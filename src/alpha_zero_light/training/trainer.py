import torch
import numpy as np
import random
import json
import sys
from pathlib import Path
from tqdm import tqdm
import os
import multiprocessing as mp
from functools import partial
import time
from alpha_zero_light.training.heuristic_opponent import HeuristicOpponent
from alpha_zero_light.training.tactical_trainer import TacticalTrainer


def _worker_self_play_games(args):
    """Worker function for parallel self-play (must be at module level for pickling)"""
    num_games, temperature, model_state, game_type, row_count, column_count, train_args = args
    
    # Import here to avoid issues
    from alpha_zero_light.model.network import ResNet
    from alpha_zero_light.mcts.mcts import MCTS
    
    # Recreate game
    if game_type == "ConnectFour":
        from alpha_zero_light.game.connect_four import ConnectFour
        game = ConnectFour(row_count=row_count, column_count=column_count, win_length=4)
    elif game_type == "TicTacToe":
        from alpha_zero_light.game.tictactoe import TicTacToe
        game = TicTacToe()
    else:
        raise ValueError(f"Unknown game type: {game_type}")
    
    # Recreate model
    model = ResNet(game, 
                  num_res_blocks=train_args.get('num_res_blocks', 10),
                  num_hidden=train_args.get('num_hidden', 128))
    model.load_state_dict(model_state)
    model.eval()
    
    # Create MCTS
    mcts = MCTS(game, train_args, model)
    
    # Play games
    worker_memory = []
    for _ in range(num_games):
        game_memory = []
        player = 1
        state = game.get_initial_state()
        
        while True:
            neutral_state = game.change_perspective(state, player)
            action_probs = mcts.search(neutral_state)
            
            game_memory.append((neutral_state, action_probs, player))
            
            temp_action_probs = action_probs ** (1 / temperature)
            temp_action_probs /= np.sum(temp_action_probs)
            action = np.random.choice(game.action_size, p=temp_action_probs)
            
            state = game.get_next_state(state, action, player)
            value, is_terminal = game.get_value_and_terminated(state, action)
            
            if is_terminal:
                for hist_state, hist_probs, hist_player in game_memory:
                    hist_outcome = value if hist_player == player else game.get_opponent_value(value)
                    worker_memory.append((
                        game.get_encoded_state(hist_state),
                        hist_probs,
                        hist_outcome
                    ))
                break
            
            player = game.get_opponent(player)
    
    return worker_memory


class AlphaZeroTrainer:
    def __init__(self, model, optimizer, game, args, mcts, evaluator=None):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = mcts
        self.evaluator = evaluator
        self.heuristic_opponent = HeuristicOpponent(game)
        self.tactical_trainer = TacticalTrainer(game)
        
        # Training history
        self.history = {
            'iterations': [],
            'total_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'eval_win_rate': [],
            'eval_wins': [],
            'eval_losses': [],
            'eval_draws': []
        }
        self.initial_lr = optimizer.param_groups[0]['lr']

    def _get_temperature(self, iteration):
        """Get temperature for current iteration based on schedule"""
        if 'temperature_schedule' in self.args:
            for schedule_entry in self.args['temperature_schedule']:
                if iteration < schedule_entry['until_iteration']:
                    return schedule_entry['temperature']
            # Return last temperature if beyond all schedules
            return self.args['temperature_schedule'][-1]['temperature']
        else:
            return self.args.get('temperature', 1.0)
    
    def _apply_learning_rate_schedule(self, iteration):
        """Apply learning rate schedule if configured"""
        if 'learning_rate_schedule' in self.args:
            for schedule_entry in self.args['learning_rate_schedule']:
                if iteration == schedule_entry['at_iteration']:
                    new_lr = self.initial_lr * schedule_entry['factor']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"📉 Learning rate adjusted to {new_lr:.6f}")
                    sys.stdout.flush()
                    # Update initial_lr for cumulative decay
                    self.initial_lr = new_lr
                    break

    def self_play_vs_random(self, temperature=None, use_heuristic=False):
        """
        Play game where model plays as either player 1 or -1 (50/50 random).
        
        Args:
            temperature: Temperature for move selection
            use_heuristic: If True, opponent uses 1-ply heuristic. If False, pure random.
        
        This provides clean value targets during warmup phase.
        CRITICAL: Randomly alternate which player the model controls so it learns
        both offensive and defensive play from both perspectives.
        """
        memory = []
        model_player = np.random.choice([1, -1])  # Model plays as either player (50/50)
        player = 1
        state = self.game.get_initial_state()
        
        if temperature is None:
            temperature = self.args.get('temperature', 1.0)
        
        while True:
            if player == model_player:
                # Model's turn - use MCTS
                neutral_state = self.game.change_perspective(state, player)
                action_probs = self.mcts.search(neutral_state)
                
                memory.append((neutral_state, action_probs, player))
                
                temperature_action_probs = action_probs ** (1 / temperature)
                temperature_action_probs /= np.sum(temperature_action_probs)
                action = np.random.choice(self.game.action_size, p=temperature_action_probs)
            else:
                # Opponent's turn
                if use_heuristic:
                    # 1-ply heuristic: win if possible, block if necessary, else random
                    action = self.heuristic_opponent.get_action(state, player)
                else:
                    # Pure random
                    valid_moves = self.game.get_valid_moves(state)
                    valid_actions = np.where(valid_moves == 1)[0]
                    action = np.random.choice(valid_actions)
            
            state = self.game.get_next_state(state, action, player)
            value, is_terminal = self.game.get_value_and_terminated(state, action)
            
            if is_terminal:
                return_memory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    # Value from model's perspective
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    return_memory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return return_memory
            
            player = self.game.get_opponent(player)

    def self_play(self, temperature=None):
        memory = []
        player = 1
        state = self.game.get_initial_state()
        
        # Use provided temperature or fall back to config
        if temperature is None:
            temperature = self.args.get('temperature', 1.0)
        
        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)
            
            memory.append((neutral_state, action_probs, player))
            
            temperature_action_probs = action_probs ** (1 / temperature)
            temperature_action_probs /= np.sum(temperature_action_probs)
            action = np.random.choice(self.game.action_size, p=temperature_action_probs)
            
            state = self.game.get_next_state(state, action, player)
            
            value, is_terminal = self.game.get_value_and_terminated(state, action)
            
            if is_terminal:
                return_memory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    return_memory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return return_memory
            
            player = self.game.get_opponent(player)
    
    def _parallel_self_play_multicore(self, num_workers, temperature):
        """Run self-play games in parallel across multiple CPU cores"""
        games_per_worker = self.args['num_self_play_iterations'] // num_workers
        remainder = self.args['num_self_play_iterations'] % num_workers
        
        # Get model state for workers (CPU only to avoid CUDA issues)
        device_backup = self.model.device if hasattr(self.model, 'device') else None
        cpu_model = self.model.cpu()
        model_state = cpu_model.state_dict()
        
        # Prepare worker arguments
        worker_tasks = [
            (games_per_worker + (1 if i < remainder else 0), temperature, model_state, 
             type(self.game).__name__, self.game.row_count if hasattr(self.game, 'row_count') else None,
             self.game.column_count if hasattr(self.game, 'column_count') else None, 
             self.args)
            for i in range(num_workers)
        ]
        
        # Run parallel workers
        ctx = mp.get_context('spawn')  # Use spawn to avoid CUDA fork issues
        with ctx.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(_worker_self_play_games, worker_tasks),
                total=num_workers,
                desc="Worker completion",
                unit="worker"
            ))
        
        # Restore model to original device
        if device_backup:
            self.model = cpu_model.to(device_backup)
        
        # Combine results
        all_memory = []
        for worker_memory in results:
            all_memory.extend(worker_memory)
        
        return all_memory
    
    def parallel_self_play(self, num_games):
        """
        Run multiple self-play games in parallel using batch inference.
        Uses smaller batches to avoid memory issues and dimension bugs.
        """
        # Process in smaller batches for stability
        max_batch_size = 256
        all_memory = []
        
        for batch_start in range(0, num_games, max_batch_size):
            batch_size = min(max_batch_size, num_games - batch_start)
            
            # Initialize batch of games - keep batch dimension (1, 1, H, W)
            states = [self.game.get_initial_state(1) for _ in range(batch_size)]
            players = [1] * batch_size
            game_histories = [[] for _ in range(batch_size)]
            active_indices = list(range(batch_size))
            
            # Loop until all games in this batch are finished
            move_count = 0
            while active_indices:
                # Prepare states for MCTS
                active_states = [states[idx] for idx in active_indices]
                active_players = [players[idx] for idx in active_indices]
                
                # Canonical states for MCTS - keep as (1, 1, H, W)
                canonical_states = [
                    self.game.change_perspective(s, p)
                    for s, p in zip(active_states, active_players)
                ]
                
                # Run Batched MCTS - squeeze to (1, H, W) for MCTS input
                action_probs_batch = self.mcts.search_batch([s.squeeze(0) for s in canonical_states])
                
                # Make moves for all active games
                next_active_indices = []
                
                for j, idx in enumerate(active_indices):
                    state = states[idx]  # (1, 1, H, W)
                    player = players[idx]
                    action_probs = action_probs_batch[j]
                    
                    # 🔍 POLICY DIAGNOSTICS (log first game of each batch)
                    if idx == 0 and len(active_indices) == batch_size:
                        import math
                        # Calculate entropy: H = -Σ(p * log(p))
                        entropy = -sum(p * math.log(p + 1e-10) for p in action_probs if p > 0)
                        max_entropy = math.log(self.game.action_size)
                        
                        # Check concentration
                        sorted_probs = sorted(action_probs, reverse=True)
                        top1 = sorted_probs[0]
                        top3 = sum(sorted_probs[:3])
                        
                        # Count how many moves have >1% probability
                        significant_moves = sum(1 for p in action_probs if p > 0.01)
                        
                        print(f"\n📊 POLICY TARGET DIAGNOSTIC (Game {idx+1}):")
                        print(f"  Entropy: {entropy:.3f} / {max_entropy:.3f} ({entropy/max_entropy*100:.1f}%)")
                        print(f"  Top-1: {top1:.3f}, Top-3: {top3:.3f}")
                        print(f"  Significant moves (>1%): {significant_moves}/{self.game.action_size}")
                        print(f"  Expected: Entropy <50%, Top-1 >30%, <20 significant moves")
                        
                        if entropy/max_entropy > 0.7:
                            print(f"  ⚠️  TOO UNIFORM - Policy targets have no signal!")
                        elif top1 > 0.5:
                            print(f"  ✅ FOCUSED - Good policy signal")
                    
                    # Store history - store squeezed (1, H, W) for MCTS compatibility
                    canonical_state = canonical_states[j].squeeze(0)  # (1, H, W)
                    game_histories[idx].append((canonical_state, action_probs, player))
                    
                    # Choose action with temperature
                    temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                    prob_sum = np.sum(temperature_action_probs)
                    
                    if prob_sum <= 0 or np.isnan(prob_sum):
                        temperature_action_probs = np.ones(self.game.action_size) / self.game.action_size
                    else:
                        temperature_action_probs /= prob_sum
                        
                    action = np.random.choice(self.game.action_size, p=temperature_action_probs)
                    
                    # Step game - expects (B, 1, H, W)
                    state = self.game.get_next_state(state, 
                                                     torch.tensor([action], device=self.game.device),
                                                     torch.tensor([player], device=self.game.device, dtype=torch.float32))
                    states[idx] = state
                    
                    # Check termination - expects (B, 1, H, W)
                    value, is_terminal = self.game.get_value_and_terminated(state, torch.tensor([action], device=self.game.device))
                    value = value.item()
                    is_terminal = is_terminal.item()
                    
                    if is_terminal:
                        # Game finished, process history
                        for hist_state, hist_probs, hist_player in game_histories[idx]:
                            hist_outcome = value if hist_player == player else -value
                            
                            # Encode state - hist_state is (1, H, W), needs (B, 1, H, W)
                            # Add batch dim, encode, then remove batch to get (3, H, W) for storage
                            encoded_state = self.game.get_encoded_state(hist_state.unsqueeze(0))  # (1, 3, H, W)
                            encoded_state = encoded_state.squeeze(0).cpu()  # (3, H, W)
                            
                            # Debug first storage
                            if len(all_memory) == 0:
                                print(f"DEBUG STORAGE: encoded shape = {encoded_state.shape}")
                            
                            all_memory.append((
                                encoded_state,  # PyTorch tensor (3, H, W)
                                hist_probs,      # numpy array from MCTS
                                hist_outcome     # Python float
                            ))
                    else:
                        # Continue game
                        players[idx] = -player
                        next_active_indices.append(idx)
                
                active_indices = next_active_indices
                move_count += 1
                
                # Progress indicator every 10 moves
                if move_count % 10 == 0:
                    completed = batch_size - len(active_indices)
                    total_completed = batch_start + completed
                    print(f"   - Progress: {total_completed}/{num_games} games finished", end='\r')
                    sys.stdout.flush()
                
        print(f"   - Progress: {num_games}/{num_games} games finished")
        sys.stdout.flush()
        return all_memory
        """
        Run multiple self-play games in parallel using batch inference.
        Uses smaller batches to avoid memory issues and dimension bugs.
        """
        # Process in smaller batches for stability
        max_batch_size = 256
        all_memory = []
        
        for batch_start in range(0, num_games, max_batch_size):
            batch_size = min(max_batch_size, num_games - batch_start)
            
            # Initialize batch of games
            states = [self.game.get_initial_state(1).squeeze(0) for _ in range(batch_size)]
            players = [1] * batch_size
            game_histories = [[] for _ in range(batch_size)]
            active_indices = list(range(batch_size))
            
            # Loop until all games in this batch are finished
            move_count = 0
            while active_indices:
                # Prepare states for MCTS
                active_states = [states[idx] for idx in active_indices]
                active_players = [players[idx] for idx in active_indices]
                
                # Canonical states for MCTS (perspective of current player)
                canonical_states = [
                    self.game.change_perspective(s.unsqueeze(0), p).squeeze(0)
                    for s, p in zip(active_states, active_players)
                ]
                
                # Run Batched MCTS
                action_probs_batch = self.mcts.search_batch(canonical_states)
                
                # Make moves for all active games
                next_active_indices = []
                
                for j, idx in enumerate(active_indices):
                    state = states[idx]
                    player = players[idx]
                    action_probs = action_probs_batch[j]
                    
                    # Store history
                    canonical_state = canonical_states[j]
                    game_histories[idx].append((canonical_state, action_probs, player))
                    
                    # Choose action with temperature
                    temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                    prob_sum = np.sum(temperature_action_probs)
                    
                    if prob_sum <= 0 or np.isnan(prob_sum):
                        temperature_action_probs = np.ones(self.game.action_size) / self.game.action_size
                    else:
                        temperature_action_probs /= prob_sum
                        
                    action = np.random.choice(self.game.action_size, p=temperature_action_probs)
                    
                    # Step game
                    state = self.game.get_next_state(state.unsqueeze(0), 
                                                     torch.tensor([action], device=self.game.device),
                                                     torch.tensor([player], device=self.game.device, dtype=torch.float32)).squeeze(0)
                    states[idx] = state
                    
                    # Check termination
                    value, is_terminal = self.game.get_value_and_terminated(
                        state.unsqueeze(0), 
                        torch.tensor([action], device=self.game.device))
                    value = value.item()
                    is_terminal = is_terminal.item()
                    
                    if is_terminal:
                        # Game finished, process history
                        for hist_state, hist_probs, hist_player in game_histories[idx]:
                            hist_outcome = value if hist_player == player else -value
                            
                            # Encode state - hist_state is (1, H, W)
                            # Keep as tensor for efficiency  
                            encoded_state = self.game.get_encoded_state(hist_state.unsqueeze(0))
                            
                            # Force reshape to exactly (3, H, W) - no ambiguity
                            # get_encoded_state returns (something, 3, H, W), we want just (3, H, W)
                            encoded_state = encoded_state.view(3, self.game.board_size, self.game.board_size).cpu()
                            
                            all_memory.append((
                                encoded_state,  # PyTorch tensor (3, H, W)
                                hist_probs,      # numpy array from MCTS
                                hist_outcome     # Python float
                            ))
                    else:
                        # Continue game
                        players[idx] = -player
                        next_active_indices.append(idx)
                
                active_indices = next_active_indices
                move_count += 1
                
                # Progress indicator every 10 moves
                if move_count % 10 == 0:
                    completed = batch_size - len(active_indices)
                    total_completed = batch_start + completed
                    print(f"   - Progress: {total_completed}/{num_games} games finished", end='\r')
                    sys.stdout.flush()
                
        print(f"   - Progress: {num_games}/{num_games} games finished")
        sys.stdout.flush()
        return all_memory

    def train(self, memory):
        random.shuffle(memory)
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        for batch_idx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batch_idx:min(len(memory), batch_idx + self.args['batch_size'])]
            states, policy_targets, value_targets = zip(*sample)
            
            # Stack tensors instead of using numpy
            # states are already tensors (3, H, W), stack to (B, 3, H, W)
            state = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32) 
                                 for s in states]).to(self.model.device)
            
            # policy_targets are numpy arrays, convert to tensor
            policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=self.model.device)
            
            # value_targets are scalars, convert to tensor  
            value_targets = torch.tensor(np.array(value_targets).reshape(-1, 1), dtype=torch.float32, device=self.model.device)
            
            out_policy, out_value = self.model(state)
            
            policy_loss = torch.nn.functional.cross_entropy(out_policy, policy_targets)
            value_loss = torch.nn.functional.mse_loss(out_value, value_targets)
            
            # Weight value loss more heavily to balance with policy loss
            # Cross entropy naturally has larger magnitude than MSE
            value_weight = self.args.get('value_loss_weight', 1.0)
            loss = policy_loss + value_weight * value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
        
        return total_loss / num_batches, total_policy_loss / num_batches, total_value_loss / num_batches
            
    def learn(self, checkpoint_dir='checkpoints'):
        """
        Main training loop with evaluation and metrics tracking
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        import time
        iteration_times = []
        
        # Create progress bar for iterations
        start_iteration = 0
        
        # Check for existing checkpoints to resume
        history_path = checkpoint_path / "training_history.json"
        if history_path.exists():
            print(f"🔄 Found existing training history at {history_path}")
            try:
                with open(history_path, 'r') as f:
                    loaded_history = json.load(f)
                
                if loaded_history['iterations']:
                    last_iteration = loaded_history['iterations'][-1]
                    model_path = checkpoint_path / f"model_{last_iteration}.pt"
                    optimizer_path = checkpoint_path / f"optimizer_{last_iteration}.pt"
                    
                    if model_path.exists() and optimizer_path.exists():
                        print(f"📥 Resuming from iteration {last_iteration + 1}...")
                        self.model.load_state_dict(torch.load(model_path, map_location=self.model.device))
                        self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.model.device))
                        self.history = loaded_history
                        start_iteration = last_iteration + 1
                        print(f"✅ Model and Optimizer state loaded.")
                    else:
                        print(f"⚠️  History found but checkpoints missing for iteration {last_iteration}. Starting from scratch.")
            except Exception as e:
                print(f"⚠️  Failed to load history: {e}. Starting from scratch.")
        
        if start_iteration >= self.args['num_iterations']:
            print(f"✨ Training already completed ({start_iteration} iterations done).")
            return self.history

        with tqdm(total=self.args['num_iterations'], initial=start_iteration, desc="Overall Progress", 
                  ncols=100, file=sys.stdout, position=0) as pbar_iterations:
            
            for iteration in range(start_iteration, self.args['num_iterations']):
                iteration_start_time = time.time()
                
                print(f"\n{'='*60}")
                print(f"📍 Iteration {iteration + 1}/{self.args['num_iterations']}")
                print(f"{'='*60}")
                sys.stdout.flush()
                
                # Get temperature for this iteration
                temperature = self._get_temperature(iteration)
                
                # Apply learning rate schedule if configured
                self._apply_learning_rate_schedule(iteration)
                
                # Determine if we're in warmup phase (playing vs opponent)
                random_opponent_iters = self.args.get('random_opponent_iterations', 0)
                in_warmup_phase = iteration < random_opponent_iters
                
                # Progressive difficulty in warmup phase with extended tactical training
                use_heuristic = False
                if in_warmup_phase:
                    # Extended warmup schedule (assumes random_opponent_iterations = 100):
                    # Iterations 0-39: Pure random (40 iterations to build fundamentals)
                    # Iterations 40-79: Heuristic opponent (40 iterations for tactical skills)
                    # Iterations 80-99: Mix (20 iterations to generalize)
                    warmup_third = random_opponent_iters // 3
                    warmup_two_thirds = 2 * warmup_third
                    
                    if iteration >= warmup_two_thirds:
                        use_heuristic = (np.random.random() < 0.5)  # 50% chance
                    elif iteration >= warmup_third:
                        use_heuristic = True  # Always heuristic
                    else:
                        use_heuristic = False  # Always random
                
                # Self-play with progress bar
                self.model.eval()
                
                if in_warmup_phase:
                    warmup_third = random_opponent_iters // 3
                    warmup_two_thirds = 2 * warmup_third
                    
                    if iteration < warmup_third:
                        opponent_desc = "Pure Random (Learning Fundamentals)"
                    elif iteration < warmup_two_thirds:
                        opponent_desc = "1-ply Heuristic (Learning Tactics: Win/Block)"
                    else:
                        opponent_desc = "Mixed Opponents + Tactical Puzzles (Mastering Patterns)"
                    
                    print(f"🎮 WARMUP PHASE: Playing vs {opponent_desc}")
                    print(f"   - {self.args['num_self_play_iterations']} games")
                    print(f"   - Model plays as Player 1, Opponent as Player -1")
                    print(f"   - Progress: {iteration + 1}/{random_opponent_iters} warmup iterations")
                else:
                    print(f"🎮 SELF-PLAY PHASE: Model vs Model")
                    print(f"   - {self.args['num_self_play_iterations']} games (standard AlphaZero)")
                
                print(f"   - MCTS searches: {self.args.get('num_searches', 50)}")
                print(f"   - Temperature: {temperature:.2f}")
                sys.stdout.flush()
                
                # Sequential self-play with GPU batching for speed
                memory = []
                
                # In mixed phase, add tactical puzzles (30% of games)
                warmup_third = random_opponent_iters // 3
                warmup_two_thirds = 2 * warmup_third
                tactical_games = 0
                
                if in_warmup_phase and iteration >= warmup_two_thirds:
                    # Mixed phase: include tactical training
                    num_tactical = int(self.args['num_self_play_iterations'] * 0.3)
                    tactical_games = num_tactical
                    print(f"   - Including {num_tactical} tactical puzzle games")
                    sys.stdout.flush()
                
                for game_num in tqdm(range(self.args['num_self_play_iterations']), desc="Self-play games", unit="game"):
                    # In mixed phase, decide game type
                    if in_warmup_phase and iteration >= warmup_two_thirds:
                        # Mixed phase: 30% tactical, 35% heuristic, 35% random
                        game_type = np.random.choice(['tactical', 'heuristic', 'random'], p=[0.3, 0.35, 0.35])
                        
                        if game_type == 'tactical':
                            game_memory = self.tactical_trainer.generate_tactical_game(self.mcts)  # Random player selection
                        elif game_type == 'heuristic':
                            game_memory = self.self_play_vs_random(temperature=temperature, use_heuristic=True)
                        else:
                            game_memory = self.self_play_vs_random(temperature=temperature, use_heuristic=False)
                    elif in_warmup_phase:
                        # Earlier warmup phases: random or heuristic
                        game_use_heuristic = use_heuristic
                        game_memory = self.self_play_vs_random(temperature=temperature, use_heuristic=game_use_heuristic)
                    else:
                        # Self-play phase
                        game_memory = self.self_play(temperature=temperature)
                    
                    memory.extend(game_memory)
                
                print(f"✅ Generated {len(memory)} training samples")
                sys.stdout.flush()
                
                # Training
                self.model.train()
                epoch_losses = []
                epoch_policy_losses = []
                epoch_value_losses = []
                
                print(f"🧠 Training neural network ({self.args['num_epochs']} epochs)...")
                sys.stdout.flush()
                
                for epoch in tqdm(range(self.args['num_epochs']), desc=f"Training", 
                                  ncols=80, file=sys.stdout, position=1, leave=False):
                    loss, policy_loss, value_loss = self.train(memory)
                    epoch_losses.append(loss)
                    epoch_policy_losses.append(policy_loss)
                    epoch_value_losses.append(value_loss)
                
                avg_loss = np.mean(epoch_losses)
                avg_policy_loss = np.mean(epoch_policy_losses)
                avg_value_loss = np.mean(epoch_value_losses)
                
                print(f"📊 Loss: {avg_loss:.4f} (Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f})")
                sys.stdout.flush()
                
                # Save metrics
                self.history['iterations'].append(iteration)
                self.history['total_loss'].append(avg_loss)
                self.history['policy_loss'].append(avg_policy_loss)
                self.history['value_loss'].append(avg_value_loss)
                
                # Evaluation - TEMPORARILY DISABLED
                # if self.evaluator and (iteration % self.args.get('eval_frequency', 5) == 0 or iteration == self.args['num_iterations'] - 1):
                #     print(f"⚔️  Evaluating model...")
                #     sys.stdout.flush()
                #     eval_results = self.evaluator.evaluate(
                #         num_games=self.args.get('num_eval_games', 20),
                #         verbose=True
                #     )
                #     print(f"🏆 Win Rate: {eval_results['win_rate']*100:.1f}% "
                #           f"(W:{eval_results['wins']} L:{eval_results['losses']} D:{eval_results['draws']})")
                #     sys.stdout.flush()
                #     
                #     self.history['eval_win_rate'].append(eval_results['win_rate'])
                #     self.history['eval_wins'].append(eval_results['wins'])
                #     self.history['eval_losses'].append(eval_results['losses'])
                #     self.history['eval_draws'].append(eval_results['draws'])
                
                # Save checkpoint
                print(f"💾 Saving checkpoint...")
                sys.stdout.flush()
                torch.save(self.model.state_dict(), checkpoint_path / f"model_{iteration}.pt")
                torch.save(self.optimizer.state_dict(), checkpoint_path / f"optimizer_{iteration}.pt")
                
                # Save training history
                with open(checkpoint_path / "training_history.json", 'w') as f:
                    json.dump(self.history, f, indent=2)
                
                # Calculate iteration time and ETA
                iteration_end_time = time.time()
                iteration_duration = iteration_end_time - iteration_start_time
                iteration_times.append(iteration_duration)
                
                # Calculate ETA
                avg_iteration_time = np.mean(iteration_times[-5:])  # Use last 5 iterations
                remaining_iterations = self.args['num_iterations'] - (iteration + 1)
                eta_seconds = avg_iteration_time * remaining_iterations
                eta_hours = eta_seconds / 3600
                eta_minutes = (eta_seconds % 3600) / 60
                
                print(f"✅ Iteration {iteration + 1} complete! (took {iteration_duration/60:.1f}min)")
                print(f"⏱️  ETA: {int(eta_hours)}h {int(eta_minutes)}m remaining")
                sys.stdout.flush()
                
                # Update progress bar
                pbar_iterations.update(1)
                pbar_iterations.set_postfix({
                    'Loss': f'{avg_loss:.3f}',
                    'ETA': f'{int(eta_hours)}h{int(eta_minutes)}m'
                })
        
        print(f"\n{'='*60}")
        print("🎉 Training Complete!")
        print(f"{'='*60}")
        sys.stdout.flush()
        
        return self.history

