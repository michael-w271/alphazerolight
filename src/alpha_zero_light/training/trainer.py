import torch
import numpy as np
import random
import json
import sys
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os

class AlphaZeroTrainer:
    def __init__(self, model, optimizer, game, args, mcts, evaluator=None):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = mcts
        self.evaluator = evaluator
        
        # Training history
        self.history = {
            'iterations': [],
            'total_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'eval_win_rate': [],
            'eval_wins': [],
            'eval_losses': [],
            'eval_draws': [],
        }

    def self_play(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()
        
        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)
            
            memory.append((neutral_state, action_probs, player))
            
            temperature_action_probs = action_probs ** (1 / self.args['temperature'])
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
    
    def parallel_self_play(self, num_games):
        """
        Run multiple self-play games in parallel using batch inference.
        This is much more efficient than multiprocessing for GPU utilization.
        """
        # INCREASED batch size for higher GPU utilization (75%)
        max_batch_size = 4096
        all_memory = []
        
        for i in range(0, num_games, max_batch_size):
            batch_size = min(max_batch_size, num_games - i)
            
            # Initialize batch of games
            states = [self.game.get_initial_state() for _ in range(batch_size)]
            players = [1] * batch_size
            game_histories = [[] for _ in range(batch_size)]
            active_indices = list(range(batch_size))
            completed_games = 0
            
            # Loop until all games in this batch are finished
            while active_indices:
                # Prepare states for MCTS
                active_states = [states[idx] for idx in active_indices]
                active_players = [players[idx] for idx in active_indices]
                
                # Canonical states for MCTS (perspective of current player)
                canonical_states = [
                    self.game.change_perspective(s, p) 
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
                    
                    # Choose action
                    temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                    temperature_action_probs /= np.sum(temperature_action_probs)
                    action = np.random.choice(self.game.action_size, p=temperature_action_probs)
                    
                    # Step game
                    state = self.game.get_next_state(state, action, player)
                    states[idx] = state
                    
                    # Check termination
                    value, is_terminal = self.game.get_value_and_terminated(state, action)
                    
                    if is_terminal:
                        # Game finished, process history
                        return_memory = []
                        for hist_state, hist_probs, hist_player in game_histories[idx]:
                            hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                            return_memory.append((
                                self.game.get_encoded_state(hist_state),
                                hist_probs,
                                hist_outcome
                            ))
                        all_memory.extend(return_memory)
                        completed_games += 1
                    else:
                        # Continue game
                        players[idx] = self.game.get_opponent(player)
                        next_active_indices.append(idx)
                
                active_indices = next_active_indices
                
        return all_memory

    def train(self, memory):
        random.shuffle(memory)
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        for batch_idx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batch_idx:min(len(memory), batch_idx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)
            
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
            
            out_policy, out_value = self.model(state)
            
            policy_loss = torch.nn.functional.cross_entropy(out_policy, policy_targets)
            value_loss = torch.nn.functional.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
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
                
                # Self-play with progress bar
                self.model.eval()
                print(f"🎮 Starting self-play ({self.args['num_self_play_iterations']} games)...")
                print(f"   - Batch Size: {self.args.get('batch_size', 'Auto')}")
                print(f"   - Device: {self.model.device}")
                sys.stdout.flush()
                
                # Run parallel self-play
                # We use a large batch size for efficiency
                memory = self.parallel_self_play(self.args['num_self_play_iterations'])
                
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
                
                # Evaluation
                if self.evaluator and (iteration % self.args.get('eval_frequency', 5) == 0 or iteration == self.args['num_iterations'] - 1):
                    print(f"⚔️  Evaluating model...")
                    sys.stdout.flush()
                    eval_results = self.evaluator.evaluate(
                        num_games=self.args.get('num_eval_games', 20),
                        verbose=True
                    )
                    print(f"🏆 Win Rate: {eval_results['win_rate']*100:.1f}% "
                          f"(W:{eval_results['wins']} L:{eval_results['losses']} D:{eval_results['draws']})")
                    sys.stdout.flush()
                    
                    self.history['eval_win_rate'].append(eval_results['win_rate'])
                    self.history['eval_wins'].append(eval_results['wins'])
                    self.history['eval_losses'].append(eval_results['losses'])
                    self.history['eval_draws'].append(eval_results['draws'])
                
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

