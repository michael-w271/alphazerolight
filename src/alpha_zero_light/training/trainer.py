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
        """Run multiple self-play games in parallel"""
        # Limit workers to 75% of CPU cores for GPU headroom
        num_workers = max(1, int(cpu_count() * 0.75))
        num_workers = min(num_workers, num_games)  # Don't use more workers than games
        
        # Split games across workers
        games_per_worker = num_games // num_workers
        remaining_games = num_games % num_workers
        
        all_memory = []
        
        # Sequential batches to control GPU usage
        batch_size = num_workers
        for batch_start in range(0, num_games, batch_size):
            batch_end = min(batch_start + batch_size, num_games)
            batch_games = batch_end - batch_start
            
            # Run batch of games
            for _ in range(batch_games):
                memory = self.self_play()
                all_memory.extend(memory)
        
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
        
        for iteration in range(self.args['num_iterations']):
            print(f"\n{'='*60}")
            print(f"📍 Iteration {iteration + 1}/{self.args['num_iterations']}")
            print(f"{'='*60}")
            sys.stdout.flush()
            
            # Self-play with progress bar
            self.model.eval()
            print(f"🎮 Starting self-play ({self.args['num_self_play_iterations']} games)...")
            sys.stdout.flush()
            
            # Parallel self-play with manual progress tracking
            memory = []
            batch_size = max(1, int(cpu_count() * 0.75))
            num_batches = (self.args['num_self_play_iterations'] + batch_size - 1) // batch_size
            
            with tqdm(total=self.args['num_self_play_iterations'], desc="Self-Play", ncols=80, file=sys.stdout) as pbar:
                for batch_idx in range(num_batches):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, self.args['num_self_play_iterations'])
                    batch_games = batch_end - batch_start
                    
                    # Run batch of games sequentially (GPU bottleneck)
                    for _ in range(batch_games):
                        game_memory = self.self_play()
                        memory.extend(game_memory)
                        pbar.update(1)
            
            print(f"✅ Generated {len(memory)} training samples")
            sys.stdout.flush()
            
            # Training
            self.model.train()
            epoch_losses = []
            epoch_policy_losses = []
            epoch_value_losses = []
            
            print(f"🧠 Training neural network ({self.args['num_epochs']} epochs)...")
            sys.stdout.flush()
            
            for epoch in tqdm(range(self.args['num_epochs']), desc=f"Training", ncols=80, file=sys.stdout):
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
            
            print(f"✅ Iteration {iteration + 1} complete!")
            sys.stdout.flush()
        
        print(f"\n{'='*60}")
        print("🎉 Training Complete!")
        print(f"{'='*60}")
        sys.stdout.flush()
        
        return self.history

