#!/usr/bin/env python3
"""
Real-time training monitor that displays progress and runs periodic evaluations.

Monitors:
- Training losses (policy, value, total)
- Model improvement metrics
- Estimated time remaining
- Live tactical skill assessment
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import subprocess

class TrainingMonitor:
    """Monitor training progress in real-time."""
    
    def __init__(self, checkpoint_dir: str, eval_interval: int = 10):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.eval_interval = eval_interval
        self.history_file = self.checkpoint_dir / 'training_history.json'
        self.last_eval_iter = -1
        self.start_time = datetime.now()
        
    def load_history(self):
        """Load training history if available."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []
    
    def display_progress(self, history):
        """Display training progress."""
        if not history:
            print("No training data yet...")
            return
        
        latest = history[-1]
        iteration = latest['iteration']
        total_iterations = 150  # Target
        
        # Clear screen
        os.system('clear' if os.name != 'nt' else 'cls')
        
        print("=" * 80)
        print(f"🚀 AlphaZero Connect4 Training Monitor - Iteration {iteration}/{total_iterations}")
        print("=" * 80)
        print()
        
        # Progress bar
        progress = iteration / total_iterations
        bar_length = 50
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"Progress: [{bar}] {progress*100:.1f}%")
        print()
        
        # Training metrics
        print("📊 Training Metrics (Latest):")
        print(f"  Total Loss:  {latest.get('total_loss', 0):.4f}")
        print(f"  Policy Loss: {latest.get('policy_loss', 0):.4f}")
        print(f"  Value Loss:  {latest.get('value_loss', 0):.4f}")
        print()
        
        # Time estimates
        elapsed = datetime.now() - self.start_time
        if iteration > 0:
            time_per_iter = elapsed / iteration
            remaining_iters = total_iterations - iteration
            eta = time_per_iter * remaining_iters
            
            print(f"⏱️  Time Elapsed: {str(elapsed).split('.')[0]}")
            print(f"⏱️  Est. Remaining: {str(eta).split('.')[0]}")
            print(f"⏱️  Time per Iter: {time_per_iter.total_seconds()/60:.1f} min")
        print()
        
        # Recent trend
        if len(history) >= 5:
            recent_losses = [h.get('total_loss', 0) for h in history[-5:]]
            trend = "📉 Decreasing" if recent_losses[-1] < recent_losses[0] else "📈 Increasing"
            print(f"Loss Trend (last 5): {trend}")
            print()
        
        # Evaluation status
        latest_model = max([int(p.stem.split('_')[1]) for p in self.checkpoint_dir.glob('model_*.pt')], default=0)
        next_eval = ((latest_model // self.eval_interval) + 1) * self.eval_interval
        print(f"📈 Latest Model: model_{latest_model}.pt")
        print(f"🎯 Next Evaluation: Iteration {next_eval}")
        print()
        
        print("=" * 80)
        print("Press Ctrl+C to stop monitoring")
        print("=" * 80)
    
    def run_evaluation(self, iteration: int):
        """Run evaluation on current model."""
        model_path = self.checkpoint_dir / f"model_{iteration}.pt"
        
        if not model_path.exists():
            return
        
        print(f"\n🔍 Running evaluation on model_{iteration}.pt...")
        
        cmd = [
            '/mnt/ssd2pro/miniforge3/envs/tetrisrl/bin/python',
            'experiments/evaluate_training.py',
            '--model', str(model_path)
        ]
        
        try:
            subprocess.run(cmd, cwd='/mnt/ssd2pro/alpha-zero-light', check=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Evaluation failed: {e}")
    
    def monitor(self):
        """Main monitoring loop."""
        print("🔍 Starting training monitor...")
        print(f"📁 Watching: {self.checkpoint_dir}")
        print()
        
        last_iteration = -1
        
        try:
            while True:
                history = self.load_history()
                self.display_progress(history)
                
                if history:
                    current_iter = history[-1]['iteration']
                    
                    # Check if new model reached evaluation milestone
                    if current_iter != last_iteration and current_iter % self.eval_interval == 0:
                        if current_iter > self.last_eval_iter:
                            self.run_evaluation(current_iter)
                            self.last_eval_iter = current_iter
                    
                    last_iteration = current_iter
                
                time.sleep(10)  # Update every 10 seconds
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Connect4 training')
    parser.add_argument('--checkpoint-dir', default='/mnt/ssd2pro/alpha-zero-checkpoints/connect4_v2',
                        help='Checkpoint directory to monitor')
    parser.add_argument('--eval-interval', type=int, default=10,
                        help='Run evaluation every N iterations')
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.checkpoint_dir, args.eval_interval)
    monitor.monitor()


if __name__ == '__main__':
    main()
