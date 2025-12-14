#!/usr/bin/env python3
"""
Enhanced training script with live monitoring and automated testing.
Creates separate terminal windows for training and testing progress.
"""
import sys
import os
import subprocess
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.alpha_zero import AlphaZero
from alpha_zero_light.training.trainer import AlphaZeroTrainer
from alpha_zero_light.mcts.mcts import MCTS
from alpha_zero_light.config_connect4 import (
    TRAINING_CONFIG, MCTS_CONFIG, MODEL_CONFIG, PATHS
)

def run_in_new_terminal(command, title):
    """Launch command in a new terminal window"""
    # Try gnome-terminal first, fallback to xterm
    try:
        subprocess.Popen([
            'gnome-terminal',
            '--title', title,
            '--',
            'bash', '-c', f'{command}; exec bash'
        ])
        return True
    except FileNotFoundError:
        try:
            subprocess.Popen([
                'xterm',
                '-title', title,
                '-e', f'{command}; bash'
            ])
            return True
        except FileNotFoundError:
            return False

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent
    monitor_script = base_dir / 'scripts' / 'monitor_training.py'
    test_script = base_dir / 'scripts' / 'test_model_progress.py'
    
    print("=" * 70)
    print("🚀 ALPHAZERO CONNECT FOUR - ENHANCED TRAINING")
    print("=" * 70)
    print()
    print("Starting training with live monitoring...")
    print()
    print("This will open:")
    print("  1. Training Progress Window - Shows iteration, loss, phase")
    print("  2. Model Testing Window - Tests model after each iteration")
    print()
    print("All results saved to:")
    print("  - training_log.txt (training output)")
    print("  - model_test_results.json (test results)")
    print("  - training_progress.json (loss history)")
    print()
    
    # Start training in new terminal
    training_cmd = f"cd {base_dir} && source env_config.sh && $PYTHON_EXEC scripts/train_connect4.py"
    
    print("📊 Opening training monitor window...")
    if run_in_new_terminal(training_cmd, "AlphaZero Training - Main"):
        print("✅ Training window opened")
    else:
        print("⚠️  Could not open separate window, running in current terminal")
        os.system(training_cmd)
        return
    
    # Start testing monitor in new terminal
    test_cmd = f"cd {base_dir} && source env_config.sh && $PYTHON_EXEC {test_script}"
    
    print("🧪 Opening model testing window...")
    if run_in_new_terminal(test_cmd, "AlphaZero Testing - Model Evaluation"):
        print("✅ Testing window opened")
    else:
        print("⚠️  Testing window not available")
    
    print()
    print("=" * 70)
    print("✅ TRAINING STARTED!")
    print("=" * 70)
    print()
    print("Monitor the separate windows for progress.")
    print("Training logs: tail -f training_log.txt")
    print("Test results: cat model_test_results.json | jq")
    print()
    print("To stop training: pkill -f train_connect4")
    print()

if __name__ == '__main__':
    main()
