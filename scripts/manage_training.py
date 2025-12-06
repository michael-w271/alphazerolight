#!/usr/bin/env python3
import argparse
import sys
import os
import shutil
from pathlib import Path

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from alpha_zero_light.config_gomoku_9x9_overnight import PATHS

def start_training():
    """Start or resume training."""
    print("🚀 Starting/Resuming Gomoku Training...")
    # We just run the training script directly, as it now handles resuming automatically
    script_path = os.path.join(os.path.dirname(__file__), 'train_gomoku_9x9_overnight.py')
    os.system(f"{sys.executable} {script_path}")

def clean_checkpoints():
    """Clean all checkpoints and history."""
    checkpoint_dir = Path(PATHS['checkpoints'])
    if checkpoint_dir.exists():
        print(f"⚠️  WARNING: This will delete all data in {checkpoint_dir}")
        response = input("Are you sure? (y/N): ")
        if response.lower() == 'y':
            shutil.rmtree(checkpoint_dir)
            print("🗑️  Checkpoints cleaned.")
        else:
            print("❌ Operation cancelled.")
    else:
        print("Checking directory does not exist.")

def main():
    parser = argparse.ArgumentParser(description="Manage Gomoku Training")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Start command
    subparsers.add_parser('start', help='Start or resume training')
    
    # Clean command
    subparsers.add_parser('clean', help='Delete all checkpoints and start fresh')
    
    args = parser.parse_args()
    
    if args.command == 'start':
        start_training()
    elif args.command == 'clean':
        clean_checkpoints()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
