#!/usr/bin/env python3
"""
Visualize training metrics from training_history.json
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from alpha_zero_light.config_connect4 import PATHS

def load_history(checkpoint_dir=None):
    """Load training history from JSON file"""
    if checkpoint_dir is None:
        checkpoint_dir = PATHS.checkpoints
    history_path = Path(checkpoint_dir) / 'training_history.json'
    
    if not history_path.exists():
        print(f"Error: Training history not found at {history_path}")
        print("Please run training first: python scripts/run_train.py")
        sys.exit(1)
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return history

def plot_training_metrics(history, output_dir=None):
    """Generate and save training plots"""
    if output_dir is None:
        output_dir = PATHS.plots
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    iterations = history['iterations']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('AlphaZero Training Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Total Loss
    ax1 = axes[0, 0]
    ax1.plot(iterations, history['total_loss'], 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss over Training')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Policy and Value Loss
    ax2 = axes[0, 1]
    ax2.plot(iterations, history['policy_loss'], 'r-', linewidth=2, marker='s', markersize=4, label='Policy Loss')
    ax2.plot(iterations, history['value_loss'], 'g-', linewidth=2, marker='^', markersize=4, label='Value Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Policy vs Value Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Win Rate
    if history['eval_win_rate']:
        # Create x-axis for evaluations
        num_evals = len(history['eval_win_rate'])
        if num_evals > 0:
            # Distribute evaluations evenly across iterations
            eval_iterations = np.linspace(iterations[0], iterations[-1], num_evals)
            
            ax3 = axes[1, 0]
            ax3.plot(eval_iterations, 
                    [wr * 100 for wr in history['eval_win_rate']], 
                    'purple', linewidth=2, marker='D', markersize=6)
            ax3.set_xlabel('Iteration (Approx)')
            ax3.set_ylabel('Win Rate (%)')
            ax3.set_title('Win Rate vs Random Player')
            ax3.set_ylim(0, 105)
            ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No evaluation data', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Win Rate vs Random Player')
    
    # Plot 4: Evaluation Results Breakdown
    if history['eval_wins']:
        ax4 = axes[1, 1]
        eval_iters = list(range(len(history['eval_wins'])))
        width = 0.25
        
        x = np.arange(len(eval_iters))
        ax4.bar(x - width, history['eval_wins'], width, label='Wins', color='green', alpha=0.8)
        ax4.bar(x, history['eval_draws'], width, label='Draws', color='orange', alpha=0.8)
        ax4.bar(x + width, history['eval_losses'], width, label='Losses', color='red', alpha=0.8)
        
        ax4.set_xlabel('Evaluation #')
        ax4.set_ylabel('Number of Games')
        ax4.set_title('Evaluation Results Breakdown')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        axes[1, 1].text(0.5, 0.5, 'No evaluation data', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Evaluation Results Breakdown')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_path / 'training_metrics.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved training metrics plot to: {plot_path}")
    
    # Show plot
    plt.show()

def print_summary(history):
    """Print training summary statistics"""
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    
    if history['iterations']:
        print(f"Total iterations: {len(history['iterations'])}")
        print(f"\nLoss Statistics:")
        print(f"  Initial loss: {history['total_loss'][0]:.4f}")
        print(f"  Final loss: {history['total_loss'][-1]:.4f}")
        print(f"  Improvement: {(history['total_loss'][0] - history['total_loss'][-1]):.4f}")
        
        if history['eval_win_rate']:
            print(f"\nEvaluation Statistics:")
            print(f"  Initial win rate: {history['eval_win_rate'][0]*100:.1f}%")
            print(f"  Final win rate: {history['eval_win_rate'][-1]*100:.1f}%")
            print(f"  Improvement: {(history['eval_win_rate'][-1] - history['eval_win_rate'][0])*100:.1f}%")
            
            final_eval_idx = -1
            print(f"\nFinal Evaluation Results:")
            print(f"  Wins: {history['eval_wins'][final_eval_idx]}")
            print(f"  Draws: {history['eval_draws'][final_eval_idx]}")
            print(f"  Losses: {history['eval_losses'][final_eval_idx]}")
    
    print("="*60)

def main():
    print("Loading training history...")
    history = load_history()
    
    print_summary(history)
    
    print("\nGenerating plots...")
    plot_training_metrics(history)
    
    print("\n✓ Visualization complete!")

if __name__ == "__main__":
    main()
