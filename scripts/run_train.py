import torch
import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS
from alpha_zero_light.training.trainer import AlphaZeroTrainer
from alpha_zero_light.training.evaluator import Evaluator
from alpha_zero_light.config_connect4 import TRAINING_CONFIG, MCTS_CONFIG, MODEL_CONFIG, PATHS

def main():
    print("="*60)
    print("AlphaZero Light - Connect Four Training")
    print("="*60)
    
    game = ConnectFour()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Create model
    model = ResNet(
        game, 
        num_res_blocks=MODEL_CONFIG['num_res_blocks'], 
        num_hidden=MODEL_CONFIG['num_hidden']
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=MODEL_CONFIG['learning_rate'], 
        weight_decay=MODEL_CONFIG['weight_decay']
    )
    
    # Combine configs for trainer
    args = {**TRAINING_CONFIG, **MCTS_CONFIG}
    
    # Create MCTS
    mcts = MCTS(game, args, model)
    
    # Create evaluator
    evaluator = Evaluator(game, model, mcts)
    
    # Create trainer
    trainer = AlphaZeroTrainer(model, optimizer, game, args, mcts, evaluator)
    
    # Print configuration
    print("Training Configuration:")
    print(f"  Iterations: {TRAINING_CONFIG['num_iterations']}")
    print(f"  Self-play games per iteration: {TRAINING_CONFIG['num_self_play_iterations']}")
    print(f"  Training epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"  Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"  MCTS searches: {MCTS_CONFIG['num_searches']}")
    print(f"  Evaluation frequency: every {TRAINING_CONFIG['eval_frequency']} iterations")
    print()
    
    # Start training
    history = trainer.learn(checkpoint_dir=PATHS.checkpoints)
    
    print(f"\nTraining history saved to: {PATHS.checkpoints}/training_history.json")
    print(f"Checkpoints saved to: {PATHS.checkpoints}/")
    print("\nTo visualize training progress, run:")
    print("  python scripts/visualize_training.py")

if __name__ == "__main__":
    main()

