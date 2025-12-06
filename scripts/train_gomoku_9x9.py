import torch
import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from alpha_zero_light.game.gomoku_gpu import GomokuGPU
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS
from alpha_zero_light.training.trainer import AlphaZeroTrainer
from alpha_zero_light.training.evaluator import Evaluator
from alpha_zero_light.config_gomoku_9x9 import TRAINING_CONFIG, MCTS_CONFIG, MODEL_CONFIG, PATHS

def main():
    print("="*60)
    print("AlphaZero Light - Gomoku 9x9 (Resumed & Optimized)")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"🚀 GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Use GomokuGPU for fast parallel self-play
    game = GomokuGPU(board_size=9)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
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
    # Signature: game, args, model
    mcts = MCTS(game, args, model)
    
    # Create evaluator
    evaluator = Evaluator(game, model, mcts)
    
    # Create trainer
    trainer = AlphaZeroTrainer(model, optimizer, game, args, mcts, evaluator)
    
    # Print configuration
    print("Training Configuration:")
    print(f"  Board Size: 9x9")
    print(f"  Iterations: {TRAINING_CONFIG['num_iterations']}")
    print(f"  Self-play games per iteration: {TRAINING_CONFIG['num_self_play_iterations']}")
    print(f"  Training epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"  Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"  MCTS searches: {MCTS_CONFIG['num_searches']}")
    print(f"  Evaluation frequency: every {TRAINING_CONFIG['eval_frequency']} iterations")
    print(f"  Checkpoints: {PATHS['checkpoints']}")
    print()
    
    # Estimate time
    est_time_per_game = 5  # seconds (much faster than 15x15)
    est_total_games = TRAINING_CONFIG['num_iterations'] * TRAINING_CONFIG['num_self_play_iterations']
    est_total_minutes = (est_time_per_game * est_total_games) / 60
    print(f"⏱️  Estimated training time: ~{est_total_minutes:.0f} minutes")
    print()
    sys.stdout.flush()
    
    # Start training
    print("🚀 Training started! Monitor progress below...")
    print("=" * 60)
    sys.stdout.flush()
    
    history = trainer.learn(checkpoint_dir=PATHS['checkpoints'])
    
    print()
    print("=" * 60)
    print("✅ Training Complete!")
    print(f"📊 Training history saved to: {PATHS['checkpoints']}/training_history.json")
    print(f"💾 Checkpoints saved to: {PATHS['checkpoints']}/")
    print("=" * 60)
    sys.stdout.flush()
    
if __name__ == "__main__":
    main()
