import torch
import sys
import os
import time

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from alpha_zero_light.game.gomoku_9x9 import Gomoku9x9
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS
from alpha_zero_light.training.trainer import AlphaZeroTrainer
from alpha_zero_light.training.evaluator import Evaluator
from alpha_zero_light.config_gomoku_9x9_overnight import TRAINING_CONFIG, MCTS_CONFIG, MODEL_CONFIG, PATHS

def main():
    print("="*60)
    print("AlphaZero Light - Gomoku 9x9 (Overnight Training)")
    print("="*60)
    print("⏰ Optimized for ~6 hours of training")
    print("🎯 Target: 20,000 self-play games (100 iterations × 200 games)")
    print("="*60)
    
    game = Gomoku9x9()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
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
    print(f"  Board Size: 9x9")
    print(f"  Iterations: {TRAINING_CONFIG['num_iterations']}")
    print(f"  Self-play games per iteration: {TRAINING_CONFIG['num_self_play_iterations']}")
    print(f"  Total games: {TRAINING_CONFIG['num_iterations'] * TRAINING_CONFIG['num_self_play_iterations']:,}")
    print(f"  Training epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"  Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"  MCTS searches: {MCTS_CONFIG['num_searches']}")
    print(f"  Evaluation frequency: every {TRAINING_CONFIG['eval_frequency']} iterations")
    print(f"  Checkpoints: {PATHS['checkpoints']}")
    print()
    
    # Estimate time
    est_time_per_game = 2.0  # seconds (conservative estimate)
    est_total_games = TRAINING_CONFIG['num_iterations'] * TRAINING_CONFIG['num_self_play_iterations']
    est_total_hours = (est_time_per_game * est_total_games) / 3600
    print(f"⏱️  Estimated training time: ~{est_total_hours:.1f} hours")
    print(f"📅 Expected completion: {time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time() + est_total_hours * 3600))}")
    print()
    sys.stdout.flush()
    
    # Start training
    print("🚀 Training started! Monitor progress below...")
    print("💡 Tip: Use 'tail -f training_log_overnight.txt' to monitor")
    print("=" * 60)
    sys.stdout.flush()
    
    start_time = time.time()
    history = trainer.learn(checkpoint_dir=PATHS['checkpoints'])
    end_time = time.time()
    
    elapsed_hours = (end_time - start_time) / 3600
    
    print()
    print("=" * 60)
    print("✅ Training Complete!")
    print(f"⏱️  Total time: {elapsed_hours:.2f} hours")
    print(f"📊 Training history saved to: {PATHS['checkpoints']}/training_history.json")
    print(f"💾 Checkpoints saved to: {PATHS['checkpoints']}/")
    print("=" * 60)
    sys.stdout.flush()
    
if __name__ == "__main__":
    main()
