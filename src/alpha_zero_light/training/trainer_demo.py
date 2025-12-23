"""
Demo training script with live telemetry for C++ viewer.

This script starts AlphaZero Connect Four training with telemetry enabled,
optimized for live visualization (single-threaded self-play for maximum frame streaming).

Usage:
    python -m alpha_zero_light.training.trainer_demo \\
        --telemetry tcp://127.0.0.1:5556 \\
        --device cuda \\
        --demo \\
        --run_dir runs/connect4_live

Arguments:
    --telemetry: ZeroMQ endpoint (default: tcp://127.0.0.1:5556)
    --device: Device for training (cuda/cpu, default: cuda)
    --demo: Enable demo mode (num_workers=1, stream all moves)
    --run_dir: Directory for checkpoints and logs
    --iterations: Number of training iterations (default: 50)
    --mcts_searches: MCTS searches per move (default: 50 for demo, 100 for normal)
"""

import argparse
import torch
import sys
from pathlib import Path

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS
from alpha_zero_light.training.trainer import AlphaZeroTrainer
from alpha_zero_light.visualization.telemetry import TelemetryPublisher
from alpha_zero_light.config_connect4 import TRAINING_CONFIG, MCTS_CONFIG, MODEL_CONFIG


def main():
    parser = argparse.ArgumentParser(description="AlphaZero Connect Four with Live Telemetry")
    parser.add_argument("--telemetry", type=str, default="tcp://127.0.0.1:5556",
                        help="ZeroMQ endpoint for telemetry")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode: num_workers=1, stream all frames")
    parser.add_argument("--run_dir", type=str, default="runs/connect4_live",
                        help="Directory for checkpoints")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of training iterations")
    parser.add_argument("--mcts_searches", type=int, default=None,
                        help="MCTS searches per move (default: 50 in demo, 100 normal)")
    
    args = parser.parse_args()
    
    # GPU verification
    print("\n" + "="*60)
    print("🚀 AlphaZero-Light Demo Trainer with Live Telemetry")
    print("="*60)
    print(f"\n📊 Configuration:")
    print(f"   Telemetry: {args.telemetry}")
    print(f"   Device: {args.device}")
    print(f"   Demo mode: {args.demo}")
    print(f"   Run directory: {args.run_dir}")
    print(f"   Iterations: {args.iterations}")
    
    # Check CUDA availability
    if args.device == "cuda":
        if torch.cuda.is_available():
            print(f"\n✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            device = torch.device("cuda")
        else:
            print("\n⚠️  CUDA not available, falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print(f"\n💻 Using CPU")
    
    # Initialize game
    game = ConnectFour()
    print(f"\n🎮 Game: {game}")
    
    # Initialize model
    model = ResNet(
        game,
        num_res_blocks=MODEL_CONFIG['num_res_blocks'],
        num_hidden=MODEL_CONFIG['num_hidden']
    ).to(device)
    
    print(f"\n🧠 Model: {MODEL_CONFIG['num_res_blocks']} ResBlocks, {MODEL_CONFIG['num_hidden']} hidden")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=MODEL_CONFIG['learning_rate'],
        weight_decay=MODEL_CONFIG['weight_decay']
    )
    
    # Training config (modify for demo mode)
    train_config = TRAINING_CONFIG.copy()
    mcts_config = MCTS_CONFIG.copy()
    
    if args.demo:
        print("\n🎬 DEMO MODE ENABLED:")
        print("   - Single-threaded self-play (num_workers=1)")
        print("   - Streaming all moves for visualization")
        print("   - Reduced MCTS searches for responsiveness")
        train_config['num_self_play_iterations'] = 10  # Fewer games per iteration
        train_config['num_parallel_workers'] = 1
        mcts_config['num_searches'] = args.mcts_searches or 50
    else:
        if args.mcts_searches:
            mcts_config['num_searches'] = args.mcts_searches
    
    train_config['num_iterations'] = args.iterations
    
    # Merge configs
    full_config = {**train_config, **mcts_config}
    
    # Initialize MCTS
    mcts = MCTS(game, full_config, model)
    
    # Initialize telemetry publisher
    print(f"\n📡 Starting telemetry publisher on {args.telemetry}...")
    telemetry = TelemetryPublisher(
        args.telemetry,
        send_frame_frequency=1,  # Stream every move in demo mode
        send_metrics_frequency=1,
        send_net_summary_frequency=5
    )
    
    # Initialize trainer with telemetry
    trainer = AlphaZeroTrainer(
        model=model,
        optimizer=optimizer,
        game=game,
        args=full_config,
        mcts=mcts,
        telemetry=telemetry
    )
    
    print(f"\n🏋️  Starting training...")
    print(f"   MCTS searches: {mcts_config['num_searches']}")
    print(f"   Games per iteration: {train_config['num_self_play_iterations']}")
    print(f"   Training epochs: {train_config['num_epochs']}")
    print("\n" + "="*60 + "\n")
    
    try:
        # Run training
        trainer.learn(checkpoint_dir=args.run_dir)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    finally:
        print("\n📡 Closing telemetry publisher...")
        telemetry.close()
        print("✅ Shutdown complete")


if __name__ == "__main__":
    main()
