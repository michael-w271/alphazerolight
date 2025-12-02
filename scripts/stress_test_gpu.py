import torch
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from alpha_zero_light.game.gomoku_gpu import GomokuGPU

def stress_test_gpu(batch_size=50000, num_steps=100):
    """
    Stress test with MASSIVE parallelism to saturate GPU.
    Target: >200W power consumption
    """
    print(f"GPU Stress Test")
    print(f"  Batch Size: {batch_size:,} parallel games")
    print(f"  Steps: {num_steps}")
    print("="*60)
    
    device = 'cuda'
    game = GomokuGPU(board_size=9, device=device)
    
    # Initialize
    states = game.get_initial_state(batch_size)
    players = torch.ones(batch_size, device=device)
    
    print("Starting stress test...")
    print("⚠️  Check GPU usage with: watch -n 0.5 nvidia-smi")
    print()
    
    start = time.time()
    
    for step in range(num_steps):
        # Random moves
        valid_moves = game.get_valid_moves(states)
        logits = torch.randn_like(valid_moves)
        logits[valid_moves == 0] = -float('inf')
        actions = torch.argmax(logits, dim=1)
        
        # Step
        states = game.get_next_state(states, actions, players)
        
        # Win check (this should use Conv2d heavily)
        wins = game.check_win(states, players[0])
        
        # Flip players
        players = -players
        
        if (step + 1) % 10 == 0:
            torch.cuda.synchronize()
            elapsed = time.time() - start
            games_per_sec = (batch_size * (step + 1)) / elapsed
            print(f"Step {step+1}/{num_steps} | {games_per_sec:,.0f} games/s")
    
    torch.cuda.synchronize()
    total_time = time.time() - start
    total_games = batch_size * num_steps
    
    print()
    print("="*60)
    print(f"Total games simulated: {total_games:,}")
    print(f"Time: {total_time:.2f}s")
    print(f"Speed: {total_games/total_time:,.0f} games/s")
    print("="*60)

if __name__ == "__main__":
    # Start with huge batch
    stress_test_gpu(batch_size=50000, num_steps=100)
