import torch
import time
import numpy as np
import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from alpha_zero_light.game.gomoku_9x9 import Gomoku9x9
from alpha_zero_light.game.gomoku_gpu import GomokuGPU

def benchmark_cpu(num_games=100, num_moves=20):
    print(f"Running CPU Benchmark ({num_games} games)...")
    game = Gomoku9x9()
    start_time = time.time()
    
    for _ in range(num_games):
        state = game.get_initial_state()
        player = 1
        for _ in range(num_moves):
            valid_moves = game.get_valid_moves(state)
            valid_indices = np.where(valid_moves)[0]
            if len(valid_indices) == 0:
                break
            action = np.random.choice(valid_indices)
            state = game.get_next_state(state, action, player)
            if game.check_win(state, action):
                break
            player = -player
            
    end_time = time.time()
    duration = end_time - start_time
    print(f"CPU Time: {duration:.4f}s | Speed: {num_games/duration:.1f} games/s")
    return num_games/duration

def benchmark_gpu(batch_size=10000, num_moves=20):
    print(f"Running GPU Benchmark ({batch_size} games parallel)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    game = GomokuGPU(board_size=9, device=device)
    
    # Warmup
    states = game.get_initial_state(100)
    game.check_win(states, 1)
    torch.cuda.synchronize()
    
    start_time = time.time()
    
    states = game.get_initial_state(batch_size)
    players = torch.ones(batch_size, device=device)
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    
    for _ in range(num_moves):
        # Get valid moves
        valid_moves = game.get_valid_moves(states) # (B, 81)
        
        # Simple random policy on GPU
        # We need to select one valid move per active game
        # For benchmark speed, just pick first valid move or random
        # Random is harder on GPU, let's try:
        
        # Mask invalid moves with -inf
        logits = torch.randn_like(valid_moves)
        logits[valid_moves == 0] = -float('inf')
        actions = torch.argmax(logits, dim=1) # (B,)
        
        # Update states
        states = game.get_next_state(states, actions, players)
        
        # Check wins
        wins = game.check_win(states, players[0]) # Check for current player
        
        # Update players
        players = -players
        
        # In real training we'd handle termination, here we just run fixed steps
        # to stress test the logic
        
    torch.cuda.synchronize()
    end_time = time.time()
    duration = end_time - start_time
    print(f"GPU Time: {duration:.4f}s | Speed: {batch_size/duration:.1f} games/s")
    return batch_size/duration

if __name__ == "__main__":
    cpu_speed = benchmark_cpu(num_games=100)
    gpu_speed = benchmark_gpu(batch_size=10000)
    
    print(f"\nSpeedup: {gpu_speed/cpu_speed:.1f}x")
