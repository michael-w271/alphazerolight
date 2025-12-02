import torch
import torch.nn as nn
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from alpha_zero_light.game.gomoku_gpu import GomokuGPU
from alpha_zero_light.model.network import ResNet

def stress_test_with_model(batch_size=10000, num_iterations=1000):
    """
    Full AlphaZero-style stress test: Game logic + MASSIVE ResNet inference
    Target: >200W GPU power
    """
    print("="*70)
    print("GPU STRESS TEST - Full AlphaZero Pipeline")
    print("="*70)
    print(f"Batch Size: {batch_size:,} parallel games")
    print(f"Iterations: {num_iterations}")
    print()
    
    device = torch.device('cuda')
    
    # Create MASSIVE model
    class DummyGame:
        def __init__(self):
            self.row_count = 9
            self.column_count = 9
            self.action_size = 81
    
    dummy_game = DummyGame()
    
    print("Creating MASSIVE ResNet...")
    print("  - 20 residual blocks")
    print("  - 512 hidden channels")
    print("  - ~50M parameters")
    
    model = ResNet(dummy_game, num_res_blocks=20, num_hidden=512).to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total parameters: {total_params:,}")
    print()
    
    # Create game
    game = GomokuGPU(board_size=9, device=device)
    
    print("Starting continuous inference loop...")
    print("🔥 Monitor GPU with: watch -n 0.5 nvidia-smi")
    print()
    
    states = game.get_initial_state(batch_size)
    players = torch.ones(batch_size, device=device)
    
    start_time = time.time()
    total_inferences = 0
    
    for iteration in range(num_iterations):
        # Encode states for model
        encoded = game.get_encoded_state(states) # (B, 3, 9, 9)
        
        # NEURAL NETWORK INFERENCE (this should push GPU hard!)
        with torch.no_grad():
            policy, value = model(encoded)
        
        # Use policy to select actions
        valid_moves = game.get_valid_moves(states)
        policy_logits = policy.clone()
        policy_logits[valid_moves == 0] = -float('inf')
        actions = torch.argmax(policy_logits, dim=1)
        
        # Step games
        states = game.get_next_state(states, actions, players)
        
        # Check wins (Conv2d operation)
        wins = game.check_win(states, players[0])
        
        # Reset won games
        if wins.any():
            states[wins] = game.get_initial_state(wins.sum().item())
        
        # Flip players
        players = -players
        
        total_inferences += batch_size
        
        # Progress every 50 iterations
        if (iteration + 1) % 50 == 0:
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            inferences_per_sec = total_inferences / elapsed
            print(f"Iter {iteration+1}/{num_iterations} | {inferences_per_sec:,.0f} inferences/s | {elapsed:.1f}s elapsed")
    
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    print()
    print("="*70)
    print(f"Total inferences: {total_inferences:,}")
    print(f"Time: {total_time:.2f}s")
    print(f"Speed: {total_inferences/total_time:,.0f} inferences/s")
    print("="*70)
    print()
    print("Check final GPU stats:")
    os.system("nvidia-smi")

if __name__ == "__main__":
    stress_test_with_model(batch_size=10000, num_iterations=1000)
