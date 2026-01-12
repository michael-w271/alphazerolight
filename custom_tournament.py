#!/usr/bin/env python3
"""
Custom tournament script for Model 120 (Old) vs Model 11 (New)
Handles different architectures (ResNet-10 vs ResNet-20)
"""
import sys
import os
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS

def load_model_smart(path, game, device):
    """Load model trying different architectures"""
    print(f"Loading {path}...")
    
    # Try New Arch (ResNet-20, 256)
    try:
        model = ResNet(game, num_res_blocks=20, num_hidden=256).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        print("  -> Detected ResNet-20 (256 hidden)")
        return model
    except Exception as e:
        pass
        
    # Try Old Arch (ResNet-10, 128)
    try:
        model = ResNet(game, num_res_blocks=10, num_hidden=128).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        print("  -> Detected ResNet-10 (128 hidden)")
        return model
    except Exception as e:
        print(f"FAILED to load model {path}: {e}")
        return None

def play_game(game, mcts1, mcts2):
    """
    Play one game. Returns 1 if mcts1 wins, -1 if mcts2 wins, 0 for draw
    """
    board = game.get_initial_state()
    current_player = 1
    move_count = 0
    max_moves = 42
    
    while move_count < max_moves:
        if current_player == 1:
            mcts = mcts1
        else:
            mcts = mcts2
        
        # Get AI's perspective
        ai_state = game.change_perspective(board.copy(), player=current_player)
        action_probs = mcts.search(ai_state)
        
        # Mask invalid moves
        valid_moves = game.get_valid_moves(board)
        action_probs *= valid_moves
        
        if np.sum(valid_moves) == 0:
            return 0
        
        action = np.argmax(action_probs)
        board = game.get_next_state(board, action, current_player)
        move_count += 1
        
        # Check terminal
        value, is_terminal = game.get_value_and_terminated(board, action)
        if is_terminal:
            if value == 1:
                return 1 if current_player == 1 else -1
            else:
                return 0
        
        current_player = game.get_opponent(current_player)
    
    return 0

def run_duel():
    print("=" * 60)
    print("🥊 PROGRESS CHECK: Model 1 vs Model 15")
    print("=" * 60)
    
    game = ConnectFour()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Define paths (Both in new directory)
    path_old = Path("/mnt/ssd2pro/alpha-zero-light/checkpoints/connect4/model_1.pt")
    path_new = Path("/mnt/ssd2pro/alpha-zero-light/checkpoints/connect4/model_15.pt")
    
    # Load
    model_old = load_model_smart(path_old, game, device)
    model_new = load_model_smart(path_new, game, device)
    
    if not model_old or not model_new:
        print("Failed to load models")
        return

    # Setup MCTS
    args = {
        'C': 2.0,
        'num_searches': 100, 
        'dirichlet_epsilon': 0.0,
        'dirichlet_alpha': 0.3,
        'mcts_batch_size': 1
    }
    
    mcts_old = MCTS(game, args, model_old)
    mcts_new = MCTS(game, args, model_new)
    
    # Stats
    wins_old = 0
    wins_new = 0
    draws = 0
    total_games = 10 # 5 games each side
    
    print(f"\nMatch Start! {total_games} games (100 MCTS searches)")
    print("-" * 60)
    
    # 5 Games: Old starts (Player 1)
    for i in range(5):
        print(f"Game {i+1}/10 (Old starts): ", end="", flush=True)
        result = play_game(game, mcts_old, mcts_new)
        if result == 1:
            wins_old += 1
            print("Old Champ Wins 🏆")
        elif result == -1:
            wins_new += 1
            print("New Challenger Wins 🌟")
        else:
            draws += 1
            print("Draw 🤝")
            
    # 5 Games: New starts (Player 1)
    for i in range(5):
        print(f"Game {i+6}/10 (New starts): ", end="", flush=True)
        result = play_game(game, mcts_new, mcts_old)
        if result == 1:
            wins_new += 1
            print("New Challenger Wins 🌟")
        elif result == -1:
            wins_old += 1
            print("Old Champ Wins 🏆")
        else:
            draws += 1
            print("Draw 🤝")
            
    print("\n" + "=" * 60)
    print("🏁 PROGRESS REPORT")
    print("=" * 60)
    print(f"Infant (Iter 1):     {wins_old}")
    print(f"Toddler (Iter 15):   {wins_new}")
    print(f"Draws:               {draws}")
    
    if wins_new > wins_old:
        print("\n📈 PROGRESS CONFIRMED! The model is learning.")
    elif wins_old > wins_new:
        print("\n📉 REGRESSION? The infant beat the toddler!")
    else:
        print("\n⛔ STALLED. No distinct improvement yet.")

if __name__ == "__main__":
    run_duel()
