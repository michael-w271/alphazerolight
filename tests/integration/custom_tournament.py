#!/usr/bin/env python3
"""
Custom tournament script for Model 5 (Old) vs Model 195 (New)
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

def play_game(game, mcts1, mcts2, device):
    """
    Play one game. Returns 1 if mcts1 wins, -1 if mcts2 wins, 0 for draw
    """
    board = game.get_initial_state()
    current_player = 1
    move_count = 0
    max_moves = 42
    
    # Print header for moves
    moves_log = []
    
    while move_count < max_moves:
        if current_player == 1:
            mcts = mcts1
            p_label = "P1"
        else:
            mcts = mcts2
            p_label = "P2"
        
        # Get AI's perspective
        ai_state = game.change_perspective(board.copy(), player=current_player)
        action_probs = mcts.search(ai_state)
        
        # Estimate value of current position for the player
        # We can get this by evaluating the root state directly with the model
        with torch.no_grad():
             encoded = game.get_encoded_state(ai_state)
             tensor = torch.tensor(encoded, dtype=torch.float32, device=device).unsqueeze(0)
             _, val_tensor = getattr(mcts, 'model')(tensor)
             root_value = val_tensor.item()

        # Mask invalid moves
        valid_moves = game.get_valid_moves(board)
        action_probs *= valid_moves
        
        if np.sum(valid_moves) == 0:
            print(f"Draw by no moves: {moves_log}")
            return 0
        
        # Select Action
        if move_count < 2: # Limit to first 2 moves (updated)
            # Softmax sampling
            action_probs_sum = np.sum(action_probs)
            if action_probs_sum > 0:
                action_probs /= action_probs_sum
            action = np.random.choice(len(action_probs), p=action_probs)
            method = "Rand"
        else:
            # Greedy
            action = np.argmax(action_probs)
            method = "MCTS"
        
        chosen_prob = action_probs[action]
        moves_log.append(f"{p_label}:{action}")
        
        print(f"Move {move_count+1}: {p_label} plays col {action} ({method}, Conf={chosen_prob:.2f}, Val={root_value:.2f})")
        
        board = game.get_next_state(board, action, current_player)
        move_count += 1
        
        # Check terminal
        value, is_terminal = game.get_value_and_terminated(board, action)
        if is_terminal:
            print(f"Game over in {move_count} moves. Moves: {', '.join(moves_log)}")
            print_board(board)
            if value == 1:
                return 1 if current_player == 1 else -1
            else:
                return 0
        
        current_player = game.get_opponent(current_player)
    
    print(f"Draw by max moves. Moves: {moves_log}")
    return 0

def print_board(board):
    print("\n  0 1 2 3 4 5 6")
    for r in range(6):
        line = "|"
        for c in range(7):
            if board[r,c] == 1:
                line += "🔴"
            elif board[r,c] == -1:
                line += "🟡"
            else:
                line += " ."
        print(line + "|")
    print("  ¯¯¯¯¯¯¯¯¯¯¯¯¯\n")

def run_duel():
    print("=" * 60)
    print("🥊 PROGRESS CHECK: Model 5 vs Model 195")
    print("=" * 60)
    
    game = ConnectFour()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Define paths
    path_old = Path("/mnt/ssd2pro/alpha-zero-light/checkpoints/connect4/model_5.pt")
    path_new = Path("/mnt/ssd2pro/alpha-zero-light/checkpoints/connect4/model_195.pt")
    
    # Load
    model_old = load_model_smart(path_old, game, device)
    model_new = load_model_smart(path_new, game, device)
    
    if not model_old or not model_new:
        print("Failed to load models")
        return

    # Setup MCTS
    args = {
        'C': 2.0,
        'num_searches': 400,  # Updated to 400 as requested
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 1.0,
        'mcts_batch_size': 1
    }
    
    mcts_old = MCTS(game, args, model_old)
    mcts_new = MCTS(game, args, model_new)
    
    # Run Puzzles first
    print("\n" + "="*60)
    print("🧩 PUZZLE EVALUATION")
    print("="*60)
    run_puzzles(game, model_old, model_new, device)
    
    # Stats
    wins_old = 0
    wins_new = 0
    draws = 0
    total_games = 20
    
    print(f"\nMatch Start! {total_games} games (400 MCTS searches)")
    print("-" * 60)
    
    # 10 Games: Old starts (Player 1)
    for i in range(10):
        print(f"Game {i+1}/{total_games} (Model 5 starts): ", end="", flush=True)
        result = play_game(game, mcts_old, mcts_new, device)
        if result == 1:
            wins_old += 1
            print("Model 5 (Old) Wins 🏆")
        elif result == -1:
            wins_new += 1
            print("Model 195 (New) Wins 🌟")
        else:
            draws += 1
            print("Draw 🤝")
            
    # 10 Games: New starts (Player 1)
    for i in range(10):
        print(f"Game {i+11}/{total_games} (Model 195 starts): ", end="", flush=True)
        result = play_game(game, mcts_new, mcts_old, device)
        if result == 1:
            wins_new += 1
            print("Model 195 (New) Wins 🌟")
        elif result == -1:
            wins_old += 1
            print("Model 5 (Old) Wins 🏆")
        else:
            draws += 1
            print("Draw 🤝")
            
    print("\n" + "=" * 60)
    print("🏁 FINAL REPORT")
    print("=" * 60)
    print(f"Model 5 (Old):       {wins_old}")
    print(f"Model 195 (New):     {wins_new}")
    print(f"Draws:               {draws}")
    
    if wins_new > wins_old:
        print("\n📈 PROGRESS: The trained model is strictly better.")
    elif wins_old > wins_new:
        print("\n📉 REGRESSION: The initialization was better? Unlikely.")
    else:
        print("\n⛔ TIED: No distinct advantage found.")

def evaluate_puzzle(game, model_old, model_new, board, description, device):
    """Compare how both models evaluate a specific board state"""
    print(f"\n📝 Puzzle: {description}")
    print_board(board)
    
    # Prepare input
    # Assume AI to move (Player 1 perspective for simplicity of evaluation function if we assume it's P1's turn)
    # We need to construct the input tensor.
    # The game.get_encoded_state takes (rows, cols).
    
    # Let's say it's P1's turn
    encoded = game.get_encoded_state(board)
    tensor = torch.tensor(encoded, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Get raw network outputs (Policy, Value)
    # Note: These are raw priors, not MCTS counts
    
    with torch.no_grad():
        pol_old, val_old = model_old(tensor)
        pol_new, val_new = model_new(tensor)
        
        pol_old = torch.softmax(pol_old, dim=1).cpu().numpy()[0]
        val_old = val_old.item()
        
        pol_new = torch.softmax(pol_new, dim=1).cpu().numpy()[0]
        val_new = val_new.item()
        
    # Mask invalid moves
    valid = game.get_valid_moves(board)
    pol_old *= valid
    pol_new *= valid
    
    # Normalize
    if np.sum(pol_old) > 0: pol_old /= np.sum(pol_old)
    if np.sum(pol_new) > 0: pol_new /= np.sum(pol_new)
    
    print(f"{'Column':<8} | {'Old P(x)':<12} | {'New P(x)':<12}")
    print("-" * 40)
    for c in range(7):
        if valid[c]:
            is_best_old = "***" if c == np.argmax(pol_old) else ""
            is_best_new = "***" if c == np.argmax(pol_new) else ""
            print(f"{c:<8} | {pol_old[c]:.4f} {is_best_old:<3} | {pol_new[c]:.4f} {is_best_new:<3}")
            
    print(f"{'Value':<8} | {val_old:.4f}       | {val_new:.4f}")
    
def run_puzzles(game, model_old, model_new, device):
    # Puzzle 1: Win in 1 (Horizontal)
    # Row 5 (Bottom): R R R . Y Y .
    board1 = np.zeros((6, 7))
    # Note: Row 5 is the BOTTOM of the board (highest index)
    # Row 0 is the TOP (lowest index)
    board1[5, 0] = 1 # R
    board1[5, 1] = 1 # R
    board1[5, 2] = 1 # R
    board1[5, 4] = -1 # Y
    board1[5, 5] = -1 # Y
    # Col 3 wins (completes row 5)
    evaluate_puzzle(game, model_old, model_new, board1, "Red to play, Win in 1 (Col 3)", device)

    # Puzzle 2: Block Win (Vertical)
    # Opponent (Yellow) has 3 stacked in Col 0. Red must play Col 0.
    board2 = np.zeros((6, 7))
    board2[5, 0] = -1 # Y (Bottom)
    board2[4, 0] = -1 # Y
    board2[3, 0] = -1 # Y
    # Row 2 is empty. Red must play Col 0 to block.
    evaluate_puzzle(game, model_old, model_new, board2, "Red to play, Must Block Col 0", device)

if __name__ == "__main__":
    run_duel()
