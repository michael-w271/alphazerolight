#!/usr/bin/env python3
"""
Quick play script to test the trained Gomoku model.
Usage: python scripts/play_gomoku.py
"""

import torch
import numpy as np
from alpha_zero_light.game.gomoku_gpu import GomokuGPU
from alpha_zero_light.model.network import AlphaZeroNet
from alpha_zero_light.mcts.mcts import MCTS

def print_board(state):
    """Print the board in a readable format"""
    board = state.squeeze().cpu().numpy()
    symbols = {1: 'X', -1: 'O', 0: '.'}
    print("\n  ", end="")
    for i in range(9):
        print(f" {i}", end="")
    print()
    for i in range(9):
        print(f"{i} ", end="")
        for j in range(9):
            print(f" {symbols[board[i, j]]}", end="")
        print()
    print()

def get_human_move(game, state):
    """Get move from human player"""
    while True:
        try:
            move_str = input("Your move (row col, e.g., '4 4'): ")
            row, col = map(int, move_str.split())
            action = row * 9 + col
            
            valid_moves = game.get_valid_moves(state.unsqueeze(0))
            if valid_moves[0, action].item() == 1:
                return action
            else:
                print("Invalid move! Try again.")
        except:
            print("Invalid input! Use format 'row col' (e.g., '4 4')")

def main():
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    game = GomokuGPU(board_size=9, device=device)
    
    model = AlphaZeroNet(game, num_res_blocks=9, num_hidden=256)
    model = model.to(device)
    
    # Load latest checkpoint
    import glob
    from pathlib import Path
    checkpoints = sorted(Path("checkpoints/gomoku_30min").glob("model_*.pt"))
    if checkpoints:
        latest = checkpoints[-1]
        print(f"Loading model: {latest}")
        model.load_state_dict(torch.load(latest, map_location=device))
        model.eval()
    else:
        print("No checkpoint found! Using untrained model.")
    
    # Setup MCTS
    mcts = MCTS(game, model, {'C': 2, 'num_searches': 100})
    
    # Game loop
    state = game.get_initial_state(1)
    player = 1  # Human is X (1), AI is O (-1)
    human_player = 1
    
    print("\n" + "="*40)
    print("Gomoku 9x9 - Play vs AI")
    print("="*40)
    print("You are X, AI is O")
    print("First to get 5 in a row wins!")
    print("="*40)
    
    while True:
        print_board(state)
        
        if player == human_player:
            # Human turn
            action = get_human_move(game, state)
        else:
            # AI turn
            print("AI is thinking...")
            neutral_state = game.change_perspective(state, player)
            action_probs = mcts.search(neutral_state.squeeze(0))
            action = np.argmax(action_probs)
            print(f"AI plays: {action // 9} {action % 9}")
        
        # Make move
        state = game.get_next_state(state,
                                    torch.tensor([action], device=device),
                                    torch.tensor([player], device=device, dtype=torch.float32))
        
        # Check win
        value, is_terminal = game.get_value_and_terminated(state, torch.tensor([action], device=device))
        
        if is_terminal.item():
            print_board(state)
            if value.item() == 1:
                winner = "You" if player == human_player else "AI"
                print(f"\n🎉 {winner} won!")
            else:
                print("\n🤝 Draw!")
            break
        
        player = -player

if __name__ == "__main__":
    main()
