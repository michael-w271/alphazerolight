#!/usr/bin/env python3
"""
Play Connect Four against the trained AlphaZero model.
You play as Red (🔴), AI plays as Yellow (🟡).
"""
import sys
import os
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS

def print_board(board):
    """Print the Connect Four board"""
    print("\n  " + " ".join(str(i) for i in range(7)))
    print("  " + "-" * 13)
    for row in board:
        print("| " + " ".join("🔴" if cell == 1 else "🟡" if cell == -1 else "⚪" for cell in row) + " |")
    print("  " + "-" * 13)
    print()

def get_column_from_action(board, action, column):
    """Find the actual row where the piece will land in the column"""
    for row in range(5, -1, -1):  # Start from bottom
        if board[row, column] == 0:
            return row
    return None

def main():
    # Setup
    print("=" * 60)
    print("🎮 AlphaZero Connect Four - Play vs AI")
    print("=" * 60)
    
    checkpoint_dir = Path("/mnt/ssd2pro/alpha-zero-checkpoints/connect4")
    checkpoint = checkpoint_dir / "model_120.pt"  # Tournament champion!
    
    if not checkpoint.exists():
        print(f"❌ Model not found at: {checkpoint}")
        return
    
    print(f"\n📁 Loading model: {checkpoint.name}")
    
    # Initialize game
    game = ConnectFour()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    
    # Load model
    model = ResNet(game, num_res_blocks=10, num_hidden=128).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    print("✅ Model loaded successfully!")
    
    # Initialize MCTS
    args = {
        'C': 2.0,
        'num_searches': 200,  # Good balance of speed and quality
        'dirichlet_epsilon': 0.0,  # No exploration noise in play mode
        'dirichlet_alpha': 0.3,
    }
    mcts = MCTS(game, args, model)
    print(f"🧠 AI configured with {args['num_searches']} MCTS searches per move")
    
    # Game loop
    board = game.get_initial_state()
    current_player = 1  # Human starts (Red)
    move_count = 0
    
    print("\n" + "=" * 60)
    print("🎯 Game Start!")
    print("   You are 🔴 (Red), AI is 🟡 (Yellow)")
    print("   Enter column number (0-6) to drop your piece")
    print("=" * 60)
    
    while True:
        print_board(board)
        
        if current_player == 1:
            # Human turn
            valid_moves = game.get_valid_moves(board)
            valid_cols = [i for i in range(7) if valid_moves[i]]
            
            if not valid_cols:
                print("🤝 Game Over - Draw!")
                break
            
            print(f"🔴 Your turn! Valid columns: {valid_cols}")
            
            while True:
                try:
                    col = input("Enter column (0-6) or 'q' to quit: ").strip()
                    if col.lower() == 'q':
                        print("👋 Thanks for playing!")
                        return
                    
                    col = int(col)
                    if col not in valid_cols:
                        print(f"❌ Invalid column! Choose from: {valid_cols}")
                        continue
                    break
                except (ValueError, KeyboardInterrupt):
                    print("❌ Invalid input! Enter a number 0-6")
                    continue
            
            action = col
            board = game.get_next_state(board, action, current_player)
            move_count += 1
            
            # Check win
            value, is_terminal = game.get_value_and_terminated(board, action)
            if is_terminal:
                print_board(board)
                if value == 1:
                    print("🎉 YOU WIN! Congratulations! 🎉")
                else:
                    print("🤝 Game Over - Draw!")
                break
            
            current_player = -1
            
        else:
            # AI turn
            print("🟡 AI is thinking...")
            
            # Get AI's perspective of the board
            ai_state = game.change_perspective(board.copy(), player=-1)
            action_probs = mcts.search(ai_state)
            
            # Get valid moves and mask
            valid_moves = game.get_valid_moves(board)
            action_probs *= valid_moves
            action = np.argmax(action_probs)
            
            print(f"🟡 AI plays column {action}")
            print(f"   Move probabilities: " + " ".join(
                f"{i}:{action_probs[i]:.2f}" for i in range(7) if valid_moves[i]
            ))
            
            board = game.get_next_state(board, action, current_player)
            move_count += 1
            
            # Check win
            value, is_terminal = game.get_value_and_terminated(board, action)
            if is_terminal:
                print_board(board)
                if value == 1:
                    print("🤖 AI WINS! Better luck next time!")
                else:
                    print("🤝 Game Over - Draw!")
                break
            
            current_player = 1
    
    print(f"\nTotal moves: {move_count}")
    print("\n" + "=" * 60)
    
    # Ask to play again
    again = input("\nPlay again? (y/n): ").strip().lower()
    if again == 'y':
        main()
    else:
        print("👋 Thanks for playing!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Game interrupted. Thanks for playing!")
