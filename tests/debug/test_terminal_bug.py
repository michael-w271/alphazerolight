#!/usr/bin/env python3
"""
Test to verify the terminal state bug in MCTS.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour


def print_board(state, title=""):
    """Print the board"""
    if title:
        print(f"\n{title}")
    print("  0 1 2 3 4 5 6")
    for row in range(6):
        print(f"{row} ", end="")
        for col in range(7):
            val = state[row, col]
            if val == 1:
                print("X ", end="")
            elif val == -1:
                print("O ", end="")
            else:
                print(". ", end="")
        print()
    print()


def main():
    game = ConnectFour(row_count=6, column_count=7, win_length=4)
    
    print("="*70)
    print("TESTING TERMINAL STATE BUG")
    print("="*70)
    
    # Create a scenario where opponent has 3 in a column
    state = game.get_initial_state()
    state = game.get_next_state(state, 0, 1)   # X plays column 0
    state = game.get_next_state(state, 5, -1)  # O plays column 5
    state = game.get_next_state(state, 1, 1)   # X plays column 1
    state = game.get_next_state(state, 5, -1)  # O plays column 5
    state = game.get_next_state(state, 2, 1)   # X plays column 2
    state = game.get_next_state(state, 5, -1)  # O plays column 5
    
    print_board(state, "Starting position (X to move):")
    print("X has 3 in bottom row (0,1,2). Playing column 3 wins!")
    print("O has 3 in column 5 (rows 3,4,5). If X doesn't block, O wins next turn.")
    
    # Simulate what MCTS does when expanding root
    print("\n" + "="*70)
    print("SIMULATING MCTS EXPAND LOGIC")
    print("="*70)
    
    # Let's say MCTS wants to try action=3 (which DOES win for X)
    action = 3
    print(f"\nTrying action {action}...")
    
    # This is what expand() does:
    parent_state = state  # Current player = 1 (X)
    child_state = parent_state.copy()
    child_state = game.get_next_state(child_state, action, 1)  # Apply move for player 1
    
    print_board(child_state, f"After applying action {action} for player 1:")
    
    # Check if this wins (SHOULD return True)
    is_win_correct = game.check_win(child_state, action)
    print(f"check_win(child_state, {action}) = {is_win_correct}")
    print(f"This is CORRECT - player 1 won by playing column {action}")
    
    # Now flip perspective (this is what expand() does next)
    child_state = game.change_perspective(child_state, player=-1)
    
    print_board(child_state, "After change_perspective (flipped to opponent):")
    
    # Now check terminal state THE WAY MCTS DOES IT
    # MCTS calls: get_value_and_terminated(child_state, action)
    # But action is still 3, and child_state is now flipped!
    value, is_terminal = game.get_value_and_terminated(child_state, action)
    
    print(f"\nget_value_and_terminated(flipped_child_state, {action}):")
    print(f"  value = {value}")
    print(f"  is_terminal = {is_terminal}")
    
    if is_terminal and value == 1:
        print("\n❌ BUG DETECTED!")
        print("MCTS thinks the OPPONENT won (value=1) after WE played column 3")
        print("This is because check_win looks at the last action in the FLIPPED state")
        print("In the flipped state, column 3 has player=1 (which represents the opponent)")
        print("So it looks like the opponent won, when actually WE won!")
    
    print("\n" + "="*70)
    print("THE FIX")
    print("="*70)
    print("\nOption 1: Check terminal BEFORE flipping perspective")
    print("Option 2: Store the terminal state info with each node")
    print("Option 3: Don't use action_taken for terminal check in flipped state")
    
    # Test the correct way
    print("\n" + "="*70)
    print("CORRECT APPROACH")
    print("="*70)
    
    child_state_unflipped = parent_state.copy()
    child_state_unflipped = game.get_next_state(child_state_unflipped, action, 1)
    
    # Check terminal BEFORE flipping
    value_correct, is_terminal_correct = game.get_value_and_terminated(child_state_unflipped, action)
    
    print(f"\nChecking terminal on UNFLIPPED state:")
    print(f"  value = {value_correct}")
    print(f"  is_terminal = {is_terminal_correct}")
    
    if is_terminal_correct and value_correct == 1:
        print("\n✅ CORRECT! Current player (1) won the game")
        print("After flipping, this should be -1 from opponent's perspective")


if __name__ == "__main__":
    main()
