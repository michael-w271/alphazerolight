#!/usr/bin/env python3
"""Debug check_win for vertical wins"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour


def test_vertical_win():
    game = ConnectFour()
    state = game.get_initial_state()
    
    print("Test: Player -1 vertical win in column 1")
    print("="*50)
    
    # Player 1 plays column 0
    state = game.get_next_state(state, 0, 1)
    print("\nAfter P1 plays col 0:")
    print_board(state)
    
    # Player -1 plays column 1 (1st piece)
    state = game.get_next_state(state, 1, -1)
    print("\nAfter P-1 plays col 1:")
    print_board(state)
    
    # Player 1 plays column 0
    state = game.get_next_state(state, 0, 1)
    print("\nAfter P1 plays col 0:")
    print_board(state)
    
    # Player -1 plays column 1 (2nd piece)
    state = game.get_next_state(state, 1, -1)
    print("\nAfter P-1 plays col 1:")
    print_board(state)
    
    # Player 1 plays column 0
    state = game.get_next_state(state, 0, 1)
    print("\nAfter P1 plays col 0:")
    print_board(state)
    
    # Player -1 plays column 1 (3rd piece)
    state = game.get_next_state(state, 1, -1)
    print("\nAfter P-1 plays col 1:")
    print_board(state)
    
    # Player 1 plays column 2
    state = game.get_next_state(state, 2, 1)
    print("\nAfter P1 plays col 2:")
    print_board(state)
    
    # Player -1 plays column 1 (4th piece - SHOULD WIN!)
    state = game.get_next_state(state, 1, -1)
    print("\nAfter P-1 plays col 1 (4th piece):")
    print_board(state)
    
    # Check if it's a win
    is_win = game.check_win(state, 1)
    print(f"\ncheck_win(state, action=1): {is_win}")
    print(f"Expected: True (4 in a row vertically)")
    
    # Debug: check what check_win is looking at
    print("\nDebug check_win logic:")
    column = 1
    for r in range(6):
        if state[r, column] != 0:
            print(f"  First non-empty row in column {column}: row {r}, value={state[r, column]}")
            player = state[r, column]
            
            # Check vertical downward
            count = 1
            r2 = r + 1
            while r2 < 6 and state[r2, column] == player:
                count += 1
                r2 += 1
            print(f"  Counting down from row {r}: count={count}")
            break
    
    if is_win:
        print("\n✅ PASS: Vertical win detected")
    else:
        print("\n❌ FAIL: Vertical win NOT detected!")
    
    return is_win


def print_board(state):
    print("  0 1 2 3 4 5 6")
    for row in range(6):
        print(f"{row} ", end="")
        for col in range(7):
            val = state[row, col]
            print("X " if val == 1 else ("O " if val == -1 else ". "), end="")
        print()


if __name__ == "__main__":
    test_vertical_win()
