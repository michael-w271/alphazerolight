#!/usr/bin/env python3
"""Reproduce the exact scenario from test_perspective_audit that failed"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour


def print_board(state, title=""):
    if title:
        print(f"\n{title}")
    print("  0 1 2 3 4 5 6")
    for row in range(6):
        print(f"{row} ", end="")
        for col in range(7):
            val = state[row, col]
            print("X " if val == 1 else ("O " if val == -1 else ". "), end="")
        print()


def test_exact_failing_case():
    game = ConnectFour()
    state = game.get_initial_state()
    
    # From test_case_2_ai_as_player_minus_1():
    state = game.get_next_state(state, 0, 1)   # P1 col 0
    state = game.get_next_state(state, 1, -1)  # P2 col 1
    state = game.get_next_state(state, 0, 1)   # P1 col 0
    state = game.get_next_state(state, 2, -1)  # P2 col 2
    state = game.get_next_state(state, 0, 1)   # P1 col 0
    state = game.get_next_state(state, 3, -1)  # P2 col 3 - SHOULD WIN!
    
    print_board(state, "Board after player -1 plays column 3:")
    
    # Check columns
    print("\nColumn contents:")
    for col in range(4):
        print(f"Column {col}: ", end="")
        for row in range(6):
            if state[row, col] != 0:
                print(f"row{row}={state[row, col]:.0f} ", end="")
        print()
    
    action = 3
    value, is_terminal = game.get_value_and_terminated(state, action)
    
    print(f"\nget_value_and_terminated(state, action={action}):")
    print(f"  value={value}, terminal={is_terminal}")
    
    # Manual check: does player -1 have 4 in a row?
    print("\nManual check for player -1 horizontal win:")
    print(f"Bottom row: {state[5, :]}")
    print(f"Player -1 at cols 1,2,3? ", end="")
    if state[5, 1] == -1 and state[5, 2] == -1 and state[5, 3] == -1:
        print("Yes - has 3 in a row (not 4)")
    
    # Check if there's actually a win
    is_win = game.check_win(state, action)
    print(f"\ncheck_win(state, action={action}): {is_win}")
    
    # Debug check_win
    print("\nDebug check_win for column 3:")
    column = 3
    for r in range(6):
        if state[r, column] != 0:
            row = r
            player = state[r, column]
            print(f"  First non-empty row: {r}, player={player}")
            
            # Check horizontal from this row
            count = 1
            # Left
            c = column - 1
            while c >= 0 and state[row, c] == player:
                count += 1
                c -= 1
            # Right
            c = column + 1
            while c < 7 and state[row, c] == player:
                count += 1
                c += 1
            print(f"  Horizontal count: {count}")
            
            # Check vertical
            count = 1
            r2 = row + 1
            while r2 < 6 and state[r2, column] == player:
                count += 1
                r2 += 1
            print(f"  Vertical count: {count}")
            break


if __name__ == "__main__":
    test_exact_failing_case()
