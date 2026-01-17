#!/usr/bin/env python3
"""
Test get_value_and_terminated to see if it returns correct values.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from alpha_zero_light.game.connect_four import ConnectFour

game = ConnectFour(6, 7, 4)

print("="*70)
print("TESTING get_value_and_terminated")
print("="*70)

# Scenario 1: Player 1 wins
print("\n1. Player 1 wins (has 4 in a row):")
state1 = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0],
])
print(state1)
value, is_terminal = game.get_value_and_terminated(state1, 3)  # Last move was column 3
print(f"   value={value}, is_terminal={is_terminal}")
print(f"   Expected: value=1 (Player 1 won)")

# Scenario 2: Player -1 wins
print("\n2. Player -1 wins (has 4 in a row):")
state2 = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, -1, 0, 0, 0],
])
print(state2)
value, is_terminal = game.get_value_and_terminated(state2, 3)
print(f"   value={value}, is_terminal={is_terminal}")
print(f"   Expected: value=1 (but this is WRONG - should reflect Player -1 won!)")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)
print("""
The function returns value=1 regardless of WHO won.
It just returns 1 if someone won, 0 if draw.

This means in the training code, we need to know which PLAYER just moved
to interpret the value correctly:

   if player who just moved is Player 1 and value=1:
       → Player 1 won (value should be +1 for Player 1's moves)
   
   if player who just moved is Player -1 and value=1:
       → Player -1 won (value should be -1 for Player 1's moves!)

The training code at line 147:
   hist_outcome = value if hist_player == player else game.get_opponent_value(value)

This works IF:
   - 'player' = the player who just moved (winner)
   - 'value' = 1
   - 'hist_player' = the player whose move we're labeling

So if Player -1 won:
   - player = -1
   - value = 1
   - For Player 1's moves: hist_player=1, player=-1, so hist_outcome = -1 ✅

BUT this only works if 'player' is correctly set to whoever just moved!
Let me verify this is the case...
""")

print("\nChecking trainer.py logic at line 143:")
print("   state = game.get_next_state(state, action, player)")
print("   value, is_terminal = game.get_value_and_terminated(state, action)")
print("   ")
print("   At this point, 'player' is the player who JUST moved.")
print("   So if is_terminal and value=1, then 'player' won.")
print("   ")
print("   For hist_player == player: they won, outcome = +1 ✅")
print("   For hist_player != player: they lost, outcome = -1 ✅")
print("   ")
print("   This SHOULD be correct!")
