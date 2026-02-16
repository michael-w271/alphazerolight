#!/usr/bin/env python3
"""
DETAILED MCTS TRACE - Step by step analysis of what happens
when searching from a position with an immediate win available.
"""
import sys
sys.path.insert(0, 'src')
import numpy as np
from alpha_zero_light.game.connect_four import ConnectFour

game = ConnectFour()

print("="*80)
print("DETAILED MCTS FLOW ANALYSIS")
print("="*80)

# Root state: Player 1 to move, has 3 in a row in column 3
root_state = np.zeros((6, 7), dtype=np.float32)
root_state[5, 3] = 1  # Bottom
root_state[4, 3] = 1  
root_state[3, 3] = 1

print("\n1. ROOT STATE (Player 1 to move):")
print("   This is from Player 1's perspective (1 = our pieces, -1 = opponent)")
for r in range(6):
    print('   ' + ' '.join(['X' if root_state[r,c]==1 else 'O' if root_state[r,c]==-1 else '.' for c in range(7)]))
print(f"   Who moves next: Player 1")
print(f"   State values: min={root_state.min()}, max={root_state.max()}")

# Test what happens when we play column 3
print("\n2. SIMULATE MOVE: Player 1 plays column 3")
after_move = game.get_next_state(root_state, 3, 1)
print("   After get_next_state(root_state, action=3, player=1):")
for r in range(6):
    print('   ' + ' '.join(['X' if after_move[r,c]==1 else 'O' if after_move[r,c]==-1 else '.' for c in range(7)]))

value, is_terminal = game.get_value_and_terminated(after_move, 3)
print(f"   get_value_and_terminated returns: value={value}, terminal={is_terminal}")
print(f"   ✓ This is correct - Player 1 won!")

# Now simulate what MCTS does when expanding this child
print("\n3. MCTS NODE EXPANSION (what mcts.py does):")
print("   In expand() method (lines 43-53 of mcts.py):")

# This is what line 49 does
child_state = root_state.copy()
child_state = game.get_next_state(child_state, 3, 1)
print("   a) get_next_state(parent_state, action=3, player=1):")
for r in range(6):
    print('      ' + ' '.join(['X' if child_state[r,c]==1 else 'O' if child_state[r,c]==-1 else '.' for c in range(7)]))
print(f"      State values: min={child_state.min()}, max={child_state.max()}")

# This is what line 50 does
child_state = game.change_perspective(child_state, player=-1)
print("   b) change_perspective(child_state, player=-1):")
print("      This multiplies ALL values by -1")
for r in range(6):
    print('      ' + ' '.join(['X' if child_state[r,c]==1 else 'O' if child_state[r,c]==-1 else '.' for c in range(7)]))
print(f"      State values: min={child_state.min()}, max={child_state.max()}")
print(f"      ⚠️  Now 1's became -1's and -1's became 1's!")

# This is stored as the child node's state
print("\n4. CHILD NODE STATE:")
print("   The child node stores this perspective-flipped state")
print("   From THIS state's perspective:")
print("   - 'Player 1' (value=1) pieces are what were originally Player -1")  
print("   - 'Player -1' (value=-1) pieces are what were originally Player 1")

# When MCTS checks if this child is terminal
print("\n5. TERMINAL CHECK (what happens in search loop):")
print("   In search() method (line 97 of mcts.py):")
print(f"   get_value_and_terminated(child_state, action=3)")

value_flipped, is_terminal_flipped = game.get_value_and_terminated(child_state, 3)
print(f"   Returns: value={value_flipped}, terminal={is_terminal_flipped}")

print("\n6. THE PROBLEM:")
print("   check_win() looks for 4-in-a-row of 'Player 1' (value=1)")
print("   In the FLIPPED child state, column 3 has four -1's, not 1's!")
print("   So check_win() returns FALSE even though this is a winning position!")

# Verify this
print("\n7. VERIFICATION - What does check_win see?")
print("   Looking at column 3 in the child (flipped) state:")
col_3_values = [child_state[r, 3] for r in range(6)]
print(f"   Column 3 values (top to bottom): {col_3_values}")
print(f"   These are all -1.0, not 1.0")
print(f"   So check_win() doesn't detect a win!")

print("\n8. ROOT CAUSE:")
print("   ❌ get_value_and_terminated() checks for 4-in-a-row of value=1")
print("   ❌ After change_perspective(player=-1), winning pieces become -1")
print("   ❌ So the immediate win is NOT DETECTED")
print("   ❌ MCTS never learns that column 3 is the winning move!")

print("\n9. THE FIX:")
print("   Option A: Don't flip perspective in expand() - keep states in original perspective")
print("   Option B: Make check_win() look for 4-in-a-row of EITHER 1 or -1")
print("   Option C: Check terminal BEFORE flipping perspective")

print("\n" + "="*80)
