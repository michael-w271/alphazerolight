#!/usr/bin/env python3
"""
Verify that the encoding fix is actually being used in the running code.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Check the trainer.py code that's actually loaded
from alpha_zero_light.training import trainer
import inspect

# Get the self_play_game function source
source = inspect.getsource(trainer.self_play_game)

print("="*70)
print("VERIFYING ENCODING FIX IS ACTIVE")
print("="*70)

if "state_from_player_perspective = game.change_perspective(hist_state, hist_player)" in source:
    print("\n✅ ENCODING FIX IS PRESENT IN LOADED CODE!")
    print("\nThe fix converts state to player perspective before encoding:")
    print("   state_from_player_perspective = game.change_perspective(hist_state, hist_player)")
    
    # Extract the relevant section
    lines = source.split('\n')
    for i, line in enumerate(lines):
        if 'state_from_player_perspective' in line:
            print(f"\nContext (lines around fix):")
            for j in range(max(0, i-3), min(len(lines), i+5)):
                print(f"  {lines[j]}")
            break
else:
    print("\n❌ ENCODING FIX NOT FOUND IN LOADED CODE!")
    print("   The code might not have reloaded properly.")
    print("   You may need to restart the training process.")

print("\n" + "="*70)
