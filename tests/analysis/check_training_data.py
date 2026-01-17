#!/usr/bin/env python3
"""
Check if training includes terminal positions or only pre-move positions.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour

def simulate_training_data_collection():
    game = ConnectFour()
    
    print("="*70)
    print("SIMULATING TRAINING DATA COLLECTION")
    print("="*70)
    
    # Simulate a simple game where player 1 wins
    memory = []
    state = game.get_initial_state()
    player = 1
    move_num = 0
    
    moves = [
        (0, 1),   # P1 plays col 0
        (0, -1),  # P2 plays col 0
        (1, 1),   # P1 plays col 1
        (1, -1),  # P2 plays col 1
        (2, 1),   # P1 plays col 2
        (2, -1),  # P2 plays col 2
        (3, 1),   # P1 plays col 3 - WINS!
    ]
    
    for col, p in moves:
        move_num += 1
        print(f"\n--- Move {move_num}: Player {p} plays column {col} ---")
        
        # Before move: store current state
        neutral_state = game.change_perspective(state, p)
        action_probs = np.ones(7) / 7  # dummy
        memory.append((neutral_state, action_probs, p))
        print(f"Stored state BEFORE move (player {p}'s perspective)")
        print(f"  State bottom row: {neutral_state[5, :]}")
        
        # Make move
        state = game.get_next_state(state, col, p)
        value, is_terminal = game.get_value_and_terminated(state, col)
        
        print(f"After move:")
        print(f"  State bottom row: {state[5, :]}")
        print(f"  value={value}, terminal={is_terminal}")
        
        if is_terminal:
            print(f"\n{'='*70}")
            print(f"GAME ENDED: Player {p} won!")
            print(f"{'='*70}")
            print(f"\nProcessing {len(memory)} historical states:")
            
            for idx, (hist_state, hist_probs, hist_player) in enumerate(memory):
                hist_outcome = value if hist_player == p else game.get_opponent_value(value)
                
                print(f"\n  State {idx}: Player {hist_player}'s turn")
                print(f"    Bottom row: {hist_state[5, :]}")
                print(f"    Outcome assigned: {hist_outcome:+.0f}")
                print(f"    Explanation: ", end="")
                
                if hist_player == p:
                    print(f"Same as winner (player {p}), gets +1")
                else:
                    print(f"Loser (player {-p}), gets -1")
            
            # Key question: Is the FINAL winning state included?
            print(f"\n{'='*70}")
            print("KEY OBSERVATION:")
            print(f"{'='*70}")
            print(f"The winning state AFTER move {move_num} is:")
            print(f"  {state[5, :]} (has 4 X's)")
            print(f"\nThis state is NOT in the training data!")
            print(f"Training data only has {len(memory)} states (pre-move states)")
            print(f"\nThe model is NEVER shown terminal positions during training.")
            print(f"It only learns to predict outcomes for non-terminal positions.")
            
            break
        
        player = game.get_opponent(player)


if __name__ == "__main__":
    simulate_training_data_collection()
