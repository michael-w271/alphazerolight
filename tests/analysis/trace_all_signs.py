#!/usr/bin/env python3
"""
Trace through EXACT sign flow from training to inference.
Find where the sign gets flipped incorrectly.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour

def trace_training_signs():
    game = ConnectFour()
    
    print("="*70)
    print("TRACING SIGN FLOW - TRAINING")
    print("="*70)
    
    # Simple game: Player 1 wins
    state = game.get_initial_state()
    
    # Move 1: Player 1 plays column 0
    player = 1
    neutral_state_move1 = game.change_perspective(state, player)
    print(f"\n--- MOVE 1: Player {player} ---")
    print(f"State before move (from player {player}'s view): {neutral_state_move1[5, :]}")
    state = game.get_next_state(state, 0, player)
    player = -player
    
    # Move 2: Player -1 plays column 1  
    neutral_state_move2 = game.change_perspective(state, player)
    print(f"\n--- MOVE 2: Player {player} ---")
    print(f"State before move (from player {player}'s view): {neutral_state_move2[5, :]}")
    state = game.get_next_state(state, 1, player)
    player = -player
    
    # ... fast forward to winning move
    state = game.get_next_state(state, 1, 1)
    state = game.get_next_state(state, 2, -1)
    state = game.get_next_state(state, 2, 1)
    state = game.get_next_state(state, 3, -1)
    
    # Final move: Player 1 wins
    player = 1
    neutral_state_final = game.change_perspective(state, player)
    print(f"\n--- FINAL MOVE: Player {player} (will win) ---")
    print(f"State before move (from player {player}'s view): {neutral_state_final[5, :]}")
    print(f"  Player 1 has pieces at cols 0,1,2 (shown as +1)")
    state = game.get_next_state(state, 3, player)
    value, is_terminal = game.get_value_and_terminated(state, 3)
    
    print(f"\nAfter move:")
    print(f"  State: {state[5, :]}")
    print(f"  value={value}, terminal={is_terminal}")
    print(f"  value is from player {player}'s perspective (who just moved)")
    
    # Now assign outcomes
    print(f"\n{'='*70}")
    print("TRAINING DATA OUTCOME ASSIGNMENT")
    print(f"{'='*70}")
    
    memory = [
        (neutral_state_move1, None, 1),
        (neutral_state_move2, None, -1),
        (neutral_state_final, None, 1),
    ]
    
    winning_player = 1
    value_from_winner_perspective = 1
    
    print(f"\nWinner: Player {winning_player}")
    print(f"Value returned: {value_from_winner_perspective}")
    
    for idx, (hist_state, _, hist_player) in enumerate(memory):
        hist_outcome = value_from_winner_perspective if hist_player == winning_player else game.get_opponent_value(value_from_winner_perspective)
        
        encoded = game.get_encoded_state(hist_state)
        
        print(f"\nTraining Example {idx}:")
        print(f"  Player who made this move: {hist_player}")
        print(f"  State (from their view): {hist_state[5, :]}")
        print(f"  Encoded - AI pieces (channel 2): {encoded[2, 5, :]}")
        print(f"  Outcome assigned: {hist_outcome:+.0f}")
        
        if hist_player == 1:
            print(f"  ✅ Player 1 won, gets +1")
        else:
            print(f"  ❌ Player -1 lost, gets -1")


def trace_inference_signs():
    game = ConnectFour()
    
    print(f"\n{'='*70}")
    print("TRACING SIGN FLOW - INFERENCE")
    print(f"{'='*70}")
    
    # Create the EXACT same position as training example 2 (player -1's second move)
    # In that example: state before move showed [-1, -1, -0, ...]
    # And it got outcome = -1 (because player -1 lost)
    
    # Build that state
    state = game.get_initial_state()
    state = game.get_next_state(state, 0, 1)   # P1 col 0
    state = game.get_next_state(state, 1, -1)  # P-1 col 1
    state = game.get_next_state(state, 1, 1)   # P1 col 1
    state = game.get_next_state(state, 2, -1)  # P-1 col 2
    state = game.get_next_state(state, 2, 1)   # P1 col 2
    state = game.get_next_state(state, 3, -1)  # P-1 col 3
    
    # Now it's player 1's turn, about to play column 3 and win
    # Player -1 sees this position
    
    player = -1
    neutral_state = game.change_perspective(state, player)
    
    print(f"\nInference: Player {player} evaluating position")
    print(f"State (from player {player}'s view): {neutral_state[5, :]}")
    print(f"This matches training example where outcome was -1")
    
    encoded = game.get_encoded_state(neutral_state)
    print(f"Encoded - AI pieces (channel 2): {encoded[2, 5, :]}")
    
    print(f"\n❓ QUESTION: What should the model predict?")
    print(f"  During training: This state (from player -1's view) got outcome = -1")
    print(f"  At inference: Model should predict value ≈ -1")
    print(f"  This means: 'I (player -1) am going to lose'")
    
    # Now check from player 1's perspective  
    print(f"\n{'='*70}")
    player = 1
    neutral_state = game.change_perspective(state, player)
    
    print(f"\nInference: Player {player} evaluating SAME position")
    print(f"State (from player {player}'s view): {neutral_state[5, :]}")
    
    encoded = game.get_encoded_state(neutral_state)
    print(f"Encoded - AI pieces (channel 2): {encoded[2, 5, :]}")
    
    print(f"\n❓ QUESTION: What should the model predict?")
    print(f"  During training: This state (from player 1's view) got outcome = +1")
    print(f"  At inference: Model should predict value ≈ +1")
    print(f"  This means: 'I (player 1) am going to win'")
    
    print(f"\n{'='*70}")
    print("CRITICAL INSIGHT")
    print(f"{'='*70}")
    print("\nThe SAME board position gets DIFFERENT encodings depending on who's turn it is:")
    print("  - Player 1's view: Their pieces = +1 in state, encoded as channel 2")
    print("  - Player -1's view: Their pieces = +1 in state, encoded as channel 2")
    print("\nBut they have OPPOSITE outcomes:")
    print("  - Player 1's view: outcome = +1 (going to win)")
    print("  - Player -1's view: outcome = -1 (going to lose)")
    print("\n✅ This is CORRECT if the model learns position-to-outcome mapping")


def trace_mcts_perspective():
    game = ConnectFour()
    
    print(f"\n{'='*70}")
    print("TRACING SIGN FLOW - MCTS")
    print(f"{'='*70}")
    
    # Position where player 1 can win at column 3
    state = game.get_initial_state()
    state = game.get_next_state(state, 0, 1)
    state = game.get_next_state(state, 1, -1)
    state = game.get_next_state(state, 1, 1)
    state = game.get_next_state(state, 2, -1)
    state = game.get_next_state(state, 2, 1)
    state = game.get_next_state(state, 3, -1)
    
    # It's player 1's turn
    player = 1
    neutral_state = game.change_perspective(state, player)
    
    print(f"\nRoot: Player {player}'s turn")
    print(f"State: {neutral_state[5, :]}")
    print(f"Player 1 can win by playing column 3")
    
    # Simulate MCTS expanding action 3
    print(f"\n--- MCTS expands action 3 ---")
    
    # 1. Apply move from current player's perspective
    child_state = neutral_state.copy()
    child_state = game.get_next_state(child_state, 3, 1)
    print(f"\n1. After applying move (current player = 1):")
    print(f"   State: {child_state[5, :]}")
    
    # 2. Check terminal BEFORE flipping
    terminal_value, is_terminal = game.get_value_and_terminated(child_state, 3)
    print(f"\n2. Check terminal (BEFORE flip):")
    print(f"   value={terminal_value}, terminal={is_terminal}")
    print(f"   This is from current player's (1) perspective")
    print(f"   value=1 means player 1 won ✅")
    
    # 3. Flip perspective
    child_state = game.change_perspective(child_state, player=-1)
    print(f"\n3. After flipping perspective:")
    print(f"   State: {child_state[5, :]}")
    print(f"   Now from opponent's perspective")
    
    # 4. Adjust terminal value
    if is_terminal and terminal_value != 0:
        terminal_value = -terminal_value
    print(f"\n4. Adjust terminal value for child node:")
    print(f"   terminal_value = {terminal_value}")
    print(f"   This is from child's (opponent's) perspective")
    print(f"   value=-1 means opponent lost (we won) ✅")
    
    # 5. Backpropagate
    print(f"\n5. Backpropagate value={terminal_value} to root")
    print(f"   Root gets: value_sum += {terminal_value}")
    print(f"   Root Q-value will be: {terminal_value}/{1} = {terminal_value}")
    
    print(f"\n❌ WAIT! This is WRONG!")
    print(f"   If we play action 3 and WIN, root should get POSITIVE value")
    print(f"   But we're adding {terminal_value} (negative) to root!")
    
    print(f"\n🔍 Checking backpropagate logic...")
    print(f"   Child has terminal_value={terminal_value} (from opponent view: loss)")
    print(f"   Backpropagate flips: root gets -({terminal_value}) = {-terminal_value}")
    print(f"   ✅ So root actually gets +1, which is CORRECT!")


if __name__ == "__main__":
    trace_training_signs()
    trace_inference_signs()
    trace_mcts_perspective()
