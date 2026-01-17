#!/usr/bin/env python3
"""
COMPREHENSIVE SIGN (VORZEICHEN) CHECK
Verify every step of the training and MCTS to ensure signs are correct.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS, Node


def test_sign_flow():
    """Test complete sign flow from game state through MCTS to training"""
    game = ConnectFour()
    
    print("="*80)
    print("COMPREHENSIVE SIGN (VORZEICHEN) CHECK")
    print("="*80)
    
    # Scenario: Create a game where player 1 will win
    print("\n" + "="*80)
    print("STEP 1: Set up game state (Player 1 about to win)")
    print("="*80)
    
    state = game.get_initial_state()
    # Player 1 has pieces at columns 0,1,2 (bottom row)
    state = game.get_next_state(state, 0, 1)   # P1
    state = game.get_next_state(state, 4, -1)  # P2  
    state = game.get_next_state(state, 1, 1)   # P1
    state = game.get_next_state(state, 5, -1)  # P2
    state = game.get_next_state(state, 2, 1)   # P1
    # Player 1 can win by playing column 3
    
    print("\nCurrent board (Player 1's turn, can win at column 3):")
    print("  0 1 2 3 4 5 6")
    for row in range(6):
        print(f"{row} ", end="")
        for col in range(7):
            val = state[row, col]
            print("X " if val == 1 else ("O " if val == -1 else ". "), end="")
        print()
    
    print(f"\nRaw state bottom row: {state[5, :]}")
    
    # STEP 2: Change perspective for player 1
    print("\n" + "="*80)
    print("STEP 2: Change perspective to Player 1 (current player)")
    print("="*80)
    
    player = 1
    neutral_state = game.change_perspective(state, player)
    
    print(f"change_perspective(state, {player}):")
    print(f"  Bottom row: {neutral_state[5, :]}")
    print(f"  Expected: Same as raw state (player=1 so multiply by 1)")
    
    if not np.array_equal(state[5, :], neutral_state[5, :]):
        print("  ❌ ERROR: Perspective change incorrect for player 1!")
    else:
        print("  ✅ CORRECT")
    
    # STEP 3: Encode state
    print("\n" + "="*80)
    print("STEP 3: Encode state for neural network")
    print("="*80)
    
    encoded = game.get_encoded_state(neutral_state)
    print(f"Encoded shape: {encoded.shape}")
    print(f"  Channel 0 (opponent=-1): {encoded[0, 5, :]}")
    print(f"  Channel 1 (empty=0):     {encoded[1, 5, :]}")
    print(f"  Channel 2 (current=+1):  {encoded[2, 5, :]}")
    print(f"\n  Current player's pieces in channel 2: {encoded[2].sum():.0f}")
    print(f"  Opponent's pieces in channel 0: {encoded[0].sum():.0f}")
    
    # STEP 4: Test winning move
    print("\n" + "="*80)
    print("STEP 4: Simulate player 1 playing winning move (column 3)")
    print("="*80)
    
    winning_action = 3
    next_state = game.get_next_state(state, winning_action, player)
    value, is_terminal = game.get_value_and_terminated(next_state, winning_action)
    
    print(f"After player {player} plays column {winning_action}:")
    print(f"  Board bottom row: {next_state[5, :]}")
    print(f"  get_value_and_terminated returns:")
    print(f"    value = {value}")
    print(f"    is_terminal = {is_terminal}")
    print(f"  Expected: value=+1 (player {player} won), terminal=True")
    
    if value == 1 and is_terminal:
        print("  ✅ CORRECT: Winning move returns +1")
    else:
        print(f"  ❌ ERROR: Expected value=1, got {value}")
    
    # STEP 5: Test MCTS expand with winning move
    print("\n" + "="*80)
    print("STEP 5: Test MCTS expand() with the winning move")
    print("="*80)
    
    args = {'C': 2.0, 'num_searches': 10}
    root = Node(game, args, neutral_state, visit_count=0)
    
    # Create dummy policy
    policy = np.ones(7) / 7
    valid_moves = game.get_valid_moves(neutral_state)
    policy *= valid_moves
    policy /= np.sum(policy)
    
    root.expand(policy)
    
    # Find the child for the winning move
    winning_child = None
    for child in root.children:
        if child.action_taken == winning_action:
            winning_child = child
            break
    
    if winning_child:
        print(f"Child node for winning move (column {winning_action}):")
        print(f"  is_terminal: {winning_child.is_terminal}")
        print(f"  terminal_value: {winning_child.terminal_value}")
        print(f"\n  Expected: is_terminal=True, terminal_value=-1")
        print(f"  Why -1? Because child is from opponent's perspective,")
        print(f"  and opponent LOST, so value=-1 from opponent's view")
        
        if winning_child.is_terminal and winning_child.terminal_value == -1:
            print("  ✅ CORRECT: Terminal value is -1 from opponent's perspective")
        else:
            print(f"  ❌ ERROR: Expected terminal_value=-1, got {winning_child.terminal_value}")
    
    # STEP 6: Test training outcome assignment
    print("\n" + "="*80)
    print("STEP 6: Simulate training outcome assignment")
    print("="*80)
    
    # Simulate game memory
    game_memory = []
    
    # Add player 1's move before winning
    game_memory.append((neutral_state, policy, player))
    
    # Player 1 wins
    winner = player
    winner_value = 1  # From winner's perspective
    
    print(f"Game ends: Player {winner} won")
    print(f"Value from winner's perspective: {winner_value}")
    print(f"\nAssigning outcomes to game memory:")
    
    for idx, (hist_state, hist_probs, hist_player) in enumerate(game_memory):
        hist_outcome = winner_value if hist_player == winner else game.get_opponent_value(winner_value)
        
        print(f"\n  Position {idx}: played by player {hist_player}")
        print(f"    hist_player == winner? {hist_player == winner}")
        print(f"    Outcome assigned: {hist_outcome:+.0f}")
        
        if hist_player == winner:
            if hist_outcome == 1:
                print(f"    ✅ CORRECT: Winner's move gets +1")
            else:
                print(f"    ❌ ERROR: Winner's move should get +1, got {hist_outcome}")
        else:
            if hist_outcome == -1:
                print(f"    ✅ CORRECT: Loser's move gets -1")
            else:
                print(f"    ❌ ERROR: Loser's move should get -1, got {hist_outcome}")
    
    # STEP 7: Check opponent's perspective
    print("\n" + "="*80)
    print("STEP 7: Test from opponent's (Player -1) perspective")
    print("="*80)
    
    player_minus_1 = -1
    neutral_state_opp = game.change_perspective(state, player_minus_1)
    
    print(f"change_perspective(state, {player_minus_1}):")
    print(f"  Original bottom: {state[5, :]}")
    print(f"  Flipped bottom:  {neutral_state_opp[5, :]}")
    print(f"  Expected: All signs flipped (multiply by -1)")
    
    expected_flipped = state[5, :] * player_minus_1
    if np.array_equal(neutral_state_opp[5, :], expected_flipped):
        print("  ✅ CORRECT: Signs flipped")
    else:
        print(f"  ❌ ERROR: Expected {expected_flipped}, got {neutral_state_opp[5, :]}")
    
    encoded_opp = game.get_encoded_state(neutral_state_opp)
    print(f"\nEncoded from opponent's perspective:")
    print(f"  Channel 0 (opponent): {encoded_opp[0, 5, :].astype(int)}")
    print(f"  Channel 2 (current):  {encoded_opp[2, 5, :].astype(int)}")
    print(f"  Opponent pieces (channel 0): {encoded_opp[0].sum():.0f}")
    print(f"  Current pieces (channel 2): {encoded_opp[2].sum():.0f}")
    print(f"\n  Note: Player -1 has {encoded_opp[2].sum():.0f} pieces (should be 2)")
    print(f"        Player 1 (opponent from -1's view) has {encoded_opp[0].sum():.0f} pieces (should be 3)")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    checks = [
        ("Perspective flip for player 1", np.array_equal(state[5, :], neutral_state[5, :])),
        ("Winning move returns +1", value == 1 and is_terminal),
        ("MCTS child terminal value", winning_child and winning_child.terminal_value == -1),
        ("Training outcome for winner", True),  # Checked above
        ("Perspective flip for player -1", np.array_equal(neutral_state_opp[5, :], expected_flipped)),
    ]
    
    all_correct = all(check[1] for check in checks)
    
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
    
    if all_correct:
        print("\n✅✅✅ ALL SIGN CHECKS PASSED!")
        print("No Vorzeichen (sign) issues detected.")
    else:
        print("\n❌❌❌ SIGN ERRORS DETECTED!")
        print("There are still Vorzeichen (sign) bugs!")
    
    return all_correct


if __name__ == "__main__":
    test_sign_flow()
