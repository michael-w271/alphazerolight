#!/usr/bin/env python3
"""
Comprehensive audit of perspective handling throughout the codebase.
Check that the AI always knows it's trying to win, never helping opponent.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS


def test_case_1_ai_as_player_1():
    """Test: AI is player 1, makes a winning move"""
    print("\n" + "="*70)
    print("TEST 1: AI is Player 1, makes winning move")
    print("="*70)
    
    game = ConnectFour()
    state = game.get_initial_state()
    
    # Set up a position where player 1 can win
    state = game.get_next_state(state, 0, 1)   # P1 col 0
    state = game.get_next_state(state, 0, -1)  # P2 col 0
    state = game.get_next_state(state, 1, 1)   # P1 col 1
    state = game.get_next_state(state, 1, -1)  # P2 col 1
    state = game.get_next_state(state, 2, 1)   # P1 col 2
    state = game.get_next_state(state, 2, -1)  # P2 col 2
    # Player 1 can win with column 3
    
    print("\nBoard before winning move (player 1 to move):")
    print("  0 1 2 3 4 5 6")
    for row in range(6):
        print(f"{row} ", end="")
        for col in range(7):
            val = state[row, col]
            print("X " if val == 1 else ("O " if val == -1 else ". "), end="")
        print()
    
    # AI is player 1, needs to see this from its own perspective
    player = 1
    neutral_state = game.change_perspective(state, player)
    
    print(f"\nAI is player {player}")
    print(f"After change_perspective(state, {player}):")
    print("  All AI pieces should be 1, opponent pieces should be -1")
    print("  Top row check: AI has pieces at columns 0,1,2?")
    bottom_row = neutral_state[5, :]
    print(f"  Bottom row: {bottom_row}")
    
    # Make winning move
    action = 3
    next_state = game.get_next_state(state, action, player)
    value, is_terminal = game.get_value_and_terminated(next_state, action)
    
    print(f"\nAI plays column {action}")
    print(f"get_value_and_terminated returns: value={value}, terminal={is_terminal}")
    print(f"Expected: value=1 (AI won), terminal=True")
    
    if value == 1 and is_terminal:
        print("✅ CORRECT: AI recognizes it won")
    else:
        print("❌ BUG: AI doesn't recognize its win!")
    
    return value == 1 and is_terminal


def test_case_2_ai_as_player_minus_1():
    """Test: AI is player -1, makes a winning move"""
    print("\n" + "="*70)
    print("TEST 2: AI is Player -1, makes winning move")
    print("="*70)
    
    game = ConnectFour()
    state = game.get_initial_state()
    
    # Set up a position where player -1 can win
    state = game.get_next_state(state, 0, 1)   # P1 col 0
    state = game.get_next_state(state, 1, -1)  # P2 col 1
    state = game.get_next_state(state, 0, 1)   # P1 col 0
    state = game.get_next_state(state, 2, -1)  # P2 col 2
    state = game.get_next_state(state, 0, 1)   # P1 col 0
    state = game.get_next_state(state, 3, -1)  # P2 col 3 - WINS!
    
    print("\nBoard after player -1 wins:")
    print("  0 1 2 3 4 5 6")
    for row in range(6):
        print(f"{row} ", end="")
        for col in range(7):
            val = state[row, col]
            print("X " if val == 1 else ("O " if val == -1 else ". "), end="")
        print()
    
    # Check what get_value_and_terminated returns
    action = 3
    value, is_terminal = game.get_value_and_terminated(state, action)
    
    print(f"\nget_value_and_terminated(state, action={action}):")
    print(f"  value={value}, terminal={is_terminal}")
    print(f"  This is from the perspective of whoever just moved (player -1)")
    print(f"  Expected: value=1 (current player -1 won), terminal=True")
    
    if value == 1 and is_terminal:
        print("✅ CORRECT: Returns 1 for the winner")
    else:
        print("❌ BUG: Doesn't return 1 for winner!")
        
    # Now check from AI perspective if AI was player -1
    player = -1  # AI is player -1
    neutral_state = game.change_perspective(state, player)
    
    print(f"\nIf AI was player {player}:")
    print(f"After change_perspective, AI's winning pieces should be represented as 1")
    bottom_row = neutral_state[5, :]
    print(f"  Bottom row: {bottom_row}")
    print(f"  Expected: [0, 1, 1, 1, 0, 0, 0] (AI's pieces as 1)")
    
    expected = np.array([0, 1, 1, 1, 0, 0, 0], dtype=np.float32)
    if np.array_equal(bottom_row, expected):
        print("✅ CORRECT: AI's pieces shown as 1 after perspective flip")
    else:
        print("❌ BUG: Perspective flip incorrect!")
    
    return value == 1 and is_terminal and np.array_equal(bottom_row, expected)


def test_case_3_mcts_expand_perspective():
    """Test: MCTS expand creates children from correct perspective"""
    print("\n" + "="*70)
    print("TEST 3: MCTS expand() creates children with correct perspective")
    print("="*70)
    
    game = ConnectFour()
    model = ResNet(game, num_res_blocks=1, num_hidden=64)
    model.eval()
    
    args = {'C': 2.0, 'num_searches': 10}
    
    # Create a simple position
    state = game.get_initial_state()
    state = game.get_next_state(state, 3, 1)  # Player 1 plays column 3
    
    print("\nBoard (player -1 to move next):")
    print("  0 1 2 3 4 5 6")
    for row in range(6):
        print(f"{row} ", end="")
        for col in range(7):
            val = state[row, col]
            print("X " if val == 1 else ("O " if val == -1 else ". "), end="")
        print()
    
    # MCTS will be called from player -1's perspective
    player = -1
    neutral_state = game.change_perspective(state, player)
    
    print(f"\nAI is player {player}")
    print(f"In neutral_state, AI's perspective has opponent's piece as -1:")
    bottom_row = neutral_state[5, :]
    print(f"  Bottom row: {bottom_row}")
    print(f"  Expected: [0, 0, 0, -1, 0, 0, 0] (opponent's piece as -1)")
    
    # Create root node and test expand
    from alpha_zero_light.mcts.mcts import Node
    root = Node(game, args, neutral_state, visit_count=0)
    
    # Create a dummy policy
    policy = np.ones(7) / 7
    valid_moves = game.get_valid_moves(neutral_state)
    policy *= valid_moves
    policy /= np.sum(policy)
    
    root.expand(policy)
    
    print(f"\nRoot expanded, created {len(root.children)} children")
    
    # Check one child (e.g., action 0)
    child_0 = next(c for c in root.children if c.action_taken == 0)
    
    print(f"\nChild for action 0:")
    print(f"  This represents the state AFTER AI plays column 0")
    print(f"  From AI's opponent's perspective (flipped)")
    
    # In child.state, what was AI's piece (1) should now be opponent's (-1)
    child_bottom = child_0.state[5, :]
    print(f"  Child bottom row: {child_bottom}")
    print(f"  Expected: AI played col 0, so position [0] should have -1 (AI's piece from opp view)")
    print(f"           And position [3] should have 1 (was opponent, now flipped)")
    
    # The child should have AI's new piece at column 0 as value 1 (before flip)
    # After flip, it becomes -1
    # And the existing opponent piece (col 3) should become 1 after flip
    
    # Actually, let's trace this more carefully:
    # 1. Root state (AI perspective): opponent piece at col 3 = -1, rest = 0
    # 2. AI plays col 0: state has AI piece (+1) at col 0, opponent (-1) at col 3
    # 3. Flip perspective: AI pieces become -1, opponent pieces become 1
    # 4. Child state: should have -1 at col 0 (AI's piece), +1 at col 3 (opponent's piece)
    
    expected_child = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
    expected_child[0] = -1
    
    if np.array_equal(child_bottom, expected_child):
        print("✅ CORRECT: Child perspective properly flipped")
    else:
        print(f"⚠️  Child state: {child_bottom}")
        print(f"   Expected:    {expected_child}")
    
    # Check terminal values
    print(f"\nChild terminal info:")
    print(f"  is_terminal: {child_0.is_terminal}")
    print(f"  terminal_value: {child_0.terminal_value}")
    print(f"  Expected: is_terminal=False (game continues)")
    
    return True


def test_case_4_training_outcome_assignment():
    """Test: Training assigns correct outcomes to historical positions"""
    print("\n" + "="*70)
    print("TEST 4: Training outcome assignment")
    print("="*70)
    
    game = ConnectFour()
    
    # Simulate a simple game where player 1 wins
    memory = []
    state = game.get_initial_state()
    player = 1
    
    # Move 1: Player 1
    neutral_state = game.change_perspective(state, player)
    action_probs = np.ones(7) / 7
    memory.append((neutral_state, action_probs, player))
    state = game.get_next_state(state, 0, player)
    player = -player
    
    # Move 2: Player -1
    neutral_state = game.change_perspective(state, player)
    action_probs = np.ones(7) / 7
    memory.append((neutral_state, action_probs, player))
    state = game.get_next_state(state, 1, player)
    player = -player
    
    # Continue until player 1 wins...
    # Simulate player 1 winning
    player_who_won = 1
    value_from_winner_perspective = 1
    
    print(f"\nGame ended: Player {player_who_won} won")
    print(f"value from winner's perspective: {value_from_winner_perspective}")
    
    # This is what training does:
    print(f"\nAssigning outcomes to {len(memory)} historical positions:")
    for idx, (hist_state, hist_probs, hist_player) in enumerate(memory):
        hist_outcome = value_from_winner_perspective if hist_player == player_who_won else game.get_opponent_value(value_from_winner_perspective)
        print(f"  Position {idx}: played by player {hist_player}, outcome={hist_outcome}")
        
        if hist_player == player_who_won:
            if hist_outcome == 1:
                print(f"    ✅ CORRECT: Winner's move gets +1")
            else:
                print(f"    ❌ BUG: Winner's move should get +1, got {hist_outcome}")
        else:
            if hist_outcome == -1:
                print(f"    ✅ CORRECT: Loser's move gets -1")
            else:
                print(f"    ❌ BUG: Loser's move should get -1, got {hist_outcome}")
    
    return True


def main():
    print("="*70)
    print("COMPREHENSIVE PERSPECTIVE AUDIT")
    print("="*70)
    print("\nChecking that AI always knows it's the AI and is trying to win...")
    
    test_case_1_ai_as_player_1()
    test_case_2_ai_as_player_minus_1()
    test_case_3_mcts_expand_perspective()
    test_case_4_training_outcome_assignment()
    
    print("\n" + "="*70)
    print("AUDIT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
