import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

from alpha_zero_light.game.gomoku import Gomoku

def test_gomoku_logic():
    game = Gomoku()
    print("Testing Gomoku Logic...")

    # Test 1: Initial State
    state = game.get_initial_state()
    assert state.shape == (15, 15)
    assert np.all(state == 0)
    print("✅ Initial state correct")

    # Test 2: Make Move
    action = 7 * 15 + 7 # Center
    next_state = game.get_next_state(state.copy(), action, 1)
    assert next_state[7, 7] == 1
    print("✅ Move execution correct")

    # Test 3: Valid Moves
    valid_moves = game.get_valid_moves(next_state)
    assert valid_moves[action] == 0
    assert np.sum(valid_moves) == 15 * 15 - 1
    print("✅ Valid moves correct")

    # Test 4: Horizontal Win
    state = np.zeros((15, 15))
    for i in range(5):
        state[0, i] = 1
    # The last move was at (0, 4)
    action = 4
    assert game.check_win(state, action)
    print("✅ Horizontal win detected")

    # Test 5: Vertical Win
    state = np.zeros((15, 15))
    for i in range(5):
        state[i, 0] = 1
    action = 4 * 15
    assert game.check_win(state, action)
    print("✅ Vertical win detected")

    # Test 6: Diagonal Win
    state = np.zeros((15, 15))
    for i in range(5):
        state[i, i] = 1
    action = 4 * 15 + 4
    assert game.check_win(state, action)
    print("✅ Diagonal win detected")

    # Test 7: Anti-Diagonal Win
    state = np.zeros((15, 15))
    for i in range(5):
        state[i, 4-i] = 1
    action = 4 * 15 + 0
    assert game.check_win(state, action)
    print("✅ Anti-diagonal win detected")

    print("🎉 All Gomoku logic tests passed!")

if __name__ == "__main__":
    test_gomoku_logic()
