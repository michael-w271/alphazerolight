"""Quick test for edge win detection"""
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from alpha_zero_light.game.gomoku_gpu import GomokuGPU

def test_edge_wins():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    game = GomokuGPU(board_size=9, device=device)
    
    # Test 1: Horizontal win at row 0
    state = game.get_initial_state(1)
    for i in range(5):
        state[0, 0, 0, i] = 1  # Top row, first 5 positions
    
    has_win = game.check_win(state, player=1)
    print(f"Test 1 - Horizontal win at row 0: {'PASS' if has_win.item() else 'FAIL'}")
    assert has_win.item(), "Should detect win at row 0"
    
    # Test 2: Vertical win at col 0
    state = game.get_initial_state(1)
    for i in range(5):
        state[0, 0, i, 0] = 1  # Left column, first 5 positions
    
    has_win = game.check_win(state, player=1)
    print(f"Test 2 - Vertical win at col 0: {'PASS' if has_win.item() else 'FAIL'}")
    assert has_win.item(), "Should detect win at col 0"
    
    # Test 3: Horizontal win at row 8
    state = game.get_initial_state(1)
    for i in range(5):
        state[0, 0, 8, i] = 1  # Bottom row, first 5 positions
    
    has_win = game.check_win(state, player=1)
    print(f"Test 3 - Horizontal win at row 8: {'PASS' if has_win.item() else 'FAIL'}")
    assert has_win.item(), "Should detect win at row 8"
    
    # Test 4: Vertical win at col 8
    state = game.get_initial_state(1)
    for i in range(5):
        state[0, 0, i, 8] = 1  # Right column, first 5 positions
    
    has_win = game.check_win(state, player=1)
    print(f"Test 4 - Vertical win at col 8: {'PASS' if has_win.item() else 'FAIL'}")
    assert has_win.item(), "Should detect win at col 8"
    
    # Test 5: No false positive
    state = game.get_initial_state(1)
    for i in range(4):  # Only 4 in a row
        state[0, 0, 0, i] = 1
    
    has_win = game.check_win(state, player=1)
    print(f"Test 5 - No false positive (4 in row): {'PASS' if not has_win.item() else 'FAIL'}")
    assert not has_win.item(), "Should NOT detect win with only 4"
    
    print("\n✅ All edge win tests passed!")

if __name__ == "__main__":
    test_edge_wins()
