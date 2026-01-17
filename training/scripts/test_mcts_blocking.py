#!/usr/bin/env python3
"""
Robust MCTS tactical blocking test.
Tests forced block and open-ended threat scenarios with correct board construction.
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS
from alpha_zero_light.config_connect4 import MODEL_CONFIG, MCTS_CONFIG


def load_model(path, game, device):
    """Load model checkpoint or use random initialization."""
    model = ResNet(game, MODEL_CONFIG['num_res_blocks'], MODEL_CONFIG['num_hidden']).to(device)
    if Path(path).exists():
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"✓ Loaded {path} on {device}")
    else:
        print(f"⚠️  {path} not found. Using random init.")
    model.eval()
    return model


def run_case(name, state, expected_block_cols, mcts, game, model, device, threshold=0.30):
    """
    Run a single tactical test case.
    
    Args:
        name: Test case name
        state: Board state in canonical form (current player = +1, opponent = -1)
        expected_block_cols: List of column indices that block the threat
        mcts: MCTS instance
        game: ConnectFour game instance
        model: Neural network model
        device: torch device
        threshold: Minimum probability threshold for PASS (default 0.30)
    """
    print("\n" + "="*70)
    print(name)
    print("="*70)
    
    # Show board encoding
    enc = game.get_encoded_state(state)
    print("\nBoard visualization (bottom row first):")
    print("  Opponent pieces (−1 in state, shown as 'O'):")
    for row in reversed(enc[0]):
        print("   ", " ".join("O" if x == 1 else "·" for x in row))
    print("  Current player pieces (+1 in state, shown as 'X'):")
    for row in reversed(enc[2]):
        print("   ", " ".join("X" if x == 1 else "·" for x in row))
    
    print("\nRaw state (opponent = -1, current = +1):")
    for row in state:
        print("   ", " ".join(f"{int(x):2}" for x in row))
    
    # Valid moves
    valid = game.get_valid_moves(state)
    valid_cols = np.where(valid)[0].tolist()
    print(f"\nValid moves: {valid_cols}")
    
    # Network-only prediction (no MCTS)
    encoded = game.get_encoded_state(state)
    encoded_tensor = torch.tensor(encoded, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        policy_logits, value = model(encoded_tensor)
        net_policy = torch.softmax(policy_logits, dim=1)[0].cpu().numpy()
        net_value = value.item()
    
    print(f"\nNetwork evaluation (no MCTS):")
    print(f"  Value: {net_value:.3f}")
    print(f"  Policy:")
    for col in range(7):
        if net_policy[col] > 0.01:
            marker = " ← BLOCK" if col in expected_block_cols else ""
            print(f"    Col {col}: {net_policy[col]:.4f}{marker}")
    
    net_block_prob = float(np.sum([net_policy[c] for c in expected_block_cols]))
    print(f"  Network block probability (sum of cols {expected_block_cols}): {net_block_prob:.1%}")
    
    # MCTS search
    print(f"\nRunning MCTS ({mcts.args['num_searches']} searches, no noise)...")
    mcts_probs = mcts.search(state, add_noise=False)
    
    print(f"MCTS policy:")
    for col in range(7):
        if mcts_probs[col] > 0.01:
            marker = " ← BLOCK" if col in expected_block_cols else ""
            print(f"    Col {col}: {mcts_probs[col]:.4f}{marker}")
    
    mcts_block_prob = float(np.sum([mcts_probs[c] for c in expected_block_cols]))
    print(f"\nMCTS block probability (sum of cols {expected_block_cols}): {mcts_block_prob:.1%}")
    
    # Verdict
    print("\n" + "-"*70)
    print("RESULTS:")
    print(f"  Network wants to block: {net_block_prob:.1%}")
    print(f"  MCTS actually blocks:   {mcts_block_prob:.1%}")
    print(f"  Threshold for PASS:     {threshold*100:.0f}%")
    
    if mcts_block_prob >= threshold:
        print("✅ PASS - MCTS correctly prioritizes blocking")
        return True
    else:
        print(f"❌ FAIL - MCTS block probability too low")
        print(f"         Expected ≥{threshold*100:.0f}%, got {mcts_block_prob:.1%}")
        return False


def main():
    print("="*70)
    print("MCTS TACTICAL BLOCKING TEST")
    print("Testing MCTS behavior after UCB sign fix")
    print("="*70)
    
    game = ConnectFour()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_path = str(BASE / 'checkpoints/connect4/model_25.pt')
    model = load_model(model_path, game, device)
    
    # Configure MCTS
    args = {**MCTS_CONFIG}
    args['num_searches'] = int(os.environ.get('MCTS_SEARCHES', args.get('num_searches', 70)))
    args['dirichlet_epsilon'] = 0.0  # Deterministic evaluation
    args['mcts_batch_size'] = int(os.environ.get('MCTS_BATCH_SIZE', '1'))
    
    print(f"\nMCTS Configuration:")
    print(f"  Searches: {args['num_searches']}")
    print(f"  C (exploration): {args['C']}")
    print(f"  Batch size: {args['mcts_batch_size']}")
    print(f"  Dirichlet noise: {args['dirichlet_epsilon']} (disabled)")
    
    mcts = MCTS(game, args, model)
    
    results = []
    
    # =======================================================================
    # CASE A: FORCED BLOCK
    # Opponent has 3 in a row at bottom: columns 0,1,2
    # Only valid block is column 3
    # =======================================================================
    state_a = game.get_initial_state()
    # In canonical form: opponent = -1, current player = +1
    state_a[5, 0] = -1  # Opponent piece at (row 5, col 0)
    state_a[5, 1] = -1  # Opponent piece at (row 5, col 1)
    state_a[5, 2] = -1  # Opponent piece at (row 5, col 2)
    # Opponent threatens to win at column 3
    
    result_a = run_case(
        name="CASE A: FORCED BLOCK (opponent at cols 0,1,2 → must block col 3)",
        state=state_a,
        expected_block_cols=[3],
        mcts=mcts,
        game=game,
        model=model,
        device=device,
        threshold=0.30
    )
    results.append(("Forced Block", result_a))
    
    # =======================================================================
    # CASE B: OPEN-ENDED THREAT
    # Opponent has 3 in a row at bottom: columns 1,2,3
    # Valid blocks are column 0 OR column 4
    # =======================================================================
    state_b = game.get_initial_state()
    state_b[5, 1] = -1  # Opponent piece at (row 5, col 1)
    state_b[5, 2] = -1  # Opponent piece at (row 5, col 2)
    state_b[5, 3] = -1  # Opponent piece at (row 5, col 3)
    # Opponent can win at either column 0 or column 4
    
    result_b = run_case(
        name="CASE B: OPEN-ENDED THREAT (opponent at cols 1,2,3 → block col 0 OR col 4)",
        state=state_b,
        expected_block_cols=[0, 4],
        mcts=mcts,
        game=game,
        model=model,
        device=device,
        threshold=0.30
    )
    results.append(("Open-Ended Threat", result_b))
    
    # =======================================================================
    # CASE C: IMMEDIATE WIN AVAILABLE
    # Current player has 3 in a row: columns 0,1,2
    # Can win immediately at column 3
    # =======================================================================
    state_c = game.get_initial_state()
    state_c[5, 0] = 1   # Current player piece at (row 5, col 0)
    state_c[5, 1] = 1   # Current player piece at (row 5, col 1)
    state_c[5, 2] = 1   # Current player piece at (row 5, col 2)
    # Current player can win at column 3
    
    result_c = run_case(
        name="CASE C: IMMEDIATE WIN (current player at cols 0,1,2 → win at col 3)",
        state=state_c,
        expected_block_cols=[3],  # Should choose winning move
        mcts=mcts,
        game=game,
        model=model,
        device=device,
        threshold=0.50  # Higher threshold for winning moves
    )
    results.append(("Immediate Win", result_c))
    
    # =======================================================================
    # SUMMARY
    # =======================================================================
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:20s}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED")
        print("MCTS UCB fix is working correctly!")
        print("="*70)
        return 0
    else:
        print("⚠️  SOME TESTS FAILED")
        print("MCTS may still have issues with tactical play.")
        print("="*70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
