#!/usr/bin/env python3
"""
Demo: Alpha-Beta Engine with WDL Neural Network Evaluation

Shows the complete Stockfish-like engine:
- Alpha-beta search with iterative deepening
- Neural network WDL evaluation (trained on solver labels)
- Transposition table and move ordering
- Zobrist hashing

This is the final hybrid engine combining all components.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from time import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.wdl_network import WDLResNet
from alpha_zero_light.engine.alphabeta import AlphaBetaEngine
from alpha_zero_light.engine.zobrist import get_zobrist_hasher


def wdl_to_score(wdl_probs):
    """
    Convert WDL probabilities to evaluation score.
    
    Args:
        wdl_probs: [p_win, p_draw, p_loss]
    
    Returns:
        Score in range [-1, 1]
    """
    p_win, p_draw, p_loss = wdl_probs
    # Expected value: win=1, draw=0, loss=-1
    return p_win - p_loss


class WDLEvaluator:
    """Neural network evaluator using WDL classification."""
    
    def __init__(self, model, game, device='cuda'):
        self.model = model
        self.game = game
        self.device = device
        self.model.eval()
    
    def evaluate(self, state, player):
        """
        Evaluate position from player's perspective.
        
        Args:
            state: Board state (6x7)
            player: Player to move (1 or -1)
        
        Returns:
            (eval_score, policy_probs): Evaluation and policy from NN
        """
        # Encode state
        encoded = np.zeros((1, 3, 6, 7), dtype=np.float32)
        encoded[0, 0] = (state == player).astype(np.float32)
        encoded[0, 1] = (state == -player).astype(np.float32)
        encoded[0, 2] = (state[0] == 0).astype(np.float32)
        
        # Get NN prediction
        with torch.no_grad():
            state_tensor = torch.from_numpy(encoded).to(self.device)
            policy_logits, wdl_logits = self.model(state_tensor)
            
            # Convert to probabilities
            policy_probs = torch.softmax(policy_logits, dim=1)[0].cpu().numpy()
            wdl_probs = torch.softmax(wdl_logits, dim=1)[0].cpu().numpy()
        
        # Convert WDL to score
        eval_score = wdl_to_score(wdl_probs)
        
        return eval_score, policy_probs


def demo_wdl_engine():
    """Demonstrate WDL-enhanced alpha-beta engine."""
    
    print("="*70)
    print(" "*20 + "WDL ALPHA-BETA ENGINE DEMO")
    print("="*70)
    
    # Load game and model
    game = ConnectFour()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check for trained WDL model
    wdl_model_path = Path('checkpoints/connect4/model_wdl_best.pt')
    if not wdl_model_path.exists():
        print(f"\n⚠ WDL model not found: {wdl_model_path}")
        print("Run training first: python training/scripts/train_supervised.py")
        return
    
    print(f"\nLoading WDL model from {wdl_model_path}...")
    model = WDLResNet.from_alphazero_checkpoint(
        'checkpoints/connect4/model_195.pt',
        game,
        device
    )
    model.load_state_dict(torch.load(wdl_model_path, map_location=device))
    model.eval()
    print(f"✓ Loaded on {device}")
    
    # Create evaluator
    evaluator = WDLEvaluator(model, game, device)
    
    # Create engine with WDL evaluation
    print(f"\nInitializing alpha-beta engine...")
    engine = AlphaBetaEngine(
        game=game,
        evaluator=evaluator.evaluate,
        use_tt=True,
        use_move_ordering=True
    )
    print(f"✓ Engine ready with WDL evaluation")
    
    # Test 1: Find winning move
    print(f"\n{'='*70}")
    print("TEST 1: Tactical Win Detection")
    print(f"{'='*70}")
    
    # Position with immediate win available
    test_state = game.get_initial_state()
    moves =  [3, 2, 3, 3, 2, 3]  # Setup position
    for move in moves:
        test_state = game.get_next_state(test_state, move, 1 if len([m for m in moves[:moves.index(move)+1]]) % 2 == 1 else -1)
    
    print("\nPosition (X to move):")
    symbols = {1: 'X', -1: 'O', 0: '.'}
    for row in range(6):
        print("  " + " ".join(symbols[test_state[row, col]] for col in range(7)))
    
    print(f"\nSearching for best move (depth 8, 1s time limit)...")
    start = time()
    best_move, best_score, stats = engine.search(test_state, player=1, max_depth=8, time_limit=1.0)
    elapsed = time() - start
    
    print(f"\n✓ Best move: Column {best_move}")
    print(f"  Evaluation: {best_score:.3f}")
    print(f"  Nodes searched: {stats['nodes_searched']:,}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  NPS: {stats['nodes_searched']/elapsed:,.0f}")
    
    # Test 2: Opening position evaluation
    print(f"\n{'='*70}")
    print("TEST 2: Opening Position Analysis")
    print(f"{'='*70}")
    
    initial_state = game.get_initial_state()
    print("\nStarting position:")
    for row in range(6):
        print("  " + " ".join('.  ' for _ in range(7)))
    
    print(f"\nSearching (depth 6)...")
    start = time()
    best_move, best_score, stats = engine.search(initial_state, player=1, max_depth=6)
    elapsed = time() - start
    
    print(f"\n✓ Best opening move: Column {best_move}")
    print(f"  Evaluation: {best_score:.3f} (center=4 is likely best)")
    print(f"  Nodes: {stats['nodes_searched']:,}")
    print(f"  Time: {elapsed:.3f}s")
    
    # Test 3: Compare WDL vs AlphaZero value head
    print(f"\n{'='*70}")
    print("TEST 3: WDL Model Confidence")
    print(f"{'='*70}")
    
    # Get WDL prediction for test position
    eval_score, policy = evaluator.evaluate(test_state, 1)
    
    # Get raw WDL probs
    encoded = np.zeros((1, 3, 6, 7), dtype=np.float32)
    encoded[0, 0] = (test_state == 1).astype(np.float32)
    encoded[0, 1] = (test_state == -1).astype(np.float32)
    encoded[0, 2] = (test_state[0] == 0).astype(np.float32)
    
    with torch.no_grad():
        state_tensor = torch.from_numpy(encoded).to(device)
        _, wdl_logits = model(state_tensor)
        wdl_probs = torch.softmax(wdl_logits, dim=1)[0].cpu().numpy()
    
    print(f"\nWDL Model Prediction:")
    print(f"  P(Win):  {wdl_probs[0]:.4f}")
    print(f"  P(Draw): {wdl_probs[1]:.4f}")
    print(f"  P(Loss): {wdl_probs[2]:.4f}")
    print(f"  → Eval: {eval_score:.3f}")
    
    print(f"\n{'='*70}")
    print("DEMO COMPLETE - Engine Ready for Deployment!")
    print(f"{'='*70}")
    print(f"\nThis engine combines:")
    print(f"  ✓ Alpha-beta search with pruning")
    print(f"  ✓ Neural network WDL evaluation (trained on solver)")
    print(f"  ✓ Transposition table")
    print(f"  ✓ Intelligent move ordering")
    print(f"\n→ Ready to compete against MCTS baseline!")


if __name__ == '__main__':
    demo_wdl_engine()
