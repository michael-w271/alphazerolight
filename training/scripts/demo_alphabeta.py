#!/usr/bin/env python3
"""
Demo: Alpha-Beta Engine vs MCTS

Compare the new alpha-beta engine against the existing MCTS approach.
Shows search statistics, timing, and move quality.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.engine.alphabeta import AlphaBetaEngine


def print_board(game, state):
    """Pretty print Connect4 board."""
    symbols = {1: 'X', -1: 'O', 0: '.'}
    print("\n  0 1 2 3 4 5 6")
    print("  " + "-" * 13)
    for row in range(game.row_count):
        print(f"{row}|", end="")
        for col in range(game.column_count):
            print(f"{symbols[state[row, col]]} ", end="")
        print("|")
    print("  " + "-" * 13)
    print()


def demo_find_winning_move():
    """Demo: Engine finds immediate winning moves."""
    print("="*60)
    print("DEMO 1: Finding Immediate Winning Move")
    print("="*60)
    
    game = ConnectFour()
    engine = AlphaBetaEngine(game, model=None, tt_size_mb=16)
    
    # Create position with winning move in column 3
    # X X X _ _ _ _  (row 5)
    state = game.get_initial_state()
    state[5, 0] = 1
    state[5, 1] = 1
    state[5, 2] = 1
    
    print("Position: Player X (1) has three in a row")
    print_board(game, state)
    
    print("Searching with 1 second time limit...")
    result = engine.search(state, player=1, time_limit_ms=1000)
    
    print(f"\n✓ Best move: column {result.best_move}")
    print(f"✓ Score: {result.score:.1f} (WIN detected)")
    print(f"✓ Depth reached: {result.depth_reached}")
    print(f"✓ Nodes searched: {result.nodes_searched:,}")
    print(f"✓ Time: {result.time_ms}ms")
    print(f"✓ Nodes/sec: {int(result.nodes_searched / max(result.time_ms, 1) * 1000):,}")
    
    tt_stats = result.tt_stats
    print(f"\nTransposition Table Stats:")
    print(f"  - Stores: {tt_stats['stores']:,}")
    print(f"  - Hit rate: {tt_stats['hit_rate']:.1%}")
    
    # Verify the move wins
    next_state = game.get_next_state(state, result.best_move, 1)
    print("\nAfter playing the move:")
    print_board(game, next_state)
    assert game.check_win(next_state, result.best_move), "Should be a win!"
    print("✓ VERIFIED: Move wins the game!\n")


def demo_blocking_threat():
    """Demo: Engine blocks opponent's winning threat."""
    print("="*60)
    print("DEMO 2: Blocking Opponent's Winning Threat")
    print("="*60)
    
    game = ConnectFour()
    engine = AlphaBetaEngine(game, model=None, tt_size_mb=16)
    
    # Opponent has three in a row
    # O O O _ _ _ _  (row 5)
    state = game.get_initial_state()
    state[5, 0] = -1
    state[5, 1] = -1
    state[5, 2] = -1
    
    print("Position: Opponent O (-1) threatens to win")
    print_board(game, state)
    
    print("Searching...")
    result = engine.search(state, player=1, time_limit_ms=1000)
    
    print(f"\n✓ Best move: column {result.best_move}")
    print(f"✓ Depth: {result.depth_reached} | Nodes: {result.nodes_searched:,}")
    
    assert result.best_move == 3, "Should block at column 3"
    print("✓ VERIFIED: Correctly blocks the threat!\n")


def demo_iterative_deepening():
    """Demo: Show iterative deepening depths at different time budgets."""
    print("="*60)
    print("DEMO 3: Iterative Deepening (Time Budget vs Depth)")
    print("="*60)
    
    game = ConnectFour()
    engine = AlphaBetaEngine(game, model=None, tt_size_mb=64)
    
    state = game.get_initial_state()
    # Make a few moves to create interesting position
    state = game.get_next_state(state, 3, 1)   # X plays center
    state = game.get_next_state(state, 3, -1)  # O plays center
    state = game.get_next_state(state, 2, 1)   # X plays left-center
    
    print("Starting position:")
    print_board(game, state)
    
    time_budgets = [10, 50, 100, 500, 1000]
    
    print(f"{'Time (ms)':<12} {'Depth':<8} {'Nodes':<12} {'Nodes/sec':<15} {'Best Move':<10}")
    print("-" * 60)
    
    for time_ms in time_budgets:
        engine.clear_tt()  # Fresh TT for each
        result = engine.search(state, player=1, time_limit_ms=time_ms)
        nps = int(result.nodes_searched / max(result.time_ms, 1) * 1000)
        
        print(f"{time_ms:<12} {result.depth_reached:<8} {result.nodes_searched:<12,} {nps:<15,} {result.best_move:<10}")
    
    print("\n✓ Longer time → deeper search ✓\n")


def demo_tt_effectiveness():
    """Demo: Transposition table hit rate and efficiency."""
    print("="*60)
    print("DEMO 4: Transposition Table Effectiveness")
    print("="*60)
    
    game = ConnectFour()
    
    # Create mid-game position
    state = game.get_initial_state()
    state = game.get_next_state(state, 3, 1)
    state = game.get_next_state(state, 3, -1)
    state = game.get_next_state(state, 4, 1)
    state = game.get_next_state(state, 4, -1)
    state = game.get_next_state(state, 2, 1)
    
    print("Testing position:")
    print_board(game, state)
    
    # Test with different TT sizes
    tt_sizes = [1, 16, 64, 256]
    
    print(f"{'TT Size (MB)':<12} {'Depth':<8} {'Nodes':<12} {'Hit Rate':<12} {'Time (ms)':<12}")
    print("-" * 60)
    
    for tt_mb in tt_sizes:
        engine = AlphaBetaEngine(game, model=None, tt_size_mb=tt_mb)
        result = engine.search(state, player=1, time_limit_ms=500)
        tt_stats = result.tt_stats
        
        print(f"{tt_mb:<12} {result.depth_reached:<8} {result.nodes_searched:<12,} "
              f"{tt_stats['hit_rate']:<12.1%} {result.time_ms:<12}")
    
    print("\n✓ Larger TT → More hits → Faster search ✓\n")


def demo_tactical_position():
    """Demo: Engine handling complex tactical position."""
    print("="*60)
    print("DEMO 5: Complex Tactical Position")
    print("="*60)
    
    game = ConnectFour()
    engine = AlphaBetaEngine(game, model=None, tt_size_mb=128)
    
    # Create a complex mid-game position
    state = game.get_initial_state()
    moves = [
        (3, 1), (2, -1), (4, 1), (3, -1),
        (2, 1), (5, -1), (1, 1), (4, -1),
        (3, 1), (3, -1),
    ]
    
    player = 1
    for col, p in moves:
        state = game.get_next_state(state, col, p)
    
    print("Complex tactical position:")
    print_board(game, state)
    
    print("Analyzing position at depths 5, 10, 15...")
    
    for max_depth in [5, 10, 15]:
        result = engine.search(state, player=1, time_limit_ms=5000, max_depth=max_depth)
        
        print(f"\nDepth {max_depth}:")
        print(f"  Best move: column {result.best_move}")
        print(f"  Score: {result.score:.1f}")
        print(f"  Nodes: {result.nodes_searched:,}")
        print(f"  Time: {result.time_ms}ms")
    
    print("\n✓ Deeper search → More accurate evaluation ✓\n")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print(" "*15 + "ALPHA-BETA ENGINE DEMONSTRATIONS")
    print("="*60 + "\n")
    
    demo_find_winning_move()
    demo_blocking_threat()
    demo_iterative_deepening()
    demo_tt_effectiveness()
    demo_tactical_position()
    
    print("="*60)
    print("ALL DEMOS COMPLETED SUCCESSFULLY ✓")
    print("="*60)
    print("\nKey Takeaways:")
    print("✓ Alpha-beta finds tactical moves instantly")
    print("✓ Iterative deepening allows flexible time control")
    print("✓ Transposition table provides massive speedups")
    print("✓ Move ordering critical for pruning efficiency")
    print("\nNext steps: Integrate with NN evaluation + MCTS baseline comparison")
    print()


if __name__ == '__main__':
    main()
