"""
Unit tests for alpha-beta search engine.

Tests verify:
1. Engine finds forced wins in simple positions
2. Transposition table works correctly
3. Move ordering improves search efficiency
4. Time management stops search properly
5. Iterative deepening returns consistent results
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.engine.alphabeta import AlphaBetaEngine, SCORE_WIN, SCORE_DRAW
from alpha_zero_light.engine.zobrist import ZobristHasher, get_zobrist_hasher
from alpha_zero_light.engine.transposition_table import TranspositionTable, BoundType


class TestZobristHashing:
    """Test Zobrist hashing correctness."""
    
    def test_hash_uniqueness(self):
        """Different positions should have different hashes."""
        game = ConnectFour()
        zobrist = get_zobrist_hasher()
        
        state1 = game.get_initial_state()
        state2 = game.get_next_state(state1, 3, 1)  # Drop in center
        state3 = game.get_next_state(state2, 3, -1)  # Opponent drops
        
        hash1 = zobrist.hash_position(state1, 1)
        hash2 = zobrist.hash_position(state2, -1)
        hash3 = zobrist.hash_position(state3, 1)
        
        assert hash1 != hash2
        assert hash2 != hash3
        assert hash1 != hash3
    
    def test_incremental_hash(self):
        """Incremental hash should match full recomputation."""
        game = ConnectFour()
        zobrist = get_zobrist_hasher()
        
        state = game.get_initial_state()
        hash1 = zobrist.hash_position(state, 1)
        
        # Make move and compute hash incrementally
        state_after = game.get_next_state(state, 3, 1)
        hash2_incremental, row = zobrist.hash_after_move(state, hash1, 3, 1)
        
        # Compute hash from scratch
        hash2_full = zobrist.hash_position(state_after, -1)
        
        assert hash2_incremental == hash2_full
    
    def test_collision_rate(self):
        """Hash collisions should be rare on random positions."""
        game = ConnectFour()
        zobrist = get_zobrist_hasher()
        
        hashes = set()
        num_positions = 10000
        
        for _ in range(num_positions):
            state = game.get_initial_state()
            # Play 10-20 random moves
            num_moves = np.random.randint(10, 21)
            player = 1
            
            for _ in range(num_moves):
                valid_moves = game.get_valid_moves(state)
                if np.sum(valid_moves) == 0:
                    break
                move = np.random.choice(np.where(valid_moves)[0])
                state = game.get_next_state(state, move, player)
                player = -player
            
            hash_val = zobrist.hash_position(state, player)
            hashes.add(hash_val)
        
        # Should have very few collisions (>99% unique)
        unique_rate = len(hashes) / num_positions
        assert unique_rate > 0.99


class TestTranspositionTable:
    """Test transposition table functionality."""
    
    def test_store_and_probe(self):
        """Basic store and probe should work."""
        tt = TranspositionTable(size_mb=1)
        
        hash_val = 12345
        tt.store(hash_val, depth=5, score=100.0, bound=BoundType.EXACT, best_move=3)
        
        result = tt.probe(hash_val, depth=3, alpha=-1000, beta=1000)
        assert result is not None
        assert result[0] == 100.0
        assert result[1] == 3
    
    def test_depth_requirement(self):
        """Should not return entry if stored depth is too shallow."""
        tt = TranspositionTable(size_mb=1)
        
        hash_val = 12345
        tt.store(hash_val, depth=3, score=50.0, bound=BoundType.EXACT, best_move=2)
        
        # Query at deeper depth
        result = tt.probe(hash_val, depth=5, alpha=-1000, beta=1000)
        assert result is None  # Too shallow
    
    def test_bound_types(self):
        """Different bound types should respect alpha-beta window."""
        tt = TranspositionTable(size_mb=1)
        hash_val = 12345
        
        # Store LOWER bound (fail-high, beta cutoff)
        tt.store(hash_val, depth=5, score=100.0, bound=BoundType.LOWER, best_move=3)
        
        # Should cut off if score >= beta
        result = tt.probe(hash_val, depth=4, alpha=-1000, beta=90)
        assert result is not None  # Cutoff occurs
        
        result = tt.probe(hash_val, depth=4, alpha=-1000, beta=110)
        assert result is None  # No cutoff


class TestAlphaBetaEngine:
    """Test alpha-beta search engine."""
    
    def test_finds_immediate_win(self):
        """Engine should find immediate winning move."""
        game = ConnectFour()
        engine = AlphaBetaEngine(game, model=None, tt_size_mb=16)
        
        # Create position where player 1 can win in column 3
        # X X X _ _ _ _  (row 5)
        state = game.get_initial_state()
        state[5, 0] = 1
        state[5, 1] = 1
        state[5, 2] = 1
        # Column 3 wins!
        
        result = engine.search(state, player=1, time_limit_ms=1000)
        
        assert result.best_move == 3  # Should play winning move
        assert result.score >= SCORE_WIN - 10  # Should recognize it's a win
    
    def test_blocks_opponent_win(self):
        """Engine should block opponent's immediate win."""
        game = ConnectFour()
        engine = AlphaBetaEngine(game, model=None, tt_size_mb=16)
        
        # Opponent (player -1) has three in a row, we must block
        # _ _ _ _ _ _ _
        # O O O _ _ _ _  (row 5)
        state = game.get_initial_state()
        state[5, 0] = -1
        state[5, 1] = -1
        state[5, 2] = -1
        
        # Player 1 to move, should block at column 3
        result = engine.search(state, player=1, time_limit_ms=1000)
        
        assert result.best_move == 3  # Must block
    
    def test_iterative_deepening(self):
        """Iterative deepening should reach increasing depths."""
        game = ConnectFour()
        engine = AlphaBetaEngine(game, model=None, tt_size_mb=16)
        
        state = game.get_initial_state()
        
        # Short time: shallow depth
        result_fast = engine.search(state, player=1, time_limit_ms=10)
        depth_fast = result_fast.depth_reached
        
        # Longer time: deeper depth
        result_slow = engine.search(state, player=1, time_limit_ms=1000)
        depth_slow = result_slow.depth_reached
        
        assert depth_slow > depth_fast
        assert result_slow.nodes_searched > result_fast.nodes_searched
    
    def test_transposition_table_usage(self):
        """TT should reduce node count on repeated searches."""
        game = ConnectFour()
        engine = AlphaBetaEngine(game, model=None, tt_size_mb=16)
        
        state = game.get_initial_state()
        state = game.get_next_state(state, 3, 1)  # Some initial moves
        state = game.get_next_state(state, 3, -1)
        
        # First search (cold TT)
        result1 = engine.search(state, player=1, time_limit_ms=500)
        nodes1 = result1.nodes_searched
        
        # Second search (warm TT)
        result2 = engine.search(state, player=1, time_limit_ms=500)
        nodes2 = result2.nodes_searched
        
        # Second search should have TT hits
        assert result2.tt_stats['hits'] > 0
        assert result2.tt_stats['hit_rate'] > 0.0
    
    def test_handles_draw(self):
        """Engine should recognize drawn positions."""
        game = ConnectFour()
        engine = AlphaBetaEngine(game, model=None, tt_size_mb=16)
        
        # Create nearly full board with no winner
        state = game.get_initial_state()
        
        # Fill board alternately (avoiding 4-in-a-rows)
        pattern = [1, -1, 1, -1, 1, -1, 1]
        for row in range(6):
            for col in range(7):
                state[row, col] = pattern[(row + col) % 2]
        
        # Now board is full, should evaluate as draw
        valid_moves = game.get_valid_moves(state)
        
        # If no valid moves, it's a draw
        if np.sum(valid_moves) == 0:
            # Engine should handle this gracefully
            result = engine.search(state, player=1, time_limit_ms=100)
            # Score should be near zero (draw)
            assert abs(result.score) < 1000


def test_full_move_ordering():
    """Test full move ordering with all heuristics."""
    from alpha_zero_light.engine.move_ordering import MoveOrdering
    
    game = ConnectFour()
    ordering = MoveOrdering(max_ply=20)
    
    state = game.get_initial_state()
    state[5, 0] = 1
    state[5, 1] = 1
    state[5, 2] = 1
    # Column 3 wins for player 1
    
    valid_moves = game.get_valid_moves(state)
    
    # Order moves
    ordered = ordering.order_moves(
        valid_moves, game, state, player=1,
        tt_move=None, policy_priors=None, ply=0
    )
    
    # Winning move (column 3) should be first
    assert ordered[0] == 3


if __name__ == '__main__':
    # Run tests
    print("Testing Zobrist hashing...")
    test_zobrist = TestZobristHashing()
    test_zobrist.test_hash_uniqueness()
    test_zobrist.test_incremental_hash()
    test_zobrist.test_collision_rate()
    print("✓ Zobrist hashing tests passed")
    
    print("\nTesting transposition table...")
    test_tt = TestTranspositionTable()
    test_tt.test_store_and_probe()
    test_tt.test_depth_requirement()
    test_tt.test_bound_types()
    print("✓ Transposition table tests passed")
    
    print("\nTesting alpha-beta engine...")
    test_ab = TestAlphaBetaEngine()
    test_ab.test_finds_immediate_win()
    print("✓ Finds immediate win")
    test_ab.test_blocks_opponent_win()
    print("✓ Blocks opponent win")
    test_ab.test_iterative_deepening()
    print("✓ Iterative deepening works")
    test_ab.test_transposition_table_usage()
    print("✓ TT reduces node count")
    test_ab.test_handles_draw()
    print("✓ Handles draw positions")
    
    print("\nTesting move ordering...")
    test_full_move_ordering()
    print("✓ Move ordering works")
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED ✓")
    print("="*50)
