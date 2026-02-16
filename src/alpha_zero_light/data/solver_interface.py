"""
Python interface to Pascal Pons' Connect4 solver.

The solver accepts positions in a specific string format where each character
represents a column number (1-7) for moves played sequentially.

Example: "44" = drop in column 4 twice (player 1, then player 2)
         "2252576253462244" = complex position

The solver returns:
- Score: positive if player-to-move wins, negative if loses, 0 if draw
- Score magnitude: number of moves until win/loss (perfect play)

Format:
- Input: sequence of column indices (1-indexed: "1234567")
- Output: "{position} {score}"
  - score > 0: player to move wins in 'score' moves
  - score < 0: player to move loses in '|score|' moves  
  - score = 0: draw with perfect play
"""

import subprocess
import os
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
from collections import OrderedDict


class Connect4Solver:
    """
    Interface to Pascal Pons' Connect4 perfect solver.
    
    The solver uses alpha-beta search with advanced optimizations to
    compute exact Win/Draw/Loss outcome for any Connect4 position.
    """
    
    def __init__(self, solver_path: Optional[str] = None, cache_size: int = 100000):
        """
        Initialize solver interface.
        
        Args:
            solver_path: Path to c4solver binary (auto-detected if None)
            cache_size: LRU cache size for solved positions
        """
        if solver_path is None:
            # Auto-detect solver in project directory
            project_root = Path(__file__).parent.parent.parent.parent
            solver_path = project_root / "solvers" / "connect4" / "c4solver"
        
        self.solver_path = Path(solver_path)
        
        if not self.solver_path.exists():
            raise FileNotFoundError(
                f"Solver binary not found at {self.solver_path}\n"
                f"Please compile the solver:\n"
                f"  cd solvers/connect4 && make"
            )
        
        # LRU cache for solved positions (position_string -> (wdl, score))
        self.cache: OrderedDict[str, Tuple[int, int]] = OrderedDict()
        self.cache_size = cache_size
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_solve_calls = 0
    
    def board_to_position_string(self, state: np.ndarray) -> str:
        """
        Convert board state to solver input format.
        
        The solver expects a sequence of moves (column indices 1-7).
        We reconstruct the move sequence from the current board state.
        
        Args:
            state: Board state (6, 7) with values {-1, 0, 1}
        
        Returns:
            Position string (e.g., "2252576253462244")
        """
        moves = []
        
        # Reconstruct move sequence by scanning columns bottom-up
        num_moves = int(np.sum(np.abs(state)))
        
        # Create a copy to track remaining pieces
        remaining = state.copy()
        
        # Alternating players starting with 1
        expected_player = 1
        
        for _ in range(num_moves):
            # Find which column has a piece at the lowest available row for expected player
            found = False
            
            for col in range(7):
                for row in range(5, -1, -1):
                    if remaining[row, col] == expected_player:
                        # This could be the next move
                        # Check if all pieces below are accounted for
                        pieces_below = np.sum(remaining[row+1:, col] != 0) if row < 5 else 0
                        
                        if pieces_below == (5 - row):
                            # This is the next move
                            moves.append(str(col + 1))  # 1-indexed
                            remaining[row, col] = 0
                            expected_player = -expected_player
                            found = True
                            break
                
                if found:
                    break
            
            if not found:
                # Fallback: linear scan
                break
        
        return ''.join(moves)
    
    def solve_position(self, position_str: str, timeout: int = 120) -> Tuple[int, int, Optional[int]]:
        """
        Solve a Connect4 position using the perfect solver.
        
        Args:
            position_str: Position string (e.g., "44" or "2252576253462244")
            timeout: Solver timeout in seconds (default 120s)
        
        Returns:
            (wdl, score, best_move):
                - wdl: -1 (loss), 0 (draw), 1 (win) for player to move
                - score: exact score (positive = win in N moves, negative = loss in N moves)
                - best_move: None (not provided by this solver directly)
        """
        # Check cache
        if position_str in self.cache:
            self.cache_hits += 1
            wdl, score = self.cache[position_str]
            return wdl, score, None
        
        self.cache_misses += 1
        self.total_solve_calls += 1
        
        # Call solver with increased timeout for complex positions
        try:
            result = subprocess.run(
                [str(self.solver_path)],
                input=position_str + '\n',
                capture_output=True,
                text=True,
                timeout=120,  # Increased to 120s for early-game positions
                cwd=str(self.solver_path.parent)
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Solver failed: {result.stderr}")
            
            # Parse output
            # Format: "{position} {score}" (opening book warning can be ignored)
            lines = result.stdout.strip().split('\n')
            result_line = None
            
            for line in lines:
                if position_str in line or line.strip().startswith(position_str):
                    result_line = line
                    break
            
            if result_line is None:
                # Try last non-empty line
                for line in reversed(lines):
                    if line.strip() and not line.startswith('Unable'):
                        result_line = line
                        break
            
            if result_line is None:
                raise RuntimeError(f"Could not parse solver output: {result.stdout}")
            
            # Extract score
            parts = result_line.split()
            if len(parts) >= 2:
                score = int(parts[1])
            else:
                raise RuntimeError(f"Unexpected output format: {result_line}")
            
            # Convert score to WDL
            if score > 0:
                wdl = 1  # Win
            elif score < 0:
                wdl = -1  # Loss
            else:
                wdl = 0  # Draw
            
            # Cache result
            self._add_to_cache(position_str, (wdl, score))
            
            return wdl, score, None
        
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Solver timeout on position: {position_str}")
        except Exception as e:
            raise RuntimeError(f"Solver error: {e}")
    
    def solve_board(self, state: np.ndarray, player: int) -> Tuple[int, int, Optional[List[int]]]:
        """
        Solve a board position.
        
        Args:
            state: Board state (6, 7)
            player: Current player to move (1 or -1)
        
        Returns:
            (wdl, score, best_moves):
                - wdl: -1/0/1 from current player's perspective
                - score: exact score
                - best_moves: None (not computed by this interface)
        """
        # Convert to position string
        position_str = self.board_to_position_string(state)
        
        # Solve
        wdl, score, _ = self.solve_position(position_str)
        
        # If player is -1, we need to flip the result
        # (solver always returns from perspective of player to move)
        # The position string reconstruction assumes player 1 moves first
        
        # Count number of pieces to determine whose turn it is
        num_pieces = int(np.sum(np.abs(state)))
        current_player = 1 if num_pieces % 2 == 0 else -1
        
        # Adjust for player perspective
        if current_player != player:
            # Wrong player, this means the position is actually for the opponent
            wdl = -wdl
            score = -score
        
        return wdl, score, None
    
    def solve_batch(
        self, 
        positions: List[str], 
        max_workers: int = 1
    ) -> List[Tuple[int, int, Optional[int]]]:
        """
        Solve multiple positions.
        
        Currently sequential (could be parallelized with multiprocessing).
        
        Args:
            positions: List of position strings
            max_workers: Number of parallel workers (not implemented yet)
        
        Returns:
            List of (wdl, score, best_move) tuples
        """
        results = []
        for pos in positions:
            try:
                result = self.solve_position(pos)
                results.append(result)
            except Exception as e:
                # On error, return unknown result
                print(f"Warning: Failed to solve {pos}: {e}")
                results.append((0, 0, None))
        
        return results
    
    def _add_to_cache(self, key: str, value: Tuple[int, int]):
        """Add entry to LRU cache."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value
            
            # Evict oldest if over capacity
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_queries = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_queries if total_queries > 0 else 0.0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_size': len(self.cache),
            'cache_capacity': self.cache_size,
            'hit_rate': hit_rate,
            'total_solve_calls': self.total_solve_calls,
        }
    
    def clear_cache(self):
        """Clear the position cache."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


def test_solver():
    """Quick test of solver interface on tactical positions."""
    solver = Connect4Solver()
    
    print("Testing Connect4 Solver Interface")
    print("="*50)
    print("Note: Testing on mid/late-game positions (solve faster)")
    
    # Test 1: Late-game tactical position (15+ moves, solves quickly)
    print("\n✓ Test 1: Tactical position (15 moves)")
    position = "444555233211111"  # Complex late-game position
    try:
        wdl, score, _ = solver.solve_position(position, timeout=60)
        print(f"  Position: {position}")
        print(f"  WDL={wdl}, Score={score}")
        print(f"  Result: Player to move {'WINS' if wdl > 0 else 'LOSES' if wdl < 0 else 'DRAWS'} in {abs(score)} moves")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 2: Another tactical scenario
    print("\n✓ Test 2: Tactical position (18 moves)")
    position2 = "225257625346224455"
    try:
        wdl2, score2, _ = solver.solve_position(position2, timeout=60)
        print(f"  Position: {position2}")
        print(f"  WDL={wdl2}, Score={score2}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 3: Cache test
    print("\n✓ Test 3: Cache test (re-solving first position)")
    try:
        wdl3, score3, _ = solver.solve_position(position, timeout=60)
        stats = solver.get_stats()
        print(f"  Cache hit! Stats:")
        print(f"    - Hits: {stats['cache_hits']}")
        print(f"    - Misses: {stats['cache_misses']}")
        print(f"    - Hit rate: {stats['hit_rate']:.1%}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "="*50)
    print("✓ SOLVER TESTS COMPLETED")
    print("="*50)
    print(f"\nFinal stats: {solver.get_stats()}")
    print("\nNote: For dataset generation, we'll focus on positions")
    print("with 10+ moves played (mid/late-game) which solve faster.")


if __name__ == '__main__':
    test_solver()
