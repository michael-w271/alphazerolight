"""
Transposition table for caching alpha-beta search results.

The transposition table stores previously computed positions to avoid redundant
work during alpha-beta search. This provides massive speedups in iterative
deepening, as positions from depth D-1 are reused when searching depth D.

Key concepts:
- Bound types: EXACT (PV node), LOWER (fail-high/beta cutoff), UPPER (fail-low/alpha cutoff)
- Replacement policy: Depth-preferred (replace shallower searches)
- Age tracking: Clear old entries between root searches
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class BoundType(Enum):
    """Type of bound stored in transposition table entry."""
    EXACT = 0   # Exact value (PV node, searched with full window)
    LOWER = 1   # Lower bound (beta cutoff, actual value >= stored value)
    UPPER = 2   # Upper bound (alpha cutoff, actual value <= stored value)


@dataclass
class TTEntry:
    """
    Transposition table entry storing cached search results.
    
    Attributes:
        zobrist_hash: Full 64-bit hash for collision detection
        depth: Search depth when this entry was stored
        score: Evaluation score (or bound)
        bound: Type of bound (EXACT/LOWER/UPPER)
        best_move: Best move found at this position (column index)
        age: Search generation (for aging out old entries)
    """
    zobrist_hash: int
    depth: int
    score: float
    bound: BoundType
    best_move: Optional[int]
    age: int = 0
    
    def is_valid(self, query_hash: int, current_age: int, max_age_diff: int = 10) -> bool:
        """
        Check if this entry is valid for the query.
        
        Args:
            query_hash: Hash being queried
            current_age: Current search generation
            max_age_diff: Maximum age difference to accept
        
        Returns:
            True if entry is valid (hash matches and not too old)
        """
        return (
            self.zobrist_hash == query_hash and 
            (current_age - self.age) <= max_age_diff
        )


class TranspositionTable:
    """
    Fixed-size transposition table with replacement policy.
    
    Implementation:
    - Power-of-2 sized table for fast modulo via bit masking
    - Always-replace or depth-preferred replacement
    - Age tracking to invalidate entries from old searches
    
    Memory usage: ~32 bytes per entry
    Example: 16M entries = 512 MB RAM
    """
    
    def __init__(self, size_mb: int = 256):
        """
        Initialize transposition table.
        
        Args:
            size_mb: Table size in megabytes (will be rounded to power of 2)
        """
        # Calculate number of entries (32 bytes per entry estimate)
        bytes_per_entry = 64  # Conservative estimate with Python overhead
        num_entries = (size_mb * 1024 * 1024) // bytes_per_entry
        
        # Round down to nearest power of 2 for fast modulo
        self.num_entries = 2 ** int(np.log2(num_entries))
        self.index_mask = self.num_entries - 1
        
        # Initialize table with None entries
        self.table: list[Optional[TTEntry]] = [None] * self.num_entries
        
        # Search generation for aging
        self.current_age = 0
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.collisions = 0
        self.stores = 0
    
    def _get_index(self, zobrist_hash: int) -> int:
        """Get table index from zobrist hash (fast modulo via bit masking)."""
        return zobrist_hash & self.index_mask
    
    def probe(
        self, 
        zobrist_hash: int, 
        depth: int, 
        alpha: float, 
        beta: float
    ) -> Optional[tuple[float, Optional[int]]]:
        """
        Probe transposition table for cached result.
        
        Returns cached score if:
        1. Hash matches (no collision)
        2. Stored depth >= query depth (deeper search is more accurate)
        3. Bound type allows cutoff given current alpha-beta window
        
        Args:
            zobrist_hash: Position hash
            depth: Current search depth
            alpha: Current alpha bound
            beta: Current beta bound
        
        Returns:
            (score, best_move) if usable entry found, None otherwise
        """
        index = self._get_index(zobrist_hash)
        entry = self.table[index]
        
        if entry is None:
            self.misses += 1
            return None
        
        # Verify hash matches (collision detection)
        if not entry.is_valid(zobrist_hash, self.current_age):
            if entry.zobrist_hash != zobrist_hash:
                self.collisions += 1
            self.misses += 1
            return None
        
        # Check if stored depth is sufficient
        if entry.depth < depth:
            self.misses += 1
            return None
        
        # Check if bound type allows cutoff
        if entry.bound == BoundType.EXACT:
            # Exact value always usable
            self.hits += 1
            return (entry.score, entry.best_move)
        elif entry.bound == BoundType.LOWER:
            # Lower bound: true value >= entry.score
            if entry.score >= beta:
                self.hits += 1
                return (entry.score, entry.best_move)
        elif entry.bound == BoundType.UPPER:
            # Upper bound: true value <= entry.score
            if entry.score <= alpha:
                self.hits += 1
                return (entry.score, entry.best_move)
        
        self.misses += 1
        return None
    
    def store(
        self, 
        zobrist_hash: int, 
        depth: int, 
        score: float, 
        bound: BoundType, 
        best_move: Optional[int]
    ):
        """
        Store search result in transposition table.
        
        Replacement policy: Always replace OR depth-preferred
        (here we use always-replace for simplicity)
        
        Args:
            zobrist_hash: Position hash
            depth: Search depth
            score: Evaluation or bound
            bound: Type of bound
            best_move: Best move found (None if terminal)
        """
        index = self._get_index(zobrist_hash)
        existing = self.table[index]
        
        # Replacement policy: prefer deeper searches
        if existing is not None:
            if existing.zobrist_hash == zobrist_hash:
                # Same position: only replace if deeper or same depth with better bound
                if depth < existing.depth:
                    return  # Don't replace deeper search with shallower
                elif depth == existing.depth and bound != BoundType.EXACT:
                    if existing.bound == BoundType.EXACT:
                        return  # Don't replace EXACT with non-EXACT
        
        # Store new entry
        self.table[index] = TTEntry(
            zobrist_hash=zobrist_hash,
            depth=depth,
            score=score,
            bound=bound,
            best_move=best_move,
            age=self.current_age
        )
        self.stores += 1
    
    def get_best_move(self, zobrist_hash: int) -> Optional[int]:
        """
        Retrieve best move from TT without score checking.
        
        Useful for move ordering even when depth/bounds don't allow cutoff.
        
        Args:
            zobrist_hash: Position hash
        
        Returns:
            Best move column if entry exists and matches, None otherwise
        """
        index = self._get_index(zobrist_hash)
        entry = self.table[index]
        
        if entry is not None and entry.is_valid(zobrist_hash, self.current_age):
            return entry.best_move
        
        return None
    
    def clear(self):
        """Clear all entries (use between games)."""
        self.table = [None] * self.num_entries
        self.current_age = 0
        self._reset_stats()
    
    def new_search(self):
        """Increment age counter for new root search (iterative deepening)."""
        self.current_age += 1
    
    def _reset_stats(self):
        """Reset statistics counters."""
        self.hits = 0
        self.misses = 0
        self.collisions = 0
        self.stores = 0
    
    def get_stats(self) -> dict:
        """
        Get transposition table statistics.
        
        Returns:
            Dictionary with hits, misses, hit rate, collisions
        """
        total_queries = self.hits + self.misses
        hit_rate = self.hits / total_queries if total_queries > 0 else 0.0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'collisions': self.collisions,
            'stores': self.stores,
            'size_entries': self.num_entries,
        }
    
    def get_fill_rate(self) -> float:
        """
        Calculate percentage of table slots occupied.
        
        Returns:
            Fill rate as percentage (0-100)
        """
        occupied = sum(1 for entry in self.table if entry is not None)
        return (occupied / self.num_entries) * 100.0
