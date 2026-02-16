"""
Zobrist hashing for Connect4 positions.

Zobrist hashing provides O(1) position lookup in transposition tables by
computing a unique hash for each board state. The hash can be incrementally
updated after each move, making it extremely efficient for alpha-beta search.

Implementation:
- Pre-generate random 64-bit keys for each (row, col, player) combination
- Hash = XOR of all keys corresponding to occupied squares
- Incremental update: hash ^= key[row][col][player]
"""

import numpy as np
from typing import Optional


class ZobristHasher:
    """
    Zobrist hashing for Connect4 board positions.
    
    Connect4 board: 6 rows × 7 columns × 2 players = 84 zobrist keys needed
    
    Features:
    - Deterministic hash generation (seeded RNG for reproducibility)
    - Fast incremental hash updates (XOR operation)
    - Collision detection via full board comparison in TT
    """
    
    def __init__(self, row_count: int = 6, column_count: int = 7, seed: int = 42):
        """
        Initialize Zobrist hash table with random 64-bit keys.
        
        Args:
            row_count: Number of rows in Connect4 board
            column_count: Number of columns in Connect4 board
            seed: Random seed for reproducibility
        """
        self.row_count = row_count
        self.column_count = column_count
        
        # Use seeded RNG for reproducible hashes
        rng = np.random.RandomState(seed)
        
        # Generate zobrist keys: [row, col, player_idx]
        # player_idx: 0 for player=-1, 1 for player=1
        self.zobrist_table = rng.randint(
            0, 2**63 - 1,
            size=(row_count, column_count, 2),
            dtype=np.uint64
        )
        
        # Side-to-move hash (XOR this if current player is -1)
        self.side_to_move_hash = rng.randint(0, 2**63 - 1, dtype=np.uint64)
    
    def hash_position(self, state: np.ndarray, player: int = 1) -> int:
        """
        Compute Zobrist hash for a board state.
        
        Args:
            state: Board state (row_count, column_count) with values in {-1, 0, 1}
            player: Current player to move (1 or -1)
        
        Returns:
            64-bit hash value (int)
        """
        hash_value = np.uint64(0)
        
        # XOR all occupied squares
        for row in range(self.row_count):
            for col in range(self.column_count):
                piece = state[row, col]
                if piece != 0:
                    # Map player {-1, 1} to index {0, 1}
                    player_idx = 0 if piece == -1 else 1
                    hash_value ^= self.zobrist_table[row, col, player_idx]
        
        # XOR side-to-move if player is -1
        if player == -1:
            hash_value ^= self.side_to_move_hash
        
        return int(hash_value)
    
    def incremental_hash(
        self, 
        current_hash: int, 
        row: int, 
        col: int, 
        player: int,
        is_undo: bool = False
    ) -> int:
        """
        Fast incremental hash update after making or undoing a move.
        
        For make_move: XOR in the new piece
        For undo_move: XOR out the piece (same operation due to XOR properties)
        
        Args:
            current_hash: Current board hash
            row: Row of the move
            col: Column of the move
            player: Player making the move (1 or -1)
            is_undo: Whether this is undoing a move (doesn't affect XOR)
        
        Returns:
            Updated hash value
        """
        player_idx = 0 if player == -1 else 1
        new_hash = current_hash ^ self.zobrist_table[row, col, player_idx]
        
        # Flip side-to-move hash (player changes)
        new_hash ^= self.side_to_move_hash
        
        return int(new_hash)
    
    def hash_after_move(
        self, 
        state: np.ndarray, 
        current_hash: int,
        action: int, 
        player: int
    ) -> tuple[int, int]:
        """
        Compute hash after making a move without modifying the board.
        
        Args:
            state: Current board state
            current_hash: Current hash value
            action: Column to drop piece
            player: Player making the move
        
        Returns:
            (new_hash, row_placed) - Updated hash and row where piece lands
        """
        # Find the row where piece will land (gravity)
        row_placed = -1
        for row in range(self.row_count - 1, -1, -1):
            if state[row, action] == 0:
                row_placed = row
                break
        
        if row_placed == -1:
            raise ValueError(f"Column {action} is full")
        
        # Incremental hash update
        new_hash = self.incremental_hash(current_hash, row_placed, action, player)
        
        return new_hash, row_placed
    
    def verify_hash(self, state: np.ndarray, player: int, claimed_hash: int) -> bool:
        """
        Verify that a claimed hash matches the actual board state.
        
        Useful for debugging hash collisions in transposition table.
        
        Args:
            state: Board state to verify
            player: Current player
            claimed_hash: Hash value to check
        
        Returns:
            True if hash matches, False otherwise
        """
        actual_hash = self.hash_position(state, player)
        return actual_hash == claimed_hash


# Global singleton instance
_global_hasher: Optional[ZobristHasher] = None


def get_zobrist_hasher(
    row_count: int = 6, 
    column_count: int = 7, 
    seed: int = 42
) -> ZobristHasher:
    """
    Get or create global Zobrist hasher singleton.
    
    This ensures all components use the same zobrist table.
    
    Args:
        row_count: Board rows
        column_count: Board columns
        seed: Random seed
    
    Returns:
        ZobristHasher instance
    """
    global _global_hasher
    
    if _global_hasher is None:
        _global_hasher = ZobristHasher(row_count, column_count, seed)
    
    return _global_hasher
