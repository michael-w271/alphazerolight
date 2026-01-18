"""
Move ordering heuristics for alpha-beta search.

Good move ordering is critical for alpha-beta pruning efficiency. The goal is to
search the best moves first to maximize beta cutoffs.

Ordering priority (high to low):
1. TT move (from transposition table at previous depth)
2. Winning moves (immediate 4-in-a-row)
3. Blocking moves (prevent opponent 4-in-a-row)
4. Policy priors from neural network (higher probability = search first)
5. Killer moves (non-capture moves that caused cutoffs at same ply)
6. History heuristic (moves that historically caused cutoffs)
7. Remaining moves (sorted by policy if available, else columns 3,2,4,1,5,0,6 for Connect4)
"""

import numpy as np
from typing import Optional


class MoveOrdering:
    """
    Move ordering heuristics for alpha-beta search.
    
    Maintains killer moves and history heuristic tables across search.
    """
    
    def __init__(self, max_ply: int = 50):
        """
        Initialize move ordering manager.
        
        Args:
            max_ply: Maximum ply depth to track (for killer/history tables)
        """
        self.max_ply = max_ply
        
        # Killer moves: 2 killers per ply (non-capture moves that caused beta cutoff)
        self.killer_moves = [[None, None] for _ in range(max_ply)]
        
        # History heuristic: [action] -> score (incremented by depth^2 on cutoff)
        # For Connect4: 7 columns
        self.history = np.zeros(7, dtype=np.int32)
    
    def reset(self):
        """Reset killer moves and history for new search."""
        self.killer_moves = [[None, None] for _ in range(self.max_ply)]
        self.history.fill(0)
    
    def update_killers(self, move: int, ply: int):
        """
        Update killer moves table when a quiet move causes beta cutoff.
        
        Args:
            move: Column index that caused cutoff
            ply: Current ply in search tree
        """
        if ply >= self.max_ply:
            return
        
        # If not already first killer, shift and add
        if self.killer_moves[ply][0] != move:
            self.killer_moves[ply][1] = self.killer_moves[ply][0]
            self.killer_moves[ply][0] = move
    
    def update_history(self, move: int, depth: int):
        """
        Update history heuristic when a move causes beta cutoff.
        
        Args:
            move: Column index that caused cutoff
            depth: Remaining depth at cutoff
        """
        # Increment by depth^2 (deeper cutoffs are more valuable)
        self.history[move] += depth * depth
    
    def order_moves(
        self,
        valid_moves: np.ndarray,
        game,
        state: np.ndarray,
        player: int,
        tt_move: Optional[int] = None,
        policy_priors: Optional[np.ndarray] = None,
        ply: int = 0
    ) -> list[int]:
        """
        Order moves for optimal alpha-beta pruning.
        
        Args:
            valid_moves: Binary mask of valid columns (7,)
            game: Game instance for checking winning/blocking moves
            state: Current board state
            player: Current player (1 or -1)
            tt_move: Best move from transposition table (highest priority)
            policy_priors: Neural network policy logits (7,)
            ply: Current ply in search tree
        
        Returns:
            List of column indices sorted by priority (best first)
        """
        moves = []
        move_scores = []
        
        # Get list of valid move indices
        valid_indices = np.where(valid_moves)[0]
        
        for move in valid_indices:
            score = 0.0
            
            # Priority 1: TT move (massive boost)
            if tt_move is not None and move == tt_move:
                score += 1_000_000
            
            # Priority 2: Winning move (immediate 4-in-a-row)
            if self._is_winning_move(game, state, move, player):
                score += 500_000
            
            # Priority 3: Blocking move (prevent opponent win)
            if self._is_blocking_move(game, state, move, player):
                score += 250_000
            
            # Priority 4: Policy priors from NN
            if policy_priors is not None:
                score += policy_priors[move] * 10_000
            
            # Priority 5: Killer moves
            if ply < self.max_ply:
                if self.killer_moves[ply][0] == move:
                    score += 5_000
                elif self.killer_moves[ply][1] == move:
                    score += 2_500
            
            # Priority 6: History heuristic
            score += self.history[move]
            
            # Priority 7: Positional preference (center columns better in Connect4)
            # Columns 3 (center) > 2,4 > 1,5 > 0,6
            center_bonus = {3: 100, 2: 75, 4: 75, 1: 50, 5: 50, 0: 25, 6: 25}
            score += center_bonus.get(move, 0)
            
            moves.append(move)
            move_scores.append(score)
        
        # Sort moves by score (descending)
        sorted_indices = np.argsort(move_scores)[::-1]
        ordered_moves = [moves[i] for i in sorted_indices]
        
        return ordered_moves
    
    def _is_winning_move(self, game, state: np.ndarray, move: int, player: int) -> bool:
        """
        Check if move wins immediately (makes 4-in-a-row).
        
        Args:
            game: Game instance
            state: Current board state
            move: Column to check
            player: Player making the move
        
        Returns:
            True if move wins game
        """
        # Simulate move
        next_state = game.get_next_state(state, move, player)
        return game.check_win(next_state, move)
    
    def _is_blocking_move(self, game, state: np.ndarray, move: int, player: int) -> bool:
        """
        Check if move blocks opponent from winning next turn.
        
        Args:
            game: Game instance
            state: Current board state
            move: Column to check
            player: Current player
        
        Returns:
            True if move blocks opponent win
        """
        opponent = -player
        
        # Check if opponent would win if they played here
        try:
            next_state = game.get_next_state(state, move, opponent)
            return game.check_win(next_state, move)
        except ValueError:
            # Column might be full
            return False


def order_moves_simple(
    valid_moves: np.ndarray,
    policy_priors: Optional[np.ndarray] = None
) -> list[int]:
    """
    Simple move ordering using only policy priors and center preference.
    
    Useful for root node or when full ordering is too expensive.
    
    Args:
        valid_moves: Binary mask of valid columns
        policy_priors: Neural network policy logits (optional)
    
    Returns:
        List of column indices sorted by priority
    """
    valid_indices = np.where(valid_moves)[0]
    
    if policy_priors is not None:
        # Sort by policy probability (higher is better)
        valid_priors = policy_priors[valid_indices]
        sorted_idx = np.argsort(valid_priors)[::-1]
        return [valid_indices[i] for i in sorted_idx]
    else:
        # Default: center-first ordering for Connect4
        # Order: 3, 2, 4, 1, 5, 0, 6
        center_order = [3, 2, 4, 1, 5, 0, 6]
        return [col for col in center_order if col in valid_indices]
