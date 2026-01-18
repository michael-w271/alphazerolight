"""
Alpha-beta negamax search engine for Connect4.

This is the core Stockfish-like search algorithm that replaces MCTS at runtime.
Unlike MCTS, alpha-beta uses exact minimax search with pruning to explore the
game tree efficiently.

Key features:
- Negamax framework (simplified minimax using negation)
- Alpha-beta pruning (cut branches that can't affect final result)
- Iterative deepening (search depth 1, then 2, then 3... until time expires)
- Transposition table integration
- Move ordering for optimal pruning
- Time management (stop cleanly when budget exhausted)
- Principal variation extraction


Algorithm overview:

    def negamax(state, depth, alpha, beta):
        # Terminal or depth limit
        if terminal or depth == 0:
            return static_eval(state)
        
        # Transposition table lookup
        if tt_entry := tt.probe(state, depth, alpha, beta):
            return tt_entry.score
        
        # Search all moves
        best_score = -infinity
        for move in ordered_moves:
            next_state = make_move(state, move)
            score = -negamax(next_state, depth-1, -beta, -alpha)
            undo_move(state, move)
            
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            
            if alpha >= beta:
                break  # Beta cutoff
        
        # Store in TT
        tt.store(state, depth, best_score, bound_type, best_move)
        return best_score
"""

import numpy as np
import time
from typing import Optional, Tuple
from dataclasses import dataclass

from alpha_zero_light.engine.zobrist import ZobristHasher, get_zobrist_hasher
from alpha_zero_light.engine.transposition_table import TranspositionTable, BoundType
from alpha_zero_light.engine.move_ordering import MoveOrdering


@dataclass
class SearchResult:
    """Result of alpha-beta search."""
    best_move: int
    score: float
    depth_reached: int
    nodes_searched: int
    time_ms: int
    principal_variation: list[int]
    tt_stats: dict


# Sentinel values for win/loss/draw
SCORE_WIN = 100000
SCORE_LOSS = -100000
SCORE_DRAW = 0
SCORE_INF = 1000000


class AlphaBetaEngine:
    """
    Alpha-beta negamax search engine with iterative deepening.
    
    This engine does NOT use perfect solving at runtime - it relies on:
    1. Alpha-beta search to explore game tree
    2. Neural network for leaf node evaluation
    3. Transposition table for caching
    4. Move ordering for efficient pruning
    """
    
    def __init__(
        self,
        game,
        evaluator=None,
        model=None,
        tt_size_mb: int = 256,
        use_killer_moves: bool = True,
        use_history_heuristic: bool = True,
        max_depth: int = 50
    ):
        """
        Initialize alpha-beta engine.
        
        Args:
            game: Connect4 game instance
            evaluator: Evaluation function (state, player) -> float (preferred)
            model: Neural network for evaluation (legacy, for backward compatibility)
            tt_size_mb: Transposition table size in MB
            use_killer_moves: Enable killer move heuristic
            use_history_heuristic: Enable history heuristic
            max_depth: Maximum search depth limit
        """
        self.game = game
        self.evaluator = evaluator  # Preferred: WDL evaluator function
        self.model = model  # Legacy: for backward compatibility
        self.max_depth = max_depth
        
        # Initialize components
        self.zobrist = get_zobrist_hasher(game.row_count, game.column_count)
        self.tt = TranspositionTable(size_mb=tt_size_mb)
        self.move_ordering = MoveOrdering(max_ply=max_depth)
        
        self.use_killer_moves = use_killer_moves
        self.use_history_heuristic = use_history_heuristic
        
        # Search statistics
        self.nodes_searched = 0
        self.start_time = 0
        self.time_limit_ms = 0
        self.stopped = False
        
        # Principal variation
        self.pv = []
    
    def search(
        self,
        state: np.ndarray,
        player: int,
        time_limit_ms: int = 1000,
        max_depth: Optional[int] = None
    ) -> SearchResult:
        """
        Main search entry point with iterative deepening.
        
        Strategy:
        - Search depth 1, then 2, then 3... until time expires
        - Always keep best move from last completed depth
        - Return gracefully when time runs out
        
        Args:
            state: Current board state
            player: Current player to move (1 or -1)
            time_limit_ms: Time budget in milliseconds
            max_depth: Override maximum depth
        
        Returns:
            SearchResult with best move, score, statistics
        """
        self.start_time = time.time() * 1000  # Convert to ms
        self.time_limit_ms = time_limit_ms
        self.stopped = False
        self.nodes_searched = 0
        
        # Reset move ordering tables for new search
        self.move_ordering.reset()
        self.tt.new_search()
        
        effective_max_depth = max_depth if max_depth is not None else self.max_depth
        
        # Check if position is already terminal
        valid_moves = self.game.get_valid_moves(state)
        if np.sum(valid_moves) == 0:
            # Board is full - draw
            return SearchResult(
                best_move=-1,  # No move available
                score=SCORE_DRAW,
                depth_reached=0,
                nodes_searched=0,
                time_ms=0,
                principal_variation=[],
                tt_stats=self.tt.get_stats()
            )
        
        # Initialize with first valid move (fallback)
        best_move = np.where(valid_moves)[0][0]
        best_score = SCORE_LOSS
        depth_reached = 0
        self.pv = []
        
        # Iterative deepening
        for depth in range(1, effective_max_depth + 1):
            if self._time_up():
                break
            
            # Search at current depth
            try:
                score, move, pv = self._search_root(state, player, depth)
                
                # Update best result if search completed
                if not self.stopped:
                    best_move = move
                    best_score = score
                    depth_reached = depth
                    self.pv = pv
                    
                    # Stop if we found a forced win/loss
                    if abs(score) >= SCORE_WIN - 100:
                        break
            except TimeoutError:
                # Time expired during search
                break
        
        elapsed_ms = int(time.time() * 1000 - self.start_time)
        
        return SearchResult(
            best_move=best_move,
            score=best_score,
            depth_reached=depth_reached,
            nodes_searched=self.nodes_searched,
            time_ms=elapsed_ms,
            principal_variation=self.pv,
            tt_stats=self.tt.get_stats()
        )
    
    def _search_root(
        self,
        state: np.ndarray,
        player: int,
        depth: int
    ) -> Tuple[float, int, list[int]]:
        """
        Root node search (no alpha-beta window).
        
        Args:
            state: Board state
            player: Player to move
            depth: Search depth
        
        Returns:
            (score, best_move, principal_variation)
        """
        valid_moves = self.game.get_valid_moves(state)
        hash_val = self.zobrist.hash_position(state, player)
        
        # Get TT move hint
        tt_move = self.tt.get_best_move(hash_val)
        
        # Get policy priors if model available
        policy_priors = self._get_policy_priors(state, player)
        
        # Order moves
        ordered_moves = self.move_ordering.order_moves(
            valid_moves, self.game, state, player,
            tt_move=tt_move, policy_priors=policy_priors, ply=0
        )
        
        best_score = SCORE_LOSS
        best_move = ordered_moves[0]
        pv = [best_move]
        alpha = -SCORE_INF
        beta = SCORE_INF
        
        for move in ordered_moves:
            if self._time_up():
                raise TimeoutError("Time limit exceeded")
            
            # Make move
            next_state = self.game.get_next_state(state, move, player)
            
            # Check if terminal
            value, terminated = self.game.get_value_and_terminated(next_state, move)
            
            if terminated:
                if value > 0:  # We won
                    score = SCORE_WIN - depth
                else:  # Draw
                    score = SCORE_DRAW
            else:
                # Change perspective and search
                next_state_opp = self.game.change_perspective(next_state, player)
                score = -self._negamax(next_state_opp, -player, depth - 1, -beta, -alpha, ply=1)
            
            if score > best_score:
                best_score = score
                best_move = move
                pv = [move]  # Simple PV for now
                alpha = max(alpha, score)
        
        # Store in TT
        self.tt.store(hash_val, depth, best_score, BoundType.EXACT, best_move)
        
        return best_score, best_move, pv
    
    def _negamax(
        self,
        state: np.ndarray,
        player: int,
        depth: int,
        alpha: float,
        beta: float,
        ply: int
    ) -> float:
        """
        Negamax alpha-beta search.
        
        Args:
            state: Board state (from current player's perspective)
            player: Current player
            depth: Remaining depth
            alpha: Alpha bound
            beta: Beta bound
            ply: Ply from root (for killer moves)
        
        Returns:
            Score from current player's perspective
        """
        self.nodes_searched += 1
        
        # Time check every ~1000 nodes
        if self.nodes_searched % 1000 == 0 and self._time_up():
            self.stopped = True
            raise TimeoutError("Time limit exceeded")
        
        # Probe transposition table
        hash_val = self.zobrist.hash_position(state, player)
        tt_result = self.tt.probe(hash_val, depth, alpha, beta)
        if tt_result is not None:
            return tt_result[0]
        
        # Terminal check or depth limit
        valid_moves = self.game.get_valid_moves(state)
        if np.sum(valid_moves) == 0:
            # Draw (board full)
            return SCORE_DRAW
        
        if depth <= 0:
            # Leaf node: evaluate
            return self._evaluate(state, player)
        
        # Get move ordering
        tt_move = self.tt.get_best_move(hash_val)
        policy_priors = self._get_policy_priors(state, player)
        
        ordered_moves = self.move_ordering.order_moves(
            valid_moves, self.game, state, player,
            tt_move=tt_move, policy_priors=policy_priors, ply=ply
        )
        
        best_score = -SCORE_INF
        best_move = None
        original_alpha = alpha
        
        for move in ordered_moves:
            # Make move
            next_state = self.game.get_next_state(state, move, player)
            
            # Check if terminal
            value, terminated = self.game.get_value_and_terminated(next_state, move)
            
            if terminated:
                if value > 0:  # Current player wins
                    score = SCORE_WIN - ply
                else:  # Draw
                    score = SCORE_DRAW
            else:
                # Recursive search
                next_state_opp = self.game.change_perspective(next_state, player)
                score = -self._negamax(next_state_opp, -player, depth - 1, -beta, -alpha, ply + 1)
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
            
            # Beta cutoff
            if alpha >= beta:
                # Update killer move (if not a winning move)
                if self.use_killer_moves and abs(score) < SCORE_WIN - 100:
                    self.move_ordering.update_killers(move, ply)
                
                # Update history
                if self.use_history_heuristic:
                    self.move_ordering.update_history(move, depth)
                
                break
        
        # Determine bound type for TT
        if best_score <= original_alpha:
            bound = BoundType.UPPER  # All moves failed low
        elif best_score >= beta:
            bound = BoundType.LOWER  # We failed high
        else:
            bound = BoundType.EXACT  # PV node
        
        # Store in TT
        self.tt.store(hash_val, depth, best_score, bound, best_move)
        
        return best_score
    
    def _evaluate(self, state: np.ndarray, player: int) -> float:
        """
        Evaluate leaf node position.
        
        Uses evaluator function if available, otherwise neural network or heuristic.
        
        Args:
            state: Board state (current player's perspective)
            player: Current player
        
        Returns:
            Score from current player's perspective
        """
        # Prefer evaluator function (WDL evaluator)
        if self.evaluator is not None:
            # Evaluator returns score in [-1, 1], scale to our range
            score = self.evaluator(state, player)
            return score * 10000
        
        if self.model is not None:
            # Use neural network evaluation
            import torch
            encoded = self.game.get_encoded_state(state)
            encoded_t = torch.tensor(encoded, dtype=torch.float32, device=self.model.device).unsqueeze(0)
            
            with torch.no_grad():
                _, value = self.model(encoded_t)
                # value is in [-1, 1], scale to our score range
                return float(value.item()) * 10000
        else:
            # Simple heuristic: count potential 4-in-a-rows
            return self._simple_heuristic(state, player)
    
    def _simple_heuristic(self, state: np.ndarray, player: int) -> float:
        """
        Simple position evaluation (when no NN available).
        
        Counts number of 2-in-a-rows and 3-in-a-rows for both players.
        """
        score = 0.0
        
        # Prefer central columns
        for col in range(self.game.column_count):
            center_dist = abs(col - 3)
            for row in range(self.game.row_count):
                if state[row, col] == player:
                    score += 10 * (4 - center_dist)
                elif state[row, col] == -player:
                    score -= 10 * (4 - center_dist)
        
        return score
    
    def _get_policy_priors(self, state: np.ndarray, player: int) -> Optional[np.ndarray]:
        """
        Get policy priors from evaluator or neural network for move ordering.
        
        Returns:
            Policy logits (7,) or None if no model/evaluator
        """
        # Try WDL evaluator first (may have get_policy_priors method)
        if self.evaluator is not None and hasattr(self.evaluator, 'get_policy_priors'):
            try:
                return self.evaluator.get_policy_priors(state, player)
            except Exception:
                pass
        
        if self.model is None:
            return None
        
        try:
            import torch
            encoded = self.game.get_encoded_state(state)
            encoded_t = torch.tensor(encoded, dtype=torch.float32, device=self.model.device).unsqueeze(0)
            
            with torch.no_grad():
                policy, _ = self.model(encoded_t)
                policy_np = policy.cpu().numpy()[0]
                return policy_np
        except Exception:
            return None
    
    def _time_up(self) -> bool:
        """Check if time limit exceeded."""
        if self.time_limit_ms <= 0:
            return False
        
        elapsed_ms = time.time() * 1000 - self.start_time
        return elapsed_ms >= self.time_limit_ms
    
    def clear_tt(self):
        """Clear transposition table."""
        self.tt.clear()
    
    def get_stats(self) -> dict:
        """Get search statistics."""
        return {
            'nodes_searched': self.nodes_searched,
            'tt_stats': self.tt.get_stats(),
            'tt_fill_rate': self.tt.get_fill_rate(),
        }
