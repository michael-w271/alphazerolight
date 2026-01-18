"""
Teacher-Solver module for Expert-Iteration training.

This module wraps the Connect4Solver to provide teacher-guided move selection
during self-play. It implements the move-source policy that decides when to use:
1. Forced-win override (solver detects imminent win)
2. Solver-guided moves (scheduled by ply)
3. Safe opening randomization (high-temp NN with safety filter)
4. NN-only moves
5. Minimal MCTS fallback (rare)

The solver provides perfect policy targets (best moves) and WDL value labels
for training, replacing the need for heavy MCTS searches.
"""

import numpy as np
import torch
import time
from typing import Tuple, Optional, Dict, List
from pathlib import Path

from alpha_zero_light.data.solver_interface import Connect4Solver


class TeacherSolver:
    """
    Teacher-solver wrapper for expert-iteration training.
    
    Provides move selection and policy/value targets during self-play by
    querying a perfect Connect4 solver instead of running MCTS.
    """
    
    def __init__(self, game, config: Dict):
        """
        Initialize teacher-solver.
        
        Args:
            game: ConnectFour game instance
            config: TEACHER_SOLVER_CONFIG dict with solver settings
        """
        self.game = game
        self.config = config
        
        # Initialize solver
        solver_path = config.get('solver_path', None)
        cache_size = config.get('cache_size', 100000)
        self.solver = Connect4Solver(solver_path=solver_path, cache_size=cache_size)
        
        # Config sections
        self.force_win_config = config.get('force_win_override', {})
        self.solver_schedule = config.get('solver_schedule', [])
        self.opening_config = config.get('opening_randomization', {})
        self.mcts_config = config.get('mcts_fallback', {})
        
        # Statistics
        self.stats = {
            'solver_calls': 0,
            'solver_timeouts': 0,
            'forced_win_overrides': 0,
            'solver_moves': 0,
            'opening_moves': 0,
            'nn_moves': 0,
            'mcts_fallback_moves': 0,
            'total_solver_time_ms': 0,
        }
    
    def query_teacher(
        self, 
        state: np.ndarray, 
        player: int, 
        ply: int,
        model: Optional[torch.nn.Module] = None,
        mcts = None
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Query teacher for policy and value targets.
        
        Implements the move-source policy:
        1. Opening plies (0-6): Safe high-temp NN + MCTS fallback (solver too slow)
        2. Forced-win override (if enabled and detected)
        3. Scheduled solver probability by ply (7+)
        4. NN-only for remaining cases
        
        Args:
            state: Board state (1, 1, 6, 7) tensor or (6, 7) numpy
            player: Current player (1 or -1)
            ply: Current ply number (number of pieces on board)
            model: Neural network model (for NN-based moves)
            mcts: MCTS instance (for fallback, optional)
        
        Returns:
            (pi_target, v_target, metadata):
                - pi_target: Policy distribution over actions (7,)
                - v_target: Scalar value target for this position
                - metadata: Dict with 'source', 'wdl', 'score', 'dtw', etc.
        """
        # 1. Opening strategy (plies 0-15): Fast MCTS with temperature control
        # - Plies 0-4: High temp (1.5) for opening variety
        # - Plies 5-15: Low temp (0.5) for fast, focused play
        # - Plies 16+: Solver takes over (much faster)
        if ply <= 15 and mcts is not None:
            try:
                state_for_mcts = self._to_numpy(state)
                canonical_state = state_for_mcts * player
                
                # Use 100-search MCTS (fast)
                old_searches = mcts.args.get('num_searches', 400)
                mcts.args['num_searches'] = 100
                
                pi = mcts.search(canonical_state)
                
                # Restore
                mcts.args['num_searches'] = old_searches
                
                self.stats['opening_moves'] += 1
                v = 0.0
                meta = {
                    'source': 'opening_mcts',
                    'success': True,
                    'mcts_searches': 100,
                    'ply': ply
                }
                return pi, v, meta
            except Exception as e:
                print(f"⚠️ Opening MCTS failed: {e}")
        
        # 2. Check forced-win override (mid/late game)
        if self.force_win_config.get('enabled', True):
            forced, pi, v, meta = self._check_forced_win(state, player)
            if forced:
                self.stats['forced_win_overrides'] += 1
                self.stats['solver_moves'] += 1
                return pi, v, meta
        
        # 3. Check solver schedule probability
        solver_prob = self._get_solver_prob(ply)
        if solver_prob > 0 and np.random.random() < solver_prob:
            # Use solver
            pi, v, meta = self._solver_policy_and_value(state, player)
            if meta['success']:
                self.stats['solver_moves'] += 1
                return pi, v, meta
            # If solver failed (timeout), fall through to NN
        
        # 4. NN-only move (model must be provided)
        if model is not None:
            pi, v, meta = self._nn_policy_and_value(state, player, model)
            self.stats['nn_moves'] += 1
            return pi, v, meta
        
        # 5. Fallback: uniform random (should rarely happen)
        meta = {'source': 'uniform_random', 'success': True}
        pi = np.ones(self.game.action_size) / self.game.action_size
        v = 0.0
        return pi, v, meta
    
    def _check_forced_win(
        self, 
        state: np.ndarray, 
        player: int
    ) -> Tuple[bool, Optional[np.ndarray], Optional[float], Dict]:
        """
        Check if position has forced win within threshold.
        
        Args:
            state: Board state
            player: Current player
        
        Returns:
            (is_forced, pi, v, metadata):
                - is_forced: True if forced win detected
                - pi: One-hot policy if forced, else None
                - v: Value target if forced, else None
                - metadata: Dict with solver info
        """
        if not self.force_win_config.get('enabled', True):
            return False, None, None, {}
        
        dtw_threshold = self.force_win_config.get('dtw_threshold_plies', 8)
        timeout_ms = self.force_win_config.get('solver_timeout_ms', 2000)
        
        # Convert state to numpy if needed
        state_np = self._to_numpy(state)
        
        # Solve position
        start_time = time.time()
        try:
            wdl, score, _ = self.solver.solve_board(state_np, player)
            solve_time_ms = (time.time() - start_time) * 1000
            self.stats['solver_calls'] += 1
            self.stats['total_solver_time_ms'] += solve_time_ms
            
            # Check if forced win
            if wdl == 1 and score > 0 and score <= dtw_threshold:
                # Forced win detected! Find best move
                best_moves = self._get_best_solver_moves(state_np, player, target_wdl=1)
                
                # Create one-hot policy over best moves
                pi = np.zeros(self.game.action_size, dtype=np.float32)
                if best_moves:
                    for move in best_moves:
                        pi[move] = 1.0 / len(best_moves)
                else:
                    # Fallback: uniform over legal moves
                    pi = self._get_legal_move_mask(state_np)
                    pi = pi / pi.sum()
                
                # Value target: +1 for win
                v = 1.0
                
                meta = {
                    'source': 'forced_solver',
                    'wdl': wdl,
                    'score': score,
                    'dtw': score,
                    'solve_time_ms': solve_time_ms,
                    'success': True,
                    'best_moves': best_moves,
                }
                
                return True, pi, v, meta
            
            # Not a forced win
            return False, None, None, {'wdl': wdl, 'score': score}
        
        except Exception as e:
            self.stats['solver_timeouts'] += 1
            return False, None, None, {'error': str(e), 'success': False}
    
    def _solver_policy_and_value(
        self, 
        state: np.ndarray, 
        player: int,
        timeout_ms: int = 2000
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Get policy and value from solver.
        
        Finds best move(s) by solving the current position and trying
        all legal moves.
        
        Args:
            state: Board state
            player: Current player
            timeout_ms: Solver timeout in milliseconds
        
        Returns:
            (pi, v, metadata)
        """
        state_np = self._to_numpy(state)
        
        start_time = time.time()
        try:
            # Solve current position
            wdl, score, _ = self.solver.solve_board(state_np, player)
            
            # Find best move(s)
            best_moves = self._get_best_solver_moves(state_np, player, target_wdl=wdl)
            
            solve_time_ms = (time.time() - start_time) * 1000
            self.stats['solver_calls'] += 1
            self.stats['total_solver_time_ms'] += solve_time_ms
            
            # Create policy distribution
            pi = np.zeros(self.game.action_size, dtype=np.float32)
            if best_moves:
                for move in best_moves:
                    pi[move] = 1.0 / len(best_moves)
            else:
                # Fallback: uniform over legal
                pi = self._get_legal_move_mask(state_np)
                pi = pi / pi.sum() if pi.sum() > 0 else pi
            
            # Value target: map WDL to scalar
            v = float(wdl)  # -1, 0, or +1
            
            meta = {
                'source': 'solver',
                'wdl': wdl,
                'score': score,
                'solve_time_ms': solve_time_ms,
                'success': True,
                'best_moves': best_moves,
            }
            
            return pi, v, meta
        
        except Exception as e:
            self.stats['solver_timeouts'] += 1
            
            # Fallback: uniform over legal moves, neutral value
            pi = self._get_legal_move_mask(state_np)
            pi = pi / pi.sum() if pi.sum() > 0 else pi
            v = 0.0
            
            meta = {
                'source': 'solver_failed',
                'error': str(e),
                'success': False,
            }
            
            return pi, v, meta
    
    def _get_best_solver_moves(
        self, 
        state_np: np.ndarray, 
        player: int,
        target_wdl: int
    ) -> List[int]:
        """
        Find all moves that maintain the target WDL outcome.
        
        Args:
            state_np: Board state (6, 7)
            player: Current player
            target_wdl: Target WDL to maintain (from current position)
        
        Returns:
            List of column indices (0-6) that achieve target WDL
        """
        legal_moves = self._get_legal_moves(state_np)
        best_moves = []
        
        for move in legal_moves:
            # Try this move
            next_state = self._apply_move(state_np, move, player)
            
            # Solve next position from opponent's perspective
            try:
                opp_wdl, opp_score, _ = self.solver.solve_board(next_state, -player)
                
                # Flip WDL to current player's perspective
                our_wdl = -opp_wdl
                
                # If this move achieves our target WDL, it's a best move
                if our_wdl == target_wdl:
                    best_moves.append(move)
                    
                    # If target was WIN, also check if this move is faster
                    # (for now, just collect all winning moves)
            except:
                # Skip if solver fails on this move
                continue
        
        return best_moves
    
    def _nn_policy_and_value(
        self, 
        state: np.ndarray, 
        player: int, 
        model: torch.nn.Module
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Get policy and value from neural network.
        
        Args:
            state: Board state
            player: Current player
            model: Neural network model
        
        Returns:
            (pi, v, metadata)
        """
        # Convert to numpy and get canonical state
        state_np = self._to_numpy(state)
        canonical_state = state_np * player
        
        # CRITICAL: Encode state for NN (3 channels: current player, opponent, empty)
        # Model expects (B, 3, H, W) not (B, 1, H, W)
        encoded_state = self.game.get_encoded_state(canonical_state)
        
        # Convert to tensor (B, 3, H, W)
        if isinstance(encoded_state, np.ndarray):
            state_tensor = torch.FloatTensor(encoded_state).unsqueeze(0)
        else:
            state_tensor = encoded_state.unsqueeze(0) if encoded_state.dim() == 3 else encoded_state
        
        if torch.cuda.is_available():
            state_tensor = state_tensor.cuda()
        
        model.eval()
        with torch.no_grad():
            policy_logits, value = model(state_tensor)
            
            # Convert to numpy
            pi = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            v = value.cpu().item()
        
        meta = {
            'source': 'nn',
            'success': True,
        }
        
        return pi, v, meta
    
    def _get_solver_prob(self, ply: int) -> float:
        """
        Get solver probability for current ply based on schedule.
        
        Args:
            ply: Current ply number
        
        Returns:
            Probability of using solver (0.0 to 1.0)
        """
        for entry in self.solver_schedule:
            ply_range = entry.get('ply_range', [0, 42])
            if ply_range[0] <= ply <= ply_range[1]:
                return entry.get('prob', 0.0)
        
        # Default: no solver
        return 0.0
    
    def _get_legal_moves(self, state_np: np.ndarray) -> List[int]:
        """Get list of legal column indices."""
        legal = []
        for col in range(7):
            if state_np[0, col] == 0:  # Top row empty
                legal.append(col)
        return legal
    
    def _get_legal_move_mask(self, state_np: np.ndarray) -> np.ndarray:
        """Get binary mask of legal moves."""
        mask = np.zeros(7, dtype=np.float32)
        for col in range(7):
            if state_np[0, col] == 0:
                mask[col] = 1.0
        return mask
    
    def _apply_move(self, state_np: np.ndarray, col: int, player: int) -> np.ndarray:
        """
        Apply a move to the board and return new state.
        
        Args:
            state_np: Current state (6, 7)
            col: Column index (0-6)
            player: Player value (1 or -1)
        
        Returns:
            New state after move
        """
        new_state = state_np.copy()
        
        # Find lowest empty row in column
        for row in range(5, -1, -1):
            if new_state[row, col] == 0:
                new_state[row, col] = player
                break
        
        return new_state
    
    def _to_numpy(self, state) -> np.ndarray:
        """Convert state to numpy (6, 7) array."""
        if isinstance(state, torch.Tensor):
            state_np = state.cpu().numpy()
            # Handle (1, 1, 6, 7) or (1, 6, 7) or (6, 7)
            while state_np.ndim > 2:
                state_np = state_np.squeeze(0)
            return state_np
        elif isinstance(state, np.ndarray):
            state_np = state
            while state_np.ndim > 2:
                state_np = state_np.squeeze(0)
            return state_np
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")
    
    def _to_tensor(self, state: np.ndarray, player: int) -> torch.Tensor:
        """
        Convert numpy state to tensor in canonical form.
        
        Args:
            state: Numpy state (6, 7)
            player: Current player (for perspective change)
        
        Returns:
            Tensor (1, 1, 6, 7) in canonical form
        """
        state_np = self._to_numpy(state)
        
        # Change perspective to player
        canonical_state = state_np * player
        
        # Convert to tensor
        state_tensor = torch.FloatTensor(canonical_state).unsqueeze(0).unsqueeze(0)
        
        if torch.cuda.is_available():
            state_tensor = state_tensor.cuda()
        
        return state_tensor
    
    def get_stats(self) -> Dict:
        """Get usage statistics."""
        total_moves = (self.stats['solver_moves'] + self.stats['nn_moves'] + 
                      self.stats['mcts_fallback_moves'] + self.stats['opening_moves'])
        
        if total_moves == 0:
            return self.stats
        
        avg_solver_time = (self.stats['total_solver_time_ms'] / self.stats['solver_calls'] 
                          if self.stats['solver_calls'] > 0 else 0)
        
        return {
            **self.stats,
            'solver_move_fraction': self.stats['solver_moves'] / total_moves,
            'forced_win_fraction': self.stats['forced_win_overrides'] / total_moves,
            'nn_move_fraction': self.stats['nn_moves'] / total_moves,
            'mcts_fallback_fraction': self.stats['mcts_fallback_moves'] / total_moves,
            'opening_fraction': self.stats['opening_moves'] / total_moves,
            'solver_timeout_rate': (self.stats['solver_timeouts'] / self.stats['solver_calls']
                                   if self.stats['solver_calls'] > 0 else 0),
            'avg_solver_time_ms': avg_solver_time,
        }
    
    def reset_stats(self):
        """Reset statistics counters."""
        for key in self.stats:
            self.stats[key] = 0
