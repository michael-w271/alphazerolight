"""
Safe opening randomization for Connect4 teacher-solver training.

Implements high-temperature NN sampling with one-ply safety filter to avoid
immediate losing blunders while maintaining opening diversity.
"""

import numpy as np
import torch
import math
from typing import Tuple, List, Dict


def safe_opening_move(
    state: np.ndarray,
    player: int,
    model: torch.nn.Module,
    game,
    config: Dict,
) -> Tuple[np.ndarray, Dict]:
    """
    Select opening move using high-temperature NN sampling with safety filter.
    
    Args:
        state: Board state (6, 7)
        player: Current player (1 or -1)
        model: Neural network model
        game: ConnectFour game instance
        config: Opening randomization config dict
    
    Returns:
        (pi, metadata):
            - pi: Policy distribution over actions
            - metadata: Dict with source, filtered_moves, etc.
    """
    temperature = config.get('temperature', 1.8)
    top_k = config.get('top_k', 4)
    safety_config = config.get('safety_filter', {})
    
    # Get NN policy
    state_tensor = _state_to_tensor(state, player)
    model.eval()
    with torch.no_grad():
        policy_logits, _ = model(state_tensor)
        pi_raw = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
    
    # Get legal moves
    legal_mask = _get_legal_mask(state)
    
    # Apply top-k filtering
    if top_k > 0 and top_k < len(pi_raw):
        # Zero out all but top-k moves
        top_k_indices = np.argsort(pi_raw)[-top_k:]
        pi_filtered = np.zeros_like(pi_raw)
        pi_filtered[top_k_indices] = pi_raw[top_k_indices]
        pi_raw = pi_filtered
    
    # Mask illegal moves
    pi_raw = pi_raw * legal_mask
    if pi_raw.sum() > 0:
        pi_raw = pi_raw / pi_raw.sum()
    else:
        # Fallback: uniform over legal
        pi_raw = legal_mask / legal_mask.sum()
    
    # Apply safety filter if enabled
    if safety_config.get('enabled', True):
        safe_moves = one_ply_safety_filter(state, player, game)
        
        # Intersect with legal moves
        safe_legal_moves = [m for m in safe_moves if legal_mask[m] > 0]
        
        if len(safe_legal_moves) > 0:
            # Restrict policy to safe moves
            pi_safe = np.zeros_like(pi_raw)
            for move in safe_legal_moves:
                pi_safe[move] = pi_raw[move]
            
            if pi_safe.sum() > 0:
                pi_safe = pi_safe / pi_safe.sum()
                pi_raw = pi_safe
            else:
                # All safe moves had zero probability
                # Fallback: uniform over safe moves
                for move in safe_legal_moves:
                    pi_safe[move] = 1.0 / len(safe_legal_moves)
                pi_raw = pi_safe
        else:
            # No safe moves found (all moves give opponent immediate win)
            # Fallback: reduce temperature and use best NN move
            print("⚠️  WARNING: No safe opening moves found! Using best NN move.")
            fallback_temp = safety_config.get('fallback_temperature', 1.0)
            temperature = fallback_temp
    
    # Apply temperature
    if temperature != 1.0:
        # Adjust probabilities with temperature
        log_probs = np.log(pi_raw + 1e-10)
        tempered_log_probs = log_probs / temperature
        pi_tempered = np.exp(tempered_log_probs)
        pi_tempered = pi_tempered / pi_tempered.sum()
        pi_raw = pi_tempered
    
    metadata = {
        'source': 'opening_random',
        'temperature': temperature,
        'top_k': top_k,
        'safety_filtered': safety_config.get('enabled', True),
    }
    
    return pi_raw, metadata


def one_ply_safety_filter(
    state: np.ndarray,
    player: int,
    game
) -> List[int]:
    """
    Filter out moves that give opponent an immediate winning move.
    
    Args:
        state: Board state (6, 7)
        player: Current player (1 or -1)
        game: ConnectFour game instance
    
    Returns:
        List of safe column indices (0-6)
    """
    legal_moves = _get_legal_moves(state)
    safe_moves = []
    
    for move in legal_moves:
        # Try this move
        next_state = _apply_move(state.copy(), move, player)
        
        # Check if opponent has an immediate winning move
        opponent = -player
        opponent_can_win = False
        
        opponent_legal_moves = _get_legal_moves(next_state)
        for opp_move in opponent_legal_moves:
            next_next_state = _apply_move(next_state.copy(), opp_move, opponent)
            
            # Check if this creates a win for opponent
            if game.check_win(next_next_state, opp_move):
                opponent_can_win = True
                break
        
        # If opponent cannot win immediately, this move is safe
        if not opponent_can_win:
            safe_moves.append(move)
    
    return safe_moves


def compute_policy_entropy(pi: np.ndarray) -> float:
    """
    Compute entropy of policy distribution.
    
    Args:
        pi: Policy distribution (action_size,)
    
    Returns:
        Entropy in nats
    """
    # Filter out zero probabilities
    pi_nonzero = pi[pi > 0]
    
    if len(pi_nonzero) == 0:
        return 0.0
    
    entropy = -np.sum(pi_nonzero * np.log(pi_nonzero + 1e-10))
    return entropy


def _get_legal_moves(state: np.ndarray) -> List[int]:
    """Get list of legal column indices."""
    legal = []
    for col in range(7):
        if state[0, col] == 0:  # Top row empty
            legal.append(col)
    return legal


def _get_legal_mask(state: np.ndarray) -> np.ndarray:
    """Get binary mask of legal moves."""
    mask = np.zeros(7, dtype=np.float32)
    for col in range(7):
        if state[0, col] == 0:
            mask[col] = 1.0
    return mask


def _apply_move(state: np.ndarray, col: int, player: int) -> np.ndarray:
    """
    Apply a move to the board.
    
    Args:
        state: Current state (6, 7) - will be modified
        col: Column index (0-6)
        player: Player value (1 or -1)
    
    Returns:
        Modified state
    """
    # Find lowest empty row in column
    for row in range(5, -1, -1):
        if state[row, col] == 0:
            state[row, col] = player
            break
    
    return state


def _state_to_tensor(state: np.ndarray, player: int) -> torch.Tensor:
    """
    Convert numpy state to tensor in canonical form.
    
    Args:
        state: Numpy state (6, 7)
        player: Current player (for perspective change)
    
    Returns:
        Tensor (1, 1, 6, 7) in canonical form
    """
    # Change perspective to player
    canonical_state = state * player
    
    # Convert to tensor
    state_tensor = torch.FloatTensor(canonical_state).unsqueeze(0).unsqueeze(0)
    
    if torch.cuda.is_available():
        state_tensor = state_tensor.cuda()
    
    return state_tensor
