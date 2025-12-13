"""
Simple heuristic opponent for Connect Four warmup training.
Uses basic 1-ply lookahead: win if possible, block if necessary, otherwise random.
"""
import numpy as np


class HeuristicOpponent:
    """
    1-ply heuristic opponent for Connect Four.
    
    Strategy:
    1. If I can win (make 4-in-a-row), do it
    2. If opponent can win next turn, block it
    3. Otherwise, pick random valid move
    """
    
    def __init__(self, game):
        self.game = game
    
    def get_action(self, state, player):
        """
        Get best heuristic move for given position.
        
        Args:
            state: Current board state
            player: Player making the move (1 or -1)
        
        Returns:
            Column index (action)
        """
        valid_moves = self.game.get_valid_moves(state)
        valid_actions = np.where(valid_moves == 1)[0]
        
        if len(valid_actions) == 0:
            raise ValueError("No valid moves available")
        
        # 1. Check if we can win
        for action in valid_actions:
            next_state = self.game.get_next_state(state, action, player)
            if self.game.check_win(next_state, action):
                return action
        
        # 2. Check if opponent can win (we must block)
        opponent = -player
        for action in valid_actions:
            next_state = self.game.get_next_state(state, action, opponent)
            if self.game.check_win(next_state, action):
                # Opponent would win here, we must block
                return action
        
        # 3. No immediate threats - pick random move
        # Slight preference for center columns (better Connect Four strategy)
        center = self.game.column_count // 2
        weights = np.array([1.0 / (1.0 + abs(i - center)) for i in valid_actions])
        weights /= weights.sum()
        
        return np.random.choice(valid_actions, p=weights)
