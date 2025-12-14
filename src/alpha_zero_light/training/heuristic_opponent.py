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
    
    def get_action_aggressive(self, state, player):
        """
        Aggressive opponent that actively creates threats.
        
        Strategy:
        1. If I can win, do it
        2. If opponent can win, block it
        3. Try to create a 3-in-a-row threat
        4. Otherwise random (prefer center)
        
        This forces the model to constantly practice blocking.
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
        
        # 2. Check if opponent can win (must block)
        opponent = -player
        for action in valid_actions:
            next_state = self.game.get_next_state(state, action, opponent)
            if self.game.check_win(next_state, action):
                return action
        
        # 3. Try to create a 3-in-a-row threat (aggressive play)
        best_threat_action = None
        for action in valid_actions:
            next_state = self.game.get_next_state(state, action, player)
            if self._creates_threat(next_state, action, player):
                best_threat_action = action
                break  # Take first threat found
        
        if best_threat_action is not None:
            return best_threat_action
        
        # 4. No threats possible - prefer center
        center = self.game.column_count // 2
        weights = np.array([1.0 / (1.0 + abs(i - center)) for i in valid_actions])
        weights /= weights.sum()
        
        return np.random.choice(valid_actions, p=weights)
    
    def _creates_threat(self, state, last_action, player):
        """
        Check if the last move created a 3-in-a-row threat.
        A threat is 3 pieces in a line with an empty space to complete 4.
        """
        # This is a simplified check - just look for 3 in any direction
        # In Connect Four, if we have 3 in a row, it's usually threatening
        
        # Get the position of the last piece
        col = last_action
        row = 0
        for r in range(self.game.row_count):
            if state[0, r, col] == player:
                row = r
                break
        
        # Check all 4 directions for sequences of 3
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal /
            (1, -1),  # Diagonal \
        ]
        
        for dr, dc in directions:
            count = 1  # Count the piece we just placed
            
            # Count in positive direction
            r, c = row + dr, col + dc
            while (0 <= r < self.game.row_count and 
                   0 <= c < self.game.column_count and 
                   state[0, r, c] == player):
                count += 1
                r += dr
                c += dc
            
            # Count in negative direction
            r, c = row - dr, col - dc
            while (0 <= r < self.game.row_count and 
                   0 <= c < self.game.column_count and 
                   state[0, r, c] == player):
                count += 1
                r -= dr
                c -= dc
            
            # If we have 3 in a row, it's a threat
            if count >= 3:
                return True
        
        return False
