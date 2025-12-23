"""
Tactical training scenarios for Connect Four.
Generates specific board positions to teach patterns like:
- Completing 4-in-a-row
- Blocking opponent threats
- Creating double threats
- Fork positions
"""

import numpy as np
from typing import List, Tuple


class TacticalTrainer:
    """Generate tactical training positions for Connect Four"""
    
    def __init__(self, game):
        self.game = game
        self.row_count = game.row_count
        self.column_count = game.column_count
        
    def create_win_in_one(self, player: int) -> Tuple[np.ndarray, int]:
        """
        Create a position where player can win in one move.
        Returns: (state, winning_action)
        """
        state = self.game.get_initial_state()
        
        # Pick a random column and build 3 pieces
        win_col = np.random.randint(0, self.column_count)
        
        # Stack 3 pieces for the player
        for _ in range(3):
            state = self.game.get_next_state(state, win_col, player)
            # Add opponent pieces randomly to other columns
            other_cols = [c for c in range(self.column_count) if c != win_col]
            if other_cols:
                opp_col = np.random.choice(other_cols)
                if self.game.get_valid_moves(state)[opp_col]:
                    state = self.game.get_next_state(state, opp_col, -player)
        
        return state, win_col
    
    def create_block_threat(self, player: int) -> Tuple[np.ndarray, int]:
        """
        Create a position where player must block opponent's win threat.
        Returns: (state, blocking_action)
        """
        state = self.game.get_initial_state()
        
        # Opponent has 3 in a column, player must block
        threat_col = np.random.randint(0, self.column_count)
        
        # Stack 3 pieces for opponent
        for _ in range(3):
            state = self.game.get_next_state(state, threat_col, -player)
            # Add player pieces randomly to other columns
            other_cols = [c for c in range(self.column_count) if c != threat_col]
            if other_cols:
                player_col = np.random.choice(other_cols)
                if self.game.get_valid_moves(state)[player_col]:
                    state = self.game.get_next_state(state, player_col, player)
        
        return state, threat_col
    
    def create_horizontal_threat(self, player: int) -> Tuple[np.ndarray, int]:
        """
        Create a position with 3 horizontal pieces and a gap.
        Player should fill the gap.
        """
        state = self.game.get_initial_state()
        
        # Build horizontal: X X _ X pattern
        start_col = np.random.randint(0, self.column_count - 3)
        gap_offset = np.random.randint(0, 3)  # Where the gap is (0, 1, or 2)
        
        # Place pieces in bottom row with a gap
        for offset in range(4):
            col = start_col + offset
            if offset == gap_offset:
                continue  # Leave gap
            state = self.game.get_next_state(state, col, player)
            
        gap_col = start_col + gap_offset
        return state, gap_col
    
    def create_double_threat(self, player: int) -> Tuple[np.ndarray, int]:
        """
        Create a position where player can create two threats at once (fork).
        This is an advanced tactic.
        """
        state = self.game.get_initial_state()
        
        # Build a setup where one move creates two winning threats
        # Example: vertical threat in col 3, horizontal setup in cols 2-5
        
        # Vertical threat setup (2 pieces in column 3)
        for _ in range(2):
            state = self.game.get_next_state(state, 3, player)
            
        # Horizontal pieces (cols 2, 4, 5 in same row)
        state = self.game.get_next_state(state, 2, player)
        state = self.game.get_next_state(state, 4, player)
        state = self.game.get_next_state(state, 5, player)
        
        # Playing col 3 creates both vertical and horizontal threats
        return state, 3
    
    def generate_tactical_game(self, mcts, player: int = None) -> List[Tuple]:
        """
        Play a game that starts from a tactical position.
        Mix of tactical puzzles and random positions.
        
        Args:
            player: Which player the model controls. If None, randomly choose 1 or -1.
        
        Returns: game memory (state, action_probs, outcome)
        
        CRITICAL: Randomly alternate which player the model controls so it learns
        both offensive and defensive play from both perspectives.
        """
        # Randomly choose which player the model controls (if not specified)
        if player is None:
            player = np.random.choice([1, -1])
        
        model_player = player  # Remember which player the model started as
        
        # Choose a random tactical scenario (70% chance) or start from empty (30%)
        scenario_type = np.random.choice([
            'win_in_one', 'block_threat', 'horizontal', 'empty'
        ], p=[0.25, 0.25, 0.20, 0.30])
        
        if scenario_type == 'win_in_one':
            state, hint_action = self.create_win_in_one(player)
        elif scenario_type == 'block_threat':
            state, hint_action = self.create_block_threat(player)
        elif scenario_type == 'horizontal':
            state, hint_action = self.create_horizontal_threat(player)
        else:
            state = self.game.get_initial_state()
            hint_action = None
        
        # Continue playing from this position
        game_memory = []
        move_count = 0
        max_moves = 42  # Maximum moves in Connect Four
        
        while move_count < max_moves:
            # Get MCTS policy
            neutral_state = self.game.change_perspective(state, player)
            action_probs = mcts.search(neutral_state)
            
            # Record this position
            game_memory.append((neutral_state, action_probs, player))
            
            # Sample action from policy
            temperature = 1.0 if move_count < 10 else 0.5
            temp_action_probs = action_probs ** (1 / temperature)
            temp_action_probs /= np.sum(temp_action_probs)
            action = np.random.choice(self.game.action_size, p=temp_action_probs)
            
            # Make move
            state = self.game.get_next_state(state, action, player)
            value, is_terminal = self.game.get_value_and_terminated(state, action)
            
            if is_terminal:
                # Assign outcomes to all moves in this game
                return_memory = []
                for hist_state, hist_probs, hist_player in game_memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    return_memory.append((
                        self.game.get_encoded_state(hist_state),
                        hist_probs,
                        hist_outcome
                    ))
                # Convert to initial model player's perspective
                model_outcome = value if model_player == player else self.game.get_opponent_value(value)
                return return_memory, model_outcome
            
            player = self.game.get_opponent(player)
            move_count += 1
        
        # Draw - no winner after max moves
        value = 0  # Draw
        return_memory = []
        for hist_state, hist_probs, hist_player in game_memory:
            return_memory.append((
                self.game.get_encoded_state(hist_state),
                hist_probs,
                0  # Draw
            ))
        return return_memory, value
