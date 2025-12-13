import numpy as np
from alpha_zero_light.game.game import Game


class ConnectFour(Game):
    """
    Connect Four (Four in a Row) game implementation.
    
    Board: 6 rows x 7 columns
    Win condition: 4 in a row (horizontal, vertical, or diagonal)
    Actions: Column index (0-6) - disc drops to lowest empty row
    """
    
    def __init__(self, row_count=6, column_count=7, win_length=4):
        self.row_count = row_count
        self.column_count = column_count
        self.win_length = win_length
        self.action_size = column_count
        
    def __repr__(self):
        return f"ConnectFour({self.row_count}x{self.column_count}, win={self.win_length})"
    
    def get_initial_state(self, batch_size=None):
        """
        Returns initial empty board state.
        
        Args:
            batch_size: If provided, returns batched states (batch_size, rows, cols)
        
        Returns:
            numpy array of zeros with shape (rows, cols) or (batch_size, rows, cols)
        """
        if batch_size is None:
            return np.zeros((self.row_count, self.column_count), dtype=np.float32)
        else:
            return np.zeros((batch_size, self.row_count, self.column_count), dtype=np.float32)
    
    def get_next_state(self, state, action, player):
        """
        Apply gravity-based move: drop disc in column to lowest empty row.
        
        Args:
            state: Current board state (rows, cols)
            action: Column index (0 to column_count-1)
            player: 1 or -1
        
        Returns:
            New state with disc placed
        """
        state = state.copy()
        
        # Find the lowest empty row in the chosen column
        column = action
        for row in range(self.row_count - 1, -1, -1):
            if state[row, column] == 0:
                state[row, column] = player
                return state
        
        # Column is full - this should not happen with proper valid_moves checking
        raise ValueError(f"Column {column} is full")
    
    def get_valid_moves(self, state):
        """
        Returns mask of valid column choices.
        
        A column is valid if its top row is empty.
        
        Args:
            state: Current board state (rows, cols) or (batch, rows, cols)
        
        Returns:
            Binary mask of length action_size (column_count)
        """
        if state.ndim == 2:
            # Single state: check top row
            return (state[0, :] == 0).astype(np.uint8)
        else:
            # Batched states
            return (state[:, 0, :] == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        """
        Check if the last move (placing in column=action) resulted in a win.
        
        Args:
            state: Current board state
            action: Column index of last move
        
        Returns:
            True if 4 in a row achieved, False otherwise
        """
        if action is None or action < 0 or action >= self.column_count:
            return False
        
        # Find the row where the last disc was placed (topmost disc in column)
        column = action
        row = -1
        for r in range(self.row_count):
            if state[r, column] != 0:
                row = r
                break
        
        if row == -1:
            return False
        
        player = state[row, column]
        
        # Check horizontal
        count = 1
        # Check left
        c = column - 1
        while c >= 0 and state[row, c] == player:
            count += 1
            c -= 1
        # Check right
        c = column + 1
        while c < self.column_count and state[row, c] == player:
            count += 1
            c += 1
        if count >= self.win_length:
            return True
        
        # Check vertical (only need to check downward)
        count = 1
        r = row + 1
        while r < self.row_count and state[r, column] == player:
            count += 1
            r += 1
        if count >= self.win_length:
            return True
        
        # Check diagonal (top-left to bottom-right)
        count = 1
        # Check up-left
        r, c = row - 1, column - 1
        while r >= 0 and c >= 0 and state[r, c] == player:
            count += 1
            r -= 1
            c -= 1
        # Check down-right
        r, c = row + 1, column + 1
        while r < self.row_count and c < self.column_count and state[r, c] == player:
            count += 1
            r += 1
            c += 1
        if count >= self.win_length:
            return True
        
        # Check anti-diagonal (top-right to bottom-left)
        count = 1
        # Check up-right
        r, c = row - 1, column + 1
        while r >= 0 and c < self.column_count and state[r, c] == player:
            count += 1
            r -= 1
            c += 1
        # Check down-left
        r, c = row + 1, column - 1
        while r < self.row_count and c >= 0 and state[r, c] == player:
            count += 1
            r += 1
            c -= 1
        if count >= self.win_length:
            return True
        
        return False
    
    def get_value_and_terminated(self, state, action):
        """
        Returns game outcome from current player's perspective.
        
        Returns:
            (value, terminated) where:
            - value: 1 if current player won, 0 otherwise
            - terminated: True if game is over (win or draw)
        """
        if self.check_win(state, action):
            return 1, True
        
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        
        return 0, False
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        """Flip the board to opponent's perspective."""
        return state * player
    
    def get_encoded_state(self, state):
        """
        Encode state as 3-channel representation:
        - Channel 0: Opponent pieces (-1 values)
        - Channel 1: Empty cells (0 values)
        - Channel 2: Current player pieces (1 values)
        
        Args:
            state: Board state (rows, cols) from current player's perspective
        
        Returns:
            Encoded state (3, rows, cols)
        """
        encoded_state = np.stack([
            state == -1,  # Opponent pieces
            state == 0,   # Empty cells
            state == 1    # Current player pieces
        ]).astype(np.float32)
        
        if len(state.shape) == 3:
            # Batched input: (batch, rows, cols) -> (batch, 3, rows, cols)
            encoded_state = np.stack([
                state == -1,
                state == 0,
                state == 1
            ], axis=1).astype(np.float32)
        
        return encoded_state
