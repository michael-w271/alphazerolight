import numpy as np
from .game import Game

class Gomoku(Game):
    def __init__(self):
        self.row_count = 15
        self.column_count = 15
        self.action_size = self.row_count * self.column_count
        self.win_length = 5

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))

    def get_next_state(self, state, action, player):
        row = action // self.column_count
        col = action % self.column_count
        state[row, col] = player
        return state

    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def check_win(self, state, action):
        if action is None:
            return False
            
        row = action // self.column_count
        col = action % self.column_count
        player = state[row, col]

        # Directions: horizontal, vertical, diagonal (down-right), diagonal (up-right)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            # Check positive direction
            for i in range(1, self.win_length):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < self.row_count and 0 <= c < self.column_count and state[r, c] == player:
                    count += 1
                else:
                    break
            
            # Check negative direction
            for i in range(1, self.win_length):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < self.row_count and 0 <= c < self.column_count and state[r, c] == player:
                    count += 1
                else:
                    break
            
            if count >= self.win_length:
                return True
                
        return False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        return encoded_state
