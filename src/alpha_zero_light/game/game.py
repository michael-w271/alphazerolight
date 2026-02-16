from abc import ABC, abstractmethod
import numpy as np

class Game(ABC):
    """
    Abstract Base Class for a game definition in the AlphaZero framework.
    """

    @abstractmethod
    def get_initial_state(self):
        """
        Returns the initial state of the game.
        """
        pass

    @abstractmethod
    def get_next_state(self, state, action, player):
        """
        Returns the next state given the current state, action, and player.
        """
        pass

    @abstractmethod
    def get_valid_moves(self, state):
        """
        Returns a binary mask of valid moves for the current state.
        """
        pass

    @abstractmethod
    def get_value_and_terminated(self, state, action):
        """
        Returns the value of the state (1 for win, 0 for draw, -1 for loss) 
        and whether the game has terminated.
        """
        pass

    @abstractmethod
    def get_opponent(self, player):
        """
        Returns the opponent of the current player.
        """
        pass
    
    @abstractmethod
    def get_opponent_value(self, value):
        """
        Returns the value from the opponent's perspective.
        """
        pass

    @abstractmethod
    def change_perspective(self, state, player):
        """
        Returns the state from the perspective of the given player.
        """
        pass

    @abstractmethod
    def get_encoded_state(self, state):
        """
        Returns the encoded state for the neural network.
        """
        pass
