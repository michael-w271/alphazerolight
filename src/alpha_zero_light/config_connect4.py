"""
Configuration for Connect Four (4-in-a-row) AlphaZero training.
"""

from dataclasses import dataclass


# Training Configuration
TRAINING_CONFIG = {
    'num_iterations': 50,              # Total training iterations
    'num_self_play_iterations': 600,    # MANY more games
    'num_epochs': 30,                   # MANY more epochs = more GPU work
    'batch_size': 1024,                 # HUGE batches = max GPU saturation
    'temperature': 1.0,                 # Temperature for move exploration
    'eval_frequency': 5,                # Evaluate every N iterations
    'num_eval_games': 20,               # Number of evaluation games
    'dirichlet_epsilon': 0.25,          # Exploration noise strength
    'dirichlet_alpha': 0.3,             # Dirichlet noise parameter
}

# MCTS Configuration
MCTS_CONFIG = {
    'C': 2.0,                           # Exploration constant
    'num_searches': 16,                 # MINIMAL searches = less CPU, rely on network quality
}

# Model Configuration
MODEL_CONFIG = {
    'num_res_blocks': 6,                # Number of residual blocks
    'num_hidden': 64,                   # Hidden layer size
    'learning_rate': 0.001,             # Adam learning rate
    'weight_decay': 0.0001,             # L2 regularization
}

# Paths Configuration
@dataclass
class PathConfig:
    checkpoints: str = "checkpoints/connect4"
    logs: str = "logs/connect4"
    plots: str = "docs/training_plots/connect4"

PATHS = PathConfig()
