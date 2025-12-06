"""
Configuration for long Gomoku 9x9 training (overnight)
Based on working 30min config
"""

import torch

# Training Configuration
TRAINING_CONFIG = {
    'num_iterations': 250,           # Long run - can stop anytime
    'num_self_play_iterations': 512, # More games for better data
    'num_epochs': 8,                 # More training epochs
    'batch_size': 256,
    'temperature': 1.5,              # More exploration
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3,
    'eval_frequency': 10,
    'num_eval_games': 20,
}

# MCTS Configuration
MCTS_CONFIG = {
    'num_searches': 100,             # MUCH more searches! (4x)
    'C': 2,
    'num_parallel_games': 512,       # Match batch size
    'num_sampling_moves': 30,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3,
}

# Model Configuration
MODEL_CONFIG = {
    'num_res_blocks': 12,            # Medium size (was 15)
    'num_hidden': 384,               # Medium capacity (was 512)
    'learning_rate': 0.002,
    'weight_decay': 0.0001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# Paths
PATHS = {
    'checkpoints': 'checkpoints/gomoku_medium',  # Medium model
    'logs': 'logs/gomoku_medium',
}
