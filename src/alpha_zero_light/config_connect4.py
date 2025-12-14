"""
Configuration for Connect Four (4-in-a-row) AlphaZero training.
"""

from dataclasses import dataclass


# Training Configuration
TRAINING_CONFIG = {
    'num_iterations': 200,              # Optimized: 100 warmup + 100 self-play
    'num_self_play_iterations': 400,    # Games per iteration
    'num_parallel_workers': 2,          # CPU cores per batch (conservative for stability)
    'games_per_batch': 50,              # Games per parallel batch (8 batches of 50 = 400 total)
    'num_epochs': 100,                  # Increased from 50 - training is fast relative to self-play
    'batch_size': 512,                  # Batch size for neural network training
    'temperature_schedule': [           # Exploration schedule
        {'until_iteration': 40, 'temperature': 1.25},
        {'until_iteration': 100, 'temperature': 1.0},
        {'until_iteration': 160, 'temperature': 0.75}
    ],
    'value_loss_weight': 2.0,           # Weight value loss more heavily
    'random_opponent_iterations': 115,  # EXTENDED: 0-27 random, 28-54 heuristic, 55-84 mixed, 85-114 strong+mixed
    'eval_frequency': 10,               
    'num_eval_games': 20,               
    'dirichlet_epsilon': 0.25,          
    'dirichlet_alpha': 0.3,            
}

# MCTS Configuration
MCTS_CONFIG = {
    'C': 2.0,                           
    'num_searches': 100,                # More searches for better tactical play
    'dirichlet_alpha': 0.3,             # Exploration noise (lower = more concentrated)
    'dirichlet_epsilon': 0.25,          # Fraction of noise to add to root
}

# Model Configuration
MODEL_CONFIG = {
    'num_res_blocks': 10,               # Reduced from 15 for faster training with parallelization
    'num_hidden': 128,                  # Reduced from 256 for faster training
    'learning_rate': 0.001,             # Moderate initial LR
    'learning_rate_schedule': [         # DISABLED - let it train smoothly
        # {'at_iteration': 75, 'factor': 0.5},
        # {'at_iteration': 150, 'factor': 0.5}
    ],
    'weight_decay': 0.0001,             
}

# Paths Configuration
@dataclass
class PathConfig:
    checkpoints: str = "checkpoints/connect4"
    logs: str = "logs/connect4"
    plots: str = "docs/training_plots/connect4"

PATHS = PathConfig()
