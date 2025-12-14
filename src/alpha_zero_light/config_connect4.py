"""
Configuration for Connect Four (4-in-a-row) AlphaZero training.
"""

from dataclasses import dataclass


# Training Configuration
TRAINING_CONFIG = {
    'num_iterations': 200,              # Extended: 120 warmup + 80 self-play
    'num_self_play_iterations': 400,    # Fewer games per iteration for faster feedback
    'num_epochs': 50,                   # Training epochs per iteration
    'batch_size': 512,                  # Batch size for neural network training
    'temperature_schedule': [           # Longer exploration phase
        {'until_iteration': 60, 'temperature': 1.25},
        {'until_iteration': 120, 'temperature': 1.0},
        {'until_iteration': 160, 'temperature': 0.75}
    ],
    'value_loss_weight': 2.0,           # Weight value loss more heavily
    'random_opponent_iterations': 120,  # EXTENDED WARMUP: 0-39 random, 40-79 heuristic, 80-119 mixed+tactical
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
    'num_res_blocks': 15,               # Larger model for better pattern recognition
    'num_hidden': 256,                  # Wider network for Connect Four complexity
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
