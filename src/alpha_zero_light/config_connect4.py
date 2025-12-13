"""
Configuration for Connect Four (4-in-a-row) AlphaZero training.
"""

from dataclasses import dataclass


# Training Configuration
TRAINING_CONFIG = {
    'num_iterations': 190,              # ~8 hours: 30 warmup + 160 self-play
    'num_self_play_iterations': 600,    # Balance zwischen Qualität und Speed
    'num_epochs': 50,                   # Mehr Training pro Iteration
    'batch_size': 512,                  # Kleinere Batches für bessere Gradienten
    'temperature_schedule': [           # Längere Exploration
        {'until_iteration': 50, 'temperature': 1.25},
        {'until_iteration': 75, 'temperature': 1.0},
        {'until_iteration': 100, 'temperature': 0.75}
    ],
    'value_loss_weight': 2.0,           # Weight value loss more heavily
    'random_opponent_iterations': 30,   # Progressive difficulty: 0-9 random, 10-19 heuristic, 20-29 mixed
    'eval_frequency': 10,               
    'num_eval_games': 20,               
    'dirichlet_epsilon': 0.25,          
    'dirichlet_alpha': 0.3,            
}

# MCTS Configuration
MCTS_CONFIG = {
    'C': 2.0,                           
    'num_searches': 50,                 # Kompromiss: besser als 32, schneller als 100
    'dirichlet_alpha': 0.3,             # Exploration noise (lower = more concentrated)
    'dirichlet_epsilon': 0.25,          # Fraction of noise to add to root
}

# Model Configuration
MODEL_CONFIG = {
    'num_res_blocks': 10,               # Bigger model for Connect Four
    'num_hidden': 128,                  # Wider network
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
