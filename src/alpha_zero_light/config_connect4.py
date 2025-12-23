"""
Configuration for Connect Four (4-in-a-row) AlphaZero training.
"""

from dataclasses import dataclass


# Training Configuration
TRAINING_CONFIG = {
    'num_iterations': 350,              # Extended long run
    'num_self_play_iterations': 150,    # Games per iteration (increased from 100)
    'num_parallel_workers': 2,          # CPU cores per batch (conservative for stability)
    'games_per_batch': 25,              # Games per parallel batch
    'num_epochs': 90,                   # Epochs per iteration (increased from 50)
    'batch_size': 512,                  # Batch size for neural network training
    'temperature_schedule': [           # Exploration schedule
        {'until_iteration': 80, 'temperature': 1.25},
        {'until_iteration': 180, 'temperature': 1.0},
        {'until_iteration': 350, 'temperature': 0.75}
    ],
    'value_loss_weight': 1.0,           # Balanced value loss
    'random_opponent_iterations': 10,   # Bootstrap phase: 0-9 (updated to match OPPONENT_MIX)
    'eval_frequency': 10,               
    'num_eval_games': 30,               
    'dirichlet_epsilon': 0.25,          
    'dirichlet_alpha': 0.3,
    
    # Auto-compare configuration (runs evaluation every N iterations)
    'auto_compare_enabled': False,       # DISABLED - Manual evaluation only
    'auto_compare_interval': 20,         # Compare every 20 iterations
    'auto_compare_lookback': 20,         # Compare current vs 20 iterations ago
    'auto_compare_mcts_searches': 50,    # MCTS searches for comparison (lightweight)
    'auto_compare_device': 'cpu',        # Use CPU to avoid GPU contention
    'auto_compare_script': 'compare_models.py',
    'auto_compare_open_terminal': False, # Don't pop terminal (logs to file)
    'auto_compare_pause_training': False, # Run eval in background (non-blocking)
    'auto_compare_log_dir': 'logs/evaluations',
    'auto_compare_eval_plan': {
        'vs_random': {
            'enabled': True,
            'games': 10,                 # 10 games vs random
            'only_as_player1': True      # Only as P1 (faster)
        },
        'head_to_head': {
            'enabled': True,
            'games_as_p1': 10,           # 10 games with new model as P1
            'games_as_p2': 10            # 10 games with new model as P2
        },
        'total_games': 30                # Total: 10 + 10 + 10 = 30
    },
    
    # Quick puzzle eval configuration (faster tactical assessment)
    'quick_eval_enabled': False,         # DISABLED - Manual evaluation only
    'quick_eval_interval': 20,           # Run every 20 iterations
    'quick_eval_lookback': 20,           # Compare current vs 20 iterations ago
    'quick_eval_mcts_searches': 100,     # MCTS searches for puzzles
    'quick_eval_h2h_games_each_side': 5, # 5 games per side (10 total H2H)
    'quick_eval_num_puzzles': 25,        # 25 harder tactical puzzles (2-3 move lookahead)
}

# MCTS Configuration
MCTS_CONFIG = {
    'C': 2.0,                           
    'num_searches': 50,                 # Reduced to 50 for faster training/visualization
    'dirichlet_alpha': 0.3,             # Exploration noise (lower = more concentrated)
    'dirichlet_epsilon': 0.25,          # Fraction of noise to add to root
    'mcts_batch_size': 1,
}

# Model Configuration
MODEL_CONFIG = {
    'num_res_blocks': 10,               # 10 residual blocks
    'num_hidden': 128,                  # 128 hidden units
    'learning_rate': 0.001,             # Standard learning rate
    'learning_rate_schedule': [],       # No LR schedule
    'weight_decay': 0.0001,             
}

# Opponent Mix Configuration - Probabilistic opponent sampling
# UPDATED: Ultra-low heuristic mix per bug report recommendations
OPPONENT_MIX = {
    'bootstrap': {
        'iterations_inclusive': [0, 9],  # Shortened to 10 iterations
        'probabilities': {
            'self_play': 0.92,   # Increased self-play dominance
            'random': 0.08,      # Pure random for basic exploration
            'heuristic': 0.0,
            'aggressive': 0.0,
            'strong': 0.0,
            'tactical': 0.0
        }
    },
    'main': {
        'iterations_inclusive': [10, 349],
        'probabilities': {
            'self_play': 0.985,  # 98.5% self-play
            'random': 0.0,
            'heuristic': 0.0,
            'aggressive': 0.01,  # Tiny adversarial diversity
            'strong': 0.005,     # Minimal strong opponent
            'tactical': 0.0
        }
    }
}

# Paths Configuration
@dataclass
class PathConfig:
    checkpoints: str = "checkpoints/connect4"
    logs: str = "logs/connect4"
    plots: str = "docs/training_plots/connect4"

PATHS = PathConfig()
