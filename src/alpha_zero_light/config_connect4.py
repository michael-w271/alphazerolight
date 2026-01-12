"""
Configuration for Connect Four (4-in-a-row) AlphaZero training.
"""

from dataclasses import dataclass


# Training Configuration - PROGRESSIVE CURRICULUM
TRAINING_CONFIG = {
    'num_iterations': 350,              # Extended long run
    'num_self_play_iterations': 150,    # Games per iteration
    'num_parallel_workers': 6,          # INCREASED: More parallel games for speed
    'games_per_batch': 25,              # Games per parallel batch
    'num_epochs': 90,                   # Epochs per iteration
    'batch_size': 1024,                 # DOUBLED: Better GPU utilization
    'temperature_schedule': [           # Exploration schedule
        {'until_iteration': 80, 'temperature': 1.25},
        {'until_iteration': 180, 'temperature': 1.0},
        {'until_iteration': 350, 'temperature': 0.75}
    ],
    # PROGRESSIVE MCTS SCHEDULE - Smart curriculum!
    'mcts_schedule': [
        {'until_iteration': 25, 'num_searches': 50},   # Weak model: quick searches
        {'until_iteration': 50, 'num_searches': 100},  # Learning: moderate depth
        {'until_iteration': 75, 'num_searches': 200},  # Improving: deeper search
        {'until_iteration': 100, 'num_searches': 300}, # Strong: tournament depth
        {'until_iteration': 350, 'num_searches': 400}, # Maximum: unbeatable
    ],
    # PROGRESSIVE EPOCHS SCHEDULE - More training as data quality improves
    'epochs_schedule': [
        {'until_iteration': 50, 'num_epochs': 60},     # Fast early learning
        {'until_iteration': 150, 'num_epochs': 90},    # Standard training
        {'until_iteration': 350, 'num_epochs': 120},   # Deep training for high-quality data
    ],
    'value_loss_weight': 1.0,           # Balanced value loss
    'random_opponent_iterations': 10,   # Bootstrap phase: 0-9
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
    'num_searches': 400,                # MAXIMUM: 400 searches for tournament-level play
    'dirichlet_alpha': 0.3,             # Exploration noise (lower = more concentrated)
    'dirichlet_epsilon': 0.25,          # Fraction of noise to add to root
    'mcts_batch_size': 1,               # Keep at 1 for tactical depth
}

# Model Configuration - MAXIMUM STRENGTH
MODEL_CONFIG = {
    'num_res_blocks': 20,               # DOUBLED: 20 residual blocks (much larger model)
    'num_hidden': 256,                  # DOUBLED: 256 hidden units (~4x parameters)
    'learning_rate': 0.001,             # Starting learning rate
    'learning_rate_schedule': [         # LR decay for better convergence
        {'until_iteration': 100, 'lr': 0.001},
        {'until_iteration': 200, 'lr': 0.0005},
        {'until_iteration': 300, 'lr': 0.0001},
    ],
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
            'self_play': 0.90,   # REDUCED: 90% self-play (more opponent diversity!)
            'random': 0.0,
            'heuristic': 0.0,
            'aggressive': 0.05,  # INCREASED: 5% adversarial play
            'strong': 0.03,      # INCREASED: 3% strong opponent (2-ply lookahead)
            'tactical': 0.02     # ADDED: 2% tactical puzzles opponent
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
