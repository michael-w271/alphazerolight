# AlphaZero Connect Four - Project Organization

## Directory Structure

```
alpha-zero-light/
├── experiments/           # Training experiments and evaluation
│   ├── connect4_v2/      # Current experiment data
│   ├── evaluate_training.py  # Comprehensive metrics evaluation
│   └── monitor_training.py   # Real-time training monitor
│
├── paper_materials/       # Scientific documentation
│   ├── data/             # Results JSON files
│   ├── figures/          # Plots and visualizations  
│   ├── tables/           # LaTeX tables
│   └── RESULTS_TEMPLATE.md  # Paper template
│
├── src/                  # Core AlphaZero implementation
│   └── alpha_zero_light/
│       ├── game/         # Game implementations
│       ├── model/        # Neural network architecture
│       ├── mcts/         # MCTS implementation
│       └── training/     # Training loop
│
├── scripts/              # Training and utility scripts
│   ├── train_connect4.py
│   └── quick_eval_puzzles.py
│
├── connect4-lab/         # Web UI for playing
│   ├── index.html       # Standalone game interface
│   └── api/server.py    # Flask API
│
├── /mnt/ssd2pro/alpha-zero-checkpoints/
│   ├── connect4/        # Original training (model_120 champion)
│   └── connect4_v2/     # New training (current)
│
└── Root files:
    ├── model_tournament.py      # Model comparison
    ├── play_connect4.py         # Terminal play
    ├── TRAINING_DOCUMENTATION.md # Technical documentation
    └── TRAINING_QUICKSTART.md    # Quick start guide
```

## Key Scripts

### Training
- `start_training_monitored.sh` - Launch training with monitoring
- `scripts/train_connect4.py` - Core training script

### Evaluation
- `experiments/evaluate_training.py` - Comprehensive metrics
- `experiments/monitor_training.py` - Real-time monitoring
- `model_tournament.py` - Head-to-head comparisons

### Playing
- `connect4-lab/index.html` - Web UI
- `play_connect4.py` - Terminal interface

## Workflow

### Starting Training
```bash
./start_training_monitored.sh
```

### Monitoring Progress
```bash
tmux attach -t connect4_training  # Window 0: training, Window 1: monitor
```

### Evaluating Models
```bash
# Single model
python experiments/evaluate_training.py --model path/to/model.pt

# Full run
python experiments/evaluate_training.py --start 0 --end 150 --step 10
```

### Generating Paper Materials
Results automatically saved to `paper_materials/`:
- `data/*.json` - Raw evaluation results
- `tables/*.tex` - LaTeX tables
- `figures/*.png` - Plots (when generated)

### Tournament Comparison
```bash
python model_tournament.py  # Edit to include v2 models
```

## Important Files

- **Configuration**: `src/alpha_zero_light/config_connect4.py`
- **Training History**: `/mnt/ssd2pro/alpha-zero-checkpoints/connect4_v2/training_history.json`
- **Checkpoints**: `/mnt/ssd2pro/alpha-zero-checkpoints/connect4_v2/model_*.pt`
- **Paper Template**: `paper_materials/RESULTS_TEMPLATE.md`
