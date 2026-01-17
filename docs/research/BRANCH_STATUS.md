# Branch 4inarowReact - Clean Structure

## Active Files (Organized)

### Training & Monitoring
```
launch_training_with_monitors.sh  # Main launcher (opens 2 terminal windows)
monitor_training.sh               # Quick status check
check_training.sh                 # Training status
```

### Experiments & Evaluation
```
experiments/
├── evaluate_training.py          # Scientific metrics evaluation
└── monitor_training.py           # Python monitor with auto-eval
```

### Connect4 Lab (Web UI)
```
connect4-lab/
├── index.html                    # Standalone game
├── api/server.py                 # Flask API
└── README.md                     # Usage guide
```

### Documentation
```
TRAINING_DOCUMENTATION.md         # Technical training docs
TRAINING_QUICKSTART.md           # Quick reference
PROJECT_STRUCTURE.md              # Organization guide
paper_materials/RESULTS_TEMPLATE.md  # Scientific paper template
```

### Key Scripts
```
model_tournament.py               # Model comparison
play_connect4.py                  # Terminal play interface
scripts/train_connect4.py         # Core training script
```

## Usage

### Start Training with Monitors
```bash
./launch_training_with_monitors.sh
```

Opens 2 terminal windows:
1. **Training Monitor** - Live log updates (5s refresh)
2. **Evaluation Monitor** - Runs tests every 10 iterations

### Check Progress
```bash
./monitor_training.sh             # Quick snapshot
tail -f training_log_v2.txt       # Live log
```

### Evaluate Models
```bash
python experiments/evaluate_training.py --model path/to/model.pt
```

## Cleaned Up
Removed duplicate/obsolete files:
- monitor_simple.sh
- start_training_v2.sh  
- start_training_monitored.sh
- setup.sh
