# AlphaZero Light - Connect Four

A lightweight implementation of AlphaZero for Connect Four (4-in-a-row), featuring self-play training, MCTS search, and comprehensive evaluation tools.

## 🎯 Quick Start

### Start Training
```bash
cd /mnt/ssd2pro/alpha-zero-light
bash training/scripts/start_training.sh
```

This will:
- Launch training with comprehensive monitoring
- Open terminal windows showing live progress
- Run evaluations every 10 iterations

### Play Against the AI
```bash
python play_connect4.py
```

### Run Tests
```bash
# Unit tests
python tests/unit/test_models.py

# MCTS tests
python tests/mcts/test_mcts_blocking.py

# Model tournaments
python tests/integration/custom_tournament.py
```

## 📁 Repository Structure

```
alpha-zero-light/
├── training/               # All training-related files
│   ├── scripts/           # Training entry points
│   │   ├── start_training.sh          # Main training launcher
│   │   ├── train_connect4.py          # Core training script
│   │   └── ...
│   ├── configs/           # Training configurations
│   │   └── training_config_v2.json
│   ├── monitors/          # Monitoring scripts
│   │   ├── monitor_full.sh
│   │   └── monitor_eval.sh
│   └── utils/             # Training utilities
│       ├── pause_training.sh
│       └── resume_training.sh
│
├── tests/                 # All test files
│   ├── unit/             # Unit tests
│   ├── mcts/             # MCTS-specific tests
│   ├── integration/      # Integration tests & tournaments
│   ├── validation/       # Validation tests
│   ├── debug/            # Debug scripts
│   └── analysis/         # Analysis & comparison tools
│
├── apps/                  # Web applications
│   ├── connect4-lab/     # React Connect4 interface
│   ├── website/          # Main website
│   └── cpp_viewer/       # C++ visualization app
│
├── docs/                  # Documentation
│   ├── research/         # Research notes and investigations
│   ├── bugs/             # Bug reports and fixes
│   ├── migration/        # Migration documentation
│   ├── training/         # Training documentation
│   └── project/          # Project structure docs
│
├── src/                   # Core library
│   └── alpha_zero_light/
│       ├── game/         # Game implementations
│       ├── mcts/         # MCTS implementation
│       ├── model/        # Neural network models
│       └── training/     # Training infrastructure
│
├── checkpoints/           # Model checkpoints
├── experiments/           # Experiments and analysis
└── paper_materials/       # Research paper materials
```

## 🚀 Training

### Main Training Script
The main training entry point is `training/scripts/start_training.sh`, which:
- Starts training with `train_connect4.py`
- Launches monitoring terminals
- Runs periodic evaluations

### Monitoring
Two monitoring windows are automatically opened:
1. **Training Monitor** (`training/monitors/monitor_full.sh`) - Real-time training progress
2. **Evaluation Monitor** (`training/monitors/monitor_eval.sh`) - Model evaluation every 10 iterations

### Configuration
Training configuration is in `training/configs/training_config_v2.json`:
- MCTS searches: Progressive curriculum (50 → 400)
- Training epochs: Progressive (60 → 120)
- Batch size: 1024
- Evaluation frequency: Every 10 iterations

### Utilities
- **Pause training**: `bash training/utils/pause_training.sh`
- **Resume training**: `bash training/utils/resume_training.sh`
- **Clean checkpoints**: `bash training/utils/clean_checkpoints.sh`
- **Fresh restart**: `bash training/utils/restart_training_fresh.sh`

## 🧪 Testing

### Unit Tests
Located in `tests/unit/`:
- `test_models.py` - Model architecture tests
- `test_model_output.py` - Model output validation
- `test_model_signs.py` - Value sign correctness

### MCTS Tests
Located in `tests/mcts/`:
- `test_mcts_blocking.py` - Threat detection
- `test_mcts_tree_depth.py` - Search depth analysis
- `test_iteration_tactical.py` - Tactical scenarios

### Integration Tests
Located in `tests/integration/`:
- `model_tournament.py` - Run tournaments between models
- `custom_tournament.py` - Custom tournament configurations
- `test_models_auto.py` - Automated model testing

### Analysis Tools
Located in `tests/analysis/`:
- `compare_models.py` - Compare model performance
- `analyze_training_data.py` - Analyze training metrics
- `trace_all_signs.py` - Debug value predictions

## 🌐 Web Applications

### Connect4 Lab (React)
Interactive Connect4 interface with AI opponent:
```bash
cd apps/connect4-lab
bash setup.sh
# Then open index.html in browser
```

### Website
Main project website:
```bash
cd apps/website
python -m http.server 8000
# Visit http://localhost:8000
```

### C++ Viewer
Real-time visualization of training:
```bash
cd apps/cpp_viewer
mkdir build && cd build
cmake ..
make
./connect4_viewer
```

## 📚 Documentation

### Training Documentation
- [Training Quickstart](docs/training/TRAINING_QUICKSTART.md) - Get started quickly
- [Training Documentation](docs/training/TRAINING_DOCUMENTATION.md) - Comprehensive guide
- [Training Controls](docs/training/TRAINING_CONTROLS.md) - Control and monitor training
- [Maximum Training Config](docs/training/MAXIMUM_TRAINING_CONFIG.md) - Advanced configuration

### Bug Reports
- [Critical Bug Fixes](docs/bugs/CRITICAL_BUG_FIXED.md) - Major bugs resolved
- [MCTS Terminal Bug](docs/bugs/CRITICAL_MCTS_TERMINAL_BUG_FIXED.md) - MCTS fixes
- [Encoding Bug](docs/bugs/ENCODING_BUG_FIXED.md) - Board encoding fixes

### Project Documentation
- [Project Structure](docs/project/PROJECT_STRUCTURE.md) - Codebase organization
- [Research Notes](docs/research/) - Investigation reports and findings

## 🔧 Development

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Environment Configuration
The `env_config.sh` file contains environment-specific paths:
```bash
source env_config.sh
```

## 📊 Checkpoints

Model checkpoints are saved in `checkpoints/connect4/`:
- `model_*.pt` - Model weights
- `optimizer_*.pt` - Optimizer state
- `training_history.json` - Training metrics

## 🎮 Interactive Play

Play against the trained AI:
```bash
python play_connect4.py [--model checkpoints/connect4/model_N.pt]
```

## 📈 Progress Tracking

Training progress is logged to:
- `training_log_v2.txt` - Main training log
- `checkpoints/connect4/training_history.json` - Metrics history
- Terminal monitors (launched automatically)

## 🤝 Contributing

This is a research project. See documentation in `docs/` for more details on the implementation and findings.

## 📄 License

See project documentation for license information.
