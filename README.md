# AlphaZero Light

A compact, scalable implementation of the AlphaZero algorithm, starting from Tic-Tac-Toe and progressing to Chess.

## Goals
- Clean, modular codebase.
- Scalable MCTS with Neural Network integration.
- Comprehensive documentation.
- Interactive UI with Streamlit.

## Setup
1. Activate the virtual environment:
   ```bash
   conda activate azl
   ```
2. Install dependencies (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```
## Quick Start

### Training
To train the AlphaZero agent with the configured environment:
```bash
./run_training.sh
```
This will use the `azl` virtual environment and the settings in `alpha_zero_light/config_connect4.py`.

### Play
To play against the trained AI:
```bash
./run_app.sh
```

### Configuration
You can adjust training parameters in `alpha_zero_light/config_connect4.py`.
The environment configuration is stored in `env_config.sh`.

### Checkpoints
Trained models and checkpoints are stored in `/mnt/ssd2pro/alpha-zero-checkpoints/connect4/`.
This external location keeps the repository clean while preserving trained models for local use.

## Website Showcase
To view the project showcase website locally:
```bash
python scripts/serve_website.py
```
Then open `http://localhost:8000` in your browser.
