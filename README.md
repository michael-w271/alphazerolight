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
This will use the `azl` virtual environment and the settings in `src/alpha_zero_light/config.py`.

### Play
To play against the trained AI:
```bash
./run_app.sh
```

### Configuration
You can adjust training parameters in `src/alpha_zero_light/config.py`.
The environment configuration is stored in `env_config.sh`.

## Live Training Viewer (New!)
A real-time C++ visualization tool that displays AlphaZero training as it happens, inspired by Snake.cpp:

### Features
- **Game State Visualization**: Watch Connect Four games with colored pieces (red/yellow) and last move highlighting
- **Neural Network Activity**: See how the network "thinks" with policy head, value head, and MCTS visualization
- **Training Metrics**: Real-time stats on games played, win rates, and model performance
- **Multi-Window Interface**: Separate panes for game, network activity, and thinking process

### Running the Viewer
1. Build the C++ viewer:
   ```bash
   cd cpp_viewer
   mkdir -p build && cd build
   cmake ..
   make
   ```
2. Start training (which launches the telemetry server):
   ```bash
   python scripts/train_connect4.py
   ```
3. Run the viewer in a separate terminal:
   ```bash
   ./cpp_viewer/build/azl_viewer
   ```

### Recent Updates (Dec 2025)
- Added Snake.cpp-style live training visualization with SDL2/ImGui
- Implemented colored Connect Four piece rendering with move highlights
- Created neural network activation visualization (policy/value heads + MCTS thinking)
- Optimized training: reduced MCTS searches to 50 for faster iteration
- Fixed telemetry streaming with ZeroMQ for reliable Python→C++ communication

## Website Showcase
To view the project showcase website locally:
```bash
python scripts/serve_website.py
```
Then open `http://localhost:8000` in your browser.
