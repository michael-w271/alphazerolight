# AlphaZero-Light C++ Viewer

Real-time visualization for AlphaZero Connect Four training, inspired by Snake.cpp.

## Building

### Dependencies

Install system packages:
```bash
# Ubuntu/Debian
sudo apt install libsdl2-dev libzmq3-dev cmake g++

# Fedora
sudo dnf install SDL2-devel zeromq-devel cmake gcc-c++
```

Clone header-only libraries:
```bash
cd cpp_viewer
git clone https://github.com/ocornut/imgui.git deps/imgui
git clone https://github.com/epezent/implot.git deps/implot
git clone https://github.com/nlohmann/json.git deps/json
```

### Compile

```bash
cd cpp_viewer
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Running

Start the Python telemetry test:
```bash
python -m alpha_zero_light.visualization.test_telemetry
```

In another terminal, run the viewer:
```bash
./build/azl_viewer --telemetry tcp://127.0.0.1:5556
```

Or start training with live visualization:
```bash
# Terminal 1: Start viewer first
./build/azl_viewer --telemetry tcp://127.0.0.1:5556

# Terminal 2: Start training
python -m alpha_zero_light.training.trainer_demo \
    --telemetry tcp://127.0.0.1:5556 \
    --device cuda \
    --demo \
    --iterations 50
```

## Controls

- **P**: Pause/Resume rendering (training continues)
- **Space**: Force render next frame
- **S**: Save episode trace
- **ESC**: Quit

## Architecture

- **Main Window**: Connect Four 6x7 board with live game state
- **Thinking Window**: MCTS visit counts, policy comparison, Q-values
- **Metrics Window**: Live loss curves and training stats
