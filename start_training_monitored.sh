#!/bin/bash
# Launch enhanced training with monitoring in separate terminals

cd "$(dirname "$0")"

echo "========================================================================"
echo "🚀 ALPHAZERO CONNECT FOUR - ENHANCED TRAINING WITH MONITORING"
echo "========================================================================"
echo ""
echo "This will open TWO terminal windows:"
echo "  1. Training Progress - Shows losses, iteration progress"
echo "  2. Model Testing - Tests model capabilities after each iteration"
echo ""
echo "All results are saved and displayed live!"
echo ""
echo "Press Enter to start..."
read

# Source environment
source env_config.sh

# Launch training in new terminal
echo "📊 Opening training window..."
gnome-terminal --title="AlphaZero Training - Main" -- bash -c "
    cd '$(pwd)'
    source ~/miniforge3/etc/profile.d/conda.sh
    conda activate azl
    source env_config.sh
    \$PYTHON_EXEC scripts/train_connect4.py 2>&1 | tee training_log.txt
    exec bash
" &

sleep 2

# Launch testing in new terminal
echo "🧪 Opening testing window..."
gnome-terminal --title="AlphaZero Testing - Model Evaluation" -- bash -c "
    cd '$(pwd)'
    source ~/miniforge3/etc/profile.d/conda.sh
    conda activate azl
    source env_config.sh
    \$PYTHON_EXEC scripts/test_model_progress.py
    exec bash
" &

echo ""
echo "========================================================================"
echo "✅ TRAINING LAUNCHED!"
echo "========================================================================"
echo ""
echo "Monitor progress in the opened windows."
echo ""
echo "Results saved to:"
echo "  - training_log.txt (training output)"
echo "  - model_test_results.json (test results with history)"
echo ""
echo "To stop: pkill -f train_connect4"
echo ""
