#!/bin/bash
# Start Gomoku 9x9 Training with Progress Tracking

echo "🚀 Starting Gomoku 9x9 Overnight Training"
echo "=========================================="
echo ""

# Source environment
source ./env_config.sh

# Check if already running
if pgrep -f "train_gomoku_9x9_overnight.py" > /dev/null; then
    echo "⚠️  Training is already running!"
    echo "   To stop it first: pkill -f train_gomoku_9x9_overnight.py"
    exit 1
fi

# Start training
echo "▶️  Starting training..."
echo "   - 100 iterations"
echo "   - 200 games per iteration"
echo "   - Batch size: 256"
echo "   - MCTS searches: 200"
echo ""
echo "📊 You will see:"
echo "   - Overall progress bar with ETA"
echo "   - Iteration time tracking"
echo "   - Updated time estimates after each iteration"
echo ""

$PYTHON_EXEC scripts/train_gomoku_9x9_overnight.py
