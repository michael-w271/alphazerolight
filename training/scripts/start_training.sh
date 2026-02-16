#!/bin/bash
# Final launcher with comprehensive monitors

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_SCRIPT="$REPO_ROOT/training/scripts/train_connect4.py"
LOG_DIR="$REPO_ROOT/artifacts/logs/training"
LOG_FILE="$LOG_DIR/training_log_v2.txt"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

echo "🚀 Launching Connect4 Maximum Strength Training"
echo "=============================================="
echo ""

# Kill existing
pkill -f "train_connect4.py" 2>/dev/null
sleep 1

# Start training
echo "Starting training..."
nohup /mnt/ssd2pro/miniforge3/envs/tetrisrl/bin/python "$TRAIN_SCRIPT" > "$LOG_FILE" 2>&1 &
echo "✅ Training started (PID: $!)"
sleep 2

# Launch comprehensive training monitor
echo "Opening training monitor..."
gnome-terminal --title="🎮 Training Monitor - Full Info" \
    --geometry=140x45 \
    --working-directory="$REPO_ROOT" \
    -- bash "$REPO_ROOT/training/monitors/monitor_full.sh" &

sleep 1

# Launch enhanced evaluation monitor  
echo "Opening evaluation monitor..."
gnome-terminal --title="🎯 Model Evaluation - Comprehensive" \
    --geometry=130x50 \
    --working-directory="$REPO_ROOT" \
    -- bash "$REPO_ROOT/training/monitors/monitor_eval.sh" &

echo ""
echo "✅ Monitoring launched!"
echo ""
echo "Windows:"
echo "  1. 🎮 Training Monitor - Full details with clean progress"
echo "  2. 🎯 Evaluation Monitor - Tactical tests every 10 iterations"
echo ""
