#!/bin/bash
# Launch training with terminal popups for monitoring

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_SCRIPT="$REPO_ROOT/training/scripts/train_connect4.py"
LOG_DIR="$REPO_ROOT/artifacts/logs/training"
LOG_FILE="$LOG_DIR/training_log_v2.txt"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

echo "🚀 Launching Connect4 Training with Terminal Monitors"
echo "======================================================"

# Kill any existing training
pkill -f "train_connect4.py" 2>/dev/null
sleep 1

# Start training in background
echo "Starting training..."
nohup /mnt/ssd2pro/miniforge3/envs/tetrisrl/bin/python "$TRAIN_SCRIPT" > "$LOG_FILE" 2>&1 &
TRAINING_PID=$!
echo "✅ Training started (PID: $TRAINING_PID)"

sleep 2

# Launch continuous monitor in terminal 1
echo "Opening training monitor window..."
gnome-terminal --title="Training Monitor - Connect4" --geometry=120x40 -- bash -c "
while true; do
    clear
    echo '============================================================'
    echo '📊 Connect4 Training Monitor - Live Updates'
    echo '============================================================'
    echo ''
    echo 'Training PID: $(pgrep -f train_connect4.py)'
    echo ''
    
    # GPU Status
    echo '🎮 GPU Status:'
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | \
        awk -F', ' '{printf \"   GPU: %s%%  |  VRAM: %s/%s MB  |  Temp: %s°C\n\", \$1, \$2, \$3, \$4}'
    echo ''
    
    # Latest log output
    echo '📋 Latest Training Output:'
    echo '------------------------------------------------------------'
    tail -n 30 \"$LOG_FILE\" | tail -n 25
    echo '------------------------------------------------------------'
    echo ''
    echo 'Updated: $(date +\"%H:%M:%S\") | Press Ctrl+C to close window'
    
    sleep 5
done
" &

sleep 1

# Launch evaluation monitor in terminal 2
echo "Opening evaluation monitor window..."
gnome-terminal --title="Evaluation Monitor - Every 10 Iterations" --geometry=100x50 -- bash -c "
echo '============================================================'
echo '🎯 Evaluation Monitor - Every 10 Iterations'
echo '============================================================'
echo ''
echo 'Waiting for checkpoints to evaluate...'
echo ''

LAST_EVAL=0

while true; do
    # Check for new models
    LATEST_MODEL=\$(ls checkpoints/connect4/model_*.pt 2>/dev/null | sed 's/.*model_//' | sed 's/.pt//' | sort -n | tail -1)
    
    if [ ! -z \"\$LATEST_MODEL\" ]; then
        # Check if this is a milestone (every 10)
        if [ \$((LATEST_MODEL % 10)) -eq 0 ] && [ \$LATEST_MODEL -gt \$LAST_EVAL ]; then
            echo ''
            echo '========================================================'
            echo \"🔍 Running Evaluation: Iteration \$LATEST_MODEL\"
            echo '========================================================'
            echo ''
            
            /mnt/ssd2pro/miniforge3/envs/tetrisrl/bin/python experiments/evaluate_training.py \
                --model checkpoints/connect4/model_\${LATEST_MODEL}.pt
            
            LAST_EVAL=\$LATEST_MODEL
            
            echo ''
            echo \"✅ Evaluation complete for iteration \$LATEST_MODEL\"
            echo \"Next evaluation at iteration \$((LATEST_MODEL + 10))\"
            echo ''
        fi
    fi
    
    # Update status
    if [ ! -z \"\$LATEST_MODEL\" ]; then
        NEXT_EVAL=\$(( (LATEST_MODEL / 10 + 1) * 10 ))
        echo -ne \"\rLatest model: \$LATEST_MODEL | Next eval: \$NEXT_EVAL | $(date +%H:%M:%S)  \"
    fi
    
    sleep 10
done
" &

echo ""
echo "✅ Monitoring windows launched!"
echo ""
echo "Windows:"
echo "  1. Training Monitor - Live log updates every 5 seconds"
echo "  2. Evaluation Monitor - Runs tests every 10 iterations"
echo ""
echo "To stop training:"
echo "  pkill -f train_connect4.py"
echo ""
