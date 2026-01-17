#!/bin/bash
# Resume Connect Four training (or start fresh if no checkpoints exist)

source ./env_config.sh

echo "🚀 Starting/Resuming Connect Four training..."
$PYTHON_EXEC scripts/train_connect4.py > training_log.txt 2>&1 &

PID=$!
echo "✅ Training started in background (PID: $PID)"
echo "   Monitor progress: tail -f training_log.txt"
echo "   Pause training: ./pause_training.sh"
