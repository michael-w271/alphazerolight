#!/bin/bash
# Resume Gomoku training (or start fresh if no checkpoints exist)

source ./env_config.sh

echo "🚀 Starting/Resuming Gomoku training..."
$PYTHON_EXEC scripts/train_gomoku.py > training_log.txt 2>&1 &

PID=$!
echo "✅ Training started in background (PID: $PID)"
echo "   Monitor progress: tail -f training_log.txt"
echo "   Pause training: ./pause_training.sh"
