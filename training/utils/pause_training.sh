#!/bin/bash
# Pause Connect Four training

PID=$(pgrep -f "train_connect4.py")

if [ -z "$PID" ]; then
    echo "❌ No training process found"
    exit 1
fi

echo "⏸️  Pausing training (PID: $PID)..."
kill -SIGTERM $PID
echo "✅ Training paused. Check artifacts/logs/training/training_log_v2.txt for final status."
echo "   To resume, run: bash training/utils/resume_training.sh"
