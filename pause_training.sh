#!/bin/bash
# Pause Gomoku training

PID=$(pgrep -f "train_gomoku.py")

if [ -z "$PID" ]; then
    echo "❌ No training process found"
    exit 1
fi

echo "⏸️  Pausing training (PID: $PID)..."
kill -SIGTERM $PID
echo "✅ Training paused. Check training_log.txt for final status."
echo "   To resume, run: ./resume_training.sh"
