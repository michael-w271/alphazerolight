#!/bin/bash
# Pause Connect Four training

PID=$(pgrep -f "train_connect4.py")

if [ -z "$PID" ]; then
    echo "❌ No training process found"
    exit 1
fi

echo "⏸️  Pausing training (PID: $PID)..."
kill -SIGTERM $PID
echo "✅ Training paused. Check training_log.txt for final status."
echo "   To resume, run: ./resume_training.sh"
