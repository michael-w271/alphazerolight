#!/bin/bash
# Check Gomoku training status

PID=$(pgrep -f "train_gomoku.py")

if [ -z "$PID" ]; then
    echo "❌ Training is not running"
    echo ""
    echo "Last 10 lines of log:"
    tail -n 10 training_log.txt
else
    echo "✅ Training is running (PID: $PID)"
    echo ""
    echo "Latest progress:"
    tail -n 20 training_log.txt | grep -E "(Iteration|Self-Play|Training|Loss|Win Rate)" | tail -n 10
fi
