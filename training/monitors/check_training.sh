#!/bin/bash
# Check Connect Four training status

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="$REPO_ROOT/artifacts/logs/training/training_log_v2.txt"

PID=$(pgrep -f "train_connect4.py")

if [ -z "$PID" ]; then
    echo "❌ Training is not running"
    echo ""
    echo "Last 10 lines of log:"
    if [ -f "$LOG_FILE" ]; then
        tail -n 10 "$LOG_FILE"
    else
        echo "Log file not found: $LOG_FILE"
    fi
else
    echo "✅ Training is running (PID: $PID)"
    echo ""
    echo "Latest progress:"
    if [ -f "$LOG_FILE" ]; then
        tail -n 20 "$LOG_FILE" | grep -E "(Iteration|Self-Play|Training|Loss|Win Rate)" | tail -n 10
    else
        echo "Log file not found: $LOG_FILE"
    fi
fi
