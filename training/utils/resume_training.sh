#!/bin/bash
# Resume Connect Four training (or start fresh if no checkpoints exist)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$REPO_ROOT/artifacts/logs/training"
LOG_FILE="$LOG_DIR/training_log_v2.txt"

mkdir -p "$LOG_DIR"
source "$REPO_ROOT/env_config.sh"

echo "🚀 Starting/Resuming Connect Four training..."
"$PYTHON_EXEC" "$REPO_ROOT/training/scripts/train_connect4.py" > "$LOG_FILE" 2>&1 &

PID=$!
echo "✅ Training started in background (PID: $PID)"
echo "   Monitor progress: tail -f $LOG_FILE"
echo "   Pause training: bash $REPO_ROOT/training/utils/pause_training.sh"
