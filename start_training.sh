#!/bin/bash
# Root launcher wrapper to keep existing commands working.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$REPO_ROOT/training/scripts/start_training.sh" "$@"
