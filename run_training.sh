#!/bin/bash
source ./env_config.sh

echo "Starting Training..."
$PYTHON_EXEC scripts/run_train.py
