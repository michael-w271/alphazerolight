#!/bin/bash
# Configuration for the project environment

# Path to the Python executable in the virtual environment
export PYTHON_EXEC="/home/michael/miniforge3/envs/azl/bin/python"
export VENV_NAME="azl"

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "Using Python: $PYTHON_EXEC"
