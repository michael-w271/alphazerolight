#!/bin/bash
# Project environment configuration

export VENV_NAME="azl"
export CONDA_ROOT="/mnt/ssd2pro/miniforge3"

# Load conda + activate env
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "$VENV_NAME" ]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "$VENV_NAME"
fi

export PYTHON_EXEC="${CONDA_ROOT}/envs/${VENV_NAME}/bin/python"

# Fail fast if env is broken
if [ ! -x "$PYTHON_EXEC" ]; then
    echo "ERROR: Python not found at $PYTHON_EXEC"
    exit 1
fi

# Add project root to PYTHONPATH so imports work everywhere
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# ---- PyTorch C++ (pip torch) paths for compiling C++/libtorch-style code ----
export TORCH_DIR="$("$PYTHON_EXEC" -c 'import os,torch; print(os.path.dirname(torch.__file__))')"
export TORCH_INCLUDE1="$TORCH_DIR/include"
export TORCH_INCLUDE2="$TORCH_DIR/include/torch/csrc/api/include"
export TORCH_LIB="$TORCH_DIR/lib"

# Help compilers & CMake find headers/libs
export CPLUS_INCLUDE_PATH="$TORCH_INCLUDE1:$TORCH_INCLUDE2:${CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$TORCH_LIB:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="$TORCH_DIR:${CMAKE_PREFIX_PATH}"

echo "Using conda env: $VENV_NAME"
echo "Python executable: $PYTHON_EXEC"
echo "TORCH_DIR: $TORCH_DIR"
