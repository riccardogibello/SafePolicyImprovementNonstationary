#!/bin/bash

# Get the current platform OS (Linux, macOS, Windows) as a user parameter
PLATFORM_OS=$1

# Name of the virtual environment
ENV_NAME=".venv"

# Get the current directory absolute path
CURRENT_DIR=$(pwd)

# Check if the environment already exists
if [ ! -d "$ENV_NAME" ]; then
    # The environment doesn't exist, create it
    python -m venv $ENV_NAME
fi

# Check if the platform OS is Linux or macOS
if [ "$PLATFORM_OS" == "Linux" ] || [ "$PLATFORM_OS" == "macOS" ]; then
    # Set the python executable path
    PYTHON_EXECUTABLE="$CURRENT_DIR/$ENV_NAME/bin/python"
    # Activate the environment
    source $ENV_NAME/bin/activate
elif [ "$PLATFORM_OS" == "Windows" ]; then
    # Set the python executable path
    PYTHON_EXECUTABLE="$CURRENT_DIR/$ENV_NAME/Scripts/python.exe"
    # Activate the environment
    source $ENV_NAME/Scripts/activate
fi

INVENV=$(python -c 'import sys; print ("1" if sys.prefix != sys.base_prefix else "0")')
# If the INVENV is set to 0 (not activated)
if [ "$INVENV" == "0" ]; then
    echo "The environment is not activated"
    exit 1
fi

# Install the requirements
pip install -r requirements.txt

# Run the Python script
python ./run.py \
    --init \
    --pythonexec $PYTHON_EXECUTABLE \
    -f bandit_swarm.jl \
    -o log_dir \
    -s 0 \
    --speeds 0 \
    --ids 1000 \
    --trials 10 \
    --eps 2000

# Check if the platform OS is Linux or macOS
if [ "$PLATFORM_OS" == "Linux" ] || [ "$PLATFORM_OS" == "macOS" ]; then
    # Deactivate the environment
    deactivate
elif [ "$PLATFORM_OS" == "Windows" ]; then
    # Deactivate the environment
    deactivate $ENV_NAME/Scripts/deactivate
fi