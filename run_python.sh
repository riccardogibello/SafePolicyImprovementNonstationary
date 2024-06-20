#!/bin/bash

# If, during docker run command, you encounter errors related to the shell script syntax,
# use this "dos2unix file_name"; 

# Get the current platform OS (Linux, macOS, Windows, Docker) as a user parameter
PLATFORM_OS=$1
echo "The platform OS is $PLATFORM_OS"

if [ "$PLATFORM_OS" = "Docker" ]; then
    echo "The platform OS is Docker"
    # Set the python executable path
    PYTHON_EXECUTABLE="/usr/bin/python3"

    # Run the Python script
    python3 ./run.py \
        --init \
        --pythonexec $PYTHON_EXECUTABLE \
        -f bandit_swarm.jl \
        -o log_dir \
        -s 0 \
        --speeds 2,3 \
        --ids 2,3 \
        --trials 10 \
        --eps 10
        # eps must be 2000
        # ids must be 1000
else
    # Get the current directory absolute path
    CURRENT_DIR=$(pwd)
    # Name of the virtual environment
    ENV_NAME=".venv"
    # Check if the environment folder already exists
    if [ ! -d "$ENV_NAME" ]; then
        # The environment doesn't exist, create it
        python -m venv $ENV_NAME
    fi
    # Check if the platform OS is Linux or macOS
    if [ "$PLATFORM_OS" = "Linux" ] || [ "$PLATFORM_OS" = "macOS" ]; then
        # Set the python executable path
        PYTHON_EXECUTABLE="$CURRENT_DIR/$ENV_NAME/bin/python"
        # Activate the environment
        source $ENV_NAME/bin/activate
    fi
    if [ "$PLATFORM_OS" = "Windows" ]; then
        # Set the python executable path
        PYTHON_EXECUTABLE="$CURRENT_DIR/$ENV_NAME/Scripts/python.exe"
        # Activate the environment
        $ENV_NAME/Scripts/activate
    fi

    INVENV=$(python -c 'import sys; print ("1" if sys.prefix != sys.base_prefix else "0")')
    # If the INVENV is set to 0 (not activated)
    if [ "$INVENV" == "0" ]; then
        echo "The environment is not activated"
        exit 1
    fi

    # Run the Python script
    python ./run.py \
        --init \
        --pythonexec $PYTHON_EXECUTABLE \
        -f bandit_swarm.jl \
        -o log_dir \
        -s 0 \
        --speeds 2,3 \
        --ids 2,3 \
        --trials 10 \
        --eps 10
        # eps must be 2000
        # ids must be 1000
fi

# Check if the platform OS is Linux or macOS or Windows
if [ "$PLATFORM_OS" = "Linux" ] || [ "$PLATFORM_OS" = "macOS" ]; then
    # Deactivate the environment
    deactivate
fi