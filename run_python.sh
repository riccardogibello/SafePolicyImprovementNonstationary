#!/bin/bash

# Name of the virtual environment
ENV_NAME=".venv"

# Check if the environment already exists
if [ ! -d "$ENV_NAME" ]; then
    # The environment doesn't exist, create it
    python3 -m venv $ENV_NAME
fi

# Activate the environment
source $ENV_NAME/bin/activate

# Install the requirements
pip install -r requirements.txt

# Run the Python script
python ./run.py \
    -f bandit_swarm.jl \
    -o log_dir \
    -s 0 \
    --speeds 0,4 \
    --ids 5 \
    --trials 10 \
    --eps 500

# Deactivate the environment
deactivate