# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install required system packages and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    git \
    jq && \
    rm -rf /var/lib/apt/lists/*

# Install Julia
RUN curl -fsSL https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.4-linux-x86_64.tar.gz | tar -xz -C /opt/ && \
    ln -s /opt/julia-1.10.4/bin/julia /usr/local/bin/julia

# Create app directory
WORKDIR /usr/src/app

# Copy the current directory contents into the container # .dockerignore: Ignore foled .vscode, log_dir, .venv
COPY . /usr/src/app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# Make run_python.sh and run_experiments.sh executable
RUN chmod +x /usr/src/app/run_experiments.sh
RUN chmod +x /usr/src/app/run_python.sh


# Initialize PYTHONPATH environment variable
ENV PYTHONPATH /usr/src/app

# Set the default command to run the python script
CMD ["python", "./run.py", "--init", "-f", "bandit_swarm.jl", "-o", "log_dir", "-s", "0", "--speeds", "2,3", "--ids", "2,3", "--trials", "10", "--eps", "2000"]

# docker build -t my_experiment .
# docker run -it --rm --name my_experiment my_experiment
# WITH MOUNT (to:from): # docker run -it --rm -v ./from_docker:./log_var --name my_experiment my_experiment