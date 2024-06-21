#
# NOTE: THIS DOCKERFILE IS GENERATED VIA "apply-templates.sh"
#
# PLEASE DO NOT EDIT IT DIRECTLY.
#

FROM debian:bookworm-slim

# Create app directory
WORKDIR /usr/src/app

# Copy the current directory contents into the container # .dockerignore: Ignore foled .vscode, log_dir, .venv
COPY . /usr/src/app

# Make run_python.sh and run_experiments.sh executable
RUN chmod +x /usr/src/app/run_experiments.sh
RUN chmod +x /usr/src/app/run_python.sh

# Python installation
RUN set -eux; \
	apt-get update; \
	apt-get install -y --no-install-recommends \
	python3 \
	python3-pip \
	libpython3-dev \
	; \
	rm -rf /var/lib/apt/lists/*

RUN mv /usr/lib/python3.11/EXTERNALLY-MANAGED /usr/lib/python3.11/EXTERNALLY-MANAGED.old

# Set environment variables
ENV PYTHON /usr/bin/python3
#ENV PYTHONPATH /usr/lib/python3.11
ENV PYCALL_DEBUG_BUILD = "yes"
# Install Pyhton dependencies
RUN pip3 install --upgrade pip && \
	pip3 install -r requirements.txt


RUN set -eux; \
	apt-get update; \
	apt-get install -y --no-install-recommends \
	ca-certificates \
	# ERROR: no download agent available; install curl, wget, or fetch
	curl \
	wget \
	tar \
	; \
	rm -rf /var/lib/apt/lists/*

ENV JULIA_PATH /usr/local/julia
ENV PATH $JULIA_PATH/bin:$PATH

# Download and extract Julia
RUN wget -O julia.tar.gz "https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.4-linux-x86_64.tar.gz"
RUN mkdir "$JULIA_PATH"
RUN tar -xzf julia.tar.gz -C "$JULIA_PATH" --strip-components 1
RUN rm julia.tar.gz

# Add Julia to the PATH
ENV PATH="$JULIA_PATH/bin:$PATH"

# Smoke test
RUN julia --version

# Run the run_python.sh script
CMD [ "sh", "run_python.sh", "Docker"]

# HOWTO:
# 1) Create a volume: docker volume create rl_recsys_volume
# 2) Build the image: docker build -f Dockerfile -t rl_project_recsys .
# 3) Run the container with mounted volume: docker run -d --name rl_recsys_cont --rm -it -v rl_recsys_volume:/usr/src/app/log_dir rl_project_recsys

# IF YOU WANT TO RUN 2 CONTAINERS IN PARALLEL:
# 1) Create a volume: docker volume create rl_recsys_volume
# 2) Modify the 'rune_experiments.sh' script to run speeds 0,1
# 3) Build the image: docker build -f Dockerfile -t rl_speed_01 .
# 4) Modify the 'rune_experiments.sh' script to run speeds 2,3
# 5) Build the image: docker build -f Dockerfile -t rl_speed_23 .
# 6) Run the container with mounted volume: docker run -d --name rl_speed_01_cont --rm -it -v rl_recsys_volume:/usr/src/app/log_dir rl_speed_01
# 7) Run the container with mounted volume: docker run -d --name rl_speed_23_cont --rm -it -v rl_recsys_volume:/usr/src/app/log_dir rl_speed_23

