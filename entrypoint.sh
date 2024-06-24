#!/bin/sh
# Prepend the fixed arguments and append the arguments passed to docker run
exec sh run_python.sh Docker "$@"