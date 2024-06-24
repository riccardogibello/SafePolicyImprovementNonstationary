#!/bin/bash

# If, during docker run command, you encounter errors related to the shell script syntax,
# use this "dos2unix file_name";

init=false
input_file=""
# This is a list that will contain the speeds at which 
# the experiments will be run
speeds=()
# This is a list that will contain incremental integer values 
# (i.e., the identifiers of each experiment). Each of them is run
# for each of the speeds in the list.
ids=()

# "$#" = a variable that holds the number of arguments passed to the script;
# As long as there are arguments to process, the script will continue to process them
while (( "$#" )); do
    # Take the first argument of the list (option)
    case $1 in
        --init)
            # --init indicates whether the Julia environment
            # must be initialized or not.
            init=true
            shift # Remove --init from processing
            ;;
        --n-proc)
            # --proc is the number of processes to be used.
            # This is not used in the Julia script.
            n_proc="$2"
            shift # Remove argument name from processing
            shift # Remove argument value from processing
            ;;
        -f)
            # -f is the name of the Julia file to be run.
            input_file="$2"
            shift # Remove argument name from processing
            shift # Remove argument value from processing
            ;;
        -o)
            # -o is the directory where the logs will be saved.
            log_dir="$2"
            shift # Remove argument name from processing
            shift # Remove argument value from processing
            ;;
        -s)
            # -s is the seed for the random number generator.
            seed="$2"
            shift # Remove argument name from processing
            shift # Remove argument value from processing
            ;;
        --speeds)
            # --speeds is a list of speeds at which the experiments will be run.
            # Set the internal field separator to a comma 
            # and split the argument into an array
            IFS=',' read -r -a speeds <<< "$2"
            shift # Remove argument name from processing
            shift # Remove argument value from processing
            ;;
        --ids)
            # --ids is the list of experiment IDs to be run. It can be either
            # a comma separated list of two values (start id and end id) 
            # or a single value (from zero to value id).
            # Get the value given as an argument
            id_nums="$2"
            shift # Remove argument name from processing
            shift # Remove argument value from processing
            ;;
        --trials)
            # --trials is the number of trials to be run for each experiment.
            trials="$2"
            shift # Remove argument name from processing
            shift # Remove argument value from processing
            ;;
        --eps)
            # --eps is the number of episodes to be run for each trial.
            eps="$2"
            shift # Remove argument name from processing
            shift # Remove argument value from processing
            ;;
        --pythonexec)
            # --pythonexec is the path to the python executable.
            pythonexe="$2"
            shift # Remove argument name from processing
            shift # Remove argument value from processing
            ;;
        *)
            echo "Invalid option: $arg" >&2
            exit 1
            ;;
    esac
done

# If the input file is not given
if [ -z "$input_file" ]; then
    echo "Input file for the experiment domain choice not given. Options: bandit_swarm.jl (RecoSys), glucose_swarm.jl (Glucose)"
    exit 1
fi

# If any of the required parameters are not given
if [ -z "$log_dir" ] || [ -z "$seed" ] || [ -z "$speeds" ] || [ -z "$id_nums" ] || [ -z "$trials" ] || [ -z "$eps" ]; then #|| [ -z "$pythonexe" ]; then
    echo "Missing required parameters. Required: log_dir, seed, speeds, ids, trials, eps, pythonexe"
    exit 1
fi

# If the julia environment is not initialized
if [ "$init" = "true" ]; then
    # Initialize the python environment variable (not sufficient
    # the outside "export" command), create and activate 
    # the julia environment, install the required packages
    julia --project=@. -e "
        ENV[\"PYTHON\"] = \"$pythonexe\";
        using Pkg;
        Pkg.activate(\".\");
 
        Pkg.instantiate();
        
        Pkg.build(\"IJulia\");
        Pkg.build(\"PyCall\");
        Pkg.add(Pkg.PackageSpec(url=\"https://github.com/ScottJordan/EvaluationOfRLAlgs.git\"));
    "
fi

# Set the environment variable for the python executable
export PYTHON=pythonexe

# For each speed in the list, representing the degree of
# non-stationarity of the environment
for speed in "${speeds[@]}"; do
    # Run all the experiments that are required
    julia --project=@. ./experiments/$input_file \
        --log-dir $log_dir \
        --n-proc $n_proc \
        --ids $id_nums \
        --seed $seed \
        --trials $trials \
        --speed $speed \
        --eps $eps
done