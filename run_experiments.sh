#!/bin/bash

# Example of a script call:
# ./run_experiments.sh -f bandit_swarm.jl -o log_dir -s 0 --speeds 0 --ids 10 --trials 10 --eps 500

init=false
input_file=""
ids=()
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
            init=true
            shift # Remove --init from processing
            ;;
        -f)
            input_file="$2"
            shift # Remove argument name from processing
            shift # Remove argument value from processing
            ;;
        -o)
            log_dir="$2"
            shift # Remove argument name from processing
            shift # Remove argument value from processing
            ;;
        -s)
            seed="$2"
            shift # Remove argument name from processing
            shift # Remove argument value from processing
            ;;
        --speeds)
            # Set the internal field separator to a comma 
            # and split the argument into an array
            IFS=',' read -r -a speeds <<< "$2"
            shift # Remove argument name from processing
            shift # Remove argument value from processing
            ;;
        --ids)
            # Get the value given as an argument
            id_nums="$2"
            # Verify if the value contains a comma
            if [[ $id_nums == *","* ]]; then
                # If it does, split the value into an array
                IFS=',' read -r -a ids <<< "$id_nums"
                # Check that ids is of two elements
                if [ ${#ids[@]} -ne 2 ]; then
                    echo "Invalid number of values for ids. Expected 2, got ${#ids[@]}"
                    exit 1
                fi
                # Set the first value to be the starting index
                id_nums=${ids[0]}
                # Set the second value to be the ending index
                id_end=${ids[1]}
                # Create an array with the values from the starting index to the ending index
                ids=($(seq $id_nums $id_end))
                echo "ids: ${ids[@]}"
            else
                # If it doesn't, create an array with the values from 1 to the given value
                ids=($(seq 1 $id_nums))
            fi
            shift # Remove argument name from processing
            shift # Remove argument value from processing
            ;;
        --trials)
            trials="$2"
            shift # Remove argument name from processing
            shift # Remove argument value from processing
            ;;
        --eps)
            eps="$2"
            shift # Remove argument name from processing
            shift # Remove argument value from processing
            ;;
        --pythonexec)
            pythonexe="$2"
            ECHO "Python executable: $pythonexe"
            shift # Remove argument name from processing
            shift # Remove argument value from processing
            ;;
        *)
            echo "Invalid option: $arg" >&2
            exit 1
            ;;
    esac
done

juliaup update

# If the input file is not given
if [ -z "$input_file" ]; then
    echo "Input file for the experiment domain choice not given. Options: bandit_swarm.jl (RecoSys), glucose_swarm.jl (Glucose)"
    exit 1
fi

# If any of the required parameters are not given
if [ -z "$log_dir" ] || [ -z "$seed" ] || [ -z "$speeds" ] || [ -z "$ids" ] || [ -z "$trials" ] || [ -z "$eps" ] || [ -z "$pythonexe" ]; then
    echo "Missing required parameters. Required: log_dir, seed, speeds, ids, trials, eps, pythonexe"
    exit 1
fi


# Print the speeds
echo "Speeds: ${speeds[@]}"


# If the julia environment is not initialized
if [ "$init" = true ]; then
    # Initialize the python environment variable, create and activate 
    # the julia environment, install the required packages
    julia --project=@. -e "
        using Pkg; 
        Pkg.activate(\".\");
        Pkg.instantiate();
        Pkg.build(\"IJulia\");
        Pkg.build(\"PyCall\");
        Pkg.add(Pkg.PackageSpec(url=\"https://github.com/ScottJordan/EvaluationOfRLAlgs.git\"));
    "
fi

export PYTHON=pythonexe
export JULIA_NUM_THREADS=16

# For each speed in the list
for speed in "${speeds[@]}"; do
    # For each experiment id in the list
    for id in "${ids[@]}"; do
        julia --project=@. ./experiments/$input_file \
            --log-dir $log_dir \
            --id $id \
            --seed $seed \
            --trials $trials \
            --speed $speed \
            --eps $eps
    done
done