import argparse
import subprocess

from codecarbon import track_emissions


@track_emissions(logging_logger=None)
def tracked_function(
        _command
):
    subprocess.run(_command, check=True)


if __name__ == '__main__':
    # Take all the parameters given as input
    parser = argparse.ArgumentParser(description="Process some integers.")

    # Add arguments
    parser.add_argument(
        '--init',
        action='store_true',
        help='Initialize the julia environment',
    )
    parser.add_argument(
        '-f',
        metavar='F',
        type=str,
        help='The julia file name to be processed.',
        choices=[
            'bandit_swarm.jl',
            'glucose_swarm.jl',
        ],
        required=True,
    )
    parser.add_argument(
        '-o',
        metavar='O',
        type=str,
        help='The output directory name.',
        required=True,
    )
    parser.add_argument(
        '-s',
        metavar='S',
        type=str,
        help='The seed.',
        required=True,
    )
    parser.add_argument(
        '--speeds',
        metavar='Sp',
        type=str,
        help='The speeds to be tested.',
        required=True,
    )
    parser.add_argument(
        '--ids',
        metavar='Ids',
        type=int,
        help='The number of experiments to be run for the given domain and speed.',
        required=True,
    )
    parser.add_argument(
        '--trials',
        metavar='T',
        type=int,
        help='The number of trials for each experiment.',
        required=True,
    )
    parser.add_argument(
        '--eps',
        metavar='E',
        type=int,
        help='The number of episodes for each trial.',
        required=True,
    )

    # Parse all the arguments
    args = parser.parse_args()
    # Get the given speeds
    speeds = args.speeds
    # Join all the speeds in a string that is comma separated
    speeds_str = ','.join([str(speed) for speed in speeds.split(',')])
    # Call the shell script with the given arguments
    # Define the command as a list
    print(str(args.trials))
    command = [
        "./run_experiments.sh",
        "--init",
        "-f", str(args.f),
        "-o", str(args.o),
        "-s", str(args.s),
        "--speeds", speeds_str,
        "--ids", str(args.ids),
        "--trials", str(args.trials),
        "--eps", str(args.eps)
    ]

    # Call the function that will be tracked
    tracked_function(
        command
    )
