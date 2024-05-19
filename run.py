import argparse
import platform
import subprocess
import sys

from codecarbon import OfflineEmissionsTracker
import logging


def tracked_function(
        _command
):
    tracker = OfflineEmissionsTracker(
        country_iso_code="ITA",
        measure_power_secs=30,
        project_name="safe_policy_experiment",
    )

    # Create a dedicated logger (log name can be the CodeCarbon project name for example)
    logger = logging.getLogger('codecarbon')
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])

    # Define a log formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)-12s: %(levelname)-8s %(message)s"
    )

    # Create file handler which logs debug messages
    fh = logging.FileHandler("codecarbon.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    consoleHandler.setLevel(logging.WARNING)
    logger.addHandler(consoleHandler)

    logger.debug("GO!")

    tracker.start()
    try:
        subprocess.run(_command, check=True)
    finally:
        tracker.stop()


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
    parser.add_argument(
        '--pythonexec',
        metavar='PE',
        type=str,
        help='The python executable to be used.',
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
    # Get the OS name
    os_name = platform.system()
    # If the OS is Windows
    if os_name == 'Windows':
        command = [
            "sh",
            "./run_experiments.sh",
        ]
    else:
        command = [
            "./run_experiments.sh",
        ]

    command.extend([
        "--init",
        "-f", str(args.f),
        "-o", str(args.o),
        "-s", str(args.s),
        "--speeds", speeds_str,
        "--ids", str(args.ids),
        "--trials", str(args.trials),
        "--eps", str(args.eps),
        '--pythonexec', str(args.pythonexec),
    ])

    # Call the function that will be tracked
    tracked_function(
        command
    )
