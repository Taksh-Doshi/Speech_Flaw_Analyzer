"""Command-line task runner for the project skeleton."""

import argparse
import subprocess
import sys


COMMANDS = {
    "setup": [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
    "test": [sys.executable, "-m", "pytest"],
    "dataset": [sys.executable, "-m", "src.speechflaw.dataset_builder.build"],
    "analyze": [sys.executable, "-m", "src.speechflaw.pipeline"],
    "app": [sys.executable, "-m", "uvicorn", "api.main:app", "--reload"],
}


def main():
    """Run a named project task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    args = parser.parse_args()
    subprocess.run(COMMANDS[args.command], check=True)


if __name__ == "__main__":
    main()
