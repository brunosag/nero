"""Main CLI entry point for NERO."""

import argparse
import sys

from nero.cli.commands import run_dummy_experiment


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NERO - NeuroEvolution vs. Gradient Research Orchestrator"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add run-dummy-experiment command
    dummy_parser = subparsers.add_parser(
        "run-dummy-experiment", help="Run a dummy experiment for integration testing"
    )
    dummy_parser.add_argument(
        "--optimizer",
        choices=["adam", "sgd"],
        default="adam",
        help="Optimizer to use (default: adam)",
    )
    dummy_parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs to train (default: 5)"
    )
    dummy_parser.add_argument(
        "--output-dir", type=str, help="Output directory for results"
    )

    args = parser.parse_args()

    if args.command == "run-dummy-experiment":
        return run_dummy_experiment(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
