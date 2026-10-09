"""Canonical command-line entry point for the Universal Representation screen."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m universal_experiment",
        description="Run the four-arm Universal Representation experiment.",
    )
    parser.add_argument("dataset", choices=("census", "aliccp"))
    args, remaining = parser.parse_known_args()

    # Keep the already verified dataset runners as implementation modules.  The
    # package CLI is the only documented entry point and forwards their flags.
    if args.dataset == "census":
        from universal_experiment.run import main as dataset_main
    else:
        from universal_experiment.aliccp import main as dataset_main

    sys.argv = [f"universal_experiment {args.dataset}", *remaining]
    dataset_main()


if __name__ == "__main__":
    main()
