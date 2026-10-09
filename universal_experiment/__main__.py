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
    parser.add_argument("--budget20", action="store_true", help="Reuse Stage-1 for the 20-epoch screen")
    parser.add_argument("--stage1-read", action="store_true",
                        help="Stage-1 old-task read-only-U four-arm screen")
    parser.add_argument("--newtask-read", action="store_true",
                        help="Read-U Stage-1 -> new-task transfer four-arm screen")
    args, remaining = parser.parse_known_args()

    # Keep the already verified dataset runners as implementation modules.  The
    # package CLI is the only documented entry point and forwards their flags.
    if args.stage1_read:
        from universal_experiment.stage1_read import main as read_main
        sys.argv = [f"universal_experiment {args.dataset}", *remaining]
        read_main(args.dataset)
        return
    if args.newtask_read:
        from universal_experiment.newtask_read import main as newtask_main
        sys.argv = [f"universal_experiment {args.dataset}", *remaining]
        newtask_main(args.dataset)
        return
    if args.budget20:
        from universal_experiment.budget import main as budget_main
        sys.argv = [f"universal_experiment {args.dataset}", *remaining]
        budget_main(args.dataset)
        return
    if args.dataset == "census":
        from universal_experiment.run import main as dataset_main
    else:
        from universal_experiment.aliccp import main as dataset_main

    sys.argv = [f"universal_experiment {args.dataset}", *remaining]
    dataset_main()


if __name__ == "__main__":
    main()
