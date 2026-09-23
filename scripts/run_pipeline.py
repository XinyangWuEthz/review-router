#!/usr/bin/env python
"""Evaluate the router using the config's frozen scorer, or explicitly retrain.

    python scripts/run_pipeline.py --config configs/baseline.yaml

Prints the run directory; point REVIEW_ROUTER_EVAL_REPORT at its report.json
to run the regression gates against it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from review_router.pipeline import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="explicitly fit a new model and calibration; never replaces the frozen artifact",
    )
    args = parser.parse_args()
    run_dir = run(args.config.resolve(), retrain=args.retrain)
    print(run_dir)
    print((run_dir / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
