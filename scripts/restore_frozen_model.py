#!/usr/bin/env python
"""Restore the pinned scorer from its trusted saved run, without training or loading pickle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from review_router.frozen import restore_frozen_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=ROOT / "configs/frozen-baseline.json")
    args = parser.parse_args()
    print(restore_frozen_model(args.source_run.resolve(), args.lock.resolve()))


if __name__ == "__main__":
    main()
