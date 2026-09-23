#!/usr/bin/env python
"""Restore the pinned scorer from a saved run or release ZIP, without training or unpickling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from review_router.frozen import restore_frozen_archive, restore_frozen_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument(
        "--source-run", type=Path, help="directory containing the trusted saved run"
    )
    sources.add_argument("--archive", type=Path, help="flat ZIP of the three pinned scorer files")
    parser.add_argument("--lock", type=Path, default=ROOT / "configs/frozen-baseline.json")
    args = parser.parse_args()
    lock_path = args.lock.resolve()
    if args.archive is not None:
        print(restore_frozen_archive(args.archive.resolve(), lock_path))
    else:
        print(restore_frozen_model(args.source_run.resolve(), lock_path))


if __name__ == "__main__":
    main()
