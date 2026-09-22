#!/usr/bin/env python
"""Write a SYNTHETIC corpus with the Jigsaw file layout, for pipeline checks only.

    python scripts/make_synthetic_corpus.py --out data/synthetic

Produces train.csv, test.csv and test_labels.csv (including unscored -1 rows so
the scored-row filter is exercised). Numbers from a run on this corpus verify
that the pipeline works; they are never results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from review_router.synthetic import write_corpus  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=6000)
    parser.add_argument("--n-test", type=int, default=3000)
    parser.add_argument("--n-unscored", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(write_corpus(args.out, args.n_train, args.n_test, args.n_unscored, args.seed))


if __name__ == "__main__":
    main()
