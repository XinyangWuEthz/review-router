#!/usr/bin/env python
"""Verify the corpus files against the pinned sha256 list; exit 1 on any mismatch.

python scripts/verify_data.py data/jigsaw configs/jigsaw.sha256
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(data_dir: Path, pins: Path) -> int:
    failures = 0
    for line in pins.read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split()
        path = data_dir / name
        if not path.is_file():
            print(f"MISSING  {name}")
            failures += 1
            continue
        actual = sha256(path)
        status = "OK      " if actual == expected else "MISMATCH"
        failures += actual != expected
        print(f"{status} {name} {actual[:16]}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
