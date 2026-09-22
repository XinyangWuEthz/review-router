#!/usr/bin/env python
"""Verify the corpus files against the pinned sha256 list; exit 1 on any mismatch.

python scripts/verify_data.py data/jigsaw configs/jigsaw.sha256
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

CORPUS_FILES = {"train.csv", "test.csv", "test_labels.csv"}


def load_pins(path: Path) -> dict[str, str]:
    """Require one SHA-256 pin for each official input, without extra paths."""
    pins: dict[str, str] = {}
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"invalid pin on line {number}")
        digest, name = parts
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError(f"invalid SHA-256 on line {number}")
        if name not in CORPUS_FILES or name in pins:
            raise ValueError(f"unexpected or duplicate corpus file on line {number}: {name}")
        pins[name] = digest.lower()
    if set(pins) != CORPUS_FILES:
        raise ValueError(f"pins must contain exactly {sorted(CORPUS_FILES)}")
    return pins


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(data_dir: Path, pins: Path) -> int:
    try:
        expected_hashes = load_pins(pins)
    except (OSError, ValueError) as error:
        print(f"INVALID PINS  {error}")
        return 1
    failures = 0
    for name, expected in expected_hashes.items():
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
