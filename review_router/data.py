"""Corpus loading, the scored-row filter and the frozen train/calib/thresh split.

drop_unscored is the one thing here that bites: Jigsaw's test.csv has 153,164
rows but only 63,978 are scored. The rest carry -1 in every label column and
silently poison any metric computed over them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = [
    "LABELS",
    "UNSCORED_SENTINEL",
    "SPLIT_NAMES",
    "Corpus",
    "drop_unscored",
    "load_corpus",
    "assign_splits",
    "label_counts",
    "file_sha256",
]

LABELS: tuple[str, ...] = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)

UNSCORED_SENTINEL = -1

SPLIT_NAMES: tuple[str, ...] = ("train", "calib", "thresh")


def drop_unscored(labels: pd.DataFrame, label_columns: tuple[str, ...] = LABELS) -> pd.DataFrame:
    """Return only the scored rows of a Jigsaw test_labels frame.

    Unscored rows carry -1 in every label column. Evaluating on them is the
    single most common error on this corpus: it silently mixes 89,186 unlabeled
    rows into the negative class.
    """
    missing = [c for c in label_columns if c not in labels.columns]
    if missing:
        raise KeyError(f"label columns missing from frame: {missing}")

    scored = (labels[list(label_columns)] != UNSCORED_SENTINEL).all(axis=1)
    return labels.loc[scored]


@dataclass(frozen=True)
class Corpus:
    """Training rows with their split assignment, and the scored test rows."""

    train: pd.DataFrame  # id, comment_text, labels..., split
    test: pd.DataFrame  # id, comment_text, labels...  (scored rows only)
    file_hashes: dict[str, str]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_splits(n: int, fractions: dict[str, float], seed: int) -> np.ndarray:
    """Seeded random split of n rows into the named fractions.

    Plain random, not stratified: the label counts per split are written to the
    manifest so any imbalance is visible rather than hidden.
    """
    if set(fractions) != set(SPLIT_NAMES):
        raise ValueError(
            f"split fractions must name exactly {SPLIT_NAMES}, got {sorted(fractions)}"
        )
    if any(not np.isfinite(value) or value <= 0 for value in fractions.values()):
        raise ValueError("split fractions must be positive and finite")
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1, got {total}")

    counts = [int(round(fractions[name] * n)) for name in SPLIT_NAMES[:-1]]
    counts.append(n - sum(counts))
    if any(count <= 0 for count in counts):
        raise ValueError("every split must contain at least one row")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    out = np.empty(n, dtype=object)
    start = 0
    for name, count in zip(SPLIT_NAMES, counts, strict=True):
        end = start + count
        out[order[start:end]] = name
        start = end
    return out


def load_corpus(data_dir: Path, split_fractions: dict[str, float], seed: int) -> Corpus:
    """Load train.csv, test.csv and test_labels.csv; merge, filter and split."""
    import pandas as pd

    paths = {name: data_dir / f"{name}.csv" for name in ("train", "test", "test_labels")}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: expected {path}")

    train = pd.read_csv(paths["train"], dtype={"id": str})
    test_text = pd.read_csv(paths["test"], dtype={"id": str})
    test_labels = pd.read_csv(paths["test_labels"], dtype={"id": str})

    for frame, name in ((train, "train"), (test_text, "test"), (test_labels, "test_labels")):
        if "id" not in frame.columns:
            raise KeyError(f"{name}.csv is missing the id column")
        if frame["id"].isna().any() or frame["id"].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name}.csv contains missing ids")
        if frame["id"].duplicated().any():
            raise ValueError(f"{name}.csv contains duplicate ids")

    if set(train["id"]) & set(test_text["id"]):
        raise ValueError("train.csv and test.csv contain overlapping ids")

    for frame, name in ((train, "train"), (test_text, "test")):
        if "comment_text" not in frame.columns:
            raise KeyError(f"{name}.csv is missing the comment_text column")

    for frame, name in ((train, "train"), (test_labels, "test_labels")):
        missing = [c for c in LABELS if c not in frame.columns]
        if missing:
            raise KeyError(f"{name}.csv is missing label columns {missing}")

    scored = drop_unscored(test_labels)
    for frame, name in ((train, "train"), (scored, "scored test_labels")):
        if not frame[list(LABELS)].isin([0, 1]).all().all():
            raise ValueError(f"{name}.csv contains non-binary labels")
    if not scored["id"].isin(test_text["id"]).all():
        raise ValueError("test.csv is missing texts for scored test_labels ids")
    test = test_text.merge(scored, on="id", how="inner", validate="one_to_one")
    if test.empty:
        raise ValueError("no scored test rows after merging test.csv with test_labels.csv")
    for frame, name in ((train, "train"), (test, "scored test")):
        if frame["comment_text"].isna().any():
            raise ValueError(f"{name}.csv contains missing comment_text values")

    train = train.copy()
    train["split"] = assign_splits(len(train), split_fractions, seed)
    train = train[["id", "comment_text", *LABELS, "split"]]
    test = test[["id", "comment_text", *LABELS]]

    hashes = {name: file_sha256(path) for name, path in paths.items()}
    return Corpus(train=train, test=test, file_hashes=hashes)


def label_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Positive count per label, plus the row total."""
    counts = {label: int(frame[label].sum()) for label in LABELS}
    counts["rows"] = int(len(frame))
    return counts
