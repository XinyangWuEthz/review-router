"""Corpus loading helpers.

The only thing here that matters today is drop_unscored: Jigsaw's test.csv has
153,164 rows but only 63,978 are scored. The rest carry -1 in every label column
and silently poison any metric computed over them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = ["LABELS", "UNSCORED_SENTINEL", "drop_unscored"]

LABELS: tuple[str, ...] = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)

UNSCORED_SENTINEL = -1


def drop_unscored(
    labels: pd.DataFrame, label_columns: tuple[str, ...] = LABELS
) -> pd.DataFrame:
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
