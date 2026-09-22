"""SYNTHETIC corpus with the Jigsaw file layout, for pipeline checks only.

Numbers from a run on this corpus verify that the pipeline works; they are
never results. See scripts/make_synthetic_corpus.py for the CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from review_router.data import LABELS

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = ["make_rows", "write_corpus"]

BENIGN_TEXT = (
    "the article section should cite a source please discuss on the talk page "
    "thanks for the edit i think the paragraph needs work agree with the revert "
    "see the guideline about notability this reference looks fine to me"
)
BENIGN = tuple(BENIGN_TEXT.split(" "))
IDENTITY = ("muslim", "jewish", "gay", "black", "women", "immigrants", "christian")
CUES = {
    "toxic": ("idiot", "stupid", "moron", "dumb", "shut up"),
    "severe_toxic": ("die", "scum", "filth", "worthless"),
    "obscene": ("crap", "damn", "bloody", "arse"),
    "threat": ("kill you", "hurt you", "find you", "destroy you"),
    "insult": ("loser", "pathetic", "clown", "fool"),
    "identity_hate": ("hate all", "disgusting", "go back", "not welcome"),
}
PREVALENCE = {
    "toxic": 0.10,
    "severe_toxic": 0.01,
    "obscene": 0.05,
    "threat": 0.006,
    "insult": 0.05,
    "identity_hate": 0.012,
}


def make_rows(n: int, rng: np.random.Generator, prefix: str) -> pd.DataFrame:
    import pandas as pd

    labels = np.zeros((n, len(LABELS)), dtype=int)
    for j, label in enumerate(LABELS):
        labels[:, j] = rng.random(n) < PREVALENCE[label]
    tox, sev = LABELS.index("toxic"), LABELS.index("severe_toxic")
    labels[labels[:, sev] == 1, tox] = 1  # severe_toxic is a subset of toxic
    texts = []
    for i in range(n):
        words = list(rng.choice(BENIGN, size=rng.integers(6, 30)))
        if rng.random() < 0.08:
            words.append(str(rng.choice(IDENTITY)))
        for j, label in enumerate(LABELS):
            if labels[i, j] and rng.random() < 0.9:
                words.append(str(rng.choice(CUES[label])))
                if label == "identity_hate":
                    words.append(str(rng.choice(IDENTITY)))
            elif not labels[i, j] and rng.random() < 0.01:
                words.append(str(rng.choice(CUES[label])))  # label noise
        rng.shuffle(words)
        texts.append(" ".join(words))
    frame = pd.DataFrame({"id": [f"{prefix}{i:07d}" for i in range(n)], "comment_text": texts})
    for j, label in enumerate(LABELS):
        frame[label] = labels[:, j]
    return frame


def write_corpus(
    out: Path, n_train: int = 6000, n_test: int = 3000, n_unscored: int = 500, seed: int = 7
) -> Path:
    """Write train.csv, test.csv and test_labels.csv (with -1 unscored rows) to `out`."""
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    make_rows(n_train, rng, "tr").to_csv(out / "train.csv", index=False)
    test = make_rows(n_test + n_unscored, rng, "te")
    test_labels = test[["id", *LABELS]].copy()
    unscored = rng.choice(len(test), size=n_unscored, replace=False)
    test_labels.loc[unscored, list(LABELS)] = -1
    test[["id", "comment_text"]].to_csv(out / "test.csv", index=False)
    test_labels.to_csv(out / "test_labels.csv", index=False)
    (out / "SYNTHETIC.txt").write_text(
        "Synthetic corpus for pipeline checks. Not Jigsaw. Never report numbers from it.\n"
    )
    return out
