from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from review_router.data import (
    LABELS,
    SPLIT_NAMES,
    assign_splits,
    drop_unscored,
    label_counts,
    load_corpus,
)

FRACTIONS = {"train": 0.6, "calib": 0.2, "thresh": 0.2}


def test_drop_unscored_removes_minus_one_rows() -> None:
    frame = pd.DataFrame({"id": ["a", "b"], **{label: [0, -1] for label in LABELS}})
    assert drop_unscored(frame)["id"].tolist() == ["a"]


def test_split_is_seeded_and_covers_every_row() -> None:
    a = assign_splits(1000, FRACTIONS, seed=3)
    b = assign_splits(1000, FRACTIONS, seed=3)
    c = assign_splits(1000, FRACTIONS, seed=4)
    assert (a == b).all()
    assert not (a == c).all()
    counts = {name: int((a == name).sum()) for name in SPLIT_NAMES}
    assert counts == {"train": 600, "calib": 200, "thresh": 200}


def test_split_rejects_fractions_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        assign_splits(10, {"train": 0.5, "calib": 0.2, "thresh": 0.2}, seed=0)


@pytest.mark.parametrize("fraction", [-0.1, 0.0, np.nan, np.inf])
def test_split_rejects_nonpositive_or_nonfinite_fractions(fraction: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        assign_splits(10, {**FRACTIONS, "train": fraction}, seed=0)


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_split_rejects_empty_partitions(n: int) -> None:
    with pytest.raises(ValueError, match="at least one row"):
        assign_splits(n, FRACTIONS, seed=0)


@pytest.fixture
def editable_corpus(tmp_path: Path) -> Path:
    pd.DataFrame(
        {
            "id": [f"train-{i}" for i in range(10)],
            "comment_text": ["training text"] * 10,
            **{label: [0, 1] * 5 for label in LABELS},
        }
    ).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"id": ["test-0", "test-1", "test-2"], "comment_text": ["test text"] * 3}).to_csv(
        tmp_path / "test.csv", index=False
    )
    pd.DataFrame(
        {"id": ["test-0", "test-1", "test-2"], **{label: [0, 1, -1] for label in LABELS}}
    ).to_csv(tmp_path / "test_labels.csv", index=False)
    return tmp_path


@pytest.mark.parametrize("name", ["train", "test", "test_labels"])
@pytest.mark.parametrize("issue", ["missing", "duplicate"])
def test_load_corpus_rejects_invalid_ids(editable_corpus: Path, name: str, issue: str) -> None:
    path = editable_corpus / f"{name}.csv"
    frame = pd.read_csv(path)
    frame.loc[1, "id"] = None if issue == "missing" else frame.loc[0, "id"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=f"{issue} ids"):
        load_corpus(editable_corpus, FRACTIONS, seed=1)


@pytest.mark.parametrize("name", ["train", "test", "test_labels"])
def test_load_corpus_rejects_missing_id_column(editable_corpus: Path, name: str) -> None:
    path = editable_corpus / f"{name}.csv"
    pd.read_csv(path).drop(columns="id").to_csv(path, index=False)
    with pytest.raises(KeyError, match="missing the id column"):
        load_corpus(editable_corpus, FRACTIONS, seed=1)


def test_load_corpus_rejects_train_test_overlap(editable_corpus: Path) -> None:
    path = editable_corpus / "train.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "id"] = "test-0"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="overlapping ids"):
        load_corpus(editable_corpus, FRACTIONS, seed=1)


def test_load_corpus_rejects_missing_scored_texts(editable_corpus: Path) -> None:
    path = editable_corpus / "test.csv"
    pd.read_csv(path).iloc[1:].to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing texts for scored"):
        load_corpus(editable_corpus, FRACTIONS, seed=1)


@pytest.mark.parametrize("name", ["train", "test"])
def test_load_corpus_rejects_missing_comment_text(editable_corpus: Path, name: str) -> None:
    path = editable_corpus / f"{name}.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "comment_text"] = None
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing comment_text"):
        load_corpus(editable_corpus, FRACTIONS, seed=1)


@pytest.mark.parametrize("name", ["train", "test_labels"])
@pytest.mark.parametrize("value", [-2.0, 2.0, 0.5, np.nan])
def test_load_corpus_rejects_nonbinary_labels(
    editable_corpus: Path, name: str, value: float
) -> None:
    path = editable_corpus / f"{name}.csv"
    frame = pd.read_csv(path)
    frame["toxic"] = frame["toxic"].astype(float)
    frame.loc[0, "toxic"] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-binary labels"):
        load_corpus(editable_corpus, FRACTIONS, seed=1)


def test_load_corpus_merges_and_filters_test(synthetic_corpus: Path) -> None:
    corpus = load_corpus(synthetic_corpus, FRACTIONS, seed=1)
    assert len(corpus.test) == 1200, "unscored rows must be dropped"
    assert (corpus.test[list(LABELS)].to_numpy() >= 0).all()
    assert set(corpus.train["split"]) == set(SPLIT_NAMES)
    assert set(corpus.file_hashes) == {"train", "test", "test_labels"}
    assert all(len(h) == 64 for h in corpus.file_hashes.values())


def test_test_ids_never_appear_in_training(synthetic_corpus: Path) -> None:
    corpus = load_corpus(synthetic_corpus, FRACTIONS, seed=1)
    assert not set(corpus.test["id"]) & set(corpus.train["id"])


def test_label_counts_reports_rows_and_positives() -> None:
    frame = pd.DataFrame({label: np.array([1, 0, 1]) for label in LABELS})
    counts = label_counts(frame)
    assert counts["rows"] == 3
    assert counts["toxic"] == 2
