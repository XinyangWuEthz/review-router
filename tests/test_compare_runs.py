from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from review_router.metrics import per_label_metrics, tier_metrics
from review_router.pipeline import _review_workload
from review_router.policy import load_policy

ROOT = Path(__file__).resolve().parent.parent


def _module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CMP = _module(ROOT / "scripts" / "compare_runs.py")

# Two labels are enough to hand-check: weight 1 (toxic-like) and weight 10 (threat-like).
WEIGHTS = np.array([1.0, 0.0, 0.0, 10.0, 0.0, 0.0])


def _y(rows: list[tuple[int, int]]) -> np.ndarray:
    y = np.zeros((len(rows), 6), dtype=int)
    for i, (a, b) in enumerate(rows):
        y[i, 0], y[i, 3] = a, b
    return y


def test_recall_table_counts_flagged_top_tier_and_allow() -> None:
    y = _y([(1, 1), (1, 0), (0, 0), (0, 1), (1, 0)])
    final = np.array(["priority_review", "human_review", "human_review", "allow", "allow"])
    out = CMP.recall_table(y, final, WEIGHTS, 5.0, "priority_review")
    assert out["flagged"] == {"k": 3, "n": 5}
    # Positives: rows 0, 1, 3, 4. Flagged among them: 0, 1.
    assert out["positives_flagged"] == {"k": 2, "n": 4, "share": 0.5}
    assert out["positives_in_top"] == {"k": 1, "n": 4, "share": 0.25}
    # High risk (weight 10 >= 5): rows 0 and 3; row 3 was left in allow.
    assert out["high_risk_flagged"]["k"] == 1
    assert out["high_risk_in_allow"] == {"k": 1, "n": 2, "share": 0.5}
    assert out["harm_total"] == pytest.approx(10 + 1 + 0 + 10 + 1)
    assert out["harm_flagged"] == pytest.approx(11)


def test_captured_at_volume_ranks_by_max_probability_with_stable_ties() -> None:
    y = _y([(0, 0), (1, 0), (0, 1), (1, 0)])
    proba = np.zeros((4, 6))
    proba[:, 0] = [0.9, 0.5, 0.1, 0.5]
    proba[:, 3] = [0.0, 0.0, 0.8, 0.0]
    # Order: row 0 (0.9), row 2 (0.8), then rows 1 and 3 tie at 0.5 -> row 1 first.
    out = CMP.captured_at_volume(proba, y, WEIGHTS, 5.0, 3)
    assert out == {"k": 3, "positives": 2, "high_risk": 1, "harm": pytest.approx(11.0)}


def test_equal_input_loads_invert_the_reference_queue_fraction() -> None:
    assert CMP.equal_input_loads(0.1, [60.0, 120.0]) == pytest.approx([600.0, 1200.0])


def test_paired_difference_and_standard_error() -> None:
    out = CMP._paired([3.0, 5.0, 7.0], [1.0, 2.0, 3.0])
    # Differences 2, 3, 4: mean 3, sample SD 1, SE 1/sqrt(3).
    assert out["mean"] == pytest.approx(3.0)
    assert out["se"] == pytest.approx(1 / np.sqrt(3))
    assert out["valid_n"] == 3


def test_undefined_metrics_do_not_become_zero_or_mismatched_seed_pairs() -> None:
    mean = CMP._mean_std([0.25, None, 0.75])
    assert mean == {"mean": 0.5, "std": pytest.approx(np.sqrt(0.125)), "valid_n": 2}
    assert CMP._mean_std([None, None]) == {"mean": None, "std": None, "valid_n": 0}
    # Only the first seed has a defined ratio in both runs; do not pair seed 2 with seed 3.
    assert CMP._paired([1.0, None, 0.5], [0.25, 0.8, None]) == {
        "mean": 0.75,
        "se": None,
        "valid_n": 1,
    }


def _stream_item(tmp_path: Path) -> dict[str, Any]:
    import pandas as pd

    (tmp_path / "policy.yaml").write_text((ROOT / "review_router" / "policy.yaml").read_text())
    # Row 0: threat, priority. Row 1: toxic, human. Row 2: threat, left in allow. Row 3: clean.
    y = _y([(0, 1), (1, 0), (0, 1), (0, 0)])
    pred = pd.DataFrame({"id": ["a", "b", "c", "d"]})
    for j, lb in enumerate(CMP.LABELS):
        pred[f"y_{lb}"] = y[:, j]
        pred[f"p_{lb}"] = [0.9, 0.6, 0.1, 0.0] if j == 0 else [0.0] * 4
    pred["final_tier"] = ["priority_review", "human_review", "allow", "allow"]
    item = {"dir": tmp_path, "run": "t", "label": "t", "pred": pred,
            "config": {"high_risk_min_weight": 5}}  # fmt: skip
    CMP._prepare_stream(item)
    return item


def test_stream_counts_high_risk_left_in_allow_as_not_reviewed(tmp_path: Path) -> None:
    item = _stream_item(tmp_path)
    cfg = CMP.SimConfig(reviewers=1, handle_minutes=2.0, horizon_hours=1.0)
    # Each comment arrives once; row 0 and row 1 are reviewed, rows 2 and 3 are not admitted.
    out = CMP._stream_metrics(
        item, np.array([0, 1, 2, 3]), np.array([0.0, 1.0, 2.0, 3.0]), cfg, "priority", 1
    )
    assert out["review_arrivals"] == 2
    assert out["high_risk_in_stream"] == 2
    assert out["high_risk_left_in_allow"] == 1
    assert out["high_risk_unhandled_in_queue"] == 0
    assert out["high_risk_not_reviewed"] == 1
    assert out["positives_completed_per_hour"] == 2  # both reviewed rows are positive, 1 hour
    assert out["harm_handled_per_hour"] == pytest.approx(10 + 1)


def test_word_queue_and_all_allow_candidate_complete_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    word = _stream_item(tmp_path)
    word["label"] = "word"
    word["model"] = {"analyzer": "word"}
    word["config"]["sim"] = {"reviewers": 1, "handle_minutes": 2.0, "horizon_hours": 1.0}
    candidate = deepcopy(word)
    candidate["label"] = "jev"
    candidate["model"] = {"kind": "external_scores", "model_name": "jev-1.13.0"}
    candidate["pred"]["final_tier"] = "allow"
    candidate["pred"][[f"p_{label}" for label in CMP.LABELS]] = 0.0
    items = [word, candidate]

    # Build real metric summaries from the existing four-comment synthetic fixture.
    # Four labels have no positive examples, so their AP/AUC are also undefined.
    for item in items:
        proba, y, final = CMP.arrays(item["pred"])
        tiers = tier_metrics(final, y.any(axis=1), ("priority_review", "human_review"), 1)
        thresholds = {tier: dict.fromkeys(CMP.LABELS) for tier in tiers}
        item["report"] = {
            "tiers": tiers,
            "threshold_selection": {"tiers": tiers},
            "per_label": per_label_metrics(y, proba, CMP.LABELS, thresholds),
            "review_workload": _review_workload(final, y, load_policy(), 5),
        }
        item["manifest"] = {"git_commit": "abc12345", "git_dirty": False}

    monkeypatch.setattr(CMP, "EQUAL_INPUT_SEEDS", tuple(range(1, 8)))
    comparison = CMP.shared_stream(items, [20.0])
    # The shipped severity ordering and the historical priority/FIFO keys all
    # remain available, including for a candidate with an empty review queue.
    strategies = {"severity", "priority", "fifo"}
    for label in ("word", "jev"):
        by_input = comparison["runs"][label]["by_input"]["20.0"]
        assert set(by_input) == strategies | {"expected_review_load_per_hour"}
    assert set(comparison["differences_vs_reference"]["jev"]["20.0"]) == strategies
    for strategy in strategies:
        assert comparison["runs"]["jev"]["by_input"]["20.0"][strategy][
            "review_arrivals"
        ]["mean"] == 0
    empty = comparison["runs"]["jev"]["by_input"]["20.0"]["priority"]
    word_metrics = comparison["runs"]["word"]["by_input"]["20.0"]["priority"]
    assert word_metrics["review_arrivals"]["mean"] > 0
    assert word_metrics["completion_ratio"]["valid_n"] == 7
    assert empty["review_arrivals"]["mean"] == 0
    assert empty["completion_ratio"] == {"mean": None, "std": None, "valid_n": 0}
    assert empty["high_risk_not_reviewed"] == empty["high_risk_in_stream"]
    assert empty["harm_handled_per_hour"]["mean"] == 0
    difference = comparison["differences_vs_reference"]["jev"]["20.0"]["priority"]
    assert difference["completion_ratio"] == {"mean": None, "se": None, "valid_n": 0}
    assert difference["high_risk_not_reviewed"]["mean"] > 0
    assert difference["high_risk_not_reviewed"]["valid_n"] == 7
    json.dumps(comparison, allow_nan=False)

    markdown = CMP.to_markdown([CMP.summarize(item) for item in items], {"equal_input": comparison})
    assert "| jev | jev-1.13.0 |" in markdown
    assert "| jev | 0 | 0.0000 | n/a [n/a, n/a] |" in markdown
    assert "| severe_toxic | n/a | n/a | n/a | n/a |" in markdown
    assert "| severity |" in markdown
    assert "n/a (n=0)" in markdown
    assert "nan" not in markdown
