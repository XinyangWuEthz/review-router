from __future__ import annotations

import numpy as np
import pytest

from review_router.metrics import per_label_metrics, tier_metrics, wilson_interval


def test_wilson_interval_thirty_of_thirty_does_not_prove_95_percent() -> None:
    low, high = wilson_interval(30, 30)
    assert high == 1.0
    assert low < 0.95, "30/30 is compatible with precision well below 0.95"


def test_wilson_interval_brackets_the_point_estimate() -> None:
    low, high = wilson_interval(45, 50)
    assert low < 0.9 < high


def test_tier_metrics_marks_small_counts_as_not_available() -> None:
    final = np.array(["priority_review"] * 5 + ["human_review"] * 40 + ["allow"] * 55, dtype=object)
    correct = np.array([True] * 5 + [True] * 36 + [False] * 4 + [False] * 55)
    out = tier_metrics(final, correct, ("priority_review", "human_review", "allow"), 30)
    assert out["priority_review"]["precision"] is None
    assert "below the minimum" in out["priority_review"]["precision_note"]
    assert out["human_review"]["precision"] == pytest.approx(0.9)
    assert out["human_review"]["coverage"] == pytest.approx(0.4)
    assert out["allow"]["precision"] is None


@pytest.mark.parametrize("higher_tier", ["priority_review", "auto_action"])
def test_per_label_metrics_reports_ap_and_threshold_stats(higher_tier: str) -> None:
    y = np.array([[1], [0], [1], [0]])
    p = np.array([[0.9], [0.8], [0.7], [0.1]])
    out = per_label_metrics(
        y, p, ("toxic",), {"human_review": {"toxic": 0.7}, higher_tier: {"toxic": None}}
    )
    m = out["toxic"]
    assert m["positives"] == 2
    assert 0 < m["average_precision"] <= 1
    assert m["at_human_review"]["n_predicted_positive"] == 3
    assert m["at_human_review"]["recall"] == 1.0
    assert m[f"at_{higher_tier}"]["threshold"] is None
