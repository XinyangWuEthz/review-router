from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_jev_experiment import calibrate_scores, paired_capture, sample_frame


def test_cohort_selection_ignores_labels_and_input_permutation() -> None:
    frame = pd.DataFrame({"id": [str(i) for i in range(50)], "toxic": [0] * 50})
    first = sample_frame(frame, 12, 42, "test")
    changed = frame.sample(frac=1, random_state=3).copy()
    changed["toxic"] = 1
    second = sample_frame(changed, 12, 42, "test")
    assert set(first.id) == set(second.id)
    assert len(first) == 12
    assert first.index.is_monotonic_increasing
    assert set(first.id) != set(sample_frame(frame, 12, 43, "test").id)


@pytest.mark.parametrize("n", [0, -1, 51, True])
def test_cohort_selection_rejects_invalid_sizes(n: int) -> None:
    with pytest.raises(ValueError, match="sample size"):
        sample_frame(pd.DataFrame({"id": range(50)}), n, 42, "test")


def test_calibration_improves_systematically_wrong_scale_on_separate_rows() -> None:
    # Both classes are separable, but raw scores are all much too high.
    calibration = np.tile(np.array([0.8] * 30 + [0.9] * 30)[:, None], (1, 6))
    y = np.tile(np.array([0] * 30 + [1] * 30)[:, None], (1, 6))
    target = np.tile(np.array([0.81, 0.89])[:, None], (1, 6))
    calibrated, metadata = calibrate_scores(calibration, y, {"test": target})
    assert (calibrated["test"][0] < 0.1).all()
    assert (calibrated["test"][1] > 0.9).all()
    assert metadata["labels"]["toxic"]["status"] == "fitted"


def test_single_class_calibration_keeps_raw_scores_and_marks_limitation() -> None:
    calibrated, metadata = calibrate_scores(
        np.full((20, 6), 0.8), np.zeros((20, 6), dtype=int),
        {"test": np.tile([0, 0.2, 0.4, 0.6, 0.8, 1], (3, 1))},
    )
    np.testing.assert_array_equal(calibrated["test"][0], [0, 0.2, 0.4, 0.6, 0.8, 1])
    assert "one class" in metadata["labels"]["threat"]["status"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_probabilities_cannot_enter_calibration(value: float) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        calibrate_scores(np.full((20, 6), value), np.zeros((20, 6)), {})


def test_paired_bootstrap_has_zero_difference_for_identical_rankings() -> None:
    y = np.zeros((40, 6), dtype=int)
    y[:10, 3] = 1
    proba = np.zeros((40, 6))
    proba[:, 3] = np.linspace(1, 0, 40)
    result = paired_capture(y, proba, proba.copy(), np.array([1, 5, 1, 10, 1, 6]),
                            5, 10, 50, 1)
    assert result["baseline_captured"] == 10
    assert result["jev_captured"] == 10
    assert result["ci95"] == [0, 0]
    assert result["difference_jev_minus_baseline"] == 0
