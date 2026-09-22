from __future__ import annotations

import numpy as np
import pytest

from review_router.model import ModelConfig, PlattCalibrator, TfidfLogitModel


def test_platt_calibrator_maps_margins_to_probabilities() -> None:
    rng = np.random.default_rng(0)
    margins = rng.normal(size=2000)
    y = (rng.random(2000) < 1 / (1 + np.exp(-(2 * margins - 1)))).astype(int)
    cal = PlattCalibrator().fit(margins, y)
    p = cal.transform(np.array([-3.0, 0.0, 3.0]))
    assert (np.diff(p) > 0).all(), "sigmoid must be monotone"
    assert 0.0 < p[0] < p[1] < p[2] < 1.0
    assert abs(cal.a - 2.0) < 0.4 and abs(cal.b + 1.0) < 0.4


def test_platt_calibrator_with_one_class_keeps_identity() -> None:
    cal = PlattCalibrator().fit(np.array([0.1, 0.2]), np.array([0, 0]))
    assert (cal.a, cal.b) == (1.0, 0.0)


def test_model_learns_cue_words() -> None:
    texts = ["nice work thanks"] * 40 + ["you stupid idiot"] * 40
    y = np.zeros((80, 6), dtype=int)
    y[40:, 0] = 1
    model = TfidfLogitModel(ModelConfig(min_df=1)).fit(texts, y, seed=0).calibrate(texts, y)
    p = model.predict_proba(["stupid idiot", "thanks"])
    assert p.shape == (2, 6)
    assert p[0, 0] > p[1, 0]


@pytest.mark.parametrize("observed_class", [0, 1])
def test_single_class_label_predicts_the_observed_class(observed_class: int) -> None:
    texts = ["alpha beta", "gamma delta", "alpha gamma", "beta delta"]
    y = np.full((4, 6), observed_class, dtype=int)
    y[:, 0] = 0
    y[:2, 0] = 1
    model = TfidfLogitModel(ModelConfig(min_df=1)).fit(texts, y, seed=0).calibrate(texts, y)
    p = model.predict_proba(["alpha beta"])
    assert np.all(np.abs(p[0, 1:] - observed_class) < 1e-6)
