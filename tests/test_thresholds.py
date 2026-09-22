from __future__ import annotations

import numpy as np

from review_router.policy import load_policy
from review_router.thresholds import (
    TierThresholds,
    apply_rules,
    choose_threshold,
    model_tier,
)

LABELS = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")


def test_choose_threshold_takes_lowest_qualifying_threshold() -> None:
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    y = np.array([1, 1, 1, 1, 0, 0])
    assert choose_threshold(y, p, floor=0.9, min_predicted_positives=2) == 0.6
    assert choose_threshold(y, p, floor=0.8, min_predicted_positives=2) == 0.5


def test_choose_threshold_returns_none_when_floor_is_unreachable() -> None:
    p = np.array([0.9, 0.8, 0.7])
    y = np.array([1, 0, 0])
    assert choose_threshold(y, p, floor=0.9, min_predicted_positives=2) is None
    assert choose_threshold(y, p, floor=0.5, min_predicted_positives=5) is None
    assert choose_threshold(np.zeros(3), p, floor=0.1, min_predicted_positives=1) is None


def test_choose_threshold_honours_floor_under_ties() -> None:
    p = np.array([0.9, 0.9, 0.9, 0.2])
    y = np.array([1, 1, 0, 0])
    # At 0.9 three rows are predicted positive with precision 2/3 < 0.9.
    assert choose_threshold(y, p, floor=0.9, min_predicted_positives=1) is None


def test_model_tier_prefers_auto_action_over_human_review() -> None:
    thresholds = TierThresholds(
        thresholds={
            "auto_action": {label: (0.95 if label == "obscene" else None) for label in LABELS},
            "human_review": {label: (0.5 if label == "toxic" else None) for label in LABELS},
        },
        labels=LABELS,
    )
    proba = np.array(
        [
            [0.9, 0, 0.99, 0, 0, 0],  # auto via obscene
            [0.6, 0, 0.1, 0, 0, 0],  # human via toxic
            [0.1, 0, 0.1, 0, 0, 0],  # allow
        ]
    )
    tiers, triggers = model_tier(proba, thresholds)
    assert tiers.tolist() == ["auto_action", "human_review", "allow"]
    assert triggers[0].tolist() == [False, False, True, False, False, False]
    assert triggers[1].tolist() == [True, False, False, False, False, False]
    assert not triggers[2].any()


def test_apply_rules_reports_matching_ids_and_action() -> None:
    policy = load_policy()
    proba = np.array(
        [
            [0.9, 0, 0, 0.5, 0, 0],  # R101 threat
            [0.1, 0, 0, 0.0, 0, 0],  # nothing
        ]
    )
    actions, ids = apply_rules(proba, LABELS, np.zeros(2), policy)
    assert actions.tolist() == ["human_review", ""]
    assert ids[0] == "R101_rare_high_harm_never_auto" and ids[1] == ""
