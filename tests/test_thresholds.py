from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from review_router.policy import DEFAULT_POLICY_PATH, Policy, load_policy
from review_router.thresholds import (
    TierThresholds,
    apply_rules,
    choose_threshold,
    model_tier,
    select_thresholds,
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


def test_model_tier_prefers_priority_review_over_human_review() -> None:
    thresholds = TierThresholds(
        thresholds={
            "priority_review": {label: (0.95 if label == "obscene" else None) for label in LABELS},
            "human_review": {label: (0.5 if label == "toxic" else None) for label in LABELS},
        },
        labels=LABELS,
    )
    proba = np.array(
        [
            [0.9, 0, 0.99, 0, 0, 0],  # priority via obscene
            [0.6, 0, 0.1, 0, 0, 0],  # human via toxic
            [0.1, 0, 0.1, 0, 0, 0],  # allow
        ]
    )
    tiers, triggers = model_tier(proba, thresholds)
    assert tiers.tolist() == ["priority_review", "human_review", "allow"]
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
    assert actions.tolist() == ["priority_review", ""]
    assert ids[0] == "R101_high_risk_priority" and ids[1] == ""


def _policy_with_subgroup(tmp_path: Path) -> Policy:
    path = tmp_path / "policy.yaml"
    path.write_text(DEFAULT_POLICY_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return load_policy(path)


def test_subgroup_threshold_is_stricter_and_disables_when_too_small(tmp_path: Path) -> None:
    policy = _policy_with_subgroup(tmp_path)
    assert [(d.tier, d.signal) for d in policy.subgroup_thresholds] == [
        ("priority_review", "identity_term_present")
    ]
    rng = np.random.default_rng(0)
    n = 4000
    y = np.zeros((n, 6), dtype=int)
    p = np.zeros((n, 6))
    # toxic: clean separation in the population, but the subgroup (last 300
    # rows) carries errors at high scores so its own threshold must sit higher.
    y[:, 0] = rng.random(n) < 0.3
    p[:, 0] = np.where(y[:, 0] == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.5, n))
    group = np.zeros(n, dtype=bool)
    group[-300:] = True
    p[-300:, 0] = np.where(
        y[-300:, 0] == 1, rng.uniform(0.9, 1.0, 300), rng.uniform(0.7, 0.93, 300)
    )
    signals = {"identity_term_present": group.astype(float)}
    thresholds = select_thresholds(y, p, LABELS, policy, signals)
    pop_t = thresholds.thresholds["priority_review"]["toxic"]
    sub_t = thresholds.subgroup["priority_review"]["identity_term_present"]["toxic"]
    assert pop_t is not None and sub_t is not None and sub_t > pop_t
    # Selection count where routed: at least 30 slice rows at or above the slice threshold.
    assert int((group & (p[:, 0] >= sub_t)).sum()) >= 30
    tiers, _ = model_tier(p, thresholds, signals)
    in_band = group & (p[:, 0] >= pop_t) & (p[:, 0] < sub_t)
    assert in_band.any()
    assert (tiers[in_band] != "priority_review").all()
    assert (tiers[~group & (p[:, 0] >= pop_t)] == "priority_review").all()
    assert "subgroup" in thresholds.as_json()

    # A subgroup slice below the minimum count disables the tier for its rows.
    tiny = TierThresholds(
        thresholds={
            "priority_review": {"toxic": 0.5, **{label: None for label in LABELS[1:]}},
            "human_review": {label: None for label in LABELS},
        },
        labels=LABELS,
        subgroup={"priority_review": {"identity_term_present": {label: None for label in LABELS}}},
    )
    proba = np.array([[0.9, 0, 0, 0, 0, 0], [0.9, 0, 0, 0, 0, 0]])
    tiers, _ = model_tier(proba, tiny, {"identity_term_present": np.array([1.0, 0.0])})
    assert tiers.tolist() == ["allow", "priority_review"]


def test_missing_subgroup_signal_is_an_error(tmp_path: Path) -> None:
    policy = _policy_with_subgroup(tmp_path)
    y = np.zeros((10, 6), dtype=int)
    with pytest.raises(KeyError, match="identity_term_present"):
        select_thresholds(y, np.zeros((10, 6)), LABELS, policy, None)


@pytest.mark.parametrize(
    "signal",
    [
        np.array(1),
        np.ones(1),
        np.ones((2, 1)),
        np.array([1.0, np.nan]),
        np.array([1.0, -1.0]),
        np.array([1.0, 0.5]),
    ],
)
def test_subgroup_signals_reject_broadcasting_and_nonbinary_values(signal: np.ndarray) -> None:
    policy = load_policy()
    y = np.zeros((2, len(LABELS)), dtype=int)
    proba = np.zeros_like(y, dtype=float)
    signals = {"identity_term_present": signal}
    with pytest.raises(ValueError, match="subgroup signal"):
        select_thresholds(y, proba, LABELS, policy, signals)

    thresholds = TierThresholds(
        thresholds={tier: dict.fromkeys(LABELS) for tier in ("priority_review", "human_review")},
        labels=LABELS,
        subgroup={"priority_review": {"identity_term_present": dict.fromkeys(LABELS)}},
    )
    with pytest.raises(ValueError, match="subgroup signal"):
        model_tier(proba, thresholds, signals)


def test_routing_requires_subgroup_signal_even_when_all_thresholds_are_disabled() -> None:
    thresholds = TierThresholds(
        thresholds={tier: dict.fromkeys(LABELS) for tier in ("priority_review", "human_review")},
        labels=LABELS,
        subgroup={"priority_review": {"identity_term_present": dict.fromkeys(LABELS)}},
    )
    with pytest.raises(KeyError, match="identity_term_present"):
        model_tier(np.zeros((2, len(LABELS))), thresholds)


def test_priority_selection_uses_95_percent_target_without_automatic_actions() -> None:
    y = np.zeros((100, len(LABELS)), dtype=int)
    y[:95, 0] = 1
    proba = np.zeros_like(y, dtype=float)
    proba[:95, 0] = 0.9
    proba[95:, 0] = 0.8
    signals = {"identity_term_present": np.zeros(100)}
    thresholds = select_thresholds(y, proba, LABELS, load_policy(), signals)
    assert thresholds.thresholds["priority_review"]["toxic"] == 0.8
    assert "auto_action" not in thresholds.thresholds
    tiers, _ = model_tier(proba, thresholds, signals)
    assert set(tiers) == {"priority_review"}


def test_legacy_threshold_map_can_be_replayed_without_renaming_its_tier() -> None:
    thresholds = TierThresholds(
        thresholds={"auto_action": {"toxic": 0.99}, "human_review": {"toxic": 0.9}},
        labels=("toxic",),
    )
    tiers, _ = model_tier(np.array([[0.995], [0.95], [0.1]]), thresholds)
    assert tiers.tolist() == ["auto_action", "human_review", "allow"]
