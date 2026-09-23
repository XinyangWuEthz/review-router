from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from review_router.metrics import wilson_interval
from review_router.policy import DEFAULT_POLICY_PATH, Policy, load_policy
from review_router.thresholds import (
    TierThresholds,
    apply_rules,
    choose_threshold,
    choose_threshold_by_segment,
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
    """The shipped policy with both tiers on the cumulative rule.

    The subgroup mechanics are independent of the population rule; the
    cumulative rule lets these fixtures set a population threshold without
    filling every declared segment with 30 rows.
    """
    raw = yaml.safe_load(DEFAULT_POLICY_PATH.read_text(encoding="utf-8"))
    raw.pop("tier_selection_rules")
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
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


def test_priority_selection_uses_95_percent_target_without_automatic_actions(
    tmp_path: Path,
) -> None:
    y = np.zeros((100, len(LABELS)), dtype=int)
    y[:95, 0] = 1
    proba = np.zeros_like(y, dtype=float)
    proba[:95, 0] = 0.9
    proba[95:, 0] = 0.8
    signals = {"identity_term_present": np.zeros(100)}
    thresholds = select_thresholds(y, proba, LABELS, _policy_with_subgroup(tmp_path), signals)
    assert thresholds.thresholds["priority_review"]["toxic"] == 0.8
    assert "auto_action" not in thresholds.thresholds
    tiers, _ = model_tier(proba, thresholds, signals)
    assert set(tiers) == {"priority_review"}
    # The shipped segment rule needs every declared segment above the edge to
    # hold 30 rows; here the segments above 0.9 are empty, so toxic gets none.
    shipped = select_thresholds(y, proba, LABELS, load_policy(), signals)
    assert shipped.thresholds["priority_review"]["toxic"] is None
    assert shipped.thresholds["human_review"] == thresholds.thresholds["human_review"]


def test_shipped_segment_rule_stops_below_a_segment_that_misses_the_target() -> None:
    # 40 rows in each declared segment from 0.8 up; only [0.8, 0.9) has 5
    # negatives (35/40 = 0.875 < 0.95). Cumulative precision at 0.8 is
    # 235/240 = 0.979, so the cumulative rule would take 0.8; the shipped
    # segment rule takes 0.9.
    edges = (0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0)
    scores, labels = [], []
    for i in range(len(edges) - 1):
        scores += list(np.linspace(edges[i], edges[i + 1], 40, endpoint=(i == len(edges) - 2)))
        labels += [0] * (5 if i == 0 else 0) + [1] * (35 if i == 0 else 40)
    n = len(scores)
    y = np.zeros((n, len(LABELS)), dtype=int)
    y[:, 0] = labels
    proba = np.zeros((n, len(LABELS)))
    proba[:, 0] = scores
    signals = {"identity_term_present": np.zeros(n)}
    thresholds = select_thresholds(y, proba, LABELS, load_policy(), signals)
    assert thresholds.thresholds["priority_review"]["toxic"] == 0.9
    assert choose_threshold(y[:, 0], proba[:, 0], 0.95, 30) == 0.8


def test_legacy_threshold_map_can_be_replayed_without_renaming_its_tier() -> None:
    thresholds = TierThresholds(
        thresholds={"auto_action": {"toxic": 0.99}, "human_review": {"toxic": 0.9}},
        labels=("toxic",),
    )
    tiers, _ = model_tier(np.array([[0.995], [0.95], [0.1]]), thresholds)
    assert tiers.tolist() == ["auto_action", "human_review", "allow"]


# ---------------------------------------------------------------- segment-agreement rule

EDGES = (0.5, 0.8, 0.9, 1.0)


def _segments(*blocks: tuple[float, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(score, n_rows, n_positive) per block, positives first so y is hand-countable."""
    scores, labels = [], []
    for score, n, positives in blocks:
        scores += [score] * n
        labels += [1] * positives + [0] * (n - positives)
    return np.array(labels), np.array(scores)


def test_segment_rule_ignores_cumulative_carry_from_the_top_segment() -> None:
    # [0.5,0.8): 10 rows, 8 positive (0.80); [0.8,0.9): 10 rows, 7 positive (0.70);
    # [0.9,1.0]: 40 rows, all positive. Cumulative precision at 0.5 is 55/60 = 0.917,
    # so the cumulative rule accepts the lowest score 0.6 (cumulative precision 0.917 at
    # floor 0.9); the segment rule does not.
    y, p = _segments((0.6, 10, 8), (0.85, 10, 7), (0.95, 40, 40))
    assert choose_threshold(y, p, floor=0.9, min_predicted_positives=5) == 0.6
    assert choose_threshold_by_segment(y, p, 0.9, min_rows=5, edges=EDGES) == 0.9


def test_segment_rule_returns_none_when_the_top_segment_is_too_small() -> None:
    y, p = _segments((0.6, 30, 30), (0.85, 30, 30), (0.95, 4, 4))
    assert choose_threshold_by_segment(y, p, 0.9, min_rows=5, edges=EDGES) is None


def test_segment_rule_places_a_score_of_exactly_one_in_the_top_segment() -> None:
    y, p = _segments((0.95, 4, 4), (1.0, 1, 1))
    # Five rows in [0.9, 1.0] including the 1.0; with min_rows 5 the segment fills.
    assert choose_threshold_by_segment(y, p, 0.9, min_rows=5, edges=EDGES) == 0.9
    assert choose_threshold_by_segment(y, p, 0.9, min_rows=6, edges=EDGES) is None


def test_wilson_variant_is_stricter_than_the_point_estimate() -> None:
    # 19 of 20 positive: point estimate 0.95 passes the 0.95 floor, the Wilson
    # 95% lower bound (about 0.76) does not.
    y, p = _segments((0.95, 20, 19))
    assert wilson_interval(19, 20)[0] < 0.95
    assert choose_threshold_by_segment(y, p, 0.95, 10, EDGES) == 0.9
    assert choose_threshold_by_segment(y, p, 0.95, 10, EDGES, use_wilson_lower_bound=True) is None


def test_segment_rule_returns_the_lowest_edge_when_every_segment_qualifies() -> None:
    y, p = _segments((0.6, 10, 10), (0.85, 10, 10), (0.95, 10, 9))
    assert choose_threshold_by_segment(y, p, 0.9, min_rows=5, edges=EDGES) == 0.5


def test_policy_without_selection_rules_reproduces_the_cumulative_thresholds(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(DEFAULT_POLICY_PATH.read_text(encoding="utf-8"))
    raw.pop("tier_selection_rules")
    raw.pop("agreement_segments")
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    policy = load_policy(path)
    assert policy.tier_selection_rules == {
        "human_review": "cumulative_precision",
        "priority_review": "cumulative_precision",
    }
    rng = np.random.default_rng(7)
    n = 3000
    y = (rng.random((n, 6)) < 0.2).astype(int)
    p = np.clip(0.6 * y + rng.normal(0.2, 0.2, (n, 6)), 0, 1)
    signals = {"identity_term_present": (rng.random(n) < 0.1).astype(float)}
    thresholds = select_thresholds(y, p, LABELS, policy, signals)
    min_pos = int(policy.gates["min_predicted_positives_for_precision"])
    for tier, floor in policy.tier_precision_floors.items():
        for j, label in enumerate(LABELS):
            assert thresholds.thresholds[tier][label] == choose_threshold(
                y[:, j], p[:, j], floor, min_pos
            )
    # The shipped policy differs on the priority tier only.
    shipped = select_thresholds(y, p, LABELS, load_policy(), signals)
    assert shipped.thresholds["human_review"] == thresholds.thresholds["human_review"]


def test_shipped_policy_uses_high_risk_edges_for_heavy_labels_only() -> None:
    policy = load_policy()
    rng = np.random.default_rng(1)
    n = 2000
    y = np.zeros((n, 6), dtype=int)
    p = np.zeros((n, 6))
    # threat (weight 10): all rows scoring in [0.3, 0.5) are positive, none above.
    y[:200, 3] = 1
    p[:200, 3] = rng.uniform(0.3, 0.5, 200)
    p[200:, 3] = rng.uniform(0.0, 0.1, n - 200)
    # toxic (weight 1): same shape, but its edges start at 0.5, so nothing qualifies.
    y[:200, 0] = 1
    p[:200, 0] = rng.uniform(0.3, 0.5, 200)
    p[200:, 0] = rng.uniform(0.0, 0.1, n - 200)
    thresholds = select_thresholds(
        y, p, LABELS, policy, {"identity_term_present": np.zeros(n)}, high_risk_min_weight=5.0
    )
    assert thresholds.thresholds["priority_review"]["threat"] is None  # [0.5, 0.8) etc. empty
    # Fill every higher-risk segment above 0.3 so the rule can land on 0.3.
    p[:200, 3] = np.concatenate(
        [
            rng.uniform(0.3, 0.5, 50),
            rng.uniform(0.5, 0.8, 50),
            rng.uniform(0.8, 0.9, 50),
            rng.uniform(0.9, 1.0, 50),
        ]
    )
    thresholds = select_thresholds(
        y, p, LABELS, policy, {"identity_term_present": np.zeros(n)}, high_risk_min_weight=5.0
    )
    assert thresholds.thresholds["priority_review"]["threat"] == 0.3
    assert thresholds.thresholds["priority_review"]["toxic"] is None
