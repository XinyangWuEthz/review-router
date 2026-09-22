"""Threshold selection from precision floors, and tier routing.

A tier's threshold is the lowest probability at which the tier's precision
floor still holds on the threshold-selection split, subject to a minimum count
of predicted positives. Lowest threshold means maximum coverage; coverage is
the dependent variable. If no threshold qualifies the tier is disabled for
that label (threshold None), which is the expected outcome for rare labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from review_router.policy import Policy

__all__ = [
    "TIER_ORDER",
    "choose_threshold",
    "TierThresholds",
    "select_thresholds",
    "model_tier",
    "apply_rules",
]

# Model-driven tiers, most severe first. `allow` is the fall-through.
TIER_ORDER: tuple[str, ...] = ("auto_action", "human_review")


def choose_threshold(
    y_true: np.ndarray, p: np.ndarray, floor: float, min_predicted_positives: int
) -> float | None:
    """Lowest threshold t such that precision(p >= t) >= floor and count(p >= t) >= min.

    Evaluated over the distinct score values so tied scores are handled by the
    same `>=` rule that routing uses.
    """
    y = np.asarray(y_true).astype(int)
    scores = np.asarray(p, dtype=float)
    if y.sum() == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    cum_tp = np.cumsum(y[order])
    counts = np.arange(1, len(scores) + 1)
    # Position of the last occurrence of each distinct score, descending.
    last_of_value = np.flatnonzero(np.diff(sorted_scores, append=-np.inf) != 0)
    best: float | None = None
    for k in last_of_value:
        n = int(counts[k])
        precision = cum_tp[k] / n
        if n >= min_predicted_positives and precision >= floor:
            best = float(sorted_scores[k])
        # Keep scanning: precision is not monotone, a lower threshold may still qualify.
    return best


@dataclass(frozen=True)
class TierThresholds:
    """thresholds[tier][label] -> probability threshold, or None when disabled."""

    thresholds: dict[str, dict[str, float | None]]
    labels: tuple[str, ...]

    def as_json(self) -> dict[str, dict[str, float | None]]:
        return {tier: dict(per_label) for tier, per_label in self.thresholds.items()}


def select_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    labels: tuple[str, ...],
    policy: Policy,
) -> TierThresholds:
    min_pos = int(policy.gates.get("min_predicted_positives_for_precision", 30))
    thresholds: dict[str, dict[str, float | None]] = {}
    for tier in TIER_ORDER:
        floor = policy.tier_precision_floors[tier]
        thresholds[tier] = {
            label: choose_threshold(y_true[:, j], proba[:, j], floor, min_pos)
            for j, label in enumerate(labels)
        }
    return TierThresholds(thresholds=thresholds, labels=labels)


def model_tier(proba: np.ndarray, thresholds: TierThresholds) -> tuple[np.ndarray, np.ndarray]:
    """Per-row model tier and the boolean (n, n_labels) mask of triggering labels.

    auto_action if any label meets its auto threshold, else human_review if any
    label meets its human threshold, else allow.
    """
    n = proba.shape[0]
    tiers = np.full(n, "allow", dtype=object)
    triggers = np.zeros(proba.shape, dtype=bool)
    for tier in reversed(TIER_ORDER):  # human_review first, auto_action overrides
        mask = np.zeros(proba.shape, dtype=bool)
        for j, label in enumerate(thresholds.labels):
            t = thresholds.thresholds[tier][label]
            if t is not None:
                mask[:, j] = proba[:, j] >= t
        hit = mask.any(axis=1)
        tiers[hit] = tier
        triggers[hit] = mask[hit]
    return tiers, triggers


def apply_rules(
    proba: np.ndarray,
    labels: tuple[str, ...],
    identity: np.ndarray,
    policy: Policy,
) -> tuple[np.ndarray, list[str]]:
    """Evaluate policy rules per row. Returns the rule action ('' if none) and matched ids.

    Rules have priority over the model tier: a matching rule's action replaces
    it. R101 exists precisely so that threat is never auto-actioned.
    """
    n = proba.shape[0]
    actions = np.full(n, "", dtype=object)
    matched: list[str] = []
    index = {label: j for j, label in enumerate(labels)}
    p_max = proba.max(axis=1)
    for i in range(n):
        signals: dict[str, float] = {f"p_{label}": float(proba[i, j]) for label, j in index.items()}
        signals["p_max"] = float(p_max[i])
        signals["identity_term_present"] = float(identity[i])
        ids = [r.id for r in policy.rules if r.matches(signals)]
        matched.append(";".join(ids))
        if ids:
            actions[i] = max(
                (r.action for r in policy.rules if r.id in ids),
                key=lambda action: policy.tiers[action],
            )
    return actions, matched
