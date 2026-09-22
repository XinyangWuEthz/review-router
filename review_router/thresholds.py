"""Threshold selection from empirical precision targets, and review tier routing.

A tier's threshold is the lowest probability at which the tier's precision
floor still holds on the threshold-selection split, subject to a minimum count
of predicted positives. Lowest threshold means maximum coverage; coverage is
the dependent variable. If no threshold qualifies the tier is disabled for
that label (threshold None), which is the expected outcome for rare labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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

# Model-driven review tiers, highest priority first. `allow` is the fall-through.
TIER_ORDER: tuple[str, ...] = ("priority_review", "human_review")
_LEGACY_TIER_ORDER: tuple[str, ...] = ("auto_action", "human_review")


def _subgroup_mask(
    signals: dict[str, np.ndarray] | None, signal: str, n: int
) -> np.ndarray:
    if signals is None or signal not in signals:
        raise KeyError(f"subgroup threshold on {signal!r} declared but no such signal was given")
    values = np.asarray(signals[signal])
    if values.shape != (n,):
        raise ValueError(f"subgroup signal {signal!r} must have shape ({n},), got {values.shape}")
    if not np.isin(values, [0, 1]).all():
        raise ValueError(f"subgroup signal {signal!r} must contain only binary values 0 or 1")
    return values.astype(bool)


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
    """thresholds[tier][label] -> probability threshold, or None when disabled.

    subgroup[tier][signal][label] -> the threshold chosen on the rows where
    that signal is 1. A row in the subgroup must clear the stricter of the
    population threshold and the subgroup threshold; None on the subgroup
    disables the tier for those rows on that label.
    """

    thresholds: dict[str, dict[str, float | None]]
    labels: tuple[str, ...]
    subgroup: dict[str, dict[str, dict[str, float | None]]] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {tier: dict(per_label) for tier, per_label in self.thresholds.items()}
        if self.subgroup:
            out["subgroup"] = {
                tier: {signal: dict(per_label) for signal, per_label in per_signal.items()}
                for tier, per_signal in self.subgroup.items()
            }
        return out


def select_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    labels: tuple[str, ...],
    policy: Policy,
    signals: dict[str, np.ndarray] | None = None,
) -> TierThresholds:
    """Population thresholds per tier and label, plus the declared subgroup thresholds.

    A subgroup threshold is chosen on the subgroup rows at or above the
    population threshold, from the same floor and minimum count, so it is
    never below the population threshold. A slice without a qualifying
    empirical threshold gets no tier on that label. This selection criterion
    does not certify a population precision bound.

    Targets are fitted independently for each label and subgroup. Combining
    labels or intersecting overlapping subgroups can change routed precision
    and counts; evaluate those routed sets separately.
    """
    min_pos = int(policy.gates.get("min_predicted_positives_for_precision", 30))
    thresholds: dict[str, dict[str, float | None]] = {}
    order = TIER_ORDER if policy.decision_mode == "human_confirmation" else _LEGACY_TIER_ORDER
    for tier in order:
        floor = policy.tier_precision_floors[tier]
        thresholds[tier] = {
            label: choose_threshold(y_true[:, j], proba[:, j], floor, min_pos)
            for j, label in enumerate(labels)
        }
    subgroup: dict[str, dict[str, dict[str, float | None]]] = {}
    for declared in policy.subgroup_thresholds:
        mask = _subgroup_mask(signals, declared.signal, len(proba))
        floor = policy.tier_precision_floors[declared.tier]
        per_label: dict[str, float | None] = {}
        for j, label in enumerate(labels):
            population = thresholds[declared.tier][label]
            if population is None:
                per_label[label] = None
                continue
            # Evaluate the slice where it is routed: at or above the population
            # threshold. A slice threshold below it would never bind, and the
            # minimum count must hold at the operative threshold.
            eligible = mask & (proba[:, j] >= population)
            per_label[label] = choose_threshold(
                y_true[eligible, j], proba[eligible, j], floor, min_pos
            )
        subgroup.setdefault(declared.tier, {})[declared.signal] = per_label
    return TierThresholds(thresholds=thresholds, labels=labels, subgroup=subgroup)


def model_tier(
    proba: np.ndarray,
    thresholds: TierThresholds,
    signals: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row model tier and the boolean (n, n_labels) mask of triggering labels.

    priority_review if any label meets its priority threshold, else human_review
    if any label meets its human threshold, else allow. Neither review tier
    authorizes an action without human confirmation. Historical threshold maps
    retain their auto_action name when replayed explicitly. Rows in a declared
    subgroup must also meet that subgroup's threshold on the label.
    """
    n = proba.shape[0]
    order = TIER_ORDER if "priority_review" in thresholds.thresholds else _LEGACY_TIER_ORDER
    if set(thresholds.thresholds) != set(order):
        raise ValueError("thresholds must contain exactly one supported pair of review tiers")
    group_masks = {
        signal: _subgroup_mask(signals, signal, n)
        for per_signal in thresholds.subgroup.values()
        for signal in per_signal
    }
    tiers = np.full(n, "allow", dtype=object)
    triggers = np.zeros(proba.shape, dtype=bool)
    for tier in reversed(order):  # human_review first, the higher tier overrides
        mask = np.zeros(proba.shape, dtype=bool)
        for j, label in enumerate(thresholds.labels):
            t = thresholds.thresholds[tier][label]
            if t is None:
                continue
            hit = proba[:, j] >= t
            for signal, per_label in thresholds.subgroup.get(tier, {}).items():
                in_group = group_masks[signal]
                sub_t = per_label[label]
                if sub_t is None:
                    hit &= ~in_group
                else:
                    hit &= ~in_group | (proba[:, j] >= max(t, sub_t))
            mask[:, j] = hit
        hit_any = mask.any(axis=1)
        tiers[hit_any] = tier
        triggers[hit_any] = mask[hit_any]
    return tiers, triggers


def apply_rules(
    proba: np.ndarray,
    labels: tuple[str, ...],
    identity: np.ndarray,
    policy: Policy,
) -> tuple[np.ndarray, list[str]]:
    """Evaluate policy rules per row. Returns the rule action ('' if none) and matched ids.

    Rules have priority over the model tier: a matching rule's action replaces
    it. The shipped R101 rule requests priority human review when p_threat >= 0.30.
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
