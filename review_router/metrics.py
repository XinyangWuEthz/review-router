"""Evaluation metrics: per-label quality, tier precision with intervals.

Review precision is a diagnostic, reported with a Wilson interval and as n/a
below the minimum count. It does not authorize automated enforcement.
"""

from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

__all__ = ["wilson_interval", "per_label_metrics", "tier_metrics"]


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _binary_at(y: np.ndarray, p: np.ndarray, threshold: float | None) -> dict[str, Any]:
    if threshold is None:
        return {"threshold": None, "n_predicted_positive": 0, "precision": None, "recall": None}
    pred = p >= threshold
    n_pred = int(pred.sum())
    tp = int((pred & (y == 1)).sum())
    positives = int((y == 1).sum())
    return {
        "threshold": threshold,
        "n_predicted_positive": n_pred,
        "precision": (tp / n_pred) if n_pred else None,
        "recall": (tp / positives) if positives else None,
    }


def per_label_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    labels: tuple[str, ...],
    thresholds: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, Any]]:
    """AP and ROC-AUC plus precision/recall at the supplied tier thresholds.

    Current maps produce at_priority_review; historical maps retain
    at_auto_action so saved version-1 reports can still be read and compared.
    """
    out: dict[str, dict[str, Any]] = {}
    for j, label in enumerate(labels):
        y = y_true[:, j]
        p = proba[:, j]
        positives = int(y.sum())
        entry: dict[str, Any] = {
            "positives": positives,
            "negatives": int(len(y) - positives),
            "average_precision": float(average_precision_score(y, p)) if positives else None,
            "roc_auc": float(roc_auc_score(y, p)) if 0 < positives < len(y) else None,
        }
        for tier, per_label in thresholds.items():
            entry[f"at_{tier}"] = _binary_at(y, p, per_label[label])
        out[label] = entry
    return out


def tier_metrics(
    final_tier: np.ndarray,
    correct: np.ndarray,
    tiers: tuple[str, ...],
    min_predicted_positives: int,
) -> dict[str, dict[str, Any]]:
    """Count, coverage, precision and Wilson 95% interval per tier.

    `correct` is the per-row correctness under the tier's own definition; it is
    only read for rows routed to a non-allow tier.
    """
    n = len(final_tier)
    out: dict[str, dict[str, Any]] = {}
    for tier in tiers:
        mask = final_tier == tier
        count = int(mask.sum())
        entry: dict[str, Any] = {
            "n_predicted_positive": count,
            "coverage": count / n if n else 0.0,
            "precision": None,
            "precision_ci95": None,
            "precision_note": None,
        }
        if tier == "allow":
            entry["precision_note"] = "allow is the fall-through; see false-negative counts"
        elif count == 0:
            entry["precision_note"] = "n/a: tier never fired"
        elif count < min_predicted_positives:
            entry["precision_note"] = (
                f"n/a: fired {count}x, below the minimum of {min_predicted_positives}"
            )
        else:
            successes = int(correct[mask].sum())
            entry["precision"] = successes / count
            entry["precision_ci95"] = list(wilson_interval(successes, count))
        out[tier] = entry
    return out
