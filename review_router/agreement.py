"""Agreement-by-confidence tables: label agreement per score segment on development data.

The method follows Thomas et al. (arXiv 2406.12800): bin the score, measure
agreement with the human labels per bin, act only in bins where agreement is
high, and report the coverage of every bin. Here agreement is the observed
share of rows in a segment whose label is positive; coverage is the share of
the split's rows the segment holds. Both are read on development splits and
describe those rows only. Nothing here is a population bound, a test result
or a permission to act without human confirmation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from review_router.metrics import wilson_interval
from review_router.policy import AgreementSegments, Policy
from review_router.thresholds import segment_masks

__all__ = ["agreement_table", "agreement_report", "oov_share", "PRIORITY"]

PRIORITY = "priority_review"


def _segment_entry(
    y: np.ndarray,
    score: np.ndarray,
    mask: np.ndarray,
    lo: float | None,
    hi: float,
    n_total: int,
    n_positive_total: int,
    min_rows: int,
) -> dict[str, Any]:
    n = int(mask.sum())
    n_positive = int(y[mask].sum())
    return {
        "lo": lo,
        "hi": hi,
        "n": n,
        "coverage": n / n_total if n_total else 0.0,
        "mean_score": float(score[mask].mean()) if n else None,
        "n_positive": n_positive,
        "agreement": n_positive / n if n else None,
        "agreement_ci95": list(wilson_interval(n_positive, n)) if n else None,
        "recall_share": n_positive / n_positive_total if n_positive_total else None,
        "meets_min_rows": n >= min_rows,
    }


def agreement_table(
    y_binary: np.ndarray, score: np.ndarray, edges: tuple[float, ...], min_rows: int
) -> list[dict[str, Any]]:
    """Per-segment agreement of a binary label with a score.

    One leading bucket (lo None) holds scores below the first edge; then
    [edges[i], edges[i+1]) with the last segment closed at 1.0. agreement is
    the mean label in the segment with a Wilson 95% interval; recall_share is
    the segment's share of all positive rows; meets_min_rows says whether the
    segment has at least min_rows rows. Observed proportions on these rows only.
    """
    y = np.asarray(y_binary).astype(int)
    s = np.asarray(score, dtype=float)
    if y.shape != s.shape:
        raise ValueError(f"y and score must have the same shape, got {y.shape} and {s.shape}")
    n_total = int(len(y))
    n_positive_total = int(y.sum())
    below = s < edges[0]
    rows = [
        _segment_entry(y, s, below, None, edges[0], n_total, n_positive_total, min_rows),
    ]
    for i, mask in enumerate(segment_masks(s, edges)):
        rows.append(
            _segment_entry(y, s, mask, edges[i], edges[i + 1], n_total, n_positive_total, min_rows)
        )
    return rows


def _segments(policy: Policy) -> AgreementSegments:
    segments = policy.agreement_segments
    if segments is None:  # pragma: no cover - Policy fills the default
        segments = AgreementSegments()
    return segments


def _rule_agreement(
    y: np.ndarray,
    proba: np.ndarray,
    labels: tuple[str, ...],
    policy: Policy,
) -> dict[str, Any]:
    """Agreement and coverage of the rows a single-condition '>=' rule on p_<label> matches.

    Rules of another shape (several conditions, other operators or signals)
    are skipped and listed under 'skipped'.
    """
    any_true: np.ndarray = np.asarray(y.any(axis=1))
    n_total = int(len(y))
    out: dict[str, Any] = {}
    skipped: list[str] = []
    for rule in policy.rules:
        if len(rule.conditions) != 1:
            skipped.append(rule.id)
            continue
        cond = rule.conditions[0]
        label = cond.signal.removeprefix("p_")
        if cond.op != ">=" or not cond.signal.startswith("p_") or label not in labels:
            skipped.append(rule.id)
            continue
        j = labels.index(label)
        mask = proba[:, j] >= cond.threshold
        n = int(mask.sum())
        n_label = int(y[mask, j].sum())
        n_any = int(any_true[mask].sum())
        out[rule.id] = {
            "action": rule.action,
            "signal": cond.signal,
            "threshold": cond.threshold,
            "n": n,
            "coverage": n / n_total if n_total else 0.0,
            "n_positive_label": n_label,
            "agreement_label": n_label / n if n else None,
            "agreement_label_ci95": list(wilson_interval(n_label, n)) if n else None,
            "n_positive_any_label": n_any,
            "agreement_any_label": n_any / n if n else None,
        }
    out["skipped"] = skipped
    return out


def agreement_report(
    y: np.ndarray,
    proba: np.ndarray,
    labels: tuple[str, ...],
    policy: Policy,
    high_risk_min_weight: float,
    final_tier: np.ndarray | None = None,
    rule_action: np.ndarray | None = None,
    strata: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Agreement tables for one split.

    per_label[label] is {'edges', 'high_risk_edges' (bool), 'table'}: the
    label's own probability against its own label, on the policy's edges
    (high-risk edges for labels whose severity weight reaches
    high_risk_min_weight); the table is wrapped so the edges it was cut on
    travel with it. pooled_review: max probability over labels against
    'any label positive', the definition both review bands are judged by.
    high_risk_rules: rows matched by each single-condition '>=' rule.
    priority_band (when final_tier is given): the pooled table on the rows
    whose final tier is priority_review, with the count of rule promotions
    and the lowest segment agreement. strata (when given): the pooled table
    per stratum value, e.g. identity_term_present or an OOV tercile.
    """
    segments = _segments(policy)
    min_rows = segments.min_rows
    p_max = proba.max(axis=1)
    any_true = y.any(axis=1).astype(int)
    per_label: dict[str, Any] = {}
    for j, label in enumerate(labels):
        weight = policy.severity_weights.get(label, 0.0)
        edges = segments.edges_for(weight, high_risk_min_weight)
        per_label[label] = {
            "edges": list(edges),
            "high_risk_edges": edges is segments.high_risk_edges,
            "table": agreement_table(y[:, j], proba[:, j], edges, min_rows),
        }
    out: dict[str, Any] = {
        "n_rows": int(len(y)),
        "min_rows": min_rows,
        "edges": list(segments.edges),
        "per_label": per_label,
        "pooled_review": agreement_table(any_true, p_max, segments.edges, min_rows),
        "high_risk_rules": _rule_agreement(y, proba, labels, policy),
    }
    if final_tier is not None:
        in_band = np.asarray(final_tier) == PRIORITY
        table = agreement_table(any_true[in_band], p_max[in_band], segments.edges, min_rows)
        agreements = [row["agreement"] for row in table[1:] if row["n"] > 0]
        out["priority_band"] = {
            "n": int(in_band.sum()),
            "table": table,
            "n_rules_promoted": (
                int((np.asarray(rule_action) == PRIORITY).sum())
                if rule_action is not None
                else None
            ),
            "lowest_segment_agreement": min(agreements) if agreements else None,
            "note": "pooled p_max against any positive label on the final priority band, rule "
            "promotions included; the leading bucket holds rows a rule promoted from below "
            "the first edge",
        }
    if strata is not None:
        out["strata"] = {
            name: {
                str(value): agreement_table(
                    any_true[values == value], p_max[values == value], segments.edges, min_rows
                )
                for value in np.unique(values)
            }
            for name, values in ((name, np.asarray(v)) for name, v in strata.items())
        }
    return out


def oov_share(model: Any, texts: list[str]) -> np.ndarray | None:
    """Per-text share of word-analyzer tokens absent from the fitted word vocabulary.

    Uses model.vectorizer.build_analyzer(), so the tokens are what the word
    block would count (including its n-grams, lowercasing and accent
    stripping). A text with no tokens has share 0.0. Returns None when the
    model has no word vectorizer (a char_wb-only model). No labels are read.
    """
    vectorizer = getattr(model, "vectorizer", None)
    if vectorizer is None:
        return None
    analyze = vectorizer.build_analyzer()
    vocabulary = vectorizer.vocabulary_
    shares = np.zeros(len(texts), dtype=float)
    for i, text in enumerate(texts):
        tokens = analyze(text)
        if tokens:
            unseen = sum(1 for token in tokens if token not in vocabulary)
            shares[i] = unseen / len(tokens)
    return shares
