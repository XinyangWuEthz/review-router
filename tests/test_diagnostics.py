"""Hand-computable checks for the development diagnostics in review_router.pipeline.

high_risk_by_band, ranking_diagnostics and cross_fitted_selection are read on
development data; these tests fix their arithmetic on tiny arrays with the
shipped policy weights (threat 10, identity_hate 6, severe_toxic 5, insult 2,
obscene 2, toxic 1) and a high-risk weight of 5.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from review_router.data import LABELS
from review_router.pipeline import (
    _cross_fit_folds,
    _review_workload,
    cross_fitted_selection,
    harm_proxy,
    ranking_diagnostics,
)
from review_router.policy import AgreementSegments, load_policy

POLICY = load_policy()
HIGH_RISK = 5.0
P, H, A = "priority_review", "human_review", "allow"


def _rows(*labels_per_row: tuple[str, ...]) -> np.ndarray:
    y = np.zeros((len(labels_per_row), len(LABELS)), dtype=int)
    for i, labels in enumerate(labels_per_row):
        for label in labels:
            y[i, LABELS.index(label)] = 1
    return y


def test_harm_proxy_is_the_max_true_label_weight() -> None:
    y = _rows(("threat",), ("identity_hate", "toxic"), ("severe_toxic", "toxic"), ("insult",), ())
    assert harm_proxy(y, POLICY).tolist() == [10.0, 6.0, 5.0, 2.0, 0.0]
    assert harm_proxy(np.zeros((0, len(LABELS)), dtype=int), POLICY).shape == (0,)


def test_high_risk_by_band_counts_shares_and_recall() -> None:
    y = _rows(
        ("threat",),  # 10, priority
        ("identity_hate", "toxic"),  # 6, human
        ("severe_toxic", "toxic"),  # 5, allow: high risk let through
        ("insult",),  # 2, human
        (),  # 0, allow
        ("toxic",),  # 1, priority
        (),  # 0, allow
        ("threat", "insult"),  # 10, human
    )
    final = np.array([P, H, A, H, A, P, A, H], dtype=object)
    out = _review_workload(final, y, POLICY, HIGH_RISK)
    assert out["n_requires_human_review"] == 5 and out["precision"] == 1.0
    assert out["high_risk_by_band"] == {
        P: {"n": 2, "n_high_risk": 1, "high_risk_share": 0.5},
        H: {"n": 3, "n_high_risk": 2, "high_risk_share": pytest.approx(2 / 3)},
        A: {"n": 3, "n_high_risk": 1, "high_risk_share": pytest.approx(1 / 3)},
    }
    assert out["n_high_risk_total"] == 4
    assert out["high_risk_recall_into_queue"] == 0.75
    assert out["high_risk_recall_into_priority"] == 0.25
    json.dumps(out, allow_nan=False)


def test_high_risk_by_band_uses_none_for_empty_bands_and_no_high_risk_rows() -> None:
    out = _review_workload(np.array([A, A], dtype=object), _rows((), ("toxic",)), POLICY, HIGH_RISK)
    assert out["high_risk_by_band"][P] == {"n": 0, "n_high_risk": 0, "high_risk_share": None}
    assert out["high_risk_by_band"][A]["high_risk_share"] == 0.0
    assert out["n_high_risk_total"] == 0
    assert out["high_risk_recall_into_queue"] is None
    assert out["high_risk_recall_into_priority"] is None


def _pool() -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    # Six queued rows; high risk (>= 5) are rows 1, 3 and 5; total harm 22.
    return {
        "harm": np.array([0.0, 10.0, 0.0, 6.0, 1.0, 5.0]),
        "priorities": {
            "fifo": np.zeros(6),
            "good": np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]),  # ranks 1, 3, 5 first
            "bad": np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3]),  # ranks 0, 2, 4 first
        },
    }


def test_ranking_diagnostics_top_k_shares_by_hand() -> None:
    out = ranking_diagnostics(_pool(), HIGH_RISK)
    assert out["n_pool"] == 6 and out["n_high_risk"] == 3
    assert out["fractions"] == [0.5, 0.67, 1.0]
    assert set(out["by_key"]) == {"fifo", "good", "bad", "oracle_true_harm"}
    assert [a["k"] for a in out["by_key"]["good"]["at_fraction"]] == [
        math.ceil(0.5 * 6),
        math.ceil(0.67 * 6),
        6,
    ]

    def shares(key: str, field: str) -> list[float]:
        return [a[field] for a in out["by_key"][key]["at_fraction"]]

    # good: top 3 = rows 1, 3, 5 (all high risk, harm 21); top 5 adds rows 4, 2.
    assert shares("good", "high_risk_share_at_k") == pytest.approx([1.0, 1.0, 1.0])
    assert shares("good", "harm_share_at_k") == pytest.approx([21 / 22, 1.0, 1.0])
    # bad: top 3 = rows 0, 2, 4 (harm 1, no high risk); top 5 adds rows 5, 3.
    assert shares("bad", "high_risk_share_at_k") == pytest.approx([0.0, 2 / 3, 1.0])
    assert shares("bad", "harm_share_at_k") == pytest.approx([1 / 22, 12 / 22, 1.0])
    # fifo: constant scores keep pool order, so top 3 = rows 0, 1, 2.
    assert shares("fifo", "high_risk_share_at_k") == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert shares("fifo", "harm_share_at_k") == pytest.approx([10 / 22, 17 / 22, 1.0])
    # The oracle captures every high-risk row at the smallest k that fits them (k = 3).
    assert shares("oracle_true_harm", "high_risk_share_at_k")[0] == 1.0
    assert shares("oracle_true_harm", "harm_share_at_k") == pytest.approx([21 / 22, 1.0, 1.0])
    assert out["by_key"]["good"]["ap_vs_high_risk"] == 1.0
    assert out["by_key"]["bad"]["ap_vs_high_risk"] == pytest.approx((1 / 4 + 2 / 5 + 3 / 6) / 3)
    assert out["by_key"]["fifo"]["ap_vs_high_risk"] == 0.5
    assert "not shippable" in out["note"]
    json.dumps(out, allow_nan=False)


def test_ranking_diagnostics_degenerate_pools() -> None:
    empty = ranking_diagnostics({"harm": np.zeros(0), "priorities": {"fifo": np.zeros(0)}}, 5.0)
    assert empty["n_pool"] == 0 and empty["by_key"] == {}
    clean = ranking_diagnostics({"harm": np.zeros(3), "priorities": {"fifo": np.zeros(3)}}, 5.0)
    entry = clean["by_key"]["fifo"]
    assert entry["ap_vs_high_risk"] is None
    assert [a["harm_share_at_k"] for a in entry["at_fraction"]] == [None, None, None]
    assert [a["high_risk_share_at_k"] for a in entry["at_fraction"]] == [None, None, None]
    all_high = ranking_diagnostics({"harm": np.full(3, 10.0), "priorities": {}}, 5.0)
    assert all_high["by_key"]["oracle_true_harm"]["ap_vs_high_risk"] is None


def test_cross_fit_folds_partition_every_row_deterministically() -> None:
    folds = _cross_fit_folds(23, 5, seed=7)
    assert folds.shape == (23,)
    assert np.bincount(folds, minlength=5).tolist() == [5, 5, 5, 4, 4]
    assert folds.tolist() == _cross_fit_folds(23, 5, seed=7).tolist()
    with pytest.raises(ValueError, match="at least two folds"):
        _cross_fit_folds(23, 1, seed=7)


def _development_pool(n: int = 400, seed: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    prevalence = np.array([0.3, 0.02, 0.2, 0.01, 0.2, 0.02])
    y = (rng.random((n, len(LABELS))) < prevalence).astype(int)
    y[y[:, LABELS.index("severe_toxic")] == 1, LABELS.index("toxic")] = 1
    proba = np.clip(0.6 * y + 0.4 * rng.random((n, len(LABELS))), 0.0, 1.0)
    identity = (rng.random(n) < 0.1).astype(float)
    return y, proba, identity


def test_cross_fitted_selection_is_deterministic_and_covers_every_row() -> None:
    y, proba, identity = _development_pool()
    first = cross_fitted_selection(y, proba, identity, POLICY, high_risk_min_weight=5.0, seed=3)
    second = cross_fitted_selection(y, proba, identity, POLICY, high_risk_min_weight=5.0, seed=3)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["k"] == 5 and len(first["thresholds_per_fold"]) == 5
    assert sum(first["fold_sizes"]) == len(y)
    assert set(first["tiers"]) == {P, H, A}
    assert sum(t["n_predicted_positive"] for t in first["tiers"].values()) == len(y)
    workload = first["review_workload"]
    assert workload["n_total"] == len(y)
    assert set(workload["high_risk_by_band"]) == {P, H, A}
    assert sum(b["n"] for b in workload["high_risk_by_band"].values()) == len(y)
    assert any(t[H]["toxic"] is not None for t in first["thresholds_per_fold"])
    assert "not distribution shift" in first["note"]
    json.dumps(first, allow_nan=False)


def test_cross_fitted_selection_uses_the_configured_high_risk_cutoff() -> None:
    policy = replace(
        POLICY,
        rules=(),
        subgroup_thresholds=(),
        agreement_segments=AgreementSegments(
            edges=(0.5, 0.9, 1.0), high_risk_edges=(0.5, 1.0), min_rows=30
        ),
    )
    y = _rows(*[("toxic",)] * 500)
    proba = np.zeros_like(y, dtype=float)
    proba[:, LABELS.index("toxic")] = 0.8
    # With cutoff 1, toxic uses the high-risk [0.5, 1.0] segment, which
    # qualifies in every fit fold. The ordinary [0.9, 1.0] segment is empty
    # and would disable priority if cross-fitting silently used cutoff 5.
    out = cross_fitted_selection(
        y, proba, np.zeros(len(y)), policy, high_risk_min_weight=1.0, seed=3
    )
    assert all(fold[P]["toxic"] == 0.5 for fold in out["thresholds_per_fold"])
    assert out["review_workload"]["n_priority_review"] == len(y)
