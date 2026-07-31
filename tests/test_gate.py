"""Regression gates.

The gate is an ordinary pytest file so CI is one `pytest -q` job. Gates whose
floor is still null in policy.yaml SKIP rather than pass — a floor of 0.0 that
nothing can fail is worse than no gate, because it reads green.

Every assertion prints the whole stats dict, so a red CI tells you the new
number immediately instead of just which boolean flipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from review_router.policy import load_policy

POLICY = load_policy()
GATES = POLICY.gates
REPORT_ENV = "REVIEW_ROUTER_EVAL_REPORT"


def _load_report(kind: str) -> dict[str, Any]:
    """Load the evaluation report, or skip if no run has produced one yet."""
    path = os.environ.get(REPORT_ENV)
    if not path or not Path(path).is_file():
        pytest.skip(
            f"no evaluation report at ${REPORT_ENV}; "
            "run scripts/run_pipeline.py and re-run the gates"
        )
    report: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    if kind not in report:
        pytest.skip(f"report has no {kind!r} section yet")
    return report


def _floor(value: float | None, name: str) -> float:
    if value is None:
        pytest.skip(f"{name} floor is null in policy.yaml — set it after the first run")
    return value


def test_per_label_average_precision() -> None:
    """Gate per label, never on a collapsed binary.

    A single pooled score hides a total collapse on `threat` behind strong
    `toxic` performance, and `threat` is the label the routing policy exists for.
    """
    report = _load_report("per_label")
    for label, spec in GATES["per_label_average_precision"].items():
        floor = _floor(spec.get("floor"), f"{label} AP")
        stats = report["per_label"][label]
        assert stats["average_precision"] >= floor, (
            f"{label} AP degraded: {stats['average_precision']:.4f} < {floor:.4f} "
            f"(was {spec.get('measured')} on {GATES.get('measured_on')}); stats={stats}"
        )


def test_auto_action_precision_is_never_relaxed() -> None:
    """Hard policy floor.

    Coverage falling to zero on the rare labels is the expected result and is
    not a failure. Precision is not negotiable — a false auto-action is an
    enforcement nobody appealed.
    """
    report = _load_report("tiers")
    stats = report["tiers"]["auto_action"]
    minimum = GATES["min_predicted_positives_for_precision"]
    if stats["n_predicted_positive"] < minimum:
        pytest.skip(
            f"auto_action fired {stats['n_predicted_positive']}x; "
            f"precision is undefined below n={minimum}"
        )
    floor = GATES["routing"]["auto_action_precision_floor"]
    assert stats["precision"] >= floor, f"auto_action precision breach: {stats}"


def test_human_review_precision() -> None:
    report = _load_report("tiers")
    floor = _floor(
        GATES["routing"].get("human_review_precision_floor"), "human_review precision"
    )
    stats = report["tiers"]["human_review"]
    assert stats["precision"] >= floor, f"human_review precision degraded: {stats}"


def test_router_beats_fifo_at_equal_capacity() -> None:
    """The whole thesis. If this fails, the router is not routing."""
    report = _load_report("simulation")
    floor = _floor(
        GATES["routing"].get("harm_per_reviewer_hour_vs_fifo_min"), "harm-vs-FIFO"
    )
    sim = report["simulation"]
    ratio = sim["harm_per_reviewer_hour"]["router"] / sim["harm_per_reviewer_hour"]["fifo"]
    assert ratio >= floor, f"router no longer beats FIFO: {sim['harm_per_reviewer_hour']}"


def test_queue_clears_under_stated_capacity() -> None:
    report = _load_report("simulation")
    sim = report["simulation"]
    depth_max = _floor(GATES["routing"].get("queue_depth_p95_max"), "queue depth p95")
    assert sim["queue_depth_p95"] <= depth_max, f"backlog growing: {sim}"
    assert sim["reviewer_utilization"] >= GATES["routing"]["reviewer_utilization_min"], (
        f"scheduler is underutilizing reviewers: {sim}"
    )


def test_hierarchy_consistency() -> None:
    """severe_toxic is an exact subset of toxic; predictions must respect that."""
    report = _load_report("consistency")
    rate = report["consistency"]["hierarchy_violation_rate"]
    ceiling = GATES["consistency"]["hierarchy_violation_rate_max"]
    assert rate <= ceiling, (
        f"severe_toxic asserted without toxic in {rate:.4%} of predictions "
        f"(max {ceiling:.4%}); stats={report['consistency']}"
    )
