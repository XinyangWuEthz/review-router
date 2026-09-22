"""Regression gates.

The gate is an ordinary pytest file so CI is one `pytest -q` job. Gates whose
floor is still null in policy.yaml SKIP rather than pass — a floor of 0.0 that
nothing can fail is worse than no gate, because it reads green.

Every assertion prints the whole stats dict, so a red CI tells you the new
number immediately instead of just which boolean flipped.
"""

from __future__ import annotations

import json
import math
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
    if not path:
        pytest.skip(
            f"${REPORT_ENV} is unset; run scripts/run_pipeline.py and point it at report.json"
        )
    if not Path(path).is_file():
        # An explicitly requested report that is missing is a failed experiment, not a skip.
        pytest.fail(f"${REPORT_ENV}={path} does not exist; the full run produced no report")
    report: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    if kind not in report:
        pytest.fail(f"explicit evaluation report has no {kind!r} section")
    return report


def _floor(value: float | None, name: str) -> float:
    if value is None:
        pytest.skip(f"{name} has no fixed acceptance limit; reported as a diagnostic")
    return value


def test_per_label_average_precision() -> None:
    """Gate per label, never on a collapsed binary.

    A single pooled score hides a total collapse on `threat` behind strong
    `toxic` performance, and `threat` is the label the routing policy exists for.
    """
    report = _load_report("per_label")
    configured = {
        label: spec
        for label, spec in GATES["per_label_average_precision"].items()
        if spec.get("floor") is not None
    }
    if not configured:
        pytest.skip("all per-label AP floors are null in policy.yaml")
    for label, spec in configured.items():
        floor = spec["floor"]
        stats = report["per_label"][label]
        assert stats["average_precision"] >= floor, (
            f"{label} AP degraded: {stats['average_precision']:.4f} < {floor:.4f} "
            f"(was {spec.get('measured')} on {GATES.get('measured_on')}); stats={stats}"
        )


def test_all_flagged_comments_require_human_capacity() -> None:
    """A priority suggestion cannot bypass human confirmation or queue capacity."""
    report = _load_report("decision_contract")
    assert report["decision_contract"] == {
        "mode": "human_confirmation",
        "requires_human_confirmation": True,
        "automatic_actions": 0,
    }
    tiers = report["tiers"]
    assert set(tiers) == {"allow", "human_review", "priority_review"}
    priority = tiers["priority_review"]["n_predicted_positive"]
    regular = tiers["human_review"]["n_predicted_positive"]
    workload = report["review_workload"]
    assert workload["n_priority_review"] == priority
    assert workload["n_human_review"] == regular
    assert workload["n_requires_human_review"] == priority + regular
    assert workload["n_total"] == priority + regular + tiers["allow"]["n_predicted_positive"]
    if priority + regular:
        assumptions = report["simulation"]["assumptions"]
        assert assumptions["queued_jobs"] == priority + regular
        assert set(assumptions["queue_tiers"]) == {"priority_review", "human_review"}


def test_configured_priority_review_precision_floor() -> None:
    """Precision is diagnostic by default; an explicit future target is checked."""
    floor = _floor(
        GATES["routing"].get("priority_review_precision_floor"), "priority review precision"
    )
    report = _load_report("tiers")
    stats = report["tiers"]["priority_review"]
    minimum = GATES["min_predicted_positives_for_precision"]
    if stats["n_predicted_positive"] < minimum:
        pytest.skip(
            f"priority_review fired {stats['n_predicted_positive']}x; "
            f"precision is undefined below n={minimum}"
        )
    assert stats["precision"] >= floor, f"priority review precision breach: {stats}"


def test_human_review_precision() -> None:
    report = _load_report("tiers")
    floor = _floor(GATES["routing"].get("human_review_precision_floor"), "human_review precision")
    stats = report["tiers"]["human_review"]
    assert stats["precision"] >= floor, f"human_review precision degraded: {stats}"


def test_router_beats_fifo_at_equal_capacity() -> None:
    """Empirical benchmark comparison, not statistical noninferiority evidence."""
    report = _load_report("simulation")
    floor = _floor(GATES["routing"].get("harm_per_reviewer_hour_vs_fifo_min"), "harm-vs-FIFO")
    sim = report["simulation"]
    ratio = sim["harm_per_reviewer_hour"]["router"] / sim["harm_per_reviewer_hour"]["fifo"]
    assert ratio >= floor, f"router no longer beats FIFO: {sim['harm_per_reviewer_hour']}"


def test_router_reaches_high_risk_earlier_than_fifo() -> None:
    """Time-to-action for high-risk items at the stated capacity, router over FIFO."""
    report = _load_report("simulation")
    ceiling = _floor(
        GATES["routing"].get("high_risk_wait_p90_vs_fifo_max"), "high-risk wait vs FIFO"
    )
    waits = report["simulation"]["high_risk_wait_p90"]
    if waits["router"] is None or waits["fifo"] is None:
        pytest.skip(f"high-risk wait p90 is undefined for this run: {waits}")
    if waits["router"] == waits["fifo"] == 0:
        pytest.skip(f"both waits are zero; no relative waiting-time evidence: {waits}")
    assert waits["router"] <= ceiling * waits["fifo"], (
        f"router no longer reaches high-risk items earlier: {waits}"
    )


@pytest.mark.parametrize("scenario", ["primary", "thesis"])
def test_high_risk_completions_do_not_fall_below_fifo(scenario: str) -> None:
    """Wait quantiles must not improve by dropping unfinished high-risk work."""
    report = _load_report("simulation")
    floor = _floor(
        GATES["routing"].get("high_risk_handled_vs_fifo_min"), "high-risk completions vs FIFO"
    )
    values = report["simulation"]["high_risk_handled"][scenario]
    if values["router"] is None or not values["fifo"]:
        pytest.skip(f"high-risk completion comparison unavailable: {values}")
    assert values["router"] / values["fifo"] >= floor, (
        f"fewer high-risk reviews completed at {scenario} load: {values}"
    )


def test_queue_depth_stays_within_configured_limit() -> None:
    report = _load_report("simulation")
    sim = report["simulation"]
    depth_max = _floor(GATES["routing"].get("queue_depth_p95_max"), "queue depth p95")
    assert sim["queue_depth_p95"] <= depth_max, f"queue depth exceeds limit: {sim}"


def test_reviewer_utilization() -> None:
    report = _load_report("simulation")
    sim = report["simulation"]
    assert sim["reviewer_utilization"] >= GATES["routing"]["reviewer_utilization_min"], (
        f"scheduler is underutilizing reviewers: {sim}"
    )


@pytest.mark.parametrize("tier", ["predicted_positive", "priority_review", "human_review"])
def test_false_positives_do_not_concentrate_on_identity_mentions(tier: str) -> None:
    """Detect a disparity in false discoveries; this does not certify fairness.

    FDR has predicted positives as its denominator. A lower interval bound
    above the ceiling detects a disparity; an interval crossing it is
    inconclusive and skips. Check pooled and each final human-review band.
    """
    report = _load_report("identity_false_positives")
    ceiling = _floor(
        GATES["fairness"].get("identity_false_discovery_rate_ratio_max"),
        "identity false-discovery ratio",
    )
    section = report["identity_false_positives"]["test"]["final_tier"][tier]
    minimum = GATES["min_predicted_positives_for_precision"]
    _check_discovery_rate_ratio(section, ceiling, minimum)


def _check_discovery_rate_ratio(section: dict[str, Any], ceiling: float, minimum: int) -> None:
    groups = (section["with_identity_term"], section["without_identity_term"])
    if any(g["n_predicted_positive"] < minimum for g in groups):
        pytest.skip(f"too few predicted positives to compare subgroup FDR: {section}")
    ci = section["false_discovery_rate_ratio_ci95"]
    if ci is None:
        pytest.skip(f"subgroup FDR ratio interval is unavailable: {section}")
    if len(ci) != 2 or not all(math.isfinite(x) for x in ci) or not 0 <= ci[0] <= ci[1]:
        pytest.fail(f"invalid subgroup FDR ratio interval: {ci}")
    assert ci[0] <= ceiling, (
        "false discoveries concentrate on identity mentions: "
        f"ratio {section['false_discovery_rate_ratio']}, 95% CI {ci}, ceiling {ceiling}; {section}"
    )
    if ci[1] > ceiling:
        pytest.skip(f"subgroup FDR disparity is inconclusive: 95% CI {ci} crosses {ceiling}")


def test_hierarchy_consistency() -> None:
    """severe_toxic is an exact subset of toxic; predictions must respect that."""
    report = _load_report("consistency")
    rate = report["consistency"]["hierarchy_violation_rate"]
    ceiling = GATES["consistency"]["hierarchy_violation_rate_max"]
    assert rate <= ceiling, (
        f"severe_toxic asserted without toxic in {rate:.4%} of predictions "
        f"(max {ceiling:.4%}); stats={report['consistency']}"
    )


def test_missing_report_at_explicit_path_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A full run that produced no report must fail, not skip."""
    monkeypatch.setenv(REPORT_ENV, str(tmp_path / "does-not-exist.json"))
    with pytest.raises(pytest.fail.Exception, match="produced no report"):
        _load_report("tiers")


def test_priority_work_cannot_be_omitted_from_reviewer_capacity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catch the old behavior that simulated only the ordinary review band."""
    path = tmp_path / "report.json"
    path.write_text(json.dumps({
        "decision_contract": {
            "mode": "human_confirmation", "requires_human_confirmation": True,
            "automatic_actions": 0,
        },
        "tiers": {
            "priority_review": {"n_predicted_positive": 10},
            "human_review": {"n_predicted_positive": 20},
            "allow": {"n_predicted_positive": 70},
        },
        "review_workload": {
            "n_total": 100, "n_priority_review": 10, "n_human_review": 20,
            "n_requires_human_review": 30,
        },
        "simulation": {"assumptions": {
            "queued_jobs": 20, "queue_tiers": ["human_review"],
        }},
    }))
    monkeypatch.setenv(REPORT_ENV, str(path))
    with pytest.raises(AssertionError):
        test_all_flagged_comments_require_human_capacity()


def test_faster_waits_do_not_hide_fewer_high_risk_completions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"simulation": {
        "high_risk_wait_p90": {"router": 1.0, "fifo": 5.0},
        "high_risk_handled": {"primary": {"router": 8.0, "fifo": 10.0}},
    }}))
    monkeypatch.setenv(REPORT_ENV, str(path))
    monkeypatch.setitem(GATES["routing"], "high_risk_handled_vs_fifo_min", 1.0)
    with pytest.raises(AssertionError, match="fewer high-risk"):
        test_high_risk_completions_do_not_fall_below_fifo("primary")


def test_positive_wait_cannot_skip_against_zero_fifo_wait(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"simulation": {
        "high_risk_wait_p90": {"router": 1.0, "fifo": 0.0},
    }}))
    monkeypatch.setenv(REPORT_ENV, str(path))
    monkeypatch.setitem(GATES["routing"], "high_risk_wait_p90_vs_fifo_max", 1.0)
    with pytest.raises(AssertionError, match="no longer reaches"):
        test_router_reaches_high_risk_earlier_than_fifo()


def test_explicit_report_with_missing_section_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "incomplete.json"
    path.write_text("{}")
    monkeypatch.setenv(REPORT_ENV, str(path))
    with pytest.raises(pytest.fail.Exception, match="no 'tiers' section"):
        _load_report("tiers")


def test_null_label_floor_does_not_skip_other_configured_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(
        GATES,
        "per_label_average_precision",
        {
            "toxic": {"floor": None},
            "threat": {"floor": 0.8},
        },
    )
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"per_label": {"threat": {"average_precision": 0.1}}}))
    monkeypatch.setenv(REPORT_ENV, str(path))
    with pytest.raises(AssertionError, match="threat AP degraded"):
        test_per_label_average_precision()


@pytest.mark.parametrize(
    "interval, expected",
    [([0.6, 1.1], "pass"), ([0.6, 1.6], "inconclusive"), ([1.4, 2.0], "disparity")],
)
def test_discovery_disparity_gate_distinguishes_evidence_from_uncertainty(
    interval: list[float], expected: str
) -> None:
    section = {
        "with_identity_term": {"n_predicted_positive": 100},
        "without_identity_term": {"n_predicted_positive": 100},
        "false_discovery_rate_ratio": sum(interval) / 2,
        "false_discovery_rate_ratio_ci95": interval,
    }
    if expected == "inconclusive":
        with pytest.raises(pytest.skip.Exception, match="inconclusive"):
            _check_discovery_rate_ratio(section, 1.25, 30)
    elif expected == "disparity":
        with pytest.raises(AssertionError, match="false discoveries concentrate"):
            _check_discovery_rate_ratio(section, 1.25, 30)
    else:
        _check_discovery_rate_ratio(section, 1.25, 30)


def test_discovery_disparity_gate_does_not_pass_small_samples() -> None:
    section = {
        "with_identity_term": {"n_predicted_positive": 5},
        "without_identity_term": {"n_predicted_positive": 100},
        "false_discovery_rate_ratio": 0.9,
        "false_discovery_rate_ratio_ci95": [0.6, 1.1],
    }
    with pytest.raises(pytest.skip.Exception, match="too few predicted positives"):
        _check_discovery_rate_ratio(section, 1.25, 30)


def test_discovery_disparity_gate_rejects_nonfinite_interval() -> None:
    section = {
        "with_identity_term": {"n_predicted_positive": 100},
        "without_identity_term": {"n_predicted_positive": 100},
        "false_discovery_rate_ratio": 0.9,
        "false_discovery_rate_ratio_ci95": [0.6, float("nan")],
    }
    with pytest.raises(pytest.fail.Exception, match="invalid subgroup FDR ratio interval"):
        _check_discovery_rate_ratio(section, 1.25, 30)
