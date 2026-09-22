"""Regression gates, in four categories: classification, routing, capacity, reproducibility.

The gate is an ordinary pytest file so CI is one `pytest -q` job. Without
REVIEW_ROUTER_EVAL_REPORT the gates SKIP (code-check layer). With it set they
read a report produced by the current code (real-evaluation layer) and FAIL,
never skip, when the report is missing, was made with a different policy or
config, or falls below a floor. Gates whose floor is still null skip rather
than pass vacuously: a floor of 0.0 that nothing can fail reads green.

Every assertion prints the whole stats dict, so a red CI tells you the new
number immediately instead of just which boolean flipped.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from review_router.policy import DEFAULT_POLICY_PATH, load_policy

POLICY = load_policy()
GATES = POLICY.gates
REPORT_ENV = "REVIEW_ROUTER_EVAL_REPORT"
REQUIRE_CLEAN_ENV = "REVIEW_ROUTER_EVAL_REQUIRE_CLEAN"
ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------- loading


def _report_path() -> Path:
    path = os.environ.get(REPORT_ENV)
    if not path:
        pytest.skip(
            f"${REPORT_ENV} is unset; run scripts/run_pipeline.py and point it at report.json"
        )
    if not Path(path).is_file():
        # An explicitly requested report that is missing is a failed experiment, not a skip.
        pytest.fail(f"${REPORT_ENV}={path} does not exist; the full run produced no report")
    return Path(path)


def _load_report(kind: str) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(_report_path().read_text(encoding="utf-8"))
    if kind not in report:
        pytest.fail(f"explicit evaluation report has no {kind!r} section")
    return report


def _run_dir() -> Path:
    return _report_path().parent


def _manifest() -> dict[str, Any]:
    path = _run_dir() / "manifest.json"
    if not path.is_file():
        pytest.fail(f"{path} is missing; a report without its manifest cannot be trusted")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _floor(value: float | None, name: str) -> float:
    if value is None:
        pytest.skip(f"{name} floor is null in policy.yaml — set it after the first run")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- provenance


def test_report_was_made_with_the_current_policy_and_config() -> None:
    """A report from a different policy or config proves nothing about this code."""
    manifest = _manifest()
    policy_sha = _sha256(DEFAULT_POLICY_PATH)
    assert manifest.get("policy_sha256") == policy_sha, (
        "report was made with a different policy.yaml: "
        f"manifest {manifest.get('policy_sha256')} vs current {policy_sha}"
    )
    recorded = Path(str(manifest.get("config_path", "")))
    candidates = [recorded, ROOT / "configs" / recorded.name]
    current = next((c for c in candidates if c.is_file()), None)
    if current is None:
        pytest.fail(f"config {recorded} named by the manifest is not in this checkout")
    assert manifest.get("config_sha256") == _sha256(current), (
        f"report was made with a different {current.name}: "
        f"manifest {manifest.get('config_sha256')} vs current {_sha256(current)}"
    )


def test_report_comes_from_a_clean_checkout_when_required() -> None:
    """In the real-evaluation workflow the report must match the commit under test."""
    if not os.environ.get(REQUIRE_CLEAN_ENV):
        pytest.skip(f"${REQUIRE_CLEAN_ENV} unset; commit match not enforced outside CI")
    manifest = _manifest()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    assert manifest.get("git_dirty") is False, "report was made from a dirty working tree"
    assert manifest.get("git_commit") == head, (
        f"report commit {manifest.get('git_commit')} is not the commit under test {head}"
    )


# ----------------------------------------------------------------------------- 1. classification


def test_per_label_average_precision() -> None:
    """Gate per label, never on a collapsed binary.

    A single pooled score hides a total collapse on `threat` behind strong
    `toxic` performance, and `threat` is the label the routing policy exists for.
    """
    report = _load_report("per_label")
    configured = {
        label: spec
        for label, spec in GATES["classification"]["per_label_average_precision"].items()
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


def test_per_label_precision_at_the_operating_point() -> None:
    """Precision at the frozen human_review threshold, per label, on the test rows."""
    report = _load_report("per_label")
    configured = {
        label: spec
        for label, spec in GATES["classification"]["per_label_precision_at_human_threshold"].items()
        if spec.get("floor") is not None
    }
    if not configured:
        pytest.skip("no per-label operating-point precision floor is set")
    minimum = GATES["min_predicted_positives_for_precision"]
    for label, spec in configured.items():
        stats = report["per_label"][label]["at_human_review"]
        assert stats["threshold"] is not None, f"{label} has no human_review threshold: {stats}"
        assert stats["n_predicted_positive"] >= minimum, (
            f"{label} operating point too thin: {stats}"
        )
        assert stats["precision"] >= spec["floor"], (
            f"{label} precision at the operating point degraded: {stats['precision']:.4f} < "
            f"{spec['floor']:.4f} (was {spec.get('measured')}); stats={stats}"
        )


def test_hierarchy_consistency() -> None:
    """severe_toxic is an exact subset of toxic; predictions must respect that."""
    report = _load_report("consistency")
    rate = report["consistency"]["hierarchy_violation_rate"]
    ceiling = GATES["classification"]["hierarchy_violation_rate_max"]
    assert rate <= ceiling, (
        f"severe_toxic asserted without toxic in {rate:.4%} of predictions "
        f"(max {ceiling:.4%}); stats={report['consistency']}"
    )


def test_predicted_volume_overshoot() -> None:
    """The model's expected positive count on the test rows versus the actual count.

    ROC-AUC is rank-based and cannot see a prevalence shift; the overshoot can.
    """
    report = _load_report("prevalence_shift")
    configured = {
        label: spec
        for label, spec in (GATES["classification"].get("volume_overshoot_max") or {}).items()
        if spec.get("ceiling") is not None
    }
    if not configured:
        pytest.skip("no volume-overshoot ceiling is set")
    for label, spec in configured.items():
        stats = report["prevalence_shift"][label]
        value = stats["volume_overshoot_model"]
        assert value is not None, f"{label}: no positives on the test rows: {stats}"
        assert value <= spec["ceiling"], (
            f"{label} predicted volume overshoots the actual count by {value:.1%} "
            f"(ceiling {spec['ceiling']:.0%}, was {spec.get('measured')}); stats={stats}"
        )


# ----------------------------------------------------------------------------- 2. routing


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
    floor = _floor(GATES["routing"].get("human_review_precision_floor"), "human_review precision")
    stats = report["tiers"]["human_review"]
    assert stats["n_predicted_positive"] >= GATES["min_predicted_positives_for_precision"], stats
    assert stats["precision"] >= floor, f"human_review precision degraded: {stats}"


# ----------------------------------------------------------------------------- 3. capacity


def _capacity(name: str) -> float:
    return _floor(GATES["capacity"].get(name), name)


def test_router_beats_fifo_at_equal_capacity() -> None:
    """The whole thesis, read at the overload load where ordering can change harm handled."""
    report = _load_report("simulation")
    floor = _capacity("harm_per_reviewer_hour_vs_fifo_min")
    sim = report["simulation"]
    ratio = sim["harm_per_reviewer_hour"]["router"] / sim["harm_per_reviewer_hour"]["fifo"]
    assert ratio >= floor, f"router no longer beats FIFO: {sim['harm_per_reviewer_hour']}"


def test_router_reaches_high_risk_earlier_than_fifo() -> None:
    """Time-to-action for high-risk items at the stated capacity, router over FIFO."""
    report = _load_report("simulation")
    ceiling = _capacity("high_risk_wait_p90_vs_fifo_max")
    waits = report["simulation"]["high_risk_wait_p90"]
    if waits["router"] is None or not waits["fifo"]:
        pytest.skip(f"high-risk wait p90 is undefined for this run: {waits}")
    ratio = waits["router"] / waits["fifo"]
    assert ratio <= ceiling, f"router no longer reaches high-risk items earlier: {waits}"


def test_queue_clears_at_the_primary_load() -> None:
    """Completion ratio, backlog and headline wait at the primary load only.

    These say nothing about other loads: the overload scenario is expected to
    leave a backlog, and that is what the harm-per-reviewer-hour gate reads.
    """
    report = _load_report("simulation")
    sim = report["simulation"]
    assert sim["completion_ratio"] >= _capacity("completion_ratio_min"), (
        f"queue no longer clears at the primary load: {sim['completion_ratio']}"
    )
    assert sim["backlog_end"] <= _capacity("backlog_end_max"), f"backlog at the horizon: {sim}"
    headline = sim["headline"]["selected"]
    assert headline["router"] is not None, headline
    assert headline["router"] <= _capacity("wait_p50_max_min"), (
        f"headline wait degraded: {headline}"
    )


def test_queue_depth_and_utilization() -> None:
    report = _load_report("simulation")
    sim = report["simulation"]
    depth_max = _capacity("queue_depth_p95_max")
    assert sim["queue_depth_p95"] <= depth_max, f"queue depth exceeds limit: {sim}"
    assert sim["reviewer_utilization"] >= GATES["capacity"]["reviewer_utilization_min"], (
        f"scheduler is underutilizing reviewers: {sim}"
    )


# ----------------------------------------------------------------------------- 4. reproducibility


def _records_rows() -> list[dict[str, Any]]:
    import pandas as pd

    path = _run_dir() / GATES["reproducibility"]["records_file"]
    if not path.is_file():
        pytest.fail(f"per-job simulation records are missing: {path}")
    frame = pd.read_csv(path, dtype={"comment_id": str})
    return [dict(row) for row in frame.to_dict(orient="records")]


def _close(a: float | None, b: float | None, tol: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def test_every_time_metric_is_recomputable_from_the_records() -> None:
    """Rebuild each (load, strategy) pool from simulation_jobs.csv and compare."""
    from review_router.simulate import JobRecords, time_metrics

    report = _load_report("simulation")
    sim = report["simulation"]
    tol = float(GATES["reproducibility"]["float_tolerance"])
    horizon = float(sim["assumptions"]["horizon_hours"]) * 60.0
    rows = _records_rows()
    for key, block in sim["time_metrics"].items():
        strategy, load = key.split("@")
        subset = [
            r
            for r in rows
            if r["strategy"] == strategy and float(r["load_per_hour"]) == float(load)
        ]
        assert subset, f"no records for {key}"
        recomputed = time_metrics(JobRecords.from_rows(subset, horizon))
        expected = block["pooled_over_seeds"]
        assert recomputed["status"] == expected["status"], key
        assert recomputed["high_risk_status"] == expected["high_risk_status"], key
        for field in ("wait", "high_risk_wait", "completion_latency", "unfinished_age_at_horizon"):
            for q in ("p50", "p90", "p99"):
                assert _close(recomputed[field][q], expected[field][q], tol), (key, field, q)
            assert recomputed[field]["n"] == expected[field]["n"], (key, field)
    # The headline reads the same pool, so it must agree too.
    head = sim["headline"]
    pool = sim["time_metrics"][f"{head['router_strategy']}@{head['load_per_hour']:g}"]
    value = pool["pooled_over_seeds"][head["metric"]][f"p{head['percentile']}"]
    assert _close(head["selected"]["router"], value, tol), head


def test_replaying_a_scenario_reproduces_the_saved_records() -> None:
    """Same config, same seed: the same job order, start and completion times."""
    import numpy as np
    import pandas as pd

    from review_router.data import LABELS
    from review_router.pipeline import load_config, queue_inputs
    from review_router.simulate import draw_scenario, simulate

    report = _load_report("simulation")
    run_dir = _run_dir()
    config = load_config(run_dir / "config.yaml")
    policy = load_policy(run_dir / "policy.yaml")
    tol = float(GATES["reproducibility"]["float_tolerance"])
    pred = pd.read_csv(run_dir / "predictions.csv", dtype={"id": str})
    proba = pred[[f"p_{label}" for label in LABELS]].to_numpy(dtype=float)
    y = pred[[f"y_{label}" for label in LABELS]].to_numpy(dtype=int)
    inputs = queue_inputs(
        proba, y, pred["final_tier"].to_numpy(), policy, config.high_risk_min_weight
    )
    load = config.primary_load_per_hour
    seed = config.sim_seeds[0]
    rows = _records_rows()
    for strategy in (config.primary_strategy, "fifo"):
        scenario = draw_scenario(seed, load, len(inputs["queued"]), config.sim)
        result, records = simulate(
            scenario,
            inputs["priorities"][strategy],
            inputs["harm"],
            inputs["high_risk"],
            config.sim,
            strategy,
        )
        saved = [
            r
            for r in rows
            if r["strategy"] == strategy
            and float(r["load_per_hour"]) == load
            and int(r["seed"]) == seed
        ]
        assert len(saved) == len(records), (strategy, len(saved), len(records))
        assert [int(r["job_index"]) for r in saved] == records.job_index.tolist(), strategy
        assert [str(r["status"]) for r in saved] == records.status.tolist(), strategy
        start_saved = np.array(
            [np.nan if pd.isna(r["start_min"]) else float(r["start_min"]) for r in saved]
        )
        assert np.allclose(start_saved, records.start_min, atol=tol, equal_nan=True), strategy
        table_row = next(
            r
            for r in report["simulation"]["table"]
            if r["strategy"] == strategy
            and float(r["load_per_hour"]) == load
            and int(r["seed"]) == seed
        )
        for field in ("n_handled", "backlog_end", "wait_p50", "wait_p90", "completion_latency_p50"):
            assert _close(table_row[field], getattr(result, field), tol), (strategy, field)


# ----------------------------------------------------------------------------- fairness


@pytest.mark.parametrize("tier", ["predicted_positive", "auto_action"])
def test_false_positives_do_not_concentrate_on_identity_mentions(tier: str) -> None:
    """Gated on the lower bound of the ratio's interval; see policy.yaml for the bases."""
    report = _load_report("identity_false_positives")
    ceiling = _floor(
        GATES["fairness"].get("identity_false_discovery_rate_ratio_max"),
        "identity false-discovery ratio",
    )
    basis = "final_tier" if tier == "predicted_positive" else "model_tier"
    section = report["identity_false_positives"]["test"][basis][tier]
    _check_discovery_rate_ratio(section, ceiling, GATES["min_predicted_positives_for_precision"])


def _check_discovery_rate_ratio(section: dict[str, Any], ceiling: float, minimum: int) -> None:
    """Detect a disparity; an interval crossing the ceiling is inconclusive and skips."""
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


# ----------------------------------------------------------------------------- negative tests
# These run in the code-check layer and prove the gates fail instead of skipping.


def _write_run(
    tmp_path: Path, report: dict[str, Any], manifest: dict[str, Any] | None = None
) -> Path:
    (tmp_path / "report.json").write_text(json.dumps(report))
    if manifest is not None:
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path / "report.json"


def test_missing_report_at_explicit_path_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(REPORT_ENV, str(tmp_path / "does-not-exist.json"))
    with pytest.raises(pytest.fail.Exception, match="produced no report"):
        _load_report("tiers")


def test_explicit_report_with_missing_section_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(REPORT_ENV, str(_write_run(tmp_path, {})))
    with pytest.raises(pytest.fail.Exception, match="no 'tiers' section"):
        _load_report("tiers")


def test_report_without_manifest_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(REPORT_ENV, str(_write_run(tmp_path, {"tiers": {}})))
    with pytest.raises(pytest.fail.Exception, match="manifest"):
        test_report_was_made_with_the_current_policy_and_config()


def test_report_made_with_another_policy_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = {"policy_sha256": "0" * 64, "config_sha256": "0" * 64, "config_path": "x.yaml"}
    monkeypatch.setenv(REPORT_ENV, str(_write_run(tmp_path, {"tiers": {}}, manifest)))
    with pytest.raises(AssertionError, match="different policy"):
        test_report_was_made_with_the_current_policy_and_config()


def test_report_made_with_another_config_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "baseline.yaml"
    config.write_text("label: x\n")
    manifest = {
        "policy_sha256": _sha256(DEFAULT_POLICY_PATH),
        "config_sha256": "0" * 64,
        "config_path": str(config),
    }
    monkeypatch.setenv(REPORT_ENV, str(_write_run(tmp_path, {"tiers": {}}, manifest)))
    with pytest.raises(AssertionError, match="different baseline.yaml"):
        test_report_was_made_with_the_current_policy_and_config()


def test_dirty_or_foreign_commit_fails_when_clean_is_required(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = {"git_dirty": True, "git_commit": "deadbeef"}
    monkeypatch.setenv(REPORT_ENV, str(_write_run(tmp_path, {"tiers": {}}, manifest)))
    monkeypatch.setenv(REQUIRE_CLEAN_ENV, "1")
    with pytest.raises(AssertionError, match="dirty"):
        test_report_comes_from_a_clean_checkout_when_required()


def test_metric_below_floor_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    report = {
        "simulation": {
            "completion_ratio": 0.5,
            "backlog_end": 0,
            "headline": {"selected": {"router": 0.1}},
        }
    }
    monkeypatch.setenv(REPORT_ENV, str(_write_run(tmp_path, report)))
    monkeypatch.setitem(GATES["capacity"], "completion_ratio_min", 0.97)
    with pytest.raises(AssertionError, match="no longer clears"):
        test_queue_clears_at_the_primary_load()


def test_missing_records_file_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(REPORT_ENV, str(_write_run(tmp_path, {"simulation": {}})))
    with pytest.raises(pytest.fail.Exception, match="records are missing"):
        _records_rows()


def test_null_label_floor_does_not_skip_other_configured_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(
        GATES["classification"],
        "per_label_average_precision",
        {"toxic": {"floor": None}, "threat": {"floor": 0.8}},
    )
    monkeypatch.setenv(
        REPORT_ENV, str(_write_run(tmp_path, {"per_label": {"threat": {"average_precision": 0.1}}}))
    )
    with pytest.raises(AssertionError, match="threat AP degraded"):
        test_per_label_average_precision()
