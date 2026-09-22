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
REQUIRE_REAL_ENV = "REVIEW_ROUTER_EVAL_REQUIRE_REAL"
CONFIG_ENV = "REVIEW_ROUTER_EVAL_CONFIG"
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
        pytest.skip(f"{name} has no fixed acceptance limit; reported as a diagnostic")
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
    requested = os.environ.get(CONFIG_ENV)
    candidates = [Path(requested)] if requested else [recorded, ROOT / "configs" / recorded.name]
    current = next((c for c in candidates if c.is_file()), None)
    if current is None:
        pytest.fail(f"config {recorded} named by the manifest is not in this checkout")
    assert manifest.get("config_sha256") == _sha256(current), (
        f"report was made with a different {current.name}: "
        f"manifest {manifest.get('config_sha256')} vs current {_sha256(current)}"
    )
    for name, hash_key in (("policy.yaml", "policy_sha256"), ("config.yaml", "config_sha256")):
        saved = _run_dir() / name
        assert saved.is_file(), f"saved {name} is missing"
        assert _sha256(saved) == manifest[hash_key], f"saved {name} differs from manifest"


def test_report_uses_the_pinned_real_corpus_when_required() -> None:
    """Real-evaluation CI cannot substitute the synthetic corpus or another export."""
    if not os.environ.get(REQUIRE_REAL_ENV):
        pytest.skip(f"${REQUIRE_REAL_ENV} unset; pinned real corpus not required")
    from scripts.verify_data import load_pins

    report = _load_report("synthetic")
    assert report["synthetic"] is False, "real evaluation cannot use synthetic data"
    expected = {
        name.removesuffix(".csv"): digest
        for name, digest in load_pins(ROOT / "configs" / "jigsaw.sha256").items()
    }
    assert _manifest().get("data_sha256") == expected, "report used an unpinned corpus"


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

    This diagnoses expected-count calibration; it does not identify the cause of a shift.
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
    assert stats["n_predicted_positive"] >= GATES["min_predicted_positives_for_precision"], stats
    assert stats["precision"] >= floor, f"human_review precision degraded: {stats}"


# ----------------------------------------------------------------------------- 3. capacity


def _capacity(name: str) -> float:
    return _floor(GATES["capacity"].get(name), name)


def test_router_beats_fifo_at_equal_capacity() -> None:
    """Empirical benchmark comparison, not statistical noninferiority evidence."""
    report = _load_report("simulation")
    floor = _floor(GATES["capacity"].get("harm_per_reviewer_hour_vs_fifo_min"), "harm-vs-FIFO")
    sim = report["simulation"]
    values = sim["harm_per_reviewer_hour"]
    if values["router"] is None or values["fifo"] is None or values["fifo"] == 0:
        pytest.skip(f"harm comparison unavailable: {values}")
    ratio = values["router"] / values["fifo"]
    assert ratio >= floor, f"router no longer beats FIFO: {sim['harm_per_reviewer_hour']}"


def test_router_reaches_high_risk_earlier_than_fifo() -> None:
    """Time-to-action for high-risk items at the stated capacity, router over FIFO."""
    report = _load_report("simulation")
    ceiling = _floor(
        GATES["capacity"].get("high_risk_wait_p90_vs_fifo_max"), "high-risk wait vs FIFO"
    )
    waits = report["simulation"]["high_risk_wait_p90"]
    if waits["router"] is None or waits["fifo"] is None:
        pytest.skip(f"high-risk wait p90 is undefined for this run: {waits}")
    if waits["router"] == waits["fifo"] == 0:
        pytest.skip(f"both waits are zero; no relative waiting-time evidence: {waits}")
    assert waits["router"] <= ceiling * waits["fifo"], (
        f"router no longer reaches high-risk items earlier: {waits}"
    )


def test_queue_clears_at_the_primary_load() -> None:
    """Completion ratio, backlog and pooled wait p50 at the primary load only.

    These say nothing about other loads: the overload scenario is expected to
    leave a backlog, and that is what the harm-per-reviewer-hour gate reads.
    """
    report = _load_report("simulation")
    sim = report["simulation"]
    assert sim["completion_ratio"] >= _capacity("completion_ratio_min"), (
        f"queue no longer clears at the primary load: {sim['completion_ratio']}"
    )
    assert sim["backlog_end"] <= _capacity("backlog_end_max"), f"backlog at the horizon: {sim}"
    primary = sim["primary"]
    key = f"{primary['strategy']}@{primary['load_per_hour']:g}"
    wait = sim["time_metrics"][key]["pooled_over_seeds"]["wait"]
    assert wait["p50"] is not None, wait
    assert wait["p50"] <= _capacity("wait_p50_max_min"), (
        f"pooled primary wait p50 degraded: {wait}"
    )


def test_queue_depth_stays_within_configured_limit() -> None:
    report = _load_report("simulation")
    sim = report["simulation"]
    depth_max = _floor(GATES["capacity"].get("queue_depth_p95_max"), "queue depth p95")
    assert sim["queue_depth_p95"] <= depth_max, f"queue depth exceeds limit: {sim}"



def test_reviewer_utilization() -> None:
    report = _load_report("simulation")
    sim = report["simulation"]
    assert sim["reviewer_utilization"] >= GATES["capacity"]["reviewer_utilization_min"], (
        f"scheduler is underutilizing reviewers: {sim}"
    )


# ----------------------------------------------------------------------------- 4. reproducibility


def _records_rows() -> list[dict[str, Any]]:
    import pandas as pd

    path = _run_dir() / GATES["reproducibility"]["records_file"]
    if not path.is_file():
        pytest.fail(f"per-job simulation records are missing: {path}")
    frame = pd.read_csv(path, dtype={"comment_id": str}, float_precision="round_trip")
    return [dict(row) for row in frame.to_dict(orient="records")]


def _close(a: float | None, b: float | None, tol: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def _assert_same(actual: Any, expected: Any, tol: float, context: str) -> None:
    """Compare a saved numerical tree with absolute tolerance, including every field."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), context
        for name, value in expected.items():
            _assert_same(actual[name], value, tol, f"{context}.{name}")
    elif isinstance(expected, float):
        assert _close(actual, expected, tol), (context, actual, expected)
    else:
        assert actual == expected, (context, actual, expected)


def test_every_time_metric_is_recomputable_from_the_records() -> None:
    """Rebuild every configured (load, strategy) pool, including sample sizes and tails."""
    import numpy as np

    from review_router.pipeline import _headline, _paired, _summarise, load_config
    from review_router.simulate import STRATEGIES, JobRecords, time_metrics

    report = _load_report("simulation")
    sim = report["simulation"]
    config = load_config(_run_dir() / "config.yaml")
    tol = float(GATES["reproducibility"]["float_tolerance"])
    horizon = config.sim.horizon_minutes
    assert sim["assumptions"]["horizon_hours"] == config.sim.horizon_hours
    rows = _records_rows()
    scenarios = {
        (strategy, load, seed)
        for strategy in STRATEGIES
        for load in config.loads_per_hour
        for seed in config.sim_seeds
    }
    saved_scenarios = {
        (r["strategy"], float(r["load_per_hour"]), int(r["seed"])) for r in rows
    }
    # Zero-arrival scenarios can have no records; table entries still establish
    # their presence. Every nonempty record scenario must belong to the config.
    assert saved_scenarios <= scenarios, "records contain unexpected scenarios"
    table_scenarios = [
        (r["strategy"], float(r["load_per_hour"]), int(r["seed"])) for r in sim["table"]
    ]
    assert set(table_scenarios) == scenarios and len(table_scenarios) == len(scenarios)
    expected_keys = {f"{strategy}@{load:g}" for strategy, load, _ in scenarios}
    assert set(sim["time_metrics"]) == expected_keys, "time-metric scenarios differ from config"
    for key, block in sim["time_metrics"].items():
        strategy, load = key.split("@")
        subset = [
            r for r in rows
            if r["strategy"] == strategy and float(r["load_per_hour"]) == float(load)
        ]
        recomputed = time_metrics(JobRecords.from_rows(subset, horizon))
        assert block["n_seeds"] == len(config.sim_seeds), key
        _assert_same(recomputed, block["pooled_over_seeds"], tol, key)
        for seed in config.sim_seeds:
            seed_rows = [r for r in subset if int(r["seed"]) == seed]
            records = JobRecords.from_rows(seed_rows, horizon)
            tm = time_metrics(records)
            table_row = next(
                r for r in sim["table"]
                if r["strategy"] == strategy and float(r["load_per_hour"]) == float(load)
                and int(r["seed"]) == seed
            )
            derived = {
                "n_arrivals": len(records),
                "n_handled": tm["status"]["completed"],
                "n_not_started": tm["status"]["not_started"],
                "n_in_progress": tm["status"]["in_progress"],
                "high_risk_arrived": sum(tm["high_risk_status"].values()),
                "high_risk_handled": tm["high_risk_status"]["completed"],
                "high_risk_not_started": tm["high_risk_status"]["not_started"],
                "high_risk_in_progress": tm["high_risk_status"]["in_progress"],
                "high_risk_unhandled": tm["high_risk_status"]["not_started"]
                + tm["high_risk_status"]["in_progress"],
                "completion_ratio": tm["completion_ratio"],
                "backlog_end": tm["status"]["not_started"],
                "unfinished_age_p50": tm["unfinished_age_at_horizon"]["p50"],
                "n_wait_samples": tm["wait"]["n"],
                "n_high_risk_wait_samples": tm["high_risk_wait"]["n"],
                "n_completion_samples": tm["completion_latency"]["n"],
            }
            for field in ("wait", "high_risk_wait", "completion_latency"):
                for percentile in ("p50", "p90", "p99"):
                    derived[f"{field}_{percentile}"] = tm[field][percentile]
            # Capacity numbers also derive from these records, with the
            # documented one-minute grid for queue depth and fixed staff time.
            grid = np.arange(0.0, horizon, 1.0)
            depth = np.searchsorted(np.sort(records.arrival_min), grid, side="right")
            depth -= np.searchsorted(
                np.sort(records.start_min[records.started]), grid, side="right"
            )
            busy = np.clip(
                np.minimum(records.completion_min[records.started], horizon)
                - records.start_min[records.started], 0.0, None,
            ).sum()
            handled_harm = float(records.harm[records.completed].sum())
            derived.update({
                "harm_arrived": float(records.harm.sum()),
                "harm_handled": handled_harm,
                "harm_per_reviewer_hour": handled_harm / (config.sim.reviewers * horizon / 60),
                "reviewer_utilization": float(busy / (config.sim.reviewers * horizon)),
                "queue_depth_p95": float(np.percentile(depth, 95)) if len(depth) else 0.0,
            })
            _assert_same(
                table_row, {"strategy": strategy, "load_per_hour": float(load),
                            "seed": seed, **derived}, tol, f"{key}/{seed}",
            )
            for column, values in (
                ("wait_min", records.wait_min),
                ("completion_latency_min", records.completion_latency_min),
            ):
                stored = np.array([r[column] for r in seed_rows], dtype=float)
                assert np.allclose(stored, values, rtol=0.0, atol=tol, equal_nan=True), (
                    key, seed, column
                )
    summary = _summarise(sim["table"])
    _assert_same(sim["summary"], summary, tol, "summary")
    _assert_same(sim["paired"], _paired(sim["table"]), tol, "paired")
    _assert_same(sim["headline"], _headline(sim["time_metrics"], config), tol, "headline")

    def mean(load: float, strategy: str, field: str) -> Any:
        value = summary[f"{strategy}@{load:g}"][field]
        return value["mean"] if value is not None else None

    def comparison(load: float, field: str) -> dict[str, Any]:
        return {
            "router": mean(load, config.primary_strategy, field),
            "fifo": mean(load, "fifo", field), "load_per_hour": load,
        }

    for field, load in (
        ("harm_per_reviewer_hour", config.thesis_load_per_hour),
        ("high_risk_wait_p90", config.primary_load_per_hour),
    ):
        _assert_same(sim[field], comparison(load, field), tol, field)
    for field in ("high_risk_handled", "high_risk_arrived"):
        for scenario, load in (
            ("primary", config.primary_load_per_hour), ("thesis", config.thesis_load_per_hour)
        ):
            _assert_same(sim[field][scenario], comparison(load, field), tol, f"{field}/{scenario}")
    for field in ("completion_ratio", "backlog_end", "queue_depth_p95", "reviewer_utilization"):
        _assert_same(
            sim[field], mean(config.primary_load_per_hour, config.primary_strategy, field),
            tol, field,
        )
    _assert_same(
        sim["high_risk_unfinished"],
        mean(config.primary_load_per_hour, config.primary_strategy, "high_risk_unhandled"),
        tol, "high_risk_unfinished",
    )


def test_replaying_a_scenario_reproduces_the_saved_records() -> None:
    """Same inputs and seed reproduce every saved scheduling field and summary."""
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
    assert GATES["reproducibility"]["scenario"] == {"load": "primary", "seed": "first"}
    pred = pd.read_csv(
        run_dir / "predictions.csv", dtype={"id": str}, float_precision="round_trip"
    )
    proba = pred[[f"p_{label}" for label in LABELS]].to_numpy(dtype=float)
    y = pred[[f"y_{label}" for label in LABELS]].to_numpy(dtype=int)
    inputs = queue_inputs(
        proba, y, pred["final_tier"].to_numpy(), policy, config.high_risk_min_weight
    )
    load = config.primary_load_per_hour
    seed = config.sim_seeds[0]
    rows = _records_rows()
    for strategy in dict.fromkeys((config.primary_strategy, "fifo")):
        scenario = draw_scenario(seed, load, len(inputs["queued"]), config.sim)
        result, records = simulate(
            scenario, inputs["priorities"][strategy], inputs["harm"], inputs["high_risk"],
            config.sim, strategy,
        )
        saved = [
            r for r in rows
            if r["strategy"] == strategy and float(r["load_per_hour"]) == load
            and int(r["seed"]) == seed
        ]
        expected = records.rows(
            comment_id=pred["id"].to_numpy()[inputs["queued"]][records.job_index]
        )
        assert len(saved) == len(expected), (strategy, len(saved), len(expected))
        numeric = (
            "arrival_min", "start_min", "completion_min", "harm", "priority_score",
            "wait_min", "completion_latency_min",
        )
        for field in numeric:
            observed = np.array([r[field] for r in saved], dtype=float)
            replayed = np.array([r[field] for r in expected], dtype=float)
            assert np.allclose(observed, replayed, rtol=0.0, atol=tol, equal_nan=True), (
                strategy, field
            )
        for field in ("job_index", "status", "high_risk", "comment_id"):
            assert [r[field] for r in saved] == [r[field] for r in expected], (strategy, field)
        table_row = next(
            r for r in report["simulation"]["table"]
            if r["strategy"] == strategy and float(r["load_per_hour"]) == load
            and int(r["seed"]) == seed
        )
        _assert_same(
            table_row, {"load_per_hour": load, "seed": seed, **result.as_json()}, tol, strategy
        )


# ----------------------------------------------------------------------------- fairness


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



@pytest.mark.parametrize("scenario", ["primary", "thesis"])
def test_high_risk_completions_do_not_fall_below_fifo(scenario: str) -> None:
    """Wait quantiles must not improve by dropping unfinished high-risk work."""
    report = _load_report("simulation")
    floor = _floor(
        GATES["capacity"].get("high_risk_handled_vs_fifo_min"), "high-risk completions vs FIFO"
    )
    values = report["simulation"]["high_risk_handled"][scenario]
    if values["router"] is None or not values["fifo"]:
        pytest.skip(f"high-risk completion comparison unavailable: {values}")
    assert values["router"] / values["fifo"] >= floor, (
        f"fewer high-risk reviews completed at {scenario} load: {values}"
    )



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
    monkeypatch.setitem(GATES["capacity"], "high_risk_handled_vs_fifo_min", 1.0)
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
    monkeypatch.setitem(GATES["capacity"], "high_risk_wait_p90_vs_fifo_max", 1.0)
    with pytest.raises(AssertionError, match="no longer reaches"):
        test_router_reaches_high_risk_earlier_than_fifo()



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



@pytest.mark.parametrize("snapshot", ["policy.yaml", "config.yaml"])
def test_replay_snapshot_tampering_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, snapshot: str
) -> None:
    config = ROOT / "configs" / "baseline.yaml"
    manifest = {
        "policy_sha256": _sha256(DEFAULT_POLICY_PATH),
        "config_sha256": _sha256(config),
        "config_path": str(config),
    }
    monkeypatch.setenv(CONFIG_ENV, str(config))
    monkeypatch.setenv(REPORT_ENV, str(_write_run(tmp_path, {}, manifest)))
    (tmp_path / "policy.yaml").write_bytes(DEFAULT_POLICY_PATH.read_bytes())
    (tmp_path / "config.yaml").write_bytes(config.read_bytes())
    test_report_was_made_with_the_current_policy_and_config()
    (tmp_path / snapshot).write_text("tampered: true\n")
    with pytest.raises(AssertionError, match=f"saved {snapshot} differs"):
        test_report_was_made_with_the_current_policy_and_config()


@pytest.mark.parametrize("synthetic, message", [(True, "synthetic"), (False, "unpinned")])
def test_real_evaluation_rejects_synthetic_or_unpinned_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, synthetic: bool, message: str
) -> None:
    monkeypatch.setenv(REQUIRE_REAL_ENV, "1")
    monkeypatch.setenv(
        REPORT_ENV,
        str(_write_run(tmp_path, {"synthetic": synthetic}, {"data_sha256": {}})),
    )
    with pytest.raises(AssertionError, match=message):
        test_report_uses_the_pinned_real_corpus_when_required()


@pytest.mark.parametrize(
    "contents",
    ["", "invalid train.csv\n", "0" * 64 + " train.csv\n", "0" * 64 + " ../train.csv\n",
     ("0" * 64 + " train.csv\n") * 2],
)
def test_incomplete_or_malformed_corpus_pins_are_rejected(tmp_path: Path, contents: str) -> None:
    from scripts.verify_data import load_pins

    path = tmp_path / "pins.sha256"
    path.write_text(contents)
    with pytest.raises(ValueError):
        load_pins(path)


def test_wait_p50_gate_ignores_configurable_headline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = {"simulation": {
        "completion_ratio": 1.0, "backlog_end": 0,
        "primary": {"strategy": "priority", "load_per_hour": 108.0},
        "time_metrics": {"priority@108": {"pooled_over_seeds": {"wait": {"p50": 2.0}}}},
        "headline": {"metric": "completion_latency", "selected": {"router": 0.0}},
    }}
    monkeypatch.setenv(REPORT_ENV, str(_write_run(tmp_path, report)))
    monkeypatch.setitem(GATES["capacity"], "wait_p50_max_min", 1.0)
    with pytest.raises(AssertionError, match="pooled primary wait p50 degraded"):
        test_queue_clears_at_the_primary_load()


def test_numerical_replay_comparison_has_no_relative_slack() -> None:
    with pytest.raises(AssertionError):
        _assert_same({"time": 400.000001}, {"time": 400.0}, 1e-9, "replay")
