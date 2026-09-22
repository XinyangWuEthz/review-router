from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RENDERER = _module(ROOT / "scripts" / "render_results.py")
GATE_TESTS = _module(ROOT / "tests" / "test_gate.py")


def _policy(path: Path, auto_gate: float = 0.93, target: float = 0.97) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "tiers": {"allow": 1, "human_review": 2, "auto_action": 3},
                "tier_precision_floors": {"human_review": 0.87, "auto_action": target},
                "gates": {
                    "routing": {"auto_action_precision_floor": auto_gate},
                    "fairness": {"identity_false_discovery_rate_ratio_max": 1.25},
                    "min_predicted_positives_for_precision": 30,
                },
            }
        )
    )


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    _policy(run / "policy.yaml")
    manifest = {
        "run_id": "fixture",
        "git_commit": None,
        "git_dirty": False,
        "seed": 1,
        "dependencies": {"scikit-learn": "fixture-version"},
        "split_label_counts": {
            name: {"rows": 100} for name in ("train", "calib", "thresh", "test_scored")
        },
    }
    report = {
        "synthetic": True,
        "tiers": {
            "auto_action": {
                "n_predicted_positive": 100,
                "coverage": 0.1,
                "precision": 0.95,
                "precision_ci95": [0.9, 0.98],
            }
        },
        "per_label": {},
        "simulation": {
            "note": "nothing was routed to human_review; simulation skipped",
            "table": [],
        },
        "identity_false_positives": {"test": {"identity_terms": ["fixture-only"]}},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "report.json").write_text(json.dumps(report))
    return run


def test_snapshot_supplies_targets_and_gates(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []
    actual_load = RENDERER.load_policy

    def load(path: Path) -> Any:
        calls.append(path)
        return actual_load(path)

    monkeypatch.setattr(RENDERER, "load_policy", load)
    text = RENDERER.render(run_dir)
    assert calls == [run_dir / "policy.yaml"]
    assert "0.87 (human) and 0.97 (auto)" in text
    assert "| auto_action precision | 0.950 | ≥ 0.930 | green |" in text


def test_explicit_policy_overrides_gates_but_not_run_targets(run_dir: Path, tmp_path: Path) -> None:
    override = tmp_path / "new-policy.yaml"
    _policy(override, auto_gate=0.96, target=0.12)
    text = RENDERER.render(run_dir, override)
    assert "0.87 (human) and 0.97 (auto)" in text
    assert "0.12 (auto)" not in text
    assert f"explicitly overridden by `{override}`" in text
    assert f"Gates read from `{override}`" in text
    assert "| auto_action precision | 0.950 | ≥ 0.960 | **red** |" in text


def test_missing_snapshot_never_falls_back_to_current_policy(run_dir: Path) -> None:
    (run_dir / "policy.yaml").unlink()
    with pytest.raises(FileNotFoundError):
        RENDERER.render(run_dir)


def test_empty_simulation_and_synthetic_provenance_are_honest(run_dir: Path) -> None:
    text = RENDERER.render(run_dir)
    assert text.startswith("## Round-1 results (synthetic pipeline check)")
    assert "SYNTHETIC corpus: pipeline check only" in text
    assert "an unavailable Git commit" in text
    assert "simulation skipped" in text
    assert "recorded 1 whole-word terms" in text
    fairness_rows = [line for line in text.splitlines() if line.startswith("| identity FDR ratio")]
    assert len(fairness_rows) == 2
    assert all(line.endswith("| unavailable |") for line in fairness_rows)


@pytest.mark.parametrize(
    "interval,count,status,outcome",
    [
        ([0.5, 1.25], 30, "green", "pass"),
        ([1.25, 1.8], 30, "inconclusive", "skip"),
        ([1.3, 1.8], 30, "**red**", "fail"),
        ([0.5, 0.9], 29, "skipped (subgroup n < 30)", "skip"),
        (None, 30, "unavailable", "skip"),
    ],
)
def test_fairness_status_matches_regression_gate(
    interval: list[float] | None, count: int, status: str, outcome: str
) -> None:
    section = {
        "with_identity_term": {"n_predicted_positive": count},
        "without_identity_term": {"n_predicted_positive": 100},
        "false_discovery_rate_ratio": 1.0,
        "false_discovery_rate_ratio_ci95": interval,
    }
    assert RENDERER.fairness_gate_status(section, 1.25, 30) == status
    check = GATE_TESTS._check_discovery_rate_ratio
    if outcome == "skip":
        with pytest.raises(pytest.skip.Exception):
            check(section, 1.25, 30)
    elif outcome == "fail":
        with pytest.raises(AssertionError):
            check(section, 1.25, 30)
    else:
        check(section, 1.25, 30)


def test_small_auto_action_sample_is_not_green(run_dir: Path) -> None:
    path = run_dir / "report.json"
    report = json.loads(path.read_text())
    report["tiers"]["auto_action"].update(n_predicted_positive=10, precision=1.0)
    path.write_text(json.dumps(report))
    assert "| auto_action precision | 1.000 | ≥ 0.930 | skipped (n < 30) |" in RENDERER.render(
        run_dir
    )


def test_fdr_and_fpr_keep_their_distinct_denominators(run_dir: Path) -> None:
    path = run_dir / "report.json"
    report = json.loads(path.read_text())
    fdr = {
        "with_identity_term": {
            "n_false_positive": 5,
            "n_predicted_positive": 50,
            "false_discovery_rate": 0.1,
        },
        "without_identity_term": {
            "n_false_positive": 5,
            "n_predicted_positive": 50,
            "false_discovery_rate": 0.1,
        },
        "false_discovery_rate_ratio": 1.0,
        "false_discovery_rate_ratio_ci95": [0.5, 1.5],
    }
    fpr = {
        "with_identity_term": {
            "n_false_positive": 5,
            "n_actual_negative": 500,
            "false_positive_rate": 0.01,
        },
        "without_identity_term": {
            "n_false_positive": 5,
            "n_actual_negative": 1000,
            "false_positive_rate": 0.005,
        },
        "false_positive_rate_ratio": 2.0,
        "false_positive_rate_ratio_ci95": [0.8, 4.0],
    }
    report["identity_false_positives"]["test"]["final_tier"] = {
        "predicted_positive": fdr,
        "clean_negative_false_positives": {"predicted_positive": fpr},
    }
    path.write_text(json.dumps(report))
    text = RENDERER.render(run_dir)
    assert "0.100 (5/50)" in text
    assert "0.010 (5/500)" in text
    assert "0.005 (5/1000)" in text
    assert "FDR) divides wrong positive decisions by all positive decisions" in text
    assert "FPR) uses all actually negative comments" in text


def test_queue_uses_run_high_risk_definition_and_named_paired_metrics() -> None:
    sim = {
        "assumptions": {
            "reviewers": 2,
            "handle_minutes": 2,
            "horizon_hours": 1,
            "capacity_per_hour": 60,
            "queued_jobs": 100,
            "queued_high_risk": 10,
            "high_risk": "harm proxy >= 9",
            "seeds": [1],
        },
        "summary": {"fifo@10": {"load_per_hour": 10, "strategy": "fifo"}},
        "paired": {
            "severity_vs_fifo@10": {
                "harm_per_reviewer_hour": {
                    "mean_diff": 1.0,
                    "n_seeds_first_better": 1,
                    "n_seeds": 1,
                }
            }
        },
    }
    text = "\n".join(RENDERER._simulation_lines(sim))
    assert "harm proxy >= 9" in text
    assert "high-risk wait p90 difference" in text
    assert "every metric" not in text
    assert "queue clears" not in text


@pytest.fixture
def priority_run(run_dir: Path) -> Path:
    policy = {
        "version": 2,
        "decision_mode": "human_confirmation",
        "tiers": {"allow": 1, "human_review": 2, "priority_review": 3},
        "tier_precision_floors": {"human_review": 0.90, "priority_review": 0.95},
        "gates": {
            "routing": {
                "priority_review_precision_floor": None,
                "human_review_precision_floor": None,
                "harm_per_reviewer_hour_vs_fifo_min": 1.0,
                "high_risk_wait_p90_vs_fifo_max": 1.0,
                "high_risk_handled_vs_fifo_min": 1.0,
                "queue_depth_p95_max": None,
                "reviewer_utilization_min": 0.60,
            },
            "fairness": {"identity_false_discovery_rate_ratio_max": 1.25},
            "min_predicted_positives_for_precision": 30,
        },
    }
    (run_dir / "policy.yaml").write_text(yaml.safe_dump(policy))
    report = json.loads((run_dir / "report.json").read_text())
    report["decision_contract"] = {
        "mode": "human_confirmation",
        "requires_human_confirmation": True,
        "automatic_actions": 0,
    }
    report["review_workload"] = {
        "n_total": 200,
        "n_requires_human_review": 100,
        "n_priority_review": 40,
        "n_human_review": 60,
        "review_fraction": 0.5,
        "n_truly_positive": 80,
        "precision": 0.8,
        "precision_ci95": [0.71, 0.87],
    }
    report["tiers"] = {
        "priority_review": {
            "n_predicted_positive": 40,
            "coverage": 0.2,
            "precision": 0.95,
            "precision_ci95": [0.83, 0.99],
        },
        "human_review": {
            "n_predicted_positive": 60,
            "coverage": 0.3,
            "precision": 0.7,
            "precision_ci95": [0.58, 0.8],
        },
    }
    report["per_label"] = {
        "toxic": {
            "positives": 80,
            "average_precision": 0.8,
            "roc_auc": 0.9,
            "at_priority_review": {"threshold": 0.8, "precision": 0.95, "recall": 0.4},
            "at_human_review": {"threshold": 0.6, "precision": 0.7, "recall": 0.5},
        }
    }
    report["simulation"] = {
        "primary": {"strategy": "priority", "load_per_hour": 108.0},
        "thesis": {"strategy": "priority", "load_per_hour": 180.0},
        "harm_per_reviewer_hour": {"router": 2.0, "fifo": 2.0},
        "high_risk_wait_p90": {"router": 2.0, "fifo": 2.0},
        "high_risk_handled": {
            "primary": {"router": 80, "fifo": 100, "load_per_hour": 108.0},
            "thesis": {"router": 100, "fifo": 100, "load_per_hour": 180.0},
        },
        "reviewer_utilization": 0.9,
        "assumptions": {
            "reviewers": 4,
            "handle_minutes": 2,
            "horizon_hours": 8,
            "capacity_per_hour": 120,
            "queued_jobs": 100,
            "queued_high_risk": 20,
            "high_risk": "harm proxy >= 5",
            "seeds": [1],
            "arrivals": "post-admission review jobs",
        },
        "summary": {
            f"{strategy}@{load}": {"load_per_hour": load, "strategy": strategy}
            for load in (108, 180)
            for strategy in ("fifo", "prob", "severity", "priority")
        },
    }
    (run_dir / "report.json").write_text(json.dumps(report))
    return run_dir


def test_priority_contract_targets_and_workload_precede_results(priority_run: Path) -> None:
    text = RENDERER.render(priority_run)
    assert text.startswith("## Review-routing results")
    assert text.index("Decision contract and evaluation criteria") < text.index("**Per label.")
    assert "Every moderation action requires human confirmation" in text
    assert "0.95 for priority_review" in text
    assert "empirical targets on the selection split" in text
    assert "100 of 200 comments requiring review" in text
    assert "40 priority reviews and 60 standard reviews" in text
    assert "Recorded automatic actions: 0" in text
    assert "| priority_review precision | 0.950 | diagnostic; no test floor | not set |" in text
    assert "| human_review precision | 0.700 | diagnostic; no test floor | not set |" in text
    assert "| priority thr | P@priority | R@priority |" in text
    assert "auto_action precision" not in text
    assert "wrong-trigger" not in text


def test_priority_queue_counts_all_flagged_rows_and_keeps_completion_check(
    priority_run: Path,
) -> None:
    text = RENDERER.render(priority_run)
    assert "sampled from both priority_review and human_review" in text
    assert "post-admission review jobs per hour" in text
    assert "| 108 | priority |" in text and "| 180 | priority |" in text
    assert (
        "high-risk completed, priority / FIFO at 108.000/h (primary) | 0.800 | ≥ 1.000 | **red**"
        in text
    )
    assert (
        "high-risk completed, priority / FIFO at 180.000/h (thesis) | 1.000 | ≥ 1.000 | green"
        in text
    )
    assert "reviewer_utilization at 108.000/h | 0.900 | ≥ 0.600 | green" in text
    assert "not statistical non-inferiority tests" in text
    assert "≥ 1.120" not in text and "≤ 0.550" not in text


@pytest.mark.parametrize("router_wait,status", [(1.0, "**red**"), (0.0, "unavailable")])
def test_zero_fifo_wait_cannot_hide_a_positive_priority_wait(
    priority_run: Path, router_wait: float, status: str
) -> None:
    path = priority_run / "report.json"
    report = json.loads(path.read_text())
    report["simulation"]["high_risk_wait_p90"] = {"router": router_wait, "fifo": 0.0}
    path.write_text(json.dumps(report))
    text = RENDERER.render(priority_run)
    row = next(line for line in text.splitlines() if line.startswith("| high_risk_wait_p90,"))
    assert row.endswith(f"| {status} |")
    if router_wait > 0:
        assert "1.000 min / zero FIFO wait" in row


def test_priority_fairness_uses_all_final_review_tiers(priority_run: Path) -> None:
    path = priority_run / "report.json"
    report = json.loads(path.read_text())
    section = {
        "with_identity_term": {"n_predicted_positive": 100},
        "without_identity_term": {"n_predicted_positive": 100},
        "false_discovery_rate_ratio": 1.5,
        "false_discovery_rate_ratio_ci95": [1.4, 1.6],
    }
    report["identity_false_positives"]["test"]["final_tier"] = {
        tier: section for tier in ("predicted_positive", "priority_review", "human_review")
    }
    path.write_text(json.dumps(report))
    text = RENDERER.render(priority_run)
    rows = [line for line in text.splitlines() if line.startswith("| identity FDR ratio")]
    assert len(rows) == 3
    assert all("final tier" in row and row.endswith("| **red** |") for row in rows)
    assert "For both priority and standard human review" in text
    assert "when no label is true" in text
    assert "none of its triggering labels" not in text


def test_report_cannot_claim_human_only_contract_with_automatic_actions(priority_run: Path) -> None:
    path = priority_run / "report.json"
    report = json.loads(path.read_text())
    report["decision_contract"]["automatic_actions"] = 1
    path.write_text(json.dumps(report))
    text = RENDERER.render(priority_run)
    row = next(
        line
        for line in text.splitlines()
        if line.startswith("| human confirmation before every moderation action")
    )
    assert row.endswith("| **red** |")


def test_legacy_snapshot_preserves_automatic_enforcement_meaning(run_dir: Path) -> None:
    text = RENDERER.render(run_dir)
    assert text.startswith("## Round-1 results")
    assert "auto_action" in text
    assert "none of its triggering labels is true" in text
    assert "priority_review" not in text
    assert "Every moderation action requires human confirmation" not in text


def test_four_category_gates_use_named_capacity_metric(priority_run: Path) -> None:
    path = priority_run / "policy.yaml"
    policy = yaml.safe_load(path.read_text())
    gates = policy["gates"]
    gates["capacity"] = {
        key: value for key, value in gates["routing"].items() if "precision_floor" not in key
    }
    gates["routing"] = {
        "priority_review_precision_floor": None, "human_review_precision_floor": None
    }
    gates["capacity"].update(completion_ratio_min=0.97, backlog_end_max=10, wait_p50_max_min=1.0)
    gates["classification"] = {
        "per_label_average_precision": {"toxic": {"floor": 0.75}},
        "per_label_precision_at_human_threshold": {"toxic": {"floor": 0.6}},
        "volume_overshoot_max": {"toxic": {"ceiling": 1.0}},
    }
    path.write_text(yaml.safe_dump(policy))
    report = json.loads((priority_run / "report.json").read_text())
    report["per_label"]["toxic"]["at_human_review"]["n_predicted_positive"] = 40
    report["prevalence_shift"] = {"toxic": {"volume_overshoot_model": 1.2}}
    sim = report["simulation"]
    sim.update(completion_ratio=0.99, backlog_end=1.0)
    # A completion-latency headline must not replace the named wait p50 gate.
    sim["headline"] = {"selected": {"router": 99}}
    sim["time_metrics"] = {"priority@108": {"pooled_over_seeds": {"wait": {"p50": 0.4}}}}
    text = "\n".join(RENDERER._gate_lines(report, gates, path, "priority_review"))
    assert "AP (1 configured labels) | see label table | snapshot/override floors | green" in text
    assert "toxic precision at human threshold | 0.700 | ≥ 0.600; n ≥ 30 | green" in text
    assert "toxic predicted volume overshoot | 1.200 | ≤ 1.000 | **red**" in text
    assert "wait_p50 at 108.000/h | 0.400 | ≤ 1.000 | green" in text
    assert "completion_ratio at 108.000/h | 0.990 | ≥ 0.970 | green" in text
    assert "(primary) | 0.800 | ≥ 1.000 | **red**" in text


def test_record_time_tables_state_populations_and_sample_counts(priority_run: Path) -> None:
    import numpy as np

    from review_router.pipeline import _headline, load_config
    from review_router.simulate import Scenario, SimConfig, simulate, time_metrics

    sim = json.loads((priority_run / "report.json").read_text())["simulation"]
    scenario = Scenario(1, 108, np.array([0, 0]), np.array([0.0, 0.5]), np.array([2.0, 2.0]))
    _, records = simulate(
        scenario, np.zeros(1), np.ones(1), np.ones(1, dtype=bool),
        SimConfig(reviewers=1, horizon_hours=3 / 60), "fifo"
    )
    tm = time_metrics(records)
    sim["time_metrics"] = {f"{s}@108": {"pooled_over_seeds": tm} for s in ("priority", "fifo")}
    sim["headline"] = _headline(
        sim["time_metrics"], load_config(ROOT / "configs" / "baseline.yaml")
    )
    text = "\n".join(RENDERER._simulation_lines(sim, "priority_review"))
    assert "all jobs that started, including reviews still in progress" in text
    assert "completed items" not in text
    assert "1 / 1 / 0" in text
    assert "n=2 / 2" in text
    assert "no / no" in text
    assert "subsets may differ" in text
