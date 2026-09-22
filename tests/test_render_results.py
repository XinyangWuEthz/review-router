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
