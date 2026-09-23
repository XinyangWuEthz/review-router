from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from record import collect_runs
from record.collect_priority import ARTIFACT_NAMES, collect
from review_router.data import LABELS, file_sha256

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def saved_run(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "run"
    run.mkdir()
    (run / "policy.yaml").write_bytes((ROOT / "review_router" / "policy.yaml").read_bytes())
    (run / "config.yaml").write_text("evaluate_test: true\n")
    thresholds = {
        tier: {label: value if label == "toxic" else None for label in LABELS}
        for tier, value in (("priority_review", 0.95), ("human_review", 0.8))
    }
    report = {
        "run_id": "fixture-v3",
        "synthetic": False,
        "policy_version": 3,
        "evaluate_test": True,
        "high_risk_min_weight": 5.0,
        "decision_contract": {"mode": "human_confirmation", "automatic_actions": 0},
        "tiers": {},
        "tiers_model_only": {},
        "per_label": {},
        "rules": {},
        "consistency": {},
        "allow_false_negatives": {},
        "simulation": {},
        "threshold_selection": {
            "tiers": {},
            "alternatives": {"cumulative_precision": {"thresholds": thresholds}},
        },
        "agreement_by_confidence": {"threshold_selection": {"n_rows": 10}},
        "ranking": {"n_pool": 2},
    }
    (run / "report.json").write_text(json.dumps(report))
    manifest = {
        "run_id": "fixture-v3",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "finished_utc": "2026-09-23T12:00:00+00:00",
        "evaluate_test": True,
        "data_sha256": {},
        "dependencies": {},
        "split_label_counts": {"test_scored": {"rows": 3}},
        "policy_sha256": file_sha256(run / "policy.yaml"),
        "config_sha256": file_sha256(run / "config.yaml"),
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    predictions: dict[str, Any] = {
        "id": ["a", "b", "c"],
        "identity_term_present": [0, 0, 0],
        "final_tier": ["human_review", "priority_review", "allow"],
    }
    for label in LABELS:
        predictions[f"y_{label}"] = [int(label == "toxic"), int(label == "threat"), 0]
        predictions[f"p_{label}"] = [0.96, 0.2, 0.1] if label == "toxic" else [0.0, 0.0, 0.0]
    predictions["p_threat"] = [0.0, 0.4, 0.0]
    pd.DataFrame(predictions).to_csv(run / "predictions.csv", index=False)
    (run / "simulation_jobs.csv").write_text("strategy,comment_id\nseverity,a\n")
    (run / "model.pkl").write_bytes(b"not a model: collection must not load it")
    ci = tmp_path / "ci.json"
    ci.write_text(
        json.dumps(
            {
                "headSha": manifest["git_commit"],
                "status": "completed",
                "conclusion": "success",
                "url": "https://github.com/example/review-router/actions/runs/123",
                "jobs": [
                    {"name": name, "status": "completed", "conclusion": "success"}
                    for name in ("evaluate", "negative-check")
                ],
            }
        )
    )
    return run, ci


def _update(path: Path, key: str, value: Any) -> None:
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))


def test_collect_preserves_diagnostics_and_replays_frozen_cumulative_thresholds(
    saved_run: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_fitting(*args: Any, **kwargs: Any) -> None:
        pytest.fail("collection must never fit thresholds")

    monkeypatch.setattr("review_router.pipeline.select_thresholds", no_fitting)
    run, ci = saved_run
    out = collect(run, ci)
    assert set(out) == {
        "run",
        "threshold_selection",
        "agreement_by_confidence",
        "ranking",
        "ci",
        "artifact_sha256",
        "test_cumulative",
    }
    source = json.loads((run / "report.json").read_text())
    for name in ("threshold_selection", "agreement_by_confidence", "ranking"):
        assert out[name] == source[name]
    assert out["ci"] == json.loads(ci.read_text())
    assert out["artifact_sha256"] == {name: file_sha256(run / name) for name in ARTIFACT_NAMES}
    alt = out["test_cumulative"]
    assert alt["same_admitted_rows"] is True
    assert alt["n_changed_band"] == 1
    assert alt["review_workload"]["n_priority_review"] == 2
    assert alt["review_workload"]["n_requires_human_review"] == 2
    assert alt["review_workload"]["precision"] == 1.0
    assert alt["tiers"]["priority_review"]["n_predicted_positive"] == 2
    assert "previously inspected test set, not fresh validation" in alt["note"]
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize(
    ("source", "key", "value", "message"),
    [
        ("ci", "headSha", "b" * 40, "headSha"),
        ("ci", "status", "in_progress", "CI run"),
        ("ci", "conclusion", "failure", "CI run"),
        ("ci", "jobs", [], "evaluate"),
        ("manifest", "git_dirty", True, "clean run"),
        ("manifest", "evaluate_test", False, "test evaluation"),
        ("report", "synthetic", True, "real, non-exploratory"),
        ("report", "exploratory", True, "real, non-exploratory"),
        ("report", "policy_version", 2, "policy version 3"),
        ("report", "evaluate_test", False, "test evaluation"),
    ],
)
def test_collect_rejects_unverified_runs(
    saved_run: tuple[Path, Path], source: str, key: str, value: Any, message: str
) -> None:
    run, ci = saved_run
    _update(ci if source == "ci" else run / f"{source}.json", key, value)
    with pytest.raises(ValueError, match=message):
        collect(run, ci)


def test_collect_requires_successful_negative_check(saved_run: tuple[Path, Path]) -> None:
    run, ci = saved_run
    metadata = json.loads(ci.read_text())
    metadata["jobs"][1]["conclusion"] = "failure"
    ci.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="negative-check"):
        collect(run, ci)


def test_same_admitted_rows_compares_identity_not_just_pool_size(
    saved_run: tuple[Path, Path],
) -> None:
    run, ci = saved_run
    predictions = pd.read_csv(run / "predictions.csv")
    predictions["final_tier"] = ["allow", "priority_review", "human_review"]
    predictions.to_csv(run / "predictions.csv", index=False)
    out = collect(run, ci)["test_cumulative"]
    assert out["review_workload"]["n_requires_human_review"] == 2
    assert out["same_admitted_rows"] is False


def test_collect_rejects_a_changed_snapshot(saved_run: tuple[Path, Path]) -> None:
    run, ci = saved_run
    (run / "config.yaml").write_text("evaluate_test: true\n# changed after the run\n")
    with pytest.raises(ValueError, match="config.yaml does not match"):
        collect(run, ci)


def test_collect_rejects_fractional_labels_before_integer_conversion(
    saved_run: tuple[Path, Path],
) -> None:
    run, ci = saved_run
    predictions = pd.read_csv(run / "predictions.csv")
    predictions["y_toxic"] = [1.5, 0.0, 0.0]
    predictions.to_csv(run / "predictions.csv", index=False)
    with pytest.raises(ValueError, match="must be binary"):
        collect(run, ci)


def test_historical_collector_cannot_overwrite_step3_with_v3(
    saved_run: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _ = saved_run
    historical = tmp_path / "human_review_run.json"
    historical.write_text("historical step 3 record\n")
    monkeypatch.setattr(collect_runs, "HERE", tmp_path)
    monkeypatch.setattr("sys.argv", ["collect_runs.py", "--human-review", str(run)])
    with pytest.raises(SystemExit) as exc:
        collect_runs.main()
    assert exc.value.code == 2
    assert historical.read_text() == "historical step 3 record\n"
