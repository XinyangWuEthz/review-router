from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from review_router.data import LABELS
from review_router.pipeline import _route, load_config, run
from review_router.policy import load_policy
from review_router.thresholds import TierThresholds

ROOT = Path(__file__).resolve().parent.parent


def _config(tmp_path: Path, corpus: Path) -> Path:
    cfg = tmp_path / "configs" / "tiny.yaml"
    cfg.parent.mkdir()
    cfg.write_text(
        f"""
label: tiny
data_dir: {corpus}
output_dir: {tmp_path / "reports"}
seed: 3
split: {{train: 0.6, calib: 0.2, thresh: 0.2}}
model: {{max_features: 5000, ngram_max: 1, min_df: 1, C: 2.0}}
simulation:
  reviewers: 2
  handle_minutes: 2.0
  horizon_hours: 1
  loads_per_hour: [30, 90]
  seeds: [1, 2]
  high_risk_min_weight: 5
  primary: {{load_per_hour: 90, strategy: severity}}
""",
        encoding="utf-8",
    )
    return cfg


def test_shipped_configs_parse() -> None:
    for name in ("baseline", "smoke"):
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        assert cfg.policy_path.is_file()
        assert cfg.sim.capacity_per_hour == 120


def test_unknown_primary_strategy_is_rejected(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        "data_dir: x\nsplit: {train: 0.6, calib: 0.2, thresh: 0.2}\n"
        "simulation: {primary: {strategy: random}}\n"
    )
    with pytest.raises(ValueError, match="unknown primary strategy"):
        load_config(cfg)


@pytest.mark.parametrize("simulation, message", [
    ("{loads_per_hour: []}", "positive finite"),
    ("{loads_per_hour: [60], primary: {load_per_hour: 108}}", "must appear"),
    ("{seeds: []}", "nonnegative integers"),
])
def test_invalid_simulation_selection_is_rejected(
    tmp_path: Path, simulation: str, message: str
) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        "data_dir: x\nsplit: {train: 0.6, calib: 0.2, thresh: 0.2}\n"
        f"simulation: {simulation}\n"
    )
    with pytest.raises(ValueError, match=message):
        load_config(cfg)


def test_human_protection_rule_overrides_automatic_model_tier() -> None:
    thresholds = TierThresholds({
        "auto_action": {label: 0.8 for label in LABELS},
        "human_review": {label: 0.5 for label in LABELS},
    }, LABELS)
    probabilities = np.array([[0.95, 0.1, 0.1, 0.4, 0.1, 0.1]])
    result = _route(
        probabilities, np.array([[1, 0, 0, 1, 0, 0]]), np.zeros(1),
        thresholds, load_policy(),
    )
    assert result["model_tier"].tolist() == ["auto_action"]
    assert result["final_tier"].tolist() == ["human_review"]
    assert result["rule_ids"] == ["R101_rare_high_harm_never_auto"]


def test_end_to_end_run_writes_every_artefact(synthetic_corpus: Path, tmp_path: Path) -> None:
    cfg_path = _config(tmp_path, synthetic_corpus)
    run_dir = run(cfg_path)
    for name in (
        "manifest.json",
        "config.yaml",
        "policy.yaml",
        "splits.csv",
        "thresholds.json",
        "model.pkl",
        "predictions.csv",
        "report.json",
        "report.md",
    ):
        assert (run_dir / name).is_file(), name
    assert (run_dir / "config.yaml").read_bytes() == cfg_path.read_bytes()
    assert (run_dir / "policy.yaml").read_bytes() == load_config(cfg_path).policy_path.read_bytes()

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert set(manifest["data_sha256"]) == {"train", "test", "test_labels"}
    assert set(manifest["split_label_counts"]) == {"train", "calib", "thresh", "test_scored"}
    assert manifest["dependencies"]["scikit-learn"]
    assert manifest["seed"] == 3 and manifest["simulation_seeds"] == [1, 2]

    report = json.loads((run_dir / "report.json").read_text())
    assert report["synthetic"] is True
    assert report["eval_slice"] == "synthetic scored test rows"
    assert "SYNTHETIC pipeline check" in (run_dir / "report.md").read_text()
    assert set(report["threshold_selection"]["tiers"]) == {"auto_action", "human_review", "allow"}
    assert set(report) >= {"per_label", "tiers", "rules", "consistency", "simulation"}
    assert set(report["per_label"]) == {
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    }
    assert set(report["tiers"]) == {"auto_action", "human_review", "allow"}
    sim = report["simulation"]
    assert set(sim["harm_per_reviewer_hour"]) == {"router", "fifo"}
    assert {row["strategy"] for row in sim["table"]} == {"fifo", "prob", "severity"}
    assert len(sim["table"]) == 2 * 2 * 3  # loads x seeds x strategies

    predictions = pd.read_csv(run_dir / "predictions.csv")
    assert len(predictions) == 1200
    assert set(predictions["final_tier"]) <= {"auto_action", "human_review", "allow"}
    counts = predictions["final_tier"].value_counts().to_dict()
    for tier, stats in report["tiers"].items():
        assert stats["n_predicted_positive"] == counts.get(tier, 0)


def test_strategies_replay_identical_arrivals(synthetic_corpus: Path, tmp_path: Path) -> None:
    run_dir = run(_config(tmp_path, synthetic_corpus))
    rows = json.loads((run_dir / "report.json").read_text())["simulation"]["table"]
    by_key: dict[tuple[float, int], set[int]] = {}
    for row in rows:
        by_key.setdefault((row["load_per_hour"], row["seed"]), set()).add(row["n_arrivals"])
    assert all(len(v) == 1 for v in by_key.values()), "arrivals differ across strategies"


def test_cli_prints_run_dir(synthetic_corpus: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path, synthetic_corpus)
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), "--config", str(cfg)],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    first_line = out.stdout.splitlines()[0]
    assert Path(first_line).is_dir()
    assert "## Queue simulation" in out.stdout
    manifest = json.loads((Path(first_line) / "manifest.json").read_text())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    assert manifest["git_commit"] == commit


def test_repeated_runs_reproduce_predictions_thresholds_and_metrics(
    synthetic_corpus: Path, tmp_path: Path
) -> None:
    config = _config(tmp_path, synthetic_corpus)
    first, second = run(config), run(config)
    assert first != second
    for name in ("splits.csv", "predictions.csv", "thresholds.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    reports = [json.loads((path / "report.json").read_text()) for path in (first, second)]
    for report in reports:
        report.pop("run_id")
    assert reports[0] == reports[1]
