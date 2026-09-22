from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from review_router.data import LABELS
from review_router.pipeline import _identity_concentration, _route, load_config, run
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


@pytest.mark.parametrize(
    "simulation, message",
    [
        ("{loads_per_hour: []}", "positive finite"),
        ("{loads_per_hour: [60], primary: {load_per_hour: 108}}", "must appear"),
        (
            "{loads_per_hour: [60], primary: {load_per_hour: 60}, thesis_load_per_hour: 180}",
            "thesis_load_per_hour must appear",
        ),
        ("{seeds: []}", "nonnegative integers"),
    ],
)
def test_invalid_simulation_selection_is_rejected(
    tmp_path: Path, simulation: str, message: str
) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        f"data_dir: x\nsplit: {{train: 0.6, calib: 0.2, thresh: 0.2}}\nsimulation: {simulation}\n"
    )
    with pytest.raises(ValueError, match=message):
        load_config(cfg)


def test_human_protection_rule_overrides_automatic_model_tier() -> None:
    thresholds = TierThresholds(
        {
            "auto_action": {label: 0.8 for label in LABELS},
            "human_review": {label: 0.5 for label in LABELS},
        },
        LABELS,
    )
    probabilities = np.array([[0.95, 0.1, 0.1, 0.4, 0.1, 0.1]])
    result = _route(
        probabilities,
        np.array([[1, 0, 0, 1, 0, 0]]),
        np.zeros(1),
        thresholds,
        load_policy(),
    )
    assert result["model_tier"].tolist() == ["auto_action"]
    assert result["final_tier"].tolist() == ["human_review"]
    assert result["rule_ids"] == ["R101_rare_high_harm_never_auto"]


def test_identity_diagnostics_distinguish_discovery_positive_and_omission_rates() -> None:
    tiers = np.array(
        ["auto_action", "allow", "auto_action", "allow", "allow"]
        + ["auto_action", "auto_action", "human_review", "allow", "allow"]
    )
    any_true = np.array([True, True, False, False, False, True, False, False, False, False])
    routing = {"any_true": any_true}
    for basis, correct_key in (
        ("population_tier", "correct_population"),
        ("model_tier", "correct_model"),
        ("final_tier", "correct"),
    ):
        routing[basis] = tiers
        routing[correct_key] = any_true
    result = _identity_concentration(
        routing, np.array([1] * 5 + [0] * 5), ["gay"] * 5 + ["article"] * 5
    )
    final = result["final_tier"]
    discovery = final["predicted_positive"]
    assert discovery["with_identity_term"]["false_discovery_rate"] == 0.5
    assert discovery["without_identity_term"]["false_discovery_rate"] == pytest.approx(2 / 3)
    assert discovery["false_discovery_rate_ratio"] == 0.75
    clean = final["clean_negative_false_positives"]["predicted_positive"]
    assert clean["with_identity_term"]["n_actual_negative"] == 3
    assert clean["with_identity_term"]["false_positive_rate"] == pytest.approx(1 / 3)
    assert clean["without_identity_term"]["false_positive_rate"] == 0.5
    assert clean["false_positive_rate_ratio"] == pytest.approx(2 / 3)
    omission = final["allow_false_omission"]["with_identity_term"]
    assert omission["n_allowed"] == 3
    assert omission["false_omission_rate"] == pytest.approx(1 / 3)
    assert result["identity_terms"] == sorted(result["identity_terms"])
    assert "gay" in result["identity_terms"]


def test_wrong_trigger_counts_as_false_discovery_but_not_a_clean_false_positive() -> None:
    tiers = np.array(["auto_action", "allow", "auto_action", "allow"])
    any_true = np.array([True, False, True, False])
    correct = np.array([False, False, True, False])
    routing = {
        "any_true": any_true,
        "population_tier": tiers,
        "model_tier": tiers,
        "final_tier": tiers,
        "correct_population": correct,
        "correct_model": correct,
        "correct": correct,
    }
    final = _identity_concentration(
        routing, np.array([1, 1, 0, 0]), ["gay", "gay", "article", "article"]
    )["final_tier"]
    assert final["auto_action"]["with_identity_term"]["false_discovery_rate"] == 1.0
    clean = final["clean_negative_false_positives"]["auto_action"]["with_identity_term"]
    assert clean["n_actual_negative"] == 1
    assert clean["false_positive_rate"] == 0.0


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
    assert set(sim["harm_per_reviewer_hour"]) == {"router", "fifo", "load_per_hour"}
    assert sim["harm_per_reviewer_hour"]["load_per_hour"] == 90  # thesis defaults to max load
    assert sim["high_risk_wait_p90"]["load_per_hour"] == 90
    identity = report["identity_false_positives"]
    assert set(identity) == {"threshold_selection", "test"}
    for name, section in identity.items():
        src = report["threshold_selection"] if name == "threshold_selection" else report
        assert set(section) >= {"population_tier", "model_tier", "final_tier"}
        assert section["identity_terms"] == sorted(section["identity_terms"])
        for basis in ("population_tier", "model_tier", "final_tier"):
            assert section[basis]["base_rate"]["with_identity_term"]["n_rows"] >= 0
        for basis in ("model_tier", "final_tier"):
            pooled = section[basis]["predicted_positive"]
            n_pos = sum(
                pooled[g]["n_predicted_positive"]
                for g in ("with_identity_term", "without_identity_term")
            )
            tiers_key = "tiers" if basis == "final_tier" else "tiers_model_only"
            assert n_pos == sum(
                src[tiers_key][t]["n_predicted_positive"] for t in ("auto_action", "human_review")
            ), (name, basis)
            for tier in ("auto_action", "human_review"):
                per_tier = section[basis][tier]
                assert (
                    per_tier["with_identity_term"]["n_predicted_positive"]
                    + per_tier["without_identity_term"]["n_predicted_positive"]
                    == src[tiers_key][tier]["n_predicted_positive"]
                ), (name, basis, tier)
    assert identity["test"]["final_tier"]["by_term"], "by-term breakdown missing"
    sub = report["subgroup_thresholds"]
    assert sub["threshold_selection"]["declared"] == [
        {"tier": "auto_action", "signal": "identity_term_present"}
    ]
    for split in ("threshold_selection", "test"):
        st = sub[split]["auto_action"]
        assert st["n_removed_by_subgroup_threshold"] <= st["n_population_tier"]
    assert set(report["threshold_selection"]["per_label"]) == set(LABELS)
    paired = sim["paired"]
    assert "severity_vs_fifo@90" in paired and "severity_vs_prob@30" in paired
    for entry in paired.values():
        for stats in entry.values():
            if stats is not None:
                assert stats["n_seeds_first_better"] + stats["n_seeds_tied"] <= stats["n_seeds"]

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


def test_render_results_replaces_readme_block(synthetic_corpus: Path, tmp_path: Path) -> None:
    run_dir = run(_config(tmp_path, synthetic_corpus))
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- results:start -->\nold\n<!-- results:end -->\nouttro\n")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "render_results.py"),
            str(run_dir),
            "--readme",
            str(readme),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    text = readme.read_text()
    assert text.startswith("intro\n<!-- results:start -->\n## Round-1 results")
    assert text.endswith("<!-- results:end -->\nouttro\n")
    assert "\nold\n" not in text
    assert "SYNTHETIC corpus" in text
    assert "| gate | measured | requirement | status |" in text
