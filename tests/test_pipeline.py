from __future__ import annotations

import json
import pickle
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from review_router.data import LABELS, Corpus, load_corpus
from review_router.pipeline import (
    _headline,
    _identity_concentration,
    _route,
    _run_simulation,
    _segment_check,
    _selection_notes,
    evaluate_scores,
    load_config,
    queue_inputs,
    run,
)
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
  primary: {{load_per_hour: 90, strategy: priority}}
""",
        encoding="utf-8",
    )
    return cfg


def test_shipped_configs_parse() -> None:
    for name in ("baseline", "smoke", "dev"):
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        assert cfg.evaluate_test is (name != "dev")
        assert cfg.policy_path.is_file()
        assert cfg.sim.capacity_per_hour == 120
        assert cfg.model.analyzer == "word"
        assert cfg.primary_strategy == "severity"
        assert load_policy(cfg.policy_path).decision_mode == "human_confirmation"


def test_paired_keys_put_the_primary_ordering_first() -> None:
    from review_router.pipeline import _ORDER_DESCRIPTIONS, _paired
    from review_router.simulate import STRATEGIES

    rows = [
        {"load_per_hour": 180.0, "seed": seed, "strategy": strategy, "high_risk_handled": rank}
        for seed in (1, 2)
        for rank, strategy in enumerate(STRATEGIES)
    ]
    legacy = {
        "priority_vs_fifo",
        "priority_vs_prob",
        "priority_vs_severity",
        "prob_vs_fifo",
        "severity_vs_fifo",
        "severity_vs_prob",
    }
    assert set(_paired(rows, "priority")) == {f"{key}@180" for key in legacy}
    by_severity = _paired(rows, "severity")
    assert list(by_severity) == [
        "severity_vs_fifo@180",
        "severity_vs_prob@180",
        "severity_vs_priority@180",
        "prob_vs_fifo@180",
        "priority_vs_fifo@180",
        "priority_vs_prob@180",
    ]
    # Differences are first minus second: severity has rank 2, priority rank 3.
    assert by_severity["severity_vs_priority@180"]["high_risk_handled"]["mean_diff"] == -1.0
    assert set(_ORDER_DESCRIPTIONS) == set(STRATEGIES)
    with pytest.raises(ValueError, match="unknown primary strategy"):
        _paired(rows, "random")


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


def test_high_risk_rule_promotes_regular_review_and_preserves_priority_review() -> None:
    thresholds = TierThresholds(
        {
            "priority_review": {label: 0.99 for label in LABELS},
            "human_review": {label: 0.5 for label in LABELS},
        },
        LABELS,
    )
    probabilities = np.array([[0.95, 0.1, 0.1, 0.4, 0.1, 0.1], [0.999, 0.1, 0.1, 0.4, 0.1, 0.1]])
    result = _route(
        probabilities,
        np.array([[1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0]]),
        np.zeros(2),
        thresholds,
        load_policy(),
    )
    assert result["model_tier"].tolist() == ["human_review", "priority_review"]
    assert result["final_tier"].tolist() == ["priority_review", "priority_review"]
    assert result["rule_ids"] == ["R101_high_risk_priority", "R101_high_risk_priority"]


def test_identity_diagnostics_distinguish_discovery_positive_and_omission_rates() -> None:
    tiers = np.array(
        ["priority_review", "allow", "priority_review", "allow", "allow"]
        + ["priority_review", "priority_review", "human_review", "allow", "allow"]
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


def test_priority_review_uses_any_true_label_even_when_the_trigger_label_is_wrong() -> None:
    thresholds = TierThresholds(
        {
            "priority_review": {label: 0.8 for label in LABELS},
            "human_review": {label: 0.5 for label in LABELS},
        },
        LABELS,
    )
    proba = np.zeros((2, len(LABELS)))
    proba[:, LABELS.index("toxic")] = 0.9
    y = np.zeros((2, len(LABELS)), dtype=int)
    y[0, LABELS.index("obscene")] = 1
    routing = _route(proba, y, np.array([1, 0]), thresholds, load_policy())
    assert routing["correct"].tolist() == [True, False]
    final = _identity_concentration(routing, np.array([1, 0]), ["gay", "article"])["final_tier"]
    assert final["priority_review"]["with_identity_term"]["false_discovery_rate"] == 0.0
    clean = final["clean_negative_false_positives"]["priority_review"]["without_identity_term"]
    assert clean["n_actual_negative"] == 1
    assert clean["false_positive_rate"] == 1.0


def test_simulation_queues_priority_items_when_no_regular_items_exist(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path, tmp_path / "unused-data"))
    proba = np.full((2, len(LABELS)), 0.1)
    y = np.zeros((2, len(LABELS)), dtype=int)
    y[0, LABELS.index("threat")] = 1
    sim, records = _run_simulation(
        proba,
        y,
        {
            "final_tier": np.array(["priority_review", "priority_review"]),
            "triggers": np.zeros_like(proba, dtype=bool),
            "rule_ids": ["", ""],
        },
        np.array(["a", "b"]),
        load_policy(),
        config,
    )
    assert sim["assumptions"]["queued_jobs"] == 2
    assert sim["assumptions"]["queued_priority_review"] == 2
    assert sim["assumptions"]["queued_human_review"] == 0
    assert {row["strategy"] for row in sim["table"]} == {"fifo", "prob", "severity", "priority"}
    assert {row["comment_id"] for row in records} == {"a", "b"}
    assert {row["strategy"] for row in records} == {"fifo", "prob", "severity", "priority"}


def test_queue_inputs_includes_both_human_tiers_and_excludes_allowed_items() -> None:
    proba = np.array([[0.99] * 6, [0.9] * 6, [0.1] * 6])
    truth = np.ones((3, 6), dtype=int)
    final = np.array(["allow", "human_review", "priority_review"])
    inputs = queue_inputs(proba, truth, final, load_policy(), 5)
    assert inputs["queued"].tolist() == [1, 2]
    assert inputs["priorities"]["severity"][0] > inputs["priorities"]["severity"][1]
    assert inputs["priorities"]["priority"][0] < inputs["priorities"]["priority"][1]


def test_simulation_skips_empty_review_pool(tmp_path: Path, synthetic_corpus: Path) -> None:
    config = load_config(_config(tmp_path, synthetic_corpus))
    routing = {"final_tier": np.array(["allow", "allow"])}
    sim, records = _run_simulation(
        np.zeros((2, 6)), np.zeros((2, 6)), routing, np.array(["a", "b"]), load_policy(), config
    )
    assert records == [] and sim["table"] == []
    assert sim["assumptions"]["queued_jobs"] == 0
    assert sim["assumptions"]["queue_tiers"] == ["priority_review", "human_review"]


@pytest.mark.parametrize("percentile", [50, 99])
def test_supplementary_high_risk_wait_always_uses_p90(
    tmp_path: Path, synthetic_corpus: Path, percentile: int
) -> None:
    config = replace(
        load_config(_config(tmp_path, synthetic_corpus)), headline_percentile=percentile
    )
    metrics = {
        "priority@90": {
            "pooled_over_seeds": {
                "wait": {"p50": 1.0, "p99": 8.0, "n": 200, "p99_reliable": True},
                "high_risk_wait": {"p50": 0.0, "p90": 2.0, "p99": 7.0, "n": 120},
            }
        },
        "fifo@90": {
            "pooled_over_seeds": {
                "wait": {"p50": 2.0, "p99": 16.0, "n": 200, "p99_reliable": True},
                "high_risk_wait": {"p50": 1.0, "p90": 8.0, "p99": 14.0, "n": 120},
            }
        },
    }
    result = _headline(metrics, config)
    assert result["selected"]["reduction"] == 0.5
    high_risk = result["supplementary_high_risk"]["wait_p90"]
    assert high_risk["router"] == 2.0 and high_risk["fifo"] == 8.0
    assert high_risk["reduction"] == 0.75
    assert "populations may differ" in result["population"]


def test_headline_with_no_samples_does_not_claim_fifo_zero(
    tmp_path: Path, synthetic_corpus: Path
) -> None:
    config = load_config(_config(tmp_path, synthetic_corpus))
    result = _headline({}, config)
    assert result["selected"]["reduction"] is None
    assert result["selected"]["note"] == "No comparable samples: a percentage is unavailable"


def _assert_agreement_sections(report: dict[str, Any]) -> None:
    """The step-3 reporting blocks: agreement by confidence and the selection diagnostics."""
    agreement = report["agreement_by_confidence"]
    assert {"calibration", "threshold_selection", "test", "optimism_gap"} <= set(agreement)
    assert agreement["calibration"] is not None and agreement["optimism_gap"] is not None
    thresh = agreement["threshold_selection"]
    assert set(thresh["per_label"]) == set(LABELS)
    pooled = thresh["pooled_review"]
    assert pooled[0]["lo"] is None and pooled[-1]["hi"] == 1.0
    assert sum(row["n"] for row in pooled) == thresh["n_rows"]
    assert abs(sum(row["coverage"] for row in pooled) - 1.0) < 1e-9
    assert "priority_band" in thresh and "n_rules_promoted" in thresh["priority_band"]
    assert set(thresh["strata"]) == {"identity_term_present", "oov_tercile"}
    assert len(agreement["strata_definition"]["oov_tercile"]["cut_points"]) == 2
    selection = report["threshold_selection"]
    assert selection["selection_rules"] == {
        "priority_review": "segment_agreement",
        "human_review": "cumulative_precision",
    }
    assert set(selection["segment_check"]) == set(LABELS)
    for check in selection["segment_check"].values():
        assert check["rule"] == "segment_agreement"
        assert (check["threshold"] is None) == (check["all_meet_floor"] is None)
        if check["threshold"] is not None:
            assert check["all_meet_floor"] is True
    alt = selection["alternatives"]["cumulative_precision"]
    assert set(alt["thresholds"]) >= {"priority_review", "human_review"}
    assert set(alt["tiers"]) == {"priority_review", "human_review", "allow"}
    assert set(alt["high_risk_by_band"]) == {"priority_review", "human_review", "allow"}
    assert set(selection["identity_share"]) == {"priority_review", "human_review"}
    for band in selection["identity_share"].values():
        assert set(band) == {"n", "n_with_identity_term", "share"}
    assert isinstance(selection["selection_notes"], list)
    priority = {k: v["threshold"] for k, v in selection["segment_check"].items()}
    if all(t is None for t in priority.values()):
        assert selection["selection_notes"] == [
            "no label qualifies for priority_review under the segment rule; the priority band "
            "contains rule promotions only"
        ]
    else:
        assert selection["selection_notes"] == []


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
        "simulation_jobs.csv",
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
    assert report["decision_contract"] == {
        "mode": "human_confirmation",
        "requires_human_confirmation": True,
        "automatic_actions": 0,
    }
    assert report["eval_slice"] == "synthetic scored test rows"
    assert "SYNTHETIC pipeline check" in (run_dir / "report.md").read_text()
    assert set(report["threshold_selection"]["tiers"]) == {
        "priority_review",
        "human_review",
        "allow",
    }
    assert set(report) >= {"per_label", "tiers", "rules", "consistency", "simulation"}
    assert set(report["per_label"]) == {
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    }
    assert set(report["tiers"]) == {"priority_review", "human_review", "allow"}
    _assert_agreement_sections(report)
    sim = report["simulation"]
    workload = report["review_workload"]
    admitted = sum(
        report["tiers"][t]["n_predicted_positive"] for t in ("priority_review", "human_review")
    )
    assert workload["n_requires_human_review"] == admitted == sim["assumptions"]["queued_jobs"]
    assert workload["n_priority_review"] + workload["n_human_review"] == admitted
    assert workload["review_fraction"] == pytest.approx(admitted / workload["n_total"])
    assert report["evaluate_test"] is True and manifest["evaluate_test"] is True
    selection = report["threshold_selection"]
    assert {"rules", "allow_false_negatives", "ranking", "cross_fitted"} <= set(selection)
    assert set(selection["allow_false_negatives"]) == set(LABELS)
    assert set(selection["rules"]["matched_per_rule"]) == {r.id for r in load_policy().rules}
    cross = selection["cross_fitted"]
    assert set(cross["tiers"]) == {"priority_review", "human_review", "allow"}
    assert sum(cross["fold_sizes"]) == selection["review_workload"]["n_total"]
    assert len(cross["thresholds_per_fold"]) == cross["k"] == 5
    for section in (workload, selection["review_workload"], cross["review_workload"]):
        bands = section["high_risk_by_band"]
        assert set(bands) == {"priority_review", "human_review", "allow"}
        assert sum(b["n"] for b in bands.values()) == section["n_total"]
        assert sum(b["n_high_risk"] for b in bands.values()) == section["n_high_risk_total"]
    queued_high_risk = sim["assumptions"]["queued_high_risk"]
    bands = workload["high_risk_by_band"]
    assert bands["priority_review"]["n_high_risk"] + bands["human_review"]["n_high_risk"] == (
        queued_high_risk
    )
    keys = {"fifo", "prob", "severity", "priority", "oracle_true_harm"}
    for pool in (report["ranking"], selection["ranking"]):
        assert set(pool) == {"n_pool", "n_high_risk", "fractions", "by_key", "note"}
        assert set(pool["by_key"]) == keys
    assert report["ranking"]["n_pool"] == admitted
    assert report["ranking"]["n_high_risk"] == queued_high_risk
    for pool in (report["ranking"], selection["ranking"]):
        if pool["n_high_risk"] > 0:
            oracle = pool["by_key"]["oracle_true_harm"]["at_fraction"]
            assert oracle[-1]["fraction"] == 1.0
            assert oracle[-1]["high_risk_share_at_k"] == 1.0
    markdown = (run_dir / "report.md").read_text()
    assert "## High-risk composition by band" in markdown
    assert "## Ranking diagnostics" in markdown
    assert "cross-fitted" in markdown
    assert sim["primary"]["strategy"] == "priority"
    assert set(sim["harm_per_reviewer_hour"]) == {"router", "fifo", "load_per_hour"}
    assert sim["harm_per_reviewer_hour"]["load_per_hour"] == 90  # thesis defaults to max load
    assert sim["high_risk_wait_p90"]["load_per_hour"] == 90
    for name in ("primary", "thesis"):
        assert sim["high_risk_arrived"][name]["router"] == sim["high_risk_arrived"][name]["fifo"]
        assert sim["high_risk_handled"][name]["load_per_hour"] == 90
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
                src[tiers_key][t]["n_predicted_positive"]
                for t in ("priority_review", "human_review")
            ), (name, basis)
            for tier in ("priority_review", "human_review"):
                per_tier = section[basis][tier]
                assert (
                    per_tier["with_identity_term"]["n_predicted_positive"]
                    + per_tier["without_identity_term"]["n_predicted_positive"]
                    == src[tiers_key][tier]["n_predicted_positive"]
                ), (name, basis, tier)
    assert identity["test"]["final_tier"]["by_term"], "by-term breakdown missing"
    sub = report["subgroup_thresholds"]
    assert sub["threshold_selection"]["declared"] == [
        {"tier": "priority_review", "signal": "identity_term_present"}
    ]
    for split in ("threshold_selection", "test"):
        st = sub[split]["priority_review"]
        assert st["n_removed_by_subgroup_threshold"] <= st["n_population_tier"]
    assert set(report["threshold_selection"]["per_label"]) == set(LABELS)
    paired = sim["paired"]
    assert "severity_vs_fifo@90" in paired and "severity_vs_prob@30" in paired
    assert "priority_vs_fifo@90" in paired and "priority_vs_severity@30" in paired
    for entry in paired.values():
        for stats in entry.values():
            if stats is not None:
                assert stats["n_seeds_first_better"] + stats["n_seeds_tied"] <= stats["n_seeds"]

    predictions = pd.read_csv(run_dir / "predictions.csv")
    assert len(predictions) == 1200
    assert set(predictions["final_tier"]) <= {"priority_review", "human_review", "allow"}
    assert (predictions["requires_human_review"] == (predictions["final_tier"] != "allow")).all()
    assert int(predictions["requires_human_review"].sum()) == admitted
    counts = predictions["final_tier"].value_counts().to_dict()
    for tier, stats in report["tiers"].items():
        assert stats["n_predicted_positive"] == counts.get(tier, 0)


def test_development_run_never_scores_the_test_rows(
    synthetic_corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from review_router.model import TfidfLogitModel

    cfg_path = _config(tmp_path, synthetic_corpus)
    full_run = run(cfg_path)
    cfg_path.write_text(cfg_path.read_text() + "evaluate_test: false\n")
    assert load_config(cfg_path).evaluate_test is False
    scored_batches: list[int] = []
    original = TfidfLogitModel.predict_proba

    def spy(self: TfidfLogitModel, texts: Any) -> np.ndarray:
        texts = list(texts)
        scored_batches.append(len(texts))
        return original(self, texts)

    monkeypatch.setattr(TfidfLogitModel, "predict_proba", spy)
    run_dir = run(cfg_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["evaluate_test"] is False
    # Only the calibration split (for its agreement table) and the threshold-selection
    # split are scored; the 1200 test rows never are.
    counts = manifest["split_label_counts"]
    assert scored_batches == [counts["calib"]["rows"], counts["thresh"]["rows"]]
    for name in ("manifest.json", "splits.csv", "thresholds.json", "model.pkl", "report.json"):
        assert (run_dir / name).is_file(), name
    assert (run_dir / "report.md").is_file()
    for name in ("predictions.csv", "simulation_jobs.csv"):
        assert not (run_dir / name).exists(), name
    # The selection stage is the full run's selection stage.
    assert (run_dir / "thresholds.json").read_bytes() == (full_run / "thresholds.json").read_bytes()
    report = json.loads((run_dir / "report.json").read_text())
    assert report["evaluate_test"] is False
    assert report["eval_slice"] == "development splits only; test rows not scored"
    assert report["decision_contract"]["automatic_actions"] == 0
    test_only = {"tiers", "simulation", "per_label", "review_workload", "rules", "ranking"}
    assert not test_only & set(report)
    selection = report["threshold_selection"]
    assert {"rules", "allow_false_negatives", "ranking", "cross_fitted"} <= set(selection)
    full_report = json.loads((full_run / "report.json").read_text())
    assert selection == full_report["threshold_selection"]
    agreement = report["agreement_by_confidence"]
    assert "test" not in agreement
    assert "test" in full_report["agreement_by_confidence"]
    assert (
        agreement["threshold_selection"]
        == full_report["agreement_by_confidence"]["threshold_selection"]
    )
    markdown = (run_dir / "report.md").read_text()
    assert "test rows not scored" in markdown
    assert "## High-risk composition by band" in markdown
    assert "## Queue simulation" not in markdown
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "render_results.py"),
            str(run_dir),
            "--readme",
            str(tmp_path / "README.md"),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode != 0
    assert "Traceback" not in proc.stderr
    assert len(proc.stderr.strip().splitlines()) == 1
    assert "test rows not scored" in proc.stderr


def test_external_scores_refuse_a_development_config(
    external_case: tuple[Path, Corpus, np.ndarray, np.ndarray, dict[str, Any]],
) -> None:
    path, corpus, p_thresh, p_test, source = external_case
    path.write_text(path.read_text() + "evaluate_test: false\n")
    with pytest.raises(ValueError, match="evaluate_test"):
        evaluate_scores(path, corpus, p_thresh, p_test, score_source=source)
    assert not load_config(path).output_dir.exists()


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
    assert text.startswith("intro\n<!-- results:start -->\n## Review-routing results")
    assert text.endswith("<!-- results:end -->\nouttro\n")
    assert "\nold\n" not in text
    assert "SYNTHETIC corpus" in text
    assert "| gate | measured | requirement | status |" in text


def test_per_job_records_reproduce_every_time_metric(
    synthetic_corpus: Path, tmp_path: Path
) -> None:
    from review_router.simulate import JobRecords, time_metrics

    run_dir = run(_config(tmp_path, synthetic_corpus))
    report = json.loads((run_dir / "report.json").read_text())
    rows = pd.read_csv(run_dir / "simulation_jobs.csv", dtype={"comment_id": str})
    expected_cols = {
        "comment_id",
        "arrival_min",
        "start_min",
        "completion_min",
        "strategy",
        "trigger_reason",
        "high_risk",
        "status",
        "wait_min",
        "completion_latency_min",
        "seed",
        "load_per_hour",
    }
    assert expected_cols <= set(rows.columns)
    assert set(rows["status"]) <= {"not_started", "in_progress", "completed"}
    sim = report["simulation"]
    horizon = sim["assumptions"]["horizon_hours"] * 60
    assert "no enforcement outcome" in sim["assumptions"]["review_completion_definition"]
    for key, block in sim["time_metrics"].items():
        strategy, load = key.split("@")
        subset = rows[(rows["strategy"] == strategy) & (rows["load_per_hour"] == float(load))]
        rec = JobRecords.from_rows(subset.to_dict(orient="records"), horizon)
        tm = time_metrics(rec)
        pooled = block["pooled_over_seeds"]
        assert tm["status"] == pooled["status"]
        assert tm["wait"]["p50"] == pytest.approx(pooled["wait"]["p50"], abs=1e-9)
        assert tm["completion_latency"]["p90"] == pytest.approx(
            pooled["completion_latency"]["p90"], abs=1e-9
        )
    head = sim["headline"]
    assert head["metric"] == "wait" and head["percentile"] == 50
    sel = head["selected"]
    if sel["fifo"]:
        assert sel["reduction"] == pytest.approx(1 - sel["router"] / sel["fifo"])
    else:
        assert sel["reduction"] is None and "absolute" in sel["note"]


def test_prevalence_shift_is_hand_computable() -> None:
    from review_router.pipeline import _prevalence_shift

    counts = {
        "train": {"rows": 60, **{label: 6 for label in LABELS}},
        "calib": {"rows": 20, **{label: 2 for label in LABELS}},
        "thresh": {"rows": 20, **{label: 2 for label in LABELS}},
    }
    y = np.zeros((10, len(LABELS)), dtype=int)
    y[:2, 0] = 1  # toxic: 2 of 10 on "test"
    proba = np.full((10, len(LABELS)), 0.3)  # model expects 3 positives per label
    out = _prevalence_shift(counts, y, proba)
    toxic = out["toxic"]
    assert toxic["prevalence_train_file"] == pytest.approx(0.10)
    assert toxic["prevalence_test"] == pytest.approx(0.20)
    assert toxic["prevalence_ratio_train_over_test"] == pytest.approx(0.5)
    assert toxic["model_expected_positives_test"] == pytest.approx(3.0)
    assert toxic["volume_overshoot_model"] == pytest.approx(0.5)  # 3 expected / 2 actual - 1
    assert toxic["volume_overshoot_train_prevalence"] == pytest.approx(
        -0.5
    )  # 1 expected / 2 actual - 1
    assert out["severe_toxic"]["volume_overshoot_model"] is None  # no positives on test


@pytest.fixture
def external_case(
    synthetic_corpus: Path,
    tmp_path: Path,
) -> tuple[Path, Corpus, np.ndarray, np.ndarray, dict[str, Any]]:
    path = _config(tmp_path, synthetic_corpus)
    config = load_config(path)
    corpus = load_corpus(config.data_dir, config.split_fractions, config.seed)
    # Deliberately retain a non-sorted cohort order; evaluation must not silently
    # sort IDs or align external scores against a newly loaded full test set.
    corpus = replace(
        corpus,
        train=corpus.train.sample(frac=1, random_state=1),
        test=corpus.test.sample(n=100, random_state=2),
    )
    thresh = corpus.train[corpus.train["split"] == "thresh"]
    rng = np.random.default_rng(13)
    p_thresh = rng.uniform(0.1, 0.9, (len(thresh), len(LABELS)))
    p_test = rng.uniform(0.1, 0.9, (len(corpus.test), len(LABELS)))
    source = {
        "model_name": "test-external-model-v1",
        "labels": list(LABELS),
        "row_ids": {
            "thresh": thresh["id"].astype(str).tolist(),
            "test": corpus.test["id"].astype(str).tolist(),
        },
        "cohort_manifest_sha256": "recorded-upstream-cohort-hash",
    }
    return path, corpus, p_thresh, p_test, source


def test_external_scores_preserve_identity_and_exploratory_provenance(
    external_case: tuple[Path, Corpus, np.ndarray, np.ndarray, dict[str, Any]],
) -> None:
    path, corpus, p_thresh, p_test, source = external_case
    run_dir = evaluate_scores(path, corpus, p_thresh, p_test, score_source=source)
    predictions = pd.read_csv(run_dir / "predictions.csv", dtype={"id": str})
    assert predictions["id"].tolist() == source["row_ids"]["test"]
    np.testing.assert_allclose(predictions[[f"p_{label}" for label in LABELS]], p_test)
    np.testing.assert_array_equal(
        predictions[[f"y_{label}" for label in LABELS]], corpus.test[list(LABELS)]
    )
    assert not (run_dir / "model.pkl").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    report = json.loads((run_dir / "report.json").read_text())
    assert manifest["score_source"] == report["score_source"] == source
    assert manifest["exploratory"] is report["exploratory"] is True
    assert report["synthetic"] is True  # "exploratory" does not disguise synthetic input
    assert report["eval_slice"].startswith("exploratory ")
    assert manifest["source_data_sha256"] == corpus.file_hashes
    assert set(manifest["data_sha256"]) == {"external_evaluation_cohort"}
    assert len(manifest["data_sha256"]["external_evaluation_cohort"]) == 64
    assert manifest["data_sha256"] != corpus.file_hashes
    assert manifest["config"]["model"] == {
        "kind": "external_scores",
        "model_name": source["model_name"],
    }
    assert manifest["unused_model_config"]["analyzer"] == "word"
    assert manifest["split_label_counts"]["test_scored"]["rows"] == 100
    assert "EXPLORATORY external-score evaluation" in (run_dir / "report.md").read_text()
    assert report["decision_contract"]["automatic_actions"] == 0


@pytest.mark.parametrize("split", ["thresh", "test"])
@pytest.mark.parametrize("invalid", ["shape", "nan", "inf", "negative", "above_one"])
def test_external_scores_reject_invalid_probabilities_before_writing(
    external_case: tuple[Path, Corpus, np.ndarray, np.ndarray, dict[str, Any]],
    split: str,
    invalid: str,
) -> None:
    path, corpus, p_thresh, p_test, source = external_case
    scores = {"thresh": p_thresh, "test": p_test}
    if invalid == "shape":
        scores[split] = scores[split][:-1]
        message = "must have shape"
    else:
        scores[split][0, 0] = {"nan": np.nan, "inf": np.inf, "negative": -0.01, "above_one": 1.01}[
            invalid
        ]
        message = r"finite probabilities in \[0, 1\]"
    with pytest.raises(ValueError, match=message):
        evaluate_scores(path, corpus, scores["thresh"], scores["test"], score_source=source)
    assert not load_config(path).output_dir.exists()


@pytest.mark.parametrize("field", ["thresh", "test", "labels"])
def test_external_scores_reject_mismatched_row_or_label_order(
    external_case: tuple[Path, Corpus, np.ndarray, np.ndarray, dict[str, Any]],
    field: str,
) -> None:
    path, corpus, p_thresh, p_test, source = external_case
    if field == "labels":
        source["labels"].reverse()
        message = "canonical order"
    else:
        source["row_ids"][field].reverse()
        message = "do not match corpus row order"
    with pytest.raises(ValueError, match=message):
        evaluate_scores(path, corpus, p_thresh, p_test, score_source=source)
    assert not load_config(path).output_dir.exists()


def test_external_scores_share_exact_baseline_evaluation(
    synthetic_corpus: Path,
    tmp_path: Path,
) -> None:
    path = _config(tmp_path, synthetic_corpus)
    baseline = run(path)
    config = load_config(path)
    corpus = load_corpus(config.data_dir, config.split_fractions, config.seed)
    with (baseline / "model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    thresh = corpus.train[corpus.train["split"] == "thresh"]
    source = {
        "model_name": "frozen-word-baseline-replay",
        "labels": list(LABELS),
        "row_ids": {"thresh": thresh["id"].tolist(), "test": corpus.test["id"].tolist()},
    }
    replay = evaluate_scores(
        path,
        corpus,
        model.predict_proba(thresh["comment_text"].tolist()),
        model.predict_proba(corpus.test["comment_text"].tolist()),
        score_source=source,
    )
    for name in ("splits.csv", "thresholds.json", "predictions.csv", "simulation_jobs.csv"):
        assert (baseline / name).read_bytes() == (replay / name).read_bytes(), name
    baseline_report = json.loads((baseline / "report.json").read_text())
    replay_report = json.loads((replay / "report.json").read_text())
    baseline_report.pop("run_id")
    replay_report.pop("run_id")
    replay_report.pop("exploratory")
    replay_report.pop("score_source")
    replay_report["eval_slice"] = replay_report["eval_slice"].removeprefix("exploratory ")
    # The replay has no model, so exactly these model-dependent sub-blocks of
    # agreement_by_confidence cannot exist there: the calibration table, the
    # optimism gap (calibration minus selection) and the oov_tercile stratum
    # (vocabulary of the word vectorizer). Everything else, including per_label,
    # pooled_review, priority_band, the identity stratum, segment_check and the
    # alternatives arm, must be identical.
    for report in (baseline_report, replay_report):
        agreement = report["agreement_by_confidence"]
        agreement.pop("calibration")
        agreement.pop("calibration_note")
        agreement.pop("optimism_gap")
        agreement["strata_definition"].pop("oov_tercile")
        for split in ("threshold_selection", "test"):
            agreement[split]["strata"].pop("oov_tercile", None)
    assert replay_report["agreement_by_confidence"]["threshold_selection"]["strata"] == {
        "identity_term_present": baseline_report["agreement_by_confidence"]["threshold_selection"][
            "strata"
        ]["identity_term_present"]
    }
    assert "priority_band" in replay_report["agreement_by_confidence"]["test"]
    assert replay_report == baseline_report


def _priority_only(threshold: float | None) -> TierThresholds:
    return TierThresholds(
        thresholds={
            "priority_review": {label: threshold for label in LABELS},
            "human_review": dict.fromkeys(LABELS, 0.5),
        },
        labels=LABELS,
    )


def test_segment_check_keeps_the_closed_top_segment_at_threshold_one() -> None:
    # A threshold of exactly 1.0 is reachable only under the cumulative rule (every qualifying
    # row scored 1.0). The last segment is closed at 1.0, so it must still be reported instead
    # of leaving segments=[] and a vacuous all_meet_floor=True.
    policy = load_policy(Path("review_router/policy.yaml"))
    y = np.ones((50, len(LABELS)), dtype=int)
    p = np.ones((50, len(LABELS)))
    check = _segment_check(_priority_only(1.0), y, p, policy, 5.0)
    for label in LABELS:
        segments = check[label]["segments"]
        assert len(segments) == 1
        assert segments[0]["lo"] == 1.0 and segments[0]["hi"] == 1.0
        assert segments[0]["n"] == 50 and segments[0]["agreement"] == 1.0
        assert check[label]["all_meet_floor"] is True


def test_selection_notes_name_the_configured_priority_rule() -> None:
    shipped = load_policy(Path("review_router/policy.yaml"))
    assert _selection_notes(_priority_only(None), shipped) == [
        "no label qualifies for priority_review under the segment rule; the priority band "
        "contains rule promotions only"
    ]
    cumulative = replace(shipped, tier_selection_rules={"priority_review": "cumulative_precision"})
    assert _selection_notes(_priority_only(None), cumulative) == [
        "no label qualifies for priority_review under the cumulative_precision rule; the "
        "priority band contains rule promotions only"
    ]
    assert _selection_notes(_priority_only(0.9), shipped) == []
