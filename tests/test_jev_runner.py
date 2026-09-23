"""Offline integration checks for the experiment runner; never model-quality evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from review_router.data import LABELS
from review_router.synthetic import CUES
from scripts import run_jev_experiment as runner

REPOSITORY = Path(__file__).resolve().parents[1]


class OfflineJev:
    """Deterministic synthetic cue scores, standing in for populated response caches."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], bool]] = []

    def predict(self, texts: list[str], *, allow_network: bool = False) -> np.ndarray:
        self.calls.append((list(texts), allow_network))
        assert not allow_network, "the default runner invocation must stay offline"
        return np.asarray(
            [
                [0.85 if any(cue in text for cue in CUES[label]) else 0.03 for label in LABELS]
                for text in texts
            ]
        )

    def summary(self) -> dict[str, Any]:
        return {
            "network_calls": 0,
            "network_successes": 0,
            "cache_hits": sum(len(texts) for texts, _ in self.calls),
            "input_tokens": 0,
            "output_tokens": 0,
            "billing_usd": None,
        }


def _configure(
    tmp_path: Path, synthetic_corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> OfflineJev:
    baseline_path = tmp_path / "baseline.yaml"
    baseline_path.write_text(
        yaml.safe_dump(
            {
                "label": "offline-runner-check",
                "data_dir": str(synthetic_corpus),
                "output_dir": str(tmp_path / "reports"),
                "policy": str(REPOSITORY / "review_router/policy.yaml"),
                "seed": 3,
                "split": {"train": 0.6, "calib": 0.2, "thresh": 0.2},
                "model": {"max_features": 5000, "ngram_max": 1, "min_df": 1, "C": 2.0},
                "simulation": {
                    "reviewers": 2,
                    "handle_minutes": 2.0,
                    "horizon_hours": 0.1,
                    "loads_per_hour": [30, 90],
                    "seeds": [1, 2],
                    "high_risk_min_weight": 5,
                    "primary": {"load_per_hour": 90, "strategy": "priority"},
                    "thesis_load_per_hour": 90,
                },
            }
        )
    )
    pilot_path = tmp_path / "pilot.yaml"
    pilot_path.write_text(
        yaml.safe_dump(
            {
                "baseline_config": str(baseline_path),
                "questions": str(REPOSITORY / "configs/jev_questions.json"),
                "model": "jev-1.13.0",
                "sample_seed": 20260923,
                "sample_sizes": {"calib": 300, "thresh": 300, "test": 300},
                "cache_dir": "cache",
                "analysis_dir": "analysis/jev",
                "bootstrap_replicates": 10,
                "workers": 1,
                "requests_per_second": 1,
                "input_usd_per_million_tokens": 0.042,
            }
        )
    )
    client = OfflineJev()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr("review_router.jev.JevClient", lambda **kwargs: client)
    monkeypatch.setattr(sys, "argv", ["run_jev_experiment.py", "--config", str(pilot_path)])
    return client


def _saved_result(tmp_path: Path) -> dict[str, Any]:
    result = json.loads((tmp_path / "record/jev_run.json").read_text())
    assert result == json.loads((tmp_path / "analysis/jev/experiment.json").read_text())
    return dict(result)


def test_offline_runner_evaluates_the_same_frozen_cohort(
    tmp_path: Path, synthetic_corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _configure(tmp_path, synthetic_corpus, monkeypatch)
    runner.main()
    result = _saved_result(tmp_path)
    assert result["status"] == "completed_exploratory"
    assert "failure" not in result
    assert [(len(texts), network) for texts, network in client.calls] == [(300, False)] * 3
    assert result["api"]["network_calls"] == 0
    assert result["comparison"]["equal_input"]["n_seeds"] == 200
    assert result["paired_capture"]["replicates"] == 10

    cohort = pd.read_csv(tmp_path / "analysis/jev/cohort.csv", dtype={"id": str})
    expected_test_ids = cohort.loc[cohort["split"] == "test", "id"].tolist()
    predictions = []
    cohort_hashes = []
    for name in ("baseline", "jev"):
        directory = Path(result[name]["run_dir"])
        prediction = pd.read_csv(directory / "predictions.csv", dtype={"id": str})
        manifest = json.loads((directory / "manifest.json").read_text())
        report = json.loads((directory / "report.json").read_text())
        assert prediction.id.tolist() == expected_test_ids
        assert report["synthetic"] is True
        assert report["exploratory"] is True
        assert manifest["score_source"]["protocol_sha256"] == result["protocol_sha256"]
        assert manifest["score_source"]["cohort_id_sha256"] == result["protocol"]["id_sha256"]
        assert manifest["score_source"]["row_ids"]["test"] == expected_test_ids
        predictions.append(prediction)
        cohort_hashes.append(manifest["data_sha256"])
    assert cohort_hashes[0] == cohort_hashes[1]
    pd.testing.assert_frame_equal(
        predictions[0][["id", *[f"y_{label}" for label in LABELS]]],
        predictions[1][["id", *[f"y_{label}" for label in LABELS]]],
    )
    assert (tmp_path / "analysis/jev/run_comparison.json").is_file()
    assert (tmp_path / "analysis/jev/run_comparison.md").is_file()


@pytest.mark.parametrize("exception", [RuntimeError, SystemExit])
def test_comparison_failure_preserves_completed_jev_evaluation(
    tmp_path: Path, synthetic_corpus: Path, monkeypatch: pytest.MonkeyPatch,
    exception: type[BaseException],
) -> None:
    client = _configure(tmp_path, synthetic_corpus, monkeypatch)

    def fail_comparison(*args: Any, **kwargs: Any) -> Any:
        raise exception("injected comparison failure")

    monkeypatch.setattr(runner, "shared_stream", fail_comparison)
    with pytest.raises(SystemExit, match="incomplete at paired_comparison"):
        runner.main()
    result = _saved_result(tmp_path)
    assert result["status"] == "incomplete"
    assert result["failure"] == {
        "stage": "paired_comparison",
        "type": exception.__name__,
        "message": "injected comparison failure",
    }
    assert result["baseline"] is not None
    assert result["jev"] is not None
    assert set(result["jev"]["brier"]) == set(LABELS)
    assert set(result["jev"]["raw_brier"]) == set(LABELS)
    assert result["jev"]["calibration"]["method"] == "Platt on logit(p)"
    assert (Path(result["jev"]["run_dir"]) / "calibration.json").is_file()
    assert (Path(result["jev"]["run_dir"]) / "report.json").is_file()
    assert result["api"] == client.summary()
    assert result["api"]["network_calls"] == 0
    assert "comparison" not in result
    assert not (tmp_path / "analysis/jev/run_comparison.json").exists()
