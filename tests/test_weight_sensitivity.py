from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from review_router.data import LABELS
from review_router.pipeline import queue_inputs
from review_router.policy import load_policy
from review_router.simulate import JobRecords, Scenario, SimConfig, SimResult, simulate

ROOT = Path(__file__).resolve().parent.parent


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "weight_sensitivity", ROOT / "scripts" / "weight_sensitivity.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SENS = _module()


@pytest.fixture
def plan() -> dict[str, Any]:
    result: dict[str, Any] = yaml.safe_load(
        (ROOT / "configs" / "severity-sensitivity.yaml").read_text()
    )
    return result


@pytest.fixture
def predictions() -> pd.DataFrame:
    # Threat outranks identity hate under default weights, but not threat_lower.
    # The allowed threat row must never enter the queue, despite its high score.
    pred = pd.DataFrame({"id": ["threat", "identity", "toxic", "allowed-threat"]})
    for label in LABELS:
        pred[f"p_{label}"] = 0.0
        pred[f"y_{label}"] = 0
    for row, label, probability in (
        (0, "threat", 0.5),
        (1, "identity_hate", 0.6),
        (2, "toxic", 0.9),
        (3, "threat", 1.0),
    ):
        pred.loc[row, f"p_{label}"] = probability
        pred.loc[row, f"y_{label}"] = 1
    pred["final_tier"] = ["human_review", "human_review", "priority_review", "allow"]
    return pred


def test_predeclared_vectors_match_baseline_and_flat_probability(
    plan: dict[str, Any],
    predictions: pd.DataFrame,
) -> None:
    assert set(plan["ordering_weights"]) == {
        "default",
        "flat",
        "compressed",
        "expanded",
        "threat_lower",
    }
    assert plan["seeds"] == list(range(1, 21))
    assert plan["loads_per_hour"] == [108, 180]
    proba = predictions[[f"p_{label}" for label in LABELS]].to_numpy()
    truth = predictions[[f"y_{label}" for label in LABELS]].to_numpy()
    tiers = predictions["final_tier"].to_numpy()
    baseline = queue_inputs(proba, truth, tiers, load_policy(), 5)
    admitted = baseline["queued"]
    scores = SENS.priorities(proba[admitted], tiers[admitted], plan)
    assert "flat" not in scores  # counted only once, as probability ordering
    for new, original in (
        ("default", "severity"),
        ("prob", "prob"),
        ("priority", "priority"),
        ("fifo", "fifo"),
    ):
        np.testing.assert_array_equal(scores[new], baseline["priorities"][original])
    np.testing.assert_allclose(scores["default"], [5.0, 3.6, 0.9], rtol=0, atol=1e-14)
    assert scores["threat_lower"].argmax() == 1
    np.testing.assert_array_equal(scores["prob"], [0.5, 0.6, 0.9])


def test_replay_changes_dispatch_but_keeps_admission_truth_and_arrivals(
    plan: dict[str, Any],
    predictions: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = deepcopy(plan)
    plan.update(seeds=[1, 2], loads_per_hour=[180])
    plan["simulation"] = {"reviewers": 1, "handle_minutes": 1.0, "horizon_hours": 1 / 60}
    drawn: list[Scenario] = []
    calls: list[tuple[Scenario, np.ndarray, np.ndarray, np.ndarray]] = []

    def draw(seed: int, load: float, n_jobs: int, config: SimConfig) -> Scenario:
        assert n_jobs == 3
        jobs = [0, 1, 2] if seed == 1 else [0, 0, 2]
        scenario = Scenario(seed, load, np.array(jobs), np.zeros(3), np.ones(3))
        drawn.append(scenario)
        return scenario

    def capture(
        scenario: Scenario,
        score: np.ndarray,
        harm: np.ndarray,
        risk: np.ndarray,
        config: SimConfig,
        arm: str,
    ) -> tuple[SimResult, JobRecords]:
        calls.append((scenario, score.copy(), harm.copy(), risk.copy()))
        return simulate(scenario, score, harm, risk, config, arm)

    monkeypatch.setattr(SENS, "draw_scenario", draw)
    monkeypatch.setattr(SENS, "simulate", capture)
    original = predictions.copy(deep=True)
    rows = SENS.replay(predictions, plan)
    pd.testing.assert_frame_equal(predictions, original)
    assert len(drawn) == 2 and len(rows) == len(calls) == 14
    for seed, scenario in enumerate(drawn, start=1):
        paired = [call for call in calls if call[0].seed == seed]
        assert len(paired) == 7 and all(call[0] is scenario for call in paired)
        for _, _, harm, risk in paired:
            np.testing.assert_array_equal(harm, [10.0, 6.0, 1.0])
            np.testing.assert_array_equal(risk, [True, True, False])
    for row in rows:
        assert row["completed"] == 1
        assert row["high_risk_arrived"] == 2 and row["other_arrived"] == 1
        assert row["high_risk_completed"] + row["other_completed"] == 1
    by_case = {(row["seed"], row["arm"]): row for row in rows}
    assert by_case[1, "default"]["threat_completed"] == 1
    assert by_case[1, "threat_lower"]["threat_completed"] == 0
    assert by_case[1, "threat_lower"]["identity_hate_completed"] == 1
    assert by_case[1, "default"]["baseline_harm_per_reviewer_hour"] == 600.0
    assert by_case[1, "threat_lower"]["baseline_harm_per_reviewer_hour"] == 360.0
    # A completed threat retains utility 10, even when its scheduling weight is 5.
    assert by_case[2, "threat_lower"]["baseline_harm_per_reviewer_hour"] == 600.0
    assert by_case[2, "threat_lower"]["threat_completed"] == 1


def test_replay_is_deterministic_and_pairs_every_arm_with_each_seed(
    plan: dict[str, Any],
    predictions: pd.DataFrame,
) -> None:
    plan = deepcopy(plan)
    plan.update(seeds=[1, 2], loads_per_hour=[40, 100])
    plan["simulation"] = {"reviewers": 1, "handle_minutes": 1.0, "horizon_hours": 0.2}
    first = SENS.replay(predictions, plan)
    assert first == SENS.replay(predictions, plan)
    assert len(first) == 2 * 2 * 7
    for load in plan["loads_per_hour"]:
        for seed in plan["seeds"]:
            paired = [r for r in first if r["seed"] == seed and r["load_per_hour"] == load]
            for metric in ("completed", "high_risk_arrived", "other_arrived"):
                assert len({r[metric] for r in paired}) == 1


def test_group_wait_excludes_never_started_but_unfinished_age_includes_in_progress() -> None:
    records = JobRecords(
        job_index=np.arange(4),
        arrival_min=np.array([0.0, 1.0, 2.0, 8.0]),
        start_min=np.array([0.0, 9.0, np.nan, np.nan]),
        completion_min=np.array([2.0, 11.0, np.nan, np.nan]),
        status=np.array(["completed", "in_progress", "not_started", "not_started"]),
        high_risk=np.array([True, True, True, False]),
        harm=np.array([10.0, 6.0, 5.0, 0.0]),
        priority=np.zeros(4),
        horizon_min=10.0,
    )
    high = SENS.group_metrics(records, records.high_risk, "high_risk")
    assert high == {
        "high_risk_arrived": 3,
        "high_risk_completed": 1,
        "high_risk_not_started": 1,
        "high_risk_unfinished": 2,
        "high_risk_wait_p90": pytest.approx(7.2),
        "high_risk_unfinished_age_p90": pytest.approx(8.9),
    }
    other = SENS.group_metrics(records, ~records.high_risk, "other")
    assert other["other_wait_p90"] is None
    assert other["other_unfinished_age_p90"] == 2.0
    assert other["other_not_started"] == other["other_unfinished"] == 1
    empty = SENS.group_metrics(records, np.zeros(4, dtype=bool), "empty")
    assert empty["empty_wait_p90"] is empty["empty_unfinished_age_p90"] is None


def test_summary_pairs_twenty_seeds_by_identity_and_keeps_missing_waits_missing() -> None:
    rows = []
    for arm in ("default", "fifo", "threat_lower"):
        # Reverse the candidate order to detect accidental positional pairing.
        seeds = range(20, 0, -1) if arm == "threat_lower" else range(1, 21)
        for seed in seeds:
            delta = 1 if seed <= 10 else (-1 if seed <= 15 else 0)
            completed = 100 + seed + ({"default": 0, "fifo": -5, "threat_lower": delta}[arm])
            wait = (
                None
                if (arm == "default" and seed == 1) or (arm == "threat_lower" and seed == 2)
                else float(seed - (2 if arm == "threat_lower" else 0))
            )
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "load_per_hour": 180,
                    "high_risk_completed": completed,
                    "high_risk_wait_p90": wait,
                }
            )
    summary, paired = SENS.summarize(rows)
    completion = paired["threat_lower_vs_default@180"]["high_risk_completed"]
    assert completion["n"] == 20 and completion["mean"] == 0.25
    assert (completion["better"], completion["tied"], completion["worse"]) == (10, 5, 5)
    assert (completion["min"], completion["max"]) == (-1, 1)
    wait = paired["threat_lower_vs_default@180"]["high_risk_wait_p90"]
    assert wait["n"] == wait["better"] == 18
    assert wait["mean"] == -2.0 and wait["tied"] == wait["worse"] == 0
    assert summary["default@180"]["high_risk_wait_p90"]["n"] == 19
    assert summary["threat_lower@180"]["high_risk_wait_p90"]["n"] == 19


@pytest.mark.parametrize("value", [-0.01, 1.01, np.nan, np.inf, -np.inf])
def test_replay_rejects_invalid_probabilities(
    predictions: pd.DataFrame,
    plan: dict[str, Any],
    value: float,
) -> None:
    predictions.loc[0, "p_toxic"] = value
    with pytest.raises(ValueError, match="finite probabilities"):
        SENS.replay(predictions, plan)


@pytest.mark.parametrize("value", [-1.0, 2.0, 0.5])
def test_replay_rejects_nonbinary_labels(
    predictions: pd.DataFrame,
    plan: dict[str, Any],
    value: float,
) -> None:
    predictions["y_toxic"] = predictions["y_toxic"].astype(float)
    predictions.loc[0, "y_toxic"] = value
    with pytest.raises(ValueError, match="labels must be binary"):
        SENS.replay(predictions, plan)


@pytest.mark.parametrize("tier", ["auto_action", "unknown", "", None])
def test_replay_rejects_unknown_bands(
    predictions: pd.DataFrame,
    plan: dict[str, Any],
    tier: str | None,
) -> None:
    predictions.loc[0, "final_tier"] = tier
    with pytest.raises(ValueError, match="unknown saved review band"):
        SENS.replay(predictions, plan)


def test_replay_rejects_empty_or_duplicate_inputs(
    predictions: pd.DataFrame,
    plan: dict[str, Any],
) -> None:
    for pred in (predictions.iloc[:0], pd.concat([predictions, predictions.iloc[:1]])):
        with pytest.raises(ValueError, match="distinct, nonempty row IDs"):
            SENS.replay(pred, plan)
    predictions["final_tier"] = "allow"
    with pytest.raises(ValueError, match="empty saved review pool"):
        SENS.replay(predictions, plan)


@pytest.mark.parametrize("row_id", [None, "", "   "])
def test_replay_rejects_missing_or_blank_ids(
    predictions: pd.DataFrame,
    plan: dict[str, Any],
    row_id: str | None,
) -> None:
    predictions.loc[0, "id"] = row_id
    with pytest.raises(ValueError, match="distinct, nonempty row IDs"):
        SENS.replay(predictions, plan)
