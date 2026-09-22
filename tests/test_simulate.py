from __future__ import annotations

import numpy as np
import pytest

from review_router.simulate import (
    STRATEGIES,
    Scenario,
    SimConfig,
    draw_scenario,
    priority_review_scores,
    simulate,
)

CFG = SimConfig(reviewers=1, handle_minutes=10.0, horizon_hours=1.0)


def _scenario(arrivals: list[float], jobs: list[int]) -> Scenario:
    return Scenario(
        seed=0,
        load_per_hour=0.0,
        job_index=np.array(jobs),
        arrival_min=np.array(arrivals, dtype=float),
        service_min=np.full(len(arrivals), CFG.handle_minutes),
    )


def test_capacity_is_reviewers_times_items_per_hour() -> None:
    assert SimConfig(reviewers=4, handle_minutes=2.0).capacity_per_hour == 120


@pytest.mark.parametrize("reviewers", [0, -1])
def test_config_rejects_nonpositive_reviewer_count(reviewers: int) -> None:
    with pytest.raises(ValueError, match="reviewers must be a positive integer"):
        SimConfig(reviewers=reviewers)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_config_rejects_invalid_service_time_and_horizon(value: float) -> None:
    with pytest.raises(ValueError, match="handle_minutes must be positive and finite"):
        SimConfig(handle_minutes=value)
    with pytest.raises(ValueError, match="horizon_hours must be positive and finite"):
        SimConfig(horizon_hours=value)


def test_draw_scenario_is_seeded_and_sorted() -> None:
    cfg = SimConfig(reviewers=4, handle_minutes=2.0, horizon_hours=8.0)
    a = draw_scenario(1, 108.0, 500, cfg)
    b = draw_scenario(1, 108.0, 500, cfg)
    c = draw_scenario(2, 108.0, 500, cfg)
    assert (a.arrival_min == b.arrival_min).all() and (a.job_index == b.job_index).all()
    assert not np.array_equal(a.arrival_min, c.arrival_min)
    assert (np.diff(a.arrival_min) >= 0).all()
    assert abs(len(a.arrival_min) - 108 * 8) < 200


def test_fifo_processes_in_arrival_order_and_counts_wait() -> None:
    # Three jobs arrive at once; one reviewer, 10 min each, 60 min horizon.
    sc = _scenario([0.0, 0.0, 0.0], [0, 1, 2])
    harm = np.array([0.0, 10.0, 0.0])
    hr = harm >= 5
    out = simulate(sc, np.zeros(3), harm, hr, CFG, "fifo")
    assert out.n_handled == 3
    assert out.wait_p50 == 10.0 and out.wait_p90 == pytest.approx(18.0)
    assert out.backlog_end == 0
    assert out.reviewer_utilization == pytest.approx(0.5)


def test_priority_pulls_high_risk_forward_under_overload() -> None:
    # Ten jobs at t=0, only six fit in the hour. Job 9 is the high-risk one.
    sc = _scenario([0.0] * 10, list(range(10)))
    harm = np.zeros(10)
    harm[9] = 10.0
    hr = harm >= 5
    fifo = simulate(sc, np.zeros(10), harm, hr, CFG, "fifo")
    sev = simulate(sc, harm.copy(), harm, hr, CFG, "severity")
    assert fifo.n_handled == sev.n_handled == 6
    assert fifo.high_risk_handled == 0 and fifo.high_risk_unhandled == 1
    assert sev.high_risk_handled == 1 and sev.high_risk_wait_p50 == 0.0
    assert sev.harm_per_reviewer_hour == 10.0 and fifo.harm_per_reviewer_hour == 0.0
    assert fifo.backlog_end == sev.backlog_end == 4


def test_priority_strategy_sorts_tier_before_severity_and_still_requires_service() -> None:
    assert "priority" in STRATEGIES
    scores = priority_review_scores(np.array([False, True, True]), np.array([10.0, 0.1, 0.3]))
    assert np.argsort(-scores).tolist() == [2, 1, 0]
    config = SimConfig(reviewers=1, handle_minutes=1.0, horizon_hours=2 / 60)
    scenario = Scenario(0, 0.0, np.arange(3), np.zeros(3), np.ones(3))
    out = simulate(
        scenario,
        scores,
        np.array([10.0, 1.0, 2.0]),
        np.array([True, False, False]),
        config,
        "priority",
    )
    assert out.n_handled == 2
    assert out.harm_handled == 3.0
    assert out.high_risk_handled == 0
    assert out.backlog_end == 1
    assert out.reviewer_utilization == 1.0


def test_idle_reviewers_do_not_start_a_later_batch_before_arrival() -> None:
    cfg = SimConfig(reviewers=2, handle_minutes=10.0, horizon_hours=1.0)
    sc = _scenario([10.0] * 5, list(range(5)))
    out = simulate(sc, np.zeros(5), np.ones(5), np.ones(5, dtype=bool), cfg, "fifo")
    assert out.n_handled == 5
    assert out.wait_p50 == 10.0
    assert out.wait_p90 == pytest.approx(16.0)
    assert out.reviewer_utilization == pytest.approx(50.0 / 120.0)


def test_late_batch_is_in_progress_at_horizon_for_all_idle_reviewers() -> None:
    cfg = SimConfig(reviewers=2, handle_minutes=10.0, horizon_hours=1.0)
    sc = _scenario([59.0, 59.0], [0, 1])
    out = simulate(sc, np.zeros(2), np.ones(2), np.ones(2, dtype=bool), cfg, "fifo")
    assert out.n_handled == out.high_risk_handled == 0
    assert out.high_risk_unhandled == 2
    assert out.wait_p50 is None
    assert out.backlog_end == 0
    assert out.reviewer_utilization == pytest.approx(2.0 / 120.0)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_simulation_rejects_invalid_scenario_service_times(value: float) -> None:
    sc = Scenario(0, 0.0, np.array([0]), np.array([0.0]), np.array([value]))
    with pytest.raises(ValueError, match="scenario service times must be positive and finite"):
        simulate(sc, np.zeros(1), np.ones(1), np.ones(1, dtype=bool), CFG, "fifo")


@pytest.mark.parametrize(
    "arrivals", [[-1.0], [60.0], [70.0], [float("inf")], [float("nan")], [10.0, 0.0]]
)
def test_simulation_rejects_invalid_arrivals(arrivals: list[float]) -> None:
    n = len(arrivals)
    sc = _scenario(arrivals, list(range(n)))
    with pytest.raises(ValueError, match="scenario arrivals must be sorted, finite"):
        simulate(sc, np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool), CFG, "fifo")
