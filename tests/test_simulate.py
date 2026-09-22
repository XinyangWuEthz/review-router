from __future__ import annotations

import numpy as np
import pytest

from review_router.simulate import (
    P99_MIN_SAMPLES,
    JobRecords,
    Scenario,
    SimConfig,
    draw_scenario,
    simulate,
    time_metrics,
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
    out, rec = simulate(sc, np.zeros(3), harm, hr, CFG, "fifo")
    assert out.n_handled == 3
    assert rec.start_min.tolist() == [0.0, 10.0, 20.0]
    assert rec.completion_latency_min.tolist() == [10.0, 20.0, 30.0]
    assert out.completion_latency_p50 == 20.0
    assert out.wait_p50 == 10.0 and out.wait_p90 == pytest.approx(18.0)
    assert out.backlog_end == 0
    assert out.reviewer_utilization == pytest.approx(0.5)


def test_priority_pulls_high_risk_forward_under_overload() -> None:
    # Ten jobs at t=0, only six fit in the hour. Job 9 is the high-risk one.
    sc = _scenario([0.0] * 10, list(range(10)))
    harm = np.zeros(10)
    harm[9] = 10.0
    hr = harm >= 5
    fifo, fifo_rec = simulate(sc, np.zeros(10), harm, hr, CFG, "fifo")
    sev, _ = simulate(sc, harm.copy(), harm, hr, CFG, "severity")
    assert fifo_rec.status.tolist() == ["completed"] * 6 + ["not_started"] * 4
    assert fifo.n_not_started == 4 and fifo.n_in_progress == 0
    assert fifo.high_risk_not_started == 1 and fifo.unfinished_age_p50 == 60.0
    assert fifo.n_handled == sev.n_handled == 6
    assert fifo.high_risk_handled == 0 and fifo.high_risk_unhandled == 1
    assert sev.high_risk_handled == 1 and sev.high_risk_wait_p50 == 0.0
    assert sev.harm_per_reviewer_hour == 10.0 and fifo.harm_per_reviewer_hour == 0.0
    assert fifo.backlog_end == sev.backlog_end == 4


def test_idle_reviewers_do_not_start_a_later_batch_before_arrival() -> None:
    cfg = SimConfig(reviewers=2, handle_minutes=10.0, horizon_hours=1.0)
    sc = _scenario([10.0] * 5, list(range(5)))
    out, _ = simulate(sc, np.zeros(5), np.ones(5), np.ones(5, dtype=bool), cfg, "fifo")
    assert out.n_handled == 5
    assert out.wait_p50 == 10.0
    assert out.wait_p90 == pytest.approx(16.0)
    assert out.reviewer_utilization == pytest.approx(50.0 / 120.0)


def test_late_batch_is_in_progress_at_horizon_for_all_idle_reviewers() -> None:
    cfg = SimConfig(reviewers=2, handle_minutes=10.0, horizon_hours=1.0)
    sc = _scenario([59.0, 59.0], [0, 1])
    out, rec = simulate(sc, np.zeros(2), np.ones(2), np.ones(2, dtype=bool), cfg, "fifo")
    assert out.n_handled == out.high_risk_handled == 0
    assert out.high_risk_unhandled == 2 and out.high_risk_in_progress == 2
    assert rec.status.tolist() == ["in_progress", "in_progress"]
    assert out.wait_p50 == 0.0  # started at once; the wait exists even though they did not finish
    assert out.completion_latency_p50 is None and out.n_completion_samples == 0
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


def test_equal_priority_ties_break_by_arrival_then_index() -> None:
    # Two reviewers idle; four jobs, all priority 1. Arrivals 0, 0, 5, 5 with
    # handle time 10: the two t=0 jobs start first (index order), the t=5
    # pair waits for the first completions at t=10.
    cfg = SimConfig(reviewers=2, handle_minutes=10.0, horizon_hours=1.0)
    sc = _scenario([0.0, 0.0, 5.0, 5.0], [3, 2, 1, 0])
    out, rec = simulate(sc, np.ones(4), np.zeros(4), np.zeros(4, dtype=bool), cfg, "severity")
    assert rec.start_min.tolist() == [0.0, 0.0, 10.0, 10.0]
    assert rec.wait_min.tolist() == [0.0, 0.0, 5.0, 5.0]
    assert out.wait_p50 == 2.5


def test_arrival_just_before_horizon_with_busy_reviewers_is_not_started() -> None:
    # One reviewer busy from 50 to 60 with job 0; job 1 arrives at 59.5 and
    # never starts because the horizon ends at 60.
    sc = _scenario([50.0, 59.5], [0, 1])
    out, rec = simulate(
        sc, np.zeros(2), np.array([0.0, 10.0]), np.array([False, True]), CFG, "fifo"
    )
    assert rec.status.tolist() == ["completed", "not_started"]
    assert out.high_risk_not_started == 1 and out.high_risk_unhandled == 1
    assert np.isnan(rec.wait_min[1]) and out.n_wait_samples == 1
    assert out.unfinished_age_p50 == pytest.approx(0.5)


def test_p99_is_flagged_unreliable_below_the_minimum_sample_size() -> None:
    sc = _scenario([0.0, 0.0, 0.0], [0, 1, 2])
    _, rec = simulate(sc, np.zeros(3), np.zeros(3), np.zeros(3, dtype=bool), CFG, "fifo")
    tm = time_metrics(rec)
    assert tm["wait"]["n"] == 3 < P99_MIN_SAMPLES
    assert tm["wait"]["p99_reliable"] is False
    assert tm["wait"]["p99"] == pytest.approx(19.8)  # the largest of 0, 10, 20 minus 1%


def test_result_fields_are_recomputable_from_the_records() -> None:
    cfg = SimConfig(reviewers=2, handle_minutes=3.0, horizon_hours=1.0)
    sc = draw_scenario(3, 70.0, 50, cfg)
    harm = np.arange(50, dtype=float) % 7
    hr = harm >= 5
    out, rec = simulate(sc, harm, harm, hr, cfg, "severity")
    tm = time_metrics(rec)
    assert tm["status"]["completed"] == out.n_handled
    assert tm["status"]["not_started"] == out.backlog_end == out.n_not_started
    assert tm["status"]["in_progress"] == out.n_in_progress
    assert tm["wait"]["p90"] == out.wait_p90 and tm["wait"]["n"] == out.n_wait_samples
    assert tm["completion_latency"]["p50"] == out.completion_latency_p50
    assert tm["high_risk_status"]["completed"] == out.high_risk_handled
    assert sum(tm["status"].values()) == out.n_arrivals == len(rec)
    # every wait equals start - arrival, computed independently from the rows
    rows = rec.rows(strategy="severity")
    for row in rows:
        if row["start_min"] is not None:
            assert row["wait_min"] == pytest.approx(row["start_min"] - row["arrival_min"])
        if row["status"] == "completed":
            assert row["completion_latency_min"] == pytest.approx(
                row["completion_min"] - row["arrival_min"]
            )
        else:
            assert row["completion_latency_min"] is None


def test_records_dataclass_exposes_masks() -> None:
    rec = JobRecords(
        job_index=np.array([0, 1]),
        arrival_min=np.array([0.0, 1.0]),
        start_min=np.array([0.0, np.nan]),
        completion_min=np.array([5.0, np.nan]),
        status=np.array(["completed", "not_started"], dtype=object),
        high_risk=np.array([True, False]),
        harm=np.array([5.0, 0.0]),
        priority=np.zeros(2),
        horizon_min=60.0,
    )
    assert rec.started.tolist() == [True, False] and rec.completed.tolist() == [True, False]
    tm = time_metrics(rec)
    assert tm["completion_ratio"] == 0.5 and tm["unfinished_age_at_horizon"]["p50"] == 59.0
