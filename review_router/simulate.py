"""Review-queue simulation: fixed reviewers, fixed handle time, replayed arrivals.

Every strategy sees the identical arrival list and handle times for a given
(seed, load); only the order in which waiting jobs are picked differs. All
quantities here are simulation outputs under stated assumptions, not
measurements of a production queue.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

__all__ = [
    "SimConfig",
    "Scenario",
    "SimResult",
    "draw_scenario",
    "simulate",
    "STRATEGIES",
    "priority_review_scores",
]

STRATEGIES: tuple[str, ...] = ("fifo", "prob", "severity", "priority")


def priority_review_scores(is_priority: np.ndarray, severity: np.ndarray) -> np.ndarray:
    """Sort priority-review jobs first, then predicted severity within each tier."""
    if is_priority.shape != severity.shape or is_priority.ndim != 1:
        raise ValueError("priority flags and severity scores must be matching vectors")
    if not np.isfinite(severity).all() or (severity < 0).any():
        raise ValueError("severity scores must be nonnegative and finite")
    band_width = float(severity.max(initial=0.0)) + 1.0
    return np.asarray(is_priority.astype(bool).astype(float) * band_width + severity)


@dataclass(frozen=True)
class SimConfig:
    reviewers: int = 4
    handle_minutes: float = 2.0
    horizon_hours: float = 8.0

    def __post_init__(self) -> None:
        if isinstance(self.reviewers, bool) or not isinstance(self.reviewers, int):
            raise ValueError("reviewers must be a positive integer")
        if self.reviewers <= 0:
            raise ValueError("reviewers must be a positive integer")
        if not isfinite(self.handle_minutes) or self.handle_minutes <= 0:
            raise ValueError("handle_minutes must be positive and finite")
        if not isfinite(self.horizon_hours) or self.horizon_hours <= 0:
            raise ValueError("horizon_hours must be positive and finite")

    @property
    def horizon_minutes(self) -> float:
        return self.horizon_hours * 60.0

    @property
    def capacity_per_hour(self) -> float:
        return self.reviewers * 60.0 / self.handle_minutes


@dataclass(frozen=True)
class Scenario:
    """One replayable arrival stream: which queued job arrives when, and its handle time."""

    seed: int
    load_per_hour: float
    job_index: np.ndarray  # index into the queued job table
    arrival_min: np.ndarray  # sorted, minutes from start
    service_min: np.ndarray


def draw_scenario(seed: int, load_per_hour: float, n_jobs: int, config: SimConfig) -> Scenario:
    """Poisson arrivals over the horizon; jobs sampled with replacement from the queue set."""
    if n_jobs <= 0:
        raise ValueError("cannot draw a scenario from an empty queue set")
    rng = np.random.default_rng([seed, int(round(load_per_hour * 1000))])
    n = int(rng.poisson(load_per_hour * config.horizon_hours))
    arrival = np.sort(rng.uniform(0.0, config.horizon_minutes, size=n))
    job_index = rng.integers(0, n_jobs, size=n)
    service = np.full(n, config.handle_minutes, dtype=float)
    return Scenario(seed, load_per_hour, job_index, arrival, service)


@dataclass(frozen=True)
class SimResult:
    strategy: str
    n_arrivals: int
    n_handled: int
    high_risk_arrived: int
    high_risk_handled: int
    high_risk_unhandled: int
    harm_arrived: float
    harm_handled: float
    harm_per_reviewer_hour: float
    wait_p50: float | None
    wait_p90: float | None
    high_risk_wait_p50: float | None
    high_risk_wait_p90: float | None
    backlog_end: int
    queue_depth_p95: float
    reviewer_utilization: float

    def as_json(self) -> dict[str, Any]:
        return dict(self.__dict__)


def simulate(
    scenario: Scenario,
    priority: np.ndarray,
    harm: np.ndarray,
    high_risk: np.ndarray,
    config: SimConfig,
    strategy: str,
) -> SimResult:
    """Run one strategy over one scenario.

    `priority`, `harm`, `high_risk` are indexed by queued-job index (not by
    arrival). Higher priority is picked first; ties fall back to arrival order,
    so an all-zero priority vector is FIFO.
    """
    n = len(scenario.arrival_min)
    if not np.isfinite(scenario.service_min).all() or (scenario.service_min <= 0).any():
        raise ValueError("scenario service times must be positive and finite")
    horizon = config.horizon_minutes
    if (
        not np.isfinite(scenario.arrival_min).all()
        or (scenario.arrival_min < 0).any()
        or (scenario.arrival_min >= horizon).any()
        or (np.diff(scenario.arrival_min) < 0).any()
    ):
        raise ValueError("scenario arrivals must be sorted, finite and within [0, horizon)")
    free_at: list[float] = [0.0] * config.reviewers
    heapq.heapify(free_at)
    waiting: list[tuple[float, float, int]] = []  # (-priority, arrival, arrival_idx)
    start = np.full(n, np.nan)
    i = 0
    clock = 0.0
    while True:
        # Idle reviewers may still have availability times before the current
        # event. Advancing to an arrival must advance every subsequent dispatch.
        t = max(clock, free_at[0])
        if not waiting:
            if i >= n:
                break
            t = max(t, float(scenario.arrival_min[i]))
        if t >= horizon:
            break
        while i < n and scenario.arrival_min[i] <= t:
            job = int(scenario.job_index[i])
            heapq.heappush(waiting, (-float(priority[job]), float(scenario.arrival_min[i]), i))
            i += 1
        if not waiting:
            continue
        _, _, k = heapq.heappop(waiting)
        heapq.heapreplace(free_at, t + float(scenario.service_min[k]))
        start[k] = t
        clock = t

    started = ~np.isnan(start)
    completion = start + scenario.service_min
    handled = started & (completion <= horizon)
    job_harm = harm[scenario.job_index]
    job_hr = high_risk[scenario.job_index].astype(bool)
    waits = start[handled] - scenario.arrival_min[handled]
    hr_waits = waits[job_hr[handled]]

    # Queue depth sampled every minute: arrived minus started.
    grid = np.arange(0.0, horizon, 1.0)
    arrived = np.searchsorted(scenario.arrival_min, grid, side="right")
    started_times = np.sort(start[started])
    begun = np.searchsorted(started_times, grid, side="right")
    depth = arrived - begun

    busy = np.clip(np.minimum(completion[started], horizon) - start[started], 0.0, None).sum()
    reviewer_minutes = config.reviewers * horizon
    harm_handled = float(job_harm[handled].sum())

    def pct(values: np.ndarray, q: float) -> float | None:
        return float(np.percentile(values, q)) if len(values) else None

    return SimResult(
        strategy=strategy,
        n_arrivals=n,
        n_handled=int(handled.sum()),
        high_risk_arrived=int(job_hr.sum()),
        high_risk_handled=int(job_hr[handled].sum()),
        high_risk_unhandled=int((job_hr & ~handled).sum()),
        harm_arrived=float(job_harm.sum()),
        harm_handled=harm_handled,
        harm_per_reviewer_hour=harm_handled / (reviewer_minutes / 60.0),
        wait_p50=pct(waits, 50),
        wait_p90=pct(waits, 90),
        high_risk_wait_p50=pct(hr_waits, 50),
        high_risk_wait_p90=pct(hr_waits, 90),
        backlog_end=int((~started).sum()),
        queue_depth_p95=float(np.percentile(depth, 95)) if len(depth) else 0.0,
        reviewer_utilization=float(busy / reviewer_minutes) if reviewer_minutes else 0.0,
    )
