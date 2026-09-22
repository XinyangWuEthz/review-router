"""Review-queue simulation: fixed reviewers, fixed handle time, replayed arrivals.

Every strategy sees the identical arrival list and handle times for a given
(seed, load); only the order in which waiting jobs are picked differs. All
quantities here are simulation outputs under stated assumptions, not
measurements of a production queue.

Every job leaves a record (arrival, start, completion, status), and every time
metric the report prints is computed from those records by `time_metrics`, so
any number can be recomputed from the saved per-job file.

Assumption stated once: a review that completes is treated as the action being
completed. There is no separate enforcement step in the simulation.
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
    "JobRecords",
    "SimResult",
    "STATUSES",
    "P99_MIN_SAMPLES",
    "draw_scenario",
    "simulate",
    "time_metrics",
    "STRATEGIES",
]

STRATEGIES: tuple[str, ...] = ("fifo", "prob", "severity")
STATUSES: tuple[str, ...] = ("not_started", "in_progress", "completed")
# Below this many samples a p99 is reported but flagged as unreliable: the
# 99th percentile of fewer than 100 values is at most the largest one or two.
P99_MIN_SAMPLES = 100


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
class JobRecords:
    """One row per arrival. Times are minutes from the start of the horizon.

    start_min is NaN for jobs never started; completion_min is start plus the
    handle time whenever the job started, even past the horizon, so a job in
    progress at the horizon keeps its would-be completion time and the status
    column says it did not finish.
    """

    job_index: np.ndarray
    arrival_min: np.ndarray
    start_min: np.ndarray
    completion_min: np.ndarray
    status: np.ndarray
    high_risk: np.ndarray
    harm: np.ndarray
    priority: np.ndarray
    horizon_min: float

    def __len__(self) -> int:
        return int(len(self.arrival_min))

    @property
    def started(self) -> np.ndarray:
        return np.asarray(~np.isnan(self.start_min), dtype=bool)

    @property
    def completed(self) -> np.ndarray:
        return np.asarray(self.status == "completed", dtype=bool)

    @property
    def wait_min(self) -> np.ndarray:
        """Start minus arrival; NaN for jobs that never started."""
        return np.asarray(self.start_min - self.arrival_min, dtype=float)

    @property
    def completion_latency_min(self) -> np.ndarray:
        """Completion minus arrival; NaN unless the job completed within the horizon."""
        out = self.completion_min - self.arrival_min
        return np.asarray(np.where(self.completed, out, np.nan), dtype=float)

    @staticmethod
    def from_rows(rows: list[dict[str, Any]], horizon_min: float) -> JobRecords:
        """Rebuild records from saved per-job rows (the inverse of rows())."""

        def col(name: str, dtype: Any) -> np.ndarray:
            values = [r[name] for r in rows]
            if dtype is float:
                return np.array(
                    [
                        np.nan
                        if v is None or v == "" or (isinstance(v, float) and np.isnan(v))
                        else float(v)
                        for v in values
                    ],
                    dtype=float,
                )
            return np.array(values, dtype=dtype)

        return JobRecords(
            job_index=col("job_index", int),
            arrival_min=col("arrival_min", float),
            start_min=col("start_min", float),
            completion_min=col("completion_min", float),
            status=np.array([str(r["status"]) for r in rows], dtype=object),
            high_risk=np.array([bool(r["high_risk"]) for r in rows]),
            harm=col("harm", float),
            priority=col("priority_score", float),
            horizon_min=horizon_min,
        )

    @staticmethod
    def concat(parts: list[JobRecords]) -> JobRecords:
        """Pool several scenarios' records (same horizon) into one table."""
        if not parts:
            raise ValueError("nothing to concatenate")
        horizon = parts[0].horizon_min
        if any(p.horizon_min != horizon for p in parts):
            raise ValueError("records with different horizons cannot be pooled")
        cat = np.concatenate
        return JobRecords(
            job_index=cat([p.job_index for p in parts]),
            arrival_min=cat([p.arrival_min for p in parts]),
            start_min=cat([p.start_min for p in parts]),
            completion_min=cat([p.completion_min for p in parts]),
            status=cat([p.status for p in parts]),
            high_risk=cat([p.high_risk for p in parts]),
            harm=cat([p.harm for p in parts]),
            priority=cat([p.priority for p in parts]),
            horizon_min=horizon,
        )

    def rows(self, **constant: Any) -> list[dict[str, Any]]:
        """Per-job dicts with the given constant columns (strategy, seed, ids...) merged in."""
        wait = self.wait_min
        latency = self.completion_latency_min
        rows = []
        for i in range(len(self)):
            row = {
                **{
                    k: (v[i] if isinstance(v, np.ndarray | list) else v)
                    for k, v in constant.items()
                },
                "job_index": int(self.job_index[i]),
                "arrival_min": float(self.arrival_min[i]),
                "start_min": None if np.isnan(self.start_min[i]) else float(self.start_min[i]),
                "completion_min": None
                if np.isnan(self.completion_min[i])
                else float(self.completion_min[i]),
                "status": str(self.status[i]),
                "high_risk": bool(self.high_risk[i]),
                "harm": float(self.harm[i]),
                "priority_score": float(self.priority[i]),
                "wait_min": None if np.isnan(wait[i]) else float(wait[i]),
                "completion_latency_min": None if np.isnan(latency[i]) else float(latency[i]),
            }
            rows.append(row)
        return rows


def _pct(values: np.ndarray, q: float) -> float | None:
    return float(np.percentile(values, q)) if len(values) else None


def _quantiles(values: np.ndarray) -> dict[str, Any]:
    n = int(len(values))
    return {
        "n": n,
        "p50": _pct(values, 50),
        "p90": _pct(values, 90),
        "p99": _pct(values, 99),
        "p99_reliable": n >= P99_MIN_SAMPLES,
        "mean": float(values.mean()) if n else None,
        "max": float(values.max()) if n else None,
    }


def time_metrics(records: JobRecords) -> dict[str, Any]:
    """Every time and completion metric, computed from the per-job records only.

    Wait = start - arrival over jobs that started (in progress or completed).
    Completion latency = completion - arrival over jobs completed within the
    horizon. Jobs that never started have no wait; they are counted, never
    given a wait of zero, and their age at the horizon is reported instead.
    """
    hr = records.high_risk.astype(bool)
    started = records.started
    completed = records.completed
    in_progress = records.status == "in_progress"
    not_started = records.status == "not_started"
    wait = records.wait_min
    latency = records.completion_latency_min
    age_unfinished = records.horizon_min - records.arrival_min
    return {
        "n_arrivals": int(len(records)),
        "status": {
            "not_started": int(not_started.sum()),
            "in_progress": int(in_progress.sum()),
            "completed": int(completed.sum()),
        },
        "high_risk_status": {
            "not_started": int((not_started & hr).sum()),
            "in_progress": int((in_progress & hr).sum()),
            "completed": int((completed & hr).sum()),
        },
        "completion_ratio": float(completed.sum() / len(records)) if len(records) else None,
        "wait": _quantiles(wait[started]),
        "high_risk_wait": _quantiles(wait[started & hr]),
        "completion_latency": _quantiles(latency[completed]),
        "high_risk_completion_latency": _quantiles(latency[completed & hr]),
        "unfinished_age_at_horizon": _quantiles(age_unfinished[~completed]),
        "high_risk_unfinished_age_at_horizon": _quantiles(age_unfinished[~completed & hr]),
    }


@dataclass(frozen=True)
class SimResult:
    strategy: str
    n_arrivals: int
    n_handled: int  # completed within the horizon
    n_not_started: int
    n_in_progress: int
    high_risk_arrived: int
    high_risk_handled: int  # completed
    high_risk_unhandled: int  # not started + in progress
    high_risk_not_started: int
    high_risk_in_progress: int
    harm_arrived: float
    harm_handled: float
    harm_per_reviewer_hour: float
    completion_ratio: float
    wait_p50: float | None  # over started jobs
    wait_p90: float | None
    wait_p99: float | None
    n_wait_samples: int
    high_risk_wait_p50: float | None
    high_risk_wait_p90: float | None
    high_risk_wait_p99: float | None
    n_high_risk_wait_samples: int
    completion_latency_p50: float | None  # over completed jobs
    completion_latency_p90: float | None
    completion_latency_p99: float | None
    n_completion_samples: int
    unfinished_age_p50: float | None  # minutes waited so far by unfinished jobs, at the horizon
    backlog_end: int  # jobs never started
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
) -> tuple[SimResult, JobRecords]:
    """Run one strategy over one scenario; return the summary and the per-job records.

    `priority`, `harm`, `high_risk` are indexed by queued-job index (not by
    arrival). Higher priority is picked first; ties fall back to arrival order,
    then to arrival index, so an all-zero priority vector is FIFO.
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
    completion = np.where(started, start + scenario.service_min, np.nan)
    completed = started & (completion <= horizon)
    status = np.full(n, "not_started", dtype=object)
    status[started & ~completed] = "in_progress"
    status[completed] = "completed"
    job_harm = harm[scenario.job_index].astype(float)
    job_hr = high_risk[scenario.job_index].astype(bool)
    records = JobRecords(
        job_index=scenario.job_index.copy(),
        arrival_min=scenario.arrival_min.copy(),
        start_min=start,
        completion_min=completion,
        status=status,
        high_risk=job_hr,
        harm=job_harm,
        priority=priority[scenario.job_index].astype(float),
        horizon_min=horizon,
    )
    tm = time_metrics(records)

    # Queue depth sampled every minute: arrived minus started.
    grid = np.arange(0.0, horizon, 1.0)
    arrived = np.searchsorted(scenario.arrival_min, grid, side="right")
    started_times = np.sort(start[started])
    begun = np.searchsorted(started_times, grid, side="right")
    depth = arrived - begun

    busy = np.clip(np.minimum(completion[started], horizon) - start[started], 0.0, None).sum()
    reviewer_minutes = config.reviewers * horizon
    harm_handled = float(job_harm[completed].sum())

    result = SimResult(
        strategy=strategy,
        n_arrivals=n,
        n_handled=tm["status"]["completed"],
        n_not_started=tm["status"]["not_started"],
        n_in_progress=tm["status"]["in_progress"],
        high_risk_arrived=int(job_hr.sum()),
        high_risk_handled=tm["high_risk_status"]["completed"],
        high_risk_unhandled=tm["high_risk_status"]["not_started"]
        + tm["high_risk_status"]["in_progress"],
        high_risk_not_started=tm["high_risk_status"]["not_started"],
        high_risk_in_progress=tm["high_risk_status"]["in_progress"],
        harm_arrived=float(job_harm.sum()),
        harm_handled=harm_handled,
        harm_per_reviewer_hour=harm_handled / (reviewer_minutes / 60.0),
        completion_ratio=float(tm["completion_ratio"] or 0.0),
        wait_p50=tm["wait"]["p50"],
        wait_p90=tm["wait"]["p90"],
        wait_p99=tm["wait"]["p99"],
        n_wait_samples=tm["wait"]["n"],
        high_risk_wait_p50=tm["high_risk_wait"]["p50"],
        high_risk_wait_p90=tm["high_risk_wait"]["p90"],
        high_risk_wait_p99=tm["high_risk_wait"]["p99"],
        n_high_risk_wait_samples=tm["high_risk_wait"]["n"],
        completion_latency_p50=tm["completion_latency"]["p50"],
        completion_latency_p90=tm["completion_latency"]["p90"],
        completion_latency_p99=tm["completion_latency"]["p99"],
        n_completion_samples=tm["completion_latency"]["n"],
        unfinished_age_p50=tm["unfinished_age_at_horizon"]["p50"],
        backlog_end=tm["status"]["not_started"],
        queue_depth_p95=float(np.percentile(depth, 95)) if len(depth) else 0.0,
        reviewer_utilization=float(busy / reviewer_minutes) if reviewer_minutes else 0.0,
    )
    return result, records
