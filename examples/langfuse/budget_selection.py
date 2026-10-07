"""Select traces for human annotation under a fixed review budget.

Pure functions over a pandas table with one row per candidate trace and one
column per automated signal (an LLM-as-a-judge score, a code check, a user
feedback flag). Every signal is read as a probability-like value in [0, 1];
a missing signal counts as no evidence (0), never as evidence.

Two policies mirror arms of the review-router benchmark: ``max_score`` is
its probability ordering and ``severity`` its default, the maximum over
signals of probability times a declared weight. ``random`` is this module's
sampling baseline, ``uncertainty`` prefers signals near 0.5 for judge
calibration, and ``mixed`` splits the budget between them by quota; these
three were not benchmark arms. Selecting within a budget reallocates fixed
reviewer capacity; it does not add capacity, and nothing here measures
real-world harm.

This file is inlined verbatim into the companion notebook; the test suite
keeps the two copies identical.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from math import isfinite, sqrt

import numpy as np
import pandas as pd

__all__ = [
    "POLICIES",
    "FLAG_THRESHOLD",
    "RESERVED_COLUMNS",
    "SelectionConfig",
    "severity_scores",
    "max_scores",
    "uncertainty_scores",
    "policy_scores",
    "rank",
    "select",
    "select_all",
    "compare_policies",
    "overlap_matrix",
    "synthetic_pool",
    "simulate_annotations",
    "annotation_precision",
    "judge_agreement",
]

POLICIES: tuple[str, ...] = ("random", "max_score", "severity", "uncertainty", "mixed")
# A signal at or above this value counts as "flagged" in the composition tables.
FLAG_THRESHOLD = 0.5
_SCORED = ("max_score", "severity", "uncertainty")
# Columns rank() and select() add; a signal with one of these names would be overwritten.
RESERVED_COLUMNS = frozenset({"policy", "score", "rank", "source", "severity_score"})


@dataclass(frozen=True)
class SelectionConfig:
    """Budget, severity weights and the quota split of the mixed policy."""

    budget: int
    weights: Mapping[str, float]
    seed: int = 0
    mixed_quota: Mapping[str, float] = field(
        default_factory=lambda: {"severity": 0.6, "uncertainty": 0.25, "random": 0.15}
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.budget, bool)
            or not isinstance(self.budget, int | np.integer)
            or self.budget <= 0
        ):
            raise ValueError("budget must be a positive integer")
        if not self.weights:
            raise ValueError("weights must name at least one signal")
        for name, weight in self.weights.items():
            if (
                isinstance(weight, bool)
                or not isinstance(weight, int | float | np.integer | np.floating)
                or not isfinite(weight)
                or weight <= 0
            ):
                raise ValueError(f"weight for {name!r} must be positive and finite")
            if name in RESERVED_COLUMNS:
                raise ValueError(f"signal name {name!r} is reserved; rename the score")
        unknown = set(self.mixed_quota) - {*_SCORED, "random"}
        if unknown:
            raise ValueError(f"mixed_quota names unknown policies: {sorted(unknown)}")
        for share in self.mixed_quota.values():
            if (
                isinstance(share, bool)
                or not isinstance(share, int | float | np.integer | np.floating)
                or not isfinite(share)
                or share < 0
            ):
                raise ValueError("mixed_quota shares must be finite and nonnegative")
        if abs(sum(self.mixed_quota.values()) - 1.0) > 1e-9:
            raise ValueError("mixed_quota shares must sum to 1")

    @property
    def signals(self) -> list[str]:
        return list(self.weights)


def _signal_matrix(pool: pd.DataFrame, weights: Mapping[str, float]) -> np.ndarray:
    missing = [name for name in weights if name not in pool.columns]
    if missing:
        raise ValueError(f"pool lacks signal columns: {missing}")
    values = pool[list(weights)].to_numpy(dtype=float)
    if np.nanmin(values, initial=0.0) < 0 or np.nanmax(values, initial=0.0) > 1:
        raise ValueError("signals must lie in [0, 1]")
    return np.asarray(np.nan_to_num(values, nan=0.0), dtype=float)


def severity_scores(pool: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    """Maximum over signals of probability times weight; the review-router default."""
    p = _signal_matrix(pool, weights)
    w = np.array([weights[name] for name in weights], dtype=float)
    return pd.Series((p * w).max(axis=1) if len(p) else [], index=pool.index, dtype=float)


def max_scores(pool: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    """Maximum probability over signals, ignoring the weights."""
    p = _signal_matrix(pool, weights)
    return pd.Series(p.max(axis=1) if len(p) else [], index=pool.index, dtype=float)


def uncertainty_scores(pool: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    """1 at a signal of 0.5, 0 at 0 or 1; highest where some judge is least sure."""
    p = _signal_matrix(pool, weights)
    closeness = 1.0 - np.abs(2.0 * p - 1.0)
    return pd.Series(closeness.max(axis=1) if len(p) else [], index=pool.index, dtype=float)


def policy_scores(pool: pd.DataFrame, policy: str, config: SelectionConfig) -> pd.Series:
    """The ranking score of one policy. random draws a seeded permutation."""
    if policy == "severity":
        return severity_scores(pool, config.weights)
    if policy == "max_score":
        return max_scores(pool, config.weights)
    if policy == "uncertainty":
        return uncertainty_scores(pool, config.weights)
    if policy == "random":
        rng = np.random.default_rng([config.seed, len(pool)])
        return pd.Series(rng.random(len(pool)), index=pool.index, dtype=float)
    raise ValueError(f"unknown policy {policy!r}; mixed has no single score")


def _check_pool(pool: pd.DataFrame) -> None:
    if "trace_id" not in pool.columns:
        raise ValueError("pool needs a trace_id column")
    if pool["trace_id"].isna().any() or (pool["trace_id"].astype(str) == "").any():
        raise ValueError("every candidate needs a non-empty trace_id")
    if pool["trace_id"].duplicated().any():
        raise ValueError("trace_id values must be unique; aggregate signals per trace first")
    clash = sorted(RESERVED_COLUMNS.intersection(pool.columns))
    if clash:
        raise ValueError(f"pool columns {clash} are reserved for the selection output; rename them")


def rank(pool: pd.DataFrame, policy: str, config: SelectionConfig) -> pd.DataFrame:
    """The whole pool ordered by one policy: score descending, ties by trace_id.

    ``score`` is the policy's own ranking score; ``severity_score`` is added to
    every row so selections from different policies share one comparable number.
    """
    _check_pool(pool)
    if policy == "mixed":
        raise ValueError("mixed selects by quota; use select()")
    scored = pool.assign(
        policy=policy,
        score=policy_scores(pool, policy, config),
        severity_score=severity_scores(pool, config.weights),
    )
    ordered = scored.sort_values(["score", "trace_id"], ascending=[False, True], kind="stable")
    return ordered.assign(rank=np.arange(1, len(ordered) + 1)).reset_index(drop=True)


def _quota_counts(budget: int, quota: Mapping[str, float]) -> dict[str, int]:
    """Largest-remainder split of the budget; the counts always sum to the budget."""
    names = list(quota)
    exact = np.array([budget * quota[name] for name in names], dtype=float)
    counts = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - counts), kind="stable")[: budget - int(counts.sum())]:
        counts[i] += 1
    return dict(zip(names, (int(c) for c in counts), strict=True))


def select(pool: pd.DataFrame, policy: str, config: SelectionConfig) -> pd.DataFrame:
    """The rows one policy sends to review, at most the budget, in service order.

    Columns: trace_id, policy, score, severity_score, rank, source (the
    sub-policy that took the row; equals policy except under mixed), plus the
    pool's own columns. Under mixed, ``score`` is the taking sub-policy's own
    score and is only comparable within one ``source``; ``severity_score`` is
    comparable across rows.
    """
    _check_pool(pool)
    if policy != "mixed":
        chosen = rank(pool, policy, config).head(config.budget)
        return chosen.assign(source=policy)
    taken: list[pd.DataFrame] = []
    seen: set[str] = set()
    for name, count in _quota_counts(config.budget, config.mixed_quota).items():
        ordered = rank(pool, name, config)
        fresh = ordered[~ordered["trace_id"].isin(seen)].head(count)
        seen.update(fresh["trace_id"])
        taken.append(fresh.assign(source=name))
    # Quotas can be starved by overlap; refill from severity so the budget is spent.
    shortfall = config.budget - sum(len(part) for part in taken)
    if shortfall > 0:
        ordered = rank(pool, "severity", config)
        fill = ordered[~ordered["trace_id"].isin(seen)].head(shortfall)
        taken.append(fill.assign(source="severity"))
    chosen = (
        pd.concat(taken, ignore_index=True) if taken else rank(pool, "severity", config).head(0)
    )
    return chosen.assign(policy="mixed", rank=np.arange(1, len(chosen) + 1))


def select_all(
    pool: pd.DataFrame, config: SelectionConfig, policies: Iterable[str] = POLICIES
) -> dict[str, pd.DataFrame]:
    return {policy: select(pool, policy, config) for policy in _names(policies)}


def _truth_label(value: object) -> float:
    if isinstance(value, str):
        return {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}.get(value.strip().lower(), 2.0)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 2.0


def _truth_mask(column: pd.Series, name: str) -> np.ndarray:
    """0/1, False/True or their strings as a boolean mask; NaN counts as not positive."""
    labels = (
        column.astype(object).where(column.notna(), 0.0).map(_truth_label).to_numpy(dtype=float)
    )
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError(f"truth column {name!r} must hold 0/1 or False/True values")
    return np.asarray(labels == 1.0)


def compare_policies(
    pool: pd.DataFrame,
    config: SelectionConfig,
    policies: Iterable[str] = POLICIES,
    truth: str | None = None,
) -> pd.DataFrame:
    """What each policy's selection contains, and what it leaves out.

    Per policy: rows selected, rows flagged (any signal >= 0.5) among them,
    per-signal flagged counts, mean severity of the selection, and
    ``flagged_left_out``: flagged rows that did not fit in the budget. With a
    boolean ``truth`` column (an oracle, only available on synthetic or
    already-labelled data; 0/1 or False/True, NaN meaning unlabelled and
    counted as not positive) it adds the true positives captured and the
    share of all true positives in the pool that the budget reaches.
    """
    _check_pool(pool)
    positives = np.zeros(len(pool), dtype=bool)
    if truth is not None:
        positives = _truth_mask(pool[truth], truth)
    flags = _signal_matrix(pool, config.weights) >= FLAG_THRESHOLD
    any_flag = pd.Series(flags.any(axis=1), index=pool.index)
    severity = severity_scores(pool, config.weights)
    rows = []
    for policy in _names(policies):
        chosen = select(pool, policy, config)
        mask = pool["trace_id"].isin(chosen["trace_id"]).to_numpy()
        row: dict[str, object] = {
            "policy": policy,
            "selected": int(mask.sum()),
            "flagged_selected": int(any_flag[mask].sum()),
            "flagged_left_out": int(any_flag[~mask].sum()),
            "mean_severity": float(severity[mask].mean()) if mask.any() else 0.0,
        }
        for j, name in enumerate(config.signals):
            row[f"{name}_flagged"] = int(flags[mask, j].sum())
        if truth is not None:
            captured = int((positives & mask).sum())
            row["true_positive_selected"] = captured
            row["true_positive_share"] = (
                captured / int(positives.sum()) if positives.any() else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def overlap_matrix(selections: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Count of trace_ids two policies both selected."""
    ids = {name: set(frame["trace_id"]) for name, frame in selections.items()}
    names = list(ids)
    matrix = pd.DataFrame(
        [[len(ids[a] & ids[b]) for b in names] for a in names], index=names, columns=names
    )
    return matrix.rename_axis(index="policy")


def synthetic_pool(
    n: int,
    weights: Mapping[str, float],
    seed: int = 0,
    prevalence: Mapping[str, float] | None = None,
    missing_rate: float = 0.05,
) -> pd.DataFrame:
    """SYNTHETIC candidates with hidden truth, for running the notebook offline.

    Each signal has a latent true label drawn at its prevalence (default 0.08)
    and a judge score drawn from Beta(5, 2) when the label is 1 and Beta(1, 6)
    when it is 0, so the judge is informative but imperfect. A share of
    signals is missing (NaN). ``true_<signal>`` columns hold the latent labels
    and ``true_high_risk`` is 1 when the maximum true-label weight reaches the
    median declared weight. Numbers from this pool verify the code; they are
    not results.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= missing_rate < 1:
        raise ValueError("missing_rate must lie in [0, 1)")
    rng = np.random.default_rng(seed)
    prevalence = dict(prevalence or {})
    frame = pd.DataFrame({"trace_id": [f"trace-{seed}-{i:05d}" for i in range(n)]})
    truth_weight = np.zeros(n)
    for name, weight in weights.items():
        y = rng.random(n) < prevalence.get(name, 0.08)
        p = np.where(y, rng.beta(5.0, 2.0, size=n), rng.beta(1.0, 6.0, size=n))
        p[rng.random(n) < missing_rate] = np.nan
        frame[name] = p
        frame[f"true_{name}"] = y.astype(int)
        truth_weight = np.maximum(truth_weight, y * float(weight))
    frame["true_high_risk"] = (truth_weight >= float(np.median(list(weights.values())))).astype(int)
    return frame


def simulate_annotations(
    selected: pd.DataFrame,
    pool: pd.DataFrame,
    signals: Iterable[str],
    seed: int = 0,
    flip: float = 0.1,
) -> pd.DataFrame:
    """SYNTHETIC annotator labels for selected rows: hidden truth with a flip rate.

    Returns long-form rows (trace_id, name, value) in the shape the Langfuse
    scores endpoint returns for ANNOTATION-source scores, so the closing-the-
    loop cells run identically offline and live.
    """
    rng = np.random.default_rng(seed)
    truth = pool.set_index("trace_id")
    names = _names(signals)
    rows = []
    for trace_id in selected["trace_id"].drop_duplicates():
        for name in names:
            label = int(truth.at[trace_id, f"true_{name}"])
            if rng.random() < flip:
                label = 1 - label
            rows.append({"trace_id": trace_id, "name": f"human_{name}", "value": float(label)})
    return pd.DataFrame(rows, columns=["trace_id", "name", "value"])


def _names(values: Iterable[str]) -> list[str]:
    """Materialise a name list; a bare string is a mistake, not a one-item list."""
    if isinstance(values, str):
        raise TypeError("pass an iterable of names, not a single string")
    return list(values)


def _wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def annotation_precision(
    selections: Mapping[str, pd.DataFrame], annotations: pd.DataFrame, positive_names: Iterable[str]
) -> pd.DataFrame:
    """Share of each policy's annotated rows that a human marked positive on any listed score.

    ``annotations`` is long-form (trace_id, name, value) with value 1 for a
    positive label. Rows a human has not labelled yet are excluded from the
    denominator, so early in a review cycle the estimate covers only the
    items reviewers reached. Intervals are 95% Wilson.
    """
    names = set(_names(positive_names))
    labelled = annotations[annotations["name"].isin(names)]
    positive_ids = set(labelled.loc[labelled["value"] >= 1, "trace_id"])
    labelled_ids = set(labelled["trace_id"])
    rows = []
    for policy, chosen in selections.items():
        ids = [t for t in chosen["trace_id"] if t in labelled_ids]
        hits = sum(t in positive_ids for t in ids)
        low, high = _wilson(hits, len(ids))
        rows.append(
            {
                "policy": policy,
                "annotated": len(ids),
                "positive": hits,
                "precision": hits / len(ids) if ids else float("nan"),
                "ci_low": low,
                "ci_high": high,
            }
        )
    columns = ["policy", "annotated", "positive", "precision", "ci_low", "ci_high"]
    return pd.DataFrame(rows, columns=columns)


def judge_agreement(
    pool: pd.DataFrame, annotations: pd.DataFrame, signals: Iterable[str]
) -> pd.DataFrame:
    """Per signal, how the judge flag (>= 0.5) agrees with the human label on annotated rows.

    Rows were chosen by the policies, not at random, so these rates describe
    the reviewed slice only; they are not the judge's accuracy on all traffic.
    """
    scores = pool.set_index("trace_id")
    rows = []
    for name in _names(signals):
        human = annotations[annotations["name"] == f"human_{name}"]
        human = human[human["trace_id"].isin(scores.index)]
        if human.empty:
            rows.append({"signal": name, "annotated": 0})
            continue
        judge = scores.loc[human["trace_id"], name].to_numpy(dtype=float)
        judge_flag = np.nan_to_num(judge, nan=0.0) >= FLAG_THRESHOLD
        label = human["value"].to_numpy(dtype=float) >= 1
        both = int((judge_flag & label).sum())
        rows.append(
            {
                "signal": name,
                "annotated": int(len(label)),
                "agreement": float((judge_flag == label).mean()),
                "judge_precision": both / int(judge_flag.sum())
                if judge_flag.any()
                else float("nan"),
                "judge_recall": both / int(label.sum()) if label.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)
