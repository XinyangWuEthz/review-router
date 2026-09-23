"""One-command experiment: data -> model -> thresholds -> routing -> queue simulation -> report.

    python scripts/run_pipeline.py --config configs/baseline.yaml

Every run writes a self-describing directory under the configured output dir:
manifest.json (data hashes, split counts, commit, dependency versions, seeds),
splits.csv, thresholds.json, model.pkl, predictions.csv, report.json (read by
tests/test_gate.py) and report.md (the tables).
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from review_router import __version__
from review_router.data import LABELS, Corpus, label_counts, load_corpus
from review_router.metrics import per_label_metrics, tier_metrics, wilson_interval
from review_router.model import ModelConfig, TfidfLogitModel
from review_router.policy import DEFAULT_POLICY_PATH, Policy, load_policy
from review_router.signals import IDENTITY_PATTERN, IDENTITY_TERMS, identity_term_present
from review_router.simulate import (
    P99_MIN_SAMPLES,
    STRATEGIES,
    JobRecords,
    SimConfig,
    SimResult,
    draw_scenario,
    priority_review_scores,
    simulate,
    time_metrics,
)
from review_router.thresholds import (
    TIER_ORDER,
    TierThresholds,
    apply_rules,
    model_tier,
    select_thresholds,
)

__all__ = ["RunConfig", "evaluate_scores", "load_config", "run"]

HUMAN = "human_review"
PRIORITY = "priority_review"
ALLOW = "allow"


@dataclass(frozen=True)
class RunConfig:
    data_dir: Path
    output_dir: Path
    seed: int
    split_fractions: dict[str, float]
    model: ModelConfig
    policy_path: Path
    sim: SimConfig
    loads_per_hour: tuple[float, ...]
    sim_seeds: tuple[int, ...]
    high_risk_min_weight: float
    primary_load_per_hour: float
    primary_strategy: str
    thesis_load_per_hour: float
    label: str
    headline_metric: str = "wait"
    headline_percentile: int = 50


def load_config(path: Path) -> RunConfig:
    path = path.resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config must be a mapping")
    base = path.parent.parent if path.parent.name == "configs" else Path.cwd()

    def resolve(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (base / p)

    sim_raw = raw.get("simulation", {})
    primary = sim_raw.get("primary", {})
    strategy = str(primary.get("strategy", "priority"))
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown primary strategy {strategy!r}; choose from {STRATEGIES}")
    loads = tuple(float(x) for x in sim_raw.get("loads_per_hour", (60, 108, 180)))
    seeds = tuple(int(x) for x in sim_raw.get("seeds", (1, 2, 3, 4, 5)))
    primary_load = float(primary.get("load_per_hour", 108))
    if not loads or any(not np.isfinite(x) or x <= 0 for x in loads):
        raise ValueError("loads_per_hour must contain positive finite arrival rates")
    if primary_load not in loads:
        raise ValueError("primary load_per_hour must appear in loads_per_hour")
    thesis_load = float(sim_raw.get("thesis_load_per_hour", max(loads)))
    if thesis_load not in loads:
        raise ValueError("thesis_load_per_hour must appear in loads_per_hour")
    if not seeds or any(x < 0 for x in seeds):
        raise ValueError("simulation seeds must contain nonnegative integers")
    headline = sim_raw.get("headline", {})
    headline_metric = str(headline.get("metric", "wait"))
    if headline_metric not in ("wait", "completion_latency"):
        raise ValueError("headline metric must be 'wait' or 'completion_latency'")
    headline_percentile = int(headline.get("percentile", 50))
    if headline_percentile not in (50, 90, 99):
        raise ValueError("headline percentile must be 50, 90 or 99")
    return RunConfig(
        data_dir=resolve(str(raw["data_dir"])),
        output_dir=resolve(str(raw.get("output_dir", "reports"))),
        seed=int(raw.get("seed", 0)),
        split_fractions={str(k): float(v) for k, v in raw["split"].items()},
        model=ModelConfig(**{k: v for k, v in (raw.get("model") or {}).items()}),
        policy_path=resolve(str(raw["policy"])) if raw.get("policy") else DEFAULT_POLICY_PATH,
        sim=SimConfig(
            reviewers=int(sim_raw.get("reviewers", 4)),
            handle_minutes=float(sim_raw.get("handle_minutes", 2.0)),
            horizon_hours=float(sim_raw.get("horizon_hours", 8.0)),
        ),
        loads_per_hour=loads,
        sim_seeds=seeds,
        high_risk_min_weight=float(sim_raw.get("high_risk_min_weight", 5.0)),
        primary_load_per_hour=primary_load,
        primary_strategy=strategy,
        thesis_load_per_hour=thesis_load,
        label=str(raw.get("label", path.stem)),
        headline_metric=headline_metric,
        headline_percentile=headline_percentile,
    )


# --------------------------------------------------------------------------- provenance


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
            cwd=Path(__file__).resolve().parent.parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _dependency_versions() -> dict[str, str]:
    import pandas
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "review_router": __version__,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "pyyaml": yaml.__version__,
    }


def _manifest(config: RunConfig, config_path: Path, corpus: Corpus, run_id: str) -> dict[str, Any]:
    splits = {
        name: label_counts(corpus.train[corpus.train["split"] == name])
        for name in ("train", "calib", "thresh")
    }
    splits["test_scored"] = label_counts(corpus.test)
    cfg = asdict(config)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()}
    return {
        "run_id": run_id,
        "label": config.label,
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "policy_path": str(config.policy_path),
        "policy_sha256": hashlib.sha256(config.policy_path.read_bytes()).hexdigest(),
        "data_sha256": corpus.file_hashes,
        "seed": config.seed,
        "simulation_seeds": list(config.sim_seeds),
        "split_label_counts": splits,
        "dependencies": _dependency_versions(),
        "config": cfg,
    }


# --------------------------------------------------------------------------- routing


def _route(
    proba: np.ndarray,
    y: np.ndarray,
    identity: np.ndarray,
    thresholds: TierThresholds,
    policy: Policy,
) -> dict[str, Any]:
    signals = {"identity_term_present": identity}
    tiers, triggers = model_tier(proba, thresholds, signals)
    population_tiers, _ = model_tier(
        proba, TierThresholds(thresholds.thresholds, thresholds.labels), None
    )
    rule_action, rule_ids = apply_rules(proba, LABELS, identity, policy)
    final = tiers.copy()
    forced = rule_action != ""
    final[forced] = rule_action[forced]

    any_true = y.any(axis=1)
    # Both flagged tiers request human review. A positive label makes the
    # comment relevant to review; no model decision is an enforcement outcome.
    correct = any_true.copy()
    correct_model = any_true.copy()
    correct_population = any_true.copy()

    return {
        "correct_model": correct_model,
        "correct_population": correct_population,
        "population_tier": population_tiers,
        "model_tier": tiers,
        "final_tier": final,
        "triggers": triggers,
        "rule_action": rule_action,
        "rule_ids": rule_ids,
        "correct": correct,
        "any_true": any_true,
    }


def _subgroup_stats(routing: dict[str, Any], y: np.ndarray, policy: Policy) -> dict[str, Any]:
    """What the declared subgroup thresholds changed.

    Rows the population threshold alone would have placed in a tier that the
    subgroup threshold did not.
    """
    population: np.ndarray = routing["population_tier"]
    model: np.ndarray = routing["model_tier"]
    any_true = y.any(axis=1)
    out: dict[str, Any] = {}
    for tier in TIER_ORDER:
        moved = (population == tier) & (model != tier)
        out[tier] = {
            "n_population_tier": int((population == tier).sum()),
            "n_removed_by_subgroup_threshold": int(moved.sum()),
            "n_removed_truly_positive_any_label": int((moved & any_true).sum()),
        }
    out["declared"] = [{"tier": d.tier, "signal": d.signal} for d in policy.subgroup_thresholds]
    return out


def _rule_stats(routing: dict[str, Any], policy: Policy) -> dict[str, Any]:
    ids: list[str] = routing["rule_ids"]
    model: np.ndarray = routing["model_tier"]
    final: np.ndarray = routing["final_tier"]
    any_true: np.ndarray = routing["any_true"]
    forced = routing["rule_action"] != ""
    per_rule = {r.id: int(sum(1 for s in ids if r.id in s.split(";"))) for r in policy.rules}
    moved_to_queue = forced & (model == ALLOW) & np.isin(final, (HUMAN, PRIORITY))
    promoted = forced & (model != PRIORITY) & (final == PRIORITY)
    return {
        "matched_per_rule": per_rule,
        "n_rows_with_any_rule": int(forced.sum()),
        "n_added_to_queue_from_allow": int(moved_to_queue.sum()),
        "n_added_to_queue_from_allow_true_positive": int((moved_to_queue & any_true).sum()),
        "n_promoted_to_priority_review": int(promoted.sum()),
        "note": "rule-forced queue entries are included in every tier and simulation metric",
    }


def _consistency(proba: np.ndarray) -> dict[str, Any]:
    j_sev = LABELS.index("severe_toxic")
    j_tox = LABELS.index("toxic")
    violations = (proba[:, j_sev] >= 0.5) & (proba[:, j_tox] < 0.5)
    return {
        "hierarchy_violation_rate": float(violations.mean()),
        "n_violations": int(violations.sum()),
        "n_rows": int(len(proba)),
    }


def _allow_false_negatives(final: np.ndarray, y: np.ndarray) -> dict[str, int]:
    allowed = final == ALLOW
    return {label: int((allowed & (y[:, j] == 1)).sum()) for j, label in enumerate(LABELS)}


def _review_workload(final: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Review demand before simulation; all flagged comments consume human capacity."""
    needs_review = np.isin(final, (PRIORITY, HUMAN))
    n = int(needs_review.sum())
    true_positive = int((needs_review & y.any(axis=1)).sum())
    return {
        "n_total": len(final),
        "n_requires_human_review": n,
        "n_priority_review": int((final == PRIORITY).sum()),
        "n_human_review": int((final == HUMAN).sum()),
        "review_fraction": n / len(final) if len(final) else 0.0,
        "n_truly_positive": true_positive,
        "precision": true_positive / n if n else None,
        "precision_ci95": list(wilson_interval(true_positive, n)) if n else None,
    }


def _rate_ratio(
    with_fp: int, with_n: int, without_fp: int, without_n: int
) -> tuple[float | None, list[float] | None]:
    """Ratio of two event rates with a 95% log-ratio (Katz) interval.

    A 0.5 continuity correction is applied to the interval when a cell is
    zero; the point ratio is None when the without-term rate is exactly zero.
    """
    if with_n == 0 or without_n == 0:
        return None, None
    a, n1, b, n2 = float(with_fp), float(with_n), float(without_fp), float(without_n)
    ratio = (a / n1) / (b / n2) if b > 0 else None
    if a == 0 or b == 0:
        a, b, n1, n2 = a + 0.5, b + 0.5, n1 + 1.0, n2 + 1.0
    estimate = (a / n1) / (b / n2)
    se = math.sqrt(max(1 / a - 1 / n1 + 1 / b - 1 / n2, 0.0))
    return ratio, [estimate * math.exp(-1.959964 * se), estimate * math.exp(1.959964 * se)]


def _subgroup_false_discoveries(
    predicted_positive: np.ndarray, correct: np.ndarray, subgroup: np.ndarray
) -> dict[str, Any]:
    positive = predicted_positive & subgroup
    n_pos = int(positive.sum())
    n_fp = int((positive & ~correct).sum())
    return {
        "n_rows": int(subgroup.sum()),
        "n_predicted_positive": n_pos,
        "n_false_positive": n_fp,
        "false_discovery_rate": (n_fp / n_pos) if n_pos else None,
        "false_discovery_rate_ci95": list(wilson_interval(n_fp, n_pos)) if n_pos else None,
    }


def _compare_subgroups(
    predicted_positive: np.ndarray, correct: np.ndarray, has_term: np.ndarray
) -> dict[str, Any]:
    with_term = _subgroup_false_discoveries(predicted_positive, correct, has_term)
    without = _subgroup_false_discoveries(predicted_positive, correct, ~has_term)
    ratio, ci = _rate_ratio(
        with_term["n_false_positive"],
        with_term["n_predicted_positive"],
        without["n_false_positive"],
        without["n_predicted_positive"],
    )
    return {
        "with_identity_term": with_term,
        "without_identity_term": without,
        "false_discovery_rate_ratio": ratio,
        "false_discovery_rate_ratio_ci95": ci,
    }


def _compare_clean_negative_subgroups(
    predicted_positive: np.ndarray, any_true: np.ndarray, has_term: np.ndarray
) -> dict[str, Any]:
    """FPR among comments with no positive label, including allowed comments."""
    groups: dict[str, dict[str, Any]] = {}
    for name, subgroup in (("with_identity_term", has_term), ("without_identity_term", ~has_term)):
        negatives = subgroup & ~any_true
        n_negative = int(negatives.sum())
        n_fp = int((negatives & predicted_positive).sum())
        groups[name] = {
            "n_actual_negative": n_negative,
            "n_false_positive": n_fp,
            "false_positive_rate": n_fp / n_negative if n_negative else None,
            "false_positive_rate_ci95": (
                list(wilson_interval(n_fp, n_negative)) if n_negative else None
            ),
        }
    with_term, without = groups["with_identity_term"], groups["without_identity_term"]
    ratio, ci = _rate_ratio(
        with_term["n_false_positive"],
        with_term["n_actual_negative"],
        without["n_false_positive"],
        without["n_actual_negative"],
    )
    return {
        **groups,
        "false_positive_rate_ratio": ratio,
        "false_positive_rate_ratio_ci95": ci,
    }


def _base_rate(any_true: np.ndarray, subgroup: np.ndarray) -> dict[str, Any]:
    n = int(subgroup.sum())
    positives = int((any_true & subgroup).sum())
    return {
        "n_rows": n,
        "n_truly_positive_any_label": positives,
        "rate": (positives / n) if n else None,
    }


def _subgroup_false_omission(
    tiers: np.ndarray, any_true: np.ndarray, subgroup: np.ndarray
) -> dict[str, Any]:
    allowed = (tiers == ALLOW) & subgroup
    n_allowed = int(allowed.sum())
    n_missed = int((allowed & any_true).sum())
    return {
        "n_allowed": n_allowed,
        "n_truly_positive": n_missed,
        "false_omission_rate": (n_missed / n_allowed) if n_allowed else None,
        "false_omission_rate_ci95": (
            list(wilson_interval(n_missed, n_allowed)) if n_allowed else None
        ),
    }


def _by_term(
    texts: list[str], predicted_positive: np.ndarray, correct: np.ndarray, top: int = 12
) -> list[dict[str, Any]]:
    """False discoveries per identity term, with predicted positives as denominator."""
    hits: dict[str, np.ndarray] = {t: np.zeros(len(texts), dtype=bool) for t in IDENTITY_TERMS}
    for i, text in enumerate(texts):
        if not predicted_positive[i]:
            continue
        for term in {m.lower() for m in IDENTITY_PATTERN.findall(text)}:
            hits[term][i] = True
    rows: list[dict[str, Any]] = []
    for term, mask in hits.items():
        n = int(mask.sum())
        if n == 0:
            continue
        fp = int((mask & ~correct).sum())
        rows.append(
            {
                "term": term,
                "n_predicted_positive": n,
                "n_false_positive": fp,
                "false_discovery_rate": fp / n,
            }
        )
    rows.sort(key=lambda r: (-int(r["n_predicted_positive"]), str(r["term"])))
    return rows[:top]


def _identity_concentration(
    routing: dict[str, Any], identity: np.ndarray, texts: list[str]
) -> dict[str, Any]:
    """Do false positives concentrate on identity-mentioning comments?

    "Identity term" is the fixed word list in review_router.signals, a proxy
    and not an annotation. Reported on three bases: the population tier (the
    classifier alone, population thresholds only), the model tier (after the
    subgroup thresholds, before rules) and the final tier (what the system
    does); pooled over the two review tiers and per tier. Both tiers require
    human confirmation. A false discovery has no positive ground-truth label.
    Conventional FPR uses all comments with no positive label as denominator
    and is reported separately. Neither diagnostic identifies protected groups.
    """
    any_true: np.ndarray = routing["any_true"]
    has_term = identity.astype(bool)
    out: dict[str, Any] = {
        "definition": "identity term = any whole-word match of "
        "review_router.signals.IDENTITY_TERMS; "
        "predicted positive = routed to priority_review or human_review; false positive = "
        "no ground-truth label is positive, for both review tiers; "
        "FDR = false positives / predicted positives; FDR ratio = "
        "with-term FDR / without-term FDR with a 95% log-ratio interval. "
        "clean_negative_false_positives separately reports FPR = clean comments routed "
        "to the tier / all clean comments in that subgroup. "
        "allow_false_omission = truly positive allowed comments / all allowed comments.",
        "identity_terms": sorted(IDENTITY_TERMS),
    }
    for basis, correct_key in (
        ("population_tier", "correct_population"),
        ("model_tier", "correct_model"),
        ("final_tier", "correct"),
    ):
        tiers = routing[basis]
        correct = routing[correct_key]
        section: dict[str, Any] = {
            "base_rate": {
                "with_identity_term": _base_rate(any_true, has_term),
                "without_identity_term": _base_rate(any_true, ~has_term),
            },
            "predicted_positive": _compare_subgroups(tiers != ALLOW, correct, has_term),
            PRIORITY: _compare_subgroups(tiers == PRIORITY, correct, has_term),
            HUMAN: _compare_subgroups(tiers == HUMAN, correct, has_term),
            "clean_negative_false_positives": {
                "predicted_positive": _compare_clean_negative_subgroups(
                    tiers != ALLOW, any_true, has_term
                ),
                PRIORITY: _compare_clean_negative_subgroups(tiers == PRIORITY, any_true, has_term),
                HUMAN: _compare_clean_negative_subgroups(tiers == HUMAN, any_true, has_term),
            },
            # False omission is conditioned on allowed comments, not all true
            # positives. It is descriptive and may depend on subgroup base rates.
            "allow_false_omission": {
                "with_identity_term": _subgroup_false_omission(tiers, any_true, has_term),
                "without_identity_term": _subgroup_false_omission(tiers, any_true, ~has_term),
            },
        }
        if basis == "final_tier":
            section["by_term"] = _by_term(texts, tiers != ALLOW, correct)
        out[basis] = section
    return out


def _prevalence_shift(
    split_counts: dict[str, dict[str, int]], y_test: np.ndarray, proba: np.ndarray
) -> dict[str, Any]:
    """Train-file versus test prevalence, and the volume the model expects on test.

    Sum of probabilities is a model-implied positive count. Comparing it with
    observed labels diagnoses aggregate calibration; it does not identify the
    cause of any difference between training and test prevalence.
    """
    train_rows = sum(split_counts[s]["rows"] for s in ("train", "calib", "thresh"))
    n_test = int(len(y_test))
    out: dict[str, Any] = {
        "definition": "volume_overshoot = model-expected positives (sum of calibrated p on the "
        "test rows) / actual positives - 1; prevalence ratio = training-file prevalence / "
        "test prevalence",
        "n_train_file_rows": int(train_rows),
        "n_test_rows": n_test,
    }
    for j, label in enumerate(LABELS):
        train_pos = sum(split_counts[s][label] for s in ("train", "calib", "thresh"))
        prev_train = train_pos / train_rows if train_rows else None
        actual = int(y_test[:, j].sum())
        expected_model = float(proba[:, j].sum())
        expected_prev = (prev_train or 0.0) * n_test
        out[label] = {
            "prevalence_train_file": prev_train,
            "prevalence_test": actual / n_test if n_test else None,
            "prevalence_ratio_train_over_test": (prev_train / (actual / n_test))
            if actual and prev_train
            else None,
            "actual_positives_test": actual,
            "model_expected_positives_test": expected_model,
            "train_prevalence_expected_positives_test": expected_prev,
            "volume_overshoot_model": (expected_model / actual - 1) if actual else None,
            "volume_overshoot_train_prevalence": (expected_prev / actual - 1) if actual else None,
        }
    return out


# --------------------------------------------------------------------------- simulation


def queue_inputs(
    proba: np.ndarray,
    y: np.ndarray,
    final: np.ndarray,
    policy: Policy,
    high_risk_min_weight: float,
) -> dict[str, Any]:
    """The combined review pool, rebuilt identically from saved prediction columns."""
    queued = np.flatnonzero(np.isin(final, (PRIORITY, HUMAN)))
    weights = np.array([policy.severity_weights.get(label, 0.0) for label in LABELS])
    p = proba[queued]
    truth = y[queued]
    harm = (truth * weights).max(axis=1)
    severity = (p * weights).max(axis=1) if len(queued) else np.zeros(0)
    return {
        "queued": queued,
        "harm": harm,
        "high_risk": harm >= high_risk_min_weight,
        "priorities": {
            "fifo": np.zeros(len(queued)),
            "prob": p.max(axis=1) if len(queued) else np.zeros(0),
            "severity": severity,
            "priority": priority_review_scores(final[queued] == PRIORITY, severity),
        },
    }


def _trigger_reasons(routing: dict[str, Any]) -> np.ndarray:
    """Why each row is where it is: the rule ids that fired and/or the model labels."""
    triggers: np.ndarray = routing["triggers"]
    out = np.empty(len(triggers), dtype=object)
    for i in range(len(triggers)):
        labels = [label for j, label in enumerate(LABELS) if triggers[i, j]]
        parts = []
        if routing["rule_ids"][i]:
            parts.append("rule:" + routing["rule_ids"][i].replace(";", "+"))
        if labels:
            parts.append("model:" + "+".join(labels))
        out[i] = "|".join(parts) if parts else "none"
    return out


def _run_simulation(
    proba: np.ndarray,
    y: np.ndarray,
    routing: dict[str, Any],
    ids: np.ndarray,
    policy: Policy,
    config: RunConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Simulate every (load, seed, strategy); return the report section and the per-job rows."""
    inputs = queue_inputs(proba, y, routing["final_tier"], policy, config.high_risk_min_weight)
    queued: np.ndarray = inputs["queued"]
    final = routing["final_tier"]
    queue_fraction = len(queued) / len(final) if len(final) else 0.0
    if len(queued) == 0:
        return {
            "note": "nothing requires human review; simulation skipped",
            "assumptions": {
                "queued_jobs": 0,
                "queue_tiers": [PRIORITY, HUMAN],
                "queue_fraction": 0.0,
            },
            "table": [],
        }, []
    harm, high_risk, priorities = inputs["harm"], inputs["high_risk"], inputs["priorities"]
    reasons = _trigger_reasons(routing)[queued]
    queued_ids = ids[queued]

    rows: list[dict[str, Any]] = []
    job_rows: list[dict[str, Any]] = []
    pooled: dict[str, list[JobRecords]] = {}
    for load in config.loads_per_hour:
        for seed in config.sim_seeds:
            scenario = draw_scenario(seed, load, len(queued), config.sim)
            for strategy in STRATEGIES:
                result, records = simulate(
                    scenario, priorities[strategy], harm, high_risk, config.sim, strategy
                )
                rows.append({"load_per_hour": load, "seed": seed, **result.as_json()})
                pooled.setdefault(_key(load, strategy), []).append(records)
                job_rows.extend(
                    records.rows(
                        strategy=strategy,
                        load_per_hour=load,
                        seed=seed,
                        comment_id=queued_ids[records.job_index],
                        trigger_reason=reasons[records.job_index],
                    )
                )

    summary = _summarise(rows)
    metrics = {
        key: {
            "pooled_over_seeds": time_metrics(JobRecords.concat(parts)),
            "n_seeds": len(parts),
        }
        for key, parts in pooled.items()
    }
    router = summary.get(_key(config.primary_load_per_hour, config.primary_strategy), {})
    fifo = summary.get(_key(config.primary_load_per_hour, "fifo"), {})
    thesis_router = summary.get(_key(config.thesis_load_per_hour, config.primary_strategy), {})
    thesis_fifo = summary.get(_key(config.thesis_load_per_hour, "fifo"), {})

    def mean(entry: dict[str, Any], field: str) -> float | None:
        value = entry.get(field)
        return float(value["mean"]) if value else None

    def high_risk_comparison(field: str) -> dict[str, dict[str, float | None]]:
        return {
            "primary": {
                "router": mean(router, field),
                "fifo": mean(fifo, field),
                "load_per_hour": config.primary_load_per_hour,
            },
            "thesis": {
                "router": mean(thesis_router, field),
                "fifo": mean(thesis_fifo, field),
                "load_per_hour": config.thesis_load_per_hour,
            },
        }

    headline = _headline(metrics, config)
    section = {
        "assumptions": {
            **asdict(config.sim),
            "capacity_per_hour": config.sim.capacity_per_hour,
            "arrivals": "Post-admission Poisson review arrivals; jobs sampled with replacement "
            "from the combined priority_review and human_review pool",
            "arrival_rates_are": "post-admission review demand, not all incoming comments",
            "queue_tiers": [PRIORITY, HUMAN],
            "queue_fraction": queue_fraction,
            "equivalent_input_loads_per_hour": [
                load / queue_fraction for load in config.loads_per_hour
            ],
            "priority_order": "priority_review first, then human_review; within each tier, "
            "higher max predicted probability times severity weight first; ties use arrival order",
            "handle_time": "deterministic",
            "review_completion_definition": (
                "simulated reviewer service finished; moderation requires human confirmation; "
                "no enforcement outcome is observed"
            ),
            "wait_definition": "start of review minus arrival, over jobs that started "
            "(completed or in progress); never-started jobs are counted, not timed",
            "completion_latency_definition": "completion minus arrival, over jobs completed "
            "within the horizon",
            "p99_min_samples": P99_MIN_SAMPLES,
            "harm_proxy": "max severity weight over TRUE labels (0 if clean); not real-world harm",
            "high_risk": f"harm proxy >= {config.high_risk_min_weight}",
            "loads_per_hour": list(config.loads_per_hour),
            "seeds": list(config.sim_seeds),
            "queued_jobs": int(len(queued)),
            "queued_high_risk": int(high_risk.sum()),
            "queued_priority_review": int((final[queued] == PRIORITY).sum()),
            "queued_human_review": int((final[queued] == HUMAN).sum()),
            "per_job_records": "simulation_jobs.csv in the run directory; every time metric "
            "here is recomputable from it",
        },
        "primary": {
            "load_per_hour": config.primary_load_per_hour,
            "strategy": config.primary_strategy,
            "note": "queue-clearing gates read this load; the router strategy is compared "
            "against fifo at the same load for time-to-action",
        },
        "thesis": {
            "load_per_hour": config.thesis_load_per_hour,
            "strategy": config.primary_strategy,
            "note": "harm handled per reviewer-hour only differs between orderings when "
            "the queue does not clear, so the FIFO comparison reads this load",
        },
        "headline": headline,
        # Gate inputs. Ordering cannot change harm handled once everything is
        # handled, so that ratio is read at the thesis (overload) load; the
        # time-to-action ratio is read at the primary (near-capacity) load.
        "harm_per_reviewer_hour": {
            "router": mean(thesis_router, "harm_per_reviewer_hour"),
            "fifo": mean(thesis_fifo, "harm_per_reviewer_hour"),
            "load_per_hour": config.thesis_load_per_hour,
        },
        "high_risk_wait_p90": {
            "router": mean(router, "high_risk_wait_p90"),
            "fifo": mean(fifo, "high_risk_wait_p90"),
            "load_per_hour": config.primary_load_per_hour,
        },
        "high_risk_handled": high_risk_comparison("high_risk_handled"),
        "high_risk_arrived": high_risk_comparison("high_risk_arrived"),
        "completion_ratio": mean(router, "completion_ratio"),
        "backlog_end": mean(router, "backlog_end"),
        "high_risk_unfinished": mean(router, "high_risk_unhandled"),
        "queue_depth_p95": mean(router, "queue_depth_p95"),
        "reviewer_utilization": mean(router, "reviewer_utilization"),
        "time_metrics": metrics,
        "summary": summary,
        "paired": _paired(rows),
        "table": rows,
    }
    return section, job_rows


def _headline(metrics: dict[str, Any], config: RunConfig) -> dict[str, Any]:
    """The configured time metric, conditional on each strategy's observed job statuses.

    Reduction = 1 - router / fifo. When the FIFO value is zero the percentage is
    meaningless and only the absolute difference is reported. The high-risk
    variant is supplementary and never a substitute for the headline.
    """
    metric, pct = config.headline_metric, config.headline_percentile
    load = config.primary_load_per_hour

    def read(strategy: str, field: str, percentile: int) -> tuple[float | None, int, bool]:
        block = metrics.get(_key(load, strategy), {}).get("pooled_over_seeds", {}).get(field, {})
        return (
            block.get(f"p{percentile}"),
            int(block.get("n", 0)),
            bool(block.get("p99_reliable", False)),
        )

    def compare(field: str, percentile: int) -> dict[str, Any]:
        r, n_r, rel_r = read(config.primary_strategy, field, percentile)
        f_, n_f, rel_f = read("fifo", field, percentile)
        out: dict[str, Any] = {
            "router": r,
            "fifo": f_,
            "n_router": n_r,
            "n_fifo": n_f,
            "absolute_difference_min": (f_ - r) if r is not None and f_ is not None else None,
            "reduction": None,
        }
        if r is not None and f_ is not None and f_ > 0:
            out["reduction"] = 1 - r / f_
        elif f_ == 0:
            out["note"] = (
                "FIFO value is zero: a percentage is meaningless, read the absolute difference"
            )
        else:
            out["note"] = "No comparable samples: a percentage is unavailable"
        if percentile == 99:
            out["p99_reliable"] = rel_r and rel_f
        return out

    return {
        "metric": metric,
        "percentile": pct,
        "load_per_hour": load,
        "router_strategy": config.primary_strategy,
        "population": (
            "jobs that started, including in-progress reviews"
            if metric == "wait"
            else "jobs completed within the horizon"
        ) + "; pooled over seeds; populations may differ by strategy; "
        "read with completion and unfinished counts",
        "selected": compare(metric, pct),
        "supplementary_high_risk": {
            "note": "reported alongside, never a substitute for the selected metric",
            "wait_p90": compare("high_risk_wait", 90) | {"percentile": 90},
        },
    }


_PAIRS: tuple[tuple[str, str], ...] = (
    ("priority", "fifo"),
    ("priority", "prob"),
    ("priority", "severity"),
    ("severity", "fifo"),
    ("prob", "fifo"),
    ("severity", "prob"),
)
_HIGHER_IS_BETTER: dict[str, bool] = {
    "high_risk_handled": True,
    "harm_per_reviewer_hour": True,
    "high_risk_unhandled": False,
    "high_risk_wait_p90": False,
    "wait_p90": False,
}


def _paired(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-seed paired differences between orderings.

    Every ordering replays the same arrivals for a given (load, seed), so the
    honest comparison is paired: how many seeds favour the first ordering, not
    whether the seed means overlap.
    """
    by = {(r["load_per_hour"], r["seed"], r["strategy"]): r for r in rows}
    loads = sorted({r["load_per_hour"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    out: dict[str, dict[str, Any]] = {}
    for load in loads:
        for first, second in _PAIRS:
            entry: dict[str, Any] = {}
            for field, higher in _HIGHER_IS_BETTER.items():
                diffs = []
                for seed in seeds:
                    x = by.get((load, seed, first), {}).get(field)
                    y = by.get((load, seed, second), {}).get(field)
                    if x is None or y is None:
                        continue
                    diffs.append(float(x) - float(y))
                if not diffs:
                    entry[field] = None
                    continue
                entry[field] = {
                    "mean_diff": float(np.mean(diffs)),
                    "min_diff": float(min(diffs)),
                    "max_diff": float(max(diffs)),
                    "n_seeds": len(diffs),
                    "n_seeds_first_better": sum(1 for d in diffs if d != 0 and (d > 0) == higher),
                    "n_seeds_tied": sum(1 for d in diffs if d == 0),
                }
            out[f"{first}_vs_{second}@{load:g}"] = entry
    return out


def _key(load: float, strategy: str) -> str:
    return f"{strategy}@{load:g}"


_SUMMARY_FIELDS = tuple(f for f in SimResult.__dataclass_fields__ if f != "strategy")


def _summarise(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """mean/std across seeds for every (strategy, load)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_key(row["load_per_hour"], row["strategy"]), []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for key, members in groups.items():
        entry: dict[str, Any] = {
            "strategy": members[0]["strategy"],
            "load_per_hour": members[0]["load_per_hour"],
            "n_seeds": len(members),
        }
        for field in _SUMMARY_FIELDS:
            values = [m[field] for m in members if m[field] is not None]
            entry[field] = (
                {"mean": float(np.mean(values)), "std": float(np.std(values))} if values else None
            )
        out[key] = entry
    return out


# --------------------------------------------------------------------------- report


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, dict) and "mean" in value:
        return f"{value['mean']:.{digits}f} ± {value['std']:.{digits}f}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _report_md(report: dict[str, Any], manifest: dict[str, Any]) -> str:
    workload = report["review_workload"]
    lines = [
        f"# Run {manifest['run_id']} ({manifest['label']})",
        "",
        f"commit `{manifest['git_commit']}`" + (" (dirty)" if manifest["git_dirty"] else ""),
        f", seed {manifest['seed']}, scored test rows "
        f"{manifest['split_label_counts']['test_scored']['rows']}.",
        "",
        "Decision mode: human confirmation. Both priority_review and human_review require "
        "reviewers; the router emits no automatic enforcement actions. This offline evaluation "
        "does not observe completed human decisions.",
        "",
        f"Review demand: {workload['n_requires_human_review']} / {workload['n_total']} comments "
        f"({_fmt(workload['review_fraction'])}), including "
        f"{workload['n_priority_review']} priority "
        f"and {workload['n_human_review']} regular reviews. Pool precision is "
        f"{_fmt(workload['precision'])}, Wilson 95% CI {workload['precision_ci95']}.",
        "",
        "## Per-label quality (scored test set)",
        "",
        "| label | positives | AP (selection) | AP | ROC-AUC | human thr | P@human | R@human "
        "| priority thr | P@priority | R@priority |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    selection = report["threshold_selection"]["per_label"]
    if report["synthetic"]:
        lines[2:2] = ["SYNTHETIC pipeline check only; these are not Jigsaw results.", ""]
    if report.get("exploratory"):
        source = report["score_source"]
        lines[2:2] = [
            f"EXPLORATORY external-score evaluation for {source['model_name']}; "
            "these results do not certify the full baseline or deployment readiness.",
            "",
        ]
    for label, m in report["per_label"].items():
        h, a = m["at_human_review"], m["at_priority_review"]
        lines.append(
            f"| {label} | {m['positives']} | {_fmt(selection[label]['average_precision'])} "
            f"| {_fmt(m['average_precision'])} | {_fmt(m['roc_auc'])} "
            f"| {_fmt(h['threshold'], 4)} | {_fmt(h['precision'])} | {_fmt(h['recall'])} "
            f"| {_fmt(a['threshold'], 4)} | {_fmt(a['precision'])} | {_fmt(a['recall'])} |"
        )
    lines += [
        "",
        "## Routing tiers (rules applied, rule-forced entries included)",
        "",
        "| tier | n | coverage | precision | 95% CI | note |",
        "|---|---:|---:|---:|---|---|",
    ]
    for tier, m in report["tiers"].items():
        ci = m["precision_ci95"]
        ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
        lines.append(
            f"| {tier} | {m['n_predicted_positive']} | {_fmt(m['coverage'])} "
            f"| {_fmt(m['precision'])} | {ci_s} | {m['precision_note'] or ''} |"
        )
    lines += [
        "",
        "Model tier before rules (for comparison):",
        "",
        "| tier | n | coverage | precision | 95% CI |",
        "|---|---:|---:|---:|---|",
    ]
    for tier, m in report["tiers_model_only"].items():
        ci = m["precision_ci95"]
        ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
        lines.append(
            f"| {tier} | {m['n_predicted_positive']} | {_fmt(m['coverage'])} "
            f"| {_fmt(m['precision'])} | {ci_s} |"
        )
    rules = report["rules"]
    lines += [
        "",
        "## Threshold-selection split (development data, not test results)",
        "",
        "Precision targets are selected per label. Combining labels, separating review "
        "priorities and applying rules can change final-tier precision. Both tiers join the queue.",
        "",
        "| final tier | n | coverage | precision |",
        "|---|---:|---:|---:|",
    ]
    for tier, m in report["threshold_selection"]["tiers"].items():
        lines.append(
            f"| {tier} | {m['n_predicted_positive']} | {_fmt(m['coverage'])} "
            f"| {_fmt(m['precision'], 4)} |"
        )
    lines += [
        "",
        f"On the test set, rules matched on {rules['n_rows_with_any_rule']} rows; "
        f"{rules['n_added_to_queue_from_allow']} moved allow -> review "
        f"({rules['n_added_to_queue_from_allow_true_positive']} truly positive); "
        f"{rules['n_promoted_to_priority_review']} were promoted to priority_review.",
        f"Hierarchy violation rate: {report['consistency']['hierarchy_violation_rate']:.4%}.",
        "",
        "## Subgroup thresholds",
        "",
        f"Declared: {report['subgroup_thresholds']['threshold_selection']['declared']}. "
        f"Thresholds on the subgroup: {report['subgroup_thresholds']['thresholds']}.",
        "",
        "| split | tier | population-tier rows | removed by subgroup threshold "
        "| of which truly positive |",
        "|---|---|---:|---:|---:|",
    ]
    for split in ("threshold_selection", "test"):
        for tier, st in report["subgroup_thresholds"][split].items():
            if tier == "declared":
                continue
            lines.append(
                f"| {split} | {tier} | {st['n_population_tier']} "
                f"| {st['n_removed_by_subgroup_threshold']} "
                f"| {st['n_removed_truly_positive_any_label']} |"
            )
    lines += [
        "",
        "## False discoveries and false positives on identity mentions",
        "",
        "Identity term = whole-word match of the fixed list in review_router/signals.py "
        "(a proxy, not an annotation). FDR = false positives / predicted positives. "
        "The FDR ratio compares comments with a term to comments without one. "
        "population_tier = classifier alone; model_tier = after subgroup thresholds, before "
        "rules; final_tier = after rules.",
        "",
        "| split | basis | tier | with term (FP / predicted positive) "
        "| without term (FP / predicted positive) | FDR ratio | 95% CI |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    miss_lines: list[str] = []
    fpr_lines = [
        "",
        "Conventional FPR uses all comments with no positive label as the denominator, "
        "including comments the system allowed. It measures a different error rate from FDR.",
        "",
        "| split | basis | tier | with term (FP / clean comments) "
        "| without term (FP / clean comments) | FPR ratio | 95% CI |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for split, section in report["identity_false_positives"].items():
        for basis in ("population_tier", "model_tier", "final_tier"):
            for tier in ("predicted_positive", PRIORITY, HUMAN):
                cmp = section[basis][tier]
                w, wo = cmp["with_identity_term"], cmp["without_identity_term"]
                ci = cmp["false_discovery_rate_ratio_ci95"]
                ci_s = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""
                lines.append(
                    f"| {split} | {basis} | {tier} | {w['n_false_positive']} / "
                    f"{w['n_predicted_positive']} | {wo['n_false_positive']} / "
                    f"{wo['n_predicted_positive']} | {_fmt(cmp['false_discovery_rate_ratio'], 2)} "
                    f"| {ci_s} |"
                )
                fpr = section[basis]["clean_negative_false_positives"][tier]
                fw, fwo = fpr["with_identity_term"], fpr["without_identity_term"]
                fci = fpr["false_positive_rate_ratio_ci95"]
                fci_s = f"[{fci[0]:.2f}, {fci[1]:.2f}]" if fci else ""
                fpr_lines.append(
                    f"| {split} | {basis} | {tier} | {fw['n_false_positive']} / "
                    f"{fw['n_actual_negative']} | {fwo['n_false_positive']} / "
                    f"{fwo['n_actual_negative']} | {_fmt(fpr['false_positive_rate_ratio'], 2)} "
                    f"| {fci_s} |"
                )
        miss = section["final_tier"]["allow_false_omission"]
        miss_lines.append(
            f"\nFalse omission rate, truly positive / all allowed comments ({split}, final tier): "
            f"{_fmt(miss['with_identity_term']['false_omission_rate'])} with an identity term "
            f"({miss['with_identity_term']['n_truly_positive']} of "
            f"{miss['with_identity_term']['n_allowed']}) vs "
            f"{_fmt(miss['without_identity_term']['false_omission_rate'])} without."
        )
    lines += fpr_lines + miss_lines
    lines += ["", "Per term (test split, final tier, most frequent first):", ""]
    lines += ["| term | predicted positive | false positives | FDR |", "|---|---:|---:|---:|"]
    for row in report["identity_false_positives"]["test"]["final_tier"]["by_term"]:
        lines.append(
            f"| {row['term']} | {row['n_predicted_positive']} | {row['n_false_positive']} "
            f"| {_fmt(row['false_discovery_rate'])} |"
        )
    lines += [
        "",
        "## Queue simulation (mean ± std over seeds)",
        "",
    ]
    sim = report["simulation"]
    if "summary" not in sim:
        lines.append(sim.get("note", "no simulation"))
    else:
        a = sim["assumptions"]
        lines.append(
            f"{a['reviewers']} reviewers, {a['handle_minutes']} min/item, {a['horizon_hours']} h "
            f"(capacity {a['capacity_per_hour']:g}/h); {a['queued_jobs']} queued jobs, "
            f"{a['queued_high_risk']} high-risk; harm proxy = {a['harm_proxy']}."
        )
        lines.append(
            f"Arrival loads are post-admission review demand. The combined review pool is "
            f"{a['queue_fraction']:.2%} of input comments; equivalent input rates under this "
            f"fixed pool fraction are {a['equivalent_input_loads_per_hour']}. "
            f"Priority order: {a['priority_order']}."
        )
        lines.append(
            "Wait quantiles are minutes until service starts, for all started jobs. "
            "High-risk unhandled includes jobs still in service at the horizon; "
            "backlog includes only jobs that have not started."
        )
        harm, wait = sim["harm_per_reviewer_hour"], sim["high_risk_wait_p90"]
        lines += [
            "",
            f"Gate inputs: harm per reviewer-hour at {harm['load_per_hour']:g}/h, "
            f"{sim['primary']['strategy']} {_fmt(harm['router'], 2)} "
            f"vs fifo {_fmt(harm['fifo'], 2)}; "
            f"high-risk wait p90 at {wait['load_per_hour']:g}/h, "
            f"{sim['primary']['strategy']} {_fmt(wait['router'], 1)} "
            f"vs fifo {_fmt(wait['fifo'], 1)} min.",
        ]
        lines += [
            "",
            "| load/h | strategy | arrivals | handled | high-risk handled | high-risk unhandled "
            "| harm/reviewer-h | wait p50 | wait p90 | HR wait p50 | HR wait p90 "
            "| backlog end | util |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for entry in sim["summary"].values():
            e = entry
            cells = [
                f"{e['load_per_hour']:g}",
                e["strategy"],
                _fmt(e["n_arrivals"], 1),
                _fmt(e["n_handled"], 1),
                _fmt(e["high_risk_handled"], 1),
                _fmt(e["high_risk_unhandled"], 1),
                _fmt(e["harm_per_reviewer_hour"], 2),
                _fmt(e["wait_p50"], 1),
                _fmt(e["wait_p90"], 1),
                _fmt(e["high_risk_wait_p50"], 1),
                _fmt(e["high_risk_wait_p90"], 1),
                _fmt(e["backlog_end"], 1),
                _fmt(e["reviewer_utilization"], 2),
            ]
            lines.append("| " + " | ".join(cells) + " |")
    if "time_metrics" in sim:
        h = sim["headline"]
        sel = h["selected"]
        lines += [
            "",
            f"Headline ({h['metric']} p{h['percentile']}, {h['router_strategy']} vs fifo at "
            f"{h['load_per_hour']:g}/h, pooled over seeds): {_fmt(sel['router'], 2)} vs "
            f"{_fmt(sel['fifo'], 2)} min (n = {sel['n_router']} / {sel['n_fifo']}); "
            + (
                f"reduction {100 * sel['reduction']:.0f}%."
                if sel.get("reduction") is not None
                else f"absolute difference {_fmt(sel['absolute_difference_min'], 2)} min "
                f"({sel.get('note', 'percentage unavailable')})."
            ),
            "",
            "Time metrics from the per-job records, pooled over seeds (minutes):",
            "",
            "| load/h | strategy | arrivals | completed | in progress | not started "
            "| HR completed / in progress / not started | wait p50 / p90 / p99 (n) "
            "| completion p50 / p90 / p99 (n) | p99 reliable |",
            "|---:|---|---:|---:|---:|---:|---|---|---|---|",
        ]
        for key, block in sim["time_metrics"].items():
            tm = block["pooled_over_seeds"]
            strategy, load = key.split("@")
            st, hr = tm["status"], tm["high_risk_status"]
            w, c = tm["wait"], tm["completion_latency"]
            lines.append(
                f"| {load} | {strategy} | {tm['n_arrivals']} | {st['completed']} | "
                f"{st['in_progress']} | {st['not_started']} | {hr['completed']} / "
                f"{hr['in_progress']} / {hr['not_started']} | {_fmt(w['p50'], 1)} / "
                f"{_fmt(w['p90'], 1)} / {_fmt(w['p99'], 1)} ({w['n']}) | {_fmt(c['p50'], 1)} / "
                f"{_fmt(c['p90'], 1)} / {_fmt(c['p99'], 1)} ({c['n']}) | "
                f"{'yes' if w['p99_reliable'] and c['p99_reliable'] else 'no'} |"
            )
    ps = report.get("prevalence_shift")
    if ps:
        lines += [
            "",
            "## Prevalence shift and predicted volume",
            "",
            "| label | prevalence train file | prevalence test | ratio | actual positives "
            "| model-expected positives | volume overshoot | ROC-AUC test | AP selection "
            "| AP test |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for label in LABELS:
            e = ps[label]
            m = report["per_label"][label]
            lines.append(
                f"| {label} | {_fmt(e['prevalence_train_file'], 4)} "
                f"| {_fmt(e['prevalence_test'], 4)} "
                f"| {_fmt(e['prevalence_ratio_train_over_test'], 2)} "
                f"| {e['actual_positives_test']} "
                f"| {_fmt(e['model_expected_positives_test'], 0)} "
                f"| {_fmt(e['volume_overshoot_model'], 2)} | {_fmt(m['roc_auc'])} "
                f"| {_fmt(report['threshold_selection']['per_label'][label]['average_precision'])} "
                f"| {_fmt(m['average_precision'])} |"
            )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- entry point


def _prepare_run(
    config_path: Path, config: RunConfig, corpus: Corpus, policy: Policy
) -> tuple[Path, dict[str, Any]]:
    if policy.decision_mode != "human_confirmation":
        raise ValueError("the current pipeline requires a human_confirmation policy")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + config.label
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = _manifest(config, config_path, corpus, run_id)
    shutil.copyfile(config_path, run_dir / "config.yaml")
    shutil.copyfile(config.policy_path, run_dir / "policy.yaml")
    corpus.train[["id", "split"]].to_csv(run_dir / "splits.csv", index=False)
    return run_dir, manifest


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    policy = load_policy(config.policy_path)
    if policy.decision_mode != "human_confirmation":
        raise ValueError("the current pipeline requires a human_confirmation policy")
    corpus = load_corpus(config.data_dir, config.split_fractions, config.seed)
    # Start provenance before training, so the run duration still includes fit/calibration.
    run_dir, manifest = _prepare_run(config_path, config, corpus, policy)

    def part(name: str) -> tuple[list[str], np.ndarray]:
        frame = corpus.train[corpus.train["split"] == name]
        return frame["comment_text"].astype(str).tolist(), frame[list(LABELS)].to_numpy(dtype=int)

    x_train, y_train = part("train")
    x_calib, y_calib = part("calib")
    x_thresh, _ = part("thresh")
    model = TfidfLogitModel(config.model).fit(x_train, y_train, config.seed)
    model.calibrate(x_calib, y_calib)
    p_thresh = model.predict_proba(x_thresh)
    p_test = model.predict_proba(corpus.test["comment_text"].astype(str).tolist())
    return _evaluate_scores(
        config, corpus, p_thresh, p_test, policy, run_dir, manifest, model=model
    )


def _validate_external_scores(
    corpus: Corpus, p_thresh: np.ndarray, p_test: np.ndarray, score_source: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, str]:
    """Validate score identity and values before writing any evaluation artefacts."""
    model_name = score_source.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("score_source must identify a nonempty model_name")
    if tuple(score_source.get("labels", ())) != LABELS:
        raise ValueError(f"score_source labels must match the canonical order {LABELS}")
    # Metadata itself must be reproducible JSON, not a repr of a live object.
    json.dumps(score_source, allow_nan=False)
    row_ids = score_source.get("row_ids")
    if not isinstance(row_ids, dict) or set(row_ids) != {"thresh", "test"}:
        raise ValueError("score_source row_ids must contain thresh and test IDs in score order")
    for name, frame in (("train", corpus.train), ("test", corpus.test)):
        if frame["id"].isna().any() or frame["id"].astype(str).str.strip().eq("").any():
            raise ValueError(f"external {name} corpus contains missing IDs")
        if frame["id"].astype(str).duplicated().any():
            raise ValueError(f"external {name} corpus contains duplicate IDs")
        if not frame[list(LABELS)].isin([0, 1]).all().all():
            raise ValueError(f"external {name} corpus contains non-binary labels")
        if frame["comment_text"].isna().any():
            raise ValueError(f"external {name} corpus contains missing texts")
    if set(corpus.train["id"].astype(str)) & set(corpus.test["id"].astype(str)):
        raise ValueError("external train and test corpus IDs overlap")
    if not corpus.train["split"].isin(("train", "calib", "thresh")).all():
        raise ValueError("external training corpus contains unknown split names")
    frames = {"thresh": corpus.train[corpus.train["split"] == "thresh"], "test": corpus.test}
    validated = []
    for name, scores in (("thresh", p_thresh), ("test", p_test)):
        expected_ids = frames[name]["id"].astype(str).tolist()
        if row_ids[name] != expected_ids:
            raise ValueError(f"score_source {name} row_ids do not match corpus row order")
        values = np.asarray(scores, dtype=float)
        if not expected_ids or values.shape != (len(expected_ids), len(LABELS)):
            raise ValueError(f"{name} scores must have shape ({len(expected_ids)}, {len(LABELS)})")
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{name} scores must be finite probabilities in [0, 1]")
        validated.append(values)
    cohort = {
        "train": corpus.train[["id", "split", "comment_text", *LABELS]].to_dict(orient="records"),
        "test": corpus.test[["id", "comment_text", *LABELS]].to_dict(orient="records"),
    }
    cohort_hash = hashlib.sha256(
        json.dumps(cohort, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
    return validated[0], validated[1], cohort_hash


def evaluate_scores(
    config_path: Path,
    corpus: Corpus,
    p_thresh: np.ndarray,
    p_test: np.ndarray,
    *,
    score_source: dict[str, Any],
    model: Any = None,
) -> Path:
    """Evaluate externally produced probabilities using the existing routing protocol.

    Columns must follow LABELS; rows must follow the corpus's threshold/test order.
    score_source requires model_name, labels, and row_ids with thresh/test lists in
    score order. Additional provenance, including calibration and sampled-cohort
    hashes, is preserved verbatim. Calibration must have happened upstream, without
    fitting on threshold-selection or test labels. Every external run is exploratory
    and cannot stand in for the pinned full-corpus baseline acceptance run.
    """
    config_path = config_path.resolve()
    config = load_config(config_path)
    policy = load_policy(config.policy_path)
    p_thresh, p_test, cohort_hash = _validate_external_scores(
        corpus, p_thresh, p_test, score_source
    )
    run_dir, manifest = _prepare_run(config_path, config, corpus, policy)
    # Retain the original corpus provenance separately. A sampled cohort must not
    # pass the existing full-corpus gate merely by carrying its source file hashes.
    manifest["source_data_sha256"] = manifest["data_sha256"]
    manifest["data_sha256"] = {"external_evaluation_cohort": cohort_hash}
    manifest["exploratory"] = True
    manifest["score_source"] = score_source
    manifest["unused_model_config"] = manifest["config"]["model"]
    manifest["config"]["model"] = {
        "kind": "external_scores", "model_name": score_source["model_name"]
    }
    return _evaluate_scores(
        config, corpus, p_thresh, p_test, policy, run_dir, manifest, model=model
    )


def _evaluate_scores(
    config: RunConfig,
    corpus: Corpus,
    p_thresh: np.ndarray,
    proba: np.ndarray,
    policy: Policy,
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    model: Any,
) -> Path:
    import pandas as pd

    run_id = manifest["run_id"]
    frame = corpus.train[corpus.train["split"] == "thresh"]
    x_thresh = frame["comment_text"].astype(str).tolist()
    y_thresh = frame[list(LABELS)].to_numpy(dtype=int)
    identity_thresh = identity_term_present(x_thresh)
    thresholds = select_thresholds(
        y_thresh, p_thresh, LABELS, policy, {"identity_term_present": identity_thresh}
    )
    selection_routing = _route(p_thresh, y_thresh, identity_thresh, thresholds, policy)
    (run_dir / "thresholds.json").write_text(json.dumps(thresholds.as_json(), indent=2))
    if model is not None:
        with (run_dir / "model.pkl").open("wb") as handle:
            pickle.dump(model, handle)

    x_test = corpus.test["comment_text"].astype(str).tolist()
    y_test = corpus.test[list(LABELS)].to_numpy(dtype=int)
    identity = identity_term_present(x_test)
    routing = _route(proba, y_test, identity, thresholds, policy)

    min_pos = int(policy.gates.get("min_predicted_positives_for_precision", 30))
    weights = np.array([policy.severity_weights.get(label, 0.0) for label in LABELS])
    predictions = pd.DataFrame({"id": corpus.test["id"].to_numpy()})
    for j, label in enumerate(LABELS):
        predictions[f"y_{label}"] = y_test[:, j]
    for j, label in enumerate(LABELS):
        predictions[f"p_{label}"] = proba[:, j]
    predictions["identity_term_present"] = identity.astype(int)
    predictions["severity_score"] = (proba * weights).max(axis=1)
    predictions["model_tier"] = routing["model_tier"]
    predictions["rule_ids"] = routing["rule_ids"]
    predictions["final_tier"] = routing["final_tier"]
    predictions["requires_human_review"] = np.isin(routing["final_tier"], (PRIORITY, HUMAN))
    predictions.to_csv(run_dir / "predictions.csv", index=False)

    synthetic = (config.data_dir / "SYNTHETIC.txt").is_file()
    report: dict[str, Any] = {
        "run_id": run_id,
        "eval_slice": "synthetic scored test rows" if synthetic else "scored test rows",
        "synthetic": synthetic,
        "policy_version": policy.version,
        "decision_contract": {
            "mode": "human_confirmation",
            "requires_human_confirmation": True,
            "automatic_actions": 0,
        },
        "review_workload": _review_workload(routing["final_tier"], y_test),
        "threshold_selection": {
            "note": "development data; per-label precision targets do not guarantee tier precision",
            "per_label": per_label_metrics(y_thresh, p_thresh, LABELS, thresholds.thresholds),
            "review_workload": _review_workload(selection_routing["final_tier"], y_thresh),
            "tiers": tier_metrics(
                selection_routing["final_tier"],
                selection_routing["correct"],
                (PRIORITY, HUMAN, ALLOW),
                min_pos,
            ),
            "tiers_model_only": tier_metrics(
                selection_routing["model_tier"],
                selection_routing["correct_model"],
                (PRIORITY, HUMAN, ALLOW),
                min_pos,
            ),
        },
        "per_label": per_label_metrics(y_test, proba, LABELS, thresholds.thresholds),
        "tiers": tier_metrics(
            routing["final_tier"], routing["correct"], (PRIORITY, HUMAN, ALLOW), min_pos
        ),
        "tiers_model_only": tier_metrics(
            routing["model_tier"], routing["correct_model"], (PRIORITY, HUMAN, ALLOW), min_pos
        ),
        "tier_correctness": {
            PRIORITY: "any label is truly positive; human confirmation is still required",
            HUMAN: "any label is truly positive",
        },
        "allow_false_negatives": _allow_false_negatives(routing["final_tier"], y_test),
        "identity_false_positives": {
            "threshold_selection": _identity_concentration(
                selection_routing, identity_thresh, x_thresh
            ),
            "test": _identity_concentration(routing, identity, x_test),
        },
        "rules": _rule_stats(routing, policy),
        "subgroup_thresholds": {
            "thresholds": thresholds.subgroup,
            "threshold_selection": _subgroup_stats(selection_routing, y_thresh, policy),
            "test": _subgroup_stats(routing, y_test, policy),
        },
        "consistency": _consistency(proba),
        "prevalence_shift": _prevalence_shift(manifest["split_label_counts"], y_test, proba),
    }
    if manifest.get("exploratory"):
        report["exploratory"] = True
        report["score_source"] = manifest["score_source"]
        report["eval_slice"] = "exploratory " + report["eval_slice"]
    simulation, job_rows = _run_simulation(
        proba, y_test, routing, corpus.test["id"].to_numpy(), policy, config
    )
    report["simulation"] = simulation
    record_columns = [
        "strategy", "load_per_hour", "seed", "comment_id", "trigger_reason", "job_index",
        "arrival_min", "start_min", "completion_min", "status", "high_risk", "harm",
        "priority_score", "wait_min", "completion_latency_min",
    ]
    pd.DataFrame(job_rows, columns=record_columns).to_csv(
        run_dir / "simulation_jobs.csv", index=False
    )
    manifest["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    (run_dir / "report.md").write_text(_report_md(report, manifest))
    return run_dir
