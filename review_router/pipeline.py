"""Router experiment: data -> frozen scorer -> thresholds -> routing -> queue -> report.

    python scripts/run_pipeline.py --config configs/baseline.yaml

Baseline and development configs load a pinned scorer. --retrain is a separate
full-training check; configs without a frozen_model, such as smoke, train normally.

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
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import average_precision_score

from review_router import __version__
from review_router.agreement import agreement_report, oov_share
from review_router.data import LABELS, Corpus, label_counts, load_corpus
from review_router.frozen import load_frozen_model
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
    segment_masks,
    select_thresholds,
)

__all__ = [
    "RunConfig",
    "cross_fitted_selection",
    "evaluate_scores",
    "harm_proxy",
    "load_config",
    "ranking_diagnostics",
    "run",
]

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
    # False stops a run after the development sections: the scored test rows
    # are never scored. For iterating on orderings and threshold rules.
    evaluate_test: bool = True
    frozen_model: Path | None = None


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
    evaluate_test = raw.get("evaluate_test", True)
    if not isinstance(evaluate_test, bool):
        raise ValueError("evaluate_test must be true or false")
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
        evaluate_test=evaluate_test,
        frozen_model=resolve(str(raw["frozen_model"])) if raw.get("frozen_model") else None,
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
        "evaluate_test": config.evaluate_test,
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


def harm_proxy(y: np.ndarray, policy: Policy) -> np.ndarray:
    """Per-row maximum severity weight over the TRUE labels; 0 for a clean row.

    The weights are a policy input applied to annotations. This is the harm
    proxy the simulator scores and the high-risk definition is built on; it
    is not a measurement of real-world harm.
    """
    weights = np.array([policy.severity_weights.get(label, 0.0) for label in LABELS])
    truth = np.asarray(y, dtype=float)
    if truth.ndim != 2 or truth.shape[1] != len(LABELS):
        raise ValueError(f"y must have shape (n, {len(LABELS)}); got {truth.shape}")
    return np.asarray((truth * weights).max(axis=1))


def _high_risk_by_band(final: np.ndarray, high_risk: np.ndarray) -> dict[str, Any]:
    """Where the high-risk rows sit: count and share per band, recall into the queue.

    High risk is a property of the true labels, so this shows how many
    high-risk rows each band holds and how many the allow band lets through.
    It measures composition, not enforcement or review outcomes.
    """
    total = int(high_risk.sum())
    bands: dict[str, Any] = {}
    for band in (PRIORITY, HUMAN, ALLOW):
        mask = final == band
        n = int(mask.sum())
        n_high = int((mask & high_risk).sum())
        bands[band] = {
            "n": n,
            "n_high_risk": n_high,
            "high_risk_share": n_high / n if n else None,
        }
    in_queue = bands[PRIORITY]["n_high_risk"] + bands[HUMAN]["n_high_risk"]
    return {
        "high_risk_by_band": bands,
        "n_high_risk_total": total,
        "high_risk_recall_into_queue": in_queue / total if total else None,
        "high_risk_recall_into_priority": (
            bands[PRIORITY]["n_high_risk"] / total if total else None
        ),
    }


def _review_workload(
    final: np.ndarray, y: np.ndarray, policy: Policy, high_risk_min_weight: float
) -> dict[str, Any]:
    """Review demand before simulation; all flagged comments consume human capacity.

    Precision is any-label precision of the combined review pool. The
    high-risk composition uses the true-label harm proxy (harm_proxy) at or
    above high_risk_min_weight, the definition the simulator uses.
    """
    needs_review = np.isin(final, (PRIORITY, HUMAN))
    n = int(needs_review.sum())
    true_positive = int((needs_review & y.any(axis=1)).sum())
    high_risk = harm_proxy(y, policy) >= high_risk_min_weight
    return {
        "n_total": len(final),
        "n_requires_human_review": n,
        "n_priority_review": int((final == PRIORITY).sum()),
        "n_human_review": int((final == HUMAN).sum()),
        "review_fraction": n / len(final) if len(final) else 0.0,
        "n_truly_positive": true_positive,
        "precision": true_positive / n if n else None,
        "precision_ci95": list(wilson_interval(true_positive, n)) if n else None,
        **_high_risk_by_band(final, high_risk),
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


# --------------------------------------------------------------------------- agreement


def _oov_terciles(
    model: Any, texts: list[str], cuts: list[float] | None = None
) -> tuple[np.ndarray | None, list[float] | None]:
    """OOV share per text and the tercile cut points (given, or computed from these texts).

    None when the model has no word vectorizer (external scores, char_wb only).
    """
    shares = oov_share(model, texts) if model is not None else None
    if shares is None:
        return None, None
    if cuts is None:
        cuts = [float(q) for q in np.quantile(shares, [1 / 3, 2 / 3])]
    return shares, cuts


def _strata(
    identity: np.ndarray, oov: np.ndarray | None, cuts: list[float] | None
) -> dict[str, np.ndarray]:
    """Row strata for the agreement tables: identity term 0/1 and, when known, OOV tercile."""
    strata: dict[str, np.ndarray] = {"identity_term_present": identity.astype(int)}
    if oov is not None and cuts is not None:
        strata["oov_tercile"] = np.digitize(oov, cuts)
    return strata


def _segment_check(
    thresholds: TierThresholds,
    y: np.ndarray,
    p: np.ndarray,
    policy: Policy,
    high_risk_min_weight: float,
) -> dict[str, Any]:
    """Per label: the priority threshold and the declared segments at or above it.

    Read on the selection split with the same edges the segment rule uses.
    all_meet_floor is whether every such segment meets the floor with
    min_rows rows, whatever rule chose the threshold; None when disabled.
    """
    segments = policy.agreement_segments
    assert segments is not None  # Policy fills the default
    floor = policy.tier_precision_floors[PRIORITY]
    rule = policy.tier_selection_rules.get(PRIORITY, "cumulative_precision")
    out: dict[str, Any] = {}
    for j, label in enumerate(LABELS):
        threshold = thresholds.thresholds[PRIORITY][label]
        edges = segments.edges_for(policy.severity_weights.get(label, 0.0), high_risk_min_weight)
        rows: list[dict[str, Any]] = []
        if threshold is not None:
            last = len(edges) - 2
            for i, mask in enumerate(segment_masks(p[:, j], edges)):
                # The last segment is closed at edges[-1], so a threshold equal to
                # edges[-1] keeps its single-value slice instead of skipping it.
                if edges[i + 1] < threshold or (edges[i + 1] <= threshold and i != last):
                    continue
                mask = mask & (p[:, j] >= threshold)
                n = int(mask.sum())
                positives = int(y[mask, j].sum())
                rows.append(
                    {
                        "lo": max(edges[i], threshold),
                        "hi": edges[i + 1],
                        "n": n,
                        "agreement": positives / n if n else None,
                        "meets_floor": n >= segments.min_rows and positives / n >= floor
                        if n
                        else False,
                    }
                )
        out[label] = {
            "tier": PRIORITY,
            "threshold": threshold,
            "rule": rule,
            "edges": list(edges),
            "segments": rows,
            "all_meet_floor": all(r["meets_floor"] for r in rows)
            if threshold is not None
            else None,
        }
    return out


def _identity_share(final: np.ndarray, identity: np.ndarray) -> dict[str, Any]:
    """Rows with an identity term per review band: a composition count, not an error rate."""
    has_term = identity.astype(bool)
    out: dict[str, Any] = {}
    for band in (PRIORITY, HUMAN):
        mask = final == band
        n = int(mask.sum())
        with_term = int((mask & has_term).sum())
        out[band] = {
            "n": n,
            "n_with_identity_term": with_term,
            "share": with_term / n if n else None,
        }
    return out


def _cumulative_alternative(
    y: np.ndarray,
    p: np.ndarray,
    identity: np.ndarray,
    policy: Policy,
    high_risk_min_weight: float,
    min_pos: int,
) -> dict[str, Any]:
    """The same split selected with both tiers on the cumulative rule, for comparison.

    Same floors, same rules and subgroup thresholds; only the population
    selection rule differs. Development-split numbers, in-sample.
    """
    alt_policy = replace(
        policy,
        tier_selection_rules={
            tier: "cumulative_precision" for tier in policy.tier_precision_floors
        },
    )
    alt = select_thresholds(
        y,
        p,
        LABELS,
        alt_policy,
        {"identity_term_present": identity},
        high_risk_min_weight=high_risk_min_weight,
    )
    routing = _route(p, y, identity, alt, alt_policy)
    final = routing["final_tier"]
    return {
        "thresholds": alt.as_json(),
        "tiers": tier_metrics(final, routing["correct"], (PRIORITY, HUMAN, ALLOW), min_pos),
        **_high_risk_by_band(final, harm_proxy(y, policy) >= high_risk_min_weight),
        "identity_share": _identity_share(final, identity),
        "note": "both tiers selected by cumulative precision at the configured floors on the "
        "same selection split; in-sample development numbers for comparison only",
    }


def _selection_notes(thresholds: TierThresholds, policy: Policy) -> list[str]:
    notes: list[str] = []
    if all(t is None for t in thresholds.thresholds[PRIORITY].values()):
        rule = policy.tier_selection_rules.get(PRIORITY, "cumulative_precision")
        rule_name = "segment" if rule == "segment_agreement" else rule
        notes.append(
            f"no label qualifies for priority_review under the {rule_name} rule; the priority "
            "band contains rule promotions only"
        )
    return notes


def _split_agreement_gap(thresh: dict[str, Any], calib: dict[str, Any]) -> dict[str, Any]:
    """Per-segment agreement on the selection split minus the calibration split."""

    def gaps(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for ra, rb in zip(a, b, strict=True):
            gap = (
                ra["agreement"] - rb["agreement"]
                if ra["agreement"] is not None and rb["agreement"] is not None
                else None
            )
            rows.append(
                {
                    "lo": ra["lo"],
                    "hi": ra["hi"],
                    "threshold_selection": ra["agreement"],
                    "calibration": rb["agreement"],
                    "gap": gap,
                }
            )
        return rows

    return {
        "pooled_review": gaps(thresh["pooled_review"], calib["pooled_review"]),
        "per_label": {
            label: gaps(thresh["per_label"][label]["table"], calib["per_label"][label]["table"])
            for label in LABELS
        },
        "note": "threshold-selection minus calibration agreement per segment. Calibration "
        "rows fitted the Platt calibrator, so this is a descriptive split comparison, not "
        "an independent estimate or bound on threshold-selection optimism. It does not "
        "measure train-to-test shift",
    }


def _agreement_by_confidence(
    corpus: Corpus,
    p_calib: np.ndarray | None,
    y_thresh: np.ndarray,
    p_thresh: np.ndarray,
    selection_routing: dict[str, Any],
    strata_thresh: dict[str, np.ndarray],
    oov_cuts: list[float] | None,
    model: Any,
    policy: Policy,
    high_risk_min_weight: float,
) -> dict[str, Any]:
    """Agreement tables on the development splits; the test block is added later if scored."""
    thresh_report = agreement_report(
        y_thresh,
        p_thresh,
        LABELS,
        policy,
        high_risk_min_weight,
        final_tier=selection_routing["final_tier"],
        rule_action=selection_routing["rule_action"],
        strata=strata_thresh,
    )
    calib_report: dict[str, Any] | None = None
    if p_calib is not None:
        calib_frame = corpus.train[corpus.train["split"] == "calib"]
        x_calib = calib_frame["comment_text"].astype(str).tolist()
        y_calib = calib_frame[list(LABELS)].to_numpy(dtype=int)
        oov_calib, _ = _oov_terciles(model, x_calib, oov_cuts)
        calib_report = agreement_report(
            y_calib,
            p_calib,
            LABELS,
            policy,
            high_risk_min_weight,
            strata=_strata(identity_term_present(x_calib), oov_calib, oov_cuts),
        )
    return {
        "calibration": calib_report,
        "calibration_note": (
            None
            if calib_report is not None
            else "not available: external scores or no calibration split scored"
        ),
        "threshold_selection": thresh_report,
        "split_agreement_gap": (
            _split_agreement_gap(thresh_report, calib_report) if calib_report is not None else None
        ),
        "strata_definition": {
            "identity_term_present": "whole-word match of review_router.signals.IDENTITY_TERMS",
            "oov_tercile": (
                {
                    "cut_points": oov_cuts,
                    "degenerate": oov_cuts[0] == oov_cuts[1],
                    "rows_per_stratum_threshold_selection": {
                        str(k): int((strata_thresh["oov_tercile"] == k).sum()) for k in range(3)
                    },
                    "definition": "share of word-analyzer tokens absent from the training "
                    "vocabulary; terciles cut at the 1/3 and 2/3 quantiles of the "
                    "threshold-selection split. Rows tied with a cut point go to the upper "
                    "stratum, so equal cut points (degenerate) collapse the terciles into "
                    "one stratum",
                }
                if oov_cuts is not None
                else None
            ),
        },
        "note": "calib is in-sample for the Platt fit and out-of-sample for thresholds; thresh "
        "is the reverse. Agreement = share of rows in the segment with the label positive; "
        "development splits only, not test results",
    }


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
    harm = harm_proxy(y[queued], policy)
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


def ranking_diagnostics(
    inputs: dict[str, Any],
    high_risk_min_weight: float,
    fractions: tuple[float, ...] = (0.5, 0.67, 1.0),
) -> dict[str, Any]:
    """How much true harm each ordering score reaches in the top part of the review pool.

    For every priority key in `inputs` (see queue_inputs) and for the reference
    key oracle_true_harm (the true-label harm proxy itself, which uses true
    labels and is not shippable), the pool is sorted by the score, descending
    and stable, and the top k = ceil(f * n) rows are read at each fraction f:
    harm_share_at_k is their share of the pool's total harm proxy and
    high_risk_share_at_k the share of the pool's high-risk rows they contain.
    ap_vs_high_risk is the average precision of the score against the
    high-risk indicator (None when the indicator is constant). fifo scores are
    constant, so its order is the pool's row order, not a simulated arrival
    stream. These are static ranking diagnostics over the whole pool; they do
    not simulate capacity or time, and say nothing about rows outside the pool.
    """
    harm = np.asarray(inputs["harm"], dtype=float)
    n = int(len(harm))
    high_risk = harm >= high_risk_min_weight
    n_high = int(high_risk.sum())
    out: dict[str, Any] = {
        "n_pool": n,
        "n_high_risk": n_high,
        "fractions": [float(f) for f in fractions],
        "by_key": {},
        "note": "oracle_true_harm ranks by the true-label harm proxy: it uses true labels and "
        "is not shippable; it is the ceiling for these shares. Shares are top-k over the whole "
        f"review pool (both bands), k = ceil(fraction * n_pool); high risk = harm proxy >= "
        f"{high_risk_min_weight:g}; fifo scores are constant, so its order is pool row order.",
    }
    if n == 0:
        return out
    total_harm = float(harm.sum())
    scores: dict[str, Any] = {**inputs["priorities"], "oracle_true_harm": harm}
    for key, score in scores.items():
        values = np.asarray(score, dtype=float)
        order = np.argsort(-values, kind="stable")
        at_fraction: list[dict[str, Any]] = []
        for f in fractions:
            k = min(n, math.ceil(f * n))
            top = order[:k]
            at_fraction.append(
                {
                    "fraction": float(f),
                    "k": int(k),
                    "harm_share_at_k": (
                        float(harm[top].sum() / total_harm) if total_harm > 0 else None
                    ),
                    "high_risk_share_at_k": (
                        float(high_risk[top].sum() / n_high) if n_high else None
                    ),
                }
            )
        out["by_key"][key] = {
            "ap_vs_high_risk": (
                float(average_precision_score(high_risk, values)) if 0 < n_high < n else None
            ),
            "at_fraction": at_fraction,
        }
    return out


def _cross_fit_folds(n: int, k: int, seed: int) -> np.ndarray:
    """Fold id per row: seeded permutation, position modulo k; every row is in one fold."""
    if k < 2:
        raise ValueError("cross-fitting needs at least two folds")
    permutation = np.random.default_rng(seed).permutation(n)
    folds = np.empty(n, dtype=int)
    folds[permutation] = np.arange(n) % k
    return folds


def cross_fitted_selection(
    y: np.ndarray,
    proba: np.ndarray,
    identity: np.ndarray,
    policy: Policy,
    *,
    high_risk_min_weight: float,
    k: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Band quality on the selection split with thresholds refit away from each row.

    Rows are split into k deterministic folds. For each fold, thresholds
    (population and subgroup, with the same signals) are selected on the other
    k-1 folds and the held-out fold is routed with them, rules included. The
    held-out tiers are concatenated back into row order and summarised like
    the in-sample selection tables, so band membership is out-of-sample
    relative to the thresholds that define it. This estimates band quality
    away from threshold fitting; it is not a formal bound on selection
    optimism. Every fold comes from
    the same train.csv distribution: it says nothing about distribution shift
    or about the scored test rows.
    """
    n = len(proba)
    folds = _cross_fit_folds(n, k, seed)
    final = np.full(n, ALLOW, dtype=object)
    correct = np.zeros(n, dtype=bool)
    per_fold: list[dict[str, Any]] = []
    for fold in range(k):
        held = folds == fold
        fit = ~held
        thresholds = select_thresholds(
            y[fit],
            proba[fit],
            LABELS,
            policy,
            {"identity_term_present": identity[fit]},
            high_risk_min_weight=high_risk_min_weight,
        )
        routing = _route(proba[held], y[held], identity[held], thresholds, policy)
        final[held] = routing["final_tier"]
        correct[held] = routing["correct"]
        per_fold.append(thresholds.as_json())
    min_pos = int(policy.gates.get("min_predicted_positives_for_precision", 30))
    return {
        "k": k,
        "seed": seed,
        "fold_sizes": [int((folds == fold).sum()) for fold in range(k)],
        "tiers": tier_metrics(final, correct, (PRIORITY, HUMAN, ALLOW), min_pos),
        "review_workload": _review_workload(final, y, policy, high_risk_min_weight),
        "thresholds_per_fold": per_fold,
        "note": "thresholds refit on k-1 folds; band membership is out-of-sample relative to "
        "the thresholds that define it; this estimates band quality in the same train.csv "
        "distribution, not distribution shift, and is not a formal bound on selection optimism",
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
            "primary_order": _ORDER_DESCRIPTIONS[config.primary_strategy],
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
            "note": "queue-clearing gates read this load; the router is the configured "
            "primary ordering (strategy), compared against fifo at the same load for "
            "time-to-action",
        },
        "thesis": {
            "load_per_hour": config.thesis_load_per_hour,
            "strategy": config.primary_strategy,
            "note": "harm handled per reviewer-hour only differs between orderings when "
            "the queue does not clear, so the FIFO comparison and the paired comparison "
            "of the primary ordering against every alternative read this load",
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
        "paired": _paired(rows, config.primary_strategy),
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
        )
        + "; pooled over seeds; populations may differ by strategy; "
        "read with completion and unfinished counts",
        "selected": compare(metric, pct),
        "supplementary_high_risk": {
            "note": "reported alongside, never a substitute for the selected metric",
            "wait_p90": compare("high_risk_wait", 90) | {"percentile": 90},
        },
    }


_HIGHER_IS_BETTER: dict[str, bool] = {
    "high_risk_handled": True,
    "harm_per_reviewer_hour": True,
    "high_risk_unhandled": False,
    "high_risk_wait_p90": False,
    "wait_p90": False,
}
# How each ordering picks the next waiting job; see queue_inputs and simulate.
_ORDER_DESCRIPTIONS: dict[str, str] = {
    "severity": "highest max predicted probability times severity weight first; "
    "ties use arrival order",
    "prob": "highest max predicted probability first; ties use arrival order",
    "priority": "priority_review band first, then human_review; within each band, highest "
    "max predicted probability times severity weight first; ties use arrival order",
    "fifo": "arrival order",
}


def _ordering_pairs(primary: str) -> list[tuple[str, str]]:
    """Strategy pairs for the paired comparison, the primary ordering first.

    The primary ordering is paired with every alternative, then the alternatives
    with each other; within a non-primary pair the strategy that appears later
    in STRATEGIES comes first. With primary 'priority' this reproduces the six
    legacy keys (priority_vs_fifo ... severity_vs_prob).
    """
    if primary not in STRATEGIES:
        raise ValueError(f"unknown primary strategy {primary!r}; choose from {STRATEGIES}")
    others = [s for s in STRATEGIES if s != primary]
    pairs = [(primary, s) for s in others]
    pairs += [(later, earlier) for i, later in enumerate(others) for earlier in others[:i]]
    return pairs


def _paired(rows: list[dict[str, Any]], primary: str) -> dict[str, dict[str, Any]]:
    """Per-seed paired differences between orderings, keyed first_vs_second@load.

    Every ordering replays the same arrivals for a given (load, seed), so the
    honest comparison is paired: how many seeds favour the first ordering, not
    whether the seed means overlap. The primary ordering is the first member of
    its pairs, so the gate reads f"{primary}_vs_{alternative}@{load}".
    """
    by = {(r["load_per_hour"], r["seed"], r["strategy"]): r for r in rows}
    loads = sorted({r["load_per_hour"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    out: dict[str, dict[str, Any]] = {}
    for load in loads:
        for first, second in _ordering_pairs(primary):
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


def _tier_row(m: dict[str, Any], digits: int = 3) -> str:
    return f"{m['n_predicted_positive']} | {_fmt(m['coverage'])} | {_fmt(m['precision'], digits)}"


def _md_high_risk_by_band(
    splits: list[tuple[str, dict[str, Any]]], high_risk_min_weight: float
) -> list[str]:
    """Count and share of high-risk rows per band, plus recall into the queue, per split."""
    lines = [
        "",
        "## High-risk composition by band",
        "",
        f"High risk = true-label harm proxy >= {high_risk_min_weight} (the simulator's "
        "definition). Share = high-risk rows / rows in the band. Recall into queue = high-risk "
        "rows in either review band / all high-risk rows in the split. Composition of what was "
        "routed, not a review outcome.",
        "",
        "| split | band | n | high-risk | share |",
        "|---|---|---:|---:|---:|",
    ]
    recall: list[str] = []
    for name, workload in splits:
        for band, m in workload["high_risk_by_band"].items():
            lines.append(
                f"| {name} | {band} | {m['n']} | {m['n_high_risk']} "
                f"| {_fmt(m['high_risk_share'])} |"
            )
        recall.append(
            f"{name}: {_fmt(workload['high_risk_recall_into_queue'])} of "
            f"{workload['n_high_risk_total']} high-risk rows enter the queue, "
            f"{_fmt(workload['high_risk_recall_into_priority'])} into priority_review"
        )
    return [*lines, "", "Recall into the queue: " + "; ".join(recall) + "."]


def _md_ranking(pools: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """High-risk share captured in the top part of each pool, per ordering key, plus AP."""
    fractions: list[float] = pools[0][1]["fractions"] if pools else []
    lines = [
        "",
        "## Ranking diagnostics (static, whole review pool)",
        "",
        "Pool rows are sorted by each ordering score; the top ceil(f * n) rows are read at each "
        "fraction f. Cells are the share of the pool's high-risk rows in that top part; AP is the "
        "average precision of the score against the high-risk indicator. oracle_true_harm uses "
        "true labels and is not shippable; it is the ceiling. fifo scores are constant, so its "
        "order is pool row order, not a simulated arrival stream. No capacity or time is "
        "simulated here.",
        "",
        "| pool | ordering | AP vs high-risk | "
        + " | ".join(f"HR share @{100 * f:.0f}%" for f in fractions)
        + " |",
        "|---|---|---:|" + "---:|" * len(fractions),
    ]
    for name, ranking in pools:
        if not ranking["by_key"]:
            lines.append(f"| {name} | (empty pool) | n/a |" + " n/a |" * len(fractions))
            continue
        for key, entry in ranking["by_key"].items():
            shares = " | ".join(_fmt(a["high_risk_share_at_k"]) for a in entry["at_fraction"])
            lines.append(
                f"| {name} (n={ranking['n_pool']}, high-risk {ranking['n_high_risk']}) | {key} "
                f"| {_fmt(entry['ap_vs_high_risk'])} | {shares} |"
            )
    return lines


def _md_test_quality(report: dict[str, Any]) -> list[str]:
    selection = report["threshold_selection"]["per_label"]
    lines = [
        "## Per-label quality (scored test set)",
        "",
        "| label | positives | AP (selection) | AP | ROC-AUC | human thr | P@human | R@human "
        "| priority thr | P@priority | R@priority |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
        lines.append(f"| {tier} | {_tier_row(m)} | {ci_s} | {m['precision_note'] or ''} |")
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
        lines.append(f"| {tier} | {_tier_row(m)} | {ci_s} |")
    return [*lines, ""]


def _md_selection(report: dict[str, Any]) -> list[str]:
    ts = report["threshold_selection"]
    cross = ts["cross_fitted"]
    rules = ts["rules"]
    lines = [
        "## Threshold-selection split (development data, not test results)",
        "",
        "Precision targets are selected per label. Combining labels, separating review "
        "priorities and applying rules can change final-tier precision. Both tiers join the queue. "
        f"In-sample: thresholds selected and read on the same rows. Cross-fitted ({cross['k']} "
        "folds): thresholds refit on the other folds, each row routed by thresholds it did not "
        "select. This estimates band quality with held-out threshold fitting on the same "
        "distribution; it is not a formal bound or a test of distribution shift.",
        "",
        "| final tier | n | coverage | precision | n (cross-fitted) | coverage (cross-fitted) "
        "| precision (cross-fitted) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tier, m in ts["tiers"].items():
        lines.append(f"| {tier} | {_tier_row(m, 4)} | {_tier_row(cross['tiers'][tier], 4)} |")
    lines += [
        "",
        f"On the selection split, rules matched on {rules['n_rows_with_any_rule']} rows; "
        f"{rules['n_added_to_queue_from_allow']} moved allow -> review "
        f"({rules['n_added_to_queue_from_allow_true_positive']} truly positive); "
        f"{rules['n_promoted_to_priority_review']} were promoted to priority_review. "
        f"Truly positive rows allowed, per label: {ts['allow_false_negatives']}.",
    ]
    return lines


def _md_subgroups(report: dict[str, Any]) -> list[str]:
    lines = [
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
    return lines


def _md_identity(report: dict[str, Any]) -> list[str]:
    lines = [
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
    return lines


def _md_simulation(report: dict[str, Any]) -> list[str]:
    lines = [
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
            f"Primary ordering ({sim['primary']['strategy']}): {a['primary_order']}. "
            f"Band-first (priority) order: {a['priority_order']}."
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
    return lines


def _md_prevalence(report: dict[str, Any]) -> list[str]:
    ps = report.get("prevalence_shift")
    if not ps:
        return []
    lines = [
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
    return lines


def _md_agreement(report: dict[str, Any]) -> list[str]:
    """Pooled agreement per segment on the selection split, the segment check and the arms."""
    abc = report["agreement_by_confidence"]
    ts = report["threshold_selection"]
    thresh = abc["threshold_selection"]
    lines = [
        "",
        "## Agreement by confidence (development splits)",
        "",
        "Score segments on the threshold-selection split; agreement = share of rows in the "
        "segment with any positive label, against the pooled max probability. Coverage is the "
        "segment's share of the split. Development data, not test results; the calibration "
        "split table and the descriptive per-segment split agreement gap are in report.json.",
        "",
        "| segment | n | coverage | agreement | 95% CI | recall share |",
        "|---|---:|---:|---:|---|---:|",
    ]
    top = thresh["edges"][-1]
    for row in thresh["pooled_review"]:
        close = "]" if row["hi"] >= top else ")"  # the last segment is closed at the top edge
        lo = (
            "< " + f"{row['hi']:g}"
            if row["lo"] is None
            else f"[{row['lo']:g}, {row['hi']:g}{close}"
        )
        ci = row["agreement_ci95"]
        ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
        lines.append(
            f"| {lo} | {row['n']} | {_fmt(row['coverage'], 4)} | {_fmt(row['agreement'])} "
            f"| {ci_s} | {_fmt(row['recall_share'])} |"
        )
    lines += [
        "",
        f"Selection rules: {ts['selection_rules']}. Priority segment check per label (segments "
        "at or above the chosen threshold, n and agreement on the selection split):",
        "",
        "| label | threshold | rule | segments (lo: n, agreement) | all meet floor |",
        "|---|---:|---|---|---|",
    ]
    for label, check in ts["segment_check"].items():
        segs = "; ".join(
            f"{seg['lo']:g}: {seg['n']}, {_fmt(seg['agreement'])}" for seg in check["segments"]
        )
        lines.append(
            f"| {label} | {_fmt(check['threshold'], 4)} | {check['rule']} | {segs or 'n/a'} "
            f"| {check['all_meet_floor']} |"
        )
    for note in ts["selection_notes"]:
        lines.append(f"\nNote: {note}.")
    alt = ts["alternatives"]["cumulative_precision"]
    lines += [
        "",
        "Arms on the selection split (in-sample): the shipped rules versus both tiers on "
        "cumulative precision at the same floors.",
        "",
        "| arm | band | n | coverage | precision | high-risk rows | identity-term share |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    arms = [
        ("shipped", ts["tiers"], ts["review_workload"]["high_risk_by_band"], ts["identity_share"]),
        ("cumulative_precision", alt["tiers"], alt["high_risk_by_band"], alt["identity_share"]),
    ]
    for name, tiers, high_risk, identity in arms:
        for band in (PRIORITY, HUMAN):
            lines.append(
                f"| {name} | {band} | {_tier_row(tiers[band], 4)} "
                f"| {high_risk[band]['n_high_risk']} | {_fmt(identity[band]['share'])} |"
            )
    shipped = {label: check["threshold"] for label, check in ts["segment_check"].items()}
    lines.append(
        f"\nShipped priority thresholds: {shipped}. "
        f"Cumulative-rule thresholds: {alt['thresholds']}."
    )
    return [*lines, ""]


def _report_md(report: dict[str, Any], manifest: dict[str, Any]) -> str:
    """The report as tables. A development report has no test sections; they say so."""
    scored = "tiers" in report  # False for a development run (evaluate_test: false)
    ts = report["threshold_selection"]
    lines = [
        f"# Run {manifest['run_id']} ({manifest['label']})",
        "",
        f"commit `{manifest['git_commit']}`"
        + (" (dirty)" if manifest["git_dirty"] else "")
        + f", seed {manifest['seed']}, "
        + (
            f"scored test rows {manifest['split_label_counts']['test_scored']['rows']}."
            if scored
            else "test rows not scored (evaluate_test: false); development splits only."
        ),
        "",
        "Decision mode: human confirmation. Both priority_review and human_review require "
        "reviewers; the router emits no automatic enforcement actions. This offline evaluation "
        "does not observe completed human decisions.",
        "",
    ]
    if report["synthetic"]:
        lines[2:2] = ["SYNTHETIC pipeline check only; these are not Jigsaw results.", ""]
    if report.get("exploratory"):
        source = report["score_source"]
        lines[2:2] = [
            f"EXPLORATORY external-score evaluation for {source['model_name']}; "
            "these results do not certify the full baseline or deployment readiness.",
            "",
        ]
    if scored:
        workload = report["review_workload"]
        lines += [
            f"Review demand: {workload['n_requires_human_review']} / {workload['n_total']} "
            f"comments ({_fmt(workload['review_fraction'])}), including "
            f"{workload['n_priority_review']} priority "
            f"and {workload['n_human_review']} regular reviews. Pool precision is "
            f"{_fmt(workload['precision'])}, Wilson 95% CI {workload['precision_ci95']}.",
            "",
        ]
        lines += _md_test_quality(report)
    else:
        lines += [
            "Development run: test rows not scored. Thresholds, band composition and ranking "
            "diagnostics below are computed on the threshold-selection split only; there is "
            "no per-label test quality, routing-tier, identity, subgroup or simulation section.",
            "",
        ]
    lines += _md_selection(report)
    lines += _md_agreement(report)
    if scored:
        rules = report["rules"]
        lines += [
            f"On the test set, rules matched on {rules['n_rows_with_any_rule']} rows; "
            f"{rules['n_added_to_queue_from_allow']} moved allow -> review "
            f"({rules['n_added_to_queue_from_allow_true_positive']} truly positive); "
            f"{rules['n_promoted_to_priority_review']} were promoted to priority_review.",
            f"Hierarchy violation rate: {report['consistency']['hierarchy_violation_rate']:.4%}.",
        ]
    composition = [
        ("selection", ts["review_workload"]),
        ("selection (cross-fitted)", ts["cross_fitted"]["review_workload"]),
    ]
    pools = [("selection", ts["ranking"])]
    if scored:
        composition.append(("test", report["review_workload"]))
        pools.append(("test", report["ranking"]))
    lines += _md_high_risk_by_band(composition, report["high_risk_min_weight"])
    lines += _md_ranking(pools)
    if scored:
        lines += _md_subgroups(report)
        lines += _md_identity(report)
        lines += _md_simulation(report)
        lines += _md_prevalence(report)
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


def run(config_path: Path, *, retrain: bool = False) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    policy = load_policy(config.policy_path)
    if policy.decision_mode != "human_confirmation":
        raise ValueError("the current pipeline requires a human_confirmation policy")
    corpus = load_corpus(config.data_dir, config.split_fractions, config.seed)
    model_artifact = None
    source = None
    if config.frozen_model is not None and not retrain:
        model, model_artifact, source = load_frozen_model(
            config.frozen_model, corpus, config.model, config.seed, config.split_fractions
        )
    # Invalid frozen inputs fail before creating an evaluation directory.
    run_dir, manifest = _prepare_run(config_path, config, corpus, policy)
    manifest["execution_mode"] = "frozen_model" if source is not None else "train"
    if source is not None:
        manifest["model_source"] = source
        assert config.frozen_model is not None
        shutil.copyfile(config.frozen_model, run_dir / "frozen-model.json")
    elif retrain:
        manifest["explicit_retrain"] = True

    def part(name: str) -> tuple[list[str], np.ndarray]:
        frame = corpus.train[corpus.train["split"] == name]
        return frame["comment_text"].astype(str).tolist(), frame[list(LABELS)].to_numpy(dtype=int)

    x_calib, y_calib = part("calib")
    x_thresh, _ = part("thresh")
    if source is None:
        x_train, y_train = part("train")
        model = TfidfLogitModel(config.model).fit(x_train, y_train, config.seed)
        model.calibrate(x_calib, y_calib)
    # The calibration split is in-sample for the Platt fit; its agreement
    # table is reported next to the selection split's, never used to select.
    p_calib = model.predict_proba(x_calib)
    p_thresh = model.predict_proba(x_thresh)
    # A development run stops before the scored test rows are scored.
    p_test = (
        model.predict_proba(corpus.test["comment_text"].astype(str).tolist())
        if config.evaluate_test
        else None
    )
    return _evaluate_scores(
        config,
        corpus,
        p_thresh,
        p_test,
        policy,
        run_dir,
        manifest,
        model=model,
        p_calib=p_calib,
        model_artifact=model_artifact,
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
    if not config.evaluate_test:
        raise ValueError(
            "external scores need a config with evaluate_test true; a development config "
            "never scores the test rows"
        )
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
        "kind": "external_scores",
        "model_name": score_source["model_name"],
    }
    return _evaluate_scores(
        config, corpus, p_thresh, p_test, policy, run_dir, manifest, model=model
    )


def _evaluate_scores(
    config: RunConfig,
    corpus: Corpus,
    p_thresh: np.ndarray,
    proba: np.ndarray | None,
    policy: Policy,
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    model: Any,
    p_calib: np.ndarray | None = None,
    model_artifact: Path | None = None,
) -> Path:
    """Select thresholds on the selection split, then evaluate the scored test rows.

    Everything under report["threshold_selection"] is computed from the
    selection split alone. proba is None for a development run
    (evaluate_test: false): the run then writes thresholds.json, model.pkl,
    the development sections, report.json and report.md, and stops without
    predictions.csv, simulation_jobs.csv or any test section. p_calib, the
    calibration split's probabilities, is None on the external-score path;
    it only feeds the agreement-by-confidence tables.
    """
    import pandas as pd

    run_id = manifest["run_id"]
    frame = corpus.train[corpus.train["split"] == "thresh"]
    x_thresh = frame["comment_text"].astype(str).tolist()
    y_thresh = frame[list(LABELS)].to_numpy(dtype=int)
    identity_thresh = identity_term_present(x_thresh)
    high_risk_min_weight = config.high_risk_min_weight
    thresholds = select_thresholds(
        y_thresh,
        p_thresh,
        LABELS,
        policy,
        {"identity_term_present": identity_thresh},
        high_risk_min_weight=high_risk_min_weight,
    )
    selection_routing = _route(p_thresh, y_thresh, identity_thresh, thresholds, policy)
    (run_dir / "thresholds.json").write_text(json.dumps(thresholds.as_json(), indent=2))
    if model_artifact is not None:
        shutil.copyfile(model_artifact, run_dir / "model.pkl")
    elif model is not None:
        with (run_dir / "model.pkl").open("wb") as handle:
            pickle.dump(model, handle)

    min_pos = int(policy.gates.get("min_predicted_positives_for_precision", 30))
    synthetic = (config.data_dir / "SYNTHETIC.txt").is_file()
    selection_final = selection_routing["final_tier"]
    oov_thresh, oov_cuts = _oov_terciles(model, x_thresh)
    strata_thresh = _strata(identity_thresh, oov_thresh, oov_cuts)
    report: dict[str, Any] = {
        "run_id": run_id,
        "eval_slice": "synthetic scored test rows" if synthetic else "scored test rows",
        "synthetic": synthetic,
        "policy_version": policy.version,
        "evaluate_test": proba is not None,
        "high_risk_min_weight": high_risk_min_weight,
        "decision_contract": {
            "mode": "human_confirmation",
            "requires_human_confirmation": True,
            "automatic_actions": 0,
        },
        "threshold_selection": {
            "note": "development data; per-label precision targets do not guarantee tier precision",
            "per_label": per_label_metrics(y_thresh, p_thresh, LABELS, thresholds.thresholds),
            "review_workload": _review_workload(
                selection_final, y_thresh, policy, high_risk_min_weight
            ),
            "tiers": tier_metrics(
                selection_final, selection_routing["correct"], (PRIORITY, HUMAN, ALLOW), min_pos
            ),
            "tiers_model_only": tier_metrics(
                selection_routing["model_tier"],
                selection_routing["correct_model"],
                (PRIORITY, HUMAN, ALLOW),
                min_pos,
            ),
            "rules": _rule_stats(selection_routing, policy),
            "allow_false_negatives": _allow_false_negatives(selection_final, y_thresh),
            "ranking": ranking_diagnostics(
                queue_inputs(p_thresh, y_thresh, selection_final, policy, high_risk_min_weight),
                high_risk_min_weight,
            ),
            "cross_fitted": cross_fitted_selection(
                y_thresh,
                p_thresh,
                identity_thresh,
                policy,
                high_risk_min_weight=high_risk_min_weight,
                seed=config.seed,
            ),
            "selection_rules": dict(policy.tier_selection_rules),
            "segment_check": _segment_check(
                thresholds, y_thresh, p_thresh, policy, high_risk_min_weight
            ),
            "identity_share": _identity_share(selection_final, identity_thresh),
            "alternatives": {
                "cumulative_precision": _cumulative_alternative(
                    y_thresh, p_thresh, identity_thresh, policy, high_risk_min_weight, min_pos
                )
            },
            "selection_notes": _selection_notes(thresholds, policy),
        },
        "agreement_by_confidence": _agreement_by_confidence(
            corpus,
            p_calib,
            y_thresh,
            p_thresh,
            selection_routing,
            strata_thresh,
            oov_cuts,
            model,
            policy,
            high_risk_min_weight,
        ),
    }
    if proba is None:
        report["eval_slice"] = "development splits only; test rows not scored"
        _write_run_outputs(run_dir, manifest, report)
        return run_dir

    x_test = corpus.test["comment_text"].astype(str).tolist()
    y_test = corpus.test[list(LABELS)].to_numpy(dtype=int)
    identity = identity_term_present(x_test)
    routing = _route(proba, y_test, identity, thresholds, policy)

    weights = np.array([policy.severity_weights.get(label, 0.0) for label in LABELS])
    predictions = pd.DataFrame({"id": corpus.test["id"].to_numpy()})
    for j, label in enumerate(LABELS):
        predictions[f"y_{label}"] = y_test[:, j]
    for j, label in enumerate(LABELS):
        predictions[f"p_{label}"] = proba[:, j]
    predictions["p_max"] = proba.max(axis=1)
    predictions["identity_term_present"] = identity.astype(int)
    predictions["severity_score"] = (proba * weights).max(axis=1)
    predictions["model_tier"] = routing["model_tier"]
    predictions["rule_ids"] = routing["rule_ids"]
    predictions["final_tier"] = routing["final_tier"]
    predictions["requires_human_review"] = np.isin(routing["final_tier"], (PRIORITY, HUMAN))
    predictions.to_csv(run_dir / "predictions.csv", index=False)

    report.update(
        {
            "review_workload": _review_workload(
                routing["final_tier"], y_test, policy, high_risk_min_weight
            ),
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
            "ranking": ranking_diagnostics(
                queue_inputs(proba, y_test, routing["final_tier"], policy, high_risk_min_weight),
                high_risk_min_weight,
            ),
        }
    )
    oov_test, _ = _oov_terciles(model, x_test, oov_cuts)
    report["agreement_by_confidence"]["test"] = agreement_report(
        y_test,
        proba,
        LABELS,
        policy,
        high_risk_min_weight,
        final_tier=routing["final_tier"],
        rule_action=routing["rule_action"],
        strata=_strata(identity, oov_test, oov_cuts),
    )
    if manifest.get("exploratory"):
        report["exploratory"] = True
        report["score_source"] = manifest["score_source"]
        report["eval_slice"] = "exploratory " + report["eval_slice"]
    simulation, job_rows = _run_simulation(
        proba, y_test, routing, corpus.test["id"].to_numpy(), policy, config
    )
    report["simulation"] = simulation
    record_columns = [
        "strategy",
        "load_per_hour",
        "seed",
        "comment_id",
        "trigger_reason",
        "job_index",
        "arrival_min",
        "start_min",
        "completion_min",
        "status",
        "high_risk",
        "harm",
        "priority_score",
        "wait_min",
        "completion_latency_min",
    ]
    pd.DataFrame(job_rows, columns=record_columns).to_csv(
        run_dir / "simulation_jobs.csv", index=False
    )
    _write_run_outputs(run_dir, manifest, report)
    return run_dir


def _write_run_outputs(run_dir: Path, manifest: dict[str, Any], report: dict[str, Any]) -> None:
    manifest["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    (run_dir / "report.md").write_text(_report_md(report, manifest))
