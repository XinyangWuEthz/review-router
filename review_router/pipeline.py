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
from review_router.metrics import per_label_metrics, tier_metrics
from review_router.model import ModelConfig, TfidfLogitModel
from review_router.policy import DEFAULT_POLICY_PATH, Policy, load_policy
from review_router.signals import identity_term_present
from review_router.simulate import (
    STRATEGIES,
    SimConfig,
    SimResult,
    draw_scenario,
    simulate,
)
from review_router.thresholds import (
    TierThresholds,
    apply_rules,
    model_tier,
    select_thresholds,
)

__all__ = ["RunConfig", "load_config", "run"]

HUMAN = "human_review"
AUTO = "auto_action"
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
    label: str


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
    strategy = str(primary.get("strategy", "severity"))
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown primary strategy {strategy!r}; choose from {STRATEGIES}")
    loads = tuple(float(x) for x in sim_raw.get("loads_per_hour", (60, 108, 180)))
    seeds = tuple(int(x) for x in sim_raw.get("seeds", (1, 2, 3, 4, 5)))
    primary_load = float(primary.get("load_per_hour", 108))
    if not loads or any(not np.isfinite(x) or x <= 0 for x in loads):
        raise ValueError("loads_per_hour must contain positive finite arrival rates")
    if primary_load not in loads:
        raise ValueError("primary load_per_hour must appear in loads_per_hour")
    if not seeds or any(x < 0 for x in seeds):
        raise ValueError("simulation seeds must contain nonnegative integers")
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
        label=str(raw.get("label", path.stem)),
    )


# --------------------------------------------------------------------------- provenance


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=10,
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
    tiers, triggers = model_tier(proba, thresholds)
    rule_action, rule_ids = apply_rules(proba, LABELS, identity, policy)
    final = tiers.copy()
    forced = rule_action != ""
    final[forced] = rule_action[forced]

    any_true = y.any(axis=1)
    trigger_true = (triggers & (y == 1)).any(axis=1)
    # auto_action is judged on the label that fired; human_review on any label.
    correct = np.where(final == AUTO, trigger_true, any_true)
    correct_model = np.where(tiers == AUTO, trigger_true, any_true)

    return {
        "correct_model": correct_model,
        "model_tier": tiers,
        "final_tier": final,
        "triggers": triggers,
        "rule_action": rule_action,
        "rule_ids": rule_ids,
        "correct": correct,
        "any_true": any_true,
    }


def _rule_stats(routing: dict[str, Any], policy: Policy) -> dict[str, Any]:
    ids: list[str] = routing["rule_ids"]
    model: np.ndarray = routing["model_tier"]
    final: np.ndarray = routing["final_tier"]
    any_true: np.ndarray = routing["any_true"]
    forced = routing["rule_action"] != ""
    per_rule = {r.id: int(sum(1 for s in ids if r.id in s.split(";"))) for r in policy.rules}
    moved_to_queue = forced & (model == ALLOW) & (final == HUMAN)
    downgraded = forced & (model == AUTO) & (final == HUMAN)
    return {
        "matched_per_rule": per_rule,
        "n_rows_with_any_rule": int(forced.sum()),
        "n_added_to_queue_from_allow": int(moved_to_queue.sum()),
        "n_added_to_queue_from_allow_true_positive": int((moved_to_queue & any_true).sum()),
        "n_downgraded_from_auto_action": int(downgraded.sum()),
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


# --------------------------------------------------------------------------- simulation


def _run_simulation(
    proba: np.ndarray, y: np.ndarray, final: np.ndarray, policy: Policy, config: RunConfig
) -> dict[str, Any]:
    queued = np.flatnonzero(final == HUMAN)
    weights = np.array([policy.severity_weights.get(label, 0.0) for label in LABELS])
    if len(queued) == 0:
        return {"note": "nothing was routed to human_review; simulation skipped", "table": []}

    p = proba[queued]
    truth = y[queued]
    harm = (truth * weights).max(axis=1)  # highest weight among the true labels
    high_risk = harm >= config.high_risk_min_weight
    priorities = {
        "fifo": np.zeros(len(queued)),
        "prob": p.max(axis=1),
        "severity": (p * weights).max(axis=1),
    }

    rows: list[dict[str, Any]] = []
    for load in config.loads_per_hour:
        for seed in config.sim_seeds:
            scenario = draw_scenario(seed, load, len(queued), config.sim)
            for strategy in STRATEGIES:
                result = simulate(
                    scenario, priorities[strategy], harm, high_risk, config.sim, strategy
                )
                rows.append({"load_per_hour": load, "seed": seed, **result.as_json()})

    summary = _summarise(rows)
    primary_key = (config.primary_load_per_hour, config.primary_strategy)
    fifo_key = (config.primary_load_per_hour, "fifo")
    router = summary.get(_key(*primary_key), {})
    fifo = summary.get(_key(*fifo_key), {})
    return {
        "assumptions": {
            **asdict(config.sim),
            "capacity_per_hour": config.sim.capacity_per_hour,
            "arrivals": "Poisson; jobs sampled with replacement from the human_review set",
            "handle_time": "deterministic",
            "harm_proxy": "max severity weight over TRUE labels (0 if clean); not real-world harm",
            "high_risk": f"harm proxy >= {config.high_risk_min_weight}",
            "loads_per_hour": list(config.loads_per_hour),
            "seeds": list(config.sim_seeds),
            "queued_jobs": int(len(queued)),
            "queued_high_risk": int(high_risk.sum()),
        },
        "primary": {
            "load_per_hour": config.primary_load_per_hour,
            "strategy": config.primary_strategy,
        },
        "harm_per_reviewer_hour": {
            "router": router.get("harm_per_reviewer_hour", {}).get("mean"),
            "fifo": fifo.get("harm_per_reviewer_hour", {}).get("mean"),
        },
        "queue_depth_p95": router.get("queue_depth_p95", {}).get("mean"),
        "reviewer_utilization": router.get("reviewer_utilization", {}).get("mean"),
        "summary": summary,
        "table": rows,
    }


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
    lines = [
        f"# Run {manifest['run_id']} ({manifest['label']})",
        "",
        f"commit `{manifest['git_commit']}`" + (" (dirty)" if manifest["git_dirty"] else ""),
        f", seed {manifest['seed']}, scored test rows "
        f"{manifest['split_label_counts']['test_scored']['rows']}.",
        "",
        "## Per-label quality (scored test set)",
        "",
        "| label | positives | AP | ROC-AUC | human thr | P@human | R@human "
        "| auto thr | P@auto | R@auto |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if report["synthetic"]:
        lines[2:2] = ["SYNTHETIC pipeline check only; these are not Jigsaw results.", ""]
    for label, m in report["per_label"].items():
        h, a = m["at_human_review"], m["at_auto_action"]
        lines.append(
            f"| {label} | {m['positives']} | {_fmt(m['average_precision'])} | {_fmt(m['roc_auc'])} "
            f"| {_fmt(h['threshold'])} | {_fmt(h['precision'])} | {_fmt(h['recall'])} "
            f"| {_fmt(a['threshold'])} | {_fmt(a['precision'])} | {_fmt(a['recall'])} |"
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
        "Precision targets are selected per label. Combining labels, removing automatic "
        "actions from the human queue and applying rules can lower final-tier precision.",
        "",
        "| final tier | n | coverage | precision |",
        "|---|---:|---:|---:|",
    ]
    for tier, m in report["threshold_selection"]["tiers"].items():
        lines.append(
            f"| {tier} | {m['n_predicted_positive']} | {_fmt(m['coverage'])} "
            f"| {_fmt(m['precision'])} |"
        )
    lines += [
        "",
        f"On the test set, rules matched on {rules['n_rows_with_any_rule']} rows; "
        f"{rules['n_added_to_queue_from_allow']} moved allow -> human_review "
        f"({rules['n_added_to_queue_from_allow_true_positive']} truly positive); "
        f"{rules['n_downgraded_from_auto_action']} moved auto_action -> human_review.",
        f"Hierarchy violation rate: {report['consistency']['hierarchy_violation_rate']:.4%}.",
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
            "Wait quantiles are minutes until service starts, for completed jobs only. "
            "High-risk unhandled includes jobs still in service at the horizon; "
            "backlog includes only jobs that have not started."
        )
        lines += [
            "",
            "| load/h | strategy | arrivals | handled | high-risk handled | high-risk unhandled "
            "| harm/reviewer-h | wait p50 | wait p90 | HR wait p50 | backlog end | util |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
                _fmt(e["backlog_end"], 1),
                _fmt(e["reviewer_utilization"], 2),
            ]
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- entry point


def run(config_path: Path) -> Path:
    import pandas as pd

    config_path = config_path.resolve()
    config = load_config(config_path)
    policy = load_policy(config.policy_path)
    corpus = load_corpus(config.data_dir, config.split_fractions, config.seed)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + config.label
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = _manifest(config, config_path, corpus, run_id)
    shutil.copyfile(config_path, run_dir / "config.yaml")
    shutil.copyfile(config.policy_path, run_dir / "policy.yaml")
    corpus.train[["id", "split"]].to_csv(run_dir / "splits.csv", index=False)

    def part(name: str) -> tuple[list[str], np.ndarray]:
        frame = corpus.train[corpus.train["split"] == name]
        return frame["comment_text"].astype(str).tolist(), frame[list(LABELS)].to_numpy(dtype=int)

    x_train, y_train = part("train")
    x_calib, y_calib = part("calib")
    x_thresh, y_thresh = part("thresh")

    model = TfidfLogitModel(config.model).fit(x_train, y_train, config.seed)
    model.calibrate(x_calib, y_calib)
    p_thresh = model.predict_proba(x_thresh)
    thresholds = select_thresholds(y_thresh, p_thresh, LABELS, policy)
    selection_routing = _route(
        p_thresh, y_thresh, identity_term_present(x_thresh), thresholds, policy
    )
    (run_dir / "thresholds.json").write_text(json.dumps(thresholds.as_json(), indent=2))
    with (run_dir / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)

    x_test = corpus.test["comment_text"].astype(str).tolist()
    y_test = corpus.test[list(LABELS)].to_numpy(dtype=int)
    proba = model.predict_proba(x_test)
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
    predictions.to_csv(run_dir / "predictions.csv", index=False)

    synthetic = (config.data_dir / "SYNTHETIC.txt").is_file()
    report: dict[str, Any] = {
        "run_id": run_id,
        "eval_slice": "synthetic scored test rows" if synthetic else "scored test rows",
        "synthetic": synthetic,
        "threshold_selection": {
            "note": "development data; per-label precision targets do not guarantee tier precision",
            "tiers": tier_metrics(
                selection_routing["final_tier"], selection_routing["correct"],
                (AUTO, HUMAN, ALLOW), min_pos,
            ),
            "tiers_model_only": tier_metrics(
                selection_routing["model_tier"], selection_routing["correct_model"],
                (AUTO, HUMAN, ALLOW), min_pos,
            ),
        },
        "per_label": per_label_metrics(y_test, proba, LABELS, thresholds.as_json()),
        "tiers": tier_metrics(
            routing["final_tier"], routing["correct"], (AUTO, HUMAN, ALLOW), min_pos
        ),
        "tiers_model_only": tier_metrics(
            routing["model_tier"], routing["correct_model"], (AUTO, HUMAN, ALLOW), min_pos
        ),
        "tier_correctness": {
            AUTO: "a label that fired the auto threshold is truly positive",
            HUMAN: "any label is truly positive",
        },
        "allow_false_negatives": _allow_false_negatives(routing["final_tier"], y_test),
        "rules": _rule_stats(routing, policy),
        "consistency": _consistency(proba),
        "simulation": _run_simulation(proba, y_test, routing["final_tier"], policy, config),
    }
    manifest["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    (run_dir / "report.md").write_text(_report_md(report, manifest))
    return run_dir
