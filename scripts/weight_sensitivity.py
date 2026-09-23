#!/usr/bin/env python
"""Bounded ordering-weight stress test using saved scores and admissions only.

No training, calibration, threshold selection or default-weight optimization.
The fixed evaluation weights and high-risk labels never change across arms.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from review_router.data import LABELS, file_sha256  # noqa: E402
from review_router.frozen import read_lock  # noqa: E402
from review_router.policy import load_policy  # noqa: E402
from review_router.simulate import (  # noqa: E402
    JobRecords,
    SimConfig,
    draw_scenario,
    priority_review_scores,
    simulate,
)


def vector(weights: dict[str, float]) -> np.ndarray:
    if set(weights) != set(LABELS):
        raise ValueError("weights must name every canonical label")
    result = np.array([weights[label] for label in LABELS], dtype=float)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("weights must be positive and finite")
    return result


def priorities(proba: np.ndarray, tiers: np.ndarray, plan: dict[str, Any]) -> dict[str, np.ndarray]:
    scores = {
        name: (proba * vector(weights)).max(axis=1)
        for name, weights in plan["ordering_weights"].items()
    }
    if not np.array_equal(scores.pop("flat"), proba.max(axis=1)):
        raise ValueError("flat weights must reproduce probability ordering")
    return {
        "fifo": np.zeros(len(proba)),
        "prob": proba.max(axis=1),
        "priority": priority_review_scores(tiers == "priority_review", scores["default"]),
        **scores,
    }


def _p90(values: np.ndarray) -> float | None:
    return float(np.percentile(values, 90)) if len(values) else None


def group_metrics(records: JobRecords, mask: np.ndarray, prefix: str) -> dict[str, Any]:
    not_started = mask & ~records.started
    unfinished = mask & ~records.completed
    return {
        f"{prefix}_arrived": int(mask.sum()),
        f"{prefix}_completed": int((mask & records.completed).sum()),
        f"{prefix}_not_started": int(not_started.sum()),
        f"{prefix}_unfinished": int(unfinished.sum()),
        f"{prefix}_wait_p90": _p90(records.wait_min[mask & records.started]),
        f"{prefix}_unfinished_age_p90": _p90(records.horizon_min - records.arrival_min[unfinished]),
    }


def replay(pred: pd.DataFrame, plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Only ordering scores vary; evaluation truth and scenarios are shared."""
    proba = pred[[f"p_{label}" for label in LABELS]].to_numpy(dtype=float)
    truth = pred[[f"y_{label}" for label in LABELS]].to_numpy(dtype=int)
    if (
        not len(pred)
        or pred["id"].isna().any()
        or pred["id"].astype(str).str.strip().eq("").any()
        or pred["id"].duplicated().any()
    ):
        raise ValueError("predictions must contain distinct, nonempty row IDs")
    if not np.isfinite(proba).all() or ((proba < 0) | (proba > 1)).any():
        raise ValueError("predictions must be finite probabilities in [0, 1]")
    if not pred[[f"y_{label}" for label in LABELS]].isin([0, 1]).all().all():
        raise ValueError("labels must be binary")
    if not pred["final_tier"].isin(["allow", "human_review", "priority_review"]).all():
        raise ValueError("unknown saved review band")
    admitted = pred["final_tier"].ne("allow").to_numpy()
    proba, truth = proba[admitted], truth[admitted]
    tiers = pred.loc[admitted, "final_tier"].to_numpy()
    if not len(proba):
        raise ValueError("empty saved review pool")
    harm = (truth * vector(plan["evaluation_weights"])).max(axis=1)
    high_risk = truth[:, [LABELS.index(label) for label in plan["high_risk_labels"]]].any(axis=1)
    config = SimConfig(**plan["simulation"])
    scores = priorities(proba, tiers, plan)
    rows = []
    for load in plan["loads_per_hour"]:
        for seed in plan["seeds"]:
            scenario = draw_scenario(seed, load, len(proba), config)
            for arm, score in scores.items():
                result, records = simulate(scenario, score, harm, high_risk, config, arm)
                row = {
                    "arm": arm,
                    "load_per_hour": load,
                    "seed": seed,
                    "completed": result.n_handled,
                    "baseline_harm_per_reviewer_hour": result.harm_per_reviewer_hour,
                    **group_metrics(records, records.high_risk, "high_risk"),
                    **group_metrics(records, ~records.high_risk, "other"),
                }
                for label in plan["high_risk_labels"]:
                    relevant = truth[scenario.job_index, LABELS.index(label)].astype(bool)
                    row[f"{label}_completed"] = int((relevant & records.completed).sum())
                rows.append(row)
    return rows


def stats(values: list[float]) -> dict[str, Any]:
    array = np.array(values, dtype=float)
    return {
        "n": len(values),
        "mean": float(array.mean()) if len(array) else None,
        "sd": float(array.std(ddof=1)) if len(array) > 1 else None,
        "min": float(array.min()) if len(array) else None,
        "max": float(array.max()) if len(array) else None,
    }


def summarize(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    summary: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    metrics = [key for key in rows[0] if key not in ("arm", "load_per_hour", "seed")]
    for load in sorted({row["load_per_hour"] for row in rows}):
        by_arm = {
            arm: {r["seed"]: r for r in rows if r["arm"] == arm and r["load_per_hour"] == load}
            for arm in dict.fromkeys(row["arm"] for row in rows)
        }
        for arm, seeds in by_arm.items():
            summary[f"{arm}@{load:g}"] = {
                metric: stats([r[metric] for r in seeds.values() if r[metric] is not None])
                for metric in metrics
            }
            for reference in ("default", "fifo"):
                fields = {}
                for metric in metrics:
                    differences = [
                        r[metric] - by_arm[reference][seed][metric]
                        for seed, r in seeds.items()
                        if r[metric] is not None and by_arm[reference][seed][metric] is not None
                    ]
                    lower_better = any(x in metric for x in ("wait", "unfinished", "not_started"))
                    direction = -1 if lower_better else 1
                    fields[metric] = {
                        **stats(differences),
                        "better": sum(direction * x > 1e-12 for x in differences),
                        "tied": sum(abs(x) <= 1e-12 for x in differences),
                        "worse": sum(direction * x < -1e-12 for x in differences),
                    }
                paired[f"{arm}_vs_{reference}@{load:g}"] = fields
    return summary, paired


def render(result: dict[str, Any]) -> str:
    lines = [
        "# Severity weight sensitivity",
        "",
        "Five weight vectors were declared before this sweep. Only scheduling scores change; "
        "saved predictions, admissions, reference harm weights and high-risk labels stay fixed. "
        "The default is retained regardless of which arm wins.",
        "",
        "Flat weights equal probability ordering and appear once as `prob`. Compression and "
        "expansion change weight contrast; `threat_lower` changes threat from 10 to 5. "
        "FIFO and band-first priority are fixed comparators.",
        "",
        "Each entry is a mean over 20 paired arrival seeds for an 8-hour shift with four "
        "reviewers and 2-minute service. Arrival loads are admitted review jobs per hour. "
        "High risk means any true severe_toxic, threat or identity_hate label; other means "
        "all remaining queued jobs, including false positives.",
        "",
        "| Load/h | Ordering | High-risk completed | High-risk wait p90, min | High-risk never "
        "started | Other completed | Other wait p90, min | Other never started | Reference "
        "harm / reviewer-h |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    fields = (
        "high_risk_completed",
        "high_risk_wait_p90",
        "high_risk_not_started",
        "other_completed",
        "other_wait_p90",
        "other_not_started",
        "baseline_harm_per_reviewer_hour",
    )
    for key, values in result["summary"].items():
        arm, load = key.split("@")
        cells = [
            f"{values[field]['mean']:.2f}" if values[field]["mean"] is not None else "n/a"
            for field in fields
        ]
        lines.append(f"| {load} | {arm} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Wait p90 is conditional on starting review. Never-started counts and unfinished-age "
        "p90 in the JSON accompany it to expose starvation. Total service capacity does not "
        "increase when the ordering changes; completions are redistributed between groups.",
        "",
        "The JSON contains per-seed outcomes and paired differences against the default and "
        "FIFO, including ranges and better/tied/worse counts. These measure simulation "
        "variability, not uncertainty about the corpus or its labels. This already-inspected "
        "benchmark is not independent evidence of real review worthiness or production benefit.",
        "",
    ]
    load = max(result["protocol"]["loads_per_hour"])
    lines += [
        f"## Paired high-risk completion differences at {load:g}/h",
        "",
        "Each difference is alternative minus default on the same arrival seed.",
        "",
        "| Alternative | Mean difference | Seed range | Better / tied / worse |",
        "|---|---:|---:|---:|",
    ]
    for arm in ("prob", "priority", "compressed", "expanded", "threat_lower"):
        entry = result["paired"][f"{arm}_vs_default@{load:g}"]["high_risk_completed"]
        lines.append(
            f"| {arm} | {entry['mean']:+.2f} | {entry['min']:+.0f} to {entry['max']:+.0f} "
            f"| {entry['better']} / {entry['tied']} / {entry['worse']} |"
        )
    default = result["summary"][f"default@{load:g}"]
    fifo = result["summary"][f"fifo@{load:g}"]
    gain = default["high_risk_completed"]["mean"] - fifo["high_risk_completed"]["mean"]
    cost = fifo["other_completed"]["mean"] - default["other_completed"]["mean"]
    lines += [
        "",
        "## Conclusion and boundary",
        "",
        f"At {load:g}/h the default completes {gain:.2f} more high-risk reviews and {cost:.2f} "
        "fewer other reviews per shift than FIFO. The priority changes who receives the fixed "
        "capacity. The reference utility is a policy-weighted label proxy, not measured harm.",
        "",
        "Retain the declared default. This sweep tests a small set of plausible perturbations; "
        "it does not establish an optimal or universally robust weight vector. In particular, "
        "an alternative performing better here is not a reason to retune against this inspected "
        "test benchmark. New traffic, different label priorities and variable handling times "
        "are outside this release's evidence.",
        "",
        f"Source evaluation: `{result['source']['run_id']}`, "
        f"commit `{result['source']['git_commit']}`.",
        "",
        "Reproduce with `python scripts/weight_sensitivity.py reports/ci-35863370097`. "
        "The source hashes and complete protocol are recorded in `severity-sensitivity.json`.",
        "",
    ]
    return "\n".join(lines)


def run(run_dir: Path, protocol_path: Path) -> dict[str, Any]:
    plan = yaml.safe_load(protocol_path.read_text())
    if plan["version"] != 1:
        raise ValueError("unsupported sensitivity protocol")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    report = json.loads((run_dir / "report.json").read_text())
    lock = read_lock(ROOT / "configs/frozen-baseline.json")
    if manifest.get("execution_mode") != "frozen_model" or report.get("synthetic"):
        raise ValueError("sensitivity source must be a real frozen-model evaluation")
    if manifest.get("git_dirty") or report.get("evaluate_test") is not True:
        raise ValueError("sensitivity source must be a clean scored evaluation")
    if file_sha256(run_dir / "model.pkl") != lock["files"]["model.pkl"]:
        raise ValueError("source model does not match the frozen model")
    for name in ("config", "policy"):
        if file_sha256(run_dir / f"{name}.yaml") != manifest[f"{name}_sha256"]:
            raise ValueError(f"source {name} snapshot hash mismatch")
    policy = load_policy(run_dir / "policy.yaml")
    if plan["evaluation_weights"] != policy.severity_weights:
        raise ValueError("reference evaluation weights must remain the source policy weights")
    if plan["ordering_weights"]["default"] != policy.severity_weights:
        raise ValueError("default ordering weights must remain the source policy weights")
    expected_risk = {
        label
        for label in LABELS
        if policy.severity_weights[label] >= manifest["config"]["high_risk_min_weight"]
    }
    if set(plan["high_risk_labels"]) != expected_risk:
        raise ValueError("high-risk labels must remain the source evaluation definition")
    if plan["simulation"] != manifest["config"]["sim"]:
        raise ValueError("reviewer capacity and service assumptions must remain fixed")
    pred = pd.read_csv(run_dir / "predictions.csv", dtype={"id": str}, float_precision="round_trip")
    if len(pred) != manifest["split_label_counts"]["test_scored"]["rows"]:
        raise ValueError("saved prediction row count differs from source manifest")
    for tier, values in report["tiers"].items():
        if pred["final_tier"].eq(tier).sum() != values["n_predicted_positive"]:
            raise ValueError("saved admissions differ from source report")
    rows = replay(pred, plan)
    summary, paired = summarize(rows)
    return {
        "protocol": plan,
        "protocol_sha256": file_sha256(protocol_path),
        "source": {
            "run_id": manifest["run_id"],
            "git_commit": manifest["git_commit"],
            "data_sha256": manifest["data_sha256"],
            "files_sha256": {
                name: file_sha256(run_dir / name)
                for name in (
                    "model.pkl",
                    "manifest.json",
                    "config.yaml",
                    "policy.yaml",
                    "thresholds.json",
                    "predictions.csv",
                )
            },
            "n_test": len(pred),
            "n_admitted": int(pred["final_tier"].ne("allow").sum()),
        },
        "summary": summary,
        "paired": paired,
        "per_seed": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/severity-sensitivity.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "record")
    args = parser.parse_args()
    result = run(args.run_dir, args.protocol)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "severity-sensitivity.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (args.out / "severity-sensitivity.md").write_text(render(result))
    print(args.out / "severity-sensitivity.md")


if __name__ == "__main__":
    main()
