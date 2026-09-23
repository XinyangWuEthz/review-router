#!/usr/bin/env python
"""Collect a verified policy-v3 CI run into the dedicated step-6 record.

    PYTHONPATH=. python record/collect_priority.py reports/<run> --ci-metadata ci.json

The cumulative arm reuses thresholds chosen on the selection split and the
saved test predictions. It does not fit thresholds, score text or load a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from record.collect_runs import pick
from review_router.data import LABELS, file_sha256
from review_router.metrics import tier_metrics
from review_router.pipeline import _allow_false_negatives, _review_workload, _route, _rule_stats
from review_router.policy import Policy, load_policy
from review_router.thresholds import TIER_ORDER, TierThresholds

HERE = Path(__file__).resolve().parent
ARTIFACT_NAMES = (
    "manifest.json",
    "report.json",
    "policy.yaml",
    "config.yaml",
    "predictions.csv",
    "simulation_jobs.csv",
)


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _validate_run(
    manifest: dict[str, Any],
    report: dict[str, Any],
    policy: Policy,
    config: dict[str, Any],
    ci: dict[str, Any],
    hashes: dict[str, str],
) -> None:
    commit = manifest.get("git_commit")
    if not isinstance(commit, str) or not commit or ci.get("headSha") != commit:
        raise ValueError("CI headSha must match manifest.git_commit")
    if manifest.get("git_dirty") is not False:
        raise ValueError("step 6 requires a clean run")
    if report.get("synthetic") is not False or report.get("exploratory"):
        raise ValueError("step 6 requires a real, non-exploratory evaluation")
    if policy.version != 3 or report.get("policy_version") != 3:
        raise ValueError("step 6 requires policy version 3 in the report and snapshot")
    if report.get("evaluate_test") is not True or manifest.get("evaluate_test") is not True:
        raise ValueError("step 6 requires a completed test evaluation")
    if config.get("evaluate_test", True) is not True:
        raise ValueError("the saved config disables test evaluation")
    if report.get("run_id") != manifest.get("run_id") or not manifest.get("finished_utc"):
        raise ValueError("report and finished manifest must identify the same run")
    contract = report.get("decision_contract") or {}
    if contract.get("mode") != "human_confirmation" or contract.get("automatic_actions") != 0:
        raise ValueError("step 6 requires the human-confirmation decision contract")
    for name in ("policy", "config"):
        if manifest.get(f"{name}_sha256") != hashes[f"{name}.yaml"]:
            raise ValueError(f"{name}.yaml does not match its manifest hash")
    if ci.get("status") != "completed" or ci.get("conclusion") != "success":
        raise ValueError("CI run must be completed with conclusion success")
    if not isinstance(ci.get("url"), str) or not ci["url"]:
        raise ValueError("CI metadata must include the run URL")
    jobs = ci.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("CI metadata must include jobs")
    for required in ("evaluate", "negative-check"):
        matches = [job for job in jobs if isinstance(job, dict) and job.get("name") == required]
        if len(matches) != 1 or any(
            job.get("status") != "completed" or job.get("conclusion") != "success"
            for job in matches
        ):
            raise ValueError(f"CI job {required} must be completed with conclusion success")


def _test_cumulative(
    run_dir: Path, report: dict[str, Any], manifest: dict[str, Any], policy: Policy
) -> dict[str, Any]:
    saved = pd.read_csv(run_dir / "predictions.csv", dtype={"id": str})
    if len(saved) != manifest["split_label_counts"]["test_scored"]["rows"]:
        raise ValueError("predictions.csv row count does not match the scored test split")
    if saved["id"].isna().any() or saved["id"].duplicated().any():
        raise ValueError("predictions.csv must contain unique nonmissing IDs")
    y = saved[[f"y_{label}" for label in LABELS]].to_numpy()
    proba = saved[[f"p_{label}" for label in LABELS]].to_numpy(dtype=float)
    identity = saved["identity_term_present"].to_numpy(dtype=float)
    original = saved["final_tier"].to_numpy()
    if not np.isin(y, [0, 1]).all() or not np.isin(identity, [0, 1]).all():
        raise ValueError("saved labels and identity signals must be binary")
    y = y.astype(int)
    if not np.isfinite(proba).all() or ((proba < 0) | (proba > 1)).any():
        raise ValueError("saved predictions must contain finite probabilities in [0, 1]")
    if not np.isin(original, (*TIER_ORDER, "allow")).all():
        raise ValueError("saved final tiers must use the human-confirmation bands")
    raw = report["threshold_selection"]["alternatives"]["cumulative_precision"]["thresholds"]
    thresholds = TierThresholds(
        thresholds={tier: raw[tier] for tier in TIER_ORDER},
        labels=LABELS,
        subgroup=raw.get("subgroup", {}),
    )
    routing = _route(proba, y, identity, thresholds, policy)
    final = routing["final_tier"]
    min_pos = int(policy.gates.get("min_predicted_positives_for_precision", 30))
    high_risk_min_weight = float(report["high_risk_min_weight"])
    return {
        "thresholds": thresholds.as_json(),
        "tiers": tier_metrics(final, routing["correct"], (*TIER_ORDER, "allow"), min_pos),
        "review_workload": _review_workload(final, y, policy, high_risk_min_weight),
        "rules": _rule_stats(routing, policy),
        "allow_false_negatives": _allow_false_negatives(final, y),
        "same_admitted_rows": bool(
            np.array_equal(np.isin(final, TIER_ORDER), np.isin(original, TIER_ORDER))
        ),
        "n_changed_band": int((final != original).sum()),
        "note": "Descriptive rerouting of the previously inspected test set, not fresh "
        "validation. Uses the same saved probabilities and labels, with cumulative thresholds "
        "copied from threshold_selection.alternatives.cumulative_precision. Thresholds were "
        "selected on the selection split; no model or threshold is fitted on test rows. "
        "All admitted rows still require human confirmation.",
    }


def collect(run_dir: Path, ci_metadata: Path) -> dict[str, Any]:
    """Validate run provenance, then collect the report and a frozen cumulative replay."""
    run_dir = run_dir.resolve()
    manifest = _json_object(run_dir / "manifest.json")
    report = _json_object(run_dir / "report.json")
    ci = _json_object(ci_metadata)
    policy = load_policy(run_dir / "policy.yaml")
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a mapping")
    hashes = {name: file_sha256(run_dir / name) for name in ARTIFACT_NAMES}
    _validate_run(manifest, report, policy, config, ci, hashes)
    return {
        # The YAML snapshot can contain dates; preserve the historical
        # collector's ISO-string representation in this JSON-ready payload.
        "run": json.loads(json.dumps(pick(run_dir), default=str, allow_nan=False)),
        "threshold_selection": report["threshold_selection"],
        "agreement_by_confidence": report["agreement_by_confidence"],
        "ranking": report["ranking"],
        "ci": ci,
        "artifact_sha256": hashes,
        "test_cumulative": _test_cumulative(run_dir, report, manifest, policy),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--ci-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "priority_run.json")
    args = parser.parse_args()
    try:
        record = collect(args.run_dir, args.ci_metadata)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
