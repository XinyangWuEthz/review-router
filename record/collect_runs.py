#!/usr/bin/env python
"""Collect the numbers the record quotes from finished runs into runs.json.

    python record/collect_runs.py --with-r103 reports/<a> --without-r103 reports/<b> --final reports/<c>

Each run directory is a pipeline output (report.json, manifest.json). Only the
fields the record pages quote are copied, so the record stays readable and the
provenance (run id, commit, dirty flag) travels with every number.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent


def pick(run_dir: Path) -> dict[str, Any]:
    r = json.loads((run_dir / "report.json").read_text())
    m = json.loads((run_dir / "manifest.json").read_text())
    sim = r["simulation"]
    out: dict[str, Any] = {
        "run_id": m["run_id"],
        "git_commit": m["git_commit"],
        "git_dirty": m["git_dirty"],
        "finished_utc": m.get("finished_utc"),
        "data_sha256": m["data_sha256"],
        "split_label_counts": m["split_label_counts"],
        "dependencies": m["dependencies"],
        "policy_snapshot": yaml.safe_load((run_dir / "policy.yaml").read_text()),
        "decision_contract": r.get("decision_contract"),
        "review_workload": r.get("review_workload"),
        "tiers": r["tiers"],
        "tiers_model_only": r["tiers_model_only"],
        "threshold_selection_tiers": r["threshold_selection"]["tiers"],
        "per_label": r["per_label"],
        "threshold_selection_per_label": r["threshold_selection"].get("per_label"),
        "rules": r["rules"],
        "consistency": r["consistency"],
        "allow_false_negatives": r["allow_false_negatives"],
        "simulation": {
            "assumptions": sim.get("assumptions"),
            "summary": sim.get("summary"),
            "paired": sim.get("paired"),
            "harm_per_reviewer_hour": sim.get("harm_per_reviewer_hour"),
            "high_risk_wait_p90": sim.get("high_risk_wait_p90"),
            "high_risk_handled": sim.get("high_risk_handled"),
            "high_risk_arrived": sim.get("high_risk_arrived"),
            "queue_depth_p95": sim.get("queue_depth_p95"),
            "reviewer_utilization": sim.get("reviewer_utilization"),
            "primary": sim.get("primary"),
            "thesis": sim.get("thesis"),
        },
        "identity_false_positives": r.get("identity_false_positives"),
        "subgroup_thresholds": r.get("subgroup_thresholds"),
    }
    thresholds_path = run_dir / "thresholds.json"
    if thresholds_path.is_file():
        out["thresholds"] = json.loads(thresholds_path.read_text())
    out["identity_summary"] = identity_summary(r.get("identity_false_positives") or {})
    return out


def _pooled(basis: dict[str, Any]) -> dict[str, Any]:
    """The pooled positive-tier comparison, whichever schema wrote it."""
    return basis.get("predicted_positive", basis)


def _ratio(cmp: dict[str, Any]) -> tuple[float | None, list[float] | None]:
    for key in ("false_discovery_rate_ratio", "false_positive_rate_ratio"):
        if key in cmp:
            return cmp.get(key), cmp.get(key + "_ci95")
    return None, None


def identity_summary(idt: dict[str, Any]) -> dict[str, Any]:
    """Numbers the record quotes, normalised across the report schemas that existed.

    Older runs report the pooled comparison directly under each basis and call
    FP / predicted-positive a "false positive rate"; the committed pipeline
    nests it under predicted_positive and calls it a false discovery rate,
    which is the correct term. Both are read here.
    """
    out: dict[str, Any] = {}
    sel = idt.get("threshold_selection") or {}
    test = idt.get("test") or {}
    if sel:
        pooled_model = _pooled(sel.get("model_tier", {}))
        pooled_final = _pooled(sel.get("final_tier", {}))
        w_model, w_final = (
            pooled_model.get("with_identity_term", {}),
            pooled_final.get("with_identity_term", {}),
        )
        if w_model and w_final:
            out["selection_identity_predicted_positive_model"] = w_model.get("n_predicted_positive")
            out["selection_identity_predicted_positive_final"] = w_final.get("n_predicted_positive")
            out["selection_identity_moved_to_positive_by_rules"] = w_final.get(
                "n_predicted_positive", 0
            ) - w_model.get("n_predicted_positive", 0)
        omission = sel.get("final_tier", {}).get("allow_false_omission") or sel.get(
            "final_tier", {}
        ).get("allow_miss_rate")
        if omission:
            w = omission["with_identity_term"]
            out["selection_identity_allowed"] = w.get("n_allowed")
            out["selection_identity_allowed_truly_positive"] = w.get("n_truly_positive")
    if test:
        for basis in ("population_tier", "model_tier", "final_tier"):
            if basis not in test:
                continue
            for tier in ("predicted_positive", "auto_action", "priority_review", "human_review"):
                cmp = (
                    _pooled(test[basis]) if tier == "predicted_positive" else test[basis].get(tier)
                )
                if not cmp:
                    continue
                ratio, ci = _ratio(cmp)
                out[f"test_{basis}_{tier}"] = {
                    "ratio": ratio,
                    "ci95": ci,
                    "with": cmp.get("with_identity_term"),
                    "without": cmp.get("without_identity_term"),
                }
        fpr = (
            test.get("final_tier", {}).get("clean_negative_false_positives", {}).get("auto_action")
        )
        if fpr:
            out["test_final_auto_clean_negative_fpr"] = fpr
        out["test_by_term"] = test.get("final_tier", {}).get("by_term")
        out["identity_terms"] = test.get("identity_terms")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--human-review", type=Path,
        help="collect a v2 human-review run into human_review_run.json; preserve round-1 history",
    )
    parser.add_argument(
        "--with-r103", type=Path, help="run made with rule R103 active"
    )
    parser.add_argument(
        "--without-r103",
        type=Path,
        help="run with R103 removed, before subgroup thresholds",
    )
    parser.add_argument(
        "--final", type=Path, help="final round-1 run (subgroup thresholds active)"
    )
    args = parser.parse_args()
    if args.human_review:
        if args.with_r103 or args.without_r103 or args.final:
            parser.error("--human-review cannot be combined with historical run arguments")
        run = pick(args.human_review.resolve())
        if (run.get("decision_contract") or {}).get("mode") != "human_confirmation":
            parser.error("--human-review requires a human-confirmation run")
        destination = HERE / "human_review_run.json"
        destination.write_text(json.dumps(run, indent=2, default=str))
        print(destination)
        return
    if not all((args.with_r103, args.without_r103, args.final)):
        parser.error("provide --human-review or all three historical run arguments")
    runs = {
        "with_r103": pick(args.with_r103.resolve()),
        "without_r103": pick(args.without_r103.resolve()),
        "final": pick(args.final.resolve()),
    }
    (HERE / "runs.json").write_text(
        json.dumps(runs, indent=2, default=lambda value: value.isoformat() if isinstance(value, date) else float(value))
    )
    print(HERE / "runs.json")


if __name__ == "__main__":
    main()
