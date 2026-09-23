#!/usr/bin/env python
"""Collect the numbers the record quotes from finished runs into runs.json.

    python record/collect_runs.py --with-r103 reports/<a> --without-r103 reports/<b> --final reports/<c>
    python record/collect_runs.py --human-review reports/<v2 run>
    python record/collect_runs.py --features analysis

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
        "prevalence_shift": r.get("prevalence_shift"),
        "simulation": {
            "note": sim.get("note"),
            "assumptions": sim.get("assumptions"),
            "summary": sim.get("summary"),
            "paired": sim.get("paired"),
            "harm_per_reviewer_hour": sim.get("harm_per_reviewer_hour"),
            "high_risk_wait_p90": sim.get("high_risk_wait_p90"),
            "high_risk_handled": sim.get("high_risk_handled"),
            "high_risk_arrived": sim.get("high_risk_arrived"),
            "headline": sim.get("headline"),
            "time_metrics": sim.get("time_metrics"),
            "completion_ratio": sim.get("completion_ratio"),
            "backlog_end": sim.get("backlog_end"),
            "high_risk_unfinished": sim.get("high_risk_unfinished"),
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


# ---------------------------------------------------------------- step 4: features
FEATURE_LABELS = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")


def _cost(run_dir: Path) -> dict[str, Any]:
    from datetime import datetime

    m = json.loads((run_dir / "manifest.json").read_text())
    started = datetime.fromisoformat(m["started_utc"].replace("Z", "+00:00"))
    finished = datetime.fromisoformat(m["finished_utc"].replace("Z", "+00:00"))
    return {
        "run_id": m["run_id"],
        "git_commit": m["git_commit"],
        "git_dirty": m["git_dirty"],
        "analyzer": (m.get("config") or {}).get("model", {}).get("analyzer", "word"),
        "seconds": round((finished - started).total_seconds()),
        "model_mb": round((run_dir / "model.pkl").stat().st_size / 1e6, 1),
    }


def _capture_curve(run_dir: Path, ks: list[int]) -> dict[str, Any]:
    """Positives and high-risk rows among the top k by max calibrated label probability."""
    import numpy as np
    import pandas as pd

    pred = pd.read_csv(run_dir / "predictions.csv", dtype={"id": str})
    policy = yaml.safe_load((run_dir / "policy.yaml").read_text())
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    hr_min = float((config.get("simulation") or {}).get("high_risk_min_weight", 5))
    weights = np.array([float(policy["severity_weights"].get(lb, 0)) for lb in FEATURE_LABELS])
    y = pred[[f"y_{lb}" for lb in FEATURE_LABELS]].to_numpy(dtype=int)
    proba = pred[[f"p_{lb}" for lb in FEATURE_LABELS]].to_numpy(dtype=float)
    order = np.argsort(-proba.max(axis=1), kind="stable")
    pos = np.cumsum(y[order].any(axis=1))
    high = np.cumsum(((y * weights).max(axis=1) >= hr_min)[order])
    flagged = pred["final_tier"] != "allow"
    harm = (y * weights).max(axis=1)
    return {
        "k": ks,
        "positives": [int(pos[k - 1]) if k else 0 for k in ks],
        "high_risk": [int(high[k - 1]) if k else 0 for k in ks],
        "operating_point": {
            "flagged": int(flagged.sum()),
            "positives": int((flagged & y.any(axis=1)).sum()),
            "high_risk": int((flagged & (harm >= hr_min)).sum()),
        },
        "total_positives": int(y.any(axis=1).sum()),
        "total_high_risk": int((harm >= hr_min).sum()),
    }


def _equal_count_precision(dirs: list[Path]) -> dict[str, Any]:
    """Per-label precision of each run at its own human-review count and at the other run's."""
    import numpy as np
    import pandas as pd

    preds = [pd.read_csv(d / "predictions.csv", dtype={"id": str}) for d in dirs]
    reports = [json.loads((d / "report.json").read_text()) for d in dirs]
    if not all((p["id"] == preds[0]["id"]).all() for p in preds):
        raise SystemExit("runs do not share predictions.csv row order")
    out: dict[str, Any] = {}
    for label in FEATURE_LABELS:
        counts = [r["per_label"][label]["at_human_review"]["n_predicted_positive"] for r in reports]
        if not all(counts):
            continue
        y = preds[0][f"y_{label}"].to_numpy()
        entry = []
        for i, pred in enumerate(preds):
            order = np.argsort(-pred[f"p_{label}"].to_numpy(), kind="stable")
            entry.append({
                "own_count": counts[i],
                "precision_at_count": {str(k): float(y[order[:k]].mean()) for k in counts},
            })
        out[label] = entry
    return out


def _gates(path: Path) -> dict[str, Any]:
    lines = path.read_text().splitlines()
    return {
        "command": lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else None,
        "failed": [ln.split("::")[1].split(" ")[0] for ln in lines if ln.startswith("FAILED")],
        "messages": [ln.split("AssertionError: ", 1)[1].split("; stats=")[0]
                     for ln in lines if "AssertionError: " in ln],
        "summary": lines[-1] if lines else None,
    }


def pick_features(analysis: Path, reports: Path) -> dict[str, Any]:
    import pandas as pd

    r1 = json.loads((analysis / "run_comparison.json").read_text())
    v2 = json.loads((analysis / "v2" / "run_comparison.json").read_text())
    audit_dir = analysis / "auto_action_fp_audit"
    meta = json.loads((audit_dir / "sample_meta.json").read_text())
    sample = pd.read_csv(audit_dir / "sample.csv", dtype=str, keep_default_na=False)

    def excerpt(text: str, n: int = 110) -> str:
        t = " ".join(text.split())
        return t if len(t) <= n else t[: n - 1] + "…"

    changed = [
        {"text": excerpt(r.text), "preread": r.preread_verdict, "human": r.human_verdict,
         "note": r.human_note}
        for r in sample.itertuples() if r.human_changed == "True"
    ]
    v2_dirs = [reports / r["run"] for r in v2["runs"]]
    r1_dirs = [reports / r["run"] for r in r1["runs"]]
    n_rows = int(v2["runs"][0]["recall"]["flagged"]["n"])
    ks = sorted({*range(0, 12001, 250), *(int(r["recall"]["flagged"]["k"]) for r in v2["runs"])})
    ks = [k for k in ks if k <= n_rows]
    verification = analysis / "v2" / "verification.json"
    return {
        "collected": date.today().isoformat(),
        "round1": {
            "runs": r1["runs"],
            "audit_overlap": r1.get("audit_overlap"),
            "paired_false_positives": json.loads(
                (analysis / "paired_false_positives.json").read_text()
            ),
            "costs": [_cost(d) for d in r1_dirs if d.is_dir()],
        },
        "audit": {
            "n_auto_action": meta["n_auto_action"],
            "n_false_positive": meta["n_false_positive"],
            "population_strata": meta["population_strata"],
            "sample_strata": meta["sample_strata"],
            "preread": meta.get("preread"),
            "human": meta.get("human"),
            "changed": changed,
        },
        "v2": {
            "runs": v2["runs"],
            "matched_volume": v2.get("matched_volume"),
            "equal_input": v2.get("equal_input"),
            "gates": {
                name: _gates(analysis / "v2" / f"gates_{name}.txt")
                for name in ("word", "word_char")
                if (analysis / "v2" / f"gates_{name}.txt").is_file()
            },
            "costs": [_cost(d) for d in v2_dirs],
            "curves": {r["label"]: _capture_curve(d, ks) for r, d in zip(v2["runs"], v2_dirs)},
            "equal_count_precision": _equal_count_precision(v2_dirs),
        },
        "verification": json.loads(verification.read_text()) if verification.is_file() else None,
    }


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
    parser.add_argument(
        "--features", type=Path,
        help="collect the step-4 feature comparison from this analysis directory",
    )
    parser.add_argument("--reports", type=Path, default=HERE.parent / "reports")
    args = parser.parse_args()
    if args.features:
        destination = HERE / "features_run.json"
        destination.write_text(
            json.dumps(pick_features(args.features.resolve(), args.reports.resolve()), indent=2,
                       ensure_ascii=False)
        )
        print(destination)
        return
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
