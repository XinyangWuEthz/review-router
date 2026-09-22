#!/usr/bin/env python
"""Side-by-side comparison of pipeline runs on the scored test rows.

    python scripts/compare_runs.py reports/<a> reports/<b> ... --audit analysis/auto_action_fp_audit

Writes analysis/run_comparison.{json,md}. The runs must share the data hash
and split fractions (checked), so any difference is the config. With --audit
it also reports how many of the audited false positives each run still sends
to auto_action, split by the pre-read verdict: a variant that only drops the
"clean" rows fixes model error; one that drops rows uniformly has just moved
the threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from review_router.data import LABELS  # noqa: E402

TIERS = ("auto_action", "human_review")


def load(run: Path) -> dict[str, Any]:
    report = json.loads((run / "report.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    config = manifest.get("config") or {}
    return {
        "run": run.name,
        "label": config.get("label", run.name),
        "model": config.get("model", {}),
        "data": manifest.get("data_sha256"),
        "split": config.get("split_fractions"),
        "report": report,
        "pred": pd.read_csv(run / "predictions.csv", dtype={"id": str}),
    }


def summarize(item: dict[str, Any]) -> dict[str, Any]:
    r = item["report"]
    out: dict[str, Any] = {"run": item["run"], "label": item["label"], "model": item["model"]}
    for tier in TIERS:
        t = r["tiers"][tier]
        sel = r["threshold_selection"]["tiers"].get(tier, {})
        out[tier] = {
            "test_precision": t["precision"],
            "test_precision_ci95": t.get("precision_ci95"),
            "test_n_predicted_positive": t["n_predicted_positive"],
            "test_coverage": t["coverage"],
            "selection_precision": sel.get("precision"),
            "selection_n_predicted_positive": sel.get("n_predicted_positive"),
        }
        if sel.get("precision") is not None and t["precision"] is not None:
            out[tier]["selection_minus_test"] = sel["precision"] - t["precision"]
    out["per_label"] = {
        lb: {
            "average_precision": r["per_label"][lb].get("average_precision"),
            "roc_auc": r["per_label"][lb].get("roc_auc"),
        }
        for lb in LABELS
    }
    return out


def audit_overlap(items: list[dict[str, Any]], audit: Path) -> dict[str, Any]:
    sample = pd.read_csv(audit / "sample.csv", dtype=str, keep_default_na=False)
    verdict_col = "human_verdict" if (sample.human_verdict != "").all() else "preread_verdict"
    out: dict[str, Any] = {"verdict_column": verdict_col, "n_audited": int(len(sample))}
    for item in items:
        pred = item["pred"].set_index("id")
        still_auto = pred.loc[sample.id, "final_tier"].to_numpy() == "auto_action"
        by = (
            pd.Series(still_auto, index=sample[verdict_col].to_numpy())
            .groupby(level=0)
            .agg(["sum", "count"])
        )
        out[item["label"]] = {
            "still_auto_action": int(still_auto.sum()),
            "by_verdict": {k: f"{int(v['sum'])}/{int(v['count'])}" for k, v in by.iterrows()},
        }
    return out


def to_markdown(summary: list[dict[str, Any]], overlap: dict[str, Any] | None) -> str:
    lines = ["# Run comparison (scored test rows)", ""]
    lines.append(
        "| run | analyzer | auto prec (95% CI) | auto n | auto sel. prec | human prec | human n |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for s in summary:
        a, h = s["auto_action"], s["human_review"]
        ci = a["test_precision_ci95"] or [float("nan")] * 2
        lines.append(
            f"| {s['label']} | {s['model'].get('analyzer', 'word')} | "
            f"{a['test_precision']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{a['test_n_predicted_positive']} | "
            f"{(a['selection_precision'] or float('nan')):.3f} | "
            f"{h['test_precision']:.3f} | {h['test_n_predicted_positive']} |"
        )
    lines += [
        "",
        "| label | "
        + " | ".join(f"AP {s['label']}" for s in summary)
        + " | "
        + " | ".join(f"AUC {s['label']}" for s in summary)
        + " |",
    ]
    lines.append("|---|" + "---:|" * (2 * len(summary)))
    for lb in LABELS:
        aps = " | ".join(f"{s['per_label'][lb]['average_precision']:.3f}" for s in summary)
        aucs = " | ".join(f"{s['per_label'][lb]['roc_auc']:.3f}" for s in summary)
        lines.append(f"| {lb} | {aps} | {aucs} |")
    if overlap:
        lines += [
            "",
            f"## Audited false positives still sent to auto_action ({overlap['verdict_column']})",
            "",
        ]
        lines.append("| run | still auto_action | by verdict |")
        lines.append("|---|---:|---|")
        for s in summary:
            o = overlap[s["label"]]
            by = ", ".join(f"{k} {v}" for k, v in o["by_verdict"].items())
            lines.append(
                f"| {s['label']} | {o['still_auto_action']}/{overlap['n_audited']} | {by} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--audit", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    items = [load(r) for r in args.runs]
    base = items[0]
    for item in items[1:]:
        if item["data"] != base["data"] or item["split"] != base["split"]:
            raise SystemExit(f"{item['run']} does not share data hashes/split with {base['run']}")
    summary = [summarize(i) for i in items]
    overlap = audit_overlap(items, args.audit) if args.audit else None
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run_comparison.json").write_text(
        json.dumps({"runs": summary, "audit_overlap": overlap}, indent=2) + "\n"
    )
    md = to_markdown(summary, overlap)
    (args.out / "run_comparison.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
