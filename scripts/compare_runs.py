#!/usr/bin/env python
"""Side-by-side comparison of pipeline runs on the scored test rows.

    python scripts/compare_runs.py reports/<a> reports/<b> ... [--audit DIR] [--equal-input]

Writes <out>/run_comparison.{json,md}. The runs must share the data hash and
split fractions (checked), so any difference comes from the config. The first
run is the reference.

Sections, each present only when the runs carry the data:

- Tiers: size, coverage, test precision with interval, selection precision.
- Workload and recall: how many rows need a reviewer, how many truly positive
  and high-risk rows reach the queue, and how many reach the top tier. High
  risk is the pipeline's definition: max severity weight over true labels at
  or above simulation.high_risk_min_weight.
- Ranking: per-label average precision and ROC-AUC.
- Queue at equal review load: the report's own simulation summary. Its
  arrival rates are post-admission review demand, so two models are compared
  at the same number of review jobs per hour even if one flags more comments.
- Queue at equal comment input (--equal-input): each run's queue is
  re-simulated at the review load it would receive from the same stream of
  incoming comments, namely input rate x that run's queue fraction. The input
  rates are the reference run's equivalent input loads, so the reference
  reproduces its own review loads. A model that flags more comments gets more
  review work, and every harm-carrying comment it leaves in allow is lost.
- Audit overlap (--audit, round-1 runs with an auto_action tier only).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from review_router.data import LABELS  # noqa: E402
from review_router.pipeline import queue_inputs  # noqa: E402
from review_router.policy import load_policy  # noqa: E402
from review_router.simulate import SimConfig, draw_scenario, simulate  # noqa: E402

TIER_ORDER = ("auto_action", "priority_review", "human_review")
SIM_FIELDS = (
    "harm_per_reviewer_hour",
    "high_risk_handled",
    "high_risk_unhandled",
    "high_risk_wait_p90",
    "completion_ratio",
    "backlog_end",
)
EQUAL_INPUT_SEEDS = tuple(range(1, 21))
EQUAL_INPUT_STRATEGIES = ("priority", "fifo")


def load(run: Path) -> dict[str, Any]:
    report = json.loads((run / "report.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    config = manifest.get("config") or {}
    return {
        "dir": run,
        "run": run.name,
        "label": config.get("label", run.name),
        "model": config.get("model", {}),
        "config": config,
        "manifest": manifest,
        "data": manifest.get("data_sha256"),
        "split": config.get("split_fractions"),
        "report": report,
        "pred": pd.read_csv(run / "predictions.csv", dtype={"id": str}),
    }


def tiers_of(report: dict[str, Any]) -> list[str]:
    return [t for t in TIER_ORDER if t in report["tiers"]]


def arrays(pred: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proba = pred[[f"p_{lb}" for lb in LABELS]].to_numpy(dtype=float)
    y = pred[[f"y_{lb}" for lb in LABELS]].to_numpy(dtype=int)
    return proba, y, pred["final_tier"].to_numpy()


def recall_table(
    y: np.ndarray, final: np.ndarray, weights: np.ndarray, high_risk_min_weight: float, top: str
) -> dict[str, Any]:
    """Where the truly positive and high-risk rows went. Hand-computable from predictions.

    "Flagged" is every tier except allow: under policy v2 that is the review
    queue; under round 1 it also includes auto_action.
    """
    flagged = final != "allow"
    in_top = final == top
    any_true = y.any(axis=1)
    harm = (y * weights).max(axis=1)
    high_risk = harm >= high_risk_min_weight

    def share(mask: np.ndarray, of: np.ndarray) -> dict[str, Any]:
        n = int(of.sum())
        k = int((mask & of).sum())
        return {"k": k, "n": n, "share": k / n if n else None}

    return {
        "flagged": {"k": int(flagged.sum()), "n": int(len(final))},
        "positives_flagged": share(flagged, any_true),
        "positives_in_top": share(in_top, any_true),
        "high_risk_flagged": share(flagged, high_risk),
        "high_risk_in_top": share(in_top, high_risk),
        "high_risk_in_allow": share(final == "allow", high_risk),
        "harm_total": float(harm.sum()),
        "harm_flagged": float(harm[flagged].sum()),
    }


def summarize(item: dict[str, Any]) -> dict[str, Any]:
    r = item["report"]
    out: dict[str, Any] = {
        "run": item["run"],
        "label": item["label"],
        "model": item["model"],
        "git_commit": item["manifest"].get("git_commit"),
        "git_dirty": item["manifest"].get("git_dirty"),
        "tiers": {},
    }
    for tier in tiers_of(r):
        t = r["tiers"][tier]
        sel = r["threshold_selection"]["tiers"].get(tier, {})
        out["tiers"][tier] = {
            "test_precision": t["precision"],
            "test_precision_ci95": t.get("precision_ci95"),
            "test_n_predicted_positive": t["n_predicted_positive"],
            "test_coverage": t["coverage"],
            "selection_precision": sel.get("precision"),
            "selection_n_predicted_positive": sel.get("n_predicted_positive"),
        }
    out["per_label"] = {
        lb: {
            "average_precision": r["per_label"][lb].get("average_precision"),
            "roc_auc": r["per_label"][lb].get("roc_auc"),
        }
        for lb in LABELS
    }
    if r.get("review_workload"):
        out["review_workload"] = r["review_workload"]
    policy = load_policy(item["dir"] / "policy.yaml")
    weights = np.array([policy.severity_weights.get(lb, 0.0) for lb in LABELS])
    _, y, final = arrays(item["pred"])
    top = next(t for t in TIER_ORDER if t in r["tiers"])
    hr_min = float(item["config"].get("high_risk_min_weight", 5))
    out["recall"] = {"top_tier": top, **recall_table(y, final, weights, hr_min, top)}
    sim = r.get("simulation") or {}
    if sim.get("summary"):
        strategy = (sim.get("primary") or {}).get("strategy")
        out["queue_equal_review_load"] = {
            "router_strategy": strategy,
            "assumption": (sim.get("assumptions") or {}).get("arrival_rates_are"),
            "queue_fraction": (sim.get("assumptions") or {}).get("queue_fraction"),
            "by_load": {
                key: {f: stats.get(f) for f in SIM_FIELDS}
                for key, stats in sim["summary"].items()
                if key.split("@")[0] in (strategy, "fifo")
            },
            "headline": sim.get("headline"),
        }
    return out


def captured_at_volume(
    proba: np.ndarray, y: np.ndarray, weights: np.ndarray, high_risk_min_weight: float, k: int
) -> dict[str, float]:
    """Positives and high-risk rows among the k rows with the highest max label probability.

    Holding k fixed separates ranking quality from how many rows a model flags.
    Ties at the cut are broken by row order (stable sort), identically for every run.
    """
    score = proba.max(axis=1)
    top = np.argsort(-score, kind="stable")[:k]
    harm = (y * weights).max(axis=1)
    return {
        "k": int(k),
        "positives": int(y[top].any(axis=1).sum()),
        "high_risk": int((harm[top] >= high_risk_min_weight).sum()),
        "harm": float(harm[top].sum()),
    }


def matched_volume(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Every run ranked at every run's flagged volume."""
    ks = sorted({int((i["pred"]["final_tier"] != "allow").sum()) for i in items})
    out: dict[str, Any] = {"score": "max calibrated label probability", "k": ks, "runs": {}}
    for item in items:
        policy = load_policy(item["dir"] / "policy.yaml")
        weights = np.array([policy.severity_weights.get(lb, 0.0) for lb in LABELS])
        proba, y, _ = arrays(item["pred"])
        hr_min = float(item["config"].get("high_risk_min_weight", 5))
        out["runs"][item["label"]] = [captured_at_volume(proba, y, weights, hr_min, k) for k in ks]
    return out


def equal_input_loads(reference_fraction: float, review_loads: list[float]) -> list[float]:
    """Incoming-comment rates at which the reference run receives review_loads."""
    return [load / reference_fraction for load in review_loads]


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}


def resimulate(item: dict[str, Any], input_rates: list[float]) -> dict[str, Any]:
    """Queue metrics for this run's review pool at a fixed incoming-comment rate."""
    config = item["config"]
    sim_cfg = SimConfig(
        **{k: config["sim"][k] for k in ("reviewers", "handle_minutes", "horizon_hours")}
    )
    policy = load_policy(item["dir"] / "policy.yaml")
    proba, y, final = arrays(item["pred"])
    inputs = queue_inputs(proba, y, final, policy, float(config["high_risk_min_weight"]))
    n_jobs = len(inputs["queued"])
    fraction = n_jobs / len(final)
    out: dict[str, Any] = {
        "queue_fraction": fraction,
        "seeds": list(EQUAL_INPUT_SEEDS),
        "by_input": {},
    }
    for rate in input_rates:
        load = rate * fraction
        per_strategy: dict[str, Any] = {}
        for strategy in EQUAL_INPUT_STRATEGIES:
            rows: dict[str, list[float]] = {}
            for seed in EQUAL_INPUT_SEEDS:
                scenario = draw_scenario(seed, load, n_jobs, sim_cfg)
                result, records = simulate(
                    scenario,
                    inputs["priorities"][strategy],
                    inputs["harm"],
                    inputs["high_risk"],
                    sim_cfg,
                    strategy,
                )
                res = result.as_json()
                completed = records.completed
                hours = sim_cfg.horizon_hours
                values = {
                    "review_arrivals": float(len(records)),
                    "completion_ratio": float(res["completion_ratio"]),
                    "backlog_end": float(res["backlog_end"]),
                    "positives_completed_per_hour": float(((records.harm > 0) & completed).sum())
                    / hours,
                    "clean_share_of_completed": float(((records.harm == 0) & completed).sum())
                    / max(int(completed.sum()), 1),
                    "high_risk_arrived": float(res["high_risk_arrived"]),
                    "high_risk_handled": float(res["high_risk_handled"]),
                    "high_risk_unhandled": float(res["high_risk_unhandled"]),
                    "harm_handled_per_hour": float(records.harm[completed].sum()) / hours,
                }
                for key, value in values.items():
                    rows.setdefault(key, []).append(value)
            per_strategy[strategy] = {k: _mean_std(v) for k, v in rows.items()}
        out["by_input"][f"{rate:.1f}"] = {"review_load_per_hour": load, **per_strategy}
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


def _f(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, dict) and "mean" in value:
        return f"{value['mean']:.{digits}f} ± {value['std']:.{digits}f}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def to_markdown(summary: list[dict[str, Any]], extra: dict[str, Any]) -> str:
    lines = ["# Run comparison (scored test rows)", ""]
    lines.append(
        "| run | analyzer | commit | tier | n | coverage | test precision (95% CI) "
        "| selection precision |"
    )
    lines.append("|---|---|---|---|---:|---:|---|---:|")
    for s in summary:
        commit = (s["git_commit"] or "")[:8] + (" dirty" if s["git_dirty"] else "")
        for tier, t in s["tiers"].items():
            ci = t["test_precision_ci95"] or [float("nan")] * 2
            lines.append(
                f"| {s['label']} | {s['model'].get('analyzer', 'word')} | {commit} | {tier} | "
                f"{t['test_n_predicted_positive']} | {t['test_coverage']:.4f} | "
                f"{_f(t['test_precision'])} [{ci[0]:.3f}, {ci[1]:.3f}] | "
                f"{_f(t['selection_precision'])} |"
            )
    if all("review_workload" in s for s in summary):
        lines += ["", "## Review workload", ""]
        lines.append("| run | needs a reviewer | share of comments | queue precision (95% CI) |")
        lines.append("|---|---:|---:|---|")
        for s in summary:
            w = s["review_workload"]
            ci = w["precision_ci95"]
            lines.append(
                f"| {s['label']} | {w['n_requires_human_review']} | {w['review_fraction']:.4f} | "
                f"{w['precision']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] |"
            )
    lines += ["", "## Where the positives went", ""]
    lines.append(
        "| run | top tier | positives flagged | positives in top tier | high-risk flagged | "
        "high-risk in top tier | high-risk left in allow |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for s in summary:
        rc = s["recall"]

        def cell(entry: dict[str, Any]) -> str:
            return f"{entry['k']}/{entry['n']} ({entry['share']:.3f})"

        lines.append(
            f"| {s['label']} | {rc['top_tier']} | {cell(rc['positives_flagged'])} | "
            f"{cell(rc['positives_in_top'])} | {cell(rc['high_risk_flagged'])} | "
            f"{cell(rc['high_risk_in_top'])} | {cell(rc['high_risk_in_allow'])} |"
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
    mv = extra.get("matched_volume")
    if mv:
        lines += [
            "",
            "## Ranking at matched volume (top k rows by max label probability)",
            "",
        ]
        lines.append("| run | k | positives | high-risk | harm proxy |")
        lines.append("|---|---:|---:|---:|---:|")
        for label, entries in mv["runs"].items():
            for e in entries:
                lines.append(
                    f"| {label} | {e['k']} | {e['positives']} | {e['high_risk']} | "
                    f"{e['harm']:.0f} |"
                )
    if all("queue_equal_review_load" in s for s in summary):
        lines += [
            "",
            "## Queue at equal review load (report simulation, 5 seeds, mean ± std)",
            "",
            "Arrival rates are review jobs per hour after admission, identical for every run.",
            "",
        ]
        lines.append("| run | strategy@load | " + " | ".join(SIM_FIELDS) + " |")
        lines.append("|---|---|" + "---:|" * len(SIM_FIELDS))
        for s in summary:
            for key, stats in s["queue_equal_review_load"]["by_load"].items():
                cells = " | ".join(_f(stats[f], 2) for f in SIM_FIELDS)
                lines.append(f"| {s['label']} | {key} | {cells} |")
    eq = extra.get("equal_input")
    if eq:
        fields = (
            "review_arrivals",
            "completion_ratio",
            "backlog_end",
            "positives_completed_per_hour",
            "clean_share_of_completed",
            "high_risk_handled",
            "high_risk_unhandled",
            "harm_handled_per_hour",
        )
        lines += [
            "",
            f"## Queue at equal comment input ({len(EQUAL_INPUT_SEEDS)} seeds, mean ± std)",
            "",
            "Input rates are the reference run's equivalent incoming-comment loads; each run's",
            "review load is input rate x its own queue fraction.",
            "",
        ]
        lines.append("| run | input/h | review load/h | strategy | " + " | ".join(fields) + " |")
        lines.append("|---|---:|---:|---|" + "---:|" * len(fields))
        for label, block in eq["runs"].items():
            for rate, entry in block["by_input"].items():
                for strategy in EQUAL_INPUT_STRATEGIES:
                    cells = " | ".join(_f(entry[strategy][f], 2) for f in fields)
                    lines.append(
                        f"| {label} | {float(rate):.0f} | {entry['review_load_per_hour']:.1f} | "
                        f"{strategy} | {cells} |"
                    )
    overlap = extra.get("audit_overlap")
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
    parser.add_argument("--equal-input", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    items = [load(r) for r in args.runs]
    base = items[0]
    for item in items[1:]:
        if item["data"] != base["data"] or item["split"] != base["split"]:
            raise SystemExit(f"{item['run']} does not share data hashes/split with {base['run']}")
    summary = [summarize(i) for i in items]
    extra: dict[str, Any] = {"matched_volume": matched_volume(items)}
    if args.audit:
        if not all("auto_action" in i["report"]["tiers"] for i in items):
            raise SystemExit("--audit needs runs with an auto_action tier (round-1 policy)")
        extra["audit_overlap"] = audit_overlap(items, args.audit)
    if args.equal_input:
        reference = base["report"]["simulation"]["assumptions"]
        rates = equal_input_loads(reference["queue_fraction"], list(reference["loads_per_hour"]))
        extra["equal_input"] = {
            "reference": base["label"],
            "input_rates_per_hour": rates,
            "runs": {i["label"]: resimulate(i, rates) for i in items},
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run_comparison.json").write_text(
        json.dumps({"runs": summary, **extra}, indent=2) + "\n"
    )
    md = to_markdown(summary, extra)
    (args.out / "run_comparison.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
