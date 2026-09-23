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
- Queue at equal comment input (--equal-input, human-confirmation runs only): common
  random numbers. For each seed one stream of incoming comments is drawn from
  the scored test rows (Poisson arrivals, rows sampled with replacement) and
  shared by every run; each run admits the comments it flags and reviews them
  with 4 reviewers. A model that flags more gets more review work, and every
  high-risk comment it leaves in allow counts as not reviewed. Severity,
  band-first priority and FIFO orderings share the same stream. Input rates
  are the reference run's equivalent input loads. Differences are paired by seed.
- Audit overlap (--audit, round-1 runs with an auto_action tier only).

Standard deviations: the report's own simulation summary uses the population
SD over its seeds (ddof=0); everything this script simulates uses the sample SD
(ddof=1), and paired differences are reported with their standard error.
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
from review_router.simulate import Scenario, SimConfig, simulate  # noqa: E402

TIER_ORDER = ("auto_action", "priority_review", "human_review")
SIM_FIELDS = (
    "harm_per_reviewer_hour",
    "high_risk_handled",
    "high_risk_unhandled",
    "high_risk_wait_p90",
    "completion_ratio",
    "backlog_end",
)
EQUAL_INPUT_SEEDS = tuple(range(1, 201))
EQUAL_INPUT_STRATEGIES = ("severity", "priority", "fifo")


def load(run: Path) -> dict[str, Any]:
    report = json.loads((run / "report.json").read_text())
    if report.get("evaluate_test") is False:
        raise SystemExit(
            f"{run} is a development run (evaluate_test: false): test rows not scored, "
            "so there are no predictions or test tiers to compare"
        )
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

    "Flagged" is every tier except allow: under human confirmation that is the review
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
            "n_seeds": len((sim.get("assumptions") or {}).get("seeds") or []),
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


def _mean_std(values: list[float | None]) -> dict[str, float | int | None]:
    """Summarize defined metrics without treating an empty denominator as zero."""
    arr = np.asarray([value for value in values if value is not None], dtype=float)
    return {
        "mean": float(arr.mean()) if len(arr) else None,
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else None,
        "valid_n": len(arr),
    }


STREAM_FIELDS = (
    "review_arrivals",
    "completion_ratio",
    "backlog_end",
    "positives_completed_per_hour",
    "harm_handled_per_hour",
    "high_risk_in_stream",
    "high_risk_left_in_allow",
    "high_risk_unhandled_in_queue",
    "high_risk_not_reviewed",
)


def _stream_metrics(
    item: dict[str, Any],
    row: np.ndarray,
    arrival: np.ndarray,
    sim_cfg: SimConfig,
    strategy: str,
    seed: int,
) -> dict[str, float | None]:
    """One run's review of one shared comment stream."""
    ctx = item["_stream"]
    admitted = ctx["flagged"][row]
    job_index = ctx["queue_pos"][row[admitted]]
    scenario = Scenario(
        seed=seed,
        load_per_hour=float(admitted.sum()) / sim_cfg.horizon_hours,
        job_index=job_index,
        arrival_min=arrival[admitted],
        service_min=np.full(int(admitted.sum()), sim_cfg.handle_minutes),
    )
    result, records = simulate(
        scenario, ctx["priorities"][strategy], ctx["harm"], ctx["high_risk"], sim_cfg, strategy
    )
    res = result.as_json()
    completed = records.completed
    hours = sim_cfg.horizon_hours
    stream_high_risk = int(ctx["high_risk_all"][row].sum())
    left_in_allow = int((ctx["high_risk_all"][row] & ~admitted).sum())
    return {
        "review_arrivals": float(len(records)),
        "completion_ratio": res["completion_ratio"],
        "backlog_end": float(res["backlog_end"]),
        "positives_completed_per_hour": float(
            (ctx["positive"][records.job_index] & completed).sum()
        )
        / hours,
        "harm_handled_per_hour": float(records.harm[completed].sum()) / hours,
        "high_risk_in_stream": float(stream_high_risk),
        "high_risk_left_in_allow": float(left_in_allow),
        "high_risk_unhandled_in_queue": float(res["high_risk_unhandled"]),
        "high_risk_not_reviewed": float(stream_high_risk - res["high_risk_handled"]),
    }


def _prepare_stream(item: dict[str, Any]) -> None:
    config = item["config"]
    policy = load_policy(item["dir"] / "policy.yaml")
    proba, y, final = arrays(item["pred"])
    hr_min = float(config["high_risk_min_weight"])
    inputs = queue_inputs(proba, y, final, policy, hr_min)
    flagged = final != "allow"
    if not np.array_equal(np.flatnonzero(flagged), inputs["queued"]):
        raise SystemExit(
            f"{item['run']}: flagged rows are not the review queue (not a human-confirmation run?)"
        )
    queue_pos = np.full(len(final), -1)
    queue_pos[inputs["queued"]] = np.arange(len(inputs["queued"]))
    weights = np.array([policy.severity_weights.get(lb, 0.0) for lb in LABELS])
    item["_stream"] = {
        "flagged": flagged,
        "queue_pos": queue_pos,
        "priorities": inputs["priorities"],
        "harm": inputs["harm"],
        "high_risk": inputs["high_risk"],
        "positive": y[inputs["queued"]].any(axis=1),
        "high_risk_all": (y * weights).max(axis=1) >= hr_min,
    }


def shared_stream(items: list[dict[str, Any]], input_rates: list[float]) -> dict[str, Any]:
    """Every run reviews the same incoming comment stream; differences are paired by seed."""
    config = items[0]["config"]
    sim_cfg = SimConfig(
        **{k: config["sim"][k] for k in ("reviewers", "handle_minutes", "horizon_hours")}
    )
    ids = items[0]["pred"]["id"].to_numpy()
    for item in items:
        if not np.array_equal(item["pred"]["id"].to_numpy(), ids):
            raise SystemExit(f"{item['run']}: predictions.csv rows differ from the reference")
        _prepare_stream(item)
    n_rows = len(ids)
    labels = [i["label"] for i in items]
    raw: dict[str, dict[str, dict[str, dict[str, list[float | None]]]]] = {
        lb: {f"{r:.1f}": {s: {} for s in EQUAL_INPUT_STRATEGIES} for r in input_rates}
        for lb in labels
    }
    for rate in input_rates:
        key = f"{rate:.1f}"
        for seed in EQUAL_INPUT_SEEDS:
            rng = np.random.default_rng([seed, int(round(rate * 1000)), 20260923])
            n = int(rng.poisson(rate * sim_cfg.horizon_hours))
            arrival = np.sort(rng.uniform(0.0, sim_cfg.horizon_minutes, size=n))
            row = rng.integers(0, n_rows, size=n)
            for item in items:
                for strategy in EQUAL_INPUT_STRATEGIES:
                    values = _stream_metrics(item, row, arrival, sim_cfg, strategy, seed)
                    bucket = raw[item["label"]][key][strategy]
                    for field, value in values.items():
                        bucket.setdefault(field, []).append(value)
    runs: dict[str, Any] = {}
    for item in items:
        fraction = float(item["_stream"]["flagged"].mean())
        runs[item["label"]] = {
            "queue_fraction": fraction,
            "by_input": {
                key: {
                    "expected_review_load_per_hour": float(key) * fraction,
                    **{
                        s: {f: _mean_std(v) for f, v in raw[item["label"]][key][s].items()}
                        for s in EQUAL_INPUT_STRATEGIES
                    },
                }
                for key in raw[item["label"]]
            },
        }
    reference = labels[0]
    differences: dict[str, Any] = {}
    for label in labels[1:]:
        differences[label] = {
            key: {
                s: {
                    f: _paired(raw[label][key][s][f], raw[reference][key][s][f])
                    for f in STREAM_FIELDS
                }
                for s in EQUAL_INPUT_STRATEGIES
            }
            for key in raw[label]
        }
    return {
        "design": "common random numbers: one incoming comment stream per seed, shared by all runs",
        "reference": reference,
        "input_rates_per_hour": input_rates,
        "seeds": [EQUAL_INPUT_SEEDS[0], EQUAL_INPUT_SEEDS[-1]],
        "n_seeds": len(EQUAL_INPUT_SEEDS),
        "runs": runs,
        "differences_vs_reference": differences,
    }


def _paired(
    a: list[float | None], b: list[float | None]
) -> dict[str, float | int | None]:
    """Pair by seed, using only seeds where both metrics are defined."""
    d = np.asarray(
        [x - y for x, y in zip(a, b, strict=True) if x is not None and y is not None],
        dtype=float,
    )
    return {
        "mean": float(d.mean()) if len(d) else None,
        "se": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None,
        "valid_n": len(d),
    }


def audit_overlap(items: list[dict[str, Any]], audit: Path) -> dict[str, Any]:
    sample = pd.read_csv(audit / "sample.csv", dtype=str, keep_default_na=False)
    filled = sample.human_verdict.str.strip() != ""
    if filled.any() and not filled.all():
        raise SystemExit(f"{audit}/sample.csv: human_verdict is only partly filled")
    verdict_col = "human_verdict" if filled.all() else "preread_verdict"
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
        count = f" (n={value['valid_n']})" if "valid_n" in value else ""
        if value["mean"] is None:
            return "n/a" + count
        return f"{_f(value['mean'], digits)} ± {_f(value['std'], digits)}" + count
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def to_markdown(summary: list[dict[str, Any]], extra: dict[str, Any]) -> str:
    lines = ["# Run comparison (scored test rows)", ""]
    lines.append(
        "| run | model | commit | tier | n | coverage | test precision (95% CI) "
        "| selection precision |"
    )
    lines.append("|---|---|---|---|---:|---:|---|---:|")
    for s in summary:
        commit = (s["git_commit"] or "")[:8] + (" dirty" if s["git_dirty"] else "")
        for tier, t in s["tiers"].items():
            ci = t["test_precision_ci95"] or [None, None]
            model = s["model"].get("model_name") or s["model"].get("analyzer", "word")
            lines.append(
                f"| {s['label']} | {model} | {commit} | {tier} | "
                f"{t['test_n_predicted_positive']} | {t['test_coverage']:.4f} | "
                f"{_f(t['test_precision'])} [{_f(ci[0])}, {_f(ci[1])}] | "
                f"{_f(t['selection_precision'])} |"
            )
    if all("review_workload" in s for s in summary):
        lines += ["", "## Review workload", ""]
        lines.append("| run | needs a reviewer | share of comments | queue precision (95% CI) |")
        lines.append("|---|---:|---:|---|")
        for s in summary:
            w = s["review_workload"]
            ci = w["precision_ci95"] or [None, None]
            lines.append(
                f"| {s['label']} | {w['n_requires_human_review']} | {w['review_fraction']:.4f} | "
                f"{_f(w['precision'])} [{_f(ci[0])}, {_f(ci[1])}] |"
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
            return f"{entry['k']}/{entry['n']} ({_f(entry['share'])})"

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
        aps = " | ".join(_f(s["per_label"][lb]["average_precision"]) for s in summary)
        aucs = " | ".join(_f(s["per_label"][lb]["roc_auc"]) for s in summary)
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
            "## Queue at equal review load (report simulation, "
            f"{summary[0]['queue_equal_review_load']['n_seeds']} seeds, mean ± population SD)",
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
        lines += [
            "",
            f"## Queue at equal comment input (common random numbers, {eq['n_seeds']} seeds)",
            "",
            "One incoming comment stream per seed, shared by every run. Each run reviews what it",
            "flags; high-risk comments it leaves in allow count as not reviewed. Mean ± sample SD.",
            "Undefined metrics remain n/a; n counts defined seeds, or jointly defined pairs.",
            "",
        ]
        lines.append("| run | input/h | strategy | " + " | ".join(STREAM_FIELDS) + " |")
        lines.append("|---|---:|---|" + "---:|" * len(STREAM_FIELDS))
        for label, block in eq["runs"].items():
            for rate, entry in block["by_input"].items():
                for strategy in EQUAL_INPUT_STRATEGIES:
                    cells = " | ".join(_f(entry[strategy][f], 2) for f in STREAM_FIELDS)
                    lines.append(f"| {label} | {float(rate):.0f} | {strategy} | {cells} |")
        for label, block in eq["differences_vs_reference"].items():
            lines += ["", f"Paired difference, {label} minus {eq['reference']} (mean ± SE):", ""]
            lines.append("| input/h | strategy | " + " | ".join(STREAM_FIELDS) + " |")
            lines.append("|---:|---|" + "---:|" * len(STREAM_FIELDS))
            for rate, per in block.items():
                for strategy in EQUAL_INPUT_STRATEGIES:
                    cells = " | ".join(
                        (
                            f"{per[strategy][f]['mean']:+.2f} ± {_f(per[strategy][f]['se'], 2)}"
                            if per[strategy][f]["mean"] is not None else "n/a"
                        )
                        + f" (n={per[strategy][f]['valid_n']})"
                        for f in STREAM_FIELDS
                    )
                    lines.append(f"| {float(rate):.0f} | {strategy} | {cells} |")
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
    labels = [i["label"] for i in items]
    for item in items:
        if labels.count(item["label"]) > 1:
            item["label"] = f"{item['label']}@{item['run'].split('-', 1)[0]}"
    base = items[0]
    for item in items[1:]:
        if item["data"] != base["data"] or item["split"] != base["split"]:
            raise SystemExit(f"{item['run']} does not share data hashes/split with {base['run']}")
    policies = {i["manifest"].get("policy_sha256") for i in items}
    if len(policies) > 1:
        print("warning: runs were made with different policy files", file=sys.stderr)
    summary = [summarize(i) for i in items]
    extra: dict[str, Any] = {
        "policy_sha256": {i["label"]: i["manifest"].get("policy_sha256") for i in items},
        "matched_volume": matched_volume(items),
    }
    if args.audit:
        if not all("auto_action" in i["report"]["tiers"] for i in items):
            raise SystemExit("--audit needs runs with an auto_action tier (round-1 policy)")
        extra["audit_overlap"] = audit_overlap(items, args.audit)
    if args.equal_input:
        reference = base["report"]["simulation"].get("assumptions") or {}
        if "queue_fraction" not in reference:
            raise SystemExit(
                "--equal-input needs human-confirmation runs (report has no queue_fraction)"
            )
        rates = equal_input_loads(reference["queue_fraction"], list(reference["loads_per_hour"]))
        extra["equal_input"] = shared_stream(items, rates)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run_comparison.json").write_text(
        json.dumps({"runs": summary, **extra}, indent=2) + "\n"
    )
    md = to_markdown(summary, extra)
    (args.out / "run_comparison.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
