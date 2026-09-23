#!/usr/bin/env python
"""Draw, then score, the human-audit sample of auto-action false positives.

    python scripts/audit_auto_action_fp.py --run reports/<run>-baseline --n 100
    python scripts/audit_auto_action_fp.py --score analysis/auto_action_fp_audit

A false positive here is an auto_action row whose triggering label(s) are 0
in test_labels.csv (the same definition the auto_action precision uses). The
sample is stratified on two facts the reader will want to compare against
the population: whether the text hits the fixed abuse lexicon, and whether
some OTHER label is true (a "wrong label, right tier" case). Strata sizes
are proportional to the population, seeded, and written next to the sample.

The verdict columns are left empty. Filling them is human work; the audit
explores disagreement with the original labels in a selected error sample.
It does not evaluate whether a comment is worth sending to human review.
Drawing refuses to overwrite a sample that already carries verdicts.

--score reads the filled sample and writes the "human" section of
sample_meta.json: verdict counts, agreement with the pre-read, and the
auto-action precision implied if the audited share of false positives that
the human judged toxic held for all of them. Two rules are reported, because
"borderline" is exactly where the two label conventions disagree:

    implied precision = 1 - n_false_positive * (1 - share_counted_correct) / n_auto_action

These are conditional sensitivity estimates, not measured relabelled precision:
the original true positives are unaudited and assumed to remain correct under
the audit rubric. The approximate 95% interval maps an unweighted Wilson interval
through the same formula; it accounts for neither stratified sampling nor
annotation uncertainty and omits the finite-population correction. Original
benchmark labels and the current policy-v2 gates are unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from review_router.data import LABELS  # noqa: E402
from review_router.metrics import wilson_interval  # noqa: E402

# Same fixed lexicon as record/diagnose.py; a hit is a text proxy, not a label.
LEXICON = (
    "fuck",
    "fucking",
    "fucked",
    "shit",
    "bullshit",
    "asshole",
    "assholes",
    "bitch",
    "bitches",
    "cunt",
    "dick",
    "twat",
    "suck",
    "sucks",
    "idiot",
    "idiots",
    "stupid",
    "moron",
    "morons",
    "loser",
    "losers",
    "faggot",
    "fag",
    "nigger",
    "retard",
    "retarded",
    "bastard",
    "crap",
    "pathetic",
    "dumb",
    "jerk",
    "wanker",
    "scum",
    "pig",
    "pigs",
)
LEX_PATTERN = re.compile(r"\b(" + "|".join(LEXICON) + r")\b", re.IGNORECASE)

VERDICTS = ("toxic", "borderline", "clean")
# Which verdicts count as "the model was right" under each relabelling rule.
RULES: dict[str, tuple[str, ...]] = {
    "toxic_only": ("toxic",),
    "toxic_or_borderline": ("toxic", "borderline"),
}
AUTO_ACTION_FLOOR = 0.99


def implied_precision(n_auto: int, n_fp: int, share_correct: float) -> float:
    """Tier precision if share_correct of the false positives were really positive."""
    return 1.0 - n_fp * (1.0 - share_correct) / n_auto


def rows_needed_for_floor(n_auto: int, n_fp: int, n_sample: int, floor: float) -> int:
    """Audited rows that must be judged toxic for the implied precision to reach floor.

    The inverse of implied_precision: the smallest count whose share makes
    1 - n_fp * (1 - share) / n_auto >= floor.
    """
    share_needed = 1.0 - n_auto * (1.0 - floor) / n_fp
    return max(0, math.ceil(n_sample * share_needed - 1e-9))


def score(sample: pd.DataFrame, n_auto: int, n_fp: int) -> dict[str, Any]:
    """Summarize a fully filled human_verdict column. Raises on blanks or unknown values."""
    verdict = sample["human_verdict"].fillna("").astype(str).str.strip().str.lower()
    unknown = sorted(set(verdict) - set(VERDICTS))
    if unknown:
        raise ValueError(f"human_verdict must be one of {VERDICTS} on every row; found {unknown}")
    n = len(sample)
    out: dict[str, Any] = {
        "n": n,
        "estimate_type": "conditional_sensitivity",
        "assumptions": [
            "Unaudited original true positives remain correct under the audit rubric.",
            "The unweighted audited share is extrapolated to all original false positives.",
        ],
        "interval_method": (
            "Approximate mapped Wilson interval; no stratification, finite-population "
            "correction or annotation uncertainty."
        ),
        "evaluation_scope": (
            "Historical auto-action false positives only; not an independent evaluation "
            "of review worthiness or current human-review precision."
        ),
        "counts": {v: int((verdict == v).sum()) for v in VERDICTS},
        "implied_precision": {},
        "floor": AUTO_ACTION_FLOOR,
        "rows_needed_toxic_for_floor": rows_needed_for_floor(n_auto, n_fp, n, AUTO_ACTION_FLOOR),
    }
    for name, correct in RULES.items():
        k = int(verdict.isin(correct).sum())
        lo, hi = wilson_interval(k, n)
        out["implied_precision"][name] = {
            "rows_counted_correct": k,
            "precision": round(implied_precision(n_auto, n_fp, k / n), 4),
            "precision_ci95": [
                round(implied_precision(n_auto, n_fp, lo), 4),
                round(implied_precision(n_auto, n_fp, hi), 4),
            ],
        }
    if "preread_verdict" in sample.columns:
        pre = sample["preread_verdict"].astype(str).str.strip().str.lower()
        out["agreement_with_preread"] = round(float((pre == verdict).mean()), 4)
        out["preread_to_human"] = {
            f"{a}->{b}": int(((pre == a) & (verdict == b)).sum())
            for a in VERDICTS
            for b in VERDICTS
            if a != b and ((pre == a) & (verdict == b)).any()
        }
    return out


def score_dir(out_dir: Path) -> dict[str, Any]:
    sample = pd.read_csv(out_dir / "sample.csv", dtype=str, keep_default_na=False)
    meta_path = out_dir / "sample_meta.json"
    meta = json.loads(meta_path.read_text())
    human = score(sample, meta["n_auto_action"], meta["n_false_positive"])
    human["protocol"] = meta.get("human", {}).get("protocol")
    meta["human"] = human
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return human


def auto_trigger(
    p: np.ndarray, in_group: np.ndarray, threshold: float, subgroup_threshold: float | None
) -> np.ndarray:
    """Whether a label triggers auto_action, with round-1 model_tier semantics.

    Outside the subgroup the population threshold applies. Inside it, a null
    subgroup threshold means the label cannot trigger at all, and otherwise the
    stricter of the two thresholds applies.
    """
    hit = p >= threshold
    if subgroup_threshold is None:
        return np.asarray(hit & ~in_group)
    return np.asarray(hit & (~in_group | (p >= max(threshold, subgroup_threshold))))


def false_positives(run: Path) -> tuple[pd.DataFrame, int]:
    report_path = run / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        if report.get("evaluate_test") is False:
            raise SystemExit(
                f"{run} is a development run (evaluate_test: false): test rows not scored, "
                "so there are no predictions to audit"
            )
    pred = pd.read_csv(run / "predictions.csv", dtype={"id": str})
    thresholds = json.loads((run / "thresholds.json").read_text())
    if "auto_action" not in thresholds:
        raise SystemExit(
            f"{run} has no auto_action tier (policy v2 routes every flag to human review); "
            "this audit applies to round-1 runs"
        )
    auto = thresholds["auto_action"]
    declared = thresholds.get("subgroup", {}).get("auto_action", {})
    sub = declared.get("identity_term_present", {})
    rows = pred[pred.final_tier == "auto_action"].copy()
    trig_cols = []
    # An absent subgroup imposes no restriction. A declared subgroup with a
    # null per-label threshold blocks that label, matching model_tier().
    in_group = (
        rows.identity_term_present.to_numpy() == 1
        if "identity_term_present" in declared
        else np.zeros(len(rows), dtype=bool)
    )
    for label in LABELS:
        t = auto.get(label)
        if t is None:
            rows[f"trig_{label}"] = False
            continue
        rows[f"trig_{label}"] = auto_trigger(
            rows[f"p_{label}"].to_numpy(), in_group, t, sub.get(label)
        )
        trig_cols.append(label)
    rows["trigger_labels"] = rows.apply(
        lambda r: "+".join(lb for lb in trig_cols if r[f"trig_{lb}"]), axis=1
    )
    rows["trigger_true"] = rows.apply(
        lambda r: any(r[f"trig_{lb}"] and r[f"y_{lb}"] == 1 for lb in trig_cols), axis=1
    )
    rows["other_labels_true"] = rows.apply(
        lambda r: "+".join(lb for lb in LABELS if r[f"y_{lb}"] == 1), axis=1
    )
    fp = rows[~rows.trigger_true].copy()
    assert len(fp) == len(rows) - int(rows.trigger_true.sum())
    return fp, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", type=Path, help="draw a sample from this run directory")
    mode.add_argument("--score", type=Path, help="score the filled sample in this directory")
    parser.add_argument("--data", type=Path, default=Path("data/jigsaw"))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--out", type=Path, default=Path("analysis/auto_action_fp_audit"))
    parser.add_argument("--force", action="store_true", help="overwrite a sample with verdicts")
    args = parser.parse_args()

    if args.score is not None:
        print(json.dumps(score_dir(args.score), indent=2))
        return

    existing = args.out / "sample.csv"
    if existing.exists() and not args.force:
        drawn = pd.read_csv(existing, dtype=str, keep_default_na=False)
        for column in ("human_verdict", "preread_verdict"):
            if column in drawn and (drawn[column].str.strip() != "").any():
                raise SystemExit(f"{existing} already has {column} values; pass --force to redraw")

    fp, n_auto = false_positives(args.run)
    test = pd.read_csv(args.data / "test.csv", dtype={"id": str}).set_index("id")
    fp["text"] = test.loc[fp.id, "comment_text"].to_numpy()
    fp["lexicon_hit"] = fp.text.map(lambda t: bool(LEX_PATTERN.search(t)))
    fp["other_label_true"] = fp.other_labels_true != ""
    fp["stratum"] = (
        np.where(fp.lexicon_hit, "lex", "nolex")
        + "/"
        + np.where(fp.other_label_true, "other", "none")
    )

    rng = np.random.default_rng(args.seed)
    pop = fp.stratum.value_counts()
    # Largest-remainder allocation so strata sum to exactly n.
    raw = pop / pop.sum() * args.n
    alloc = np.floor(raw).astype(int)
    for s in (raw - alloc).sort_values(ascending=False).index[: args.n - int(alloc.sum())]:
        alloc[s] += 1
    parts = []
    for stratum, k in alloc.items():
        pool = fp[fp.stratum == stratum]
        parts.append(pool.iloc[rng.choice(len(pool), size=min(k, len(pool)), replace=False)])
    sample = pd.concat(parts).sort_values("p_toxic", ascending=False)
    sample["human_verdict"] = ""  # one of VERDICTS
    sample["human_note"] = ""

    args.out.mkdir(parents=True, exist_ok=True)
    cols = [
        "id",
        "stratum",
        "trigger_labels",
        "other_labels_true",
        "lexicon_hit",
        "identity_term_present",
        *[f"p_{lb}" for lb in LABELS],
        "text",
        "human_verdict",
        "human_note",
    ]
    sample[cols].to_csv(args.out / "sample.csv", index=False)
    meta = {
        "run": str(args.run),
        "seed": args.seed,
        "n_auto_action": n_auto,
        "n_false_positive": int(len(fp)),
        "precision": round(1 - len(fp) / n_auto, 4),
        "population_strata": {k: int(v) for k, v in pop.items()},
        "sample_strata": {k: int(v) for k, v in sample.stratum.value_counts().items()},
        "n_sample": int(len(sample)),
        "verdicts": list(VERDICTS),
        "definition": "auto_action row whose triggering label(s) are 0 in test_labels.csv",
    }
    (args.out / "sample_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
