#!/usr/bin/env python
"""Draw the human-audit sample of auto-action false positives from a run.

    python scripts/audit_auto_action_fp.py --run reports/<run>-baseline --n 100

A false positive here is an auto_action row whose triggering label(s) are 0
in test_labels.csv (the same definition the auto_action precision uses). The
sample is stratified on two facts the reader will want to compare against
the population: whether the text hits the fixed abuse lexicon, and whether
some OTHER label is true (a "wrong label, right tier" case). Strata sizes
are proportional to the population, seeded, and written next to the sample.

The verdict columns are left empty. Filling them is human work; the audit
answers whether 0.904 measures model error or a looser labelling convention
on the scored test rows.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from review_router.data import LABELS  # noqa: E402

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


def false_positives(run: Path) -> pd.DataFrame:
    pred = pd.read_csv(run / "predictions.csv", dtype={"id": str})
    thresholds = json.loads((run / "thresholds.json").read_text())
    auto = thresholds["auto_action"]
    sub = thresholds.get("subgroup", {}).get("auto_action", {}).get("identity_term_present", {})
    rows = pred[pred.final_tier == "auto_action"].copy()
    trig_cols = []
    for label in LABELS:
        t = auto.get(label)
        t_sub = sub.get(label)
        if t is None:
            rows[f"trig_{label}"] = False
            continue
        thr = np.where(
            (rows.identity_term_present == 1) & (t_sub is not None),
            t_sub if t_sub is not None else t,
            t,
        )
        rows[f"trig_{label}"] = rows[f"p_{label}"].to_numpy() >= thr
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
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/jigsaw"))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--out", type=Path, default=Path("analysis/auto_action_fp_audit"))
    args = parser.parse_args()

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
        "sample_strata": {k: int(v) for k, v in alloc.items()},
        "n_sample": int(len(sample)),
        "verdicts": list(VERDICTS),
        "definition": "auto_action row whose triggering label(s) are 0 in test_labels.csv",
    }
    (args.out / "sample_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
