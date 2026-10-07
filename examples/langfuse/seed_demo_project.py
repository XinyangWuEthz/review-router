#!/usr/bin/env python
"""Seed a Langfuse project with SYNTHETIC traces and judge scores for the cookbook.

A fresh project has no scores, so the notebook's live path would find no
candidates. This writes `n` traces, each with a short input/output and one
score per signal declared in the notebook, drawn from the same synthetic
generator the notebook uses offline (hidden truth, informative but imperfect
judge). The scores are what a managed LLM-as-a-judge evaluator or a user
feedback hook would otherwise write; nothing here is a result.

Usage:
  export LANGFUSE_PUBLIC_KEY=pk-lf-... LANGFUSE_SECRET_KEY=sk-lf-...
  export LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or the US region
  python examples/langfuse/seed_demo_project.py --n 300 --seed 7
Without credentials the script prints what it would write and exits.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

HERE = Path(__file__).resolve().parent
# Must match WEIGHTS in the notebook; the key is the score name the notebook reads.
WEIGHTS = {
    "safety_violation": 10.0,
    "hallucination": 6.0,
    "negative_feedback": 5.0,
    "off_topic": 2.0,
    "tone": 1.0,
}
BOOLEAN_SIGNALS = {"negative_feedback"}  # written as true/false, like a thumbs-down hook
QUESTIONS = [
    "How do I reset my password?",
    "Can I get a refund for last month's invoice?",
    "The export keeps failing with a timeout.",
    "Does the plan include SSO?",
    "Why was my account suspended?",
]


def _selection_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("budget_selection", HERE / "budget_selection.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["budget_selection"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=300, help="traces to write")
    parser.add_argument("--seed", type=int, default=7, help="generator seed")
    args = parser.parse_args()

    pool = _selection_module().synthetic_pool(args.n, WEIGHTS, seed=args.seed)
    live = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))
    n_scores = int(pool[list(WEIGHTS)].notna().sum().sum())
    print(f"{len(pool)} synthetic traces, {n_scores} scores across {list(WEIGHTS)}")
    if not live:
        print("no LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY set: dry run, nothing written")
        return 0

    from langfuse import Langfuse

    langfuse = Langfuse()
    if not langfuse.auth_check():
        print("Langfuse rejected the credentials", file=sys.stderr)
        return 1
    rng = np.random.default_rng(args.seed)
    for row in pool.itertuples(index=False):
        question = QUESTIONS[int(rng.integers(len(QUESTIONS)))]
        span = langfuse.start_observation(
            name="support-agent",
            as_type="span",
            input={"question": question},
            output={"answer": f"(synthetic answer for {row.trace_id})"},
            metadata={"synthetic": True, "seed": args.seed},
        )
        for name in WEIGHTS:
            value = getattr(row, name)
            if np.isnan(value):
                continue  # the judge did not run on this trace
            if name in BOOLEAN_SIGNALS:
                langfuse.create_score(
                    trace_id=span.trace_id,
                    name=name,
                    value=float(value >= 0.5),
                    data_type="BOOLEAN",
                )
            else:
                langfuse.create_score(
                    trace_id=span.trace_id, name=name, value=float(value), data_type="NUMERIC"
                )
        span.end()
    langfuse.flush()
    print(f"wrote {len(pool)} traces; scores arrive within a few seconds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
