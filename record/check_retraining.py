#!/usr/bin/env python
"""Compare two saved runs without retraining or recomputing their predictions.

From the repository root, after downloading both CI artifact directories:

    PYTHONPATH=. python record/check_retraining.py \\
        reports/ci-35856953771 reports/ci-35857528118 \\
        --ci-run-ids 35856953771 35857528118 \\
        --train-csv data/jigsaw/train.csv --trust-model-pickles

Only use --trust-model-pickles for artifacts from a trusted run: pickle loading
can execute code. The inspection environment is recorded because sklearn does
not support loading estimators across versions. This script inspects saved
vocabularies and counts their terms; it never invokes the saved classifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import subprocess
import warnings
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from review_router.data import LABELS, file_sha256

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _source_hash(commit: str, path: str) -> str:
    content = subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    return hashlib.sha256(content).hexdigest()


def compare(first: Path, second: Path, ci_ids: list[int], train_csv: Path) -> dict[str, Any]:
    """Inspect trusted run artifacts; callers must authorize pickle loading."""
    dirs = (first, second)
    manifests = [_read(path / "manifest.json") for path in dirs]
    reports = [_read(path / "report.json") for path in dirs]
    if any(file_sha256(train_csv) != m["data_sha256"]["train"] for m in manifests):
        raise ValueError("training CSV does not match both recorded data hashes")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        models = [pickle.loads((path / "model.pkl").read_bytes()) for path in dirs]
    vocabularies = [model.vectorizer.vocabulary_ for model in models]
    vocab_sets = [set(vocab) for vocab in vocabularies]
    common = vocab_sets[0] & vocab_sets[1]
    exclusive = [sorted(vocab_sets[i] - vocab_sets[1 - i]) for i in range(2)]
    swapped = set(exclusive[0]) | set(exclusive[1])
    splits = pd.read_csv(second / "splits.csv", dtype={"id": str})
    train_ids = set(splits.loc[splits["split"] == "train", "id"])
    frame = pd.read_csv(train_csv, dtype={"id": str}, usecols=["id", "comment_text"])
    texts = frame.loc[frame["id"].isin(train_ids), "comment_text"].fillna("").astype(str)
    analyze = models[1].vectorizer.build_analyzer()
    tf: Counter[str] = Counter()
    df: Counter[str] = Counter()
    for text in texts:
        counts = Counter(term for term in analyze(text) if term in swapped)
        tf.update(counts)
        df.update(counts.keys())
    common_sorted = sorted(common)
    idfs = [
        model.vectorizer.idf_[[vocab[word] for word in common_sorted]]
        for model, vocab in zip(models, vocabularies, strict=True)
    ]
    predictions = [pd.read_csv(path / "predictions.csv", dtype={"id": str}) for path in dirs]
    a, b = predictions
    if not a["id"].equals(b["id"]):
        raise ValueError("saved prediction IDs/order differ")
    label_cols = [f"y_{label}" for label in LABELS]
    if not a[label_cols].equals(b[label_cols]):
        raise ValueError("saved test labels differ")
    score_cols = [f"p_{label}" for label in LABELS]
    score_diff = np.abs(a[score_cols].to_numpy() - b[score_cols].to_numpy())
    admitted = [pred["final_tier"] != "allow" for pred in predictions]
    changed = admitted[0] != admitted[1]
    runs = []
    for path, manifest, report, ci_id in zip(dirs, manifests, reports, ci_ids, strict=True):
        runs.append(
            {
                "ci_run_id": ci_id,
                "ci_url": f"https://github.com/XinyangWuEthz/review-router/actions/runs/{ci_id}",
                "run_id": manifest["run_id"],
                "git_commit": manifest["git_commit"],
                "git_dirty": manifest["git_dirty"],
                "data_sha256": manifest["data_sha256"],
                "config_sha256": manifest["config_sha256"],
                "policy_sha256": manifest["policy_sha256"],
                "splits_sha256": file_sha256(path / "splits.csv"),
                "model_sha256": file_sha256(path / "model.pkl"),
                "predictions_sha256": file_sha256(path / "predictions.csv"),
                "training_source_sha256": {
                    file: _source_hash(manifest["git_commit"], file)
                    for file in ("review_router/model.py", "review_router/data.py")
                },
                "seed": manifest["seed"],
                "dependencies": manifest["dependencies"],
                "model_config": manifest["config"]["model"],
                "review_workload": report["review_workload"],
                "overload": {
                    strategy: {
                        metric: report["simulation"]["summary"][f"{strategy}@180"][metric]
                        for metric in (
                            "high_risk_arrived",
                            "high_risk_handled",
                            "harm_per_reviewer_hour",
                        )
                    }
                    for strategy in ("severity", "priority")
                },
            }
        )
    same_inputs = {
        key: runs[0][key] == runs[1][key]
        for key in (
            "data_sha256",
            "config_sha256",
            "splits_sha256",
            "seed",
            "dependencies",
            "model_config",
            "training_source_sha256",
        )
    }
    tf_counts = dict(sorted(Counter(tf[t] for t in swapped).items()))
    df_counts = dict(sorted(Counter(df[t] for t in swapped).items()))
    differing_inputs = [key for key, matches in same_inputs.items() if not matches]
    conclusions = [
        "All checked data, split, seed, model-setting, dependency and training-source inputs match."
        if not differing_inputs
        else f"Recorded inputs differ: {', '.join(differing_inputs)}; cross-run changes are confounded."
    ]
    if swapped:
        conclusions.append(
            f"The vocabularies share {len(common)} terms and have "
            f"{len(exclusive[0])} and {len(exclusive[1])} exclusive terms."
        )
        at_cap = all(
            len(vocab) == model.config.max_features
            for vocab, model in zip(vocab_sets, models, strict=True)
        )
        if at_cap and len(tf_counts) == len(df_counts) == 1:
            conclusions.append(
                f"Every swapped term has raw term frequency {next(iter(tf_counts))} and "
                f"document frequency {next(iter(df_counts))}, supporting tied-term selection "
                "at the feature cap. Platform-dependent ordering of ties is a suspected "
                "mechanism; the specific CPU or library cause has not been established."
            )
        else:
            conclusions.append(
                "Swapped-term frequency counts are recorded; they do not isolate a single "
                "tied boundary or establish the cause of the vocabulary difference."
            )
    else:
        conclusions.append("The saved vocabulary term sets match.")
    if (score_diff > 0).any():
        conclusions.append("Retraining did not reproduce exact saved probability values.")
        if not differing_inputs:
            conclusions.append(
                "Because the training implementation is unchanged, this exposes a "
                "pre-existing reproducibility limitation."
            )
    else:
        conclusions.append("The saved probability values match exactly.")
    if changed.any():
        conclusions.append(
            "Changes in admitted row identities can also change which comments "
            "index-based simulation sampling draws."
        )
    conclusions.append(
        "Compare orderings and cumulative/segment bands within the same saved run; "
        "cross-run metric differences alone do not isolate a policy effect."
    )
    return {
        "runs": runs,
        "same_inputs": same_inputs,
        "vocabulary": {
            "sizes": [len(vocab) for vocab in vocab_sets],
            "common_terms": len(common),
            "exclusive_counts": [len(terms) for terms in exclusive],
            "exclusive_terms": exclusive,
            "common_terms_max_abs_idf_difference": float(np.max(np.abs(idfs[0] - idfs[1]))),
            "training_rows_inspected": len(texts),
            "swapped_term_frequency_counts": tf_counts,
            "swapped_document_frequency_counts": df_counts,
        },
        "test_prediction_changes": {
            "n_rows": len(a),
            "max_abs_probability_difference": float(score_diff.max()),
            "mean_abs_probability_difference": float(score_diff.mean()),
            "n_rows_with_any_exact_score_difference": int((score_diff > 0).any(axis=1).sum()),
            "n_rows_changing_final_band": int((a["final_tier"] != b["final_tier"]).sum()),
            "n_rows_changing_admission": int(changed.sum()),
            "admission_changes": [
                {
                    "id": str(a.loc[i, "id"]),
                    "first_tier": a.loc[i, "final_tier"],
                    "second_tier": b.loc[i, "final_tier"],
                }
                for i in a.index[changed]
            ],
        },
        "inspection_environment": {
            "python": platform.python_version(),
            "dependencies": {name: version(name) for name in ("numpy", "scikit-learn", "pandas")},
            "pickle_version_warnings": sorted({str(w.message) for w in caught}),
            "note": "Only trusted project CI pickles were loaded. Saved vocabularies/IDFs were "
            "inspected; classifiers were not invoked. Cross-version estimator loading is "
            "unsupported by sklearn; all numerical score comparisons use saved CSV predictions.",
        },
        "conclusion": " ".join(conclusions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_run", type=Path)
    parser.add_argument("second_run", type=Path)
    parser.add_argument("--ci-run-ids", type=int, nargs=2, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--trust-model-pickles", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "record" / "retraining_check.json")
    args = parser.parse_args()
    if not args.trust_model_pickles:
        parser.error("--trust-model-pickles is required to inspect trusted CI model artifacts")
    result = compare(args.first_run, args.second_run, args.ci_run_ids, args.train_csv)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
