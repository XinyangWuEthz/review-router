#!/usr/bin/env python
"""Compare round-1 auto-action precision across selection, cross-validation and test rows.

    python record/diagnose.py reports/<run_id>
    python record/diagnose.py --render-only

Reads the run's model.pkl, splits.csv, thresholds.json, predictions.csv and manifest.json,
recomputes everything from the corpus, and writes results.json plus figures/*.png next to
this file. render.py then turns results.json and runs.json into the record pages. Nothing here
changes the deployed policy using scored test rows. Post-hoc scans on those rows
describe this fitted model; they do not establish a deployment threshold or a causal explanation.
"""

from __future__ import annotations

import argparse
import html
import json
import pickle
import re
import sys
from contextlib import chdir
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from review_router.data import LABELS, file_sha256, load_corpus  # noqa: E402
from review_router.metrics import wilson_interval  # noqa: E402
from review_router.pipeline import load_config  # noqa: E402
from review_router.policy import load_policy  # noqa: E402
from review_router.thresholds import choose_threshold  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
AUTO_LABELS = ("toxic", "obscene", "insult")
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8983"
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e1",
        "grid.linewidth": 0.8,
        "axes.edgecolor": "#c9c8c3",
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
    }
)


# ----------------------------------------------------------------------------- helpers
def choose_threshold_lb(y: np.ndarray, p: np.ndarray, floor: float, min_pos: int) -> float | None:
    """Select by the pointwise Wilson 95% lower bound; scanning has no joint 95% guarantee."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if y.sum() == 0:
        return None
    order = np.argsort(-p, kind="stable")
    ps, cum = p[order], np.cumsum(y[order])
    counts = np.arange(1, len(p) + 1)
    last = np.flatnonzero(np.diff(ps, append=-np.inf) != 0)
    best: float | None = None
    for k in last:
        n = int(counts[k])
        if n >= min_pos and wilson_interval(int(cum[k]), n)[0] >= floor:
            best = float(ps[k])
    return best


RULES: dict[str, dict[str, Any]] = {
    "point_min30": {"label": "point precision >= 0.99, n >= 30 (current)", "lb": False, "min": 30},
    "lb_min30": {"label": "Wilson lower bound >= 0.99, n >= 30", "lb": True, "min": 30},
    "point_min100": {"label": "point precision >= 0.99, n >= 100", "lb": False, "min": 100},
    "point_min300": {"label": "point precision >= 0.99, n >= 300", "lb": False, "min": 300},
    "lb_min100": {"label": "Wilson lower bound >= 0.99, n >= 100", "lb": True, "min": 100},
}


def select(
    rule: dict[str, Any], Y: np.ndarray, P: np.ndarray, floor: float
) -> dict[str, float | None]:
    fn = choose_threshold_lb if rule["lb"] else choose_threshold
    return {label: fn(Y[:, j], P[:, j], floor, rule["min"]) for j, label in enumerate(LABELS)}


def tier_masks(
    P: np.ndarray, Y: np.ndarray, thr: dict[str, float | None]
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(P.shape, dtype=bool)
    for j, label in enumerate(LABELS):
        if thr.get(label) is not None:
            mask[:, j] = P[:, j] >= thr[label]
    hit = mask.any(axis=1)
    correct = (mask & (Y == 1)).any(axis=1)
    return hit, correct


def summarise(hit: np.ndarray, correct: np.ndarray) -> dict[str, Any]:
    n = int(hit.sum())
    tp = int((hit & correct).sum())
    return {
        "n": n,
        "tp": tp,
        "precision": (tp / n) if n else None,
        "ci95": list(wilson_interval(tp, n)) if n else None,
        "coverage": n / len(hit),
    }


def per_label(P: np.ndarray, Y: np.ndarray, thr: dict[str, float | None]) -> dict[str, Any]:
    out = {}
    for j, label in enumerate(LABELS):
        t = thr.get(label)
        if t is None:
            out[label] = {"threshold": None, "n": 0, "tp": 0, "precision": None, "ci95": None}
            continue
        hit = P[:, j] >= t
        out[label] = {"threshold": t, **summarise(hit, Y[:, j] == 1)}
    return out


def cv(
    rule: dict[str, Any],
    Y: np.ndarray,
    P: np.ndarray,
    floor: float,
    seeds: tuple[int, ...],
    k: int = 5,
) -> dict[str, Any]:
    """Choose on k-1 folds, evaluate on the held-out fold, pool the held-out rows; repeat per seed."""
    n = len(Y)
    tier_runs, label_runs, thr_runs, fold_count_runs = [], [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        fold = rng.permutation(n) % k
        hit_all = np.zeros(n, dtype=bool)
        correct_all = np.zeros(n, dtype=bool)
        label_hit = np.zeros((n, len(LABELS)), dtype=bool)
        fold_thr = []
        held_counts: dict[str, list[int]] = {label: [] for label in LABELS}
        sel_counts: dict[str, list[int]] = {label: [] for label in LABELS}
        for f in range(k):
            train, hold = fold != f, fold == f
            thr = select(rule, Y[train], P[train], floor)
            fold_thr.append(thr)
            h, c = tier_masks(P[hold], Y[hold], thr)
            hit_all[hold], correct_all[hold] = h, c
            for j, label in enumerate(LABELS):
                if thr[label] is not None:
                    label_hit[hold, j] = P[hold, j] >= thr[label]
                    held_counts[label].append(int((P[hold, j] >= thr[label]).sum()))
                    sel_counts[label].append(int((P[train, j] >= thr[label]).sum()))
        tier_runs.append(summarise(hit_all, correct_all))
        label_runs.append(
            {label: summarise(label_hit[:, j], Y[:, j] == 1) for j, label in enumerate(LABELS)}
        )
        thr_runs.append(fold_thr)
        fold_count_runs.append({"held_out": held_counts, "selection": sel_counts})

    def mean_of(vals: list[float | None]) -> float | None:
        v = [x for x in vals if x is not None]
        return float(np.mean(v)) if v else None

    return {
        "tier": {
            "precision_mean": mean_of([r["precision"] for r in tier_runs]),
            "precision_per_seed": [r["precision"] for r in tier_runs],
            "coverage_mean": float(np.mean([r["coverage"] for r in tier_runs])),
            "n_mean": float(np.mean([r["n"] for r in tier_runs])),
        },
        "per_label": {
            label: {
                "precision_mean": mean_of([r[label]["precision"] for r in label_runs]),
                "n_mean": float(np.mean([r[label]["n"] for r in label_runs])),
                "threshold_fold_values": [t[label] for run in thr_runs for t in run],
                "held_out_fold_counts": [
                    c for run in fold_count_runs for c in run["held_out"][label]
                ],
                "selection_fold_counts": [
                    c for run in fold_count_runs for c in run["selection"][label]
                ],
            }
            for label in LABELS
        },
    }


def reliability(p: np.ndarray, y: np.ndarray, edges: list[float]) -> list[dict[str, Any]]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        n = int(m.sum())
        rows.append(
            {
                "bin": f"[{lo:g}, {hi:g}{')' if hi < 1.0 else ']'}",
                "n": n,
                "mean_p": float(p[m].mean()) if n else None,
                "observed": float(y[m].mean()) if n else None,
                "ci95": list(wilson_interval(int(y[m].sum()), n)) if n else None,
            }
        )
    return rows


def best_precision_scan(p: np.ndarray, y: np.ndarray, min_pos: int) -> dict[str, Any]:
    """Highest precision over EVERY distinct score with at least min_pos rows above it."""
    order = np.argsort(-p, kind="stable")
    ps, cum = p[order], np.cumsum(y[order])
    counts = np.arange(1, len(p) + 1)
    last = np.flatnonzero(np.diff(ps, append=-np.inf) != 0)
    best = {"threshold": None, "n": 0, "precision": -1.0}
    for k in last:
        n = int(counts[k])
        if n < min_pos:
            continue
        prec = float(cum[k] / n)
        if prec > best["precision"]:
            best = {"threshold": float(ps[k]), "n": n, "precision": prec}
    best["n_distinct_thresholds_checked"] = int(sum(1 for k in last if counts[k] >= min_pos))
    return best


def precision_curve(p: np.ndarray, y: np.ndarray, grid: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for t in grid:
        hit = p >= t
        n = int(hit.sum())
        out.append(
            {"threshold": float(t), "n": n, "precision": float(y[hit].mean()) if n else None}
        )
    return out


TOKEN = re.compile(r"(?u)\b\w\w+\b")

# Fixed profanity, slur and insult terms for a reproducible text diagnostic.
# A match depends on context and does not establish abuse or an incorrect label.
# These terms may also occur among the classifier's features.
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


def lexicon_share(texts: list[str], mask: np.ndarray) -> dict[str, Any]:
    idx = np.flatnonzero(mask)
    hits = sum(1 for i in idx if LEX_PATTERN.search(texts[i]))
    n = int(len(idx))
    return {
        "n": n,
        "n_with_lexicon_hit": int(hits),
        "share": (hits / n) if n else None,
        "ci95": list(wilson_interval(int(hits), n)) if n else None,
    }


def oov_rate(text: str, vocab: set[str]) -> float | None:
    """Approximate token coverage; unlike TF-IDF preprocessing, accents are not stripped."""
    toks = [w.lower() for w in TOKEN.findall(text)]
    if not toks:
        return None
    return 1 - sum(1 for w in toks if w in vocab) / len(toks)


def text_stats(
    texts: list[str], vocab: set[str], rng: np.random.Generator, sample: int = 15000
) -> dict[str, Any]:
    idx = rng.choice(len(texts), size=min(sample, len(texts)), replace=False)
    lengths, oov, empty = [], [], 0
    for i in idx:
        t = texts[i]
        lengths.append(len(t))
        toks = [w.lower() for w in TOKEN.findall(t)]
        if not toks:
            empty += 1
            continue
        in_vocab = sum(1 for w in toks if w in vocab)
        oov.append(1 - in_vocab / len(toks))
        if in_vocab == 0:
            empty += 1
    lengths_a = np.array(lengths)
    return {
        "sample": int(len(idx)),
        "chars_median": float(np.median(lengths_a)),
        "chars_p90": float(np.percentile(lengths_a, 90)),
        "chars_mean": float(lengths_a.mean()),
        "oov_token_rate_mean": float(np.mean(oov)) if oov else None,
        "oov_token_rate_p90": float(np.percentile(oov, 90)) if oov else None,
        "share_no_vocab_token": empty / len(idx),
        "length_hist": np.histogram(np.clip(lengths_a, 0, 2000), bins=40, range=(0, 2000))[
            0
        ].tolist(),
    }


# ----------------------------------------------------------------------------- main
def main(run_dir: Path) -> None:
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    for name, hash_key in (("config.yaml", "config_sha256"), ("policy.yaml", "policy_sha256")):
        if file_sha256(run_dir / name) != manifest[hash_key]:
            raise ValueError(f"{name} does not match the saved run manifest")
    # Saved baseline paths refer to the project root, independently of the caller's cwd.
    with chdir(ROOT):
        config = load_config(run_dir / "config.yaml")
    policy = load_policy(run_dir / "policy.yaml")
    floor = policy.tier_precision_floors["auto_action"]
    report = json.loads((run_dir / "report.json").read_text())
    thresholds = json.loads((run_dir / "thresholds.json").read_text())
    with (run_dir / "model.pkl").open("rb") as fh:
        model = pickle.load(fh)
    corpus = load_corpus(config.data_dir, config.split_fractions, config.seed)
    if corpus.file_hashes != manifest["data_sha256"]:
        raise ValueError("corpus SHA256 hashes do not match the saved run manifest")
    splits = pd.read_csv(run_dir / "splits.csv", dtype={"id": str})
    if not np.array_equal(
        splits[["id", "split"]].to_numpy(), corpus.train[["id", "split"]].to_numpy()
    ):
        raise ValueError("saved split ids or assignments do not match the corpus")

    thresh = corpus.train[corpus.train["split"] == "thresh"]
    x_thresh = thresh["comment_text"].astype(str).tolist()
    Y_thresh = thresh[list(LABELS)].to_numpy(dtype=int)
    P_thresh = model.predict_proba(x_thresh)
    x_test = corpus.test["comment_text"].astype(str).tolist()
    Y_test = corpus.test[list(LABELS)].to_numpy(dtype=int)
    P_test = model.predict_proba(x_test)
    pred = pd.read_csv(run_dir / "predictions.csv", dtype={"id": str})
    if not np.array_equal(pred["id"].to_numpy(), corpus.test["id"].to_numpy()):
        raise ValueError("saved prediction ids do not match the corpus")
    if not np.array_equal(pred[[f"y_{l}" for l in LABELS]].to_numpy(), Y_test):
        raise ValueError("saved prediction labels do not match the corpus")
    assert np.allclose(pred[[f"p_{l}" for l in LABELS]].to_numpy(), P_test, atol=1e-9), (
        "predictions mismatch"
    )
    auto_thr = {label: thresholds["auto_action"][label] for label in LABELS}
    results: dict[str, Any] = {
        "provenance": {
            "run_id": manifest["run_id"],
            "git_commit": manifest["git_commit"],
            "git_dirty": manifest["git_dirty"],
            "data_sha256": manifest["data_sha256"],
            "seed": manifest["seed"],
            "floor": floor,
            "min_predicted_positives": policy.gates["min_predicted_positives_for_precision"],
            "auto_thresholds": auto_thr,
            "rows": {"thresh": int(len(thresh)), "test": int(len(corpus.test))},
        }
    }

    # ---- step 1: reproduce the red light (population thresholds, model only; and the report's tiers)
    hit_s, cor_s = tier_masks(P_thresh, Y_thresh, auto_thr)
    hit_t, cor_t = tier_masks(P_test, Y_test, auto_thr)
    results["step1"] = {
        "tier_in_sample": summarise(hit_s, cor_s),
        "tier_test": summarise(hit_t, cor_t),
        "per_label_in_sample": per_label(P_thresh, Y_thresh, auto_thr),
        "per_label_test": per_label(P_test, Y_test, auto_thr),
        "report_tiers": {
            "selection_final": report["threshold_selection"]["tiers"]["auto_action"]["precision"],
            "test_final": report["tiers"]["auto_action"]["precision"],
            "test_final_n": report["tiers"]["auto_action"]["n_predicted_positive"],
        },
        "share_of_auto_rows_by_label_test": {
            label: int((P_test[:, j] >= auto_thr[label]).sum())
            if auto_thr[label] is not None
            else 0
            for j, label in enumerate(LABELS)
        },
    }

    # ---- step 2: held-out threshold evaluation inside the selection split
    seeds = (0, 1, 2)
    cv_point = cv(RULES["point_min30"], Y_thresh, P_thresh, floor, seeds)
    results["step2"] = {"seeds": list(seeds), "folds": 5, "cv": cv_point}

    # ---- step 3: observed differences between splits
    edges = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0]
    rel = {}
    for j, label in enumerate(LABELS):
        if label not in AUTO_LABELS:
            continue
        rel[label] = {
            "thresh": reliability(P_thresh[:, j], Y_thresh[:, j], edges),
            "test": reliability(P_test[:, j], Y_test[:, j], edges),
        }
    grid = np.concatenate([np.linspace(0.90, 0.99, 19), np.linspace(0.991, 0.9999, 30)])
    jt = LABELS.index("toxic")
    curves = {
        "thresh": precision_curve(P_thresh[:, jt], Y_thresh[:, jt], grid),
        "test": precision_curve(P_test[:, jt], Y_test[:, jt], grid),
    }
    best_test = best_precision_scan(P_test[:, jt], Y_test[:, jt], 30)
    rng = np.random.default_rng(7)
    vocab = set(model.vectorizer.vocabulary_)
    train_texts = (
        corpus.train[corpus.train["split"] == "train"]["comment_text"].astype(str).tolist()
    )
    results["step3"] = {
        "ap": {
            label: {
                "selection": report["threshold_selection"]["per_label"][label]["average_precision"],
                "test": report["per_label"][label]["average_precision"],
            }
            for label in LABELS
        },
        "prevalence": {
            label: {"thresh": float(Y_thresh[:, j].mean()), "test": float(Y_test[:, j].mean())}
            for j, label in enumerate(LABELS)
        },
        "reliability_bins": edges,
        "reliability": rel,
        "precision_curve_toxic": curves,
        "best_toxic_precision_on_test_n30": best_test,
        "text": {
            "train": text_stats(train_texts, vocab, rng),
            "thresh": text_stats(x_thresh, vocab, rng),
            "test": text_stats(x_test, vocab, rng),
        },
    }

    # ---- step 4: what the test "false positives" look like (toxic trigger)
    t_tox = auto_thr["toxic"]
    fp_mask = (P_test[:, jt] >= t_tox) & (Y_test[:, jt] == 0)
    other_true = (Y_test[fp_mask][:, [j for j in range(len(LABELS)) if j != jt]] == 1).any(axis=1)
    fp_idx = np.flatnonzero(fp_mask)
    hit_flags = np.array([bool(LEX_PATTERN.search(x_test[i])) for i in fp_idx])
    n_examples = min(12, len(fp_idx))
    n_hit_examples = int(round(n_examples * hit_flags.mean())) if len(fp_idx) else 0
    rng2 = np.random.default_rng(11)
    # Stratified by lexicon hit so the sample carries the population share.
    sample_idx = np.concatenate(
        [
            rng2.choice(
                fp_idx[hit_flags], size=min(n_hit_examples, int(hit_flags.sum())), replace=False
            ),
            rng2.choice(
                fp_idx[~hit_flags],
                size=min(n_examples - n_hit_examples, int((~hit_flags).sum())),
                replace=False,
            ),
        ]
    )
    examples = []
    for i in sorted(sample_idx, key=lambda k: -P_test[k, jt]):
        labels_true = [label for j, label in enumerate(LABELS) if Y_test[i, j] == 1]
        examples.append(
            {
                "id": corpus.test["id"].iloc[i],
                "p_toxic": float(P_test[i, jt]),
                "other_labels_true": labels_true,
                "lexicon_hit": bool(LEX_PATTERN.search(x_test[i])),
                "text": html.escape(x_test[i][:240].replace("\n", " ")),
            }
        )

    # P(toxic = 1 | lexicon hit) per split: is lexicon-bearing text labelled toxic less often on test?
    def p_toxic_given_hit(texts: list[str], Y: np.ndarray) -> dict[str, Any]:
        hit = np.array([bool(LEX_PATTERN.search(t)) for t in texts])
        n = int(hit.sum())
        pos = int((hit & (Y[:, jt] == 1)).sum())
        return {
            "n_with_hit": n,
            "n_toxic": pos,
            "share": pos / n if n else None,
            "ci95": list(wilson_interval(pos, n)) if n else None,
        }

    # OOV rate by group on test (the vocabulary is the model's):
    vocab_ = set(model.vectorizer.vocabulary_)

    def mean_oov(idx: np.ndarray) -> float | None:
        vals = [v for v in (oov_rate(x_test[i], vocab_) for i in idx) if v is not None]
        return float(np.mean(vals)) if vals else None

    tp_idx = np.flatnonzero((P_test[:, jt] >= t_tox) & (Y_test[:, jt] == 1))
    oov_by_group = {
        "test_fp_lexicon_hit": mean_oov(fp_idx[hit_flags]),
        "test_fp_no_lexicon_hit": mean_oov(fp_idx[~hit_flags]),
        "test_true_positives": mean_oov(tp_idx),
    }
    per_other = {
        label: int((Y_test[fp_mask][:, j] == 1).sum())
        for j, label in enumerate(LABELS)
        if label != "toxic"
    }
    tp_mask = (P_test[:, jt] >= t_tox) & (Y_test[:, jt] == 1)
    neg_mask = Y_test[:, jt] == 0
    sel_fp = (P_thresh[:, jt] >= t_tox) & (Y_thresh[:, jt] == 0)
    sel_tp = (P_thresh[:, jt] >= t_tox) & (Y_thresh[:, jt] == 1)
    lexicon = {
        "test_false_positives": lexicon_share(x_test, fp_mask),
        "test_true_positives": lexicon_share(x_test, tp_mask),
        "test_all_toxic_zero": lexicon_share(x_test, neg_mask),
        "selection_false_positives": lexicon_share(x_thresh, sel_fp),
        "selection_true_positives": lexicon_share(x_thresh, sel_tp),
        "selection_all_toxic_zero": lexicon_share(x_thresh, Y_thresh[:, jt] == 0),
        "lexicon": list(LEXICON),
    }
    results["step4"] = {
        "lexicon": lexicon,
        "p_toxic_given_lexicon_hit": {
            "selection": p_toxic_given_hit(x_thresh, Y_thresh),
            "test": p_toxic_given_hit(x_test, Y_test),
        },
        "oov_by_group": oov_by_group,
        "example_seed": 11,
        "n_examples_with_lexicon_hit": int(sum(e["lexicon_hit"] for e in examples)),
        "toxic_threshold": t_tox,
        "n_toxic_auto_test": int((P_test[:, jt] >= t_tox).sum()),
        "n_false_positive": int(fp_mask.sum()),
        "n_fp_with_other_label_true": int(other_true.sum()),
        "fp_other_label_counts": per_other,
        "examples": examples,
        "n_fp_all_labels_zero": int((~other_true).sum()),
    }

    # ---- step 5: candidate selection rules (CV inside thresh, test as a readout)
    cands = {}
    for key, rule in RULES.items():
        thr_full = select(rule, Y_thresh, P_thresh, floor)
        h_s, c_s = tier_masks(P_thresh, Y_thresh, thr_full)
        h_t, c_t = tier_masks(P_test, Y_test, thr_full)
        cands[key] = {
            "label": rule["label"],
            "thresholds": thr_full,
            "in_sample": summarise(h_s, c_s),
            "cv": cv(rule, Y_thresh, P_thresh, floor, seeds)["tier"],
            "test_readout": summarise(h_t, c_t),
            "test_per_label": per_label(P_test, Y_test, thr_full),
        }
    results["step5"] = {"rules": cands}

    # ---- step 6: descriptive differences, not an identified causal decomposition
    # Legacy JSON keys "optimism" and "shift" are retained for existing renderers.
    s1, s2 = results["step1"], results["step2"]["cv"]["tier"]
    results["step6"] = {
        "in_sample": s1["tier_in_sample"]["precision"],
        "held_out_cv": s2["precision_mean"],
        "test": s1["tier_test"]["precision"],
        "optimism": s1["tier_in_sample"]["precision"] - s2["precision_mean"],
        "shift": s2["precision_mean"] - s1["tier_test"]["precision"],
        "best_rule_by_cv": max(cands, key=lambda k: cands[k]["cv"]["precision_mean"] or 0),
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2, default=float))
    figures(results)
    print(HERE / "results.json")


# ----------------------------------------------------------------------------- figures
def _bar_end(ax: Any, bars: Any, fmt: str = "{:.3f}", dy: float = 2, fontsize: int = 8) -> None:
    for b in bars:
        h = b.get_height()
        if np.isnan(h):
            continue
        ax.annotate(
            fmt.format(h),
            (b.get_x() + b.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#52514e",
            xytext=(0, dy),
            textcoords="offset points",
        )


def _legend_below(ax: Any, ncol: int = 3) -> None:
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=ncol, fontsize=8)


def figures(r: dict[str, Any]) -> None:
    FIG.mkdir(exist_ok=True)
    floor = r["provenance"]["floor"]
    # fig1: precision at the frozen thresholds, selection vs test, per label + tier
    s1 = r["step1"]
    cats = [*AUTO_LABELS, "population\nauto tier"]
    sel = [s1["per_label_in_sample"][l]["precision"] for l in AUTO_LABELS] + [
        s1["tier_in_sample"]["precision"]
    ]
    tst = [s1["per_label_test"][l]["precision"] for l in AUTO_LABELS] + [
        s1["tier_test"]["precision"]
    ]
    ns = [s1["per_label_test"][l]["n"] for l in AUTO_LABELS] + [s1["tier_test"]["n"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(cats))
    w = 0.34
    b1 = ax.bar(x - w / 2 - 0.01, sel, w, color=BLUE, label="selection split (in-sample)")
    b2 = ax.bar(x + w / 2 + 0.01, tst, w, color=ORANGE, label="scored test rows")
    _bar_end(ax, b1)
    _bar_end(ax, b2)
    ax.axhline(floor, color=GRAY, lw=1, ls="-")
    ax.text(
        len(cats) - 0.5, floor + 0.002, f"floor {floor}", ha="right", fontsize=8, color="#52514e"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n test = {n})" for c, n in zip(cats, ns)])
    ax.set_ylim(0.85, 1.005)
    ax.set_ylabel("precision at frozen population thresholds")
    _legend_below(ax, 2)
    ax.set_title("Gap 1: the same thresholds, two splits")
    fig.tight_layout()
    fig.savefig(FIG / "gap1_precision_by_split.png")
    plt.close(fig)

    # fig2: in-sample vs held-out CV vs test, tier and per label
    cvr = r["step2"]["cv"]
    cats = [*AUTO_LABELS, "population\nauto tier"]
    ins = sel
    ho = [cvr["per_label"][l]["precision_mean"] for l in AUTO_LABELS] + [
        cvr["tier"]["precision_mean"]
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(cats))
    w = 0.26
    b1 = ax.bar(
        x - w - 0.01, ins, w, color=BLUE, label="in-sample (chosen and scored on the same rows)"
    )
    b2 = ax.bar(x, ho, w, color=AQUA, label="held-out folds within selection split, 5 folds x 3 seeds")
    b3 = ax.bar(x + w + 0.01, tst, w, color=ORANGE, label="official scored test rows")
    _bar_end(ax, b1, dy=9, fontsize=7)
    _bar_end(ax, b2, dy=2, fontsize=7)
    _bar_end(ax, b3, dy=2, fontsize=7)
    ax.axhline(floor, color=GRAY, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("precision")
    _legend_below(ax, 1)
    ax.set_title("Gap 2: selection, cross-validation and test precision", pad=14)
    fig.tight_layout()
    fig.savefig(FIG / "gap2_optimism_vs_shift.png")
    plt.close(fig)

    # fig3a: AP per label, selection vs test
    ap = r["step3"]["ap"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(LABELS))
    w = 0.34
    b1 = ax.bar(
        x - w / 2 - 0.01,
        [ap[l]["selection"] for l in LABELS],
        w,
        color=BLUE,
        label="selection split",
    )
    b2 = ax.bar(
        x + w / 2 + 0.01, [ap[l]["test"] for l in LABELS], w, color=ORANGE, label="scored test rows"
    )
    _bar_end(ax, b1)
    _bar_end(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("average precision")
    ax.legend(frameon=False)
    ax.set_title("Gap 3a: AP before any threshold is chosen")
    fig.tight_layout()
    fig.savefig(FIG / "gap3a_ap_shift.png")
    plt.close(fig)

    # fig3b: reliability in the top band
    rel = r["step3"]["reliability"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for ax, label in zip(axes, AUTO_LABELS):
        for split, color in (("thresh", BLUE), ("test", ORANGE)):
            rows = [b for b in rel[label][split] if b["n"] >= 20]
            xs = [b["mean_p"] for b in rows]
            ys = [b["observed"] for b in rows]
            lo = [b["observed"] - b["ci95"][0] for b in rows]
            hi = [b["ci95"][1] - b["observed"] for b in rows]
            ax.errorbar(
                xs,
                ys,
                yerr=[lo, hi],
                fmt="o-",
                color=color,
                ms=5,
                lw=1.5,
                capsize=2,
                label="selection split" if split == "thresh" else "scored test rows",
            )
        ax.plot([0.3, 1], [0.3, 1], color=GRAY, lw=1, ls="-")
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0.3, 1.02)
        ax.set_title(label)
        ax.set_xlabel("mean predicted probability in bin")
    axes[0].set_ylabel("observed positive rate")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "Gap 3b: reliability in the high-probability band (bins with >= 20 rows)",
        fontweight="bold",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG / "gap3b_reliability.png")
    plt.close(fig)

    # fig3c: precision vs threshold for toxic
    cur = r["step3"]["precision_curve_toxic"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for split, color, name in (
        ("thresh", BLUE, "selection split"),
        ("test", ORANGE, "scored test rows"),
    ):
        pts = [c for c in cur[split] if c["n"] >= 30]
        ax.plot(
            [c["threshold"] for c in pts],
            [c["precision"] for c in pts],
            color=color,
            lw=2,
            label=name,
        )
    t = r["provenance"]["auto_thresholds"]["toxic"]
    ax.axvline(t, color=GRAY, lw=1)
    ax.text(t, 0.77, f"chosen {t:.4f} ", fontsize=8, color="#52514e", ha="right")
    ax.axhline(floor, color=GRAY, lw=1)
    ax.set_xlim(0.9, 1.0)
    ax.set_ylim(0.75, 1.01)
    ax.set_xlabel("toxic threshold")
    ax.set_ylabel("precision of p_toxic >= threshold")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Gap 3c: toxic precision by threshold (at least 30 predictions)")
    fig.tight_layout()
    fig.savefig(FIG / "gap3c_precision_curve_toxic.png")
    plt.close(fig)

    # fig3d: text statistics
    tx = r["step3"]["text"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    edges = np.linspace(0, 2000, 41)
    centers = (edges[:-1] + edges[1:]) / 2
    for split, color, name in (
        ("train", BLUE, "train split"),
        ("test", ORANGE, "scored test rows"),
    ):
        h = np.array(tx[split]["length_hist"], dtype=float)
        h = h / h.sum()
        axes[0].plot(centers, h, color=color, lw=2, label=name)
    axes[0].set_xlabel("comment length (chars, clipped at 2000)")
    axes[0].set_ylabel("share of comments")
    axes[0].legend(frameon=False)
    axes[0].set_title("Length distribution")
    names = ["train", "thresh", "test"]
    vals = [tx[n]["oov_token_rate_mean"] for n in names]
    b = axes[1].bar(names, vals, 0.5, color=[BLUE, BLUE, ORANGE])
    _bar_end(axes[1], b)
    axes[1].set_ylabel("mean share of tokens absent from the fitted vocabulary")
    axes[1].set_title("Approximate out-of-vocabulary rate")
    fig.suptitle("Gap 3d: text length and vocabulary coverage by split", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "gap3d_text_shift.png")
    plt.close(fig)

    # fig4: composition of the toxic false positives on test
    s4 = r["step4"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1, 1.25]})
    cats = ["another label is true", "all six labels are 0"]
    vals = [s4["n_fp_with_other_label_true"], s4["n_fp_all_labels_zero"]]
    b = axes[0].barh(cats, vals, 0.5, color=[YELLOW, ORANGE])
    for bar, v in zip(b, vals):
        axes[0].annotate(
            str(v),
            (v, bar.get_y() + bar.get_height() / 2),
            va="center",
            ha="left",
            xytext=(3, 0),
            textcoords="offset points",
            fontsize=9,
            color="#52514e",
        )
    axes[0].set_xlabel(
        f"test rows above toxic threshold with toxic = 0 (total {s4['n_false_positive']})"
    )
    axes[0].set_title("Label composition of the false positives")
    lex = s4["lexicon"]
    groups = [
        ("test: false positives\n(toxic = 0, p >= thr)", "test_false_positives", ORANGE),
        ("test: true positives\n(toxic = 1, p >= thr)", "test_true_positives", ORANGE),
        ("test: all toxic = 0 rows", "test_all_toxic_zero", GRAY),
        ("selection: false positives", "selection_false_positives", BLUE),
        ("selection: true positives", "selection_true_positives", BLUE),
        ("selection: all toxic = 0 rows", "selection_all_toxic_zero", GRAY),
    ]
    names = [g[0] for g in groups]
    shares = [lex[g[1]]["share"] or 0 for g in groups]
    colors = [g[2] for g in groups]
    b = axes[1].barh(names, shares, 0.55, color=colors)
    for bar, g in zip(b, groups):
        e = lex[g[1]]
        axes[1].annotate(
            f"{e['share']:.2f} (n={e['n']})",
            (e["share"] or 0, bar.get_y() + bar.get_height() / 2),
            va="center",
            ha="left",
            xytext=(3, 0),
            textcoords="offset points",
            fontsize=8,
            color="#52514e",
        )
    axes[1].set_xlim(0, 1.3)
    axes[1].set_xlabel("share containing a fixed-lexicon term")
    axes[1].set_title("Lexicon matches by label and prediction")
    axes[1].invert_yaxis()
    axes[1].tick_params(axis="y", labelsize=8)
    fig.suptitle("Gap 4: toxic false positives, labels and lexicon matches", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "gap4_fp_composition.png")
    plt.close(fig)

    # fig5: candidate rules
    s5 = r["step5"]["rules"]
    keys = list(s5)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(keys))
    w = 0.26
    b1 = axes[0].bar(
        x - w - 0.01,
        [s5[k]["in_sample"]["precision"] for k in keys],
        w,
        color=BLUE,
        label="in-sample",
    )
    b2 = axes[0].bar(
        x, [s5[k]["cv"]["precision_mean"] for k in keys], w, color=AQUA, label="held-out CV"
    )
    b3 = axes[0].bar(
        x + w + 0.01,
        [s5[k]["test_readout"]["precision"] for k in keys],
        w,
        color=ORANGE,
        label="test readout",
    )
    _bar_end(axes[0], b1, dy=9, fontsize=7)
    _bar_end(axes[0], b2, dy=2, fontsize=7)
    _bar_end(axes[0], b3, dy=2, fontsize=7)
    axes[0].axhline(floor, color=GRAY, lw=1)
    axes[0].set_ylim(0.85, 1.02)
    axes[0].set_ylabel("population auto-tier precision")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(keys, rotation=20, ha="right", fontsize=8)
    _legend_below(axes[0], 3)
    axes[0].set_title("Precision under each selection rule", pad=14)
    b4 = axes[1].bar(
        x - 0.17,
        [s5[k]["cv"]["n_mean"] for k in keys],
        0.32,
        color=AQUA,
        label=f"held-out CV (per {r['provenance']['rows']['thresh']:,} rows)",
    )
    b5 = axes[1].bar(
        x + 0.17,
        [s5[k]["test_readout"]["n"] for k in keys],
        0.32,
        color=ORANGE,
        label=f"test (per {r['provenance']['rows']['test']:,} rows)",
    )
    _bar_end(axes[1], b4, "{:.0f}")
    _bar_end(axes[1], b5, "{:.0f}")
    axes[1].set_ylabel("rows selected by population thresholds")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(keys, rotation=20, ha="right", fontsize=8)
    _legend_below(axes[1], 2)
    axes[1].set_title("Selected counts (split sizes differ)")
    fig.suptitle(
        "Gap 5: precision and counts under candidate threshold rules", fontweight="bold", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(FIG / "gap5_selection_rules.png")
    plt.close(fig)

    # fig6: observed precision at three evaluation stages
    s6 = r["step6"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    stages = ["selection split", "held-out folds\nwithin selection split", "scored test rows"]
    vals = [s6["in_sample"], s6["held_out_cv"], s6["test"]]
    b = ax.bar(stages, vals, 0.5, color=[BLUE, AQUA, ORANGE])
    _bar_end(ax, b)
    ax.axhline(floor, color=GRAY, lw=1)
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("population auto-tier precision")
    ax.annotate(
        f"selection - CV: {s6['optimism']:+.3f}",
        (0.5, (vals[0] + vals[1]) / 2 + 0.02),
        ha="center",
        fontsize=9,
        color="#52514e",
    )
    ax.annotate(
        f"CV - test: {s6['shift']:+.3f}",
        (1.5, (vals[1] + vals[2]) / 2 + 0.02),
        ha="center",
        fontsize=9,
        color="#52514e",
    )
    ax.set_title("Gap 6: observed differences, not causal attribution")
    fig.tight_layout()
    fig.savefig(FIG / "gap6_decomposition.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, help="saved run directory to diagnose")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="regenerate figures from record/results.json without recomputing diagnostics",
    )
    args = parser.parse_args()
    if args.render_only:
        if args.run_dir is not None:
            parser.error("run_dir cannot be used with --render-only")
        figures(json.loads((HERE / "results.json").read_text()))
    elif args.run_dir is None:
        parser.error("run_dir is required unless --render-only is supplied")
    else:
        main(args.run_dir.resolve())
