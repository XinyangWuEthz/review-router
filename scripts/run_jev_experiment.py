#!/usr/bin/env python
"""Compare cached Jev scores with a word baseline on one frozen pilot cohort.

Preparation and replay are offline. --allow-network explicitly enables billed
Jev calls; credentials come only from TYPESAFE_API_KEY in the process environment.
No labels are sent to Jev. The independent human evaluation remains deferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from review_router.data import LABELS, Corpus, file_sha256, label_counts, load_corpus  # noqa: E402
from review_router.model import PlattCalibrator, TfidfLogitModel  # noqa: E402
from review_router.pipeline import evaluate_scores, load_config  # noqa: E402
from review_router.policy import load_policy  # noqa: E402
from scripts.compare_runs import (  # noqa: E402
    equal_input_loads,
    load,
    matched_volume,
    shared_stream,
    summarize,
    to_markdown,
)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sample_frame(frame: pd.DataFrame, n: int, seed: int, split: str) -> pd.DataFrame:
    """Select by ID hash, without labels; keep original corpus order for evaluation."""
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0 or n > len(frame):
        raise ValueError(f"{split} sample size must be in 1..{len(frame)}, got {n}")
    ranks = frame["id"].map(lambda identifier: digest([seed, split, str(identifier)]))
    chosen = set(frame.loc[ranks.sort_values(kind="stable").index[:n], "id"])
    return frame.loc[frame["id"].isin(chosen)].copy()


def pilot_corpus(corpus: Corpus, sizes: dict[str, int], seed: int) -> Corpus:
    if set(sizes) != {"calib", "thresh", "test"}:
        raise ValueError("sample_sizes must specify calib, thresh and test")
    selected = []
    for name in ("train", "calib", "thresh"):
        frame = corpus.train[corpus.train["split"] == name]
        selected.append(frame if name == "train" else sample_frame(frame, sizes[name], seed, name))
    return Corpus(
        train=pd.concat(selected).sort_index(),
        test=sample_frame(corpus.test, sizes["test"], seed, "test"),
        file_hashes=corpus.file_hashes,
    )


def parts(corpus: Corpus) -> dict[str, pd.DataFrame]:
    return {
        **{n: corpus.train[corpus.train["split"] == n] for n in ("train", "calib", "thresh")},
        "test": corpus.test,
    }


def calibrate_scores(
    calibration: np.ndarray, y: np.ndarray, targets: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Platt scaling on logits, fitted only on the designated calibration rows."""
    def logits(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if p.ndim != 2 or p.shape[1] != len(LABELS):
            raise ValueError("expected six probability columns")
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("probabilities must be finite and in [0, 1]")
        q = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(q / (1 - q))

    margins = logits(calibration)
    if margins.shape != y.shape or not np.isin(y, (0, 1)).all():
        raise ValueError("calibration labels must be binary and match the score matrix")
    output = {name: np.empty_like(logits(p)) for name, p in targets.items()}
    target_logits = {name: logits(p) for name, p in targets.items()}
    parameters: dict[str, Any] = {"method": "Platt on logit(p)", "clip_epsilon": 1e-6}
    parameters["labels"] = {}
    for j, label in enumerate(LABELS):
        calibrator = PlattCalibrator().fit(margins[:, j], y[:, j])
        fitted = len(np.unique(y[:, j])) == 2
        parameters["labels"][label] = {
            "a": calibrator.a, "b": calibrator.b,
            "positive": int(y[:, j].sum()), "negative": int(len(y) - y[:, j].sum()),
            "status": "fitted" if fitted else "one class: raw probability retained",
        }
        for name, p in target_logits.items():
            output[name][:, j] = (
                calibrator.transform(p[:, j]) if fitted else targets[name][:, j]
            )
    return output, parameters


def brier(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {label: float(np.mean((p[:, j] - y[:, j]) ** 2)) for j, label in enumerate(LABELS)}


def run_summary(directory: Path) -> dict[str, Any]:
    item = load(directory)
    return {
        **summarize(item), "run_dir": str(directory),
        "threshold_selection": item["report"]["threshold_selection"]["per_label"],
        "thresholds": json.loads((directory / "thresholds.json").read_text()),
    }


def paired_capture(
    y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, weights: np.ndarray,
    high_risk_min: float, k: int, repeats: int, seed: int,
) -> dict[str, Any]:
    """Paired row bootstrap, conditional on fitted models and a fixed top-k budget."""
    if not 0 < k <= len(y) or repeats < 1:
        raise ValueError("nonzero top-k and positive bootstrap repetitions required")
    high = (y * weights).max(axis=1) >= high_risk_min
    scores = [p.max(axis=1) for p in (baseline, candidate)]

    def captured(index: np.ndarray, score: np.ndarray) -> int:
        top = np.argsort(-score[index], kind="stable")[:k]
        return int(high[index[top]].sum())

    original = np.arange(len(y))
    counts = [captured(original, score) for score in scores]
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repeats):
        index = rng.integers(0, len(y), len(y))
        differences.append(captured(index, scores[1]) - captured(index, scores[0]))
    return {
        "ranking": "max calibrated label probability, stable ties in corpus order",
        "k": k, "total_high_risk": int(high.sum()),
        "baseline_captured": counts[0], "jev_captured": counts[1],
        "difference_jev_minus_baseline": counts[1] - counts[0],
        "ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
        "replicates": repeats, "seed": seed,
        "scope": "ranking diagnostic, not the actual rule-based router at equal budget; "
        "conditional on this cohort, fitted models and fixed k; no retraining uncertainty",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/jev_pilot.yaml")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()
    if args.prepare_only and args.allow_network:
        parser.error("--prepare-only and --allow-network cannot be combined")
    path = args.config.resolve()
    spec = yaml.safe_load(path.read_text())
    baseline_path = (ROOT / spec["baseline_config"]).resolve()
    cfg = load_config(baseline_path)
    corpus = pilot_corpus(
        load_corpus(cfg.data_dir, cfg.split_fractions, cfg.seed),
        spec["sample_sizes"], int(spec["sample_seed"]),
    )
    frames = parts(corpus)
    questions_path = ROOT / spec["questions"]
    questions = json.loads(questions_path.read_text())
    out = ROOT / spec["analysis_dir"]
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "scope": "exploratory paired pilot on original Jigsaw labels; not a fresh holdout",
        "model": spec["model"], "question_sha256": file_sha256(questions_path),
        "experiment_config_sha256": file_sha256(path),
        "baseline_config_sha256": file_sha256(baseline_path),
        "policy_sha256": file_sha256(cfg.policy_path),
        "source_data_sha256": corpus.file_hashes,
        "sample_seed": spec["sample_seed"], "requested_sizes": spec["sample_sizes"],
        "sampling": "lowest SHA256(seed, split, id), without labels; original order retained",
        "counts": {name: label_counts(frame) for name, frame in frames.items()},
        "id_sha256": {name: digest(frame.id.tolist()) for name, frame in frames.items()},
        "comparison": "same calibration, threshold-selection and test rows for both methods",
        "primary_diagnostic": "high-risk captured at word baseline flagged count, max-score top-k",
        "operational_check": "same policy and common incoming comment streams, 200 paired seeds",
        "limitations": [
            "Existing test data have been inspected repeatedly.",
            "Jev pretraining overlap with this public corpus is unknown.",
            "No independent human review-worthiness evaluation; original labels are a proxy.",
            "Small rare-label counts can make calibration or threshold selection inconclusive.",
            "Bootstrap holds models and thresholds fixed; simulation SE covers only queue seeds.",
            "No BERT run: this comparison cannot establish superiority over BERT.",
        ],
    }
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise SystemExit("Frozen protocol differs: use a new analysis_dir for a new experiment")
    write_json(protocol_path, protocol)
    pd.concat([
        frame[["id"]].assign(split=name) for name, frame in frames.items() if name != "train"
    ]).to_csv(out / "cohort.csv", index=False)

    result: dict[str, Any] = {
        "updated_utc": datetime.now(UTC).isoformat(),
        "status": "prepared", "protocol": protocol, "protocol_sha256": digest(protocol),
        "baseline": None, "jev": None,
        "conclusion_zh": "尚无 Jev 预测，不能判断它是否改善分类或审核队列。默认仍保持 word。",
    }
    policy = load_policy(cfg.policy_path)
    result["selection_rule"] = {
        "min_predicted_positives": int(
            policy.gates.get("min_predicted_positives_for_precision", 30)
        ),
        "precision_targets": policy.tier_precision_floors,
    }

    def save() -> None:
        result["updated_utc"] = datetime.now(UTC).isoformat()
        write_json(out / "experiment.json", result)
        write_json(ROOT / "record/jev_run.json", result)

    save()
    text = {name: frame.comment_text.astype(str).tolist() for name, frame in frames.items()}
    labels = {name: frame[list(LABELS)].to_numpy(dtype=int) for name, frame in frames.items()}
    common_source = {
        "labels": list(LABELS),
        "row_ids": {name: frames[name].id.tolist() for name in ("thresh", "test")},
        "protocol_sha256": digest(protocol), "cohort_id_sha256": protocol["id_sha256"],
        "exploratory": True,
    }

    def evaluate(name: str, scores: dict[str, np.ndarray], source: dict[str, Any],
                 model: Any = None) -> Path:
        raw = yaml.safe_load(baseline_path.read_text())
        raw.update(label=f"jev-pilot-{name}", data_dir=str(cfg.data_dir),
                   output_dir=str(cfg.output_dir), policy=str(cfg.policy_path))
        run_config = cfg.output_dir / "jev-configs" / f"{name}.yaml"
        run_config.parent.mkdir(parents=True, exist_ok=True)
        run_config.write_text(yaml.safe_dump(raw, sort_keys=False))
        return evaluate_scores(
            run_config, corpus, scores["thresh"], scores["test"],
            score_source={**common_source, **source}, model=model,
        )

    stage = "word_baseline"
    client: Any = None
    try:
        print("Fitting the paired word baseline on the original train split...", flush=True)
        start = time.perf_counter()
        baseline = TfidfLogitModel(cfg.model).fit(text["train"], labels["train"], cfg.seed)
        baseline.calibrate(text["calib"], labels["calib"])
        started_predicting = time.perf_counter()
        base_scores = {name: baseline.predict_proba(text[name]) for name in ("thresh", "test")}
        inference_seconds = time.perf_counter() - started_predicting
        baseline_seconds = time.perf_counter() - start
        base_dir = evaluate(
            "word", base_scores,
            {"model_name": "word-tfidf-logistic",
             "calibration": "Platt on margins, shared pilot calibration split"}, baseline,
        )
        result["baseline"] = {
            **run_summary(base_dir),
            "fit_calibration_prediction_seconds": baseline_seconds,
            "prediction_seconds": inference_seconds,
            "brier": brier(labels["test"], base_scores["test"]),
        }
        result["status"] = "awaiting_jev_scores"
        save()
        print("Baseline:", base_dir, flush=True)
        if args.prepare_only:
            print("Prepared. No API calls made. Next: --allow-network, or replay cached responses.")
            return

        from review_router.jev import JevClient

        stage = "jev_client_setup"
        client = JevClient(
            model=spec["model"], questions=questions, cache_dir=ROOT / spec["cache_dir"],
            workers=int(spec["workers"]), requests_per_second=float(spec["requests_per_second"]),
        )
        raw_scores = {}
        for name in ("calib", "thresh", "test"):
            stage = f"jev_predict_{name}"
            print(f"Jev {name}: {len(text[name])} comments", flush=True)
            raw_scores[name] = client.predict(text[name], allow_network=args.allow_network)
        result["status"] = "jev_scores_cached"
        result["api"] = client.summary()
        save()
        stage = "jev_calibration"
        calibrated, calibration = calibrate_scores(
            raw_scores["calib"], labels["calib"],
            {name: raw_scores[name] for name in ("thresh", "test")},
        )
        api = client.summary()
        stage = "jev_evaluation"
        jev_dir = evaluate("jev", calibrated, {
            "model_name": spec["model"], "questions_sha256": file_sha256(questions_path),
            "calibration": calibration, "api": api,
        })
        write_json(jev_dir / "calibration.json", calibration)
        result["jev"] = {
            **run_summary(jev_dir), "calibration": calibration,
            "raw_brier": brier(labels["test"], raw_scores["test"]),
            "brier": brier(labels["test"], calibrated["test"]),
        }
        result["api"] = api
        result["published_input_usd_per_million_tokens"] = spec["input_usd_per_million_tokens"]
        result["status"] = "jev_evaluated"
        result["conclusion_zh"] = "两种方法已各自完成评估，配对比较尚未完成，暂不更换默认模型。"
        save()
        stage = "paired_comparison"
        items = [load(base_dir), load(jev_dir)]
        for item in items[1:]:
            assert item["data"] == items[0]["data"]
            assert item["manifest"]["policy_sha256"] == items[0]["manifest"]["policy_sha256"]
            assert item["pred"]["id"].equals(items[0]["pred"]["id"])
            assert np.array_equal(
                item["pred"][[f"y_{lb}" for lb in LABELS]],
                items[0]["pred"][[f"y_{lb}" for lb in LABELS]],
            )
        fraction = float(items[0]["report"]["review_workload"]["review_fraction"])
        comparison: dict[str, Any] = {"matched_volume": matched_volume(items)}
        if fraction > 0:
            comparison["equal_input"] = shared_stream(
                items, equal_input_loads(fraction, list(cfg.loads_per_hour))
            )
            policy = load_policy(cfg.policy_path)
            result["paired_capture"] = paired_capture(
                labels["test"], base_scores["test"], calibrated["test"],
                np.array([policy.severity_weights[lb] for lb in LABELS]),
                cfg.high_risk_min_weight,
                int(items[0]["report"]["review_workload"]["n_requires_human_review"]),
                int(spec["bootstrap_replicates"]), int(spec["sample_seed"]),
            )
        comparison["runs"] = [summarize(item) for item in items]
        write_json(out / "run_comparison.json", comparison)
        (out / "run_comparison.md").write_text(to_markdown(comparison["runs"], comparison))
        result["comparison"] = comparison
        result["status"] = "completed_exploratory"
        result["conclusion_zh"] = (
            "已完成固定样本上的探索性对照。应结合等标记量高风险召回区间、"
            "同输入流下的积压与完成量判断是否值得扩大实验；本轮不自动更换默认模型。"
        )
        save()
        print("Saved:", out / "experiment.json", flush=True)
    except (Exception, SystemExit) as exc:
        result["status"] = "incomplete"
        result["failure"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
        if client is not None:
            result["api"] = client.summary()
        result["conclusion_zh"] = (
            f"实验在 {stage} 阶段停止，尚不能作完整效果判断。"
            "已完成的评估和成功响应保留，可修复后重放缓存。默认仍保持 word。"
        )
        save()
        raise SystemExit(f"Experiment incomplete at {stage}: {exc}") from exc


if __name__ == "__main__":
    main()
