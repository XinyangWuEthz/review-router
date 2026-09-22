# Round 1: one-command reproducible experiment

Question: with identical reviewer capacity, does risk-ordered review handle
high-harm comments earlier than first-in-first-out (FIFO)?

This spec records the user's round-1 protocol plus the concrete decisions taken
where the protocol left room. Anything marked *assumption* is a stated default,
chosen to keep the run reproducible, and lives in `configs/*.yaml`.

## Deliverable

```bash
python scripts/run_pipeline.py --config configs/baseline.yaml
```

writes `reports/<run_id>/` containing `manifest.json` (data hashes, split
counts, git commit, dependency versions, seeds), `splits.csv`,
`thresholds.json`, `predictions.csv` (one row per scored test comment),
`report.json` (consumed by `tests/test_gate.py`) and `report.md` (the tables).

## 1. Frozen data split

- Corpus: official Jigsaw 2018 files `train.csv`, `test.csv`, `test_labels.csv`.
  Test text is joined to labels on `id`; unscored rows (`-1`) are dropped with
  the existing `drop_unscored`.
- `train.csv` is split 60/20/20 into `train` / `calib` / `thresh` by a seeded
  shuffle. *Assumption:* plain random split, not stratified; the per-set label
  counts are written to the manifest so imbalance is visible rather than hidden.
- The scored test set is used once, for the final evaluation, and never for
  fitting, calibration or threshold choice.
- `splits.csv` records `id, split` for every training comment.

## 2. Model

- Word TF-IDF (1-2 grams, `min_df=3`, `sublinear_tf`, `max_features=200k`) fitted
  on `train` only; six independent `LogisticRegression` heads, one per label.
- Calibration: Platt (sigmoid) scaling fitted on `calib` over each head's
  decision function. Implemented directly (a 1-D logistic regression on the
  margin) so there is no dependence on scikit-learn's prefit-calibration API,
  which changed across versions.
- Hierarchy is *not* enforced in round 1; the violation rate
  (`p_severe_toxic >= 0.5 and p_toxic < 0.5`) is measured and reported.

## 3. Thresholds and routing

- On `thresh`, per label and per tier, choose the lowest probability threshold
  whose precision is at or above the tier floor (`human_review` 0.90,
  `auto_action` 0.99 from `policy.yaml`) with at least
  `min_predicted_positives_for_precision` (30) predicted positives. Lowest
  threshold = maximum coverage. No qualifying threshold disables that tier for
  that label.
- Model tier per comment: `auto_action` if any label meets its auto threshold;
  else `human_review` if any label meets its human threshold; else `allow`.
- Rules in `policy.yaml` have priority. If any rule matches, its action replaces
  the model tier (R101 exists precisely to keep `threat` out of auto-action).
  Rule-forced queue entries are counted in the report and included in every
  downstream metric.
- Reported per tier: count, precision, coverage, Wilson 95% interval.
  Precision is `n/a` when the tier fired fewer than 30 times. Tier correctness:
  an automatic action is correct if at least one label triggering its automatic
  threshold is truly positive; a human review is correct if any label is positive.
- The precision targets apply to individual labels during threshold selection,
  not to the final combined tiers. Report the final-tier metrics on that split
  as development diagnostics as well as on the held-out test set.
- `identity_term_present` is a word-boundary match against a small fixed list
  of identity terms in `review_router/signals.py` (a policy input, not a
  measurement).

## 4. Queue simulation

- Job set: the scored test comments routed to `human_review` (model or rule).
- Reviewers 4, deterministic 2-minute handle time, 8-hour horizon
  (*assumption*, capacity 120/h). Load levels 60, 108, 180 arrivals/h.
- Arrivals: Poisson process, drawn once per (seed, load); jobs are sampled from
  the queued set with replacement. Every strategy replays the identical arrival
  list and handle times. Seeds: 5 fixed.
- Strategies: `fifo`; `prob` (max label probability); `severity`
  (`max_j p_j * w_j` with `severity_weights` from policy). Ties → arrival order.
- Harm proxy for a job = maximum severity weight among its *true* labels
  (0 if clean); no double counting on overlapping labels. High-risk job = true
  label with weight ≥ 5 (threat, identity_hate, severe_toxic).
- Metrics per (strategy, load), mean ± std across seeds: high-risk handled in
  horizon, weighted harm handled per reviewer-hour, wait p50/p90, backlog at
  end, high-risk not completed by the horizon, reviewer utilisation, queue depth p95.
  Wait quantiles include completed jobs only. Backlog excludes jobs in service;
  the high-risk unhandled count includes them.

## 5. Report and gates

- `report.json` sections: `per_label` (AP, precision, recall at the human
  threshold), `tiers`, `rules`, `consistency`, `simulation` (full table plus a
  primary scenario summary: `harm_per_reviewer_hour.{router,fifo}`,
  `queue_depth_p95`, `reviewer_utilization` at load 108/h, router = severity).
- `tests/test_gate.py` change: when `REVIEW_ROUTER_EVAL_REPORT` points to a
  path that does not exist, the gates fail instead of skipping. Unset means skip.

## 6. Data availability

No Jigsaw files or Kaggle credentials are on this machine. The pipeline is
verified end to end on a synthetic corpus with the Jigsaw schema
(`scripts/make_synthetic_corpus.py`, `configs/smoke.yaml`). Numbers from the
synthetic run are pipeline checks only and are never reported as results.
