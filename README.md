# review-router

A capacity-constrained content-policy router that sends comments to
**allow / human-review / priority-review**. Both review bands require a human
to confirm any action and consume the same reviewer capacity. The current
baseline uses TF-IDF classifiers; additional model signals are planned.

A classifier's ROC-AUC does not determine the precision or workload at its
chosen operating point. This project measures the routing layer above a
classifier: what gets queued, which comments receive priority, and what remains
unfinished when reviewer capacity is limited.

The earlier automatic-action experiment required 99% precision and measured
90.4%. It is preserved in the [round-1 record](record/step2-round1.html).
The current workflow evaluates assistance to human reviewers. Its success
criteria concern review workload and ordering under limited capacity; no
automatic enforcement capability is implemented or claimed.

## Why operating-point quality matters

At a fixed recall and false-positive rate, class balance determines precision.
On the 63,978 **scored** Jigsaw test rows, assuming 50% recall:

| Label | positives / negatives | precision @ FPR 1% | @ FPR 0.1% |
|---|---|---|---|
| toxic | 6,090 / 57,888 | 84.0% | 98.1% |
| obscene | 3,691 / 60,287 | 75.4% | 96.8% |
| insult | 3,427 / 60,551 | 73.9% | 96.6% |
| identity_hate | 712 / 63,266 | 36.0% | 84.9% |
| severe_toxic | 367 / 63,611 | 22.4% | 74.3% |
| **threat** | **211 / 63,767** | **14.2%** | 62.3% |

Reproduce: `python -c "from review_router.ceilings import ceiling_table; print(ceiling_table())"`

For `threat`, a 1% false-positive rate produces ~638 false positives against
~106 true positives. Even at FPR 0.1%, the implied precision is only 62.3%.
This motivates evaluating rare labels at the operating point used by the queue.
It does not establish a universal precision ceiling: a lower FPR can give
higher precision. The policy sends comments with a threat probability of at
least 0.30 to a human.

These are identities over the label counts, not model results.

## Design

- **Empirical targets define review bands.** Initial threshold-selection
  targets are 0.95 for `priority_review` and 0.90 for `human_review`.
  These are development settings, fixed before this run, not guarantees of
  test precision or permission to enforce. The pipeline selects the largest
  qualifying set per label and reports the resulting precision and coverage.
- **Policy as reviewable config.** Tiers, severity weights, rules and CI floors
  live in [`review_router/policy.yaml`](review_router/policy.yaml); every rule
  carries a prose rationale for *why its tier is what it is*. The loader rejects
  unknown actions and operators, and that rejection is tested.
- **Segment agreement selects the priority band.** The `priority_review`
  threshold for a label is the lowest declared score edge such that every
  segment at or above it meets the 0.95 target on its own with at least 30
  rows on the selection split. This is a project extension inspired by the
  confidence-threshold analysis of Thomas et al. (arXiv 2406.12800), whose
  paper does not prescribe the per-segment 95% / 30-row rule. The cumulative rule let a strong top
  segment carry weaker ones below it (toxic selected 0.847 while the
  [0.95, 0.98) bin alone reached 0.937); the segment rule reports the
  agreement and coverage of every segment instead. The `human_review` band
  and the identity-term slice keep the cumulative rule.
- **Rules and models coexist.** Models supply scores. Threat and label
  inconsistency rules promote comments into priority review.
  Where a slice needs extra care, the floor is required on that slice
  (`subgroup_thresholds`) rather than typed in as a probability constant.
- **All review work consumes capacity.** Both review bands enter the simulator.
  The queue is served in the configured primary ordering, predicted severity
  (highest max predicted probability times severity weight first). The
  band-first ordering, priority band first and then severity within each band,
  is kept as a compared alternative, as are FIFO and probability ordering, all
  on the same admitted comments and arrivals. The paired per-seed comparison
  of the primary ordering against every alternative is gated at the overload
  load.

## Reproducing the human-review experiment

The current experiment asks: **with the same reviewer capacity, does the router
(the review queue served in predicted-severity order) handle at least as much
high-risk work as first-in-first-out, and reach it sooner?** Every assumption
of the protocol is a line in `configs/baseline.yaml` or
`review_router/policy.yaml`, and the pipeline module docstring walks the stages.

```bash
pip install -c configs/frozen-ml.txt -e ".[ml,dev]"
# Put the official Jigsaw 2018 files in data/jigsaw/: train.csv, test.csv, test_labels.csv
# (Kaggle: jigsaw-toxic-comment-classification-challenge; the corpus is not redistributed here.)
# One-time restore from the saved, accepted CI run:
gh run download 35857528118 --repo XinyangWuEthz/review-router \
  --name evaluation-210d9857220b23d175d4aff5a54ab221ad86280b \
  --dir reports/ci-35857528118
python scripts/restore_frozen_model.py --source-run reports/ci-35857528118
python scripts/run_pipeline.py --config configs/baseline.yaml
```

For development iterations, run the same protocol without the scored test rows:

```bash
python scripts/run_pipeline.py --config configs/dev.yaml
```

Development runs stop before the test rows are scored: they write the
thresholds, the model and the development sections of the report, but no
predictions, simulation or test section, and they are not a gate input. Use
them to compare orderings and threshold rules on the selection split; read the
high-risk-by-band and ranking tables there first.

One run writes `reports/<run_id>/` with:

| file | contents |
|---|---|
| `manifest.json` | data file hashes, split label counts, git commit, dependency versions, seeds, the config |
| `config.yaml`, `policy.yaml` | snapshots of the exact experiment and policy inputs |
| `splits.csv` | every training id and its split (`train` 60% / `calib` 20% / `thresh` 20%) |
| `thresholds.json` | per-label thresholds for each tier, `null` where the precision floor is unreachable |
| `predictions.csv` | one row per scored test comment: labels, calibrated probabilities, model tier, matched rules, final tier and human-review requirement |
| `simulation_jobs.csv` | every simulated arrival, with strategy, seed, times, status, risk and routing reason |
| `model.pkl` | fitted TF-IDF vocabulary, classifiers and calibrators |
| `frozen-model.json` | scorer lock snapshot for frozen runs; its hash and source training commit are in the manifest |
| `report.json` | decision contract, total review workload, per-label and tier quality with Wilson intervals, subgroup diagnostics, rule effects, four-strategy simulation |
| `report.md` | the same, as tables |

The scorer is frozen from the accepted run at commit `210d9857220b`. Its word
vocabulary, IDF, six logistic heads and Platt calibration parameters are reused
without fitting. `configs/frozen-baseline.json` pins the model, source manifest,
split assignments and model implementation. Both baseline and development runs
reject changed data, model settings, splits or incompatible ML dependencies;
missing artifacts never trigger training. Each report records the original
training commit separately from the current router commit and copies the exact
model bytes. The current project studies the router on top of this fixed scorer.
The [freeze record](record/frozen-router.md) documents the real-data replay and
its numerical limits.

Keep `models/frozen-baseline/` backed up: GitHub Actions artifacts have limited
retention, and its cache is not permanent storage. The restore command can use
any saved copy whose files match the lock. CI uses the frozen scorer by default.
Freezing prevents vocabulary-cap tie selection from changing router inputs.
It does not fix the underlying full-training reproducibility issue. Minor
floating-point inference differences can still occur across platforms. Use the
saved predictions for exact score reuse in queue comparisons, and the shared
incoming stream when admission rules change.

Full-training checks are explicit and do not replace the pinned model:

```bash
python scripts/run_pipeline.py --config configs/baseline.yaml --retrain
```

The original scorer was fitted on `train` and Platt-calibrated on `calib`.
Router iterations still choose tier thresholds on `thresh` from the targets in
`policy.yaml`, apply rules, then simulate a fixed-capacity review queue
(4 reviewers, 2 min per item, 8 h, loads of 60 / 108 / 180 per hour, 5 seeds)
replayed under FIFO, probability, severity and priority-band ordering. Every
strategy sees the identical arrivals. Model fitting, calibration and threshold
selection use separate portions of the training data. The scored test set has
now been inspected across several policy iterations, so these results are an
iterative benchmark rather than a fresh, untouched final evaluation. Further
threshold selection should use development data, with new held-out
data needed for an independent confirmation.

The 90% and 95% targets select thresholds independently for each label.
Combining labels, separating review bands and applying rules can change the
final tiers' precision. For both bands, review is useful when any annotated
label is positive. The report shows final-tier quality
on both the threshold-selection split and the held-out test set; these targets
are not guarantees. Simulation wait quantiles include every job that started,
including reviews still in progress. Read them alongside completed and
unfinished high-risk counts and the remaining backlog.

Arrival rates are measured after admission to the combined human queue. They
are conditional ranking experiments, not a fixed incoming-comment traffic
comparison with the old automatic-action policy. The report records the
admission fraction and equivalent incoming rate so workload changes remain visible.

The initial benchmark requirements are a harm-per-reviewer-hour ratio of at
least 1.0 versus FIFO at overload, high-risk wait p90 no higher than FIFO near
capacity, and high-risk completed counts no lower than FIFO at both loads.
The completion checks prevent shorter waits from hiding unfinished work.
At the overload load the primary ordering's mean per-seed difference against
each alternative ordering (FIFO, probability, priority band) must be no worse
than -1.0 high-risk completions and -0.5 harm per reviewer-hour; wait quantiles
are reported and do not decide.
These are empirical comparisons, not statistical noninferiority guarantees.
Per-tier precision and coverage are diagnostic; there is no 99% test-precision gate.
At the primary load of 108 reviews/hour, the added capacity checks require
at least 97% completion, mean end backlog at most 10, pooled wait p50 at most
1 minute, mean queue-depth p95 at most 39, and utilization at least 60%.
These limits were adopted before the merged baseline rerun.

Agreement-by-confidence tables (label agreement and coverage per score
segment, pooled and per label, with identity and out-of-vocabulary strata) are
computed from the calibration and threshold-selection splits. They are
development diagnostics; `configs/dev.yaml` omits test scoring entirely.

Regression gates read the report. The README results block uses the report and
the policy snapshot saved with that run:

```bash
REVIEW_ROUTER_EVAL_REPORT=reports/<run_id>/report.json pytest -q tests/test_gate.py
```

```bash
python scripts/render_results.py reports/<run_id>
```

Gates whose floor is still `null` in `policy.yaml` skip; an explicitly requested
report that is missing or lacks a required section fails. Identity-disparity
checks report inconclusive intervals as skipped rather than treating them as
evidence that the groups have equal error rates.

To check the pipeline without the corpus, generate a synthetic stand-in with
the same file layout. Its numbers are pipeline checks, never results:

```bash
python scripts/make_synthetic_corpus.py --out data/synthetic
python scripts/run_pipeline.py --config configs/smoke.yaml
```

## Per-job simulation records and the headline time metric

Every simulated arrival leaves a row in `simulation_jobs.csv` in the run
directory: comment id, arrival, start and completion time, ordering strategy,
load, seed, trigger reason (the rule ids or model labels that put it in the
queue), whether it is high-risk, and its status at the horizon: `not_started`,
`in_progress` or `completed`. Two times are derived from it:

| metric | definition | population |
|---|---|---|
| wait | start of review minus arrival | every job that started, finished or not |
| completion latency | completion minus arrival | every job completed within the horizon |

Both are reported at p50, p90 and p99, with the sample size; a p99 on fewer
than 100 samples is printed but flagged unreliable. Jobs that never started are
counted by status, with their high-risk share and their age at the horizon;
they are never given a wait of zero. Completion means the simulated review
finished. Any moderation action still requires human confirmation; the
simulation does not observe enforcement outcomes.

The protocol selects one headline: `configs/baseline.yaml` names the metric
and percentile (wait p50), read for the router and FIFO at the primary load,
pooled over seeds. They use identical arrivals and the same metric definition,
but their started and completed subsets may differ. Reduction is 1 minus the
router value over the FIFO value; when the FIFO value is zero only the absolute
difference is reported. The high-risk p90 comparison is reported alongside and
is never a substitute. Every number in the report's time tables is recomputed
from the records file by the gates.

## Two-layer CI

- **Code checks** (`ci.yml`, pushes to `main` and pull requests): lint, types, unit
  tests, the synthetic pipeline test. The regression gates skip here because no
  report is passed in.
- **Real evaluation** (`real-eval.yml`, pushes to `main`, weekly, on demand):
  fetches the three Jigsaw files, verifies them against `configs/jigsaw.sha256`,
  runs the pipeline with the code under test, archives the run directory as a
  workflow artifact, then points the gates at the report it just produced with
  `REVIEW_ROUTER_EVAL_REQUIRE_CLEAN=1`. It never re-checks an old report, and
  a second job tampers with a synthetic run in five ways (missing report,
  policy mismatch, metric below floor, primary ordering losing to an
  alternative, foreign commit) and fails unless every one is blocked.

The gates sit in four categories in `policy.yaml`: classification (per-label
AP and precision at the operating point, hierarchy consistency, predicted
volume overshoot), routing (tier precision with a minimum sample size),
capacity (read at the stated primary or overload scenario), and reproducibility (a scenario is
replayed from `predictions.csv` and compared with the saved records under a
stated float tolerance). Determinism is a check, not a precision floor, and a
queue-depth ceiling that holds at the primary load says nothing about other
loads. Both review bands require human confirmation. Their test precision
remains diagnostic; the historical 99% automatic-action gate does not apply
to this workflow. High-risk completions must also remain at least as high as
FIFO at the primary and overload scenarios.

<!-- results:start -->
## Review-routing results (Jigsaw scored test rows), run `20260923T115815829404Z-human-review-baseline`

Generated from this run's `report.json` and `manifest.json`. The run used commit `210d9857220b`, seed 20260922, scikit-learn 1.9.1. The input files' SHA-256 hashes are recorded in the manifest.

Development split: 95,743 / 31,914 / 31,914 rows for train / calibration / threshold selection; 63,978 scored test rows evaluated separately. Precision targets come from the run snapshot `reports/ci-35857528118/policy.yaml`.

**Decision contract and evaluation criteria.** Every moderation action requires human confirmation. The model routes comments to priority_review, human_review or allow. Both review bands require human confirmation; queue precedence is determined by the configured primary ordering. Automatic actions must remain zero.

The run's per-label selection targets are 0.95 for priority_review and 0.90 for human_review. These are empirical targets on the selection split, not guarantees of test precision. Per-tier test precision remains diagnostic when its gate is not set.

Comparison gates come from `reports/ci-35857528118/policy.yaml`. The router is the configured primary ordering, severity. It serves the combined review pool by predicted severity; the review band does not change queue precedence. The band-first priority ordering is reported as an alternative. At equal reviewer capacity, the router/FIFO harm-per-reviewer-hour ratio at 180.000/h must be ≥ 1.000; the high-risk wait p90 ratio at 108.000/h must be ≤ 1.000. The high-risk completed-count ratio must be ≥ 1.000 at both named loads. These comparisons use ratios of seed means. They are descriptive checks that the measured result is no worse than FIFO, not statistical non-inferiority tests. At 180.000/h the primary ordering's mean per-seed difference against each of fifo, prob, priority must be at least -1.000 high-risk completions and -0.500 harm per reviewer-hour; wait quantiles are reported and do not decide. Historical automatic-enforcement results use a different contract.

**Per label.** Thresholds were selected for precision targets 0.90 (human) and 0.95 (priority). Selection and test AP are both shown. An AP difference describes a measured performance gap; it does not by itself identify the cause.

| label | positives | AP selection | AP test | ROC-AUC test | human thr | P@human | R@human | priority thr | P@priority | R@priority |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| toxic | 6090 | 0.859 | 0.752 | 0.957 | 0.6575 | 0.663 | 0.680 | 0.9800 | 0.880 | 0.393 |
| severe_toxic | 367 | 0.492 | 0.311 | 0.982 | n/a | n/a | n/a | n/a | n/a | n/a |
| obscene | 3691 | 0.874 | 0.773 | 0.972 | 0.6575 | 0.773 | 0.616 | 0.9950 | 0.940 | 0.318 |
| threat | 211 | 0.459 | 0.458 | 0.992 | n/a | n/a | n/a | n/a | n/a | n/a |
| insult | 3427 | 0.774 | 0.696 | 0.965 | 0.9179 | 0.878 | 0.289 | 0.9950 | 0.943 | 0.120 |
| identity_hate | 712 | 0.415 | 0.480 | 0.974 | n/a | n/a | n/a | n/a | n/a | n/a |

**Total human-review workload.** Both flagged tiers enter the same review pool and consume reviewer capacity. The report records 6312 of 63978 comments requiring review, a fraction of 0.099: 2823 priority reviews and 3489 standard reviews. Of the flagged comments, 4235 have at least one positive label. Combined review precision is 0.671, with 95% interval 0.659 to 0.682. Recorded automatic actions: 0.


**Routing tiers.** Matching policy rules promoted 99 comments to priority review and added 18 comments from allow to the review pool. These aggregate counts can overlap. Both review tiers count a comment as positive when any label is true.

| tier | n | coverage | precision (test) | 95% CI | precision (selection split) | target |
|---|---:|---:|---:|---|---:|---:|
| priority_review | 2823 | 0.044 | 0.874 | 0.862 to 0.886 | 0.9953 | 0.950 |
| human_review | 3489 | 0.055 | 0.506 | 0.490 to 0.523 | 0.8710 | 0.900 |
| allow | 57666 | 0.901 | n/a | n/a | n/a | n/a |

Per-label precision targets need not hold after pooling labels, partitioning the review pool or applying rules. Test precision is measured separately. The gates below retain the declared limits; changes to models, calibration or threshold selection need a fresh evaluation.

**Identity mentions.** The run recorded 63 whole-word terms in its `report.json`. This word-list diagnostic is a proxy, not an identity annotation.

False discovery rate (FDR) divides wrong positive decisions by all positive decisions in each slice. For both priority and standard human review, a flagged comment is a false discovery when no label is true. Ratios compare comments with an identity term to those without one.

| decision basis | tier | FDR with term (wrong / decisions) | FDR without term | ratio | ratio 95% CI |
|---|---|---:|---:|---:|---|
| population thresholds | priority_review | 0.122 (44/362) | 0.112 (265/2362) | 1.08 | 0.80 to 1.46 |
| after subgroup threshold, before rules | priority_review | 0.122 (44/362) | 0.112 (265/2362) | 1.08 | 0.80 to 1.46 |
| final routing | predicted_positive | 0.335 (271/809) | 0.328 (1806/5503) | 1.02 | 0.92 to 1.13 |
| final routing | priority_review | 0.122 (45/369) | 0.126 (310/2454) | 0.97 | 0.72 to 1.29 |
| final routing | human_review | 0.514 (226/440) | 0.491 (1496/3049) | 1.05 | 0.95 to 1.15 |

Conventional false-positive rate (FPR) uses all actually negative comments in the slice as its denominator, including allowed comments. Here 'negative' means no positive label. It measures the fraction of clean comments sent to each tier. Both review tiers use the same any-positive-label definition of correctness.

| final tier | FPR with term (clean routed / clean total) | FPR without term | ratio | ratio 95% CI |
|---|---:|---:|---:|---|
| predicted_positive | 0.065 (271/4144) | 0.034 (1806/53591) | 1.94 | 1.71 to 2.20 |
| priority_review | 0.011 (45/4144) | 0.006 (310/53591) | 1.88 | 1.38 to 2.56 |
| human_review | 0.055 (226/4144) | 0.028 (1496/53591) | 1.95 | 1.70 to 2.24 |

The subgroup threshold uses the run's priority_review precision target of 0.95 on identity-term comments in the threshold-selection split, at or above each population threshold. A label without a qualifying slice threshold cannot trigger priority_review on that slice. These are empirical selection targets, not statistical guarantees on test data.

Slice thresholds: toxic: 0.9801; severe_toxic: no qualifying slice threshold; obscene: 0.9952; threat: no qualifying slice threshold; insult: no qualifying slice threshold; identity_hate: no qualifying slice threshold. They removed 0 of 1040 population priority-review candidates on selection data and 0 of 2724 on test data. Of the removed test decisions, 0 had a positive label.

False omission rate uses allowed comments as its denominator. It is 0.058 with a term (240/4113 allowed comments have a positive label) and 0.033 without (1768/53553). The overall positive-label base rates are 0.158 and 0.093, respectively.

The five terms with the most positive decisions are shown below; comments can match multiple terms.

| term | wrong decisions | positive decisions | FDR |
|---|---:|---:|---:|
| gay | 88 | 360 | 0.244 |
| black | 34 | 78 | 0.436 |
| white | 27 | 70 | 0.386 |
| woman | 21 | 47 | 0.447 |
| women | 14 | 39 | 0.359 |

**Prevalence and predicted volume.** Expected positives are the sum of calibrated probabilities on test rows. Overshoot is expected divided by actual positives, minus one. This measures aggregate probability calibration; it does not by itself establish why the distributions differ.

| label | train-file prevalence | test prevalence | actual positives | model-expected positives | volume overshoot |
|---|---:|---:|---:|---:|---:|
| toxic | 0.0958 | 0.0952 | 6090 | 8715.3 | 0.431 |
| severe_toxic | 0.0100 | 0.0057 | 367 | 702.9 | 0.915 |
| obscene | 0.0529 | 0.0577 | 3691 | 4359.8 | 0.181 |
| threat | 0.0030 | 0.0033 | 211 | 307.6 | 0.458 |
| insult | 0.0494 | 0.0536 | 3427 | 3778.6 | 0.103 |
| identity_hate | 0.0088 | 0.0111 | 712 | 776.5 | 0.091 |

**Queue simulation.** 4 reviewers, 2 min per item, 8 h, capacity 120/h. The sampled pool has 6312 queued comments and 916 high-risk comments; high-risk means harm proxy >= 5.0. The run uses 5 seeds with identical arrivals and handle times for every ordering within each seed.

Arrival rates are post-admission review jobs per hour, sampled from both priority_review and human_review. They are not incoming platform-comment rates. Every admitted item uses reviewer capacity. Primary ordering (severity): highest max predicted probability times severity weight first; ties use arrival order. Priority-band ordering: priority_review first, then human_review; within each tier, higher max predicted probability times severity weight first; ties use arrival order. Arrival assumption: Post-admission Poisson review arrivals; jobs sampled with replacement from the combined priority_review and human_review pool. Handle times: deterministic. Harm proxy: max severity weight over TRUE labels (0 if clean); not real-world harm. Waits are minutes until review starts, measured over all jobs that started, including reviews still in progress at the horizon. High-risk left includes jobs still in service; backlog counts jobs not yet started. The table reports seed means and standard deviations. Completion counts at a finite horizon do not establish queue stability.

| load/h | ordering | handled | high-risk handled | high-risk left | harm / reviewer-h | high-risk wait p50 | high-risk wait p90 | wait p90, all items | backlog at end |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | fifo | 468 ± 26 | 66.2 ± 8.7 | 0.0 ± 0.0 | 27.5 ± 2.5 | 0.0 ± 0.0 | 0.3 ± 0.2 | 0.3 ± 0.1 | 0 ± 1 |
| 60 | prob | 468 ± 26 | 66.2 ± 8.7 | 0.0 ± 0.0 | 27.5 ± 2.5 | 0.0 ± 0.0 | 0.2 ± 0.1 | 0.3 ± 0.1 | 0 ± 1 |
| 60 | severity | 468 ± 26 | 66.2 ± 8.7 | 0.0 ± 0.0 | 27.5 ± 2.5 | 0.0 ± 0.0 | 0.2 ± 0.1 | 0.3 ± 0.1 | 0 ± 1 |
| 60 | priority | 468 ± 26 | 66.2 ± 8.7 | 0.0 ± 0.0 | 27.5 ± 2.5 | 0.0 ± 0.0 | 0.2 ± 0.1 | 0.3 ± 0.1 | 0 ± 1 |
| 108 | fifo | 867 ± 20 | 122.6 ± 13.2 | 0.4 ± 0.5 | 50.6 ± 3.1 | 1.6 ± 0.8 | 5.9 ± 2.7 | 6.0 ± 2.9 | 1 ± 2 |
| 108 | prob | 867 ± 20 | 122.8 ± 13.1 | 0.2 ± 0.4 | 50.7 ± 3.1 | 0.3 ± 0.0 | 1.2 ± 0.1 | 3.9 ± 1.0 | 1 ± 2 |
| 108 | severity | 867 ± 20 | 122.8 ± 13.1 | 0.2 ± 0.4 | 50.6 ± 3.1 | 0.2 ± 0.0 | 1.0 ± 0.1 | 4.4 ± 2.0 | 1 ± 2 |
| 108 | priority | 867 ± 20 | 122.8 ± 13.1 | 0.2 ± 0.4 | 50.7 ± 3.1 | 0.2 ± 0.0 | 1.1 ± 0.1 | 4.4 ± 2.0 | 1 ± 2 |
| 180 | fifo | 953 ± 4 | 134.0 ± 9.1 | 66.8 ± 9.6 | 55.8 ± 2.1 | 82.2 ± 5.8 | 140.0 ± 8.5 | 140.4 ± 7.9 | 465 ± 41 |
| 180 | prob | 953 ± 4 | 176.2 ± 10.9 | 24.6 ± 4.1 | 68.5 ± 2.6 | 0.5 ± 0.2 | 2.3 ± 0.7 | 11.7 ± 1.5 | 465 ± 41 |
| 180 | severity | 953 ± 4 | 184.6 ± 10.0 | 16.2 ± 4.5 | 71.0 ± 2.1 | 0.5 ± 0.3 | 1.6 ± 0.3 | 11.1 ± 1.7 | 465 ± 41 |
| 180 | priority | 953 ± 4 | 184.2 ± 10.1 | 16.6 ± 4.4 | 70.9 ± 2.1 | 0.5 ± 0.2 | 2.0 ± 0.3 | 11.3 ± 1.2 | 465 ± 41 |

Paired comparisons use shared arrivals within each seed. Differences are first ordering minus second. Each cell gives the mean difference and the number of seeds in which the first ordering was better on that metric.

| comparison | harm / reviewer-h difference | high-risk handled difference | high-risk wait p90 difference |
|---|---:|---:|---:|
| severity_vs_fifo@60 | 0.00; better 0/5 | 0.00; better 0/5 | -0.14; better 4/5 |
| severity_vs_prob@60 | 0.00; better 0/5 | 0.00; better 0/5 | 0.00; better 0/5 |
| severity_vs_priority@60 | 0.00; better 0/5 | 0.00; better 0/5 | 0.00; better 0/5 |
| prob_vs_fifo@60 | 0.00; better 0/5 | 0.00; better 0/5 | -0.14; better 4/5 |
| priority_vs_fifo@60 | 0.00; better 0/5 | 0.00; better 0/5 | -0.14; better 4/5 |
| priority_vs_prob@60 | 0.00; better 0/5 | 0.00; better 0/5 | 0.00; better 0/5 |
| severity_vs_fifo@108 | 0.03; better 1/5 | 0.20; better 1/5 | -4.94; better 5/5 |
| severity_vs_prob@108 | -0.01; better 0/5 | 0.00; better 0/5 | -0.19; better 5/5 |
| severity_vs_priority@108 | -0.01; better 0/5 | 0.00; better 0/5 | -0.06; better 2/5 |
| prob_vs_fifo@108 | 0.04; better 2/5 | 0.20; better 1/5 | -4.76; better 5/5 |
| priority_vs_fifo@108 | 0.04; better 2/5 | 0.20; better 1/5 | -4.89; better 5/5 |
| priority_vs_prob@108 | 0.00; better 0/5 | 0.00; better 0/5 | -0.13; better 5/5 |
| severity_vs_fifo@180 | 15.29; better 5/5 | 50.60; better 5/5 | -138.40; better 5/5 |
| severity_vs_prob@180 | 2.56; better 5/5 | 8.40; better 5/5 | -0.65; better 4/5 |
| severity_vs_priority@180 | 0.14; better 4/5 | 0.40; better 2/5 | -0.37; better 5/5 |
| prob_vs_fifo@180 | 12.73; better 5/5 | 42.20; better 5/5 | -137.75; better 5/5 |
| priority_vs_fifo@180 | 15.14; better 5/5 | 50.20; better 5/5 | -138.03; better 5/5 |
| priority_vs_prob@180 | 2.41; better 5/5 | 8.00; better 5/5 | -0.28; better 2/5 |

Harm and high-risk status use the same policy weights as severity ordering. These comparisons are conditional on that weight vector and the stated arrival model.

**Headline time metric.** wait p50 at 108/h, pooled over seeds: severity 0.403 vs FIFO 1.417 min (n=4356 / 4356). Reduction: 71.6%.

Waiting time uses all started reviews. Completion latency uses reviews completed within the horizon. Never-started jobs have no observed wait; their counts and age at the horizon remain in the records. Completion means simulated review service finished; actual moderation outcomes are not observed. The two strategies use the same arrivals, but their started and completed subsets may differ.

| load/h | ordering | completed / in progress / not started | high-risk completed / in progress / not started | wait p50 / p90 / p99 (n) | completion latency p50 / p90 / p99 (n) | p99 reliable, wait / completion |
|---:|---|---|---|---|---|---|
| 60 | fifo | 2339 / 8 / 2 | 331 / 0 / 0 | 0.000 / 0.348 / 1.470 (2347) | 2.000 / 2.326 / 3.471 (2339) | yes / yes |
| 60 | prob | 2339 / 8 / 2 | 331 / 0 / 0 | 0.000 / 0.270 / 1.555 (2347) | 2.000 / 2.262 / 3.557 (2339) | yes / yes |
| 60 | severity | 2339 / 8 / 2 | 331 / 0 / 0 | 0.000 / 0.265 / 1.496 (2347) | 2.000 / 2.261 / 3.496 (2339) | yes / yes |
| 60 | priority | 2339 / 8 / 2 | 331 / 0 / 0 | 0.000 / 0.265 / 1.496 (2347) | 2.000 / 2.258 / 3.496 (2339) | yes / yes |
| 108 | fifo | 4337 / 19 / 6 | 613 / 2 / 0 | 1.417 / 6.443 / 13.526 (4356) | 3.420 / 8.465 / 15.528 (4337) | yes / yes |
| 108 | prob | 4337 / 19 / 6 | 614 / 1 / 0 | 0.400 / 3.749 / 42.424 (4356) | 2.401 / 5.751 / 44.659 (4337) | yes / yes |
| 108 | severity | 4337 / 19 / 6 | 614 / 1 / 0 | 0.403 / 3.799 / 38.223 (4356) | 2.406 / 5.806 / 40.351 (4337) | yes / yes |
| 108 | priority | 4337 / 19 / 6 | 614 / 1 / 0 | 0.401 / 3.761 / 38.223 (4356) | 2.402 / 5.773 / 40.351 (4337) | yes / yes |
| 180 | fifo | 4766 / 20 / 2324 | 670 / 2 / 332 | 78.873 / 140.272 / 165.950 (4786) | 80.392 / 141.590 / 167.251 (4766) | yes / yes |
| 180 | prob | 4766 / 20 / 2324 | 881 / 4 / 119 | 0.706 / 11.212 / 118.125 (4786) | 2.706 / 13.146 / 119.750 (4766) | yes / yes |
| 180 | severity | 4766 / 20 / 2324 | 923 / 2 / 79 | 0.690 / 11.055 / 102.447 (4786) | 2.690 / 13.026 / 104.108 (4766) | yes / yes |
| 180 | priority | 4766 / 20 / 2324 | 921 / 2 / 81 | 0.696 / 11.313 / 100.323 (4786) | 2.695 / 13.201 / 102.361 (4766) | yes / yes |

All times are minutes. A p99 with fewer than 100 observations is flagged unreliable. Counts pool seeds, whereas the preceding simulation table reports seed means.

Gates read from `reports/ci-35857528118/policy.yaml`:

For identity FDR ratios, green requires the entire 95% interval to be at or below the ceiling. An interval entirely above it is red; a crossing interval is inconclusive. Subgroups below the minimum count are skipped, and missing estimates are unavailable. Neither state is a passing fairness result.

| gate | measured | requirement | status |
|---|---:|---|---|
| human confirmation before every moderation action | mode=human_confirmation; automatic actions=0 | human_confirmation; required=true; actions=0 | green |
| per-label AP (6 configured labels) | see label table | snapshot/override floors | green |
| toxic precision at human threshold | 0.663 | ≥ 0.643; n ≥ 30 | green |
| obscene precision at human threshold | 0.773 | ≥ 0.753; n ≥ 30 | green |
| insult precision at human threshold | 0.878 | ≥ 0.858; n ≥ 30 | green |
| severe_toxic precision at human threshold | n/a | ≥ n/a; n ≥ 30 | not set |
| threat precision at human threshold | n/a | ≥ n/a; n ≥ 30 | not set |
| identity_hate precision at human threshold | n/a | ≥ n/a; n ≥ 30 | not set |
| severe_toxic predicted volume overshoot | 0.915 | ≤ 1.000 | green |
| priority_review precision | 0.874 | diagnostic; no test floor | not set |
| human_review precision | 0.506 | diagnostic; no test floor | not set |
| harm_per_reviewer_hour, severity / FIFO at 180.000/h | 1.274 | ≥ 1.000 | green |
| high_risk_wait_p90, severity / FIFO at 108.000/h | 0.168 | ≤ 1.000 | green |
| high-risk completed, severity / FIFO at 108.000/h (primary) | 1.002 | ≥ 1.000 | green |
| high-risk completed, severity / FIFO at 180.000/h (thesis) | 1.378 | ≥ 1.000 | green |
| ordering vs fifo at 180.000/h: high-risk completed mean diff | 50.600 | ≥ -1.000 | green |
| ordering vs fifo at 180.000/h: harm per reviewer-hour mean diff | 15.287 | ≥ -0.500 | green |
| ordering vs prob at 180.000/h: high-risk completed mean diff | 8.400 | ≥ -1.000 | green |
| ordering vs prob at 180.000/h: harm per reviewer-hour mean diff | 2.556 | ≥ -0.500 | green |
| ordering vs priority at 180.000/h: high-risk completed mean diff | 0.400 | ≥ -1.000 | green |
| ordering vs priority at 180.000/h: harm per reviewer-hour mean diff | 0.144 | ≥ -0.500 | green |
| completion_ratio at 108.000/h | 0.994 | ≥ 0.970 | green |
| backlog_end at 108.000/h | 1.200 | ≤ 10 | green |
| wait_p50 at 108.000/h | 0.403 | ≤ 1.000 | green |
| queue_depth_p95 at 108.000/h | 14.220 | ≤ 39 | green |
| reviewer_utilization at 108.000/h | 0.905 | ≥ 0.600 | green |
| hierarchy violation rate | 0.0000 | ≤ 0.005 | green |
| identity FDR ratio, pooled, final tier | 1.021 [0.92 to 1.13] | ≤ 1.250 | green |
| identity FDR ratio, priority_review, final tier | 0.965 [0.72 to 1.29] | ≤ 1.250 | inconclusive |
| identity FDR ratio, human_review, final tier | 1.047 [0.95 to 1.15] | ≤ 1.250 | green |

Reproducibility checks use `simulation_jobs.csv` with absolute tolerance 1e-09. The gate suite recomputes record-derived metrics and replays the configured scenario. Policy/config hashes and, when required, a clean matching commit are checked separately. This table displays measured criteria; rendering it does not execute or certify those checks.
<!-- results:end -->

## Status

Built:

- [x] Arithmetic precision ceilings over the scored test set (`ceilings.py`)
- [x] Policy schema, loader, validation and routing (`policy.py`, `policy.yaml`)
- [x] Corpus loader, scored-row filter, frozen seeded split (`data.py`)
- [x] TF-IDF + six logistic heads with Platt calibration (`model.py`)
- [x] Threshold selection from precision floors; model tiers; rule priority (`thresholds.py`)
- [x] Review-queue simulator with replayed arrivals and four orderings (`simulate.py`)
- [x] Per-label and per-tier metrics with Wilson intervals (`metrics.py`)
- [x] One-command pipeline with manifest, predictions and report (`pipeline.py`, `scripts/run_pipeline.py`)
- [x] Regression gates in four categories, with provenance checks and controlled failures
- [x] Per-job simulation records, a declared headline time metric, and scenario replay
- [x] Per-label prevalence shift and predicted volume overshoot
- [x] Real-evaluation CI with pinned corpus hashes and archived runs
- [x] Identity-mention FDR and FPR reported per review band; R103 retired, with subgroup threshold selection for the priority band
- [x] Human confirmation required for both review bands, with all admitted comments counted against reviewer capacity
- [x] Historical false-positive audit and character n-gram comparison, recorded in [step 4](record/step4-features.html). Under policy v2, word+char has about 15% fewer high-risk misses at matched volume. At its selected thresholds it needs about 12% more review work, with about 6x pipeline time. It remains an experimental option; the default stays `word`.
- [x] Severity ordering, development-only evaluation, cross-fitted band diagnostics and segment thresholds, evaluated in [step 6](record/step6-priority.html). The matched-score comparison raises priority-band precision from 76.30% to 87.42% while retaining the same 6,312 admitted comments. It does not improve total high-risk recall.

The project record, one HTML page per step with what was done and why, lives
under `record/` ([index](record/index.html)); the README keeps only the
current results block above.

Not done yet:

- [ ] Sentence-embedding signal and multi-model fusion, judged under the same protocol
- [ ] Hierarchy constraint (`severe_toxic` is an exact subset of `toxic`); today only the violation rate is reported
- [ ] Sensitivity sweep over the severity-weight vector
- [ ] Make vocabulary-cap tie selection deterministic across training environments, then verify repeated training. The [retraining check](record/retraining_check.json) records the observed variation.
- [ ] Validate frozen ranking and threshold choices on new independent data
- [ ] Optional independent human evaluation of review worthiness, deferred due to annotation workload. The [record](record/step4-features.html#future-review-evaluation) describes the possible scope; collection and annotation have not started.

## Honest scope and limitations

- **Saved simulation replay and fresh training have different guarantees.**
  Replay gates reproduce queue results from saved predictions. Two clean CI
  runs with identical data, split, seed and recorded dependency versions selected
  different terms tied at the TF-IDF feature cap. Two comments changed admission,
  which also changed the comments sampled by the queue simulation. Comparisons
  in step 6 therefore use the same saved scores and paired arrivals within one
  run. Exact retraining reproducibility remains unresolved.
- **The corpus is not the target domain.** Jigsaw is English Wikipedia talk-page
  argument, mean ~394 characters, written by editors about article disputes.
  Classifier weights do not transfer to short, multilingual, emoji-heavy
  comments with video context. The routing, calibration and capacity methodology
  is the transferable part; that is the only part claimed.
- **Label positivity is a proxy for review value.** Current evaluation uses the
  original Jigsaw labels. An ambiguous comment may merit human review even if
  no violation is confirmed; context may still be missing after review. The
  historical audit judged 46 of 100 sampled false positives toxic, 23 borderline
  and 31 clean. Its 0.948/0.970 precision estimates assume unaudited true positives
  stay correct and extrapolate the sampled verdicts. They are sensitivity
  estimates, not independent measurements of review value. Original labels and
  current acceptance criteria remain unchanged.
- **Exposure is simulated.** Jigsaw has no view counts. Any harm-weighted
  quantity is conditional on a stated exposure model and a stated severity
  weight vector. The current simulation counts weighted labels handled within
  its horizon and does not model views or measure prevented exposure.
  Sensitivity to the weight vector is still to be evaluated.
- **No identity annotations exist in the 2018 corpus.** Subgroup / BPSN / BNSP
  AUCs cannot be computed on it. Fairness work here must either pull the 2019
  Civil Comments release or use a templated probe set, labelled as synthetic.
- **No temporal split.** Adversarial drift is the defining property of this
  domain and the 2018 files carry no reliable timestamp. Temporal robustness
  has not been evaluated. Obfuscation perturbations could provide a separate,
  weaker robustness check.

## Development notes

Built with AI-assisted development tooling; problem framing, design decisions,
code review and validation are my own.

## Licence

MIT
