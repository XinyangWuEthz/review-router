# Experiment protocol

The [README](../README.md) describes the release method and main results.
This page keeps the execution contract, metric definitions and validation
details. Run commands from the repository root.

## Run the experiment

The current experiment asks: **with the same reviewer capacity, does the router
(the review queue served in predicted-severity order) handle at least as much
high-risk work as first-in-first-out, and reach it sooner?** Every assumption
of the protocol is a line in `configs/baseline.yaml` or
`review_router/policy.yaml`, and the pipeline module docstring walks the stages.

```bash
pip install -c configs/frozen-ml.txt -e ".[ml,dev]"
# Put the official Jigsaw 2018 files in data/jigsaw/: train.csv, test.csv, test_labels.csv
# (Kaggle: jigsaw-toxic-comment-classification-challenge; the corpus is not redistributed here.)
# One-time restore of the accepted scorer from the v0.1.0 release:
gh release download v0.1.0 --repo XinyangWuEthz/review-router \
  --pattern frozen-baseline-v0.1.0.zip --dir models
python scripts/restore_frozen_model.py --archive models/frozen-baseline-v0.1.0.zip
python scripts/run_pipeline.py --config configs/baseline.yaml
```

For development iterations, run the same protocol without scoring the test rows:

```bash
python scripts/run_pipeline.py --config configs/dev.yaml
```

Development runs stop before the test rows are scored: they write the
thresholds, the model and the development sections of the report, but no
predictions, simulation or test section, and they are not a gate input. Use
them to compare threshold rules and ranking diagnostics on the selection split;
they do not simulate a review queue.

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
The [freeze record](../record/frozen-router.md) documents the real-data replay and
its numerical limits.

The release asset preserves the pinned scorer beyond GitHub Actions artifact
retention. The restore command verifies the files against the checked-in lock.
An existing accepted run can also be restored with `--source-run <run_dir>`.
CI uses the frozen scorer by default.
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
These are empirical regression comparisons. They do not establish statistical noninferiority.
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

The renderer updates only the README results markers and defaults to the compact
overview. `--detailed` renders all diagnostics and gates; an explicit `--policy`
also selects the detailed view with overridden comparison gates. Tier targets
always come from the saved run. Use `--readme <path>` to write into another
Markdown file containing the same results markers.

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


## Reproduce the bounded weight check

This experiment replays the accepted frozen-model evaluation from commit
`f19cc5b3f53b`. It does not train, recalibrate, reselect thresholds or change
admissions. The [declared protocol](../configs/severity-sensitivity.yaml) uses
five weight vectors, 20 paired arrival seeds and loads of 108 and 180 review
jobs per hour. True-label utility and high-risk membership stay fixed across
all arms, including flat weights. Flat weights appear once as probability
ordering. FIFO and band-first ordering are fixed comparators.

```bash
gh release download v0.1.0 --repo XinyangWuEthz/review-router \
  --pattern baseline-evaluation-v0.1.0.zip --dir reports
unzip reports/baseline-evaluation-v0.1.0.zip -d reports/ci-35863370097
python scripts/weight_sensitivity.py reports/ci-35863370097
```

The command writes [the report](../record/severity-sensitivity.md) and
[per-seed results with source hashes](../record/severity-sensitivity.json).
Unlike a full baseline run, this replay needs only the saved evaluation bundle.
The 20 seeds describe variation in simulated arrivals, not uncertainty about
Jigsaw labels or unseen data. Wait quantiles include only jobs that started;
never-started counts and unfinished ages expose work that the queue has not
reached. The default weight vector is retained independently of the ranking in
this bounded comparison.

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

