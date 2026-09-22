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
- **Rules and models coexist.** Models supply scores. Threat and label
  inconsistency rules promote comments into priority review.
  Where a slice needs extra care, the floor is required on that slice
  (`subgroup_thresholds`) rather than typed in as a probability constant.
- **All review work consumes capacity.** Both review bands enter the simulator.
  Priority ordering serves the priority band first, then uses predicted
  severity within each band. FIFO, probability and severity orderings provide
  comparisons on the same admitted comments and arrivals. Confidence and harm
  are different signals; priority ordering may lose to severity ordering.

## Reproducing the human-review experiment

The current experiment asks: **with the same reviewer capacity, does priority
review handle at least as much high-risk work as first-in-first-out, and reach
it sooner?** Every
assumption of the protocol is a line in `configs/baseline.yaml` or
`review_router/policy.yaml`, and the pipeline module docstring walks the stages.

```bash
pip install -e ".[ml,dev]"
# Put the official Jigsaw 2018 files in data/jigsaw/: train.csv, test.csv, test_labels.csv
# (Kaggle: jigsaw-toxic-comment-classification-challenge; the corpus is not redistributed here.)
python scripts/run_pipeline.py --config configs/baseline.yaml
```

One run writes `reports/<run_id>/` with:

| file | contents |
|---|---|
| `manifest.json` | data file hashes, split label counts, git commit, dependency versions, seeds, the config |
| `config.yaml`, `policy.yaml` | snapshots of the exact experiment and policy inputs |
| `splits.csv` | every training id and its split (`train` 60% / `calib` 20% / `thresh` 20%) |
| `thresholds.json` | per-label thresholds for each tier, `null` where the precision floor is unreachable |
| `predictions.csv` | one row per scored test comment: labels, calibrated probabilities, model tier, matched rules, final tier and human-review requirement |
| `model.pkl` | fitted TF-IDF vocabulary, classifiers and calibrators |
| `report.json` | decision contract, total review workload, per-label and tier quality with Wilson intervals, subgroup diagnostics, rule effects, four-strategy simulation |
| `report.md` | the same, as tables |

The pipeline is: TF-IDF + six logistic heads fitted on `train`, Platt-calibrated
on `calib`, tier thresholds chosen on `thresh` from the precision floors in
`policy.yaml`, rules applied with priority, then a fixed-capacity review queue
(4 reviewers, 2 min per item, 8 h, loads of 60 / 108 / 180 per hour, 5 seeds)
replayed under FIFO, probability, severity and priority-band ordering. Every
strategy sees the identical arrivals. Model fitting, calibration and threshold
selection use separate portions of the training data. The scored test set has
now been inspected across several policy iterations, so these results are an
iterative benchmark rather than a fresh, untouched final evaluation. Further
model and threshold selection should use development data, with new held-out
data needed for an independent confirmation.

The 90% and 95% targets select thresholds independently for each label.
Combining labels, separating review bands and applying rules can change the
final tiers' precision. For both bands, review is useful when any annotated
label is positive. The report shows final-tier quality
on both the threshold-selection split and the held-out test set; these targets
are not guarantees. Simulation wait quantiles include completed jobs only, so
read them alongside the unhandled high-risk count and the remaining backlog.

Arrival rates are measured after admission to the combined human queue. They
are conditional ranking experiments, not a fixed incoming-comment traffic
comparison with the old automatic-action policy. The report records the
admission fraction and equivalent incoming rate so workload changes remain visible.

The initial benchmark requirements are a harm-per-reviewer-hour ratio of at
least 1.0 versus FIFO at overload, high-risk wait p90 no higher than FIFO near
capacity, and high-risk completed counts no lower than FIFO at both loads.
The completion checks prevent shorter waits from hiding unfinished work.
These are empirical comparisons, not statistical noninferiority guarantees.
Precision and coverage are diagnostic; there is no 99% test-precision gate.

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

<!-- results:start -->
## Review-routing results (Jigsaw scored test rows), run `20260922T130141204182Z-human-review-baseline`

Generated from this run's `report.json` and `manifest.json`. The run used commit `a7ee1ab214b0` with uncommitted changes (manifest `git_dirty: true`), seed 20260922, scikit-learn 1.9.1. The input files' SHA-256 hashes are recorded in the manifest.

Development split: 95,743 / 31,914 / 31,914 rows for train / calibration / threshold selection; 63,978 scored test rows evaluated separately. Precision targets come from the run snapshot `reports/20260922T130141204182Z-human-review-baseline/policy.yaml`.

**Decision contract and evaluation criteria.** Every moderation action requires human confirmation. The model routes comments to priority_review, human_review or allow; priority_review schedules a human review earlier and does not authorize an automatic moderation action. Automatic actions must remain zero.

The run's per-label selection targets are 0.95 for priority_review and 0.90 for human_review. These are empirical targets on the selection split, not guarantees of test precision. Per-tier test precision remains diagnostic when its gate is not set.

Comparison gates come from `reports/20260922T130141204182Z-human-review-baseline/policy.yaml`. At equal reviewer capacity, the router/FIFO harm-per-reviewer-hour ratio at 180.000/h must be ≥ 1.000; the high-risk wait p90 ratio at 108.000/h must be ≤ 1.000. The high-risk completed-count ratio must be ≥ 1.000 at both named loads. These comparisons use ratios of seed means. They are descriptive checks that the measured result is no worse than FIFO, not statistical non-inferiority tests. Historical automatic-enforcement results use a different contract.

**Per label.** Thresholds were selected for precision targets 0.90 (human) and 0.95 (priority). Selection and test AP are both shown. An AP difference describes a measured performance gap; it does not by itself identify the cause.

| label | positives | AP selection | AP test | ROC-AUC test | human thr | P@human | R@human | priority thr | P@priority | R@priority |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| toxic | 6090 | 0.859 | 0.752 | 0.957 | 0.6576 | 0.663 | 0.680 | 0.8474 | 0.755 | 0.574 |
| severe_toxic | 367 | 0.492 | 0.311 | 0.982 | n/a | n/a | n/a | n/a | n/a | n/a |
| obscene | 3691 | 0.874 | 0.773 | 0.972 | 0.6573 | 0.773 | 0.616 | 0.9258 | 0.875 | 0.474 |
| threat | 211 | 0.459 | 0.458 | 0.992 | n/a | n/a | n/a | n/a | n/a | n/a |
| insult | 3427 | 0.774 | 0.696 | 0.965 | 0.9176 | 0.879 | 0.290 | 0.9903 | 0.934 | 0.144 |
| identity_hate | 712 | 0.415 | 0.480 | 0.974 | n/a | n/a | n/a | n/a | n/a | n/a |

**Total human-review workload.** Both flagged tiers enter the same review pool and consume reviewer capacity. The report records 6310 of 63978 comments requiring review, a fraction of 0.099: 4625 priority reviews and 1685 standard reviews. Of the flagged comments, 4233 have at least one positive label. Combined review precision is 0.671, with 95% interval 0.659 to 0.682. Recorded automatic actions: 0.


**Routing tiers.** Matching policy rules promoted 44 comments to priority review and added 18 comments from allow to the review pool. These aggregate counts can overlap. Both review tiers count a comment as positive when any label is true.

| tier | n | coverage | precision (test) | 95% CI | precision (selection split) | target |
|---|---:|---:|---:|---|---:|---:|
| priority_review | 4625 | 0.072 | 0.763 | 0.750 to 0.775 | 0.9724 | 0.950 |
| human_review | 1685 | 0.026 | 0.419 | 0.396 to 0.443 | 0.7974 | 0.900 |
| allow | 57668 | 0.901 | n/a | n/a | n/a | n/a |

Per-label precision targets need not hold after pooling labels, partitioning the review pool or applying rules. Test precision is measured separately. The gates below retain the declared limits; changes to models, calibration or threshold selection need a fresh evaluation.

**Identity mentions.** The run recorded 63 whole-word terms in its `report.json`. This word-list diagnostic is a proxy, not an identity annotation.

False discovery rate (FDR) divides wrong positive decisions by all positive decisions in each slice. For both priority and standard human review, a flagged comment is a false discovery when no label is true. Ratios compare comments with an identity term to those without one.

| decision basis | tier | FDR with term (wrong / decisions) | FDR without term | ratio | ratio 95% CI |
|---|---|---:|---:|---:|---|
| population thresholds | priority_review | 0.242 (144/595) | 0.237 (960/4049) | 1.02 | 0.88 to 1.19 |
| after subgroup threshold, before rules | priority_review | 0.212 (113/532) | 0.237 (960/4049) | 0.90 | 0.75 to 1.06 |
| final routing | predicted_positive | 0.336 (272/810) | 0.328 (1805/5500) | 1.02 | 0.92 to 1.14 |
| final routing | priority_review | 0.212 (114/537) | 0.241 (984/4088) | 0.88 | 0.74 to 1.05 |
| final routing | human_review | 0.579 (158/273) | 0.581 (821/1412) | 1.00 | 0.89 to 1.11 |

Conventional false-positive rate (FPR) uses all actually negative comments in the slice as its denominator, including allowed comments. Here 'negative' means no positive label. It measures the fraction of clean comments sent to each tier. Both review tiers use the same any-positive-label definition of correctness.

| final tier | FPR with term (clean routed / clean total) | FPR without term | ratio | ratio 95% CI |
|---|---:|---:|---:|---|
| predicted_positive | 0.066 (272/4144) | 0.034 (1805/53591) | 1.95 | 1.72 to 2.20 |
| priority_review | 0.028 (114/4144) | 0.018 (984/53591) | 1.50 | 1.24 to 1.81 |
| human_review | 0.038 (158/4144) | 0.015 (821/53591) | 2.49 | 2.11 to 2.94 |

The subgroup threshold uses the run's priority_review precision target of 0.95 on identity-term comments in the threshold-selection split, at or above each population threshold. A label without a qualifying slice threshold cannot trigger priority_review on that slice. These are empirical selection targets, not statistical guarantees on test data.

Slice thresholds: toxic: 0.8949; severe_toxic: no qualifying slice threshold; obscene: 0.9460; threat: no qualifying slice threshold; insult: no qualifying slice threshold; identity_hate: no qualifying slice threshold. They removed 16 of 1635 population priority-review candidates on selection data and 63 of 4644 on test data. Of the removed test decisions, 32 had a positive label.

False omission rate uses allowed comments as its denominator. It is 0.058 with a term (240/4112 allowed comments have a positive label) and 0.033 without (1770/53556). The overall positive-label base rates are 0.158 and 0.093, respectively.

The five terms with the most positive decisions are shown below; comments can match multiple terms.

| term | wrong decisions | positive decisions | FDR |
|---|---:|---:|---:|
| gay | 88 | 360 | 0.244 |
| black | 35 | 79 | 0.443 |
| white | 27 | 70 | 0.386 |
| woman | 21 | 47 | 0.447 |
| women | 14 | 39 | 0.359 |

**Queue simulation.** 4 reviewers, 2 min per item, 8 h, capacity 120/h. The sampled pool has 6310 queued comments and 916 high-risk comments; high-risk means harm proxy >= 5.0. The run uses 5 seeds with identical arrivals and handle times for every ordering within each seed.

Arrival rates are post-admission review jobs per hour, sampled from both priority_review and human_review. They are not incoming platform-comment rates. Every admitted item uses reviewer capacity. Priority ordering: priority_review first, then human_review; within each tier, higher max predicted probability times severity weight first; ties use arrival order. Arrival assumption: Post-admission Poisson review arrivals; jobs sampled with replacement from the combined priority_review and human_review pool. Handle times: deterministic. Harm proxy: max severity weight over TRUE labels (0 if clean); not real-world harm. Waits are minutes until review starts, measured over completed items. High-risk left includes jobs still in service; backlog counts jobs not yet started. The table reports seed means and standard deviations. Completion counts at a finite horizon do not establish queue stability.

| load/h | ordering | handled | high-risk handled | high-risk left | harm / reviewer-h | high-risk wait p50 | high-risk wait p90 | wait p90, all items | backlog at end |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | fifo | 468 ± 26 | 66.6 ± 13.5 | 0.0 ± 0.0 | 27.7 ± 3.5 | 0.0 ± 0.0 | 0.3 ± 0.1 | 0.3 ± 0.2 | 0 ± 1 |
| 60 | prob | 468 ± 26 | 66.6 ± 13.5 | 0.0 ± 0.0 | 27.7 ± 3.5 | 0.0 ± 0.0 | 0.2 ± 0.1 | 0.3 ± 0.1 | 0 ± 1 |
| 60 | severity | 468 ± 26 | 66.6 ± 13.5 | 0.0 ± 0.0 | 27.7 ± 3.5 | 0.0 ± 0.0 | 0.2 ± 0.1 | 0.3 ± 0.1 | 0 ± 1 |
| 60 | priority | 468 ± 26 | 66.6 ± 13.5 | 0.0 ± 0.0 | 27.7 ± 3.5 | 0.0 ± 0.0 | 0.2 ± 0.1 | 0.3 ± 0.1 | 0 ± 1 |
| 108 | fifo | 867 ± 20 | 123.4 ± 17.7 | 0.4 ± 0.5 | 50.4 ± 3.5 | 1.3 ± 0.4 | 5.6 ± 2.0 | 6.0 ± 2.9 | 1 ± 2 |
| 108 | prob | 867 ± 20 | 123.6 ± 17.4 | 0.2 ± 0.4 | 50.5 ± 3.4 | 0.2 ± 0.0 | 1.1 ± 0.1 | 3.9 ± 1.1 | 1 ± 2 |
| 108 | severity | 867 ± 20 | 123.6 ± 17.4 | 0.2 ± 0.4 | 50.5 ± 3.4 | 0.2 ± 0.1 | 1.0 ± 0.1 | 4.0 ± 1.4 | 1 ± 2 |
| 108 | priority | 867 ± 20 | 123.6 ± 17.4 | 0.2 ± 0.4 | 50.5 ± 3.4 | 0.2 ± 0.1 | 1.1 ± 0.1 | 4.0 ± 1.4 | 1 ± 2 |
| 180 | fifo | 953 ± 4 | 137.6 ± 3.0 | 69.4 ± 8.6 | 56.1 ± 1.3 | 76.8 ± 5.2 | 138.5 ± 9.5 | 139.9 ± 7.8 | 465 ± 41 |
| 180 | prob | 953 ± 4 | 183.6 ± 6.1 | 23.4 ± 4.5 | 69.6 ± 2.0 | 0.5 ± 0.2 | 2.1 ± 0.5 | 10.2 ± 0.8 | 465 ± 41 |
| 180 | severity | 953 ± 4 | 189.0 ± 5.7 | 18.0 ± 5.1 | 71.6 ± 1.6 | 0.5 ± 0.2 | 1.9 ± 0.7 | 11.5 ± 0.7 | 465 ± 41 |
| 180 | priority | 953 ± 4 | 184.2 ± 5.6 | 22.8 ± 5.3 | 70.2 ± 1.7 | 0.5 ± 0.2 | 1.6 ± 0.3 | 10.3 ± 0.8 | 465 ± 41 |

Paired comparisons use shared arrivals within each seed. Differences are first ordering minus second. Each cell gives the mean difference and the number of seeds in which the first ordering was better on that metric.

| comparison | harm / reviewer-h difference | high-risk handled difference | high-risk wait p90 difference |
|---|---:|---:|---:|
| priority_vs_fifo@60 | 0.00; better 0/5 | 0.00; better 0/5 | -0.08; better 5/5 |
| priority_vs_prob@60 | 0.00; better 0/5 | 0.00; better 0/5 | 0.00; better 0/5 |
| priority_vs_severity@60 | 0.00; better 0/5 | 0.00; better 0/5 | 0.00; better 1/5 |
| severity_vs_fifo@60 | 0.00; better 0/5 | 0.00; better 0/5 | -0.08; better 5/5 |
| prob_vs_fifo@60 | 0.00; better 0/5 | 0.00; better 0/5 | -0.08; better 5/5 |
| severity_vs_prob@60 | 0.00; better 0/5 | 0.00; better 0/5 | -0.00; better 1/5 |
| priority_vs_fifo@108 | 0.06; better 2/5 | 0.20; better 1/5 | -4.55; better 5/5 |
| priority_vs_prob@108 | 0.01; better 1/5 | 0.00; better 0/5 | -0.04; better 3/5 |
| priority_vs_severity@108 | 0.00; better 0/5 | 0.00; better 0/5 | 0.01; better 0/5 |
| severity_vs_fifo@108 | 0.06; better 2/5 | 0.20; better 1/5 | -4.56; better 5/5 |
| prob_vs_fifo@108 | 0.04; better 2/5 | 0.20; better 1/5 | -4.51; better 5/5 |
| severity_vs_prob@108 | 0.01; better 1/5 | 0.00; better 0/5 | -0.05; better 3/5 |
| priority_vs_fifo@180 | 14.16; better 5/5 | 46.60; better 5/5 | -136.90; better 5/5 |
| priority_vs_prob@180 | 0.62; better 5/5 | 0.60; better 3/5 | -0.47; better 5/5 |
| priority_vs_severity@180 | -1.38; better 0/5 | -4.80; better 0/5 | -0.26; better 4/5 |
| severity_vs_fifo@180 | 15.53; better 5/5 | 51.40; better 5/5 | -136.64; better 5/5 |
| prob_vs_fifo@180 | 13.54; better 5/5 | 46.00; better 5/5 | -136.43; better 5/5 |
| severity_vs_prob@180 | 1.99; better 5/5 | 5.40; better 5/5 | -0.21; better 4/5 |

Harm and high-risk status use the same policy weights as severity ordering. These comparisons are conditional on that weight vector and the stated arrival model.

Gates read from `reports/20260922T130141204182Z-human-review-baseline/policy.yaml`:

For identity FDR ratios, green requires the entire 95% interval to be at or below the ceiling. An interval entirely above it is red; a crossing interval is inconclusive. Subgroups below the minimum count are skipped, and missing estimates are unavailable. Neither state is a passing fairness result.

| gate | measured | requirement | status |
|---|---:|---|---|
| human confirmation before every moderation action | mode=human_confirmation; automatic actions=0 | human_confirmation; required=true; actions=0 | green |
| per-label AP (6 configured labels) | see label table | snapshot/override floors | green |
| priority_review precision | 0.763 | diagnostic; no test floor | not set |
| human_review precision | 0.419 | diagnostic; no test floor | not set |
| harm_per_reviewer_hour, priority / FIFO at 180.000/h | 1.252 | ≥ 1.000 | green |
| high_risk_wait_p90, priority / FIFO at 108.000/h | 0.189 | ≤ 1.000 | green |
| high-risk completed, priority / FIFO at 108.000/h (primary) | 1.002 | ≥ 1.000 | green |
| high-risk completed, priority / FIFO at 180.000/h (thesis) | 1.339 | ≥ 1.000 | green |
| queue_depth_p95 at 108.000/h | 14.220 | ≤ n/a | not set |
| reviewer_utilization at 108.000/h | 0.905 | ≥ 0.600 | green |
| hierarchy violation rate | 0.0000 | ≤ 0.005 | green |
| identity FDR ratio, pooled, final tier | 1.023 [0.92 to 1.14] | ≤ 1.250 | green |
| identity FDR ratio, priority_review, final tier | 0.882 [0.74 to 1.05] | ≤ 1.250 | green |
| identity FDR ratio, human_review, final tier | 0.995 [0.89 to 1.11] | ≤ 1.250 | green |
<!-- results:end -->

## Status

Built:

- [x] Arithmetic precision ceilings over the scored test set (`ceilings.py`)
- [x] Policy schema, loader, validation and routing (`policy.py`, `policy.yaml`)
- [x] Corpus loader, scored-row filter, frozen seeded split (`data.py`)
- [x] TF-IDF + six logistic heads with Platt calibration (`model.py`)
- [x] Threshold selection from precision floors; model tiers; rule priority (`thresholds.py`)
- [x] Review-queue simulator with replayed arrivals and three orderings (`simulate.py`)
- [x] Per-label and per-tier metrics with Wilson intervals (`metrics.py`)
- [x] One-command pipeline with manifest, predictions and report (`pipeline.py`, `scripts/run_pipeline.py`)
- [x] Regression-gate harness (`tests/test_gate.py`) with floors set from the measured round-1 run
- [x] Identity-mention FDR and FPR reported per review band; R103 retired, with subgroup threshold selection for the priority band
- [x] Human confirmation required for both review bands, with all admitted comments counted against reviewer capacity

The project record, one HTML page per step with what was done and why, lives
under `record/` ([index](record/index.html)); the README keeps only the
current results block above.

Not done yet:

- [ ] Sentence-embedding signal and multi-model fusion, judged under the same protocol
- [ ] Hierarchy constraint (`severe_toxic` is an exact subset of `toxic`); today only the violation rate is reported
- [ ] Sensitivity sweep over the severity-weight vector
- [ ] Improve priority review using development-data calibration and ranking diagnostics, then validate frozen choices on new independent data

## Honest scope and limitations

- **The corpus is not the target domain.** Jigsaw is English Wikipedia talk-page
  argument, mean ~394 characters, written by editors about article disputes.
  Classifier weights do not transfer to short, multilingual, emoji-heavy
  comments with video context. The routing, calibration and capacity methodology
  is the transferable part; that is the only part claimed.
- **Labels are a perception signal, not a policy signal.** Annotators were asked
  whether a comment was rude enough to make them leave a discussion. Inter-
  annotator agreement puts a ceiling on measured performance that is label
  noise, not model capacity.
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
