# review-router

A capacity-constrained content-policy router that sends comments to
**allow / human-review / auto-action** and evaluates the resulting queue with a
fixed number of reviewers. The current baseline uses TF-IDF classifiers;
additional model signals are planned.

A classifier's ROC-AUC does not determine the precision or workload at its
chosen operating point. This project measures the routing layer above a
classifier: what gets acted on, what reaches human review, and what remains
unfinished when reviewer capacity is limited.

## The result the project is built around

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

- **Precision floors, not thresholds.** Each tier declares a precision floor
  (`auto_action: 0.99`, `human_review: 0.90`) and the threshold is read off the
  precision-recall curve. Coverage is the dependent variable, never tuned.
  The floors come from cost asymmetry: a false auto-action is an unappealed
  wrong enforcement; a false queue entry costs a few reviewer-minutes.
- **Policy as reviewable config.** Tiers, severity weights, rules and CI floors
  live in [`review_router/policy.yaml`](review_router/policy.yaml); every rule
  carries a prose rationale for *why its tier is what it is*. The loader rejects
  unknown actions and operators, and that rejection is tested.
- **Rules and models coexist.** Models supply scores and rules override the
  resulting tier when the policy requires human review.
  Where a slice needs extra care, the floor is required on that slice
  (`subgroup_thresholds`) rather than typed in as a probability constant.
- **Capacity is the point.** An unbounded queue makes routing trivial. Arrival
  rate, reviewer count and handle time are explicit inputs, and the simulation
  reports the overflow point at which the backlog stops clearing.

## Reproducing the round-1 experiment

Round 1 asks one question: **with the same reviewer capacity, does risk-ordered
review reach high-harm comments earlier than first-in-first-out?** Every
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
| `predictions.csv` | one row per scored test comment: labels, calibrated probabilities, model tier, matched rules, final tier |
| `model.pkl` | fitted TF-IDF vocabulary, classifiers and calibrators |
| `report.json` | per-label AP / precision / recall, tier precision with Wilson intervals, rule effects, hierarchy-violation rate, the three-strategy simulation table |
| `report.md` | the same, as tables |

The pipeline is: TF-IDF + six logistic heads fitted on `train`, Platt-calibrated
on `calib`, tier thresholds chosen on `thresh` from the precision floors in
`policy.yaml`, rules applied with priority, then a fixed-capacity review queue
(4 reviewers, 2 min per item, 8 h, loads of 60 / 108 / 180 per hour, 5 seeds)
replayed under FIFO, probability-first and severity-first ordering. Every
strategy sees the identical arrivals. Model fitting, calibration and threshold
selection use separate portions of the training data. The scored test set has
now been inspected across several policy iterations, so these results are an
iterative benchmark rather than a fresh, untouched final evaluation. Further
model and threshold selection should use development data, with new held-out
data needed for an independent confirmation.

The 90% and 99% targets select thresholds independently for each label.
Combining labels, removing automatic actions from the human queue and applying
rules can lower the final tiers' precision. The report shows final-tier quality
on both the threshold-selection split and the held-out test set; these targets
are not guarantees. Simulation wait quantiles include completed jobs only, so
read them alongside the unhandled high-risk count and the remaining backlog.

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
## Round-1 results (Jigsaw scored test rows), run `20260922T112845423138Z-baseline`

Generated from this run's `report.json` and `manifest.json`. The run used commit `3429f1fe4170`, seed 20260922, scikit-learn 1.9.1. The input files' SHA-256 hashes are recorded in the manifest.

Development split: 95,743 / 31,914 / 31,914 rows for train / calibration / threshold selection; 63,978 scored test rows evaluated separately. Precision targets come from the run snapshot `reports/20260922T112845423138Z-baseline/policy.yaml`.

**Per label.** Thresholds were selected for precision targets 0.90 (human) and 0.99 (auto). Selection and test AP are both shown. An AP difference describes a measured performance gap; it does not by itself identify the cause.

| label | positives | AP selection | AP test | ROC-AUC test | human thr | P@human | R@human | auto thr | P@auto | R@auto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| toxic | 6090 | 0.859 | 0.752 | 0.957 | 0.6576 | 0.663 | 0.680 | 0.9893 | 0.898 | 0.349 |
| severe_toxic | 367 | 0.492 | 0.311 | 0.982 | n/a | n/a | n/a | n/a | n/a | n/a |
| obscene | 3691 | 0.874 | 0.773 | 0.972 | 0.6573 | 0.773 | 0.616 | 0.9984 | 0.952 | 0.266 |
| threat | 211 | 0.459 | 0.458 | 0.992 | n/a | n/a | n/a | n/a | n/a | n/a |
| insult | 3427 | 0.774 | 0.696 | 0.965 | 0.9176 | 0.879 | 0.290 | 0.9998 | 0.985 | 0.038 |
| identity_hate | 712 | 0.415 | 0.480 | 0.974 | n/a | n/a | n/a | n/a | n/a | n/a |

**Routing tiers.** Matching policy rules moved 145 auto-actions and 18 allows to human review. These counts aggregate all matching rules.

| tier | n | coverage | precision (test) | 95% CI | precision (selection split) | target |
|---|---:|---:|---:|---|---:|---:|
| auto_action | 2125 | 0.033 | 0.904 | 0.891 to 0.916 | 0.9917 | 0.990 |
| human_review | 4185 | 0.065 | 0.551 | 0.536 to 0.566 | 0.8901 | 0.900 |
| allow | 57668 | 0.901 | n/a | n/a | n/a | n/a |

Per-label precision targets need not hold after pooling labels, removing auto-actions from the human queue, or applying rules. Test precision is measured separately. The gates below retain the declared limits; changes to models, calibration or threshold selection need a fresh evaluation.

**Identity mentions.** The run recorded 63 whole-word terms in its `report.json`. This word-list diagnostic is a proxy, not an identity annotation.

False discovery rate (FDR) divides wrong positive decisions by all positive decisions in each slice. For auto-action, a decision is wrong when none of its triggering labels is true; for human review, it is wrong when no label is true. Ratios compare comments with an identity term to those without one.

| decision basis | tier | FDR with term (wrong / decisions) | FDR without term | ratio | ratio 95% CI |
|---|---|---:|---:|---:|---|
| population thresholds | auto_action | 0.109 (35/322) | 0.099 (202/2045) | 1.10 | 0.78 to 1.54 |
| after subgroup threshold, before rules | auto_action | 0.058 (13/225) | 0.099 (202/2045) | 0.58 | 0.34 to 1.01 |
| final routing | predicted_positive | 0.337 (273/810) | 0.329 (1811/5500) | 1.02 | 0.92 to 1.14 |
| final routing | auto_action | 0.059 (12/205) | 0.100 (192/1920) | 0.59 | 0.33 to 1.03 |
| final routing | human_review | 0.431 (261/605) | 0.452 (1619/3580) | 0.95 | 0.86 to 1.05 |

Conventional false-positive rate (FPR) uses all actually negative comments in the slice as its denominator, including allowed comments. Here 'negative' means no positive label. It measures the fraction of clean comments sent to each tier; it does not count wrong-trigger auto-actions on comments that have another positive label.

| final tier | FPR with term (clean routed / clean total) | FPR without term | ratio | ratio 95% CI |
|---|---:|---:|---:|---|
| predicted_positive | 0.066 (272/4144) | 0.034 (1805/53591) | 1.95 | 1.72 to 2.20 |
| auto_action | 0.003 (11/4144) | 0.003 (186/53591) | 0.76 | 0.42 to 1.40 |
| human_review | 0.063 (261/4144) | 0.030 (1619/53591) | 2.08 | 1.84 to 2.37 |

The subgroup threshold uses the run's auto-action precision target of 0.99 on identity-term comments in the threshold-selection split, at or above each population threshold. A label without a qualifying slice threshold cannot trigger auto-action on that slice. These are empirical selection targets, not statistical guarantees on test data.

Slice thresholds: toxic: 0.9974; severe_toxic: no qualifying slice threshold; obscene: no qualifying slice threshold; threat: no qualifying slice threshold; insult: no qualifying slice threshold; identity_hate: no qualifying slice threshold. They removed 31 of 921 population auto-actions on selection data and 97 of 2367 on test data. Of the removed test decisions, 77 had a positive label.

False omission rate uses allowed comments as its denominator. It is 0.058 with a term (240/4112 allowed comments have a positive label) and 0.033 without (1770/53556). The overall positive-label base rates are 0.158 and 0.093, respectively.

The five terms with the most positive decisions are shown below; comments can match multiple terms.

| term | wrong decisions | positive decisions | FDR |
|---|---:|---:|---:|
| gay | 89 | 360 | 0.247 |
| black | 35 | 79 | 0.443 |
| white | 27 | 70 | 0.386 |
| woman | 21 | 47 | 0.447 |
| women | 14 | 39 | 0.359 |

**Queue simulation.** 4 reviewers, 2 min per item, 8 h, capacity 120/h. The sampled pool has 4185 queued comments and 429 high-risk comments; high-risk means harm proxy >= 5.0. The run uses 5 seeds with identical arrivals and handle times for every ordering within each seed.

Arrival assumption: Poisson; jobs sampled with replacement from the human_review set. Handle times: deterministic. Harm proxy: max severity weight over TRUE labels (0 if clean); not real-world harm. Waits are minutes until review starts, measured over completed items. High-risk left includes jobs still in service; backlog counts jobs not yet started. The table reports seed means and standard deviations. Completion counts at a finite horizon do not establish queue stability.

| load/h | ordering | handled | high-risk handled | high-risk left | harm / reviewer-h | high-risk wait p50 | high-risk wait p90 | wait p90, all items | backlog at end |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | fifo | 468 ± 26 | 42.6 ± 5.9 | 0.0 ± 0.0 | 21.1 ± 1.3 | 0.0 ± 0.0 | 0.4 ± 0.2 | 0.3 ± 0.2 | 0 ± 1 |
| 60 | prob | 468 ± 26 | 42.6 ± 5.9 | 0.0 ± 0.0 | 21.1 ± 1.3 | 0.0 ± 0.0 | 0.3 ± 0.2 | 0.3 ± 0.1 | 0 ± 1 |
| 60 | severity | 468 ± 26 | 42.6 ± 5.9 | 0.0 ± 0.0 | 21.1 ± 1.3 | 0.0 ± 0.0 | 0.3 ± 0.2 | 0.3 ± 0.1 | 0 ± 1 |
| 108 | fifo | 867 ± 20 | 88.2 ± 4.6 | 0.8 ± 0.7 | 39.7 ± 1.3 | 1.5 ± 0.9 | 6.2 ± 2.8 | 6.0 ± 2.9 | 1 ± 2 |
| 108 | prob | 867 ± 20 | 88.2 ± 4.6 | 0.8 ± 0.7 | 39.7 ± 1.3 | 0.3 ± 0.1 | 1.4 ± 0.2 | 4.2 ± 1.2 | 1 ± 2 |
| 108 | severity | 867 ± 20 | 88.0 ± 4.7 | 1.0 ± 1.1 | 39.6 ± 1.3 | 0.3 ± 0.1 | 1.1 ± 0.3 | 4.4 ± 1.6 | 1 ± 2 |
| 180 | fifo | 953 ± 4 | 100.0 ± 3.8 | 47.6 ± 6.4 | 44.6 ± 0.9 | 81.8 ± 5.4 | 143.3 ± 9.3 | 139.9 ± 7.8 | 465 ± 41 |
| 180 | prob | 953 ± 4 | 124.6 ± 3.1 | 23.0 ± 5.9 | 53.1 ± 1.0 | 0.5 ± 0.2 | 3.1 ± 1.4 | 11.0 ± 2.5 | 465 ± 41 |
| 180 | severity | 953 ± 4 | 133.8 ± 5.3 | 13.8 ± 2.3 | 56.0 ± 1.3 | 0.5 ± 0.2 | 2.2 ± 1.1 | 11.5 ± 1.4 | 465 ± 41 |

Paired comparisons use shared arrivals within each seed. Differences are first ordering minus second. Each cell gives the mean difference and the number of seeds in which the first ordering was better on that metric.

| comparison | harm / reviewer-h difference | high-risk handled difference | high-risk wait p90 difference |
|---|---:|---:|---:|
| severity_vs_fifo@60 | 0.00; better 0/5 | 0.00; better 0/5 | -0.07; better 3/5 |
| prob_vs_fifo@60 | 0.00; better 0/5 | 0.00; better 0/5 | -0.06; better 4/5 |
| severity_vs_prob@60 | 0.00; better 0/5 | 0.00; better 0/5 | -0.00; better 1/5 |
| severity_vs_fifo@108 | -0.04; better 0/5 | -0.20; better 0/5 | -5.11; better 5/5 |
| prob_vs_fifo@108 | 0.01; better 1/5 | 0.00; better 0/5 | -4.80; better 5/5 |
| severity_vs_prob@108 | -0.04; better 0/5 | -0.20; better 0/5 | -0.30; better 5/5 |
| severity_vs_fifo@180 | 11.40; better 5/5 | 33.80; better 5/5 | -141.12; better 5/5 |
| prob_vs_fifo@180 | 8.46; better 5/5 | 24.60; better 5/5 | -140.18; better 5/5 |
| severity_vs_prob@180 | 2.94; better 5/5 | 9.20; better 5/5 | -0.94; better 3/5 |

Harm and high-risk status use the same policy weights as severity ordering. These comparisons are conditional on that weight vector and the stated arrival model.

Gates read from `reports/20260922T112845423138Z-baseline/policy.yaml`:

For identity FDR ratios, green requires the entire 95% interval to be at or below the ceiling. An interval entirely above it is red; a crossing interval is inconclusive. Subgroups below the minimum count are skipped, and missing estimates are unavailable. Neither state is a passing fairness result.

| gate | measured | requirement | status |
|---|---:|---|---|
| per-label AP (6 configured labels) | see label table | snapshot/override floors | green |
| auto_action precision | 0.904 | ≥ 0.990 | **red** |
| human_review precision | 0.551 | ≥ 0.520 | green |
| harm_per_reviewer_hour, severity / FIFO at 180.000/h | 1.255 | ≥ 1.120 | green |
| high_risk_wait_p90, severity / FIFO at 108.000/h | 0.174 | ≤ 0.550 | green |
| queue_depth_p95 at 108.000/h | 14.220 | ≤ 39 | green |
| reviewer_utilization at 108.000/h | 0.905 | ≥ 0.600 | green |
| hierarchy violation rate | 0.0000 | ≤ 0.005 | green |
| identity FDR ratio, pooled, final tier | 1.024 [0.92 to 1.14] | ≤ 1.250 | green |
| identity FDR ratio, auto_action, after subgroup threshold before rules | 0.585 [0.34 to 1.01] | ≤ 1.250 | green |
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
- [x] Identity-mention false-positive concentration measured per tier and per term on every run, and gated; rule R103 retired and replaced by a subgroup threshold (auto-action must clear its floor on the identity-term slice too)

Not done yet:

- [ ] Sentence-embedding signal and multi-model fusion, judged under the same protocol
- [ ] Hierarchy constraint (`severe_toxic` is an exact subset of `toxic`); today only the violation rate is reported
- [ ] Sensitivity sweep over the severity-weight vector
- [ ] Investigate the selection-to-test precision gap using development-data calibration, ranking diagnostics and threshold uncertainty; retain the 0.99 auto-action requirement

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
