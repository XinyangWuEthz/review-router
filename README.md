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
## Round-1 results

The results block will be generated from a baseline run made after this
implementation is committed, using `scripts/render_results.py`.
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
