# review-router

A capacity-constrained content-policy router: fuse signals from several expert
models, route each item to **allow / human-review / auto-action**, and evaluate
the routing under a review queue that has a fixed number of reviewers.

The modelling problem on the Jigsaw toxic-comment corpus is solved — the 2018
winning ensemble scored 0.9886 mean column-wise ROC-AUC and an off-the-shelf
single BERT scores 0.9864. The open problem is that a model at that AUC is still
roughly 40% precise at a usable threshold, which is what drowns a human review
queue. This project is about the layer above the classifier.

## The result the project is built around

Precision is capped by class balance before any model exists. On the 63,978
**scored** Jigsaw test rows, at 50% recall:

| Label | positives / negatives | ceiling @ FPR 1% | @ FPR 0.1% |
|---|---|---|---|
| toxic | 6,090 / 57,888 | 84.0% | 98.1% |
| obscene | 3,691 / 60,287 | 75.4% | 96.8% |
| insult | 3,427 / 60,551 | 73.9% | 96.6% |
| identity_hate | 712 / 63,266 | 36.0% | 84.9% |
| severe_toxic | 367 / 63,611 | 22.4% | 74.3% |
| **threat** | **211 / 63,767** | **14.2%** | 62.3% |

Reproduce: `python -c "from review_router.ceilings import ceiling_table; print(ceiling_table())"`

For `threat`, a 1% false-positive rate produces ~638 false positives against
~106 true positives. The rare, highest-harm labels only become actionable at
FPR ≈ 0.1% — the extreme left edge of the ROC curve, the region that contributes
almost nothing to the ROC-AUC integral. That is why the official metric is
uninformative at the operating point a review queue actually runs at, and it is
why `threat` is routed to a human by policy rather than auto-actioned.

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
- **Rules and models coexist.** Rules are the precision-critical fast path and
  the place where policy reasoning is legible; the model supplies coverage.
- **Capacity is the point.** An unbounded queue makes routing trivial. Arrival
  rate, reviewer count and handle time are explicit inputs, and the simulation
  reports the overflow point at which the backlog stops clearing.

## Reproducing the round-1 experiment

Round 1 asks one question: **with the same reviewer capacity, does risk-ordered
review reach high-harm comments earlier than first-in-first-out?** The protocol
is written down in
[`docs/superpowers/specs/2026-09-22-round1-reproducible-experiment-design.md`](docs/superpowers/specs/2026-09-22-round1-reproducible-experiment-design.md).

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
strategy sees the identical arrivals. The scored test set is touched once, for
the final evaluation.

The 90% and 99% targets select thresholds independently for each label.
Combining labels, removing automatic actions from the human queue and applying
rules can lower the final tiers' precision. The report shows final-tier quality
on both the threshold-selection split and the held-out test set; these targets
are not guarantees. Simulation wait quantiles include completed jobs only, so
read them alongside the unhandled high-risk count and the remaining backlog.

Regression gates read the report:

```bash
REVIEW_ROUTER_EVAL_REPORT=reports/<run_id>/report.json pytest -q tests/test_gate.py
```

Gates whose floor is still `null` in `policy.yaml` skip; an explicitly requested
report that is missing or lacks a required section fails.

To check the pipeline without the corpus, generate a synthetic stand-in with
the same file layout. Its numbers are pipeline checks, never results:

```bash
python scripts/make_synthetic_corpus.py --out data/synthetic
python scripts/run_pipeline.py --config configs/smoke.yaml
```

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
- [x] Regression-gate harness (`tests/test_gate.py`) — floors are set from a measured run

Not done yet:

- [ ] Run `configs/baseline.yaml` on the real Jigsaw files and set the gate floors from it
- [ ] Sentence-embedding signal and multi-model fusion, judged under the same protocol
- [ ] Hierarchy constraint (`severe_toxic` is an exact subset of `toxic`); today only the violation rate is reported
- [ ] Sensitivity sweep over the severity-weight vector
- [ ] False-positive concentration on identity-mentioning text

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
  weight vector, both of which are policy inputs and are swept for sensitivity —
  never presented as measurements.
- **No identity annotations exist in the 2018 corpus.** Subgroup / BPSN / BNSP
  AUCs cannot be computed on it. Fairness work here must either pull the 2019
  Civil Comments release or use a templated probe set, labelled as synthetic.
- **No temporal split.** Adversarial drift is the defining property of this
  domain and the 2018 files carry no reliable timestamp. Robustness is probed
  with obfuscation perturbations instead, which is a weaker substitute.

## Development notes

Built with AI-assisted development tooling; problem framing, design decisions,
code review and validation are my own.

## Licence

MIT
