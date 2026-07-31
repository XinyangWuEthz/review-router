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

## Status

Built:

- [x] Arithmetic precision ceilings over the scored test set (`ceilings.py`)
- [x] Policy schema, loader, validation and routing (`policy.py`, `policy.yaml`)
- [x] Scored-row filter for the Jigsaw test set (`data.py`)
- [x] Regression-gate harness (`tests/test_gate.py`) — gates skip until floors
      are set from a measured run

Not built yet:

- [ ] Expert signals: TF-IDF, sentence-embedding similarity, rule heuristics
- [ ] Calibrated meta-classifier over the fused signals, with a hierarchy
      constraint (`severe_toxic` is an exact subset of `toxic`)
- [ ] Review-queue simulator: arrivals, reviewer pool, priority queue, clock
- [ ] Routing metrics: harm averted per reviewer-hour, time-to-action p50/p90/p99
      by severity, queue depth, reviewer utilisation
- [ ] False-positive concentration on identity-mentioning text
- [ ] `benchmarks/run.py` with frozen queries and a dated, version-stamped table

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
