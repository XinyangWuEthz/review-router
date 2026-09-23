# v0.1.0 method freeze

This release fixes the project scope: a human-confirmed review router over a
fixed classifier, evaluated under limited reviewer capacity. The classifier
supplies calibrated label probabilities. The project measures admission,
review-band composition and queue ordering. It does not automate moderation or
claim an independently validated model of review worthiness.

## Default method

- Frozen word TF-IDF, six logistic classifiers and Platt calibration from the
  accepted scorer at `210d9857220b`.
- Policy v3: cumulative selection for ordinary review, segment agreement for
  priority review, existing subgroup constraints and rules R101/R102.
- Combined review queue ordered by maximum calibrated probability times severity
  weight. Default weights remain threat 10, identity hate 6, severe toxicity 5,
  insult 2, obscenity 2 and toxicity 1.
- Fixed Jigsaw corpus and train/calibration/selection split. Numeric thresholds
  are selected on the declared development split; exact replay uses the saved
  probabilities and admissions in the evaluation archive.
- Four reviewers, two minutes per review and eight-hour simulations. FIFO,
  probability ordering and band-first ordering are comparison arms.

The release tag fixes the code and configuration. Scorer bytes and data hashes
are pinned separately. Future scorer, policy or default-weight changes belong
to a new experiment and release, not a replacement of these assets.

## Evidence and limits

The [five-seed baseline](../record/step6-priority.html) shows more high-risk
completions than FIFO under overload. The [bounded sensitivity experiment](../record/severity-sensitivity.md)
keeps saved scores, admissions, arrival streams, evaluation weights and high-risk
labels fixed while testing five declared ordering-weight vectors over 20 seeds
at 108 and 180 admitted reviews/hour.

At 180/h, the default completes 189.35 high-risk reviews per shift versus
135.60 for FIFO. The three non-flat perturbations produce 187.20–191.80; flat
weights, equivalent to probability ordering, produce 181.00. The default is
retained without selecting the best result from this sweep.

The cost is a redistribution of fixed capacity. Against FIFO, the default
completes 53.75 more high-risk reviews and 53.75 fewer other reviews per shift.
The mean total is unchanged at 955.25. Other queued jobs never started rise
from 393.60 to 447.60. Wait quantiles alone do not describe this cost.

These results use an already-inspected benchmark and Jigsaw label proxies.
Simulation seeds do not quantify label or corpus uncertainty. Independent human
evaluation, other domains and variable review durations remain outside this
release. The weight sweep is a bounded stress test, not proof that the default
weights are optimal or that review value has been measured in production.

## Durable reproduction assets

The GitHub release contains:

| Asset | Contents |
|---|---|
| `frozen-baseline-v0.1.0.zip` | Pinned model, original training manifest and split assignments |
| `baseline-evaluation-v0.1.0.zip` | Accepted frozen evaluation at `f19cc5b`, including exact predictions, thresholds, policy/config snapshots, reports and per-job records |
| `SHA256SUMS.txt` | Archive checksums, also recorded in `configs/release-assets.json` |

Original comment texts are not included. Restore commands validate model-file hashes
before loading. See [the reproduction protocol](experiment.md) for setup and
the exact sensitivity command. The published archives remove the dependency on
expiring Actions artifacts for this release.
