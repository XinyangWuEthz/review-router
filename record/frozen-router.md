# Frozen scorer for router development

Decision, 2026-09-23: keep the existing training data, word features, classifier
and Platt parameters fixed. The project now studies the router on top of this
scorer. Threshold selection on development data, review rules and queue ordering
remain experimental choices.

The pinned scorer comes from [CI run 35857528118](https://github.com/XinyangWuEthz/review-router/actions/runs/35857528118),
training commit `210d9857220b23d175d4aff5a54ab221ad86280b`.
Its model SHA-256 is
`ac31ea513bf55ebc2500edf88e68f69c4e49f5e4b73d1c9a6f1a84fc5b378eef`.
The checked-in lock is `configs/frozen-baseline.json`; the model itself lives in
the ignored `models/frozen-baseline/` directory. Preserve a copy independently of
the expiring GitHub Actions artifact and cache.

Both `configs/dev.yaml` and `configs/baseline.yaml` load this scorer by default.
They verify its bytes, training corpus, row split assignments, model settings,
model implementation and inference dependency versions. Missing or incompatible
artifacts fail before evaluation, with no training fallback. Each new run saves
the exact model bytes and lock, and separates the original training provenance
from the current router code. The explicit `--retrain` option is reserved for
full-training checks and never replaces the frozen artifact.

## Verification against the saved baseline

Local validation run: `20260923T124601460734Z-human-review-baseline`.
This run was made from the working tree, so it is local validation rather than
a clean-commit CI acceptance run. The validation replaced `fit` and `calibrate`
with functions that fail immediately if called.

| Comparison with the saved CI run | Result |
| --- | --- |
| Frozen model bytes | Identical |
| Scored test rows | Same 63,978 IDs in the same order |
| Largest absolute probability difference | `1.1102230246251565e-15` |
| Changed final review bands | 0 |
| Combined review workload | Identical |
| Simulation summary | Identical |
| Simulated comment IDs, arrivals and service times | Identical |
| Per-job priority scores | Maximum difference `8.881784197001252e-15` |

The source ran on Linux with Python 3.12; this replay ran on macOS with Python
3.13 and the same NumPy, SciPy and scikit-learn versions. Some selected threshold
values differ at about `1e-16`, consistent with the observed floating-point
inference differences. The replay preserves the bands and queue results; it
does not claim byte-identical predictions across platforms.

The full unit suite, lint, type checks, real-data gates and all five negative
controls passed. The real-data gate check did not enforce a clean commit on this
working-tree run. The two unset band-precision gates and the inconclusive
identity FDR comparison remain diagnostic skips.

Freezing prevents vocabulary-cap tie selection from changing router inputs
because vocabulary selection is no longer executed. It does not repair the
underlying retraining instability. Use saved predictions when exact score reuse
is needed, and the shared incoming stream when comparing policies that change
admission. Independent final evaluation on new data remains deferred.
