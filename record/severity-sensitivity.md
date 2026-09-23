# Severity weight sensitivity

Five weight vectors were declared before this sweep. Only scheduling scores change; saved predictions, admissions, reference harm weights and high-risk labels stay fixed. The default is retained regardless of which arm wins.

Flat weights equal probability ordering and appear once as `prob`. Compression and expansion change weight contrast; `threat_lower` changes threat from 10 to 5. FIFO and band-first priority are fixed comparators.

Each entry is a mean over 20 paired arrival seeds for an 8-hour shift with four reviewers and 2-minute service. Arrival loads are admitted review jobs per hour. High risk means any true severe_toxic, threat or identity_hate label; other means all remaining queued jobs, including false positives.

| Load/h | Ordering | High-risk completed | High-risk wait p90, min | High-risk never started | Other completed | Other wait p90, min | Other never started | Reference harm / reviewer-h |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 108 | fifo | 122.00 | 5.16 | 0.45 | 739.65 | 5.29 | 2.70 | 50.23 |
| 108 | prob | 122.10 | 1.42 | 0.40 | 739.55 | 4.49 | 2.75 | 50.28 |
| 108 | priority | 122.20 | 1.15 | 0.30 | 739.45 | 4.78 | 2.85 | 50.30 |
| 108 | default | 122.25 | 1.11 | 0.30 | 739.40 | 4.78 | 2.85 | 50.30 |
| 108 | compressed | 122.15 | 1.12 | 0.35 | 739.50 | 4.58 | 2.80 | 50.28 |
| 108 | expanded | 122.25 | 1.11 | 0.25 | 739.40 | 4.88 | 2.90 | 50.30 |
| 108 | threat_lower | 122.20 | 1.11 | 0.35 | 739.45 | 4.80 | 2.80 | 50.28 |
| 180 | fifo | 135.60 | 139.54 | 70.05 | 819.65 | 139.42 | 393.60 | 55.98 |
| 180 | prob | 181.00 | 2.69 | 24.30 | 774.25 | 13.19 | 439.35 | 69.01 |
| 180 | priority | 189.20 | 2.28 | 16.20 | 766.05 | 13.83 | 447.45 | 71.49 |
| 180 | default | 189.35 | 1.76 | 16.05 | 765.90 | 13.40 | 447.60 | 71.60 |
| 180 | compressed | 187.20 | 1.72 | 18.20 | 768.05 | 14.27 | 445.45 | 70.98 |
| 180 | expanded | 191.80 | 1.83 | 13.60 | 763.45 | 13.42 | 450.05 | 72.01 |
| 180 | threat_lower | 189.25 | 1.79 | 16.15 | 766.00 | 13.17 | 447.50 | 71.57 |

Wait p90 is conditional on starting review. Never-started counts and unfinished-age p90 in the JSON accompany it to expose starvation. Total service capacity does not increase when the ordering changes; completions are redistributed between groups.

The JSON contains per-seed outcomes and paired differences against the default and FIFO, including ranges and better/tied/worse counts. These measure simulation variability, not uncertainty about the corpus or its labels. This already-inspected benchmark is not independent evidence of real review worthiness or production benefit.

## Paired high-risk completion differences at 180/h

Each difference is alternative minus default on the same arrival seed.

| Alternative | Mean difference | Seed range | Better / tied / worse |
|---|---:|---:|---:|
| prob | -8.35 | -18 to +2 | 1 / 0 / 19 |
| priority | -0.15 | -1 to +1 | 2 / 13 / 5 |
| compressed | -2.15 | -7 to +3 | 3 / 1 / 16 |
| expanded | +2.45 | +0 to +8 | 17 / 3 / 0 |
| threat_lower | -0.10 | -1 to +2 | 2 / 13 / 5 |

## Conclusion and boundary

At 180/h the default completes 53.75 more high-risk reviews and 53.75 fewer other reviews per shift than FIFO. The priority changes who receives the fixed capacity. The reference utility is a policy-weighted label proxy, not measured harm.

Retain the declared default. This sweep tests a small set of plausible perturbations; it does not establish an optimal or universally robust weight vector. In particular, an alternative performing better here is not a reason to retune against this inspected test benchmark. New traffic, different label priorities and variable handling times are outside this release's evidence.

Source evaluation: `20260923T125405590508Z-human-review-baseline`, commit `f19cc5b3f53b821a5e7cbb8b3cb0d0e15979e66d`.

Reproduce with `python scripts/weight_sensitivity.py reports/ci-35863370097`. The source hashes and complete protocol are recorded in `severity-sensitivity.json`.
