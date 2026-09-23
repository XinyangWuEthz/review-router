# Run comparison (scored test rows)

| run | analyzer | commit | tier | n | coverage | test precision (95% CI) | selection precision |
|---|---|---|---|---:|---:|---|---:|
| human-review-baseline | word | 01897e45 | priority_review | 4625 | 0.0723 | 0.763 [0.750, 0.775] | 0.972 |
| human-review-baseline | word | 01897e45 | human_review | 1685 | 0.0263 | 0.419 [0.396, 0.443] | 0.797 |
| word-char | word+char_wb | 01897e45 | priority_review | 5030 | 0.0786 | 0.756 [0.744, 0.768] | 0.972 |
| word-char | word+char_wb | 01897e45 | human_review | 2063 | 0.0322 | 0.387 [0.366, 0.408] | 0.794 |

## Review workload

| run | needs a reviewer | share of comments | queue precision (95% CI) |
|---|---:|---:|---|
| human-review-baseline | 6310 | 0.0986 | 0.671 [0.659, 0.682] |
| word-char | 7093 | 0.1109 | 0.649 [0.638, 0.660] |

## Where the positives went

| run | top tier | positives flagged | positives in top tier | high-risk flagged | high-risk in top tier | high-risk left in allow |
|---|---|---:|---:|---:|---:|---:|
| human-review-baseline | priority_review | 4233/6243 (0.678) | 3527/6243 (0.565) | 916/1080 (0.848) | 838/1080 (0.776) | 164/1080 (0.152) |
| word-char | priority_review | 4603/6243 (0.737) | 3805/6243 (0.609) | 958/1080 (0.887) | 874/1080 (0.809) | 122/1080 (0.113) |

| label | AP human-review-baseline | AP word-char | AUC human-review-baseline | AUC word-char |
|---|---:|---:|---:|---:|
| toxic | 0.752 | 0.780 | 0.957 | 0.964 |
| severe_toxic | 0.311 | 0.319 | 0.982 | 0.985 |
| obscene | 0.773 | 0.796 | 0.972 | 0.978 |
| threat | 0.458 | 0.504 | 0.992 | 0.993 |
| insult | 0.696 | 0.731 | 0.965 | 0.971 |
| identity_hate | 0.480 | 0.562 | 0.974 | 0.984 |

## Ranking at matched volume (top k rows by max label probability)

| run | k | positives | high-risk | harm proxy |
|---|---:|---:|---:|---:|
| human-review-baseline | 6310 | 4234 | 916 | 11879 |
| human-review-baseline | 7093 | 4488 | 936 | 12390 |
| word-char | 6310 | 4338 | 941 | 12209 |
| word-char | 7093 | 4601 | 958 | 12707 |

## Queue at equal review load (report simulation, 5 seeds, mean ± std)

Arrival rates are review jobs per hour after admission, identical for every run.

| run | strategy@load | harm_per_reviewer_hour | high_risk_handled | high_risk_unhandled | high_risk_wait_p90 | completion_ratio | backlog_end |
|---|---|---:|---:|---:|---:|---:|---:|
| human-review-baseline | fifo@60 | 27.67 ± 3.48 | 66.60 ± 13.53 | 0.00 ± 0.00 | 0.26 ± 0.10 | 1.00 ± 0.00 | 0.40 ± 0.80 |
| human-review-baseline | priority@60 | 27.67 ± 3.48 | 66.60 ± 13.53 | 0.00 ± 0.00 | 0.19 ± 0.09 | 1.00 ± 0.00 | 0.40 ± 0.80 |
| human-review-baseline | fifo@108 | 50.41 ± 3.46 | 123.40 ± 17.67 | 0.40 ± 0.49 | 5.60 ± 1.99 | 0.99 ± 0.00 | 1.20 ± 1.60 |
| human-review-baseline | priority@108 | 50.47 ± 3.40 | 123.60 ± 17.44 | 0.20 ± 0.40 | 1.06 ± 0.08 | 0.99 ± 0.00 | 1.20 ± 1.60 |
| human-review-baseline | fifo@180 | 56.09 ± 1.28 | 137.60 ± 3.01 | 69.40 ± 8.64 | 138.94 ± 9.44 | 0.67 ± 0.02 | 464.80 ± 41.50 |
| human-review-baseline | priority@180 | 70.24 ± 1.71 | 184.20 ± 5.64 | 22.80 ± 5.34 | 1.63 ± 0.31 | 0.67 ± 0.02 | 464.80 ± 41.50 |
| word-char | fifo@60 | 26.05 ± 0.68 | 63.00 ± 2.00 | 0.40 ± 0.49 | 0.37 ± 0.14 | 1.00 ± 0.00 | 0.40 ± 0.80 |
| word-char | priority@60 | 26.05 ± 0.68 | 63.00 ± 2.00 | 0.40 ± 0.49 | 0.24 ± 0.12 | 1.00 ± 0.00 | 0.40 ± 0.80 |
| word-char | fifo@108 | 48.21 ± 2.34 | 118.60 ± 13.72 | 0.40 ± 0.49 | 5.88 ± 3.07 | 0.99 ± 0.00 | 1.20 ± 1.60 |
| word-char | priority@108 | 48.26 ± 2.36 | 118.80 ± 13.89 | 0.20 ± 0.40 | 1.29 ± 0.40 | 0.99 ± 0.00 | 1.20 ± 1.60 |
| word-char | fifo@180 | 53.92 ± 3.36 | 128.40 ± 17.11 | 63.40 ± 6.09 | 139.47 ± 6.66 | 0.67 ± 0.02 | 464.80 ± 41.50 |
| word-char | priority@180 | 67.62 ± 4.01 | 172.20 ± 22.46 | 19.60 ± 2.06 | 1.55 ± 0.59 | 0.67 ± 0.02 | 464.80 ± 41.50 |

## Queue at equal comment input (20 seeds, mean ± std)

Input rates are the reference run's equivalent incoming-comment loads; each run's
review load is input rate x its own queue fraction.

| run | input/h | review load/h | strategy | review_arrivals | completion_ratio | backlog_end | positives_completed_per_hour | clean_share_of_completed | high_risk_handled | high_risk_unhandled | harm_handled_per_hour |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| human-review-baseline | 608 | 60.0 | priority | 480.25 ± 22.35 | 1.00 ± 0.00 | 0.25 ± 0.64 | 39.83 ± 2.62 | 0.33 ± 0.03 | 69.40 ± 10.22 | 0.25 ± 0.55 | 112.33 ± 10.28 |
| human-review-baseline | 608 | 60.0 | fifo | 480.25 ± 22.35 | 1.00 ± 0.00 | 0.25 ± 0.64 | 39.83 ± 2.62 | 0.33 ± 0.03 | 69.40 ± 10.22 | 0.25 ± 0.55 | 112.33 ± 10.28 |
| human-review-baseline | 1095 | 108.0 | priority | 868.55 ± 33.06 | 0.99 ± 0.00 | 3.15 ± 3.27 | 71.80 ± 3.19 | 0.33 ± 0.01 | 126.10 ± 13.93 | 0.50 ± 0.83 | 202.49 ± 11.88 |
| human-review-baseline | 1095 | 108.0 | fifo | 868.55 ± 33.06 | 0.99 ± 0.00 | 3.15 ± 3.27 | 71.72 ± 3.19 | 0.33 ± 0.01 | 125.75 ± 13.87 | 0.85 ± 0.88 | 202.14 ± 11.89 |
| human-review-baseline | 1825 | 180.0 | priority | 1422.90 ± 30.99 | 0.67 ± 0.01 | 463.65 ± 30.04 | 93.92 ± 1.91 | 0.21 ± 0.02 | 187.00 ± 10.46 | 22.75 ± 4.29 | 283.02 ± 9.09 |
| human-review-baseline | 1825 | 180.0 | fifo | 1422.90 ± 30.99 | 0.67 ± 0.01 | 463.65 ± 30.04 | 80.61 ± 1.83 | 0.32 ± 0.02 | 138.60 ± 7.88 | 71.15 ± 9.69 | 225.68 ± 6.58 |
| word-char | 608 | 67.4 | priority | 532.15 ± 20.56 | 1.00 ± 0.00 | 0.05 ± 0.22 | 42.81 ± 1.99 | 0.35 ± 0.02 | 70.90 ± 7.20 | 0.20 ± 0.41 | 117.96 ± 6.10 |
| word-char | 608 | 67.4 | fifo | 532.15 ± 20.56 | 1.00 ± 0.00 | 0.05 ± 0.22 | 42.81 ± 1.99 | 0.35 ± 0.02 | 70.90 ± 7.20 | 0.20 ± 0.41 | 117.96 ± 6.10 |
| word-char | 1095 | 121.4 | priority | 973.85 ± 40.77 | 0.96 ± 0.02 | 34.05 ± 24.39 | 77.30 ± 2.41 | 0.34 ± 0.01 | 132.30 ± 13.08 | 1.50 ± 0.95 | 216.63 ± 13.43 |
| word-char | 1095 | 121.4 | fifo | 973.85 ± 40.77 | 0.96 ± 0.02 | 34.05 ± 24.39 | 75.92 ± 1.88 | 0.35 ± 0.01 | 128.45 ± 11.85 | 5.35 ± 4.31 | 211.47 ± 11.65 |
| word-char | 1825 | 202.3 | priority | 1626.00 ± 37.01 | 0.59 ± 0.01 | 666.05 ± 37.02 | 96.41 ± 1.66 | 0.19 ± 0.01 | 191.35 ± 16.79 | 27.10 ± 4.82 | 291.93 ± 12.59 |
| word-char | 1825 | 202.3 | fifo | 1626.00 ± 37.01 | 0.59 ± 0.01 | 666.05 ± 37.02 | 77.12 ± 1.88 | 0.35 ± 0.02 | 127.40 ± 14.32 | 91.05 ± 11.56 | 212.91 ± 11.05 |
