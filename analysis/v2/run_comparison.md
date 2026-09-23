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

## Queue at equal review load (report simulation, 5 seeds, mean ± population SD)

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

## Queue at equal comment input (common random numbers, 200 seeds)

One incoming comment stream per seed, shared by every run. Each run reviews what it
flags; high-risk comments it leaves in allow count as not reviewed. Mean ± sample SD.

| run | input/h | strategy | review_arrivals | completion_ratio | backlog_end | positives_completed_per_hour | harm_handled_per_hour | high_risk_in_stream | high_risk_left_in_allow | high_risk_unhandled_in_queue | high_risk_not_reviewed |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| human-review-baseline | 608 | priority | 481.13 ± 20.72 | 1.00 ± 0.00 | 0.07 ± 0.33 | 40.21 ± 2.08 | 113.00 ± 7.28 | 82.69 ± 9.06 | 12.71 ± 3.57 | 0.24 ± 0.49 | 12.96 ± 3.60 |
| human-review-baseline | 608 | fifo | 481.13 ± 20.72 | 1.00 ± 0.00 | 0.07 ± 0.33 | 40.21 ± 2.08 | 112.99 ± 7.28 | 82.69 ± 9.06 | 12.71 ± 3.57 | 0.25 ± 0.50 | 12.96 ± 3.60 |
| human-review-baseline | 1095 | priority | 864.95 ± 31.23 | 0.99 ± 0.01 | 4.00 ± 4.94 | 72.00 ± 3.03 | 202.86 ± 10.79 | 149.19 ± 12.21 | 22.50 ± 4.74 | 0.73 ± 0.83 | 23.23 ± 4.76 |
| human-review-baseline | 1095 | fifo | 864.95 ± 31.23 | 0.99 ± 0.01 | 4.00 ± 4.94 | 71.89 ± 2.99 | 202.47 ± 10.64 | 149.19 ± 12.21 | 22.50 ± 4.74 | 1.04 ± 1.17 | 23.54 ± 4.76 |
| human-review-baseline | 1825 | priority | 1440.21 ± 34.41 | 0.66 ± 0.02 | 480.64 ± 34.39 | 94.33 ± 1.79 | 283.79 ± 9.87 | 246.01 ± 14.91 | 36.73 ± 5.91 | 21.65 ± 4.73 | 58.38 ± 7.27 |
| human-review-baseline | 1825 | fifo | 1440.21 ± 34.41 | 0.66 ± 0.02 | 480.64 ± 34.39 | 80.18 ± 1.83 | 225.59 ± 8.42 | 246.01 ± 14.91 | 36.73 ± 5.91 | 69.52 ± 8.97 | 106.25 ± 10.92 |
| word-char | 608 | priority | 540.34 ± 22.19 | 1.00 ± 0.00 | 0.20 ± 0.74 | 43.74 ± 2.17 | 120.92 ± 7.45 | 82.69 ± 9.06 | 9.38 ± 3.12 | 0.24 ± 0.47 | 9.62 ± 3.14 |
| word-char | 608 | fifo | 540.34 ± 22.19 | 1.00 ± 0.00 | 0.20 ± 0.74 | 43.74 ± 2.17 | 120.91 ± 7.46 | 82.69 ± 9.06 | 9.38 ± 3.12 | 0.25 ± 0.49 | 9.63 ± 3.12 |
| word-char | 1095 | priority | 973.36 ± 32.04 | 0.96 ± 0.02 | 32.10 ± 22.44 | 77.25 ± 2.66 | 215.05 ± 10.35 | 149.19 ± 12.21 | 16.66 ± 3.92 | 1.23 ± 1.09 | 17.89 ± 4.05 |
| word-char | 1095 | fifo | 973.36 ± 32.04 | 0.96 ± 0.02 | 32.10 ± 22.44 | 75.97 ± 2.20 | 210.32 ± 8.75 | 149.19 ± 12.21 | 16.66 ± 3.92 | 4.88 ± 3.65 | 21.54 ± 5.27 |
| word-char | 1825 | priority | 1617.84 ± 35.59 | 0.59 ± 0.01 | 658.04 ± 35.59 | 96.55 ± 1.59 | 292.70 ± 9.33 | 246.01 ± 14.91 | 27.56 ± 4.93 | 25.46 ± 5.04 | 53.02 ± 6.83 |
| word-char | 1825 | fifo | 1617.84 ± 35.59 | 0.59 ± 0.01 | 658.04 ± 35.59 | 77.57 ± 1.81 | 214.79 ± 8.11 | 246.01 ± 14.91 | 27.56 ± 4.93 | 88.38 ± 9.19 | 115.94 ± 10.77 |

Paired difference, word-char minus human-review-baseline (mean ± SE):

| input/h | strategy | review_arrivals | completion_ratio | backlog_end | positives_completed_per_hour | harm_handled_per_hour | high_risk_in_stream | high_risk_left_in_allow | high_risk_unhandled_in_queue | high_risk_not_reviewed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 608 | priority | +59.20 ± 0.78 | -0.00 ± 0.00 | +0.13 ± 0.04 | +3.53 ± 0.06 | +7.92 ± 0.16 | +0.00 ± 0.00 | -3.33 ± 0.16 | -0.01 ± 0.01 | -3.33 ± 0.16 |
| 608 | fifo | +59.20 ± 0.78 | -0.00 ± 0.00 | +0.13 ± 0.04 | +3.53 ± 0.06 | +7.91 ± 0.16 | +0.00 ± 0.00 | -3.33 ± 0.16 | +0.00 ± 0.01 | -3.33 ± 0.16 |
| 1095 | priority | +108.41 ± 1.17 | -0.03 ± 0.00 | +28.11 ± 1.46 | +5.25 ± 0.08 | +12.20 ± 0.22 | +0.00 ± 0.00 | -5.83 ± 0.22 | +0.49 ± 0.06 | -5.34 ± 0.22 |
| 1095 | fifo | +108.41 ± 1.17 | -0.03 ± 0.00 | +28.11 ± 1.46 | +4.07 ± 0.12 | +7.85 ± 0.35 | +0.00 ± 0.00 | -5.83 ± 0.22 | +3.84 ± 0.23 | -2.00 ± 0.30 |
| 1825 | priority | +177.62 ± 1.43 | -0.07 ± 0.00 | +177.40 ± 1.43 | +2.22 ± 0.07 | +8.91 ± 0.25 | +0.00 ± 0.00 | -9.18 ± 0.28 | +3.81 ± 0.32 | -5.36 ± 0.27 |
| 1825 | fifo | +177.62 ± 1.43 | -0.07 ± 0.00 | +177.40 ± 1.43 | -2.61 ± 0.08 | -10.80 ± 0.32 | +0.00 ± 0.00 | -9.18 ± 0.28 | +18.86 ± 0.33 | +9.69 ± 0.35 |
