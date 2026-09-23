# Run comparison (scored test rows)

| run | analyzer | commit | tier | n | coverage | test precision (95% CI) | selection precision |
|---|---|---|---|---:|---:|---|---:|
| baseline | word | a7ee1ab2 dirty | auto_action | 2125 | 0.0332 | 0.904 [0.891, 0.916] | 0.992 |
| baseline | word | a7ee1ab2 dirty | human_review | 4185 | 0.0654 | 0.551 [0.536, 0.566] | 0.890 |
| char-ngram | char_wb | df31a3ae dirty | auto_action | 2514 | 0.0393 | 0.898 [0.886, 0.909] | 0.990 |
| char-ngram | char_wb | df31a3ae dirty | human_review | 4420 | 0.0691 | 0.511 [0.497, 0.526] | 0.877 |
| word-char | word+char_wb | df31a3ae dirty | auto_action | 2460 | 0.0385 | 0.907 [0.895, 0.918] | 0.992 |
| word-char | word+char_wb | df31a3ae dirty | human_review | 4633 | 0.0724 | 0.511 [0.496, 0.525] | 0.881 |

## Where the positives went

| run | top tier | positives flagged | positives in top tier | high-risk flagged | high-risk in top tier | high-risk left in allow |
|---|---|---:|---:|---:|---:|---:|
| baseline | auto_action | 4233/6243 (0.678) | 1928/6243 (0.309) | 916/1080 (0.848) | 487/1080 (0.451) | 164/1080 (0.152) |
| char-ngram | auto_action | 4525/6243 (0.725) | 2265/6243 (0.363) | 954/1080 (0.883) | 579/1080 (0.536) | 126/1080 (0.117) |
| word-char | auto_action | 4603/6243 (0.737) | 2237/6243 (0.358) | 958/1080 (0.887) | 528/1080 (0.489) | 122/1080 (0.113) |

| label | AP baseline | AP char-ngram | AP word-char | AUC baseline | AUC char-ngram | AUC word-char |
|---|---:|---:|---:|---:|---:|---:|
| toxic | 0.752 | 0.774 | 0.780 | 0.957 | 0.963 | 0.964 |
| severe_toxic | 0.311 | 0.309 | 0.319 | 0.982 | 0.984 | 0.985 |
| obscene | 0.773 | 0.789 | 0.796 | 0.972 | 0.976 | 0.978 |
| threat | 0.458 | 0.454 | 0.504 | 0.992 | 0.991 | 0.993 |
| insult | 0.696 | 0.724 | 0.731 | 0.965 | 0.970 | 0.971 |
| identity_hate | 0.480 | 0.560 | 0.562 | 0.974 | 0.983 | 0.984 |

## Ranking at matched volume (top k rows by max label probability)

| run | k | positives | high-risk | harm proxy |
|---|---:|---:|---:|---:|
| baseline | 6310 | 4234 | 916 | 11879 |
| baseline | 6934 | 4437 | 929 | 12278 |
| baseline | 7093 | 4488 | 936 | 12390 |
| char-ngram | 6310 | 4320 | 931 | 12135 |
| char-ngram | 6934 | 4526 | 954 | 12559 |
| char-ngram | 7093 | 4584 | 958 | 12671 |
| word-char | 6310 | 4338 | 941 | 12209 |
| word-char | 6934 | 4553 | 956 | 12627 |
| word-char | 7093 | 4601 | 958 | 12707 |

## Queue at equal review load (report simulation, 5 seeds, mean ± std)

Arrival rates are review jobs per hour after admission, identical for every run.

| run | strategy@load | harm_per_reviewer_hour | high_risk_handled | high_risk_unhandled | high_risk_wait_p90 | completion_ratio | backlog_end |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | fifo@60 | 21.14 ± 1.31 | 42.60 ± 5.85 | 0.00 ± 0.00 | 0.39 ± 0.24 | 1.00 ± 0.00 | 0.40 ± 0.80 |
| baseline | severity@60 | 21.14 ± 1.31 | 42.60 ± 5.85 | 0.00 ± 0.00 | 0.32 ± 0.21 | 1.00 ± 0.00 | 0.40 ± 0.80 |
| baseline | fifo@108 | 39.66 ± 1.30 | 88.20 ± 4.58 | 0.80 ± 0.75 | 6.18 ± 2.76 | 0.99 ± 0.00 | 1.20 ± 1.60 |
| baseline | severity@108 | 39.62 ± 1.31 | 88.00 ± 4.69 | 1.00 ± 1.10 | 1.08 ± 0.29 | 0.99 ± 0.00 | 1.20 ± 1.60 |
| baseline | fifo@180 | 44.64 ± 0.91 | 100.00 ± 3.79 | 47.60 ± 6.37 | 143.68 ± 8.96 | 0.67 ± 0.02 | 464.80 ± 41.50 |
| baseline | severity@180 | 56.04 ± 1.26 | 133.80 ± 5.34 | 13.80 ± 2.32 | 2.15 ± 1.06 | 0.67 ± 0.02 | 464.80 ± 41.50 |
| char-ngram | fifo@60 | 19.02 ± 1.88 | 40.20 ± 6.43 | 0.40 ± 0.80 | 0.53 ± 0.44 | 1.00 ± 0.00 | 0.40 ± 0.80 |
| char-ngram | severity@60 | 19.02 ± 1.88 | 40.20 ± 6.43 | 0.40 ± 0.80 | 0.35 ± 0.27 | 1.00 ± 0.00 | 0.40 ± 0.80 |
| char-ngram | fifo@108 | 37.77 ± 1.96 | 82.20 ± 5.91 | 0.00 ± 0.00 | 5.67 ± 2.74 | 0.99 ± 0.00 | 1.20 ± 1.60 |
| char-ngram | severity@108 | 37.73 ± 2.00 | 82.00 ± 6.16 | 0.20 ± 0.40 | 1.37 ± 0.19 | 0.99 ± 0.00 | 1.20 ± 1.60 |
| char-ngram | fifo@180 | 40.32 ± 1.66 | 81.40 ± 7.23 | 44.00 ± 8.56 | 140.51 ± 9.76 | 0.67 ± 0.02 | 464.80 ± 41.50 |
| char-ngram | severity@180 | 50.01 ± 1.30 | 110.80 ± 7.39 | 14.60 ± 7.12 | 1.87 ± 0.85 | 0.67 ± 0.02 | 464.80 ± 41.50 |
| word-char | fifo@60 | 18.99 ± 1.24 | 38.40 ± 5.68 | 0.00 ± 0.00 | 0.57 ± 0.29 | 1.00 ± 0.00 | 0.40 ± 0.80 |
| word-char | severity@60 | 18.99 ± 1.24 | 38.40 ± 5.68 | 0.00 ± 0.00 | 0.36 ± 0.21 | 1.00 ± 0.00 | 0.40 ± 0.80 |
| word-char | fifo@108 | 38.54 ± 1.38 | 84.00 ± 5.69 | 0.40 ± 0.49 | 5.88 ± 2.97 | 0.99 ± 0.00 | 1.20 ± 1.60 |
| word-char | severity@108 | 38.57 ± 1.40 | 84.20 ± 5.88 | 0.20 ± 0.40 | 1.26 ± 0.41 | 0.99 ± 0.00 | 1.20 ± 1.60 |
| word-char | fifo@180 | 40.06 ± 1.30 | 87.00 ± 8.74 | 39.80 ± 8.47 | 140.42 ± 7.85 | 0.67 ± 0.02 | 464.80 ± 41.50 |
| word-char | severity@180 | 50.00 ± 2.39 | 117.00 ± 12.77 | 9.80 ± 4.53 | 1.50 ± 0.44 | 0.67 ± 0.02 | 464.80 ± 41.50 |

## Audited false positives still sent to auto_action (human_verdict)

| run | still auto_action | by verdict |
|---|---:|---|
| baseline | 100/100 | borderline 23/23, clean 31/31, toxic 46/46 |
| char-ngram | 54/100 | borderline 14/23, clean 8/31, toxic 32/46 |
| word-char | 68/100 | borderline 18/23, clean 14/31, toxic 36/46 |
