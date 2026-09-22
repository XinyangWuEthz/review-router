# Run comparison (scored test rows)

| run | analyzer | auto prec (95% CI) | auto n | auto sel. prec | human prec | human n |
|---|---|---|---:|---:|---:|---:|
| baseline | word | 0.904 [0.891, 0.916] | 2125 | 0.992 | 0.551 | 4185 |
| char-ngram | char_wb | 0.898 [0.886, 0.909] | 2514 | 0.990 | 0.511 | 4420 |
| word-char | word+char_wb | 0.907 [0.895, 0.918] | 2460 | 0.992 | 0.511 | 4633 |

| label | AP baseline | AP char-ngram | AP word-char | AUC baseline | AUC char-ngram | AUC word-char |
|---|---:|---:|---:|---:|---:|---:|
| toxic | 0.752 | 0.774 | 0.780 | 0.957 | 0.963 | 0.964 |
| severe_toxic | 0.311 | 0.309 | 0.319 | 0.982 | 0.984 | 0.985 |
| obscene | 0.773 | 0.789 | 0.796 | 0.972 | 0.976 | 0.978 |
| threat | 0.458 | 0.454 | 0.504 | 0.992 | 0.991 | 0.993 |
| insult | 0.696 | 0.724 | 0.731 | 0.965 | 0.970 | 0.971 |
| identity_hate | 0.480 | 0.560 | 0.562 | 0.974 | 0.983 | 0.984 |

## Audited false positives still sent to auto_action (preread_verdict)

| run | still auto_action | by verdict |
|---|---:|---|
| baseline | 100/100 | borderline 33/33, clean 27/27, toxic 40/40 |
| char-ngram | 54/100 | borderline 20/33, clean 7/27, toxic 27/40 |
| word-char | 68/100 | borderline 26/33, clean 11/27, toxic 31/40 |
