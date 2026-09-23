# Round 2: why auto-action precision is 0.904, and whether character n-grams help

Question: the auto-action tier reaches 0.992 precision on the selection split
and 0.904 on the scored test rows against a 0.99 floor. Is the gap model
error, or a looser labelling convention on the test rows? And does a feature
change that is robust to unseen spellings (character n-grams) close it?

Everything here is computed from run directories under `reports/` by the two
scripts in `scripts/`; nothing is hand-edited except the audit verdicts.

## Step 1: audit of auto-action false positives (`auto_action_fp_audit/`)

`scripts/audit_auto_action_fp.py` draws a seeded, stratified sample of 100 of
the 204 false positives of run `20260922T131503586656Z-baseline` (auto_action
rows whose triggering label is 0 in `test_labels.csv`, the precision
definition the gate uses). Strata are proportional to the population:

| stratum (lexicon hit / another label true) | population | sample |
|---|---:|---:|
| lex / none | 133 | 65 |
| nolex / none | 64 | 31 |
| lex / other | 5 | 3 |
| nolex / other | 2 | 1 |

`sample.csv` carries the text and two verdict columns. `preread_verdict` is
an LLM pre-read (Claude, recorded as such in `sample_meta.json`).
`human_verdict` is the audit result: the repository owner read all 100 rows
against the pre-read and marked only the rows where they disagreed
(`human_changed`, with `human_note` where they gave a reason). The rule, as
refined before the human pass:

- **toxic**: attacks a person or group, explicitly or by sarcasm, insinuation
  or a rhetorical question aimed at the reader.
- **borderline**: profanity or rudeness with nobody attacked, including
  self-directed and quoted text.
- **clean**: nothing offensive.

The human pass changed 11 of 100 pre-read verdicts, in both directions:

| change | rows | examples |
|---|---:|---|
| borderline to toxic | 7 | "Your timing sucks", "Cool story bro'. What the fuck does it have to do with anything?", "..why are you acting dumb??" |
| borderline to clean | 3 | "Alexhead8835 SHUT UP >:(", "WHAT THE HELL Justin", "are you gay?" |
| toxic to clean | 1 | "u gay bro." |

Final verdicts: toxic 46, borderline 23, clean 31. Scored with
`python scripts/audit_auto_action_fp.py --score analysis/auto_action_fp_audit`:

| relabel rule | conditional precision estimate (approximate 95% interval) |
|---|---:|
| as measured | 0.904 |
| audited "toxic" rows are label noise | 0.948 [0.939, 0.958] |
| "toxic" and "borderline" both count as toxic | 0.970 [0.961, 0.978] |

These estimates assume the 1921 unaudited original true positives remain correct
under the audit rubric and extrapolate the unweighted audited share to all 204
original false positives. They are not measured precision after relabelling.
The interval maps a Wilson interval through this formula; it omits stratification,
the finite-population correction and annotation uncertainty. The 89% agreement
with the visible LLM pre-read does not measure independent annotator reliability.
Reaching the historical 0.99 target would need at most 21 false positives among
2125 auto-action rows, which means at least 90 of the 100 audited rows judged
toxic. The audit found 46, or 69 under the generous rule.

Under these assumptions, changing the label convention alone does not reach
the old target. Relative to the pre-read, the human pass raised the toxic-only
estimate (0.942 to 0.948) because sarcasm and insinuation count as attacks,
and lowered the estimate including borderline (0.974 to 0.970) because shouting at
someone without an insult, and asking about an identity, are not abuse. The
31 rows judged clean disagree with the model under this audit rubric. Examples
include an abuse word
used without a target ("the animation is stupid", "awh that sucks", the TV
series *2 Stupid Dogs*, a quoted film line, a regex abuse-filter list), and a
crude or identity word with no insult at all ("WHAT THE HELL Justin", "u gay
bro.", "sex == hello =="). Four of the 31 mention an identity term. Clean
verdicts are denser where the fixed lexicon does not hit (14 of 31 rows) than
where it does (16 of 68). Most of those 14 carry a word the lexicon omits,
such as "hell", "gay", "ass" or "poop", so a missing lexicon hit does not
mean the model saw nothing crude.

## Step 2: character n-grams (`run_comparison.md`, `paired_false_positives.json`)

`model.analyzer` now accepts `word` (round 1), `char_wb` and `word+char_wb`.
Two runs, identical to the baseline except for that key
(`configs/char_ngram.yaml`, `configs/word_char.yaml`):

| run | auto precision (95% CI) | auto n | selection precision |
|---|---:|---:|---:|
| baseline (word) | 0.904 [0.891, 0.916] | 2125 | 0.992 |
| char-ngram | 0.898 [0.886, 0.909] | 2514 | 0.990 |
| word+char | 0.907 [0.895, 0.918] | 2460 | 0.992 |

Ranking improves everywhere (average precision up on five of six labels,
identity_hate 0.480 to 0.562). Precision at the floor does not move: all
three intervals overlap and none approaches 0.99.

The paired view explains why. Word+char drops 62 of the baseline's 204 false
positives, including 17 of the 31 audited rows judged clean, but its threshold
admits 517 rows the baseline did not, and 87 of those are new false
positives. This comparison shows a change in which rows are admitted, but does
not isolate whether spelling coverage or interpretation of context caused it.

## Step 3: the same comparison under policy v2 (`v2/`)

Policy v2 removed the auto-action tier, so the feature question became: does
word+char route comments to human review better? Two runs from clean commit
01897e4 differ only in `model.analyzer` (`configs/baseline.yaml`,
`configs/word_char.yaml`). `scripts/compare_runs.py ... --equal-input` wrote
`v2/run_comparison.{md,json}`; `v2/gates_*.txt` holds the regression gates
against each report; `v2/verification.json` summarises an independent
four-agent check (raw output in `v2/verification_raw.json`).

| | word | word+char |
|---|---:|---:|
| comments needing review | 6310 | 7093 |
| queue precision | 0.671 | 0.649 |
| positives flagged | 4233 | 4603 |
| high-risk flagged (of 1080) | 916 | 958 |
| high-risk missed at 6310 flagged (matched volume) | 164 | 139 |
| full pipeline run, s | 29 | 170 |
| model.pkl, MB | 17.6 | 42.4 |

- **Ranking gain, small and real.** At matched volume word+char captures
  about 2.5% more positives and about 15% fewer high-risk misses, mostly
  identity_hate. Paired bootstrap intervals exclude zero.
- **Positives versus high risk.** About 70% of the extra positives come from
  flagging more rows; for high-risk rows more than half of the gain is
  ranking. At equal per-label counts word+char has higher precision on
  toxic, obscene and insult.
- **Same comment stream.** Common-random-number simulation (200 seeds,
  priority ordering): word+char handles 8 to 12 more harm units per hour
  and leaves 3.3 to 5.4 fewer high-risk comments unreviewed per 8-hour
  shift, because fewer stay in allow. It needs about 12% more review work;
  at the middle input rate its queue passes capacity and the end backlog
  grows by about 28. Under FIFO at overload the result reverses.
- **Gates.** Word passes all 47 gates that ran. Word+char fails two whose
  limits were set from the word run: toxic operating-point precision by
  0.001, inside sampling error, and severe_toxic predicted-volume overshoot
  (1.076 against 1.0), a robust calibration regression on a label that sets
  no routing threshold.

## Conclusion

- In this sample, 46 rows were judged toxic, 23 borderline and 31 clean.
  The conditional estimates are 0.948 and 0.970, both below the old 0.99
  target. Some examples suggest difficulty interpreting targets and context;
  they do not prove that every bag-of-n-grams model must make these errors.
  The audit does not establish which comments are worth human review.
- Decision (2026-09-23, project owner): do not adopt character n-grams for
  now; `model.analyzer` stays in the code with default `word`. The reason is
  cost (about 6x pipeline time, 2.4x model size), about 12% more review
  work, and unresolved regression failures. Further threshold selection would
  need development data followed by independent evaluation. Current gates stay
  unchanged; the measured gain on high-risk comments remains recorded.
- If independent-data validation is resumed, reconsider word+char at equal
  review volume with thresholds and limits selected without using the final
  evaluation set.
- An independent human evaluation of review worthiness is an optional future
  direction, deferred on 2026-09-23 because of annotation workload. Ambiguous
  comments may deserve review even when no violation is confirmed, and some
  require context that is unavailable. Keep the original Jigsaw labels and
  existing gates; use this audit as development evidence, not final evaluation.
