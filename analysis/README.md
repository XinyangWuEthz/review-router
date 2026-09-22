# Round 2, steps 1 and 2: why auto-action precision is 0.904, and whether features fix it

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

`sample.csv` carries the text and two verdict columns. `human_verdict` is
empty and is the deliverable of this step; it has to be filled by a person.
`preread_verdict` is an LLM pre-read (Claude, recorded as such in
`sample_meta.json`) under one written rule: **toxic** = abuse directed at a
person or group; **borderline** = profanity, self-directed or undirected
rudeness; **clean** = no abuse. Its purpose is to give a first estimate and a
reading order, not to replace the human pass.

Pre-read result, 100 rows: toxic 40, borderline 33, clean 27.

What that implies for the floor if the human pass agreed with the pre-read:

| relabel rule | implied auto-action precision |
|---|---:|
| as measured | 0.904 |
| the 40% "toxic" rows are label noise | 0.942 |
| "toxic" and "borderline" both count as toxic | 0.974 |

Even the most generous reading leaves the tier below 0.99. The 27 "clean"
rows are real model errors, and they share one shape: a lexicon word used
without a target ("the animation is stupid", "awh that sucks", a TV series
called *2 Stupid Dogs*, "Big Dumb Object", a quoted film line, a regex
abuse-filter list). The word-level model fires on the token, not on whether
anyone is being attacked.

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
positives, including 16 of the 27 "clean" audited rows, but its threshold
admits 517 rows the baseline did not, and 87 of those are new false
positives. Character n-grams fix the OOV cases and then buy coverage with
the same non-directed lexical errors at a different set of rows.

## Conclusion for the next step

- The gap is roughly half label convention (pending the human verdicts) and
  half a real error mode, non-directed use of abuse vocabulary, that
  bag-of-n-grams features cannot express at any threshold.
- Keep `word+char_wb` as the round-2 default for ranking quality; do not
  expect it to satisfy the auto-action floor.
- Reaching 0.99 needs either a feature that sees the target of the abuse
  (a sentence encoder or a small transformer) or a narrower trigger
  (several labels jointly high plus a directed-address cue). Both must be
  judged under the same split and floors, with thresholds chosen on a
  selection split drawn from the test distribution and reported on rows
  that selection never touched.
- Until then the protocol's own rule applies: the tier stays red or is
  disabled, the floor is not lowered.
