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

| relabel rule | implied auto-action precision (95% CI) |
|---|---:|
| as measured | 0.904 |
| audited "toxic" rows are label noise | 0.948 [0.939, 0.958] |
| "toxic" and "borderline" both count as toxic | 0.970 [0.961, 0.978] |

The interval maps the Wilson interval of the audited share through the
implied-precision formula and ignores the finite-population correction, so it
is conservative. Reaching 0.99 would need at most 21 false positives among
2125 auto-action rows, which means at least 90 of the 100 audited rows judged
toxic. The audit found 46, or 69 under the generous rule.

So the gap is not label noise alone. Relative to the pre-read, the human pass
raised the lower bound (0.942 to 0.948) because sarcasm and insinuation count
as attacks, and lowered the upper bound (0.974 to 0.970) because shouting at
someone without an insult, and asking about an identity, are not abuse. The
31 clean rows are real model errors. They come in two shapes: an abuse word
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
positives. Character n-grams fix the OOV cases and then buy coverage with
the same non-directed lexical errors at a different set of rows.

## Conclusion for the next step

- The human audit puts the label-convention share of the gap between 46%
  and 69% of the false positives. The rest, 31%, are real errors: abuse or
  crude vocabulary with nobody attacked, which bag-of-n-grams features cannot
  tell apart from an attack at any threshold. Even a full relabel under the
  generous rule leaves the tier at 0.970.
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
