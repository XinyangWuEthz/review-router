# Langfuse cookbook: prioritize traces for human annotation under a fixed budget

A cookbook prepared for contribution to the
[Langfuse cookbook](https://github.com/langfuse/langfuse-docs/tree/main/cookbook).
It applies the review-router question, which risks a fixed review capacity
reaches, to Langfuse annotation queues: given the automated scores already on a
project's traces and a budget of B human reviews per run, which traces go into
the queue?

| File | Role |
|---|---|
| `example_annotation_queue_prioritization.ipynb` | The cookbook. Runs offline on a synthetic pool without credentials; reads and writes a Langfuse project when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set. |
| `seed_demo_project.py` | Writes synthetic traces and judge scores into an empty project so the live path has candidates. Dry-runs without credentials. |
| `budget_selection.py` | The selection logic: five policies, the comparison tables, the synthetic pool and the closing-the-loop estimates. Inlined verbatim into the notebook cell tagged `budget_selection`; `tests/test_langfuse_example.py` fails if the two copies drift. |

`max_score` mirrors the benchmark's probability ordering and `severity` its
default ordering, the maximum over signals of probability times a declared
weight. `random` is the notebook's sampling baseline; `uncertainty` (judge
calibration) and `mixed` (quota split) are additions for annotation queues and
have no benchmark counterpart. Selection under a budget reallocates reviewer
time; it does not add any, and the notebook reports the displaced work next to
every gain.

## Run

```bash
pip install -e ".[ml,dev]" langfuse nbconvert ipykernel
python -m pytest -q tests/test_langfuse_example.py
# Execute offline (synthetic pool); outputs are rewritten in place.
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.record_timing=False \
  examples/langfuse/example_annotation_queue_prioritization.ipynb
```

Clear the `%pip` cell's output before committing; the tests reject per-cell
timing metadata, error outputs and HTML table outputs (the notebook renders
tables as text so the generated MDX compiles).

Set the two keys and `LANGFUSE_BASE_URL` to run against a project. The live
path uses the public API as exposed by `langfuse` 4.17.0: the v3 scores read
(`api.scores_v3.get_many_v3`, cursor-paged, with the `subject`, `annotation`
and `details` field groups), `api.score_configs`, `api.annotation_queues` and
`api.scores.create` (synchronous, so a failed decision write raises). The
page-based `api.scores.get_many` is the deprecated v2 read path and is not
used: the SDK marks it for removal on Langfuse Cloud on 2026-11-16 and on
self-hosted deployments at the Langfuse v4 upgrade. The live path was run
twice against a Langfuse Cloud project on 2026-10-07; the log below records
what each run confirmed.

## Run against a free Langfuse project

1. Sign up at cloud.langfuse.com (the free Hobby plan is enough), create a
   project, and under Project Settings > API Keys create a key pair. Note the
   region: `https://cloud.langfuse.com` for EU, `https://us.cloud.langfuse.com`
   for US.
2. Seed the empty project with synthetic traces and judge scores named like
   the notebook's signals; a fresh project has no scores, so the live path
   would otherwise find no candidates:

   ```bash
   export LANGFUSE_PUBLIC_KEY=pk-lf-...
   export LANGFUSE_SECRET_KEY=sk-lf-...
   export LANGFUSE_BASE_URL=https://cloud.langfuse.com
   python examples/langfuse/seed_demo_project.py --n 300
   ```

   Wait a minute for ingestion, then confirm in the UI that the traces carry
   `safety_violation`, `hallucination`, `negative_feedback`, `off_topic` and
   `tone` scores.
3. Execute the notebook against the project into a copy, so the committed
   offline outputs stay untouched:

   ```bash
   jupyter nbconvert --to notebook --execute \
     --output live-run.ipynb --output-dir /tmp/langfuse-live \
     examples/langfuse/example_annotation_queue_prioritization.ipynb
   ```

   Or open it in Jupyter with the three variables exported and run all cells.
   Step 3 should report a few hundred candidates, Step 5 a queue named
   `daily-review` with 40 items, and Step 6 "no completed reviews in queue ... yet".
4. In the UI, open Annotation Queues > daily-review, note the order the items
   are served in, and score ten or so items with the `human_*` configs,
   completing each one (Step 6 reads completed items only).
5. Run the notebook again. Step 3 excludes the traces selected on the first
   run, Step 5 adds a second batch, and Step 6 now shows the precision and
   agreement tables rebuilt from the stored `review_policy` scores.
6. Tick the checklist below and record each outcome for the pull request
   text. Do not commit the live-executed copy; re-execute offline (recipe
   above) before committing any notebook change.

## Before opening the langfuse-docs pull request

Read from the Langfuse server source and API definition while writing, so the
notebook text already relies on them; confirm once in the UI of a live project:

- [x] Insertion order is the priority: the queue page lists the rank-1 trace
      first while the items endpoint returns it last (newest first), and the
      first ten items served for annotation were ranks 1 to 10 (all from the
      severity quota). Confirmed on 2026-10-07.
- [x] The API does not reject a trace already in the queue; the notebook's
      `queued_trace_ids` check is what prevents duplicates. Confirmed on
      2026-10-07: adding an already-queued trace created a second item, which
      `delete_queue_item` removed.
- [x] `create_score(..., data_type="CATEGORICAL")` without a config id is
      accepted for the `review_policy` marker, and its `metadata` comes back
      through the `details` field group: the second run excluded "40 selected
      earlier" and attributed the annotated traces to four policies.

Changed after those runs, in response to the langfuse-docs review, and not yet
run against a project:

- [ ] `api.scores.create(id=..., data_type="CATEGORICAL")` without a config id
      is accepted for `review_policy`, a second write with the same id on the same
      UTC day replaces the first (on a later day it adds a second record, and
      Step 6 keeps the latest), and the `queue_id` and `signals` metadata come back through
      `details`. The 2026-10-07 runs covered `create_score`, not this call.
- [ ] `list_queue_items(status="COMPLETED")` returns only completed items, which
      is what Step 6 now reads.
- [ ] `list_queues` returns `score_config_ids` for an existing queue, which
      `ensure_queue` now checks.

Confirmed on the same project:

- [x] `api.scores_v3.get_many_v3(queue_id=..., source="ANNOTATION")` returns
      the queue's annotation scores with `subject` and `queue_id` populated:
      50 labels on 10 traces after scoring ten items in the UI.
- [x] A second run after annotations exist rebuilds the per-policy table in
      Step 6 from the stored `review_policy` metadata.
- [x] The langfuse.com and api.reference.langfuse.com links resolve; they were
      derived from the langfuse-docs file paths and the API tag names and
      opened in a browser on 2026-10-07.

Live run log:

- 2026-10-07, Langfuse Cloud, langfuse 4.17.0, project seeded with 300
  traces: first run read 1428 scores on 300 candidates through the v3
  endpoint (trace ids resolved from `subject`), created the five boolean
  score configs and the queue with plain-string arguments, added 40 of 40
  items, and took the empty-queue branch in Step 6.
- 2026-10-07, second run after scoring ten items in the UI: Step 3 excluded
  10 annotated, 40 selected earlier and 40 already queued (260 candidates);
  Step 5 added a second batch of 40 to the same queue; Step 6 read 50
  annotation labels and attributed them to severity (10), mixed (10),
  max_score (5) and random (1); the one deliberate disagreement showed up
  as negative_feedback agreement 0.9. Signals with no flagged and no
  positive rows render empty judge precision and recall cells, as designed.

Mechanics of the contribution, from the langfuse-docs README and recent
cookbook pull requests:

1. Copy the notebook to `cookbook/example_annotation_queue_prioritization.ipynb`
   in a fork of `langfuse/langfuse-docs`. The first markdown cell already
   carries the `NOTEBOOK_METADATA` comment that `scripts/move_docs.py` turns
   into front matter.
2. Run `bash scripts/update_cookbook_docs.sh` (needs `uv`) and commit the
   generated `content/guides/cookbook/example_annotation_queue_prioritization.mdx`.
3. Add the slug to `content/guides/cookbook/meta.json` and an entry
   `{"notebook": "example_annotation_queue_prioritization.ipynb", "docsPath": null, "isGuide": true}`
   to `cookbook/_routes.json`.
4. Run `pnpm run format`.
5. Use `langfuse-docs-pr.md` as the pull request text: offline reproduction,
   SDK version tested, no real credentials, and the live checks above with
   their outcome.
