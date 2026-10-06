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
`create_score`. The page-based `api.scores.get_many` is the deprecated v2 read
path and is not used: the SDK marks it for removal on Langfuse Cloud on
2026-11-16 and on self-hosted deployments at the Langfuse v4 upgrade. The live
path was checked against the SDK's signatures and types, not yet against a
project.

## Before opening the langfuse-docs pull request

Read from the Langfuse server source and API definition while writing, so the
notebook text already relies on them; confirm once in the UI of a live project:

- [ ] Annotators are served pending items oldest first, so insertion order is
      the priority (the items endpoint lists newest first). If the served
      order differs, the fallback in the notebook text applies: one queue per
      tier.
- [ ] The API does not reject a trace already in the queue; the notebook's
      `queued_trace_ids` check is what prevents duplicates.
- [ ] `create_score(..., data_type="CATEGORICAL")` without a config id is
      accepted for the `review_policy` marker, and its `metadata` comes back
      through the `details` field group.

Still to verify on a live project:

- [ ] `api.scores_v3.get_many_v3(queue_id=..., source="ANNOTATION")` returns
      the queue's annotation scores with `subject` and `queue_id` populated.
- [ ] A second run after annotations exist rebuilds the per-policy table in
      Step 6 from the stored `review_policy` metadata.
- [ ] The langfuse.com and api.reference.langfuse.com links resolve; they were
      derived from the langfuse-docs file paths and the API tag names, not
      opened from this sandbox.

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
5. In the pull request text: offline reproduction, SDK version tested, no real
   credentials, and the live checks above with their outcome.
