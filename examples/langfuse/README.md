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

The policies mirror the benchmark's queue orderings: `random` and `max_score`
are baselines, `severity` is the default ordering (maximum of probability
times a declared weight), `uncertainty` targets judge calibration, and
`mixed` splits the budget by quota. Selection under a budget reallocates
reviewer time; it does not add any, and the notebook reports the displaced
work next to every gain.

## Run

```bash
pip install -e ".[ml,dev]" langfuse nbconvert ipykernel
python -m pytest -q tests/test_langfuse_example.py
# Execute offline (synthetic pool); outputs are rewritten in place.
jupyter nbconvert --to notebook --execute --inplace \
  examples/langfuse/example_annotation_queue_prioritization.ipynb
```

Set the two keys and `LANGFUSE_BASE_URL` to run against a project. The live
path is written against the public API as exposed by `langfuse` 4.17.0
(`api.scores.get_many`, `api.score_configs`, `api.annotation_queues`,
`create_score`); it was checked against the SDK's signatures, not yet against
a live project.

## Before opening the langfuse-docs pull request

Verified on a live project, once:

- [ ] The queue lists items in insertion order. If not, the fallback in the
      notebook text applies: one queue per tier.
- [ ] Adding a trace that is already in the queue is rejected or ignored;
      the notebook skips known ids first, so either behaviour is fine.
- [ ] `create_score(..., data_type="CATEGORICAL")` without a config id is
      accepted for the `review_policy` marker.
- [ ] `api.scores.get_many(queue_id=...)` returns the annotation scores for the queue.
- [ ] The langfuse.com and api.reference.langfuse.com links resolve; they were
      derived from the langfuse-docs file paths, not opened from this sandbox.

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
