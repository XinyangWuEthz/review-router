"""The Langfuse cookbook example: selection module, notebook sync and notebook shape."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "langfuse"
MODULE = EXAMPLE / "budget_selection.py"
NOTEBOOK = EXAMPLE / "example_annotation_queue_prioritization.ipynb"
WEIGHTS = {"safety_violation": 10.0, "hallucination": 6.0, "negative_feedback": 5.0, "tone": 1.0}


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("budget_selection", MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["budget_selection"] = module
    spec.loader.exec_module(module)
    return module


BS = _module()


@pytest.fixture
def config() -> Any:
    return BS.SelectionConfig(budget=4, weights=WEIGHTS, seed=3)


@pytest.fixture
def pool() -> pd.DataFrame:
    # Hand-built rows with known orderings under each policy.
    return pd.DataFrame(
        {
            "trace_id": ["a", "b", "c", "d", "e", "f"],
            "safety_violation": [0.05, 0.95, 0.00, 0.50, np.nan, 0.10],
            "hallucination": [0.90, 0.10, 0.20, 0.10, 0.60, 0.10],
            "negative_feedback": [0.00, 0.00, 1.00, 0.00, 0.00, 0.00],
            "tone": [0.30, 0.20, 0.10, 0.10, 0.10, 0.99],
        }
    )


def test_severity_is_max_probability_times_weight(pool: pd.DataFrame) -> None:
    scores = BS.severity_scores(pool, WEIGHTS)
    expected = [0.9 * 6, 0.95 * 10, 1.0 * 5, 0.5 * 10, 0.6 * 6, 0.1 * 10]
    assert scores.tolist() == pytest.approx(expected)


def test_missing_signal_is_no_evidence(pool: pd.DataFrame) -> None:
    # Row e has NaN safety; it must not rank as if safety were high, nor raise.
    assert BS.max_scores(pool, WEIGHTS)[pool["trace_id"] == "e"].item() == pytest.approx(0.6)


def test_uncertainty_prefers_half_over_confident_scores(pool: pd.DataFrame) -> None:
    scores = BS.uncertainty_scores(pool, WEIGHTS)
    assert scores.idxmax() == 3  # row d: safety exactly 0.5
    assert scores.iloc[3] == pytest.approx(1.0)


def test_signals_outside_unit_interval_are_rejected(pool: pd.DataFrame) -> None:
    bad = pool.assign(tone=[1.5, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        BS.severity_scores(bad, WEIGHTS)


def test_pool_must_name_every_signal(pool: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="lacks signal columns"):
        BS.severity_scores(pool.drop(columns=["tone"]), WEIGHTS)


@pytest.mark.parametrize(
    "policy, expected",
    [
        ("severity", ["b", "a", "c", "d"]),  # c and d tie at 5.0; id breaks it
        ("max_score", ["c", "f", "b", "a"]),
        ("uncertainty", ["d", "e", "a", "b"]),  # b and c tie at 0.4; id breaks it
    ],
)
def test_ranked_selection_order(
    pool: pd.DataFrame, config: Any, policy: str, expected: list[str]
) -> None:
    chosen = BS.select(pool, policy, config)
    assert len(chosen) == config.budget
    assert chosen["rank"].tolist() == [1, 2, 3, 4]
    assert (chosen["source"] == policy).all()
    assert chosen["trace_id"].tolist() == expected


def test_ties_break_by_trace_id_for_determinism(config: Any) -> None:
    tied = pd.DataFrame({"trace_id": list("zyx"), **{name: [0.5] * 3 for name in WEIGHTS}})
    assert BS.rank(tied, "severity", config)["trace_id"].tolist() == ["x", "y", "z"]


def test_random_is_seeded_and_reproducible(pool: pd.DataFrame) -> None:
    first = BS.select(pool, "random", BS.SelectionConfig(budget=3, weights=WEIGHTS, seed=7))
    again = BS.select(pool, "random", BS.SelectionConfig(budget=3, weights=WEIGHTS, seed=7))
    other = BS.select(pool, "random", BS.SelectionConfig(budget=3, weights=WEIGHTS, seed=8))
    assert first["trace_id"].tolist() == again["trace_id"].tolist()
    assert first["trace_id"].tolist() != other["trace_id"].tolist()


def test_budget_larger_than_pool_returns_whole_pool(pool: pd.DataFrame) -> None:
    config = BS.SelectionConfig(budget=50, weights=WEIGHTS)
    for policy in BS.POLICIES:
        assert sorted(BS.select(pool, policy, config)["trace_id"]) == sorted(pool["trace_id"])


def test_mixed_quota_counts_sum_to_budget() -> None:
    quota = {"severity": 0.6, "uncertainty": 0.25, "random": 0.15}
    for budget in (1, 2, 7, 20, 33):
        counts = BS._quota_counts(budget, quota)
        assert sum(counts.values()) == budget
        assert all(count >= 0 for count in counts.values())
    assert BS._quota_counts(20, quota) == {"severity": 12, "uncertainty": 5, "random": 3}


def test_mixed_selection_is_unique_and_spends_the_budget(pool: pd.DataFrame) -> None:
    config = BS.SelectionConfig(
        budget=5, weights=WEIGHTS, seed=1, mixed_quota={"severity": 0.6, "uncertainty": 0.4}
    )
    chosen = BS.select(pool, "mixed", config)
    assert len(chosen) == 5
    assert chosen["trace_id"].is_unique
    assert chosen["rank"].tolist() == [1, 2, 3, 4, 5]
    assert (chosen["policy"] == "mixed").all()
    assert set(chosen["source"]) <= {"severity", "uncertainty"}
    # Severity's quota (3) comes first and in severity order.
    assert chosen["trace_id"].tolist()[:3] == ["b", "a", "c"]


def test_mixed_refills_from_severity_when_quotas_overlap() -> None:
    # Every signal 0.5: uncertainty and severity agree on every row, so the
    # uncertainty quota is fully starved by severity's picks and must be refilled.
    tied = pd.DataFrame({"trace_id": list("abcdef"), **{name: [0.5] * 6 for name in WEIGHTS}})
    config = BS.SelectionConfig(
        budget=4, weights=WEIGHTS, mixed_quota={"severity": 0.5, "uncertainty": 0.5}
    )
    chosen = BS.select(tied, "mixed", config)
    assert len(chosen) == 4
    assert chosen["trace_id"].is_unique


def test_duplicate_or_blank_trace_ids_are_rejected(pool: pd.DataFrame, config: Any) -> None:
    with pytest.raises(ValueError, match="unique"):
        BS.select(pd.concat([pool, pool.head(1)]), "severity", config)
    with pytest.raises(ValueError, match="non-empty trace_id"):
        BS.select(pool.assign(trace_id=["", "b", "c", "d", "e", "f"]), "severity", config)
    with pytest.raises(ValueError, match="non-empty trace_id"):
        BS.select(pool.assign(trace_id=[None, "b", "c", "d", "e", "f"]), "severity", config)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"budget": 0}, "positive integer"),
        ({"budget": True}, "positive integer"),
        ({"weights": {}}, "at least one signal"),
        ({"weights": {"x": 0}}, "positive and finite"),
        ({"mixed_quota": {"severity": 0.5, "fifo": 0.5}}, "unknown policies"),
        ({"mixed_quota": {"severity": 0.7, "random": 0.2}}, "sum to 1"),
        ({"mixed_quota": {"severity": 1.5, "random": -0.5}}, "nonnegative"),
        ({"mixed_quota": {"severity": float("nan"), "random": 1.0}}, "finite"),
    ],
)
def test_config_validation(kwargs: dict[str, Any], message: str) -> None:
    base: dict[str, Any] = {"budget": 3, "weights": WEIGHTS}
    with pytest.raises(ValueError, match=message):
        BS.SelectionConfig(**{**base, **kwargs})


def test_compare_policies_accounts_for_every_flagged_row(pool: pd.DataFrame, config: Any) -> None:
    table = BS.compare_policies(pool, config)
    flagged = int((pool[list(WEIGHTS)].fillna(0) >= 0.5).any(axis=1).sum())
    assert set(table["policy"]) == set(BS.POLICIES)
    assert (table["selected"] == config.budget).all()
    assert (table["flagged_selected"] + table["flagged_left_out"] == flagged).all()
    severity = table.set_index("policy").loc["severity"]
    assert severity["safety_violation_flagged"] == 2  # b and d


def test_compare_policies_with_oracle_truth(config: Any) -> None:
    pool = BS.synthetic_pool(600, WEIGHTS, seed=5)
    table = BS.compare_policies(pool, config, truth="true_high_risk").set_index("policy")
    assert (table["true_positive_selected"] <= config.budget).all()
    assert ((table["true_positive_share"] >= 0) & (table["true_positive_share"] <= 1)).all()


def test_synthetic_pool_is_reproducible_and_bounded() -> None:
    pool = BS.synthetic_pool(300, WEIGHTS, seed=2)
    assert pool.equals(BS.synthetic_pool(300, WEIGHTS, seed=2))
    assert pool["trace_id"].is_unique
    for name in WEIGHTS:
        assert pool[name].dropna().between(0, 1).all()
        assert pool[name].isna().any()  # some signals are missing on purpose
        assert set(pool[f"true_{name}"].unique()) <= {0, 1}
    assert pool["true_high_risk"].isin([0, 1]).all()
    with pytest.raises(ValueError, match="positive"):
        BS.synthetic_pool(0, WEIGHTS)


def test_overlap_matrix_counts_shared_traces(pool: pd.DataFrame, config: Any) -> None:
    selections = BS.select_all(pool, config)
    matrix = BS.overlap_matrix(selections)
    assert list(matrix.index) == list(BS.POLICIES)
    assert all(matrix.loc[p, p] == len(selections[p]) for p in BS.POLICIES)
    assert matrix.equals(matrix.T)
    # severity picks b, a, c, d; max_score picks c, f, b, a: three in common.
    assert matrix.loc["severity", "max_score"] == 3
    assert matrix.loc["severity", "uncertainty"] == len(
        set(selections["severity"]["trace_id"]) & set(selections["uncertainty"]["trace_id"])
    )


def test_truth_nan_counts_as_unlabelled_and_non_binary_is_rejected(
    pool: pd.DataFrame, config: Any
) -> None:
    # severity selects b, a, c, d. Truth: a, c, f positive; b and e unlabelled.
    labelled = pool.assign(truth=[1.0, np.nan, 1.0, 0.0, np.nan, 1.0])
    table = BS.compare_policies(labelled, config, ["severity"], truth="truth").iloc[0]
    assert table["true_positive_selected"] == 2
    assert table["true_positive_share"] == pytest.approx(2 / 3)
    nullable = pool.assign(truth=pd.array([True, None, True, False, None, True], dtype="boolean"))
    assert BS.compare_policies(nullable, config, ["severity"], truth="truth").equals(
        BS.compare_policies(labelled, config, ["severity"], truth="truth")
    )
    with pytest.raises(ValueError, match="0/1"):
        BS.compare_policies(pool.assign(truth=2), config, ["severity"], truth="truth")


@pytest.mark.parametrize("name", sorted(BS.RESERVED_COLUMNS))
def test_reserved_column_names_are_rejected(pool: pd.DataFrame, config: Any, name: str) -> None:
    with pytest.raises(ValueError, match="reserved"):
        BS.select(pool.assign(**{name: 0.1}), "severity", config)
    with pytest.raises(ValueError, match="reserved"):
        BS.SelectionConfig(budget=3, weights={**WEIGHTS, name: 1.0})


def test_every_selection_carries_the_severity_score(pool: pd.DataFrame, config: Any) -> None:
    expected = BS.severity_scores(pool, WEIGHTS).set_axis(pool["trace_id"])
    for policy in BS.POLICIES:
        chosen = BS.select(pool, policy, config).set_index("trace_id")
        assert chosen["severity_score"].tolist() == pytest.approx(
            expected.loc[chosen.index].tolist()
        )
    max_score = BS.select(pool, "max_score", config)
    assert not np.allclose(max_score["score"], max_score["severity_score"])


def test_string_truth_and_bare_policy_strings(pool: pd.DataFrame, config: Any) -> None:
    words = pool.assign(truth=["True", "false", "TRUE", "False", "false", "true"])
    table = BS.compare_policies(words, config, ["severity"], truth="truth").iloc[0]
    assert table["true_positive_selected"] == 2  # a and c among b, a, c, d
    with pytest.raises(ValueError, match="0/1"):
        BS.compare_policies(pool.assign(truth="yes"), config, ["severity"], truth="truth")
    with pytest.raises(TypeError, match="single string"):
        BS.compare_policies(pool, config, "severity")
    with pytest.raises(TypeError, match="single string"):
        BS.select_all(pool, config, "severity")


def test_numpy_numbers_are_accepted_as_budget_and_weights(pool: pd.DataFrame) -> None:
    config = BS.SelectionConfig(
        budget=np.int64(3), weights={name: np.float32(w) for name, w in WEIGHTS.items()}
    )
    assert len(BS.select(pool, "severity", config)) == 3
    assert BS.select(pool, "severity", config)["severity_score"].iloc[0] == pytest.approx(9.5)


def test_name_iterables_are_materialised_and_bare_strings_rejected(config: Any) -> None:
    pool = BS.synthetic_pool(30, WEIGHTS, seed=2)
    chosen = BS.select(pool, "severity", config)
    labels = BS.simulate_annotations(chosen, pool, (name for name in WEIGHTS), flip=0.0)
    assert len(labels) == len(chosen) * len(WEIGHTS)
    with pytest.raises(TypeError, match="single string"):
        BS.annotation_precision({"severity": chosen}, labels, "human_tone")
    with pytest.raises(TypeError, match="single string"):
        BS.judge_agreement(pool, labels, "tone")


def test_annotation_precision_counts_only_labelled_rows(pool: pd.DataFrame, config: Any) -> None:
    selections = {"severity": BS.select(pool, "severity", config)}  # b, a, d, c
    annotations = pd.DataFrame(
        {
            "trace_id": ["b", "b", "a", "d"],
            "name": ["human_safety", "human_tone", "human_safety", "human_safety"],
            "value": [1.0, 0.0, 0.0, 1.0],
        }
    )
    table = BS.annotation_precision(selections, annotations, ["human_safety"]).iloc[0]
    assert table["annotated"] == 3  # c has no label yet
    assert table["positive"] == 2
    assert table["precision"] == pytest.approx(2 / 3)
    assert 0 <= table["ci_low"] < table["precision"] < table["ci_high"] <= 1
    empty = BS.annotation_precision(selections, annotations.head(0), ["human_safety"]).iloc[0]
    assert empty["annotated"] == 0 and np.isnan(empty["precision"])
    # No policy at all (a live run before any annotation): the frame keeps its columns.
    none = BS.annotation_precision({}, annotations.head(0), ["human_safety"])
    assert none.empty and list(none.columns)[0] == "policy"
    none.set_index("policy")


def test_judge_agreement_on_annotated_rows(pool: pd.DataFrame) -> None:
    annotations = pd.DataFrame(
        {
            "trace_id": ["b", "a", "f", "e"],
            "name": ["human_safety_violation"] * 3 + ["human_hallucination"],
            "value": [1.0, 0.0, 1.0, 1.0],
        }
    )
    table = BS.judge_agreement(pool, annotations, ["safety_violation", "hallucination", "tone"])
    table = table.set_index("signal")
    safety = table.loc["safety_violation"]
    # Judge flags b only; humans mark b and f. Agreement 2/3, precision 1, recall 1/2.
    assert safety["annotated"] == 3
    assert safety["judge_missing"] == 0
    assert safety["agreement"] == pytest.approx(2 / 3)
    assert safety["judge_precision"] == pytest.approx(1.0)
    assert safety["judge_recall"] == pytest.approx(0.5)
    assert table.loc["tone", "annotated"] == 0
    assert table.loc["tone", "judge_missing"] == 0


def test_judge_agreement_leaves_out_rows_the_judge_did_not_score(pool: pd.DataFrame) -> None:
    # Row e has no safety score. Read as 0, the human's yes on e would count as a judge miss;
    # it is no prediction at all, so it is counted apart instead.
    annotations = pd.DataFrame(
        {
            "trace_id": ["b", "a", "e"],
            "name": ["human_safety_violation"] * 3,
            "value": [1.0, 0.0, 1.0],
        }
    )
    safety = BS.judge_agreement(pool, annotations, ["safety_violation"]).iloc[0]
    assert safety["annotated"] == 2
    assert safety["judge_missing"] == 1
    assert safety["agreement"] == pytest.approx(1.0)  # b flagged and positive, a neither
    assert safety["judge_recall"] == pytest.approx(1.0)
    only_missing = BS.judge_agreement(pool, annotations.tail(1), ["safety_violation"]).iloc[0]
    assert only_missing["annotated"] == 0
    assert only_missing["judge_missing"] == 1
    assert np.isnan(only_missing["agreement"])


def test_simulated_annotations_cover_every_selected_row(config: Any) -> None:
    pool = BS.synthetic_pool(200, WEIGHTS, seed=9)
    chosen = BS.select(pool, "mixed", config)
    annotations = BS.simulate_annotations(chosen, pool, WEIGHTS, seed=1, flip=0.0)
    assert len(annotations) == len(chosen) * len(WEIGHTS)
    truth = pool.set_index("trace_id")
    for row in annotations.itertuples():
        signal = row.name.removeprefix("human_")
        assert row.value == truth.at[row.trace_id, f"true_{signal}"]


# ---------------------------------------------------------------- notebook


def _notebook() -> dict[str, Any]:
    result: dict[str, Any] = json.loads(NOTEBOOK.read_text())
    return result


def _source(cell: dict[str, Any]) -> str:
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else str(source)


def test_notebook_inlines_the_module_verbatim() -> None:
    cells = [
        c
        for c in _notebook()["cells"]
        if "budget_selection" in c.get("metadata", {}).get("tags", [])
    ]
    assert len(cells) == 1, "exactly one code cell must carry the budget_selection tag"
    assert cells[0]["cell_type"] == "code"
    assert _source(cells[0]).strip() == MODULE.read_text().strip()


def test_notebook_carries_langfuse_docs_metadata_header() -> None:
    first = _notebook()["cells"][0]
    assert first["cell_type"] == "markdown"
    header = _source(first).lstrip()
    assert header.startswith("<!-- NOTEBOOK_METADATA")
    for key in ("source", "title", "sidebarTitle", "description", "category"):
        assert f'{key}: "' in header.split("-->", 1)[0]
    assert "\n# " in header  # the H1 follows the metadata comment


def test_notebook_structure_is_valid_and_offline_safe() -> None:
    nb = _notebook()
    assert nb["nbformat"] == 4
    kinds = {c["cell_type"] for c in nb["cells"]}
    assert kinds <= {"markdown", "code"}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert "execution" not in cell["metadata"], "strip per-cell timing before committing"
        for out in cell["outputs"]:
            assert out["output_type"] != "error", cell["id"]
            # HTML tables carry a <style> block that does not compile as MDX in langfuse-docs,
            # and plain-text tables become indented blocks that MDX renders as prose.
            assert "text/html" not in out.get("data", {}), cell["id"]
            if out["output_type"] == "execute_result":
                plain = out["data"].get("text/plain", "")
                plain = "".join(plain) if isinstance(plain, list) else plain
                assert "\n" not in plain.strip(), (
                    f"{cell['id']}: render tables with show(), not the DataFrame repr"
                )
    markdown_outputs = sum(
        "text/markdown" in out.get("data", {})
        for cell in nb["cells"]
        if cell["cell_type"] == "code"
        for out in cell["outputs"]
    )
    assert markdown_outputs >= 5, "tables must be emitted as text/markdown"
    code = "\n".join(_source(c) for c in nb["cells"] if c["cell_type"] == "code")
    # Live calls are gated so the notebook runs end to end without credentials.
    assert "LANGFUSE_PUBLIC_KEY" in code and "LANGFUSE_SECRET_KEY" in code
    for call in ("create_queue(", "create_queue_item(", "scores.create(", "get_many_v3("):
        assert call in code
    assert "scores.get_many(" not in code, "the v2 scores endpoint is deprecated; use scores_v3"
    # create_score batches in the background and never raises, so it cannot guarantee that a
    # queued trace has its decision; the synchronous endpoint must run before the queue insert.
    assert "langfuse.create_score(" not in code
    push = code[code.index("def push_selection") :]
    assert push.index("scores.create(") < push.index("create_queue_item("), (
        "push_selection must record the decision before adding the queue item"
    )
    assert "pk-lf-" not in code.replace("pk-lf-...", "") and "sk-lf-" not in code.replace(
        "sk-lf-...", ""
    ), "no real-looking credentials in the notebook"
