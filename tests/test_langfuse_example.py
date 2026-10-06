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
        ("uncertainty", ["d", "e", "a", "a"]),
    ],
)
def test_ranked_selection_order(
    pool: pd.DataFrame, config: Any, policy: str, expected: list[str]
) -> None:
    chosen = BS.select(pool, policy, config)
    assert len(chosen) == config.budget
    assert chosen["rank"].tolist() == [1, 2, 3, 4]
    assert (chosen["source"] == policy).all()
    if policy == "uncertainty":
        # d (0.5 safety) first, e (0.6 hallucination) second; the rest tie-break by id.
        assert chosen["trace_id"].tolist()[:2] == ["d", "e"]
    else:
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


def test_mixed_refills_from_severity_when_quotas_overlap(config: Any) -> None:
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


def test_overlap_matrix_diagonal_is_selection_size(pool: pd.DataFrame, config: Any) -> None:
    selections = BS.select_all(pool, config)
    matrix = BS.overlap_matrix(selections)
    assert list(matrix.index) == list(BS.POLICIES)
    assert all(matrix.loc[p, p] == len(selections[p]) for p in BS.POLICIES)
    assert matrix.equals(matrix.T)


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
    assert safety["agreement"] == pytest.approx(2 / 3)
    assert safety["judge_precision"] == pytest.approx(1.0)
    assert safety["judge_recall"] == pytest.approx(0.5)
    assert table.loc["tone", "annotated"] == 0


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
    code = "\n".join(_source(c) for c in nb["cells"] if c["cell_type"] == "code")
    # Live calls are gated so the notebook runs end to end without credentials.
    assert "LANGFUSE_PUBLIC_KEY" in code and "LANGFUSE_SECRET_KEY" in code
    for call in ("create_queue(", "create_queue_item(", "create_score(", "get_many("):
        assert call in code
    assert "pk-lf-" not in code.replace("pk-lf-...", "") and "sk-lf-" not in code.replace(
        "sk-lf-...", ""
    ), "no real-looking credentials in the notebook"
