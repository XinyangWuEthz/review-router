from __future__ import annotations

import numpy as np

from review_router.agreement import agreement_report, agreement_table, oov_share
from review_router.model import ModelConfig, TfidfLogitModel
from review_router.policy import load_policy

LABELS = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")


def test_agreement_table_is_hand_computable() -> None:
    #            below  | [0.5,0.8) | [0.8,1.0]
    score = np.array([0.1, 0.4, 0.6, 0.7, 0.9, 1.0])
    y = np.array([0, 1, 1, 0, 1, 1])
    table = agreement_table(y, score, edges=(0.5, 0.8, 1.0), min_rows=2)
    assert [(row["lo"], row["hi"]) for row in table] == [(None, 0.5), (0.5, 0.8), (0.8, 1.0)]
    assert [row["n"] for row in table] == [2, 2, 2]
    assert [row["coverage"] for row in table] == [1 / 3, 1 / 3, 1 / 3]
    assert [row["n_positive"] for row in table] == [1, 1, 2]
    assert [row["agreement"] for row in table] == [0.5, 0.5, 1.0]
    assert [row["recall_share"] for row in table] == [0.25, 0.25, 0.5]
    assert all(row["meets_min_rows"] for row in table)
    assert table[2]["mean_score"] == 0.95
    assert table[1]["agreement_ci95"][0] < 0.5 < table[1]["agreement_ci95"][1]

    # An empty segment reports None for the proportions and fails min_rows.
    empty = agreement_table(np.array([1]), np.array([0.2]), edges=(0.5, 1.0), min_rows=1)
    assert empty[1] == {
        "lo": 0.5,
        "hi": 1.0,
        "n": 0,
        "coverage": 0.0,
        "mean_score": None,
        "n_positive": 0,
        "agreement": None,
        "agreement_ci95": None,
        "recall_share": 0.0,
        "meets_min_rows": False,
    }


def test_agreement_report_uses_high_risk_edges_for_heavy_labels_and_reads_rules() -> None:
    policy = load_policy()
    rng = np.random.default_rng(0)
    n = 200
    y = (rng.random((n, 6)) < 0.3).astype(int)
    proba = rng.random((n, 6))
    final = np.where(proba.max(axis=1) > 0.9, "priority_review", "allow")
    rule_action = np.where(proba[:, 3] >= 0.3, "priority_review", "")
    report = agreement_report(
        y,
        proba,
        LABELS,
        policy,
        5.0,
        final_tier=final,
        rule_action=rule_action,
        strata={"identity_term_present": (rng.random(n) < 0.2).astype(int)},
    )
    assert report["per_label"]["threat"]["high_risk_edges"] is True
    assert report["per_label"]["toxic"]["high_risk_edges"] is False
    assert report["per_label"]["threat"]["edges"] == [0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0]
    assert sum(row["n"] for row in report["pooled_review"]) == n
    band = report["priority_band"]
    assert band["n"] == int((final == "priority_review").sum())
    assert band["n_rows_matching_priority_rule"] == int((rule_action == "priority_review").sum())
    r101 = report["high_risk_rules"]["R101_high_risk_priority"]
    assert r101["n"] == int((proba[:, 3] >= 0.3).sum())
    assert r101["agreement_label"] == y[proba[:, 3] >= 0.3, 3].mean()
    assert report["high_risk_rules"]["skipped"] == ["R102_hierarchy_violation"]
    assert set(report["strata"]["identity_term_present"]) == {"0", "1"}


def _fit(analyzer: str) -> TfidfLogitModel:
    texts = ["the cat sat", "the dog ran", "a cat and a dog", "the bird flew"]
    y = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    return TfidfLogitModel(ModelConfig(min_df=1, ngram_max=1, analyzer=analyzer)).fit(
        texts, y, seed=0
    )


def test_priority_minimum_includes_rule_promotions_below_the_first_edge() -> None:
    # R101 promotes 30 false positives at threat=0.35; the other 30 priority
    # rows are true toxic positives in the top score segment. Omitting the
    # leading bucket would incorrectly report a minimum agreement of 1.0.
    y = np.zeros((60, len(LABELS)), dtype=int)
    proba = np.zeros_like(y, dtype=float)
    proba[:30, LABELS.index("threat")] = 0.35
    proba[30:, LABELS.index("toxic")] = 0.999
    y[30:, LABELS.index("toxic")] = 1
    report = agreement_report(
        y,
        proba,
        LABELS,
        load_policy(),
        5.0,
        final_tier=np.full(60, "priority_review"),
        rule_action=np.array(["priority_review"] * 30 + [""] * 30),
    )
    band = report["priority_band"]
    assert band["table"][0]["n"] == 30
    assert band["table"][0]["agreement"] == 0.0
    assert band["lowest_segment_agreement"] == 0.0


def test_priority_rule_match_count_includes_already_priority_rows() -> None:
    # A high toxic score can put the row into model priority while R101 also
    # matches it. The diagnostic counts that match without claiming a promotion.
    y = np.array([[1, 0, 0, 0, 0, 0]])
    proba = np.array([[0.999, 0.0, 0.0, 0.35, 0.0, 0.0]])
    report = agreement_report(
        y,
        proba,
        LABELS,
        load_policy(),
        5.0,
        final_tier=np.array(["priority_review"]),
        rule_action=np.array(["priority_review"]),
    )
    band = report["priority_band"]
    assert band["n_rows_matching_priority_rule"] == 1
    assert "n_rules_promoted" not in band
    assert "already in the model's priority band" in band["note"]


def test_oov_share_is_one_for_unseen_text_and_low_for_training_text() -> None:
    model = _fit("word")
    shares = oov_share(model, ["zebra quokka", "the cat sat", "the cat zebra", ""])
    assert shares is not None
    assert shares.tolist() == [1.0, 0.0, 1 / 3, 0.0]


def test_oov_share_is_none_without_a_word_vectorizer() -> None:
    assert oov_share(_fit("char_wb"), ["anything"]) is None
    assert oov_share(_fit("word+char_wb"), ["anything"]) is not None
