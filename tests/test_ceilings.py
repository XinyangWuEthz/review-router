from __future__ import annotations

import pytest

from review_router.ceilings import (
    SCORED_TEST_COUNTS,
    SCORED_TEST_ROWS,
    LabelCounts,
    ceiling_table,
    precision_ceiling,
)


def test_counts_sum_to_the_scored_test_size() -> None:
    """Every label partitions the same 63,978 scored rows."""
    for label, counts in SCORED_TEST_COUNTS.items():
        assert counts.total == SCORED_TEST_ROWS, label


def test_rare_labels_are_not_auto_actionable_at_one_percent_fpr() -> None:
    """The project's thesis, as an assertion.

    At 50% recall and a 1% false-positive rate the class balance alone caps
    threat precision near 14% — nowhere near the 0.99 auto-action floor.
    """
    threat = precision_ceiling(SCORED_TEST_COUNTS["threat"], recall=0.5, fpr=0.01)
    assert threat == pytest.approx(0.142, abs=0.001)

    toxic = precision_ceiling(SCORED_TEST_COUNTS["toxic"], recall=0.5, fpr=0.01)
    assert toxic == pytest.approx(0.840, abs=0.001)

    assert toxic > 5 * threat, "the common/rare precision gap is the whole point"


def test_tightening_fpr_lifts_the_rare_label_ceiling() -> None:
    """Rare labels only become usable at the extreme left edge of the ROC curve."""
    counts = SCORED_TEST_COUNTS["threat"]
    at_1pct = precision_ceiling(counts, recall=0.5, fpr=0.01)
    at_0_1pct = precision_ceiling(counts, recall=0.5, fpr=0.001)
    assert at_0_1pct == pytest.approx(0.623, abs=0.001)
    assert at_0_1pct > 4 * at_1pct


def test_ceiling_is_monotone_in_fpr() -> None:
    counts = SCORED_TEST_COUNTS["insult"]
    ceilings = [precision_ceiling(counts, 0.5, fpr) for fpr in (0.05, 0.01, 0.001)]
    assert ceilings == sorted(ceilings)


def test_ceiling_table_covers_every_label() -> None:
    table = ceiling_table()
    assert set(table) == set(SCORED_TEST_COUNTS)
    for per_fpr in table.values():
        assert set(per_fpr) == {0.01, 0.001}


def test_zero_recall_gives_zero_precision() -> None:
    assert precision_ceiling(SCORED_TEST_COUNTS["toxic"], recall=0.0, fpr=0.01) == 0.0


def test_perfect_separation_gives_perfect_precision() -> None:
    assert precision_ceiling(LabelCounts(10, 100), recall=1.0, fpr=0.0) == 1.0


@pytest.mark.parametrize(("recall", "fpr"), [(-0.1, 0.01), (1.1, 0.01), (0.5, -0.1), (0.5, 2.0)])
def test_out_of_range_inputs_are_rejected(recall: float, fpr: float) -> None:
    with pytest.raises(ValueError):
        precision_ceiling(SCORED_TEST_COUNTS["toxic"], recall=recall, fpr=fpr)
