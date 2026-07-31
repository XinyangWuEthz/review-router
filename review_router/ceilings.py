"""Arithmetic precision ceilings for the Jigsaw scored test set.

The point of this module: on this corpus the official metric (mean column-wise
ROC-AUC) is saturated and says nothing about the operating point a review queue
actually runs at. Class prevalence alone caps how precise any detector can be at
a given false-positive rate, and for the rare high-harm labels that cap is low
enough to decide the routing policy on its own.

Every number here is arithmetic on published label counts, not a model result.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LabelCounts", "SCORED_TEST_COUNTS", "precision_ceiling", "ceiling_table"]


@dataclass(frozen=True)
class LabelCounts:
    """Positive and negative counts for one label on a fixed evaluation slice."""

    positives: int
    negatives: int

    @property
    def total(self) -> int:
        return self.positives + self.negatives

    @property
    def prevalence(self) -> float:
        return self.positives / self.total


# Jigsaw Toxic Comment Classification Challenge (2018), scored test rows only.
#
# test.csv has 153,164 rows but only 63,978 are scored; the other 89,186 carry
# -1 in every column of test_labels.csv and must be filtered out before any
# evaluation. See review_router.data.drop_unscored.
SCORED_TEST_ROWS = 63_978

SCORED_TEST_COUNTS: dict[str, LabelCounts] = {
    "toxic": LabelCounts(positives=6_090, negatives=57_888),
    "obscene": LabelCounts(positives=3_691, negatives=60_287),
    "insult": LabelCounts(positives=3_427, negatives=60_551),
    "identity_hate": LabelCounts(positives=712, negatives=63_266),
    "severe_toxic": LabelCounts(positives=367, negatives=63_611),
    "threat": LabelCounts(positives=211, negatives=63_767),
}


def precision_ceiling(counts: LabelCounts, recall: float, fpr: float) -> float:
    """Precision attainable at a hypothetical (recall, FPR) operating point.

    TP = recall * P, FP = fpr * N, so precision = TP / (TP + FP). This is an
    identity over the label counts — it describes what the class balance
    permits, not what any model achieves.
    """
    if not 0.0 <= recall <= 1.0:
        raise ValueError(f"recall must be in [0, 1], got {recall}")
    if not 0.0 <= fpr <= 1.0:
        raise ValueError(f"fpr must be in [0, 1], got {fpr}")

    true_positives = recall * counts.positives
    false_positives = fpr * counts.negatives
    if true_positives + false_positives == 0.0:
        return 0.0
    return true_positives / (true_positives + false_positives)


def ceiling_table(
    recall: float = 0.5,
    fprs: tuple[float, ...] = (0.01, 0.001),
    counts: dict[str, LabelCounts] | None = None,
) -> dict[str, dict[float, float]]:
    """Precision ceiling per label at each false-positive rate."""
    table = counts if counts is not None else SCORED_TEST_COUNTS
    return {
        label: {fpr: precision_ceiling(c, recall, fpr) for fpr in fprs}
        for label, c in table.items()
    }
