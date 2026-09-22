"""Round-1 model: word TF-IDF, six independent logistic heads, Platt calibration.

Deliberately simple so the whole path (data -> prediction -> routing ->
evaluation) can be re-run in minutes. Embeddings and fusion are later rounds
and must be judged under this same protocol.

Platt scaling is implemented directly (a one-dimensional logistic regression
on each head's margin, fitted on the calibration split) rather than through
scikit-learn's prefit-calibration wrapper, whose API changed between releases.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from review_router.data import LABELS

__all__ = ["ModelConfig", "PlattCalibrator", "ConstantHead", "TfidfLogitModel"]


@dataclass(frozen=True)
class ModelConfig:
    max_features: int = 200_000
    ngram_max: int = 2
    min_df: int = 3
    C: float = 4.0
    max_iter: int = 1000


@dataclass
class PlattCalibrator:
    """p = sigmoid(a * margin + b), with (a, b) fitted on held-out margins."""

    a: float = 1.0
    b: float = 0.0

    def fit(self, margins: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        if len(np.unique(y)) < 2:
            # Nothing to calibrate against; keep the identity mapping.
            self.a, self.b = 1.0, 0.0
            return self
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(margins.reshape(-1, 1), y)
        self.a = float(lr.coef_[0, 0])
        self.b = float(lr.intercept_[0])
        return self

    def transform(self, margins: np.ndarray) -> np.ndarray:
        z = self.a * margins + self.b
        return 1.0 / (1.0 + np.exp(-z))


@dataclass
class ConstantHead:
    """Stand-in for a label with a single class in the training split.

    scikit-learn's liblinear cannot fit one-class targets. The fixed margin
    reflects the observed class: negative for all-zero targets and positive
    for all-one targets.
    """

    margin: float = -20.0

    def decision_function(self, x: spmatrix) -> np.ndarray:
        return np.full(x.shape[0], self.margin)


@dataclass
class TfidfLogitModel:
    config: ModelConfig = field(default_factory=ModelConfig)
    labels: tuple[str, ...] = LABELS
    vectorizer: TfidfVectorizer | None = None
    heads: dict[str, LogisticRegression | ConstantHead] = field(default_factory=dict)
    calibrators: dict[str, PlattCalibrator] = field(default_factory=dict)

    def fit(self, texts: Iterable[str], y: np.ndarray, seed: int) -> TfidfLogitModel:
        """Fit the vocabulary and the six heads on the training split only."""
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, self.config.ngram_max),
            min_df=self.config.min_df,
            max_features=self.config.max_features,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )
        x = self.vectorizer.fit_transform(texts)
        self.heads = {}
        self.calibrators = {}
        for j, label in enumerate(self.labels):
            head: LogisticRegression | ConstantHead
            if len(np.unique(y[:, j])) < 2:
                head = ConstantHead(margin=20.0 if y[0, j] == 1 else -20.0)
            else:
                head = LogisticRegression(
                    C=self.config.C,
                    max_iter=self.config.max_iter,
                    solver="liblinear",
                    random_state=seed,
                )
                head.fit(x, y[:, j])
            self.heads[label] = head
            self.calibrators[label] = PlattCalibrator()
        return self

    def _features(self, texts: Iterable[str]) -> spmatrix:
        if self.vectorizer is None:
            raise RuntimeError("model is not fitted")
        return self.vectorizer.transform(texts)

    def decision(self, texts: Iterable[str]) -> np.ndarray:
        """Uncalibrated margins, shape (n, n_labels)."""
        x = self._features(texts)
        return np.column_stack([self.heads[label].decision_function(x) for label in self.labels])

    def calibrate(self, texts: Iterable[str], y: np.ndarray) -> TfidfLogitModel:
        """Fit one Platt calibrator per head on the calibration split."""
        margins = self.decision(texts)
        for j, label in enumerate(self.labels):
            self.calibrators[label] = PlattCalibrator().fit(margins[:, j], y[:, j])
        return self

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        """Calibrated probabilities, shape (n, n_labels), columns in self.labels order."""
        margins = self.decision(texts)
        return np.column_stack(
            [
                self.calibrators[label].transform(margins[:, j])
                for j, label in enumerate(self.labels)
            ]
        )
