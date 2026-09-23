from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CMP = _module(ROOT / "scripts" / "compare_runs.py")

# Two labels are enough to hand-check: weight 1 (toxic-like) and weight 10 (threat-like).
WEIGHTS = np.array([1.0, 0.0, 0.0, 10.0, 0.0, 0.0])


def _y(rows: list[tuple[int, int]]) -> np.ndarray:
    y = np.zeros((len(rows), 6), dtype=int)
    for i, (a, b) in enumerate(rows):
        y[i, 0], y[i, 3] = a, b
    return y


def test_recall_table_counts_flagged_top_tier_and_allow() -> None:
    y = _y([(1, 1), (1, 0), (0, 0), (0, 1), (1, 0)])
    final = np.array(["priority_review", "human_review", "human_review", "allow", "allow"])
    out = CMP.recall_table(y, final, WEIGHTS, 5.0, "priority_review")
    assert out["flagged"] == {"k": 3, "n": 5}
    # Positives: rows 0, 1, 3, 4. Flagged among them: 0, 1.
    assert out["positives_flagged"] == {"k": 2, "n": 4, "share": 0.5}
    assert out["positives_in_top"] == {"k": 1, "n": 4, "share": 0.25}
    # High risk (weight 10 >= 5): rows 0 and 3; row 3 was left in allow.
    assert out["high_risk_flagged"]["k"] == 1
    assert out["high_risk_in_allow"] == {"k": 1, "n": 2, "share": 0.5}
    assert out["harm_total"] == pytest.approx(10 + 1 + 0 + 10 + 1)
    assert out["harm_flagged"] == pytest.approx(11)


def test_captured_at_volume_ranks_by_max_probability_with_stable_ties() -> None:
    y = _y([(0, 0), (1, 0), (0, 1), (1, 0)])
    proba = np.zeros((4, 6))
    proba[:, 0] = [0.9, 0.5, 0.1, 0.5]
    proba[:, 3] = [0.0, 0.0, 0.8, 0.0]
    # Order: row 0 (0.9), row 2 (0.8), then rows 1 and 3 tie at 0.5 -> row 1 first.
    out = CMP.captured_at_volume(proba, y, WEIGHTS, 5.0, 3)
    assert out == {"k": 3, "positives": 2, "high_risk": 1, "harm": pytest.approx(11.0)}


def test_equal_input_loads_invert_the_reference_queue_fraction() -> None:
    assert CMP.equal_input_loads(0.1, [60.0, 120.0]) == pytest.approx([600.0, 1200.0])
