from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _module(ROOT / "scripts" / "audit_auto_action_fp.py")


def _sample(human: list[str], preread: list[str] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame({"id": [str(i) for i in range(len(human))], "human_verdict": human})
    if preread is not None:
        frame["preread_verdict"] = preread
    return frame


def test_implied_precision_is_hand_computable() -> None:
    # 100 auto-action rows, 10 false positives; half of the audited ones are really toxic.
    assert AUDIT.implied_precision(100, 10, 0.5) == pytest.approx(0.95)
    assert AUDIT.implied_precision(100, 10, 1.0) == pytest.approx(1.0)
    assert AUDIT.implied_precision(100, 10, 0.0) == pytest.approx(0.90)


def test_rows_needed_for_the_floor() -> None:
    # At most 1 false positive may remain of 10, so 9 of 10 audited rows must be toxic.
    assert AUDIT.rows_needed_for_floor(100, 10, 10, 0.99) == 9
    # The round-1 case: 2125 rows allow 21 false positives of 204 -> 90 of 100.
    assert AUDIT.rows_needed_for_floor(2125, 204, 100, 0.99) == 90


def test_score_counts_both_rules_and_agreement() -> None:
    human = ["toxic"] * 5 + ["borderline"] * 3 + ["clean"] * 2
    preread = ["toxic"] * 4 + ["borderline"] * 4 + ["clean"] * 2
    out = AUDIT.score(_sample(human, preread), n_auto=100, n_fp=10)
    assert out["counts"] == {"toxic": 5, "borderline": 3, "clean": 2}
    assert out["implied_precision"]["toxic_only"]["precision"] == pytest.approx(0.95)
    assert out["implied_precision"]["toxic_or_borderline"]["precision"] == pytest.approx(0.98)
    lo, hi = out["implied_precision"]["toxic_only"]["precision_ci95"]
    assert lo < 0.95 < hi
    assert out["agreement_with_preread"] == pytest.approx(0.9)
    assert out["preread_to_human"] == {"borderline->toxic": 1}


def test_score_is_case_and_space_insensitive() -> None:
    out = AUDIT.score(_sample([" Toxic", "CLEAN ", "clean"]), n_auto=10, n_fp=3)
    assert out["counts"] == {"toxic": 1, "borderline": 0, "clean": 2}


@pytest.mark.parametrize("bad", ["", "maybe"])
def test_score_rejects_blank_or_unknown_verdicts(bad: str) -> None:
    with pytest.raises(ValueError, match="human_verdict"):
        AUDIT.score(_sample(["toxic", bad]), n_auto=10, n_fp=2)


def test_score_dir_writes_the_human_section(tmp_path: Path) -> None:
    _sample(["toxic", "clean"]).to_csv(tmp_path / "sample.csv", index=False)
    (tmp_path / "sample_meta.json").write_text(
        json.dumps({"n_auto_action": 10, "n_false_positive": 2, "human": {"protocol": "p"}})
    )
    AUDIT.score_dir(tmp_path)
    meta = json.loads((tmp_path / "sample_meta.json").read_text())
    assert meta["human"]["counts"]["toxic"] == 1
    assert meta["human"]["protocol"] == "p"
    assert meta["n_auto_action"] == 10
