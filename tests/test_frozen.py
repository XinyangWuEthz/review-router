from __future__ import annotations

import json
import stat
import subprocess
import sys
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from review_router import frozen
from review_router.data import file_sha256, load_corpus
from review_router.model import TfidfLogitModel
from review_router.pipeline import load_config, run

ROOT = Path(__file__).resolve().parent.parent


def _write_config(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory: pytest.TempPathFactory, synthetic_corpus: Path) -> Path:
    directory = tmp_path_factory.mktemp("frozen-source")
    config = _write_config(
        directory / "train.yaml",
        {
            "label": "frozen-test",
            "data_dir": str(synthetic_corpus),
            "output_dir": str(directory / "reports"),
            "seed": 3,
            "split": {"train": 0.6, "calib": 0.2, "thresh": 0.2},
            "model": {"max_features": 5000, "ngram_max": 1, "min_df": 1, "C": 2.0},
            "simulation": {
                "reviewers": 2,
                "handle_minutes": 2.0,
                "horizon_hours": 1,
                "loads_per_hour": [90],
                "seeds": [1],
                "high_risk_min_weight": 5,
                "primary": {"load_per_hour": 90, "strategy": "severity"},
            },
        },
    )
    return run(config)


@pytest.fixture
def frozen_case(tmp_path: Path, trained_run: Path) -> tuple[Path, Path, Path]:
    lock_path = tmp_path / "frozen-model.json"
    lock_path.write_text(
        json.dumps(
            {
                "version": 1,
                "artifact_dir": "artifacts/pinned-model",
                "model_code_sha256": file_sha256(ROOT / "review_router" / "model.py"),
                "files": {name: file_sha256(trained_run / name) for name in frozen.FILES},
            }
        )
    )
    destination = frozen.restore_frozen_model(trained_run, lock_path)
    cfg = yaml.safe_load((trained_run / "config.yaml").read_text())
    cfg.update(frozen_model=str(lock_path), output_dir=str(tmp_path / "evaluations"))
    config_path = _write_config(tmp_path / "evaluate.yaml", cfg)
    return config_path, lock_path, destination


def _forbid_training(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        pytest.fail("frozen evaluation must not train or recalibrate")

    monkeypatch.setattr(TfidfLogitModel, "fit", fail)
    monkeypatch.setattr(TfidfLogitModel, "calibrate", fail)


@pytest.mark.parametrize("absolute_directory", [False, True])
def test_restored_model_reproduces_router_outputs_without_training(
    trained_run: Path,
    frozen_case: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    absolute_directory: bool,
) -> None:
    config_path, lock_path, destination = frozen_case
    if absolute_directory:
        lock = json.loads(lock_path.read_text())
        lock["artifact_dir"] = str(destination)
        lock_path.write_text(json.dumps(lock))
    # Restore is idempotent for the pinned bytes, with either path representation.
    assert frozen.restore_frozen_model(trained_run, lock_path) == destination
    _forbid_training(monkeypatch)
    replay = run(config_path)
    for name in (
        "model.pkl",
        "splits.csv",
        "thresholds.json",
        "predictions.csv",
        "simulation_jobs.csv",
    ):
        assert (replay / name).read_bytes() == (trained_run / name).read_bytes(), name
    assert (replay / "frozen-model.json").read_bytes() == lock_path.read_bytes()
    manifest = json.loads((replay / "manifest.json").read_text())
    source = json.loads((trained_run / "manifest.json").read_text())
    assert manifest["execution_mode"] == "frozen_model"
    assert manifest["model_source"] == {
        "lock_sha256": file_sha256(lock_path),
        "model_sha256": file_sha256(trained_run / "model.pkl"),
        "source_run_id": source["run_id"],
        "source_git_commit": source["git_commit"],
        "source_dependencies": source["dependencies"],
    }
    original_report = json.loads((trained_run / "report.json").read_text())
    replay_report = json.loads((replay / "report.json").read_text())
    for key in ("threshold_selection", "agreement_by_confidence", "simulation", "per_label"):
        assert replay_report[key] == original_report[key], key


def test_frozen_development_run_only_predicts_calibration_and_selection_rows(
    frozen_case: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _, _ = frozen_case
    raw = yaml.safe_load(config_path.read_text())
    raw["evaluate_test"] = False
    _write_config(config_path, raw)
    config = load_config(config_path)
    corpus = load_corpus(config.data_dir, config.split_fractions, config.seed)
    expected = [
        corpus.train.loc[corpus.train["split"] == split, "comment_text"].tolist()
        for split in ("calib", "thresh")
    ]
    calls: list[list[str]] = []
    original_predict = TfidfLogitModel.predict_proba

    def predict(self: TfidfLogitModel, texts: Iterable[str]) -> np.ndarray:
        batch = list(texts)
        calls.append(batch)
        return original_predict(self, batch)

    _forbid_training(monkeypatch)
    monkeypatch.setattr(TfidfLogitModel, "predict_proba", predict)
    result = run(config_path)
    assert calls == expected
    assert not (result / "predictions.csv").exists()
    assert not (result / "simulation_jobs.csv").exists()
    report = json.loads((result / "report.json").read_text())
    assert report["evaluate_test"] is False
    assert "test" not in report["agreement_by_confidence"]


@pytest.mark.parametrize(
    "mismatch, message",
    [
        ("data", "data differs"),
        ("model_config", "model config differs"),
        ("seed", "seed or split fractions differ"),
        ("split_fractions", "seed or split fractions differ"),
        ("split_rows", "row IDs or split assignments differ"),
        ("model_code", "model implementation differs"),
        ("artifact_hash", "artifact hash mismatch"),
        ("lock_version", "unsupported frozen model lock"),
        ("dependency", "frozen model needs numpy=="),
        ("missing", "artifact missing"),
    ],
)
def test_incompatible_frozen_inputs_fail_before_unpickling_or_creating_a_run(
    frozen_case: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
    message: str,
) -> None:
    config_path, lock_path, destination = frozen_case
    config = yaml.safe_load(config_path.read_text())
    lock = json.loads(lock_path.read_text())
    if mismatch == "data":
        manifest_path = destination / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["data_sha256"]["train"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
        lock["files"]["manifest.json"] = file_sha256(manifest_path)
    elif mismatch == "model_config":
        config["model"]["C"] = 3.0
    elif mismatch == "seed":
        config["seed"] += 1
    elif mismatch == "split_fractions":
        config["split"] = {"train": 0.5, "calib": 0.3, "thresh": 0.2}
    elif mismatch == "split_rows":
        splits_path = destination / "splits.csv"
        splits = pd.read_csv(splits_path, dtype=str)
        splits.iloc[::-1].to_csv(splits_path, index=False)
        lock["files"]["splits.csv"] = file_sha256(splits_path)
    elif mismatch == "model_code":
        lock["model_code_sha256"] = "0" * 64
    elif mismatch == "artifact_hash":
        (destination / "model.pkl").write_bytes(b"untrusted pickle")
    elif mismatch == "lock_version":
        lock["version"] = 99
    elif mismatch == "dependency":
        original_version = frozen.version
        monkeypatch.setattr(
            frozen, "version", lambda name: "0.0.0" if name == "numpy" else original_version(name)
        )
    elif mismatch == "missing":
        (destination / "model.pkl").unlink()
    _write_config(config_path, config)
    lock_path.write_text(json.dumps(lock))
    _forbid_training(monkeypatch)

    def never_unpickle(*args: Any, **kwargs: Any) -> None:
        pytest.fail("incompatible artifacts must be rejected before unpickling")

    monkeypatch.setattr(frozen.pickle, "load", never_unpickle)
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        run(config_path)
    assert not load_config(config_path).output_dir.exists()


def test_restore_rejects_corrupt_source_and_preserves_existing_pin(
    trained_run: Path, frozen_case: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    _, lock_path, destination = frozen_case
    pinned = {name: (destination / name).read_bytes() for name in frozen.FILES}
    source = tmp_path / "corrupt-source"
    source.mkdir()
    for name, contents in pinned.items():
        (source / name).write_bytes(contents)
    (source / "model.pkl").write_bytes(b"changed model")
    with pytest.raises(ValueError, match="artifact hash mismatch: model.pkl"):
        frozen.restore_frozen_model(source, lock_path)
    assert {name: (destination / name).read_bytes() for name in frozen.FILES} == pinned
    (destination / "model.pkl").write_bytes(b"different existing model")
    with pytest.raises(ValueError, match="refusing to replace a different frozen artifact"):
        frozen.restore_frozen_model(trained_run, lock_path)
    assert (destination / "model.pkl").read_bytes() == b"different existing model"


@pytest.fixture
def archive_case(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "archive-source"
    source.mkdir()
    for name in frozen.FILES:
        (source / name).write_bytes(f"pinned {name}".encode())
    lock_path = tmp_path / "archive-lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "version": 1,
                "artifact_dir": "restored",
                "files": {name: file_sha256(source / name) for name in frozen.FILES},
            }
        )
    )
    archive = tmp_path / "frozen-baseline.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in frozen.FILES:
            bundle.write(source / name, arcname=name)
    return archive, lock_path, tmp_path / "restored"


def test_archive_restore_verifies_without_unpickling_and_is_idempotent(
    archive_case: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, lock_path, destination = archive_case

    def never_unpickle(*args: Any, **kwargs: Any) -> None:
        pytest.fail("restoring the archive must not unpickle its contents")

    monkeypatch.setattr(frozen.pickle, "load", never_unpickle)
    assert not destination.exists()
    assert frozen.restore_frozen_archive(archive, lock_path) == destination
    assert frozen.restore_frozen_archive(archive, lock_path) == destination
    lock = json.loads(lock_path.read_text())
    assert {name: file_sha256(destination / name) for name in frozen.FILES} == lock["files"]


@pytest.mark.parametrize(
    "mutation", ["missing", "unexpected", "nested", "traversal", "absolute", "symlink", "duplicate"]
)
def test_archive_rejects_invalid_members_before_installing(
    archive_case: tuple[Path, Path, Path], mutation: str
) -> None:
    archive, lock_path, destination = archive_case
    with zipfile.ZipFile(archive) as bundle:
        contents = {name: bundle.read(name) for name in frozen.FILES}
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, content in contents.items():
            if name == "model.pkl":
                if mutation == "missing":
                    continue
                if mutation in ("nested", "traversal", "absolute"):
                    name = {
                        "nested": "folder/model.pkl",
                        "traversal": "../model.pkl",
                        "absolute": "/model.pkl",
                    }[mutation]
                elif mutation == "symlink":
                    info = zipfile.ZipInfo(name)
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                    bundle.writestr(info, b"../outside-model")
                    continue
            bundle.writestr(name, content)
        if mutation == "unexpected":
            bundle.writestr("extra.txt", b"extra")
        elif mutation == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                bundle.writestr("model.pkl", contents["model.pkl"])
    with pytest.raises(ValueError, match="frozen archive"):
        frozen.restore_frozen_archive(archive, lock_path)
    assert not destination.exists()
    assert not (archive.parent / "model.pkl").exists()


def test_archive_rejects_corruption_and_preserves_existing_files(
    archive_case: tuple[Path, Path, Path],
) -> None:
    archive, lock_path, destination = archive_case
    frozen.restore_frozen_archive(archive, lock_path)
    pinned = {name: (destination / name).read_bytes() for name in frozen.FILES}
    corrupt = archive.with_name("corrupt.zip")
    with zipfile.ZipFile(corrupt, "w") as bundle:
        for name, content in pinned.items():
            bundle.writestr(name, b"changed model" if name == "model.pkl" else content)
    with pytest.raises(ValueError, match="artifact hash mismatch: model.pkl"):
        frozen.restore_frozen_archive(corrupt, lock_path)
    assert {name: (destination / name).read_bytes() for name in frozen.FILES} == pinned
    (destination / "model.pkl").write_bytes(b"different existing model")
    with pytest.raises(ValueError, match="refusing to replace a different frozen artifact"):
        frozen.restore_frozen_archive(archive, lock_path)
    assert (destination / "model.pkl").read_bytes() == b"different existing model"


def test_restore_cli_accepts_archive_and_rejects_two_sources(
    archive_case: tuple[Path, Path, Path],
) -> None:
    archive, lock_path, destination = archive_case
    command = [
        sys.executable,
        str(ROOT / "scripts/restore_frozen_model.py"),
        "--archive",
        str(archive),
        "--lock",
        str(lock_path),
    ]
    restored = subprocess.run(command, check=True, capture_output=True, text=True)
    assert restored.stdout.strip() == str(destination)
    assert set(path.name for path in destination.iterdir()) == set(frozen.FILES)
    rejected = subprocess.run(
        [*command, "--source-run", str(destination)], check=False, capture_output=True, text=True
    )
    assert rejected.returncode == 2
    assert "not allowed with argument" in rejected.stderr


def test_explicit_retrain_bypasses_lock_without_replacing_pinned_artifacts(
    frozen_case: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, lock_path, destination = frozen_case
    pinned = {name: (destination / name).read_bytes() for name in frozen.FILES}
    # A new training experiment can run even when the old lock cannot be loaded.
    lock_path.write_text("invalid lock, deliberately unused by --retrain")
    calls: list[str] = []
    original_fit = TfidfLogitModel.fit
    original_calibrate = TfidfLogitModel.calibrate

    def fit(
        self: TfidfLogitModel, texts: Iterable[str], y: np.ndarray, seed: int
    ) -> TfidfLogitModel:
        calls.append("fit")
        return original_fit(self, texts, y, seed)

    def calibrate(self: TfidfLogitModel, texts: Iterable[str], y: np.ndarray) -> TfidfLogitModel:
        calls.append("calibrate")
        return original_calibrate(self, texts, y)

    monkeypatch.setattr(TfidfLogitModel, "fit", fit)
    monkeypatch.setattr(TfidfLogitModel, "calibrate", calibrate)
    result = run(config_path, retrain=True)
    assert calls == ["fit", "calibrate"]
    assert {name: (destination / name).read_bytes() for name in frozen.FILES} == pinned
    assert (result / "model.pkl").is_file()
    assert not (result / "frozen-model.json").exists()
    manifest = json.loads((result / "manifest.json").read_text())
    assert manifest["execution_mode"] == "train"
    assert manifest["explicit_retrain"] is True
    assert "model_source" not in manifest
