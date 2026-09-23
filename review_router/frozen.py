"""Load the project's pinned scorer without fitting vocabulary, heads or calibration.

Only restore artifacts from the trusted project run named in the checked-in lock.
The hashes are checked before unpickling; a missing artifact never triggers training.
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from review_router.data import LABELS, Corpus, file_sha256
from review_router.model import ModelConfig, TfidfLogitModel

FILES = ("model.pkl", "manifest.json", "splits.csv")


def read_lock(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("version") != 1 or set(lock.get("files", {})) != set(FILES):
        raise ValueError("unsupported frozen model lock")
    return dict(lock)


def _verify_files(directory: Path, lock: dict[str, Any]) -> None:
    for name in FILES:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(
                f"frozen model artifact missing: {path}; restore it with "
                "scripts/restore_frozen_model.py --source-run <saved-run>; "
                "training is never an automatic fallback"
            )
        if file_sha256(path) != lock["files"][name]:
            raise ValueError(f"frozen model artifact hash mismatch: {name}")


def restore_frozen_model(source_run: Path, lock_path: Path) -> Path:
    """Copy verified artifacts to the lock's local model directory, without unpickling."""
    lock = read_lock(lock_path)
    _verify_files(source_run, lock)
    destination = (lock_path.parent / str(lock["artifact_dir"])).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = destination / name
        if target.exists() and file_sha256(target) != lock["files"][name]:
            raise ValueError(f"refusing to replace a different frozen artifact: {target}")
    for name in FILES:
        target = destination / name
        if not target.exists():
            shutil.copyfile(source_run / name, target)
    return destination


def load_frozen_model(
    lock_path: Path,
    corpus: Corpus,
    model_config: ModelConfig,
    seed: int,
    split_fractions: dict[str, float],
) -> tuple[TfidfLogitModel, Path, dict[str, Any]]:
    import pandas as pd

    lock = read_lock(lock_path)
    directory = (lock_path.parent / str(lock["artifact_dir"])).resolve()
    _verify_files(directory, lock)
    source = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if file_sha256(Path(__file__).with_name("model.py")) != lock["model_code_sha256"]:
        raise ValueError("model implementation differs from the frozen scorer")
    if source["data_sha256"] != corpus.file_hashes:
        raise ValueError("data differs from the frozen model's corpus")
    cfg = source["config"]
    if cfg["model"] != asdict(model_config):
        raise ValueError("model config differs from the frozen model")
    if cfg["seed"] != seed or cfg["split_fractions"] != split_fractions:
        raise ValueError("seed or split fractions differ from the frozen model")
    expected = corpus.train[["id", "split"]].reset_index(drop=True)
    saved = pd.read_csv(directory / "splits.csv", dtype=str)
    if not saved.equals(expected):
        raise ValueError("row IDs or split assignments differ from the frozen model")
    for dependency in ("scikit-learn", "numpy", "scipy"):
        if version(dependency) != source["dependencies"][dependency]:
            raise ValueError(
                f"frozen model needs {dependency}=={source['dependencies'][dependency]}; "
                f"installed {version(dependency)}"
            )
    model_path = directory / "model.pkl"
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, TfidfLogitModel) or model.labels != LABELS:
        raise ValueError("frozen model type or label order is invalid")
    if model.config != model_config or set(model.heads) != set(LABELS):
        raise ValueError("frozen model configuration or classifier heads are invalid")
    if set(model.calibrators) != set(LABELS) or model.n_features == 0:
        raise ValueError("frozen model is missing fitted features or calibration")
    provenance = {
        "lock_sha256": file_sha256(lock_path),
        "model_sha256": lock["files"]["model.pkl"],
        "source_run_id": source["run_id"],
        "source_git_commit": source["git_commit"],
        "source_dependencies": source["dependencies"],
    }
    return model, model_path, provenance
