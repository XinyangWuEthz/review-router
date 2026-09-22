from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sklearn", reason="ML extras not installed")
pytest.importorskip("pandas", reason="ML extras not installed")


@pytest.fixture(scope="session")
def synthetic_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from review_router.synthetic import write_corpus

    return write_corpus(
        tmp_path_factory.mktemp("corpus"), n_train=2500, n_test=1200, n_unscored=200, seed=11
    )
