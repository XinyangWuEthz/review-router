#!/usr/bin/env python
"""Controlled failures: tamper with a real run in three ways; the gates must fail each time.

    python scripts/ci_negative_check.py

Uses a synthetic corpus so it needs no data access. Exit code 0 means every
tampered run was blocked; any tampered run that passed the gates is an error.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def gates(report: Path, extra_env: dict[str, str] | None = None) -> int:
    env = {**os.environ, "REVIEW_ROUTER_EVAL_REPORT": str(report), **(extra_env or {})}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_gate.py", "-p", "no:cacheprovider"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    ).returncode


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="negative-check-"))
    corpus = work / "corpus"
    subprocess.run(
        [sys.executable, "scripts/make_synthetic_corpus.py", "--out", str(corpus)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    config = work / "configs" / "smoke.yaml"
    config.parent.mkdir()
    text = (ROOT / "configs" / "smoke.yaml").read_text()
    text = text.replace("data_dir: data/synthetic", f"data_dir: {corpus}")
    text = text.replace("output_dir: reports", f"output_dir: {work / 'reports'}")
    config.write_text(text)
    run_dir = Path(
        subprocess.run(
            [sys.executable, "scripts/run_pipeline.py", "--config", str(config)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0]
    )
    outcomes: dict[str, bool] = {}

    # 1. The report the run produced is missing.
    outcomes["missing report"] = gates(run_dir / "does-not-exist.json") != 0

    # 2. The report does not match the current policy (manifest hash tampered).
    tampered = work / "tampered"
    shutil.copytree(run_dir, tampered)
    manifest = json.loads((tampered / "manifest.json").read_text())
    manifest["policy_sha256"] = "0" * 64
    (tampered / "manifest.json").write_text(json.dumps(manifest))
    outcomes["policy mismatch"] = gates(tampered / "report.json") != 0

    # 3. A metric below its floor (completion ratio collapsed).
    below = work / "below"
    shutil.copytree(run_dir, below)
    report = json.loads((below / "report.json").read_text())
    report["simulation"]["completion_ratio"] = 0.0
    (below / "report.json").write_text(json.dumps(report))
    outcomes["metric below floor"] = gates(below / "report.json") != 0

    # 4. A report from another commit when a clean checkout is required.
    foreign = work / "foreign"
    shutil.copytree(run_dir, foreign)
    manifest = json.loads((foreign / "manifest.json").read_text())
    manifest["git_dirty"] = False
    manifest["git_commit"] = "0" * 40
    (foreign / "manifest.json").write_text(json.dumps(manifest))
    outcomes["foreign commit"] = (
        gates(foreign / "report.json", {"REVIEW_ROUTER_EVAL_REQUIRE_CLEAN": "1"}) != 0
    )

    for name, blocked in outcomes.items():
        print(f"{'BLOCKED' if blocked else 'PASSED (BAD)'}  {name}")
    shutil.rmtree(work, ignore_errors=True)
    return 0 if all(outcomes.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
