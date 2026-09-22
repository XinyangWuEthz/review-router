#!/usr/bin/env python
"""Prove individual regression gates accept controls and reject deliberate mutations.

A synthetic pipeline run supplies the report structure. Controlled fixtures
isolate provenance and capacity checks; they are not real-data performance
results. Unrelated model failures, skipped tests and collection errors do not
count as catching a mutation.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def check_gate(
    report: Path,
    test: str,
    *,
    config: Path,
    should_fail: bool,
    extra_env: dict[str, str] | None = None,
) -> bool:
    env = dict(os.environ)
    for key in ("REVIEW_ROUTER_EVAL_REQUIRE_CLEAN", "REVIEW_ROUTER_EVAL_REQUIRE_REAL"):
        env.pop(key, None)
    env.update({
        "REVIEW_ROUTER_EVAL_REPORT": str(report),
        "REVIEW_ROUTER_EVAL_CONFIG": str(config),
        **(extra_env or {}),
    })
    with tempfile.TemporaryDirectory(prefix="gate-result-") as temporary:
        junit = Path(temporary) / "result.xml"
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", f"tests/test_gate.py::{test}",
             "-p", "no:cacheprovider", f"--junitxml={junit}"],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        cases = ET.parse(junit).getroot().findall(".//testcase") if junit.is_file() else []
    correct = (
        result.returncode == (1 if should_fail else 0)
        and len(cases) == 1
        and cases[0].find("error") is None
        and cases[0].find("skipped") is None
        and (cases[0].find("failure") is not None) == should_fail
    )
    if not correct:
        print(f"Unexpected gate result: {test}, expected failure={should_fail}")
        print(result.stdout)
        print(result.stderr)
    return correct


def _read(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="negative-check-") as temporary:
        work = Path(temporary)
        corpus = work / "corpus"
        subprocess.run(
            [sys.executable, "scripts/make_synthetic_corpus.py", "--out", str(corpus)],
            cwd=ROOT, check=True, capture_output=True,
        )
        config = work / "smoke.yaml"
        settings = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text())
        settings["data_dir"] = str(corpus)
        settings["output_dir"] = str(work / "reports")
        config.write_text(yaml.safe_dump(settings))
        completed = subprocess.run(
            [sys.executable, "scripts/run_pipeline.py", "--config", str(config)],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        run_dir = Path(completed.stdout.splitlines()[0])
        outcomes: dict[str, bool] = {}

        def exercise(
            name: str, test: str, control: Path, mutation: Path,
            extra_env: dict[str, str] | None = None,
        ) -> None:
            control_ok = check_gate(
                control, test, config=config, should_fail=False, extra_env=extra_env
            )
            mutation_ok = check_gate(
                mutation, test, config=config, should_fail=True, extra_env=extra_env
            )
            outcomes[name] = control_ok and mutation_ok

        exercise(
            "missing report", "test_all_flagged_comments_require_human_capacity",
            run_dir / "report.json", run_dir / "does-not-exist.json",
        )
        tampered = work / "policy-mismatch"
        shutil.copytree(run_dir, tampered)
        manifest = _read(tampered / "manifest.json")
        manifest["policy_sha256"] = "0" * 64
        _write(tampered / "manifest.json", manifest)
        exercise(
            "policy mismatch", "test_report_was_made_with_the_current_policy_and_config",
            run_dir / "report.json", tampered / "report.json",
        )

        # This control isolates the completion floor from the other capacity
        # checks. Its stipulated metrics are a fixture, not a benchmark result.
        capacity = work / "capacity-control"
        shutil.copytree(run_dir, capacity)
        report = _read(capacity / "report.json")
        sim = report["simulation"]
        sim["completion_ratio"], sim["backlog_end"] = 1.0, 0
        primary = sim["primary"]
        key = f"{primary['strategy']}@{primary['load_per_hour']:g}"
        sim["time_metrics"][key]["pooled_over_seeds"]["wait"]["p50"] = 0.0
        _write(capacity / "report.json", report)
        below = work / "capacity-mutation"
        shutil.copytree(capacity, below)
        report["simulation"]["completion_ratio"] = 0.0
        _write(below / "report.json", report)
        exercise(
            "metric below floor", "test_queue_clears_at_the_primary_load",
            capacity / "report.json", below / "report.json",
        )

        # A stipulated clean manifest allows this check to run while a
        # developer has local edits. Only the commit identity is then mutated.
        current = work / "commit-control"
        shutil.copytree(run_dir, current)
        manifest = _read(current / "manifest.json")
        manifest["git_dirty"] = False
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        _write(current / "manifest.json", manifest)
        foreign = work / "foreign-commit"
        shutil.copytree(current, foreign)
        manifest["git_commit"] = "0" * 40
        _write(foreign / "manifest.json", manifest)
        exercise(
            "foreign commit", "test_report_comes_from_a_clean_checkout_when_required",
            current / "report.json", foreign / "report.json",
            {"REVIEW_ROUTER_EVAL_REQUIRE_CLEAN": "1"},
        )
        for name, caught in outcomes.items():
            print(f"{'CONTROL PASSED / MUTATION BLOCKED' if caught else 'FAILED'}  {name}")
        return 0 if all(outcomes.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
