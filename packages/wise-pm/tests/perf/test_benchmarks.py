"""Performance gate: ``benchmarks/bench_synthetic.py`` at full scale.

The benchmark (250 000 cases / 1 600 000 events) runs in a subprocess and
prints one JSON report; the tests here assert the step budgets recorded in
``benchmarks/README.md``. They are skipped unless ``WISE_RUN_BENCH=1`` is
set, so the ordinary suite stays fast::

    WISE_RUN_BENCH=1 python -m pytest -q tests/perf
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_SCRIPT = REPO_ROOT / "benchmarks" / "bench_synthetic.py"
ENABLED = os.environ.get("WISE_RUN_BENCH") == "1"

#: Upper bounds in seconds per step (see benchmarks/README.md for the 0.1.0 baseline).
#: Multiplier for slower machines (hosted CI runners set WISE_BENCH_BUDGET_FACTOR=2.0); budgets below are laptop values.
BUDGET_FACTOR = float(os.environ.get("WISE_BENCH_BUDGET_FACTOR", "1.0"))
STEP_BUDGETS_S: dict[str, float] = {"build": 3.0 * BUDGET_FACTOR, "score": 3.0 * BUDGET_FACTOR, "prioritize": 0.5 * BUDGET_FACTOR}
PEAK_BUDGET_GB = 1.5
FULL_CASES = 250_000
FULL_EVENTS = 1_600_000

pytestmark = [
    pytest.mark.bench,
    pytest.mark.skipif(not ENABLED, reason="benchmark; set WISE_RUN_BENCH=1 to run it"),
]


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    """Run the benchmark once at ``--scale 1`` and parse its JSON report."""
    proc = subprocess.run(
        [sys.executable, "-W", "error", str(BENCH_SCRIPT), "--scale", "1"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert proc.returncode == 0, f"benchmark failed (exit {proc.returncode}):\n{proc.stderr}"
    parsed: dict[str, Any] = json.loads(proc.stdout)
    return parsed


def test_runs_at_full_size(report: dict[str, Any]) -> None:
    assert report["n_cases"] == FULL_CASES
    assert report["n_events"] == FULL_EVENTS
    assert report["scale"] == 1
    assert set(report["versions"]) == {"python", "numpy", "pandas", "wise"}


@pytest.mark.parametrize(("step", "budget"), sorted(STEP_BUDGETS_S.items()))
def test_step_within_budget(report: dict[str, Any], step: str, budget: float) -> None:
    seconds = report[step]
    assert seconds < budget, f"{step} took {seconds:.3f} s, budget {budget} s (report: {json.dumps(report)})"


def test_peak_memory_within_budget(report: dict[str, Any]) -> None:
    peak = report["peak_gb"]
    assert peak is not None, "run without --no-tracemalloc"
    assert peak < PEAK_BUDGET_GB, f"tracemalloc peak {peak:.3f} GB, budget {PEAK_BUDGET_GB} GB"


def test_scores_are_consistent(report: dict[str, Any]) -> None:
    """The timed run must have produced a sound result, not just a fast one."""
    assert report["decomposition_error"] < 1e-9
    assert report["scored_share"] > 0.99
