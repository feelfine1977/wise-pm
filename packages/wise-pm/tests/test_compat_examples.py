"""The 0.1.0 quick start and both example scripts run unchanged apart from deprecation warnings.

Decision record 0001, section 2. Each program is run as 0.1.0 shipped it (the
examples from ``examples/``, unchanged since the ``v0.1.0`` tag; the quick start of the
0.1.0 README under ``tests/data/compat``) with every warning an error except
``DeprecationWarning`` and, for a schema version 2 norm file, the
``LegacySemanticsWarning`` of decision 0001 section 4. Its output is compared with the
output 0.1.0 produced on the same input, recorded under ``tests/data/compat`` by
running this module against the source of the tag::

    git archive v0.1.0 src | tar -x -C /tmp/v010
    PYTHONPATH=/tmp/v010/src python packages/wise-pm/tests/test_compat_examples.py --record

Printed tables are compared as text (they round to four decimals); the quick start's
frames within 1e-9 (docs/semantics/parity.md).
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))  # for --record, run as a script

from _support.legacy_inputs import write_bpic19_csv, write_readme_csv

EXAMPLES = TESTS.parent / "examples"
RECORDS = TESTS / "data" / "compat"
README_QUICKSTART = RECORDS / "readme_quickstart_0_1_0.py.txt"
#: Warning categories the programs may emit; any other warning is an error.
ALLOWED = ("DeprecationWarning", "LegacySemanticsWarning")
WARNING_LINE = re.compile(r"^\S.*:\d+: (\w+): ", re.M)


def _normalise(output: str, directory: Path) -> str:
    """The run's directory and timings replaced, trailing blanks dropped (pandas pads its index rows)."""
    output = output.replace(str(directory), "<dir>")
    output = re.sub(r"\(\d+\.\ds\)", "(<time>)", output)
    output = re.sub(r"in \d+\.\ds;", "in <time>;", output)
    return "".join(line.rstrip() + "\n" for line in output.splitlines())


def _run(script: str, directory: Path, *args: str) -> str:
    """Run ``examples/<script>`` from a copy in ``directory``; its normalised standard output."""
    for name in (script, "running_p2p_norm.json", "bpic19_norm.json"):
        shutil.copy(EXAMPLES / name, directory / name)
    # LegacySemanticsWarning is a FutureWarning; the category check below tells it from any other one
    flags = ["-W", "error", "-W", "always::DeprecationWarning", "-W", "always::FutureWarning"]
    env = {**os.environ, "PYTHONWARNINGS": ""}
    done = subprocess.run(
        [sys.executable, *flags, script, *args], cwd=directory, capture_output=True, text=True, env=env, check=False
    )
    assert done.returncode == 0, done.stderr
    assert set(WARNING_LINE.findall(done.stderr)) <= set(ALLOWED), done.stderr
    return _normalise(done.stdout, directory)


def _summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "shape": list(frame.shape),
        "columns": [str(c) for c in frame.columns],
        "nan": [int(n) for n in frame.isna().sum()],
        "sum": [float(s) for s in frame.sum(numeric_only=False)],
    }


def readme_quickstart(directory: Path) -> dict[str, Any]:
    """Run the 0.1.0 README quick start in ``directory``; what it computed, and the warnings it emitted."""
    write_readme_csv(directory / "log.csv")
    namespace: dict[str, Any] = {}
    cwd = Path.cwd()
    os.chdir(directory)
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            exec(compile(README_QUICKSTART.read_text(encoding="utf-8"), "README quick start (0.1.0)", "exec"), namespace)
    finally:
        os.chdir(cwd)
    result, backlog = namespace["result"], namespace["backlog"]
    table = backlog if isinstance(backlog, pd.DataFrame) else backlog.table
    return {
        "violations": _summary(result.violations),
        "scores": _summary(result.scores),
        "contributions_Finance": _summary(result.contributions["Finance"]),
        "backlog": json.loads(table.reset_index().to_json(orient="split", double_precision=15)),
        "norm": json.loads((directory / "norm_v1.json").read_text(encoding="utf-8")),
        "warnings": sorted({w.category.__name__ for w in seen}),
    }


def _close(actual: Any, expected: Any, path: str = "") -> None:
    if isinstance(expected, float) and isinstance(actual, float | int):
        assert math.isnan(expected) == math.isnan(actual), path
        assert math.isnan(expected) or abs(actual - expected) <= 1e-9, path
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for k in expected:
            _close(actual[k], expected[k], f"{path}.{k}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _close(a, e, f"{path}[{i}]")
    else:
        assert actual == expected, path


@pytest.mark.spec("decision:0001/2")
def test_the_0_1_0_readme_quick_start_computes_what_0_1_0_computed(tmp_path: Path) -> None:
    got = readme_quickstart(tmp_path)
    expected = json.loads((RECORDS / "readme_quickstart_0_1_0.json").read_text(encoding="utf-8"))
    assert set(got.pop("warnings")) <= {"DeprecationWarning"}
    expected.pop("warnings")
    _close(got, expected)


@pytest.mark.spec("decision:0001/2")
def test_the_quickstart_example_prints_what_0_1_0_printed(tmp_path: Path) -> None:
    printed = _run("quickstart.py", tmp_path)
    assert printed == (RECORDS / "quickstart_0_1_0.txt").read_text(encoding="utf-8")
    written = json.loads((tmp_path / "running_p2p_norm.json").read_text(encoding="utf-8"))
    assert written == json.loads((EXAMPLES / "running_p2p_norm.json").read_text(encoding="utf-8"))


@pytest.mark.spec("decision:0001/2")
def test_the_bpic19_example_prints_what_0_1_0_printed_on_a_log_in_the_challenge_layout(tmp_path: Path) -> None:
    write_bpic19_csv(tmp_path / "bpic19.csv")
    printed = _run("bpic19_evaluation.py", tmp_path, "bpic19.csv")
    assert printed == (RECORDS / "bpic19_evaluation_0_1_0.txt").read_text(encoding="utf-8")


def test_an_unexpected_warning_fails_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "warns.py"
    monkeypatch.setattr(sys.modules[__name__], "EXAMPLES", tmp_path)
    for name in ("running_p2p_norm.json", "bpic19_norm.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    script.write_text("import warnings\nwarnings.warn('x', FutureWarning)\nwarnings.warn('y', UserWarning)\n", encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(AssertionError, match="FutureWarning"):
        _run("warns.py", run)
    script.write_text("import warnings\nwarnings.warn('y', UserWarning)\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="UserWarning"):
        _run("warns.py", run)


def _record() -> None:  # pragma: no cover - run against the 0.1.0 source, see the module docstring
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw).resolve()
        (RECORDS / "quickstart_0_1_0.txt").write_text(_run("quickstart.py", directory), encoding="utf-8")
        write_bpic19_csv(directory / "bpic19.csv")
        printed = _run("bpic19_evaluation.py", directory, "bpic19.csv")
        (RECORDS / "bpic19_evaluation_0_1_0.txt").write_text(printed, encoding="utf-8")
        recorded = readme_quickstart(directory)
        (RECORDS / "readme_quickstart_0_1_0.json").write_text(json.dumps(recorded, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__" and sys.argv[1:] == ["--record"]:
    _record()
