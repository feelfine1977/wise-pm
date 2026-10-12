"""The interface named after pm4py is deprecated, and nothing else needs pm4py.

``EventLog.from_xes``, ``from_pm4py`` and ``to_pm4py`` (with the optional extra
``pm4py``) stay in 0.2.0 because the 0.1.0 compatibility record pins them; each use
warns that 0.3.0 removes it. pm4py is imported only inside the first two, so the
package imports, scores and ranks without it, and the suite runs in an environment
without it: the workspace's development group does not install it, and the tests
of these methods use a stand-in module.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import textwrap
import types
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import wise
from wise.compat import PATHS

PACKAGE = Path(wise.__file__).resolve().parent
TESTS = Path(__file__).resolve().parent
WORKSPACE = TESTS.parents[2] / "pyproject.toml"

CASE, ACTIVITY, TIME = "case:concept:name", "concept:name", "time:timestamp"


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            CASE: ["a", "a", "b"],
            ACTIVITY: ["x", "y", "x"],
            TIME: pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )


def _pm4py(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stand-in pm4py module with the two functions the deprecated methods call."""
    fake = types.ModuleType("pm4py")
    fake.read_xes = lambda path: _frame()  # type: ignore[attr-defined]
    fake.convert_to_dataframe = lambda obj: _frame()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pm4py", fake)


def _warned_once(call: Callable[[], Any], name: str) -> Any:
    """``call()``'s value; it must emit exactly one warning, a DeprecationWarning naming 0.3.0 and the replacement."""
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        value = call()
    messages = [(w.category, str(w.message)) for w in seen]
    assert len(messages) == 1, messages
    category, message = messages[0]
    assert category is DeprecationWarning
    assert message.startswith(f"EventLog.{name} is deprecated and will be removed in 0.3.0; ")
    assert message.endswith(PATHS[f"wise.log.EventLog.{name}"].replacement)
    return value


def _from_xes(monkeypatch: pytest.MonkeyPatch) -> None:
    _pm4py(monkeypatch)
    log = _warned_once(lambda: wise.EventLog.from_xes("log.xes"), "from_xes")
    assert log.fingerprint() == wise.EventLog(_frame()).fingerprint()


def _from_pm4py(monkeypatch: pytest.MonkeyPatch) -> None:
    log = _warned_once(lambda: wise.EventLog.from_pm4py(_frame()), "from_pm4py")
    assert log.fingerprint() == wise.EventLog(_frame()).fingerprint()
    _pm4py(monkeypatch)
    log = _warned_once(lambda: wise.EventLog.from_pm4py(object()), "from_pm4py")
    assert log.fingerprint() == wise.EventLog(_frame()).fingerprint()


def _to_pm4py(monkeypatch: pytest.MonkeyPatch) -> None:
    own = _frame().rename(columns={CASE: "case", ACTIVITY: "activity", TIME: "time"})
    log = wise.EventLog(own, case_col="case", activity_col="activity", timestamp_col="time")
    frame = _warned_once(log.to_pm4py, "to_pm4py")
    assert frame[CASE].tolist() == ["a", "a", "b"]
    assert frame[ACTIVITY].tolist() == ["x", "y", "x"]
    assert frame[TIME].tolist() == list(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    renamed = log.events.rename(columns={log.case_col: CASE, log.activity_col: ACTIVITY, log.timestamp_col: TIME})
    pd.testing.assert_frame_equal(frame, renamed, check_exact=True)  # the replacement the warning names


DRIVERS: dict[str, Callable[[pytest.MonkeyPatch], None]] = {
    "from_xes": _from_xes,
    "from_pm4py": _from_pm4py,
    "to_pm4py": _to_pm4py,
}


@pytest.mark.spec("decision:0001/2")
@pytest.mark.parametrize("name", sorted(DRIVERS))
def test_each_method_named_after_pm4py_warns_once_naming_0_3_0_and_returns_what_it_returned(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    DRIVERS[name](monkeypatch)


@pytest.mark.spec("decision:0001/2")
def test_the_compatibility_record_lists_them_as_withdrawn_in_0_3_0() -> None:
    withdrawn = {name: path for name, path in PATHS.items() if path.status == "withdrawn"}
    assert set(withdrawn) == {f"wise.log.EventLog.{name}" for name in DRIVERS}
    assert {path.removed_in for path in withdrawn.values()} == {"0.3.0"}
    assert {path.removed_in for path in PATHS.values() if path.status != "withdrawn"} == {"1.0"}


def _imports_of_pm4py(path: Path) -> list[tuple[str, ...]]:
    """Each import of pm4py in ``path``, as the names of the functions and classes around it."""
    found: list[tuple[str, ...]] = []

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if (isinstance(child, ast.Import) and any(a.name.split(".")[0] == "pm4py" for a in child.names)) or (
                isinstance(child, ast.ImportFrom) and (child.module or "").split(".")[0] == "pm4py"
            ):
                found.append(scope)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child, (*scope, child.name))
            else:
                visit(child, scope)

    visit(ast.parse(path.read_text(encoding="utf-8")), ())
    return found


def test_pm4py_is_imported_only_inside_the_two_deprecated_readers() -> None:
    imports = {
        f"{path.relative_to(root.parent)}:{'.'.join(scope)}"
        for root in (PACKAGE, TESTS)
        for path in sorted(root.rglob("*.py"))
        for scope in _imports_of_pm4py(path)
    }
    assert imports == {"wise/log.py:EventLog.from_pm4py", "wise/log.py:EventLog.from_xes"}


def test_the_package_imports_scores_and_ranks_with_pm4py_unavailable() -> None:
    script = textwrap.dedent(
        """
        import importlib, pkgutil, sys
        sys.modules["pm4py"] = None  # any import of pm4py now fails
        import wise
        for module in pkgutil.walk_packages(wise.__path__, "wise."):
            importlib.import_module(module.name)
        result = wise.score(wise.running_p2p_log(), wise.running_p2p_norm())
        backlog = wise.prioritize(result, by="company", view="Finance")
        assert len(backlog.table) > 0
        try:
            wise.EventLog.from_xes("log.xes")
        except ImportError as exc:
            assert "wise-pm[pm4py]" in str(exc), exc
        else:
            raise AssertionError("from_xes read a file without pm4py")
        assert sys.modules["pm4py"] is None
        print("ok")
        """
    )
    run = subprocess.run(
        [sys.executable, "-W", "ignore::DeprecationWarning", "-c", script], capture_output=True, text=True, check=False
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "ok"


@pytest.mark.skipif(not WORKSPACE.exists(), reason="the workspace configuration is not part of the package")
def test_the_environment_the_suite_runs_in_does_not_install_pm4py() -> None:
    # CI and the release workflow sync the workspace with its development group only
    groups = re.search(r"^\[dependency-groups\]\n(.*?)(?=^\[)", WORKSPACE.read_text(encoding="utf-8"), flags=re.M | re.S)
    assert groups is not None
    assert "wise-pm[stats]" in groups.group(1)
    assert "pm4py" not in groups.group(1)
