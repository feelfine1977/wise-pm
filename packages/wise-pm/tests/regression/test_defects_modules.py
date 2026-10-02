"""Defect ledger C21: modules of the package reach into each other's private names."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import wise

PACKAGE = Path(wise.__file__).parent


def _private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _imported_modules(tree: ast.Module) -> set[str]:
    """Names bound to a module of the package by ``import wise.x as m`` or ``from . import x``."""
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bound |= {a.asname or a.name.split(".")[0] for a in node.names if a.name.split(".")[0] == "wise"}
        elif isinstance(node, ast.ImportFrom) and (node.level > 0 or (node.module or "").split(".")[0] == "wise"):
            bound |= {a.asname or a.name for a in node.names}
    return bound


def _private_reaches(source: str) -> list[tuple[int, str]]:
    """``(line, name)`` for each underscore name imported from, or read off, another module of the package.

    Underscore modules (``_json``, ``_version``) are internal to the package; importing their public
    names is crossing their interface.
    """
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.level > 0 or (node.module or "").split(".")[0] == "wise"):
            hits += [(node.lineno, a.name) for a in node.names if _private(a.name)]
    modules = _imported_modules(tree)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in modules
            and _private(node.attr)
        ):
            hits.append((node.lineno, node.attr))
    return sorted(hits)


def test_the_scan_finds_private_imports_and_attribute_reads() -> None:
    source = (
        "from .prioritization import _frame, prioritize\n"
        "from ..constraints import _UNITS as units\n"
        "from . import prioritization\n"
        "x = prioritization._keys('a')\n"
        "from ._version import __version__\n"
        "from ._json import jsonable as _jsonable\n"
        "import wise._json\n"
        "from pandas import _libs\n"
        "y = self._own\n"
    )
    assert _private_reaches(source) == [(1, "_frame"), (2, "_UNITS"), (4, "_keys")]


@pytest.mark.regression
@pytest.mark.spec("decision:0001/1")
# C21 closed 2026-10-02: modules cross each other's public interface only (decision record 0001, one owner per computation)
def test_no_module_imports_a_private_name_from_another_module() -> None:
    offenders = {
        str(path.relative_to(PACKAGE)): hits
        for path in sorted(PACKAGE.rglob("*.py"))
        if (hits := _private_reaches(path.read_text(encoding="utf-8")))
    }
    assert offenders == {}
