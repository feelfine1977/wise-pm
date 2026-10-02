"""The type-checking and layering configuration covers the library as it is laid out now.

Every module of the package is checked by mypy without a relaxation, and every
module is placed in the layer contract of import-linter, so a new module cannot
slip past either check. The configuration lives in the workspace ``pyproject.toml``,
which is not part of the packaged tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

import wise

PACKAGE = Path(wise.__file__).parent
CONFIG = PACKAGE.parents[3] / "pyproject.toml"

pytestmark = [
    pytest.mark.skipif(not CONFIG.exists(), reason="the workspace configuration is not part of the package"),
    pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib needs Python 3.11"),
]


def _config() -> dict[str, Any]:
    import tomllib

    with CONFIG.open("rb") as f:
        return tomllib.load(f)


def _modules() -> set[str]:
    """``wise.<name>`` for every module and subpackage directly under the package."""
    names = {p.stem for p in PACKAGE.glob("*.py") if p.stem != "__init__"}
    names |= {p.name for p in PACKAGE.iterdir() if (p / "__init__.py").exists()}
    return {f"wise.{n}" for n in names}


def test_no_module_of_the_library_has_a_type_checking_relaxation() -> None:
    relaxed = {
        module
        for override in _config()["tool"]["mypy"].get("overrides", [])
        for module in override["module"]
        if module == "wise" or module.startswith("wise.")
    }
    assert relaxed == set()


def test_the_layer_contract_places_every_module_of_the_library() -> None:
    contracts = _config()["tool"]["importlinter"]["contracts"]
    layers = next(c for c in contracts if c["type"] == "layers")["layers"]
    placed = {name.strip() for layer in layers for name in layer.split("|")}
    assert placed == _modules()
    assert layers[0] == "wise.compat"  # nothing in the library may depend on the compatibility paths
