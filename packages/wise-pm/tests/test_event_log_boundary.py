"""Modules outside the event log and its case index cross the public interface only."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pandas as pd
import pytest

import wise
import wise.derive
from wise.log import EventLog
from wise.recipes import Recipe

PACKAGE = Path(wise.__file__).parent
OWNERS = {"log.py", "case_index.py"}


def _private_members() -> set[str]:
    log = wise.running_p2p_log()
    names = {n for n in dir(EventLog) if n.startswith("_") and not n.endswith("__")}
    names |= {n for n in vars(log) if n.startswith("_")}
    return names


def _reaches(source: str, private: set[str]) -> list[tuple[int, str]]:
    """``(line, name)`` for every attribute access or ``getattr``-style lookup of a name in ``private``."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in private:
            hits.append((node.lineno, node.attr))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "setattr", "hasattr", "delattr"}
            and len(node.args) > 1
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in private
        ):
            hits.append((node.lineno, str(node.args[1].value)))
    return sorted(hits)


def test_the_scan_finds_reads_writes_and_lookups() -> None:
    private = {"_codes", "_recipe_cache"}
    source = "a = log._codes\nlog._recipe_cache['x'] = 1\ngetattr(log, '_codes')\nlog.case_index.case_codes\n"
    assert _reaches(source, private) == [(1, "_codes"), (2, "_recipe_cache"), (3, "_codes")]


@pytest.mark.spec("decision:0001")
def test_no_module_outside_the_event_log_touches_its_private_members() -> None:
    private = _private_members()
    assert {"_codes", "_ts", "_valid", "_series", "_to_ts", "_index"} <= private
    offenders = {
        path.name: hits
        for path in sorted(PACKAGE.rglob("*.py"))
        if path.name not in OWNERS and (hits := _reaches(path.read_text(encoding="utf-8"), private))
    }
    assert offenders == {}


RECIPE = {"name": "n_gr", "kind": "count", "activities": ["Record Goods Receipt"]}


@pytest.fixture
def computations(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []
    compute = wise.derive.compute_recipe

    def counting(log: EventLog, recipe: Recipe) -> pd.Series:
        seen.append(recipe.name)
        return compute(log, recipe)

    monkeypatch.setattr(wise.derive, "compute_recipe", counting)
    return seen


@pytest.mark.spec("IV-C.norm")
def test_an_unchanged_recipe_is_not_computed_again(computations: list[str]) -> None:
    log = wise.running_p2p_log()
    log.derive([RECIPE], overwrite=False)
    log.derive([RECIPE], overwrite=False)
    log.derive([dict(RECIPE)], overwrite=False)  # an equal recipe is the same recipe
    assert computations == ["n_gr"]
    assert log.cases.loc["B", "n_gr"] == 4


@pytest.mark.spec("IV-C.norm")
def test_a_changed_recipe_another_log_or_overwrite_computes_again(computations: list[str]) -> None:
    log, other = wise.running_p2p_log(), wise.running_p2p_log()
    log.derive([RECIPE], overwrite=False)
    other.derive([RECIPE], overwrite=False)  # each log keeps its own record
    log.derive([RECIPE])  # overwrite is the default
    log.derive([{**RECIPE, "activities": ["Clear Invoice"]}], overwrite=False)
    assert computations == ["n_gr"] * 4
    assert log.cases.loc["B", "n_gr"] == 1


@pytest.mark.spec("IV-C.norm")
def test_scoring_twice_reuses_the_derived_attributes(computations: list[str]) -> None:
    log = wise.running_p2p_log()
    norm = dataclasses.replace(wise.running_p2p_norm(), derived_attributes=(RECIPE,))
    wise.score(log, norm)
    wise.score(log, norm)
    assert computations == ["n_gr"]
