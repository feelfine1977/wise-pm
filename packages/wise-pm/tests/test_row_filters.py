"""Recipe where-clauses and the ``where=`` filters of results, drivers and penalty mass.

Both are expressed in the selection grammar (``wise.selection``). The running example::

    case  company  vendor  events
    A     A        V1      Create PO, GR (100), IR (100), Clear Invoice
    B     A        V2      Create PO, GR (25) x4, IR (100), Clear Invoice
    C     B        V1      Create PO, GR (100), IR (82), Clear Invoice
    D     B        V1      Create PO, GR (100), IR (100), Cancel IR
    E     B        V2      Create PO, GR (100)

Expected counts are read off this table by hand; the 0.1.0 filters, frozen in
``_support.legacy_filters``, are the oracle for every form 0.1.0 accepted.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import pandas as pd
import pytest

import wise
from _support.legacy_filters import legacy_recipe_mask, legacy_where_mask
from wise.derive import compute_recipe
from wise.errors import LogSchemaError, NormError, NotScoredError
from wise.selection import LEGACY, EventTable, parse_recipe_where

CASES = ["A", "B", "C", "D", "E"]


@pytest.fixture
def log() -> wise.EventLog:
    return wise.running_p2p_log()


@pytest.fixture
def result(log: wise.EventLog) -> wise.ScoreResult:
    return wise.score(log, wise.running_p2p_norm())


def _count(log: wise.EventLog, where: Any) -> list[float]:
    out = compute_recipe(log, {"name": "n", "kind": "count_events", "where": where})
    assert out.index.tolist() == CASES
    return out.tolist()


# ------------------------------------------------------------------ recipe where-clauses
#: Every clause form of 0.1.0 with its per-case event count, hand-derived from the table above.
RECIPE_CLAUSES: list[tuple[Any, list[float]]] = [
    ({"column": "activity", "in": ["Record Goods Receipt"]}, [1, 4, 1, 1, 1]),
    ({"column": "activity", "not_in": ["Record Goods Receipt", "Create Purchase Order Item"]}, [2, 2, 2, 2, 0]),
    ({"column": "activity", "regex": "^Record"}, [2, 5, 2, 2, 1]),
    ({"column": "activity", "regex": "(Goods|Invoice) Receipt"}, [2, 5, 2, 3, 1]),
    # 0.1.0 handed the pattern to pandas as given, so a compiled pattern with flags worked
    ({"column": "activity", "regex": re.compile("record", re.IGNORECASE)}, [2, 5, 2, 2, 1]),
    ({"column": "vendor", "eq": "V2"}, [0, 7, 0, 0, 2]),
    ({"column": "amount", "eq": 100.0}, [2, 1, 1, 2, 1]),
    # amount missing on Create PO / Clear Invoice / Cancel IR: out of 'in', inside 'not_in'
    ({"column": "amount", "in": [25.0]}, [0, 4, 0, 0, 0]),
    ({"column": "amount", "not_in": [100.0]}, [2, 6, 3, 2, 1]),
    # 0.1.0 took the first of in / not_in / regex / eq and ignored any other key
    ({"column": "activity", "eq": "Clear Invoice", "in": ["Record Invoice Receipt"]}, [1, 1, 1, 1, 0]),
    ({"column": "activity", "regex": "Invoice", "note": "ignored"}, [2, 2, 2, 2, 0]),
    ({}, [4, 7, 4, 4, 2]),
    (None, [4, 7, 4, 4, 2]),
]


@pytest.mark.spec("decision:0001/1")
@pytest.mark.parametrize(("where", "expected"), RECIPE_CLAUSES, ids=[repr(c[0]) for c in RECIPE_CLAUSES])
def test_recipe_where_clauses_of_0_1_0_select_the_same_events(log: wise.EventLog, where: Any, expected: list[float]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a pattern with groups is not a warning here, as in 0.1.0
        assert _count(log, where) == expected
        mask = parse_recipe_where(where).evaluate(EventTable(log), missing=LEGACY)
    assert mask.astype(bool).to_numpy().tolist() == legacy_recipe_mask(log.events, where).tolist()


def test_recipe_where_clause_filters_nunique_and_agg(log: wise.EventLog) -> None:
    nunique = {"name": "k", "kind": "nunique", "column": "activity", "where": {"column": "amount", "not_in": [100.0]}}
    assert compute_recipe(log, nunique).tolist() == [2, 3, 3, 2, 1]
    agg = {"name": "s", "kind": "agg", "column": "amount", "agg": "sum", "where": {"column": "activity", "regex": "Goods"}}
    assert compute_recipe(log, agg).tolist() == [100, 100, 100, 100, 100]


@pytest.mark.spec("decision:0001/1")
def test_recipe_where_clause_accepts_the_selection_grammar(log: wise.EventLog) -> None:
    gr_without_clearing = {"all": [{"attr": "activity", "eq": "Record Goods Receipt"}, {"lacks": "Clear Invoice"}]}
    assert _count(log, gr_without_clearing) == [0, 0, 0, 1, 1]
    assert _count(log, {"vendor": ["V2"], "activity": ["Record Goods Receipt"]}) == [0, 4, 0, 0, 1]
    assert _count(log, {"not": {"attr": "amount", "isna": True}}) == [2, 5, 2, 2, 1]


@pytest.mark.parametrize(
    ("where", "error", "match"),
    [
        ({"column": "nope", "in": [1]}, LogSchemaError, "unknown event column 'nope'"),
        ({"column": "nope", "note": 1}, LogSchemaError, "unknown event column 'nope'"),
        ({"in": ["A"]}, LogSchemaError, "unknown event column"),
        ({"column": "activity"}, NormError, "needs one of in/not_in/regex/eq"),
        ({"column": "activity", "regex": "("}, NormError, "regex"),
        ({"all": [{"attr": "nope", "eq": 1}]}, LogSchemaError, "unknown event column 'nope'"),
    ],
    ids=["unknown column", "unknown column without operator", "no column", "no operator", "bad pattern", "rule form"],
)
def test_malformed_recipe_where_clauses_are_refused(log: wise.EventLog, where: Any, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        _count(log, where)


# ------------------------------------------------------------------ where= on results, drivers, penalty mass
def _worst(result: wise.ScoreResult, where: Any) -> list[str]:
    return sorted(result.worst_cases("Finance", where=where).index)


#: The scalar dict and boolean Series forms of 0.1.0, with the cases they select.
LEGACY_WHERE: list[tuple[Any, list[str]]] = [
    ({"company": "B"}, ["C", "D", "E"]),
    ({"company": "A", "vendor": "V2"}, ["B"]),
    ({"case": "C"}, ["C"]),
    ({"n_events": 4}, ["A", "C", "D"]),
    ({"vendor": None}, []),
    ({}, CASES),
    (None, CASES),
]


@pytest.mark.parametrize(("where", "expected"), LEGACY_WHERE, ids=[repr(w[0]) for w in LEGACY_WHERE])
def test_scalar_where_keeps_its_0_1_0_meaning(result: wise.ScoreResult, where: Any, expected: list[str]) -> None:
    assert _worst(result, where) == expected
    oracle = legacy_where_mask(result.cases, where)
    assert sorted(oracle.index[oracle.to_numpy()]) == expected


def test_boolean_series_where_keeps_its_0_1_0_meaning(result: wise.ScoreResult) -> None:
    assert _worst(result, result.cases["company"] != "A") == ["C", "D", "E"]
    partial = pd.Series([True, True, False], index=["A", "E", "B"])  # labels it lacks are not selected
    assert _worst(result, partial) == ["A", "E"]
    pm = wise.penalty_mass(result, "Finance", "vendor", where=partial)
    assert pm["n_cases"].to_dict() == {"V1": 1, "V2": 1}


@pytest.mark.spec("decision:0001/1")
@pytest.mark.parametrize(
    ("where", "expected"),
    [
        ({"vendor": ["V2"]}, ["B", "E"]),
        ({"case": ("A", "E")}, ["A", "E"]),
        ({"company": {"B"}, "vendor": ["V1", "V9"]}, ["C", "D"]),
        ({"any": [{"attr": "vendor", "eq": "V2"}, {"has": "Cancel Invoice Receipt"}]}, ["B", "D", "E"]),
        ({"not": {"attr": "company", "in": ["A"]}}, ["C", "D", "E"]),
        ({"attr": "case", "regex": "^[AE]$"}, ["A", "E"]),
        ({"lacks": "Clear Invoice"}, ["D", "E"]),
    ],
    ids=["list", "tuple of case ids", "set and list", "any with has", "not", "regex on case id", "lacks"],
)
def test_where_accepts_the_selection_grammar(result: wise.ScoreResult, where: Any, expected: list[str]) -> None:
    assert _worst(result, where) == expected
    as_series = result.cases.index.isin(expected)
    mask = pd.Series(as_series, index=result.cases.index)
    drivers = wise.constraint_drivers(result, "Finance", where)
    pd.testing.assert_frame_equal(drivers, wise.constraint_drivers(result, "Finance", mask))
    pm = wise.penalty_mass(result, "Finance", "company", where=where)
    pd.testing.assert_frame_equal(pm, wise.penalty_mass(result, "Finance", "company", where=mask))
    assert pm["n_cases"].sum() == len(expected)


@pytest.mark.parametrize(
    ("where", "match"),
    [
        ({"nope": 1}, "unknown case attribute 'nope' in where"),
        ({"vendor": []}, "empty list"),
        ({"attr": "vendor", "like": "V%"}, "operator"),
        (["A"], "must be a mapping"),
    ],
    ids=["unknown attribute", "empty list", "unknown operator", "not a mapping"],
)
def test_malformed_where_is_refused_with_norm_error(result: wise.ScoreResult, where: Any, match: str) -> None:
    with pytest.raises(NormError, match=match):
        result.worst_cases("Finance", where=where)
    with pytest.raises(NormError, match=match):
        wise.constraint_drivers(result, "Finance", where)
    with pytest.raises(NormError, match=match):
        wise.penalty_mass(result, "Finance", "vendor", where=where)


def test_where_that_selects_no_scored_case(result: wise.ScoreResult) -> None:
    assert _worst(result, {"vendor": ["V9"]}) == []
    with pytest.raises(NotScoredError):
        wise.penalty_mass(result, "Finance", "vendor", where={"vendor": ["V9"]})


def test_prioritization_does_not_reach_into_scoring_privates() -> None:
    import wise.prioritization as prioritization

    assert not hasattr(prioritization, "_where_mask")
