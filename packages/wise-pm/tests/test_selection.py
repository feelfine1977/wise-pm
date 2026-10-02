"""The selection module: one rule grammar, two table adapters, an explicit missing-value policy.

Three cases throughout (``ft`` and ``val`` are case attributes, case 3 has both missing)::

    case  trace  ft  val
    1     A B    X   5
    2     A      Y   1
    3     B      -   -

Truth values are written ``T`` (in scope), ``F`` (out of scope) and ``U`` (unknown).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import wise
from wise.errors import NormError
from wise.selection import (
    LEGACY,
    THREE_VALUED,
    AllOf,
    AnyOf,
    CaseTable,
    Compare,
    EventTable,
    MissingPolicy,
    Not,
    Occurs,
    parse,
    parse_recipe_where,
    parse_where,
    select_cases,
)

T, F, U = True, False, None


@pytest.fixture(scope="module")
def log() -> wise.EventLog:
    df = pd.DataFrame(
        {
            "case": [1, 1, 2, 3],
            "activity": ["A", "B", "A", "B"],
            "time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-01"]),
            "ft": ["X", "Z", "Y", None],
            "val": [5.0, 5.0, 1.0, np.nan],
        }
    )
    return wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="time", case_attributes=["ft", "val"])


def _cases(log: wise.EventLog, rule: Any, missing: MissingPolicy) -> list[bool | None]:
    out = parse(rule).evaluate(CaseTable(log.cases, log), missing=missing)
    assert str(out.dtype) == "boolean"
    assert out.index.equals(log.cases.index)
    return [None if v is pd.NA else bool(v) for v in out]


# ------------------------------------------------------------------ truth table
#: Operator by missing value, hand-derived. Legacy is 0.1.0: pandas comparisons on the raw
#: column, so a missing value is False under in/eq/gt/gte/lt/lte/notna and True under
#: not_in/ne/isna. Three-valued (decision 0001, section 3): unknown for the missing case under
#: every operator, including the negations and ``not``.
TRUTH_TABLE = [
    # rule                                    legacy      three-valued
    ({"attr": "ft", "in": ["X"]}, [T, F, F], [T, F, U]),
    ({"attr": "ft", "not_in": ["X"]}, [F, T, T], [F, T, U]),
    ({"attr": "ft", "eq": "X"}, [T, F, F], [T, F, U]),
    ({"attr": "ft", "ne": "X"}, [F, T, T], [F, T, U]),
    ({"attr": "val", "gt": 3}, [T, F, F], [T, F, U]),
    ({"attr": "val", "gte": 5}, [T, F, F], [T, F, U]),
    ({"attr": "val", "lt": 3}, [F, T, F], [F, T, U]),
    ({"attr": "val", "lte": 1}, [F, T, F], [F, T, U]),
    ({"attr": "ft", "isna": True}, [F, F, T], [F, F, U]),
    ({"attr": "ft", "isna": False}, [T, T, F], [T, T, U]),
    ({"attr": "ft", "notna": True}, [T, T, F], [T, T, U]),
    ({"attr": "ft", "notna": False}, [F, F, T], [F, F, U]),
    ({"attr": "ft", "regex": "^[XY]$"}, [T, T, F], [T, T, U]),
    ({"attr": "val", "regex": "^5"}, [T, F, F], [T, F, U]),
    ({"not": {"attr": "ft", "eq": "X"}}, [F, T, T], [F, T, U]),
    ({"not": {"attr": "ft", "in": ["X"]}}, [F, T, T], [F, T, U]),
    ({"not": {"attr": "val", "gt": 3}}, [F, T, T], [F, T, U]),
    ({"not": {"attr": "ft", "regex": "X"}}, [F, T, T], [F, T, U]),
    ({"ft": ["X", "Y"]}, [T, T, F], [T, T, U]),
    ({"has": "A"}, [T, T, F], [T, T, F]),
    ({"lacks": ["A"]}, [F, F, T], [F, F, T]),
    ({"not": {"has": "B"}}, [F, T, F], [F, T, F]),
]


@pytest.mark.spec("decision:0001/3")
@pytest.mark.parametrize(("rule", "legacy", "three_valued"), TRUTH_TABLE, ids=[str(r[0]) for r in TRUTH_TABLE])
def test_operator_by_missing_value_truth_table(
    log: wise.EventLog, rule: dict[str, Any], legacy: list[bool | None], three_valued: list[bool | None]
) -> None:
    assert _cases(log, rule, LEGACY) == legacy
    assert _cases(log, rule, THREE_VALUED) == three_valued


#: Kleene logic for ``all`` / ``any``: a definite part decides, otherwise unknown stays unknown.
COMBINATIONS = [
    # case 3: eq -> F / U, has B -> T, has A -> F
    ({"all": [{"attr": "ft", "eq": "X"}, {"has": "B"}]}, [T, F, F], [T, F, U]),
    ({"all": [{"attr": "ft", "eq": "X"}, {"has": "A"}]}, [T, F, F], [T, F, F]),
    ({"any": [{"attr": "ft", "eq": "X"}, {"has": "B"}]}, [T, F, T], [T, F, T]),
    ({"any": [{"attr": "ft", "eq": "X"}, {"has": "A"}]}, [T, T, F], [T, T, U]),
    ({"not": {"all": [{"attr": "ft", "eq": "X"}, {"has": "B"}]}}, [F, T, T], [F, T, U]),
    ({"all": [{"attr": "ft", "ne": "Y"}, {"attr": "val", "lt": 10}]}, [T, F, F], [T, F, U]),
]


@pytest.mark.spec("decision:0001/3")
@pytest.mark.parametrize(("rule", "legacy", "three_valued"), COMBINATIONS, ids=[str(r[0]) for r in COMBINATIONS])
def test_combinators_follow_kleene_logic_under_three_valued(
    log: wise.EventLog, rule: dict[str, Any], legacy: list[bool | None], three_valued: list[bool | None]
) -> None:
    assert _cases(log, rule, LEGACY) == legacy
    assert _cases(log, rule, THREE_VALUED) == three_valued


@pytest.mark.spec("IV-C.applicability")
@pytest.mark.parametrize("rule", [r[0] for r in TRUTH_TABLE + COMBINATIONS])
def test_applicability_is_the_legacy_evaluation(log: wise.EventLog, rule: dict[str, Any]) -> None:
    mask = wise.NormConstraint("c", "L", wise.Presence("A"), applicability=rule).applies_to(log.cases, log)
    assert mask.dtype == bool
    assert mask.tolist() == _cases(log, rule, LEGACY)


def test_empty_rule_selects_every_row(log: wise.EventLog) -> None:
    empty: tuple[Any, ...] = (None, {})
    for rule in empty:
        sel = parse(rule)
        assert sel.rule is None
        assert (sel.columns, sel.activities, sel.document) == ((), (), {})
        assert _cases(log, rule, THREE_VALUED) == [T, T, T]


def test_policy_is_an_explicit_argument(log: wise.EventLog) -> None:
    sel = parse({"attr": "ft", "eq": "X"})
    with pytest.raises(TypeError):
        sel.evaluate(CaseTable(log.cases, log))  # type: ignore[call-arg]
    with pytest.raises(NormError, match="missing-value policy"):
        sel.evaluate(CaseTable(log.cases, log), missing="kleene")  # type: ignore[arg-type]


# ------------------------------------------------------------------ parsing
def test_rule_form_parses_into_a_tree_that_states_its_inputs() -> None:
    sel = parse(
        {
            "all": [
                {"attr": "ft", "in": ["X"]},
                {"any": [{"has": ["A", "B"]}, {"not": {"attr": "val", "gte": 3}}]},
                {"lacks": "C"},
                {"attr": "ft", "ne": "Y"},
            ]
        }
    )
    assert sel.rule == AllOf(
        (
            Compare("ft", "in", ["X"]),
            AnyOf((Occurs(("A", "B"), True), Not(Compare("val", "gte", 3)))),
            Occurs(("C",), False),
            Compare("ft", "ne", "Y"),
        )
    )
    assert sel.columns == ("ft", "val")
    assert sel.activities == ("A", "B", "C")


def test_short_form_is_an_and_of_membership_tests() -> None:
    sel = parse({"ft": "X", "val": (np.float64(1.0), 2)})
    assert sel.rule == AllOf((Compare("ft", "in", ["X"]), Compare("val", "in", [1.0, 2])))
    assert sel.document == {"ft": ["X"], "val": [1.0, 2]}
    assert type(sel.document["val"][0]) is float
    assert sel.columns == ("ft", "val")
    assert sel.activities == ()


def test_rule_form_document_is_a_copy() -> None:
    source: dict[str, Any] = {"any": [{"attr": "ft", "in": ["X"]}]}
    sel = parse(source)
    source["any"].append({"has": "A"})
    assert sel.document == {"any": [{"attr": "ft", "in": ["X"]}]}


@pytest.mark.parametrize(
    "make",
    [
        lambda values: {"attr": "ft", "in": values},
        lambda values: {"not": {"attr": "ft", "not_in": values}},
        lambda values: {"ft": values},
    ],
    ids=["rule form", "nested rule form", "short form"],
)
def test_changing_the_source_after_parsing_changes_nothing(log: wise.EventLog, make: Any) -> None:
    values = ["X"]
    nc = wise.NormConstraint("c", "L", wise.Presence("A"), applicability=make(values))
    values.append("Y")
    assert nc.applies_to(log.cases, log).tolist() == [T, F, F]
    assert nc.applicability == make(["X"])


#: Every rule 0.1.0 refused, with the class it refused them with.
REFUSED: list[Any] = [
    ["ft"],
    "ft",
    {"all": []},
    {"any": {"attr": "ft", "eq": "X"}},
    {"all": [{"attr": "ft", "eq": "X"}], "any": [{"has": "A"}]},
    {"all": ["ft"]},
    {"not": {"attr": "ft"}, "has": "A"},
    {"not": {"ft": ["X"], "lacks": "A"}},
    {"has": "A", "attr": "x"},
    {"has": None},
    {"lacks": 5},
    {"attr": "x"},
    {"attr": "x", "in": [1], "eq": 2},
    {"attr": "x", "like": "X%"},
    {"flow_type": []},
    {"any": [{"not": {"foo": "bar"}}]},
    {"attr": "ft", "regex": "("},
    {"attr": "ft", "regex": 5},
]


@pytest.mark.spec("IV-C.applicability")
@pytest.mark.parametrize("rule", REFUSED, ids=[repr(r) for r in REFUSED])
def test_malformed_rules_are_refused_with_norm_error(rule: Any) -> None:
    with pytest.raises(NormError):
        parse(rule)
    with pytest.raises(NormError):
        wise.NormConstraint("c", "L", wise.Presence("A"), applicability=rule)


def test_missing_inputs_are_refused_by_the_case_adapter(log: wise.EventLog) -> None:
    with pytest.raises(NormError, match="case attribute 'nope'"):
        parse({"attr": "nope", "eq": 1}).evaluate(CaseTable(log.cases, log), missing=LEGACY)
    with pytest.raises(NormError, match="needs the event log"):
        parse({"has": "A"}).evaluate(CaseTable(log.cases), missing=LEGACY)
    nc = wise.NormConstraint("c", "L", wise.Presence("A"), applicability={"nope": [1]})
    with pytest.raises(NormError, match="constraint 'c' is conditioned on case attribute 'nope'"):
        nc.applies_to(log.cases, log)


# ------------------------------------------------------------------ event table
@pytest.mark.spec("decision:0001/3")
def test_rule_evaluates_over_the_event_table(log: wise.EventLog) -> None:
    # events in log order: (1, A, X), (1, B, Z), (2, A, Y), (3, B, missing)
    assert log.events["activity"].tolist() == ["A", "B", "A", "B"]
    table = EventTable(log)

    def events(rule: Any, missing: MissingPolicy) -> list[bool | None]:
        out = parse(rule).evaluate(table, missing=missing)
        assert out.index.equals(log.events.index)
        return [None if v is pd.NA else bool(v) for v in out]

    assert events({"attr": "ft", "ne": "X"}, LEGACY) == [F, T, T, T]
    assert events({"attr": "ft", "ne": "X"}, THREE_VALUED) == [F, T, T, U]
    assert events({"activity": ["B"]}, THREE_VALUED) == [F, T, F, T]
    # has / lacks test the event's case: cases 1 and 2 contain A, case 3 does not
    assert events({"has": "A"}, THREE_VALUED) == [T, T, T, F]
    assert events({"all": [{"lacks": "A"}, {"attr": "val", "lt": 3}]}, THREE_VALUED) == [F, F, F, U]
    assert events({"all": [{"lacks": "A"}, {"attr": "val", "lt": 3}]}, LEGACY) == [F, F, F, F]
    with pytest.raises(NormError, match="event column 'nope'"):
        events({"attr": "nope", "eq": 1}, LEGACY)


def test_rule_attributes_stays_importable_from_the_norm_module() -> None:
    from wise.norm import _rule_attributes

    rule = {"all": [{"attr": "ft", "eq": "X"}, {"has": "A"}, {"not": {"attr": "val", "lt": 1}}]}
    assert _rule_attributes(rule) == {"ft", "val"}
    assert _rule_attributes({}) == set()


# ------------------------------------------------------------------ filters
def test_recipe_where_clause_parses_into_one_leaf() -> None:
    sel = parse_recipe_where({"column": "activity", "eq": "B", "in": ["A"], "note": "x"})
    assert sel.rule == Compare("activity", "in", ["A"])
    assert sel.document == {"column": "activity", "eq": "B", "in": ["A"], "note": "x"}
    assert sel.columns == ("activity",)
    assert parse_recipe_where({"has": "A"}).rule == Occurs(("A",), True)
    assert parse_recipe_where(None).rule is None
    with pytest.raises(NormError, match="needs one of in/not_in/regex/eq"):
        parse_recipe_where({"column": "activity", "ne": "A"})


@pytest.mark.spec("decision:0001/1")
def test_where_filter_reads_a_scalar_as_equality_and_a_list_as_membership() -> None:
    sel = parse_where({"ft": "X", "val": (np.float64(1.0), 2)})
    assert sel.rule == AllOf((Compare("ft", "eq", "X"), Compare("val", "in", [1.0, 2])))
    assert sel.document == {"all": [{"attr": "ft", "eq": "X"}, {"attr": "val", "in": [1.0, 2]}]}
    assert parse_where({"not": {"has": "A"}}).rule == Not(Occurs(("A",), True))
    assert parse_where({}).rule is None


def test_select_cases_is_a_plain_mask_with_unknown_rows_left_out(log: wise.EventLog) -> None:
    out = select_cases(log.cases, {"attr": "ft", "ne": "X"})
    assert out.dtype == bool
    assert out.tolist() == [F, T, T]  # legacy: case 3 (ft missing) is in scope under ne
    assert select_cases(log.cases, {"case": [1, 3]}).tolist() == [T, F, T]
    nullable = log.cases.assign(ft=log.cases["ft"].astype("string"))
    assert select_cases(nullable, {"ft": "X"}).tolist() == [T, F, F]  # <NA> == "X" is unknown: not selected
    with pytest.raises(NormError, match="needs the event log"):
        select_cases(log.cases, {"has": "A"})
    assert select_cases(log.cases, {"has": "B"}, log=log).tolist() == [T, F, T]
