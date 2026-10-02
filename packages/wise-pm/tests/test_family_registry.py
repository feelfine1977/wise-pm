"""The evaluation record, the public ``evaluate`` and the registry of constraint families.

A toy family registered here is parsed from a norm document, checked against
a log and scored through the public interface only. Expectations are derived
by hand from the three-case log below.

======  ==================================
case    events (activity, day, amount)
======  ==================================
1       (A, 0, 4)  (A, 1, 3)  (B, 2, 0)
2       (A, 0, 2)  (B, 1, 9)
3       (B, 0, 1)
======  ==================================
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

import wise
from wise.constraints import Constraint, as_labels
from wise.errors import NormError
from wise.families import presence
from wise.family import (
    CENSORED,
    EVALUATED,
    MISSING_ACTIVITY,
    NOT_EVALUABLE,
    OUT_OF_SCOPE,
    REASONS,
    VALUED,
    Evaluation,
    Family,
    Inputs,
    families,
    family,
    family_of,
    register,
    unregister,
)

# Schema version 2 documents are read here; each read announces the 0.1.0 semantics (LegacySemanticsWarning).
pytestmark = pytest.mark.filterwarnings("ignore::wise.LegacySemanticsWarning")

NAN = math.nan
CASES = ["1", "2", "3"]


def three_cases() -> wise.EventLog:
    rows = [
        ("1", "A", 0, 4.0, "X"),
        ("1", "A", 1, 3.0, "X"),
        ("1", "B", 2, 0.0, "X"),
        ("2", "A", 0, 2.0, "Y"),
        ("2", "B", 1, 9.0, "Y"),
        ("3", "B", 0, 1.0, "X"),
    ]
    df = pd.DataFrame(rows, columns=["case", "activity", "day", "amount", "region"])
    df["time"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(df["day"], unit="D")
    return wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="time", case_attributes=["region"])


# ------------------------------------------------------------------ a toy family
@dataclass(frozen=True)
class Ceiling(Constraint):
    """The amounts on ``activity`` must not exceed ``ceiling`` in total: ``ν = 1[Σ amount > ceiling]``."""

    type: ClassVar[str] = "ceiling"
    activity: str
    column: str = "amount"
    ceiling: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "activity", as_labels(self.activity, what="activity")[0])
        if isinstance(self.ceiling, bool) or not isinstance(self.ceiling, int | float):
            raise NormError(f"ceiling: ceiling must be a number, got {self.ceiling!r}")
        object.__setattr__(self, "ceiling", float(self.ceiling))

    def activities(self) -> tuple[str, ...]:
        return (self.activity,)


def _ceiling_requires(c: Ceiling) -> Inputs:
    return Inputs(activities=c.activities(), case_attributes=("region",), event_columns=(c.column,))


def _ceiling_evaluate(c: Ceiling, log: wise.EventLog) -> Evaluation:
    total = log.total(c.column, c.activity)
    present = log.count(c.activity) > 0
    value = (total > c.ceiling).astype(float).where(present)
    return Evaluation.of(value, log.case_ids, why=[(~present, MISSING_ACTIVITY)])


CEILING = Family(
    name="ceiling",
    constraint=Ceiling,
    requires=_ceiling_requires,
    evaluate=_ceiling_evaluate,
    numbers=frozenset({"ceiling"}),
    aliases=("ceil",),
)


@pytest.fixture
def ceiling() -> Iterator[Family[Ceiling]]:
    register(CEILING)
    try:
        yield CEILING
    finally:
        unregister("ceiling")


def toy_norm(params: dict[str, Any], type_name: str = "ceil") -> dict[str, Any]:
    return {
        "schema_version": 2,
        "name": "toy",
        "constraints": [
            {"id": "cap", "layer": "L", "type": type_name, "params": params},
            {"id": "has_b", "layer": "L", "type": "presence", "params": {"activity": "B"}},
        ],
        "views": [{"name": "v", "constraint_weights": {"cap": 3, "has_b": 1}}],
    }


@pytest.mark.spec("IV-D")
def test_a_registered_family_is_parsed_checked_and_scored(ceiling: Family[Ceiling]) -> None:
    log = three_cases()
    norm = wise.Norm.from_dict(toy_norm({"activity": "A", "ceiling": "5"}))
    assert norm.get_constraint("cap").constraint == Ceiling("A", ceiling=5.0)
    assert wise.Norm.loads(norm.dumps()) == norm
    assert norm.check(log) == []
    assert norm.attributes() == ["region"]

    # A totals 7, 2, (none): cap ν = 1, 0, not evaluable; has_b ν = 0, 0, 0
    result = wise.score(log, norm, mode="flat")
    assert result.violations["cap"].tolist() == pytest.approx([1.0, 0.0, NAN], nan_ok=True)
    assert result.violations["has_b"].tolist() == [0.0, 0.0, 0.0]
    # weights 3 and 1: S = 1 - 3/4 * 1 = 0.25, 1, and case 3 only has has_b: 1
    assert result.scores["v"].tolist() == pytest.approx([0.25, 1.0, 1.0])

    ev = wise.evaluate(norm.get_constraint("cap"), log)
    assert ev.reason.tolist() == [EVALUATED, EVALUATED, MISSING_ACTIVITY]


@pytest.mark.spec("IV-C.norm")
def test_the_norm_check_reports_what_a_family_says_it_reads(ceiling: Family[Ceiling]) -> None:
    df = three_cases().events.drop(columns=["amount"])
    log = wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="time")
    norm = wise.Norm.from_dict(toy_norm({"activity": "Z"}, type_name="ceiling"))
    assert norm.check(log) == [
        "constraint 'cap': activity 'Z' never occurs in the log",
        "constraint 'cap': ceiling attribute 'region' is neither a case attribute nor derived",
        "constraint 'cap': ceiling attribute 'amount' is not an event column",
    ]


@pytest.mark.spec("IV-C.norm")
def test_a_family_parses_its_parameters_from_a_document(ceiling: Family[Ceiling]) -> None:
    with pytest.raises(NormError, match="ceiling: parameter ceiling='high' is not numeric"):
        wise.Norm.from_dict(toy_norm({"activity": "A", "ceiling": "high"}))
    with pytest.raises(NormError, match=r"ceiling: unknown parameters \['m'\]"):
        wise.Norm.from_dict(toy_norm({"activity": "A", "m": 1}))
    with pytest.raises(NormError, match="ceiling must be a number"):
        wise.Norm.from_dict(toy_norm({"activity": "A", "ceiling": [1]}))


def test_registering_and_removing_a_family_changes_the_known_types() -> None:
    before = dict(wise.CONSTRAINT_TYPES)
    register(CEILING)
    try:
        assert wise.CONSTRAINT_TYPES["ceiling"] is Ceiling
        assert family("ceil") is CEILING
        assert family_of(Ceiling("A")) is CEILING
        assert families()[-1] is CEILING
    finally:
        assert unregister("ceil") is CEILING
    assert before == wise.CONSTRAINT_TYPES
    assert "ceil" not in wise.constraints.TYPE_ALIASES
    with pytest.raises(NormError, match="unknown constraint type 'ceiling'"):
        wise.constraint_from_dict("ceiling", {"activity": "A"})
    with pytest.raises(NormError, match="does not belong to a registered constraint family"):
        wise.evaluate(Ceiling("A"), three_cases())


@pytest.mark.parametrize(
    ("fam", "message"),
    [
        pytest.param(
            Family("presence", wise.Presence, presence.requires, presence.evaluate), "'presence' is already registered", id="name"
        ),
        pytest.param(
            Family("ceiling", Ceiling, CEILING.requires, CEILING.evaluate, aliases=("pres",)),
            "'pres' is already registered",
            id="alias",
        ),
        pytest.param(Family("pres", wise.Presence, presence.requires, presence.evaluate), "has type 'presence'", id="type"),
    ],
)
def test_the_registry_refuses_a_clash(fam: Family[Any], message: str) -> None:
    with pytest.raises(NormError, match=message):
        register(fam)
    assert "ceiling" not in wise.CONSTRAINT_TYPES


def test_the_built_in_families_are_registered_in_catalogue_order() -> None:
    names = [f.name for f in families()]
    assert names == ["presence", "exclusion", "singularity", "lag", "precedence", "balance", "metric"]
    assert list(wise.CONSTRAINT_TYPES) == names
    assert all(wise.CONSTRAINT_TYPES[f.name] is f.constraint for f in families())
    assert wise.constraints.TYPE_ALIASES == {
        "pres": "presence",
        "excl": "exclusion",
        "sing": "singularity",
        "order": "precedence",
        "bal": "balance",
    }


# ------------------------------------------------------------------ evaluate
@pytest.mark.spec("IV-C.applicability")
def test_evaluate_gives_the_value_and_the_reason_per_case() -> None:
    log = three_cases()
    # B before the first A: 0, 0, (no A); applicability region == X keeps cases 1 and 3
    nc = wise.NormConstraint("p", "L", wise.Precedence("A", "B"), applicability={"region": ["X"]})
    ev = wise.evaluate(nc, log)
    assert isinstance(ev, wise.Evaluation)
    assert (ev.value.name, ev.reason.name) == ("p", "p")
    assert ev.value.tolist() == pytest.approx([0.0, NAN, NAN], nan_ok=True)
    assert ev.reason.tolist() == [EVALUATED, OUT_OF_SCOPE, "missing_anchor"]
    assert ev.in_scope.tolist() == [True, False, True]
    assert ev.evaluated.tolist() == [True, False, False]
    assert set(ev.reason) <= set(REASONS)

    bare = wise.evaluate(nc.constraint, log)
    assert bare.reason.tolist() == [EVALUATED, EVALUATED, "missing_anchor"]
    assert bare.value.name is None


@pytest.mark.spec("IV-D")
def test_evaluate_constraint_and_the_violation_matrix_are_read_from_evaluate(p2p_log: wise.EventLog, p2p_norm: wise.Norm) -> None:
    V, S = wise.violation_matrix(p2p_log, p2p_norm, return_scope=True)
    for nc in p2p_norm.constraints:
        ev = wise.evaluate(nc, p2p_log)
        pd.testing.assert_series_equal(wise.evaluate_constraint(p2p_log, nc), ev.value)
        pd.testing.assert_series_equal(V[nc.id], ev.value)
        pd.testing.assert_series_equal(S[nc.id], ev.in_scope.rename(nc.id))


def test_evaluate_names_the_constraint_when_its_family_refuses_the_log() -> None:
    df = pd.DataFrame(
        {"case": ["1", "1"], "activity": ["A", "B"], "time": pd.to_datetime(["2024-01-01", "2024-01-02"]), "amount": [-1.0, 1.0]}
    )
    log = wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="time")
    nc = wise.NormConstraint("bal", "L", wise.Balance("amount", "A", "amount", "B"))
    with pytest.raises(NormError, match=r"^constraint 'bal': balance totals must be non-negative"):
        wise.evaluate(nc, log)
    with pytest.raises(NormError, match=r"^balance totals must be non-negative"):
        wise.evaluate(nc.constraint, log)


# ------------------------------------------------------------------ the record
def test_an_evaluation_names_the_reason_of_every_missing_value() -> None:
    index = pd.Index(["a", "b", "c", "d"])
    ev = Evaluation.of(
        [0.5, NAN, NAN, NAN],
        index,
        why=[(np.array([True, True, False, False]), "missing_anchor"), (np.array([False, True, True, False]), MISSING_ACTIVITY)],
    )
    # the first reason that covers a case wins; a value is never given a reason
    assert ev.reason.tolist() == [EVALUATED, "missing_anchor", MISSING_ACTIVITY, NOT_EVALUABLE]
    scoped = ev.within(pd.Series([True, False, True, True], index=index))
    assert scoped.reason.tolist() == [EVALUATED, OUT_OF_SCOPE, MISSING_ACTIVITY, NOT_EVALUABLE]
    assert scoped.in_scope.tolist() == [True, False, True, True]


def test_a_family_can_name_a_reason_of_its_own() -> None:
    index = pd.Index(["a", "b"])
    ev = Evaluation.of([NAN, 1.0], index, why=[(np.array([True, False]), "too_early")])
    assert ev.reason.tolist() == ["too_early", EVALUATED]
    assert ev.within(pd.Series([True, False], index=index)).reason.tolist() == ["too_early", OUT_OF_SCOPE]


def test_a_censored_case_has_a_value_and_counts_as_evaluated() -> None:
    index = pd.Index(["a", "b", "c"])
    ev = Evaluation.of([0.5, NAN, 1.0], index, why=[(np.array([True, True, False]), CENSORED)])
    assert ev.reason.tolist() == [CENSORED, NOT_EVALUABLE, EVALUATED]  # a valued reason never covers a missing value
    assert ev.evaluated.tolist() == [True, False, True]
    assert ev.in_scope.all()
    assert Evaluation(ev.value, ev.reason.astype(str)).reason.tolist() == [CENSORED, NOT_EVALUABLE, EVALUATED]
    assert {EVALUATED, CENSORED} == VALUED


def test_an_evaluation_refuses_values_and_reasons_that_disagree() -> None:
    index = pd.Index(["a", "b"])
    message = r"value exactly where its reason is one of \['censored', 'evaluated'\]"
    with pytest.raises(NormError, match=message):
        Evaluation(pd.Series([0.0, 1.0], index=index), pd.Series([EVALUATED, OUT_OF_SCOPE], index=index))
    with pytest.raises(NormError, match=message):
        Evaluation(pd.Series([0.0, NAN], index=index), pd.Series([EVALUATED, EVALUATED], index=index))
    with pytest.raises(NormError, match=message):
        Evaluation(pd.Series([0.0, NAN], index=index), pd.Series([EVALUATED, CENSORED], index=index))
    with pytest.raises(NormError, match="share one index"):
        Evaluation(pd.Series([0.0, 1.0], index=index), pd.Series([EVALUATED, EVALUATED], index=pd.Index(["a", "c"])))
