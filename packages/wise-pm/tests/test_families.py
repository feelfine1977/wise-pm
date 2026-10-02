"""The six constraint families at their own interface (paper Sec. IV-B).

Each family parses its parameters from a document, states the inputs it reads
and evaluates on a log to a value and a reason per case. Every expectation is
derived by hand from the four-case log below.

======  ===============================================  ========
case    events (activity, day, amount)                   ``size``
======  ===============================================  ========
1       (A, 0, 10)  (B, 1, 10)  (C, 2, 0)                3
2       (B, 0, 5)   (A, 1, 0)   (A, 2, 0)                7
3       (C, 0, 0)                                        missing
4       (A, 0, 8)   (C, 1, 0)   (A, 2, 0)  (A, 3, 0)     12
======  ===============================================  ========

Counts: ``A`` 1, 2, 0, 3; ``B`` 1, 1, 0, 0; ``C`` 1, 0, 1, 1. First ``C``:
day 2, none, day 0, day 1. First ``B``: day 1, day 0, none, none.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest

import wise
from wise.errors import NormError
from wise.family import (
    EVALUATED,
    MISSING_ACTIVITY,
    MISSING_ANCHOR,
    MISSING_ATTRIBUTE,
    Evaluation,
    Inputs,
    family,
)

NAN = math.nan
CASES = ["1", "2", "3", "4"]
E, ANCHOR, ACTIVITY, ATTRIBUTE = EVALUATED, MISSING_ANCHOR, MISSING_ACTIVITY, MISSING_ATTRIBUTE


def four_cases() -> wise.EventLog:
    rows = [
        ("1", "A", 0, 10.0, 3.0),
        ("1", "B", 1, 10.0, 3.0),
        ("1", "C", 2, 0.0, 3.0),
        ("2", "B", 0, 5.0, 7.0),
        ("2", "A", 1, 0.0, 7.0),
        ("2", "A", 2, 0.0, 7.0),
        ("3", "C", 0, 0.0, None),
        ("4", "A", 0, 8.0, 12.0),
        ("4", "C", 1, 0.0, 12.0),
        ("4", "A", 2, 0.0, 12.0),
        ("4", "A", 3, 0.0, 12.0),
    ]
    df = pd.DataFrame(rows, columns=["case", "activity", "day", "amount", "size"])
    df["time"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(df["day"], unit="D")
    return wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="time", case_attributes=["size"])


@pytest.fixture(scope="module")
def log() -> wise.EventLog:
    return four_cases()


def check(ev: Evaluation, values: list[float], reasons: list[str]) -> None:
    assert list(ev.value.index) == CASES
    assert ev.value.dtype == float
    assert ev.value.tolist() == pytest.approx(values, nan_ok=True)
    assert ev.reason.tolist() == reasons


def run(type_name: str, params: dict[str, Any], log: wise.EventLog) -> Evaluation:
    fam = family(type_name)
    ev: Evaluation = fam.evaluate(fam.parse(params), log)
    return ev


def refused(type_name: str, params: dict[str, Any], message: str) -> None:
    with pytest.raises(NormError, match=message):
        family(type_name).parse(params)


# ------------------------------------------------------------------ presence
@pytest.mark.spec("IV-B.presence")
@pytest.mark.parametrize(
    ("params", "values"),
    [
        ({"activity": "A"}, [0.0, 0.0, 1.0, 0.0]),
        ({"activity": "A", "m": 2}, [0.5, 0.0, 1.0, 0.0]),
        ({"activity": "A", "m": "2"}, [0.5, 0.0, 1.0, 0.0]),
        ({"activity": ["B", "C"], "m": 2}, [0.0, 0.5, 0.5, 0.5]),
    ],
)
def test_presence_truth_table(log: wise.EventLog, params: dict[str, Any], values: list[float]) -> None:
    check(run("presence", params, log), values, [E] * 4)


@pytest.mark.spec("IV-B.presence")
def test_presence_parses_and_states_its_inputs() -> None:
    fam = family("pres")
    assert fam is family("presence")
    c = fam.parse({"activity": ["A", "B"], "m": "3"})
    assert c == wise.Presence(["A", "B"], m=3)
    assert isinstance(c.m, int)
    assert fam.requires(c) == Inputs(activities=("A", "B"))


@pytest.mark.spec("IV-B.presence")
@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"activity": "A", "m": 0}, "m must be an integer >= 1"),
        ({"activity": "A", "m": 1.5}, "m must be an integer >= 1"),
        ({"activity": "A", "m": "2.7"}, "m must be an integer >= 1"),
        ({"activity": "A", "m": "two"}, "parameter m='two' is not numeric"),
        ({"activity": "A", "m": True}, "m must be a number"),
        ({"activity": []}, "empty activity"),
        ({"activity": "A", "k": 1}, r"unknown parameters \['k'\]"),
    ],
)
def test_presence_refuses_bad_parameters(params: dict[str, Any], message: str) -> None:
    refused("presence", params, message)


# ----------------------------------------------------------------- exclusion
@pytest.mark.spec("IV-B")
@pytest.mark.parametrize(
    ("params", "values", "reasons"),
    [
        ({"activity": "C"}, [1.0, 0.0, 1.0, 1.0], [E] * 4),
        ({"activity": "A", "after": "C"}, [0.0, NAN, 0.0, 1.0], [E, ANCHOR, E, E]),
        ({"activity": "A", "before": "B"}, [1.0, 0.0, NAN, NAN], [E, E, ANCHOR, ANCHOR]),
        ({"activity": "B", "after": "A", "before": "C"}, [1.0, NAN, NAN, 0.0], [E, ANCHOR, ANCHOR, E]),
    ],
)
def test_exclusion_truth_table(log: wise.EventLog, params: dict[str, Any], values: list[float], reasons: list[str]) -> None:
    check(run("exclusion", params, log), values, reasons)


@pytest.mark.spec("IV-B")
def test_exclusion_parses_and_states_its_inputs() -> None:
    fam = family("excl")
    c = fam.parse({"activity": "A", "after": "B", "before": ["C", "D"]})
    assert c == wise.Exclusion("A", after="B", before=["C", "D"])
    assert fam.requires(c) == Inputs(activities=("A", "B", "C", "D"))


@pytest.mark.spec("IV-B")
@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"activity": None}, "activity must be given"),
        ({"activity": "A", "after": []}, "empty after"),
        ({"activity": "A", "before": 3}, "before must be a label or a list of labels"),
        ({"activity": "A", "m": 1}, r"unknown parameters \['m'\]"),
    ],
)
def test_exclusion_refuses_bad_parameters(params: dict[str, Any], message: str) -> None:
    refused("exclusion", params, message)


# --------------------------------------------------------------- singularity
@pytest.mark.spec("IV-B.singularity")
@pytest.mark.parametrize(
    ("params", "values", "reasons"),
    [
        ({"activity": "A"}, [0.0, 1.0, 0.0, 1.0], [E] * 4),
        ({"activity": "A", "k": 1, "K": 2}, [0.0, 0.5, 0.0, 1.0], [E] * 4),
        ({"activity": "A", "k": "0", "K": "4"}, [0.25, 0.5, 0.0, 0.75], [E] * 4),
        ({"activity": "A", "k": 0, "K": 1, "after": "C"}, [0.0, NAN, 0.0, 1.0], [E, ANCHOR, E, E]),
    ],
)
def test_singularity_truth_table(log: wise.EventLog, params: dict[str, Any], values: list[float], reasons: list[str]) -> None:
    check(run("singularity", params, log), values, reasons)


@pytest.mark.spec("IV-B.singularity")
def test_singularity_parses_and_states_its_inputs() -> None:
    fam = family("sing")
    c = fam.parse({"activity": "A", "k": 2.0, "K": "3", "before": "B"})
    assert (c.k, c.K) == (2, 3.0)
    assert isinstance(c.k, int)
    assert fam.requires(c) == Inputs(activities=("A", "B"))


@pytest.mark.spec("IV-B.singularity")
@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"activity": "A", "k": -1}, "k must be an integer >= 0"),
        ({"activity": "A", "k": 0.9}, "k must be an integer >= 0"),
        ({"activity": "A", "K": 0}, "K must be a finite number > 0"),
        ({"activity": "A", "K": "inf"}, "K must be a finite number > 0"),
        ({"activity": "A", "K": "big"}, "parameter K='big' is not numeric"),
    ],
)
def test_singularity_refuses_bad_parameters(params: dict[str, Any], message: str) -> None:
    refused("singularity", params, message)


# ---------------------------------------------------------------- precedence
@pytest.mark.spec("IV-B")
@pytest.mark.parametrize(
    ("params", "values", "reasons"),
    [
        # B before the first A: 0, 1, (no A), 0 (no B at all)
        ({"a": "A", "b": "B"}, [0.0, 1.0, NAN, 0.0], [E, E, ANCHOR, E]),
        ({"a": "A", "b": "B", "missing_a": "violate"}, [0.0, 1.0, 1.0, 0.0], [E] * 4),
        ({"a": "A", "b": "B", "missing_b": "skip"}, [0.0, 1.0, NAN, NAN], [E, E, ANCHOR, ACTIVITY]),
        (
            {"a": "A", "b": "B", "missing_a": "violate", "missing_b": "skip"},
            [0.0, 1.0, NAN, NAN],
            [E, E, ACTIVITY, ACTIVITY],
        ),
        # A before the first C: 1, (no C), 0, 1
        ({"a": "C", "b": "A", "K": 2}, [0.5, NAN, 0.0, 0.5], [E, ANCHOR, E, E]),
        ({"a": "C", "b": "A", "k": 1, "K": 2}, [0.0, NAN, 0.0, 0.0], [E, ANCHOR, E, E]),
    ],
)
def test_precedence_truth_table(log: wise.EventLog, params: dict[str, Any], values: list[float], reasons: list[str]) -> None:
    check(run("precedence", params, log), values, reasons)


@pytest.mark.spec("IV-B")
def test_precedence_parses_and_states_its_inputs() -> None:
    fam = family("order")
    c = fam.parse({"a": "A", "b": ["B", "C"], "k": "1", "K": 2})
    assert c == wise.Precedence("A", ["B", "C"], k=1, K=2.0)
    assert fam.requires(c) == Inputs(activities=("A", "B", "C"))


@pytest.mark.spec("IV-B")
@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"a": "A", "b": "B", "k": 1.5}, "k must be an integer >= 0"),
        ({"a": "A", "b": "B", "k": -1}, "k must be an integer >= 0"),
        ({"a": "A", "b": "B", "K": -1}, "K must be a finite number > 0"),
        ({"a": "A", "b": "B", "missing_a": "satisfy"}, "missing_a must be 'skip' or 'violate'"),
        ({"a": "A", "b": "B", "missing_b": "violate"}, "missing_b must be 'satisfy' or 'skip'"),
        ({"a": "A", "b": []}, "empty b"),
    ],
)
def test_precedence_refuses_bad_parameters(params: dict[str, Any], message: str) -> None:
    refused("precedence", params, message)


# ------------------------------------------------------------------- balance
BALANCE = {"attr_x": "amount", "activities_x": "A", "attr_y": "amount", "activities_y": "B"}


@pytest.mark.spec("IV-B.balance")
@pytest.mark.parametrize(
    ("params", "values"),
    [
        # tot_x (A) 10, 0, 0, 8; tot_y (B) 10, 5, 0, 0; d = 0, 1, 0 (both empty), 1
        (BALANCE, [0.0, 1.0, 0.0, 1.0]),
        ({**BALANCE, "tau": 0.2, "width": "1"}, [0.0, 0.8, 0.0, 0.8]),
        ({**BALANCE, "tau": 0.5, "width": 0}, [0.0, 1.0, 0.0, 1.0]),
        # last A amount 10, 0, (none) 0, 0; last B amount 10, 5, (none) 0, (none) 0
        ({**BALANCE, "agg": "last"}, [0.0, 1.0, 0.0, 0.0]),
    ],
)
def test_balance_truth_table(log: wise.EventLog, params: dict[str, Any], values: list[float]) -> None:
    check(run("balance", params, log), values, [E] * 4)


@pytest.mark.spec("IV-B.balance")
def test_balance_parses_and_states_its_inputs() -> None:
    fam = family("bal")
    c = fam.parse({"attr_x": "net", "activities_x": ["A", "B"], "attr_y": "gross", "activities_y": "C", "tau": "0.1"})
    assert c.tau == 0.1
    assert fam.requires(c) == Inputs(activities=("A", "B", "C"), event_columns=("net", "gross"))


@pytest.mark.spec("IV-A")
def test_balance_refuses_negative_totals() -> None:
    df = pd.DataFrame(
        {"case": ["1", "1"], "activity": ["A", "B"], "time": pd.to_datetime(["2024-01-01", "2024-01-02"]), "amount": [-1.0, 1.0]}
    )
    neg = wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="time")
    with pytest.raises(NormError, match="balance totals must be non-negative"):
        run("balance", BALANCE, neg)


@pytest.mark.spec("IV-B.balance")
@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({**BALANCE, "tau": 1.5}, r"tau must be in \[0, 1\]"),
        ({**BALANCE, "tau": "x"}, "parameter tau='x' is not numeric"),
        ({**BALANCE, "width": -1}, "width must be a finite number >= 0"),
        ({**BALANCE, "eps": 0}, "eps must be a finite number > 0"),
        ({**BALANCE, "agg": "median"}, "unknown agg 'median'"),
    ],
)
def test_balance_refuses_bad_parameters(params: dict[str, Any], message: str) -> None:
    refused("balance", params, message)


# -------------------------------------------------------------------- metric
@pytest.mark.spec("IV-B.sat")
@pytest.mark.parametrize(
    ("params", "values"),
    [
        # size 3, 7, missing, 12
        ({"attribute": "size", "threshold": 5, "width": 4}, [0.0, 0.5, NAN, 1.0]),
        ({"attribute": "size", "threshold": "5", "width": "4", "direction": "low"}, [0.5, 0.0, NAN, 0.0]),
        ({"attribute": "size", "threshold": 7, "width": 0}, [0.0, 0.0, NAN, 1.0]),
    ],
)
def test_metric_truth_table(log: wise.EventLog, params: dict[str, Any], values: list[float]) -> None:
    check(run("metric", params, log), values, [E, E, ATTRIBUTE, E])


@pytest.mark.spec("IV-B.sat")
def test_metric_parses_and_states_its_inputs() -> None:
    fam = family("metric")
    c = fam.parse({"attribute": "touches", "threshold": "2"})
    assert (c.threshold, c.width, c.direction) == (2.0, 1.0, "high")
    assert fam.requires(c) == Inputs(case_attributes=("touches",))


@pytest.mark.spec("IV-B.sat")
@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"attribute": ""}, "attribute must be given"),
        ({"attribute": "x", "threshold": "inf"}, "threshold must be finite"),
        ({"attribute": "x", "threshold": None}, "threshold must be finite"),
        ({"attribute": "x", "width": -1}, "width must be a finite number >= 0"),
        ({"attribute": "x", "direction": "up"}, "direction must be 'high' or 'low'"),
    ],
)
def test_metric_refuses_bad_parameters(params: dict[str, Any], message: str) -> None:
    refused("metric", params, message)
