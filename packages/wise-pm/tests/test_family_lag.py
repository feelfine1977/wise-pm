"""The Lag family at its own interface (paper Sec. IV-B).

``ν = sat(t_b − t_a; δ, Δ)`` with ``δ = 2``, ``Δ = 4`` days unless a row says
otherwise, so a lag of 0 or 1 day gives 0, 3 gives 0.25, 4 gives 0.5, 5 gives
0.75 and 6 or more gives 1. The observation window ends on day 10, so an
activation on day 7 left open has waited 3 days (0.25), one on day 5 has
waited 5 days (0.75). Every expectation is derived by hand from the log below
(``A`` activates, ``B`` responds).

======  ==========================  ======  ===================================
case    events (activity, day)      flow    what it exercises
======  ==========================  ======  ===================================
1       A 0, B 3                    X       one pair, lag 3
2       A 0, A 4, B 5               X       first (5) and last (1) differ; each
3       B 0, A 2, B 6               X       a response before the activation
4       A 7                         Y       no response: violate, skip, censor
5       B 1                         Y       no activation
6       B 2, A 2                    X       a tie, the response listed first
7       A 0, B 1, A 5               Y       each: one answered, one open
======  ==========================  ======  ===================================

Per mode the pairs (activation day → response day) are:

* first, first_after: 1 0→3, 2 0→5, 3 2→6, 4 7→none, 6 2→2, 7 0→1;
* last, first_after: 1 0→3, 2 4→5, 3 2→6, 4 7→none, 6 2→2, 7 5→none;
* first, first_overall: as first/first_after, except 3 2→none (the first
  ``B`` is on day 0, before the activation);
* last, first_overall: as last/first_after, except 3 2→none and 7 5→none
  (the first ``B`` is on day 1);
* each: 1 0→3; 2 0→5 and 4→5; 3 2→6; 4 7→none; 6 2→2; 7 0→1 and 5→none.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

import wise
from wise.errors import NormError
from wise.families import lag as lag_family
from wise.family import CENSORED, EVALUATED, MISSING_ACTIVATION, MISSING_RESPONSE, OUT_OF_SCOPE, Evaluation, Inputs, family

NAN = math.nan
CASES = ["1", "2", "3", "4", "5", "6", "7"]
E, ACT, RESP, CEN, OUT = EVALUATED, MISSING_ACTIVATION, MISSING_RESPONSE, CENSORED, OUT_OF_SCOPE


def seven_cases() -> wise.EventLog:
    rows = [
        ("1", "A", 0, "X"),
        ("1", "B", 3, "X"),
        ("2", "A", 0, "X"),
        ("2", "A", 4, "X"),
        ("2", "B", 5, "X"),
        ("3", "B", 0, "X"),
        ("3", "A", 2, "X"),
        ("3", "B", 6, "X"),
        ("4", "A", 7, "Y"),
        ("5", "B", 1, "Y"),
        ("6", "B", 2, "X"),
        ("6", "A", 2, "X"),
        ("7", "A", 0, "Y"),
        ("7", "B", 1, "Y"),
        ("7", "A", 5, "Y"),
    ]
    df = pd.DataFrame(rows, columns=["case", "activity", "day", "flow"])
    start = pd.Timestamp("2024-01-01")
    df["time"] = start + pd.to_timedelta(df["day"], unit="D")
    return wise.EventLog(
        df,
        case_col="case",
        activity_col="activity",
        timestamp_col="time",
        case_attributes=["flow"],
        window=(start, start + pd.Timedelta(days=10)),
    )


@pytest.fixture(scope="module")
def log() -> wise.EventLog:
    return seven_cases()


def check(ev: Evaluation, values: list[float], reasons: list[str]) -> None:
    assert list(ev.value.index) == CASES
    assert ev.value.dtype == float
    assert ev.value.tolist() == pytest.approx(values, nan_ok=True)
    assert ev.reason.tolist() == reasons
    assert ev.evaluated.tolist() == [not math.isnan(v) for v in values]


def run(params: dict[str, Any], log: wise.EventLog) -> Evaluation:
    fam = family("lag")
    ev: Evaluation = fam.evaluate(fam.parse({"a": "A", "b": "B", "delta": 2, "width": 4, **params}), log)
    return ev


# --------------------------------------------------------------- truth table
TABLE = [
    # first activation, first response at or after it
    ({}, [0.25, 0.75, 0.5, 1.0, 1.0, 0.0, 0.0], [E, E, E, E, E, E, E]),
    ({"missing_b": "skip"}, [0.25, 0.75, 0.5, NAN, 1.0, 0.0, 0.0], [E, E, E, RESP, E, E, E]),
    ({"missing_b": "censor"}, [0.25, 0.75, 0.5, 0.25, 1.0, 0.0, 0.0], [E, E, E, CEN, E, E, E]),
    ({"missing_a": "skip"}, [0.25, 0.75, 0.5, 1.0, NAN, 0.0, 0.0], [E, E, E, E, ACT, E, E]),
    ({"missing": "skip"}, [0.25, 0.75, 0.5, NAN, NAN, 0.0, 0.0], [E, E, E, RESP, ACT, E, E]),
    ({"missing_a": "skip", "missing_b": "censor"}, [0.25, 0.75, 0.5, 0.25, NAN, 0.0, 0.0], [E, E, E, CEN, ACT, E, E]),
    # last activation
    ({"activation": "last"}, [0.25, 0.0, 0.5, 1.0, 1.0, 0.0, 1.0], [E] * 7),
    ({"activation": "last", "missing": "skip"}, [0.25, 0.0, 0.5, NAN, NAN, 0.0, NAN], [E, E, E, RESP, ACT, E, RESP]),
    ({"activation": "last", "missing_b": "censor"}, [0.25, 0.0, 0.5, 0.25, 1.0, 0.0, 0.75], [E, E, E, CEN, E, E, CEN]),
    # first response overall: a response before the activation counts as missing
    ({"response": "first_overall"}, [0.25, 0.75, 1.0, 1.0, 1.0, 0.0, 0.0], [E] * 7),
    ({"response": "first_overall", "missing": "skip"}, [0.25, 0.75, NAN, NAN, NAN, 0.0, 0.0], [E, E, RESP, RESP, ACT, E, E]),
    ({"activation": "last", "response": "first_overall"}, [0.25, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0], [E] * 7),
    (
        {"activation": "last", "response": "first_overall", "missing": "skip"},
        [0.25, 0.0, NAN, NAN, NAN, 0.0, NAN],
        [E, E, RESP, RESP, ACT, E, RESP],
    ),
    # each activation, averaged per case: 2 is (0.75 + 0) / 2, 7 is (0 + ν of the open one) / 2
    ({"activation": "each"}, [0.25, 0.375, 0.5, 1.0, 1.0, 0.0, 0.5], [E] * 7),
    ({"activation": "each", "missing": "skip"}, [0.25, 0.375, 0.5, NAN, NAN, 0.0, 0.0], [E, E, E, RESP, ACT, E, E]),
    ({"activation": "each", "missing_b": "censor"}, [0.25, 0.375, 0.5, 0.25, 1.0, 0.0, 0.375], [E, E, E, CEN, E, E, CEN]),
    (
        {"activation": "each", "missing_a": "skip", "missing_b": "censor"},
        [0.25, 0.375, 0.5, 0.25, NAN, 0.0, 0.375],
        [E, E, E, CEN, ACT, E, CEN],
    ),
    # unbounded lag: only whether a response follows matters
    ({"delta": None}, [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], [E] * 7),
    ({"delta": None, "missing": "skip"}, [0.0, 0.0, 0.0, NAN, NAN, 0.0, 0.0], [E, E, E, RESP, ACT, E, E]),
    ({"delta": None, "activation": "last"}, [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0], [E] * 7),
    ({"delta": None, "response": "first_overall"}, [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], [E] * 7),
    ({"delta": None, "activation": "each"}, [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.5], [E] * 7),
    ({"delta": None, "activation": "each", "missing": "skip"}, [0.0, 0.0, 0.0, NAN, NAN, 0.0, 0.0], [E, E, E, RESP, ACT, E, E]),
    # a step at δ (Δ = 0) and another unit: 3 days = 72 h
    ({"delta": 72, "width": 0, "unit": "h"}, [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0], [E] * 7),
]


@pytest.mark.spec("IV-B.lag")
@pytest.mark.parametrize(("params", "values", "reasons"), TABLE)
def test_lag_truth_table(log: wise.EventLog, params: dict[str, Any], values: list[float], reasons: list[str]) -> None:
    check(run(params, log), values, reasons)


@pytest.mark.spec("IV-B.lag")
def test_a_response_at_the_activation_time_counts_whatever_the_row_order(log: wise.EventLog) -> None:
    # case 6 lists B before A at the same instant; a lag of 0 is within any bound
    for activation in ("first", "last", "each"):
        ev = run({"activation": activation, "delta": 0, "width": 0, "missing": "skip"}, log)
        assert ev.value["6"] == 0.0
        assert ev.reason["6"] == E


@pytest.mark.spec("IV-B.lag")
def test_scoring_adds_out_of_scope_and_keeps_the_lag_reasons(log: wise.EventLog) -> None:
    flow_y = {"flow": ["Y"]}
    skipped = wise.NormConstraint("l", "L", wise.Lag("A", "B", delta=2, width=4, missing="skip"), applicability=flow_y)
    check(wise.evaluate(skipped, log), [NAN, NAN, NAN, NAN, NAN, NAN, 0.0], [OUT, OUT, OUT, RESP, ACT, OUT, E])
    censored = wise.NormConstraint(
        "l", "L", wise.Lag("A", "B", delta=2, width=4, missing_a="skip", missing_b="censor", activation="last"), applicability=flow_y
    )
    check(wise.evaluate(censored, log), [NAN, NAN, NAN, 0.25, NAN, NAN, 0.75], [OUT, OUT, OUT, CEN, ACT, OUT, CEN])


@pytest.mark.spec("IV-B.lag")
def test_scores_count_a_censored_case_and_leave_out_the_skipped_ones(log: wise.EventLog) -> None:
    norm = wise.Norm(
        (wise.NormConstraint("l", "L", wise.Lag("A", "B", delta=2, width=4, missing_a="skip", missing_b="censor")),),
        (wise.Layer("L"),),
        (wise.View("v", constraint_weights={"l": 1.0}),),
    )
    res = wise.score(log, norm)
    assert res.scores["v"].tolist() == pytest.approx([0.75, 0.25, 0.5, 0.75, NAN, 1.0, 1.0], nan_ok=True)
    assert res.in_scope["l"].all()
    assert res.applicable["l"].tolist() == [True, True, True, True, False, True, True]


# ------------------------------------------------------------------- pairing
@pytest.mark.spec("IV-B.lag")
@pytest.mark.parametrize(
    ("activation", "response", "negative", "per_case"),
    [
        ("first", "first_after", False, [3.0, 5.0, 4.0, NAN, NAN, 0.0, 1.0]),
        ("last", "first_after", False, [3.0, 1.0, 4.0, NAN, NAN, 0.0, NAN]),
        ("first", "first_overall", False, [3.0, 5.0, NAN, NAN, NAN, 0.0, 1.0]),
        ("first", "first_overall", True, [3.0, 5.0, -2.0, NAN, NAN, 0.0, 1.0]),
        ("last", "first_overall", True, [3.0, 1.0, -2.0, NAN, NAN, 0.0, -4.0]),
    ],
)
def test_pairs_give_one_lag_per_activated_case(
    log: wise.EventLog, activation: Any, response: Any, negative: bool, per_case: list[float]
) -> None:
    p = lag_family.pairs(log, "A", "B", activation=activation, response=response, negative=negative)
    assert p.case.tolist() == [0, 1, 2, 3, 5, 6]  # case 5 has no activation
    assert p.per_case().tolist() == pytest.approx(per_case, nan_ok=True)
    hours = lag_family.pairs(log, "A", "B", activation=activation, response=response, unit="h", negative=negative)
    assert hours.lag.tolist() == pytest.approx([24 * x for x in p.lag.tolist()], nan_ok=True)


@pytest.mark.spec("IV-B.lag")
def test_pairs_give_one_lag_per_activation_for_each(log: wise.EventLog) -> None:
    p = lag_family.pairs(log, "A", "B", activation="each")
    assert p.case.tolist() == [0, 1, 1, 2, 3, 5, 6, 6]
    assert p.lag.tolist() == pytest.approx([3.0, 5.0, 1.0, 4.0, NAN, 0.0, 1.0, NAN], nan_ok=True)
    assert p.t_a.dt.day.tolist() == [1, 1, 5, 3, 8, 3, 1, 6]  # day d of the log is 2024-01-(d + 1)
    with pytest.raises(NormError, match="one lag per case"):
        p.per_case()


@pytest.mark.spec("IV-B.lag")
@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"activation": "every"}, "activation must be 'first', 'last' or 'each'"),
        ({"response": "next"}, "response must be 'first_after' or 'first_overall'"),
        ({"activation": "each", "response": "first_overall"}, "activation='each' requires response='first_after'"),
        ({"unit": "fortnight"}, "unknown unit 'fortnight'"),
    ],
)
def test_pairs_refuse_unknown_options(log: wise.EventLog, options: dict[str, Any], message: str) -> None:
    with pytest.raises(NormError, match=message):
        lag_family.pairs(log, "A", "B", **options)


# ------------------------------------------------------- parse and refusals
@pytest.mark.spec("IV-B.lag")
def test_lag_parses_and_states_its_inputs() -> None:
    fam = family("lag")
    c = fam.parse({"a": ["A", "C"], "b": "B", "delta": "2.5", "width": "4", "unit": "H", "missing": "skip"})
    assert c == wise.Lag(["A", "C"], "B", delta=2.5, width=4.0, unit="h", missing_a="skip", missing_b="skip")
    assert fam.requires(c) == Inputs(activities=("A", "C", "B"))
    assert fam.parse({"a": "A", "b": "B", "delta": None}).delta is None


@pytest.mark.spec("IV-B.lag")
@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"delta": -1}, "delta must be a finite number >= 0"),
        ({"width": "wide"}, "parameter width='wide' is not numeric"),
        ({"unit": "fortnight"}, "unit must be one of"),
        ({"missing_a": "censor"}, "missing_a must be 'skip' or 'violate'"),
        ({"missing_b": "ignore"}, "missing_b must be 'violate', 'skip' or 'censor'"),
        ({"missing": "censor"}, "missing must be 'violate' or 'skip'"),
        ({"activation": "every"}, "activation must be 'first', 'last' or 'each'"),
        ({"response": "next"}, "response must be 'first_after' or 'first_overall'"),
        ({"activation": "each", "response": "first_overall"}, "activation='each' requires response='first_after'"),
        ({"missing_b": "censor", "delta": None}, "missing_b='censor' needs a time bound"),
        ({"missing_b": "censor", "response": "first_overall"}, "missing_b='censor' requires response='first_after'"),
        ({"a": []}, "empty"),
        ({"k": 1}, r"unknown parameters \['k'\]"),
    ],
)
def test_lag_refuses_bad_parameters(params: dict[str, Any], message: str) -> None:
    with pytest.raises(NormError, match=message):
        family("lag").parse({"a": "A", "b": "B", "delta": 2, **params})


def test_the_pairs_of_a_log_without_activations_are_empty(log: wise.EventLog) -> None:
    for activation in ("first", "each"):
        p = lag_family.pairs(log, "Z", "B", activation=activation)
        assert len(p.case) == 0
        assert p.lag.dtype == float
    assert np.isnan(lag_family.pairs(log, "Z", "B").per_case()).all()
