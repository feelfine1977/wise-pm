"""Event log (``wise.log``): regression tests for defects C6–C11 and characterisation tests.

Regression tests pin the desired behaviour and are ``xfail(strict=True)`` with the
exception the current code produces; they turn green when the fix lands.
Characterisation tests pin behaviour that is already correct and must stay.
"""

from typing import Any

import pandas as pd
import pytest

import wise
from wise.errors import LogSchemaError, WiseError

PO = "Create Purchase Order Item"
GR = "Record Goods Receipt"
INV = "Record Invoice Receipt"
CLR = "Clear Invoice"
CINV = "Cancel Invoice Receipt"

COLS: dict[str, Any] = {"case_col": "case", "activity_col": "activity", "timestamp_col": "time"}


def build_log(events: pd.DataFrame, **kwargs) -> wise.EventLog:
    """EventLog over the running example's column names."""
    return wise.EventLog(events, **COLS, **kwargs)


def with_nat(events: pd.DataFrame, case: str, activity: str) -> pd.DataFrame:
    """Copy of ``events`` with the timestamp of the first ``(case, activity)`` event set to NaT."""
    out = events.copy()
    pos = out.index[(out["case"] == case) & (out["activity"] == activity)][0]
    out.loc[pos, "time"] = pd.NaT
    return out


@pytest.fixture
def p2p_events() -> pd.DataFrame:
    return wise.running_p2p_events()


# ---------------------------------------------------------------- C6: silent empty logs


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="C6: null lifecycle transitions never match keep_transitions and are dropped silently, leaving an empty log",
)
def test_all_null_lifecycle_column_raises_schema_error(p2p_events):
    with pytest.raises(LogSchemaError):
        build_log(p2p_events.assign(lc=None), lifecycle_col="lc")


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="C6: a single null lifecycle transition is dropped silently instead of being reported",
)
def test_one_null_lifecycle_transition_raises_schema_error(p2p_events):
    lifecycle: list[str | None] = ["complete"] * len(p2p_events)
    lifecycle[3] = None
    with pytest.raises(LogSchemaError):
        build_log(p2p_events.assign(lc=lifecycle), lifecycle_col="lc")


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="C6: an EventLog with zero events is built silently as 'EventLog(0 events, 0 cases)'",
)
def test_log_with_zero_events_raises_schema_error(p2p_events):
    with pytest.raises(LogSchemaError):
        build_log(p2p_events.iloc[:0])


# ---------------------------------------------------------------- C7: mutable public state


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="C7: log.events is the live internal frame, so an in-place edit desynchronises it from the index",
)
def test_mutating_returned_events_frame_leaves_log_unchanged(p2p_events):
    log = build_log(p2p_events)
    original_activities = log.events[log.activity_col].tolist()
    ev = log.events
    ev[log.activity_col] = "ZZZ"
    assert log.events[log.activity_col].tolist() == original_activities
    assert log.count(GR)["B"] == 4


# ---------------------------------------------------------------- C8: NaT rules differ per primitive


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="C8: the count recipe goes through count_scoped and masks NaT events while log.count includes them",
)
def test_count_recipe_agrees_with_count_when_nat_events_are_kept(p2p_events):
    log = build_log(with_nat(p2p_events, "E", GR), missing_timestamps="keep")
    assert log.n_missing_timestamps == 1
    log.derive([{"name": "n_gr", "kind": "count", "activities": [GR]}])
    pd.testing.assert_series_equal(log.cases["n_gr"], log.count(GR), check_names=False)


# ---------------------------------------------------------------- C9: errors escaping the hierarchy


@pytest.mark.regression
# C9 closed 2026-09-27: every public entry point raises a WiseError subclass (decision record 0001, error contract)
def test_unknown_missing_timestamps_option_raises_wise_error(p2p_events):
    with pytest.raises(WiseError):
        build_log(p2p_events, missing_timestamps="x")


@pytest.mark.regression
# C9 closed 2026-09-27: every public entry point raises a WiseError subclass (decision record 0001, error contract)
def test_reversed_window_raises_wise_error(p2p_events):
    with pytest.raises(WiseError):
        build_log(p2p_events, window=("2024-02-01", "2024-01-01"))


@pytest.mark.regression
# C9 closed 2026-09-27: every public entry point raises a WiseError subclass (decision record 0001, error contract)
def test_unknown_exposure_agg_raises_wise_error(p2p_events):
    with pytest.raises(WiseError):
        build_log(p2p_events, exposure_col="amount", exposure_agg="nonsense")


# ---------------------------------------------------------------- C11: LogSchemaError is a KeyError


@pytest.mark.regression
# C11 closed 2026-09-27: library errors inherit no builtin (decision record 0001, error contract)
def test_except_key_error_does_not_swallow_log_schema_error():
    caught_as_key_error = False
    try:
        raise LogSchemaError("x")
    except KeyError:
        caught_as_key_error = True
    except LogSchemaError:
        pass
    assert not caught_as_key_error


# ---------------------------------------------------------------- characterisation: missing timestamps


def test_drop_removes_exactly_the_nat_events(p2p_events):
    log = build_log(with_nat(p2p_events, "B", GR), missing_timestamps="drop")
    assert len(log.events) == len(p2p_events) - 1
    assert log.events["time"].notna().all()
    assert log.count(GR).to_dict() == {"A": 1.0, "B": 3.0, "C": 1.0, "D": 1.0, "E": 1.0}


def test_drop_refactorises_case_ids_when_a_case_loses_all_its_events(p2p_events):
    events = p2p_events.copy()
    events.loc[events["case"] == "E", "time"] = pd.NaT
    log = build_log(events, missing_timestamps="drop")
    assert list(log.case_ids) == list("ABCD")
    assert len(log) == 4
    assert len(log.cases) == 4
    assert log.count(GR).to_dict() == {"A": 1.0, "B": 4.0, "C": 1.0, "D": 1.0}
    assert log.trace("D")["activity"].tolist() == [PO, GR, INV, CINV]


def test_keep_retains_nat_events_and_reports_their_number(p2p_events):
    events = p2p_events.copy()
    events.loc[events["case"] == "E", "time"] = pd.NaT
    log = build_log(events, missing_timestamps="keep")
    assert len(log.events) == len(p2p_events)
    assert int(log.events["time"].isna().sum()) == 2
    assert log.n_missing_timestamps == 2
    assert log.validate()["missing_timestamps"] == 2
    assert "E" in log.case_ids
    assert pd.isna(log.cases.loc["E", "first_ts"])


# ---------------------------------------------------------------- characterisation: trace ordering


@pytest.mark.spec("IV-A")
def test_trace_returns_the_events_of_one_case_in_timestamp_order(p2p_events):
    shuffled = p2p_events.sample(frac=1, random_state=7).reset_index(drop=True)
    trace = build_log(shuffled).trace("B")
    assert set(trace["case"]) == {"B"}
    assert trace["activity"].tolist() == [PO, GR, GR, GR, GR, INV, CLR]
    assert trace["time"].is_monotonic_increasing
