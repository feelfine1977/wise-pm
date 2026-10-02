"""Event log (``wise.log``): regression tests for defects C6–C11, C17 and C22 and characterisation tests.

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
@pytest.mark.spec("IV-A")
# C6 closed 2026-10-02: a log is a set of cases with at least one event each (Sec. IV-A); the parse step
# refuses null lifecycle transitions instead of dropping them, and refuses a log without events
def test_all_null_lifecycle_column_raises_schema_error(p2p_events):
    with pytest.raises(LogSchemaError, match=rf"lifecycle column 'lc' has {len(p2p_events)} null transitions \(first at row 0\)"):
        build_log(p2p_events.assign(lc=None), lifecycle_col="lc")


@pytest.mark.regression
@pytest.mark.spec("IV-A")
# C6 closed 2026-10-02 (Sec. IV-A): one null transition is reported with its row, not dropped
def test_one_null_lifecycle_transition_raises_schema_error(p2p_events):
    lifecycle: list[str | None] = ["complete"] * len(p2p_events)
    lifecycle[3] = None
    with pytest.raises(LogSchemaError, match=r"lifecycle column 'lc' has 1 null transition \(first at row 3\)"):
        build_log(p2p_events.assign(lc=lifecycle), lifecycle_col="lc")
    assert len(build_log(p2p_events.assign(lc="complete"), lifecycle_col="lc").events) == len(p2p_events)


@pytest.mark.regression
@pytest.mark.spec("IV-A")
# C6 closed 2026-10-02 (Sec. IV-A): an event frame without rows is refused
def test_log_with_zero_events_raises_schema_error(p2p_events):
    with pytest.raises(LogSchemaError, match="event frame has no rows"):
        build_log(p2p_events.iloc[:0])


@pytest.mark.regression
@pytest.mark.spec("IV-A")
# C6 closed 2026-10-02 (Sec. IV-A): a log whose filters remove every event is refused with the filter named
def test_log_emptied_by_its_filters_raises_schema_error(p2p_events):
    with pytest.raises(LogSchemaError, match=r"no event of lifecycle column 'lc' has a transition in \['complete'\]"):
        build_log(p2p_events.assign(lc="start"), lifecycle_col="lc")
    with pytest.raises(LogSchemaError, match="every event has a null or unparseable timestamp"):
        build_log(p2p_events.assign(time=pd.NaT), missing_timestamps="drop")


# ---------------------------------------------------------------- C7: mutable public state


@pytest.mark.regression
@pytest.mark.spec("IV-A")
# The event-log side of C7 (Sec. IV-A: the log is fixed once built): log.events hands out a copy. C7 closed 2026-10-02
# together with the norm side (test_defects_norm.py) and the per-case primitives below.
def test_mutating_returned_events_frame_leaves_log_unchanged(p2p_events):
    log = build_log(p2p_events)
    original_activities = log.events[log.activity_col].tolist()
    ev = log.events
    ev[log.activity_col] = "ZZZ"
    assert log.events[log.activity_col].tolist() == original_activities
    assert log.count(GR)["B"] == 4


@pytest.mark.regression
@pytest.mark.spec("IV-A")
# C7 closed 2026-10-02 (Sec. IV-A): the per-case primitives hand out a copy of what they remember, so writing into one
# does not change the log's later answers
def test_writing_into_a_returned_primitive_leaves_log_unchanged(p2p_events):
    log = build_log(p2p_events)
    counts, first, last = log.count(GR), log.first_ts(GR), log.last_ts(GR)
    expected = (counts.copy(), first.copy(), last.copy())
    counts.iloc[:] = -1.0
    first.iloc[:] = pd.NaT
    last.iloc[:] = pd.NaT
    pd.testing.assert_series_equal(log.count(GR), expected[0])
    pd.testing.assert_series_equal(log.first_ts(GR), expected[1])
    pd.testing.assert_series_equal(log.last_ts(GR), expected[2])
    assert log.count(GR)["B"] == 4


# ---------------------------------------------------------------- C8, C22: NaT rules differ per primitive


def uniform_log(events: pd.DataFrame) -> wise.EventLog:
    """EventLog that keeps NaT events under the uniform rule for untimed events."""
    return wise.EventLog(events, schema=wise.LogSchema(**COLS, missing_timestamps="keep", untimed_events="uniform"))


@pytest.mark.regression
@pytest.mark.spec("IV-A")
# C8 closed 2026-10-02 (Sec. IV-A; decision record 0001, section 4): under LogSchema(untimed_events="uniform") an event
# without a timestamp counts wherever no time is compared, so the count recipe and Presence see the same events
def test_count_recipe_agrees_with_presence_under_the_uniform_rule(p2p_events):
    log = uniform_log(with_nat(p2p_events, "E", GR))
    assert log.n_missing_timestamps == 1
    log.derive([{"name": "n_gr", "kind": "count", "activities": [GR]}])
    pd.testing.assert_series_equal(log.cases["n_gr"], log.count(GR), check_names=False)
    assert log.cases.loc["E", "n_gr"] == 1.0
    presence = wise.evaluate_constraint(log, wise.NormConstraint("p", "L", wise.Presence(GR)))
    # sat(1 - n_gr; 0, 1): the recipe's view of "at least one goods receipt"
    from_recipe = wise.evaluate_constraint(log, wise.NormConstraint("m", "L", wise.Metric("n_gr", 1, 1, "low")))
    assert presence.tolist() == from_recipe.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.mark.regression
@pytest.mark.spec("decision:0001")
# C8 (decision record 0001, section 4): the default rule is the 0.1.0 one, so classic numbers do not move
def test_the_legacy_default_keeps_the_0_1_0_count_recipe(p2p_events):
    log = build_log(with_nat(p2p_events, "E", GR), missing_timestamps="keep")
    assert log.schema.untimed_events.value == "legacy"
    log.derive([{"name": "n_gr", "kind": "count", "activities": [GR]}])
    assert log.cases.loc["E", "n_gr"] == 0.0
    assert log.count(GR)["E"] == 1.0


@pytest.mark.regression
@pytest.mark.spec("decision:0001")
# C22 closed 2026-10-02 (decision record 0001, section 4): one rule, keyed on the log schema, says what an untimed
# event contributes to each primitive (docs/semantics/log-policies.md); case E's goods receipt is untimed here
def test_one_rule_per_schema_states_what_an_untimed_event_contributes(p2p_events):
    events = with_nat(p2p_events, "E", GR)
    for log, unscoped in ((build_log(events, missing_timestamps="keep"), 0.0), (uniform_log(events), 1.0)):
        assert log.count(GR)["E"] == 1.0  # counted
        assert log.total("amount", GR)["E"] == 100.0  # counted
        assert log.count_scoped(GR)["E"] == unscoped  # ignored under legacy, counted under uniform
        assert log.count_after(GR, PO)["E"] == 0.0  # it cannot be placed after the order
        assert pd.isna(log.first_ts(GR)["E"])  # it never defines a time
        assert pd.isna(log.first_after(PO, GR)[1]["E"])
        assert log.trace("E")["activity"].tolist() == [PO, GR]


# ---------------------------------------------------------------- C17: varying case attributes


@pytest.mark.regression
@pytest.mark.spec("IV-A")
# C17 closed 2026-10-02 (Sec. IV-A, att_β(σ) is one value per case): the default keeps the first non-null value and
# records the varying attributes with their case counts; case_attribute_values="strict" refuses the log
def test_varying_case_attributes_are_recorded_or_refused(p2p_events):
    events = p2p_events.copy()
    events.loc[(events["case"] == "B") & (events["activity"] == CLR), "company"] = "Z"
    events.loc[events["case"].isin(["C", "D"]) & (events["activity"] == PO), "vendor"] = "V9"
    log = build_log(events, case_attributes=["company", "vendor", "flow_type"])
    assert log.cases["company"].to_dict() == {"A": "A", "B": "A", "C": "B", "D": "B", "E": "B"}
    assert log.cases.loc["C", "vendor"] == "V9"
    report = log.validate()
    assert report["varying_within_case[company]"] == 1
    assert report["varying_within_case[vendor]"] == 2
    assert "varying_within_case[flow_type]" not in report
    with pytest.raises(LogSchemaError, match=r"case attributes vary within a case: 'company' \(1 case\), 'vendor' \(2 cases\)"):
        wise.EventLog(
            events, schema=wise.LogSchema(**COLS, case_attributes=["company", "vendor"], case_attribute_values="strict")
        )
    strict = wise.LogSchema(**COLS, case_attributes=["company", "vendor", "flow_type"], case_attribute_values="strict")
    assert wise.EventLog(p2p_events, schema=strict).cases["vendor"].tolist() == ["V1", "V2", "V1", "V1", "V2"]


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
