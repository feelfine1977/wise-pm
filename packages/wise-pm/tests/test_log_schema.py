"""The log schema and the parse step (``wise.LogSchema``, ``wise.parse``).

The schema names the columns of an event frame and carries every policy of
building a log as an enumerated value; ``parse`` turns a frame and a schema
into the events, the case index and the case table. The expectations below are
derived by hand from the four-event frame of :func:`_frame`.
"""

from __future__ import annotations

import dataclasses
import sys
import types
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

import wise
from wise.errors import LogSchemaError
from wise.log_schema import (
    ACTIVITY_COL,
    CASE_COL,
    TIMESTAMP_COL,
    CaseAttributeValues,
    DateOrder,
    Duplicates,
    ExposureAggregation,
    MissingTimestamps,
    TimeZone,
    UntimedEvents,
)

COLS: dict[str, Any] = {"case_col": "case", "activity_col": "activity", "timestamp_col": "time"}
SCHEMA = wise.LogSchema(**COLS)

POLICIES: dict[str, tuple[type[Enum], Enum]] = {
    "missing_timestamps": (MissingTimestamps, MissingTimestamps.RAISE),
    "untimed_events": (UntimedEvents, UntimedEvents.LEGACY),
    "duplicates": (Duplicates, Duplicates.KEEP),
    "time_zone": (TimeZone, TimeZone.AS_PARSED),
    "date_order": (DateOrder, DateOrder.MONTH_FIRST),
    "exposure_agg": (ExposureAggregation, ExposureAggregation.MAX),
    "case_attribute_values": (CaseAttributeValues, CaseAttributeValues.FIRST_NON_NULL),
}


def _frame() -> pd.DataFrame:
    """Case b has two events, the second a start transition; case a has one event and a duplicate of it."""
    return pd.DataFrame(
        {
            "case": ["b", "a", "b", "a"],
            "activity": ["X", "Y", "X", "Y"],
            "time": pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-03", "2024-01-01"]),
            "lc": ["complete", "COMPLETE", "start", "complete"],
            "amount": [2.0, 5.0, 3.0, 5.0],
            "org": ["n", "m", "n", "m"],
        }
    )


# ------------------------------------------------------------------ the schema value
@pytest.mark.spec("decision:0001")
def test_every_policy_is_an_enumerated_value_with_its_documented_default() -> None:
    schema = wise.LogSchema()
    doc = wise.LogSchema.__doc__ or ""
    for name, (enum, default) in POLICIES.items():
        value = getattr(schema, name)
        assert type(value) is enum
        assert value is default
        assert f"{name}" in doc
        assert f"default ``{default.value!r}``" in doc, name
    policy_fields = {f.name for f in dataclasses.fields(wise.LogSchema) if isinstance(getattr(schema, f.name), Enum)}
    assert policy_fields == set(POLICIES)


@pytest.mark.spec("decision:0001")
def test_the_default_columns_are_the_pm4py_names() -> None:
    schema = wise.LogSchema()
    assert (schema.case_col, schema.activity_col, schema.timestamp_col) == ("case:concept:name", "concept:name", "time:timestamp")
    assert schema.case_attributes == ()
    assert schema.exposure_col is schema.order_col is schema.event_id_col is schema.lifecycle_col is None
    assert schema.keep_transitions == ("complete",)
    assert schema.keep_columns is None
    assert schema.timestamp_format is None


def test_policy_names_are_accepted_and_unknown_ones_refused() -> None:
    schema = wise.LogSchema(missing_timestamps="drop", exposure_agg="sum", time_zone="utc")
    assert schema.missing_timestamps is MissingTimestamps.DROP
    assert schema.exposure_agg is ExposureAggregation.SUM
    assert schema.time_zone is TimeZone.UTC
    with pytest.raises(LogSchemaError, match=r"missing_timestamps must be one of \['raise', 'drop', 'keep'\], got 'x'"):
        wise.LogSchema(missing_timestamps="x")  # type: ignore[arg-type]
    with pytest.raises(LogSchemaError, match="exposure_agg must be one of"):
        wise.LogSchema(exposure_agg="idxmax")


def test_column_lists_are_normalised_and_reserved_names_refused() -> None:
    schema = wise.LogSchema(case_attributes="org", keep_transitions=["Complete", "START", "complete"], keep_columns="amount")
    assert schema.case_attributes == ("org",)
    assert schema.keep_transitions == ("complete", "start")
    assert schema.keep_columns == ("amount",)
    with pytest.raises(LogSchemaError, match="'n_events' is reserved"):
        wise.LogSchema(case_attributes=["n_events"])


def test_the_schema_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        SCHEMA.case_col = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, {}),
        ({"utc": True}, {"time_zone": TimeZone.UTC}),
        ({"dayfirst": True}, {"date_order": DateOrder.DAY_FIRST}),
        ({"dedupe": True}, {"duplicates": Duplicates.DROP}),
        ({"missing_timestamps": "keep"}, {"missing_timestamps": MissingTimestamps.KEEP}),
        ({"exposure_col": "amount", "exposure_agg": "sum"}, {"exposure_col": "amount", "exposure_agg": ExposureAggregation.SUM}),
        ({"case_attributes": "org", "keep_columns": ["amount"]}, {"case_attributes": ("org",), "keep_columns": ("amount",)}),
        ({"lifecycle_col": "lc", "keep_transitions": "Start"}, {"lifecycle_col": "lc", "keep_transitions": ("start",)}),
        ({"order_col": "seq", "event_id_col": "eid"}, {"order_col": "seq", "event_id_col": "eid"}),
        ({"timestamp_format": "%d/%m/%Y"}, {"timestamp_format": "%d/%m/%Y"}),
    ],
)
def test_from_options_maps_the_constructor_options_onto_the_schema(options: dict[str, Any], expected: dict[str, Any]) -> None:
    assert wise.LogSchema.from_options(**COLS, **options) == dataclasses.replace(SCHEMA, **expected)


# ------------------------------------------------------------------ digest
ALTERNATIVES: dict[str, Any] = {
    "case_col": "case2",
    "activity_col": "activity2",
    "timestamp_col": "time2",
    "case_attributes": ("org",),
    "exposure_col": "amount",
    "order_col": "seq",
    "event_id_col": "eid",
    "lifecycle_col": "lc",
    "keep_transitions": ("start",),
    "keep_columns": ("amount",),
    "timestamp_format": "%Y",
    "missing_timestamps": MissingTimestamps.KEEP,
    "untimed_events": UntimedEvents.UNIFORM,
    "duplicates": Duplicates.DROP,
    "time_zone": TimeZone.UTC,
    "date_order": DateOrder.DAY_FIRST,
    "exposure_agg": ExposureAggregation.SUM,
    "case_attribute_values": CaseAttributeValues.STRICT,
}


@pytest.mark.spec("decision:0001")
def test_the_digest_is_stable_text() -> None:
    digest = SCHEMA.digest
    assert len(digest) == 64
    assert int(digest, 16) >= 0
    assert wise.LogSchema(**COLS).digest == digest
    assert wise.LogSchema(**COLS, missing_timestamps="raise").digest == digest


@pytest.mark.spec("decision:0001")
def test_every_role_and_policy_has_an_alternative_here() -> None:
    names = {f.name for f in dataclasses.fields(wise.LogSchema)}
    single_valued = {name for name, (enum, _) in POLICIES.items() if len(enum) == 1}
    assert names - single_valued == set(ALTERNATIVES)


@pytest.mark.spec("decision:0001")
@pytest.mark.parametrize("field", sorted(ALTERNATIVES))
def test_the_digest_changes_with_every_column_role_and_policy(field: str) -> None:
    changed = dataclasses.replace(SCHEMA, **{field: ALTERNATIVES[field]})
    assert changed != SCHEMA
    assert changed.digest != SCHEMA.digest


@pytest.mark.spec("decision:0001")
def test_the_digest_ignores_spellings_of_the_same_schema() -> None:
    a = wise.LogSchema(**COLS, case_attributes="org", keep_transitions=["START", "complete"], keep_columns=["x", "amount"])
    b = wise.LogSchema(
        **COLS, case_attributes=["org"], keep_transitions=("complete", "start", "Start"), keep_columns=("amount", "x")
    )
    assert a == b
    assert a.digest == b.digest
    # the order of case attributes is the column order of the case table: a different role
    c = wise.LogSchema(**COLS, case_attributes=["org", "region"])
    assert c.digest != wise.LogSchema(**COLS, case_attributes=["region", "org"]).digest


_schemas = st.builds(
    wise.LogSchema,
    case_col=st.sampled_from(["case", "c"]),
    case_attributes=st.lists(st.sampled_from(["org", "region"]), max_size=2, unique=True),
    exposure_col=st.sampled_from([None, "amount"]),
    lifecycle_col=st.sampled_from([None, "lc"]),
    keep_transitions=st.lists(st.sampled_from(["complete", "Complete", "start"]), max_size=2),
    keep_columns=st.sampled_from([None, (), ("amount",)]),
    missing_timestamps=st.sampled_from(MissingTimestamps),
    untimed_events=st.sampled_from(UntimedEvents),
    case_attribute_values=st.sampled_from(CaseAttributeValues),
    duplicates=st.sampled_from(Duplicates),
    exposure_agg=st.sampled_from([ExposureAggregation.MAX, ExposureAggregation.SUM]),
)


@pytest.mark.spec("decision:0001")
@given(_schemas, _schemas)
def test_two_schemas_share_a_digest_exactly_when_they_are_equal(a: wise.LogSchema, b: wise.LogSchema) -> None:
    assert (a.digest == b.digest) == (a == b)


# ------------------------------------------------------------------ parse
@pytest.mark.spec("IV-A")
def test_parse_builds_events_index_and_case_table() -> None:
    schema = wise.LogSchema(**COLS, case_attributes="org", exposure_col="amount", exposure_agg="sum", lifecycle_col="lc")
    parsed = wise.parse(_frame(), schema)
    assert parsed.schema is schema
    # the start transition of b is dropped; events are sorted by (case, time)
    assert parsed.events["case"].tolist() == ["a", "a", "b"]
    assert parsed.events["activity"].tolist() == ["Y", "Y", "X"]
    assert list(parsed.index.case_codes) == [0, 0, 1]
    assert parsed.index.activity_labels == ("Y", "X")
    assert list(parsed.cases.index) == ["a", "b"]
    assert parsed.cases["n_events"].tolist() == [2, 1]
    assert parsed.cases["exposure"].tolist() == [10.0, 2.0]
    assert parsed.cases["org"].tolist() == ["m", "n"]
    assert parsed.cases["first_ts"].tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert parsed.tz is None
    assert (parsed.n_missing_timestamps, parsed.n_null_activities) == (0, 0)


@pytest.mark.spec("IV-A")
def test_policies_are_read_from_the_schema_not_from_the_data() -> None:
    frame = _frame()
    kept = wise.parse(frame, SCHEMA)
    dropped = wise.parse(frame, dataclasses.replace(SCHEMA, duplicates=Duplicates.DROP))
    assert (len(kept.events), len(dropped.events)) == (4, 3)
    aware = frame.assign(time=frame["time"].dt.tz_localize("Europe/Berlin"))
    assert str(wise.parse(aware, SCHEMA).tz) == "Europe/Berlin"
    assert str(wise.parse(aware, dataclasses.replace(SCHEMA, time_zone=TimeZone.UTC)).tz) == "UTC"
    assert wise.parse(aware, SCHEMA).schema == SCHEMA


@pytest.mark.spec("IV-A")
def test_dropping_duplicates_keeps_the_same_event_whatever_the_row_order() -> None:
    # rows 1 and 3 agree in case, activity and time and differ in amount (7 and 5) and lc ('COMPLETE', 'complete');
    # the first other column in name order that differs is amount, so the row with amount 5 survives
    frame = _frame().assign(amount=[2.0, 7.0, 3.0, 5.0])
    schema = wise.LogSchema(**COLS, exposure_col="amount", exposure_agg="sum")
    kept = wise.parse(frame, schema)
    drop = dataclasses.replace(schema, duplicates=Duplicates.DROP)
    assert kept.events["amount"].tolist() == [7.0, 5.0, 2.0, 3.0]
    assert kept.cases["exposure"].tolist() == [12.0, 5.0]
    for rows in ([0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2]):
        dropped = wise.parse(frame.iloc[rows], drop)
        assert dropped.events["amount"].tolist() == [5.0, 2.0, 3.0]
        assert dropped.events["lc"].tolist() == ["complete", "complete", "start"]
        assert dropped.cases["exposure"].tolist() == [5.0, 5.0]
        assert dropped.cases["n_events"].tolist() == [1, 2]


@pytest.mark.spec("IV-A")
def test_the_order_column_chooses_the_survivor_before_the_other_columns() -> None:
    # seq puts the amount-7 row of case a first, although its amount is the larger one
    frame = _frame().assign(amount=[2.0, 7.0, 3.0, 5.0], seq=[0, 1, 0, 2])
    schema = wise.LogSchema(**COLS, order_col="seq", duplicates="drop")
    for rows in ([0, 1, 2, 3], [3, 2, 1, 0]):
        assert wise.parse(frame.iloc[rows], schema).events["amount"].tolist() == [7.0, 2.0, 3.0]
    # equal order values fall back to the other columns: amount 5 survives
    tied = frame.assign(seq=[0, 1, 0, 1])
    for rows in ([0, 1, 2, 3], [3, 2, 1, 0]):
        assert wise.parse(tied.iloc[rows], schema).events["amount"].tolist() == [5.0, 2.0, 3.0]


@pytest.mark.spec("IV-A")
def test_values_that_do_not_compare_still_choose_one_survivor() -> None:
    # rows 1 and 3 of the frame are equal outside the column added here, which alone decides
    # equal values of different types rank by type name: 'builtins.bool:True' < 'builtins.int:1'
    # nested values do not compare and rank by repr: "builtins.list:['p']" < "builtins.list:['q']"
    pair = _frame().iloc[[1, 3]].reset_index(drop=True).assign(lc="complete")
    schema = dataclasses.replace(SCHEMA, duplicates=Duplicates.DROP)
    for column, values, survivor in (("flag", [1, True], True), ("tags", [["q"], ["p"]], ["p"])):
        frame = pair.assign(**{column: pd.Series(values, dtype=object)})
        for rows in ([0, 1], [1, 0]):
            assert wise.parse(frame.iloc[rows], schema).events[column].tolist() == [survivor]


@pytest.mark.spec("IV-A")
def test_equal_values_with_other_bits_and_names_with_one_spelling_still_choose_one_survivor() -> None:
    # 0.0 and -0.0 are equal floats that the fingerprint tells apart: 0.0 survives in any row order
    pair = _frame().iloc[[1, 3]].reset_index(drop=True).assign(lc="complete", amount=[-0.0, 0.0])
    schema = dataclasses.replace(SCHEMA, duplicates=Duplicates.DROP)
    logs = [wise.EventLog(pair.iloc[rows], schema=schema) for rows in ([0, 1], [1, 0])]
    assert [np.signbit(log.events["amount"]).tolist() for log in logs] == [[False], [False]]
    assert logs[0].fingerprint() == logs[1].fingerprint()
    # columns 1 and "1" read alike; the type name orders them (builtins.int < builtins.str) in any column order,
    # so the row with the smaller value in column 1 survives
    named = pair.drop(columns="amount").assign(**{"1": [5, 6]})
    named[1] = [8, 7]
    for columns in (list(named.columns), list(named.columns)[::-1]):
        assert wise.parse(named[columns], schema).events[[1, "1"]].values.tolist() == [[7, 6]]


@pytest.mark.spec("IV-A")
def test_the_parsed_record_counts_the_dropped_events_and_their_cases() -> None:
    # case a: one duplicate; case b: two events at 2024-01-02 duplicate the first one; c: none
    extra = pd.DataFrame(
        {
            "case": ["b", "b", "c"],
            "activity": ["X", "X", "X"],
            "time": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
            "lc": ["complete"] * 3,
            "amount": [1.0, 1.0, 1.0],
            "org": ["n", "n", "o"],
        }
    )
    frame = pd.concat([_frame(), extra], ignore_index=True)
    dropped = wise.parse(frame, dataclasses.replace(SCHEMA, duplicates=Duplicates.DROP))
    assert (dropped.n_dropped_duplicates, dropped.n_cases_with_dropped_duplicates) == (3, 2)
    assert len(dropped.events) == 7 - 3
    kept = wise.parse(frame, SCHEMA)
    assert (kept.n_dropped_duplicates, kept.n_cases_with_dropped_duplicates) == (0, 0)
    clean = wise.parse(_frame().iloc[:3], dataclasses.replace(SCHEMA, duplicates=Duplicates.DROP))
    assert (clean.n_dropped_duplicates, clean.n_cases_with_dropped_duplicates) == (0, 0)


@pytest.mark.spec("IV-A")
def test_validate_reports_the_dropped_duplicates_when_the_log_drops_them() -> None:
    log = wise.EventLog(_frame(), **COLS, dedupe=True)
    report = log.validate()
    assert (report["dropped_duplicates"], report["cases_with_dropped_duplicates"]) == (1, 1)
    assert (log.n_dropped_duplicates, log.n_cases_with_dropped_duplicates) == (1, 1)
    clean = wise.EventLog(_frame().iloc[:3], **COLS, dedupe=True).validate()
    assert (clean["dropped_duplicates"], clean["cases_with_dropped_duplicates"]) == (0, 0)
    kept = wise.EventLog(_frame(), **COLS)
    assert (kept.n_dropped_duplicates, kept.n_cases_with_dropped_duplicates) == (0, 0)
    assert "dropped_duplicates" not in kept.validate().index
    assert "cases_with_dropped_duplicates" not in kept.validate().index


@pytest.mark.spec("IV-A")
def test_parse_records_case_attributes_that_vary_within_a_case() -> None:
    # case b says 'n' then 'o', case a says 'm' twice; a null never counts as a second value
    frame = _frame().assign(org=pd.Series(["n", "m", "o", None]), region=["r", "r", "r", "r"])
    parsed = wise.parse(frame, dataclasses.replace(SCHEMA, case_attributes=("region", "org")))
    assert dict(parsed.varying_case_attributes) == {"org": 1}
    assert parsed.cases["org"].tolist() == ["m", "n"]
    with pytest.raises(TypeError):
        parsed.varying_case_attributes["org"] = 2  # type: ignore[index]
    assert dict(wise.parse(_frame(), dataclasses.replace(SCHEMA, case_attributes=("org",))).varying_case_attributes) == {}


@pytest.mark.spec("IV-A")
def test_the_strict_policy_refuses_varying_case_attributes_by_name() -> None:
    frame = _frame().assign(org=pd.Series(["n", "m", "o", None]), region=["r", "s", "t", "u"])
    strict = dataclasses.replace(SCHEMA, case_attributes=("org", "region"), case_attribute_values=CaseAttributeValues.STRICT)
    with pytest.raises(LogSchemaError, match=r"vary within a case: 'org' \(1 case\), 'region' \(2 cases\)"):
        wise.parse(frame, strict)
    assert wise.parse(_frame(), dataclasses.replace(strict, case_attributes=("org",))).cases["org"].tolist() == ["m", "n"]


def test_parse_refuses_a_frame_without_the_named_columns() -> None:
    with pytest.raises(LogSchemaError, match=r"missing columns \['lc'\]"):
        wise.parse(_frame().drop(columns="lc"), dataclasses.replace(SCHEMA, lifecycle_col="lc"))


@pytest.mark.spec("IV-A")
def test_parse_refuses_a_log_without_events() -> None:
    with pytest.raises(LogSchemaError, match="event frame has no rows"):
        wise.parse(_frame().iloc[:0], SCHEMA)
    with pytest.raises(LogSchemaError, match=r"no event of lifecycle column 'lc' has a transition in \['end'\]"):
        wise.parse(_frame(), dataclasses.replace(SCHEMA, lifecycle_col="lc", keep_transitions=("end",)))
    no_time = _frame().assign(time=pd.Series(pd.NaT, index=range(4), dtype="datetime64[ns]"))
    with pytest.raises(LogSchemaError, match="every event has a null or unparseable timestamp"):
        wise.parse(no_time, dataclasses.replace(SCHEMA, missing_timestamps=MissingTimestamps.DROP))


@pytest.mark.spec("IV-A")
def test_parse_refuses_null_lifecycle_transitions() -> None:
    frame = _frame()
    frame.loc[2, "lc"] = None
    with pytest.raises(LogSchemaError, match=r"lifecycle column 'lc' has 1 null transition \(first at row 2\)"):
        wise.parse(frame, dataclasses.replace(SCHEMA, lifecycle_col="lc"))
    assert len(wise.parse(frame, SCHEMA).events) == 4  # without a lifecycle role the column is plain data


# ------------------------------------------------------------------ the EventLog façade
@pytest.mark.spec("IV-A")
def test_the_log_carries_its_schema_and_accepts_one() -> None:
    log = wise.EventLog(_frame(), **COLS, case_attributes="org")
    assert log.schema == wise.LogSchema(**COLS, case_attributes=("org",))
    schema = wise.LogSchema(**COLS, lifecycle_col="lc")
    from_schema = wise.EventLog(_frame(), schema=schema, window=("2024-01-01", None))
    assert from_schema.schema is schema
    assert len(from_schema.events) == 3
    assert from_schema.window == (pd.Timestamp("2024-01-01"), None)
    assert from_schema.case_attributes == []


def test_a_schema_and_column_or_policy_options_together_are_refused() -> None:
    with pytest.raises(LogSchemaError, match="pass either schema= or the column and policy options"):
        wise.EventLog(_frame(), schema=SCHEMA, dedupe=True)


@pytest.fixture
def parse_calls(monkeypatch: pytest.MonkeyPatch) -> list[wise.LogSchema]:
    """Record every schema the event log hands to the parse step."""
    calls: list[wise.LogSchema] = []
    real = wise.parse

    def spy(events: pd.DataFrame, schema: wise.LogSchema) -> wise.ParsedLog:
        calls.append(schema)
        return real(events, schema)

    monkeypatch.setattr("wise.log.parse", spy)
    return calls


@pytest.mark.spec("IV-A")
def test_from_csv_goes_through_the_parse_step(tmp_path: Path, parse_calls: list[wise.LogSchema]) -> None:
    path = tmp_path / "log.csv"
    _frame().to_csv(path, index=False)
    log = wise.EventLog.from_csv(str(path), **COLS, lifecycle_col="lc")
    assert parse_calls == [wise.LogSchema(**COLS, lifecycle_col="lc")]
    assert log.events["case"].tolist() == ["a", "a", "b"]
    frame = _frame()
    frame.loc[0, "lc"] = None
    frame.to_csv(path, index=False)
    with pytest.raises(LogSchemaError, match="null transition"):
        wise.EventLog.from_csv(str(path), schema=wise.LogSchema(**COLS, lifecycle_col="lc"))


@pytest.mark.spec("IV-A")
def test_from_pm4py_goes_through_the_parse_step(monkeypatch: pytest.MonkeyPatch, parse_calls: list[wise.LogSchema]) -> None:
    wise.EventLog.from_pm4py(_frame(), **COLS)
    fake = types.ModuleType("pm4py")
    fake.convert_to_dataframe = lambda obj: _frame()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pm4py", fake)
    wise.EventLog.from_pm4py(object(), schema=SCHEMA)
    assert parse_calls == [SCHEMA, SCHEMA]
    with pytest.raises(LogSchemaError, match="no rows"):
        wise.EventLog.from_pm4py(_frame().iloc[:0], **COLS)


@pytest.mark.spec("IV-A")
def test_from_xes_goes_through_the_parse_step(monkeypatch: pytest.MonkeyPatch, parse_calls: list[wise.LogSchema]) -> None:
    def read_xes(path: str) -> pd.DataFrame:
        return _frame().rename(columns={"case": CASE_COL, "activity": ACTIVITY_COL, "time": TIMESTAMP_COL})

    fake = types.ModuleType("pm4py")
    fake.read_xes = read_xes  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pm4py", fake)
    log = wise.EventLog.from_xes("log.xes", lifecycle_col="lc")
    assert parse_calls == [wise.LogSchema(lifecycle_col="lc")]
    assert np.array_equal(log.cases["n_events"].to_numpy(), [2, 1])
