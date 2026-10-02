"""The ``EventLog`` constructor over the parse step builds the 0.1.0 log for every option.

Each example draws an event frame (string, integer or categorical case ids;
null and categorical activities; naive, aware, string and unparseable
timestamps; lifecycle transitions; ties; exact duplicates and events equal
only in case, activity and time; non-numeric and negative exposure) and a
combination of the constructor's options, then builds the log both ways: through ``EventLog`` (and through ``parse`` with the matching
``LogSchema``), and through the frozen 0.1.0 construction in
``_support.legacy_log``. The two agree exactly, except where 0.1.0 accepted
input silently and wrongly: an event frame without rows, a lifecycle column
with nulls, and a log that loses every event to its filters are refused.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, note
from hypothesis import strategies as st

import wise
from _support.legacy_log import LegacyLog, legacy_build
from wise.errors import LogSchemaError
from wise.log_schema import ExposureAggregation

pytestmark = pytest.mark.property

COLS = {"case_col": "case", "activity_col": "activity", "timestamp_col": "time"}
BASE = pd.Timestamp("2024-03-01 08:00")


def _frame(rows: list[dict[str, Any]], case_kind: str, act_kind: str, ts_kind: str, amount_kind: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["case", "activity", "day", "lc", "seq", "eid", "amount", "attr", "extra"])
    if case_kind == "int":
        df["case"] = df["case"].map(lambda c: None if pd.isna(c) else int(c[1:]))
    elif case_kind == "category":
        df["case"] = pd.Categorical(df["case"], categories=["c0", "c1", "c2", "c3", "c9"])
    if act_kind == "category":
        df["activity"] = pd.Categorical(df["activity"], categories=["A", "B", "C", "Z"])
    time = BASE + pd.to_timedelta(df["day"].astype(float), unit="D")
    if ts_kind == "aware":
        time = time.dt.tz_localize("Europe/Amsterdam")
    elif ts_kind == "iso":
        time = time.dt.strftime("%Y-%m-%d %H:%M").where(time.notna(), None).astype(object)
    elif ts_kind == "dmy":
        time = time.dt.strftime("%d/%m/%Y").where(time.notna(), "not a date").astype(object)
    df["time"] = time
    if amount_kind == "text":
        df["amount"] = df["amount"].astype(object)
        df.loc[df.index[::3], "amount"] = "n/a"
    return df.drop(columns="day")


_row = st.fixed_dictionaries(
    {
        "case": st.sampled_from(["c0", "c1", "c2", "c3"]),
        "activity": st.sampled_from(["A", "B", "C", None]),
        "day": st.integers(0, 6).map(lambda v: v / 2),
        "lc": st.sampled_from(["complete", "complete", "COMPLETE", "start"]),
        "seq": st.integers(0, 3),
        "eid": st.integers(0, 4),
        "amount": st.integers(-1, 8).map(lambda v: v / 4),
        "attr": st.sampled_from(["p", "q", None]),
        "extra": st.integers(0, 2),
    }
)


@st.composite
def frames(draw: st.DrawFn) -> pd.DataFrame:
    rows = draw(st.lists(_row, min_size=0, max_size=12))
    for column, null in (("case", None), ("day", float("nan")), ("lc", None)):
        if rows and draw(st.integers(0, 4)) == 0:
            rows[draw(st.integers(0, len(rows) - 1))][column] = null
    if draw(st.booleans()) and rows:
        rows = rows + rows[: draw(st.integers(0, len(rows)))]  # exact duplicates
    for twin in draw(st.lists(_row, max_size=3)) if rows else []:
        source = rows[draw(st.integers(0, len(rows) - 1))]  # equal in case, activity and time only
        rows.append({**twin, "case": source["case"], "activity": source["activity"], "day": source["day"]})
    kinds = (
        draw(st.sampled_from(["str", "int", "category"])),
        draw(st.sampled_from(["object", "category"])),
        draw(st.sampled_from(["naive", "aware", "iso", "dmy"])),
        draw(st.sampled_from(["float", "text"])),
    )
    note(f"kinds: {kinds}")
    return _frame(rows, *kinds)


@st.composite
def options(draw: st.DrawFn) -> dict[str, Any]:
    opts: dict[str, Any] = dict(COLS)
    pick = draw
    if pick(st.booleans()):
        opts["case_attributes"] = pick(st.sampled_from(["attr", ["attr"], (), ["extra", "attr"]]))
    if pick(st.booleans()):
        opts["exposure_col"] = "amount"
    if pick(st.booleans()):
        opts["exposure_agg"] = pick(st.sampled_from([a.value for a in ExposureAggregation]))
    roles = (("order_col", "seq"), ("event_id_col", "eid"), ("lifecycle_col", "lc"))
    opts.update({key: column for key, column in roles if pick(st.booleans())})
    if pick(st.booleans()):
        opts["keep_transitions"] = pick(st.sampled_from(["complete", ("Start", "complete"), ["START"], ()]))
    if pick(st.booleans()):
        opts["utc"] = pick(st.booleans())
    if pick(st.booleans()):
        opts["timestamp_format"] = pick(st.sampled_from(["%d/%m/%Y", "%Y-%m-%d %H:%M"]))
    if pick(st.booleans()):
        opts["dayfirst"] = pick(st.booleans())
    if pick(st.booleans()):
        opts["missing_timestamps"] = pick(st.sampled_from(["raise", "drop", "keep"]))
    if pick(st.booleans()):
        opts["keep_columns"] = pick(st.sampled_from([None, "extra", ["seq", "attr"], []]))
    if pick(st.booleans()):
        opts["dedupe"] = pick(st.booleans())
    note(f"options: {opts}")
    return opts


def _assert_same(log: wise.EventLog | wise.ParsedLog, old: LegacyLog) -> None:
    pd.testing.assert_frame_equal(log.events, old.events, check_exact=True)
    pd.testing.assert_frame_equal(log.cases, old.cases, check_exact=True)
    new_index, old_index = (log.case_index if isinstance(log, wise.EventLog) else log.index), old.index
    for name in ("case_codes", "timestamps", "valid", "activity_codes", "starts", "sizes"):
        np.testing.assert_array_equal(getattr(new_index, name), getattr(old_index, name), err_msg=name)
    assert new_index.activity_labels == old_index.activity_labels
    assert new_index.n_cases == old_index.n_cases
    assert str(log.tz) == str(old.tz)
    assert log.n_missing_timestamps == old.n_missing_timestamps
    assert log.n_null_activities == old.n_null_activities


@given(frames(), options())
def test_the_constructor_builds_the_0_1_0_log_or_refuses_what_0_1_0_accepted_wrongly(
    frame: pd.DataFrame, opts: dict[str, Any]
) -> None:
    try:
        old: LegacyLog | None = legacy_build(frame, **opts)
        old_error: Exception | None = None
    except LogSchemaError as exc:
        old, old_error = None, exc

    lifecycle_nulls = "lifecycle_col" in opts and bool(frame["lc"].isna().any())
    refused_now = len(frame) == 0 or lifecycle_nulls or (old is not None and len(old.events) == 0)

    if old_error is not None or refused_now:
        with pytest.raises(LogSchemaError):
            wise.EventLog(frame, **opts)
        return
    assert old is not None
    log = wise.EventLog(frame, **opts)
    _assert_same(log, old)
    assert log.case_attributes == list(wise.LogSchema.from_options(**opts).case_attributes)
    assert log.schema == wise.LogSchema.from_options(**opts)
    _assert_same(wise.parse(frame, log.schema), old)
