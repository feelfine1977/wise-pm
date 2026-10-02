"""EventLog on top of its case index: public timestamp conversion, the 0.1.0 path and the read-only arrays."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import wise
from wise.case_index import CaseIndex
from wise.log import EventLog


def _log(tz: str | None = None) -> wise.EventLog:
    df = pd.DataFrame(
        {
            "case": ["b", "a", "a", "b"],
            "activity": ["X", "Y", "X", "Y"],
            "time": pd.to_datetime(pd.Series(["2024-01-03", "2024-01-02", "2024-01-01", None])),
        }
    )
    if tz is not None:
        df["time"] = df["time"].dt.tz_localize(tz)
    return wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="time", missing_timestamps="keep")


@pytest.mark.spec("IV-A")
def test_to_timestamp_matches_the_log_time_zone() -> None:
    naive, aware = _log(), _log("Europe/Berlin")
    assert naive.to_timestamp(None) is None
    assert naive.to_timestamp("2024-01-02") == pd.Timestamp("2024-01-02")
    assert naive.to_timestamp(pd.Timestamp("2024-01-02 01:00", tz="Europe/Berlin")) == pd.Timestamp("2024-01-02")
    t = aware.to_timestamp("2024-01-02 01:00")
    assert t == pd.Timestamp("2024-01-02 00:00", tz="UTC")
    assert str(t.tzinfo) == "Europe/Berlin"  # type: ignore[union-attr]
    assert aware.window_end == pd.Timestamp("2024-01-03", tz="Europe/Berlin")


@pytest.mark.spec("decision:0001")
def test_the_private_conversion_still_works_with_a_deprecation_warning() -> None:
    log = _log("Europe/Berlin")
    with pytest.warns(DeprecationWarning, match="EventLog.to_timestamp"):
        old = log._to_ts("2024-01-02 01:00")
    assert old == log.to_timestamp("2024-01-02 01:00")


@pytest.mark.spec("IV-A")
def test_the_case_index_is_aligned_with_events_and_cases() -> None:
    log = _log()
    idx = log.case_index
    assert isinstance(idx, CaseIndex)
    assert (idx.n_cases, idx.n_events) == (len(log), len(log.events))
    assert list(log.case_ids[idx.case_codes]) == list(log.events["case"])  # events are in index order
    assert list(log.events["activity"]) == ["X", "Y", "X", "Y"]  # a: day 1, day 2; b: day 3, no timestamp
    assert idx.activity_labels == tuple(log.activity_labels)
    np.testing.assert_array_equal(idx.sizes, log.cases["n_events"].to_numpy())


def _read_with_warning(log: EventLog, name: str, replacement: str) -> np.ndarray:
    with pytest.warns(DeprecationWarning, match=rf"EventLog\.{name} .* use EventLog\.case_index\.{replacement}") as seen:
        arr = getattr(log, name)
    assert len(seen) == 1
    assert isinstance(arr, np.ndarray)
    return arr


INDEX_ARRAYS = {"_codes": "case_codes", "_ts": "timestamps", "_valid": "valid", "_starts": "starts"}


@pytest.mark.spec("decision:0001/2")
@pytest.mark.parametrize(("name", "replacement"), INDEX_ARRAYS.items())
def test_the_arrays_read_by_0_1_0_callers_warn_and_equal_the_case_index(name: str, replacement: str) -> None:
    log = _log()
    np.testing.assert_array_equal(_read_with_warning(log, name, replacement), getattr(log.case_index, replacement))


def test_the_arrays_read_by_0_1_0_callers_cannot_be_written() -> None:
    log = _log()
    arrays = [_read_with_warning(log, name, replacement) for name, replacement in INDEX_ARRAYS.items()]
    np.testing.assert_array_equal(arrays[0], [0, 0, 1, 1])
    for arr in arrays:
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = arr[-1]
    for arr in (*arrays, log.case_index.activity_codes, log.case_index.sizes):
        assert arr.base is None or not arr.base.flags.writeable  # no writable array behind the view
    with pytest.raises(AttributeError):
        log.case_index._codes = np.zeros(4, dtype=np.int64)
    np.testing.assert_array_equal(log.count("X").to_numpy(), [1.0, 1.0])


@pytest.mark.spec("IV-A")
def test_total_keeps_the_dtype_0_1_0_returned() -> None:
    # case a has X and Y, case b only X: a count over X covers every case and stays integer,
    # a count over Y leaves case b empty and becomes float with 0 there
    df = pd.DataFrame(
        {
            "case": ["a", "a", "b"],
            "activity": ["X", "Y", "X"],
            "time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "cost": [5.0, 7.0, 2.0],
        }
    )
    log = wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="time")
    over_x = log.total("cost", "X", agg="count")  # type: ignore[arg-type]  # accepted at run time, as in 0.1.0
    assert over_x.dtype == np.int64
    assert over_x.tolist() == [1, 1]
    over_y = log.total("cost", "Y", agg="count")  # type: ignore[arg-type]
    assert over_y.dtype == np.float64
    assert over_y.tolist() == [1.0, 0.0]
    assert log.total("cost", "X").tolist() == [5.0, 2.0]
