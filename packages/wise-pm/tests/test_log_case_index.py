"""EventLog on top of its case index: public timestamp conversion, the 0.1.0 path and the read-only arrays."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import wise
from wise.case_index import CaseIndex


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


def test_the_arrays_read_by_0_1_0_callers_cannot_be_written() -> None:
    log = _log()
    np.testing.assert_array_equal(log._codes, [0, 0, 1, 1])
    for arr in (log._codes, log._ts, log._valid, log._starts):
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = arr[-1]
    with pytest.raises(AttributeError):
        log.case_index._codes = np.zeros(4, dtype=np.int64)
    np.testing.assert_array_equal(log.count("X").to_numpy(), [1.0, 1.0])
