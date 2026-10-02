"""The case index: the immutable per-case kernel behind the event log (paper Sec. IV-A).

Every expectation below is derived by hand from the small logs built in this
module; nothing goes through :class:`wise.EventLog`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from wise.case_index import NAT, CaseIndex, event_order
from wise.errors import LogSchemaError, WiseError

DAY = 86_400_000_000_000
LABELS = ("A", "B", "C")


def _days(*values: int | None) -> np.ndarray:
    return np.array([NAT if v is None else v * DAY for v in values], dtype=np.int64)


def small_index(tz: Any = None) -> CaseIndex:
    """Three cases, events already in (case, timestamp) order.

    ======  ========  =========
    case    activity  timestamp
    ======  ========  =========
    0       A         day 1
    0       B         day 2
    0       B         day 2
    0       C         none
    1       B         day 5
    1       A         day 7
    1       (none)    day 7
    2       C         none
    ======  ========  =========
    """
    return CaseIndex(
        np.array([0, 0, 0, 0, 1, 1, 1, 2]),
        _days(1, 2, 2, None, 5, 7, 7, None),
        np.array([0, 1, 1, 2, 1, 0, -1, 2]),
        LABELS,
        tz=tz,
    )


def _eq(actual: np.ndarray, expected: list[Any]) -> None:
    np.testing.assert_array_equal(actual, np.array(expected))


# ------------------------------------------------------------------ structure
@pytest.mark.spec("IV-A")
def test_structure_of_the_index() -> None:
    idx = small_index()
    assert (idx.n_cases, idx.n_events) == (3, 8)
    assert idx.activity_labels == LABELS
    _eq(idx.case_codes, [0, 0, 0, 0, 1, 1, 1, 2])
    _eq(idx.activity_codes, [0, 1, 1, 2, 1, 0, 3, 2])  # an event without a label gets code len(labels)
    _eq(idx.valid, [True, True, True, False, True, True, True, False])
    _eq(idx.starts, [0, 4, 7])
    _eq(idx.sizes, [4, 3, 1])
    assert idx.timestamps[3] == NAT
    assert idx.tz is None
    assert "3 cases" in repr(idx)


def test_a_case_without_events_has_an_empty_slice() -> None:
    idx = CaseIndex(np.array([0, 2]), _days(1, 2), np.array([0, 0]), ("A",), n_cases=3)
    _eq(idx.starts, [0, 1, 1])
    _eq(idx.sizes, [1, 0, 1])
    _eq(idx.count("A"), [1, 0, 1])
    _eq(idx.first("A"), [DAY, NAT, 2 * DAY])


def test_an_empty_index() -> None:
    idx = CaseIndex(np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64), ())
    assert (idx.n_cases, idx.n_events) == (0, 0)
    assert len(idx.count("A")) == 0
    assert len(idx.run_starts()) == 0


# ------------------------------------------------------------------ counts
@pytest.mark.spec("IV-A")
@pytest.mark.parametrize(
    ("select", "expected"),
    [
        ("B", [2, 1, 0]),  # a bare string is one label
        (["A"], [1, 1, 0]),
        (("A", "C"), [2, 1, 1]),
        ({"C"}, [1, 0, 1]),  # events without a timestamp still count
        (["Z"], [0, 0, 0]),  # an unknown label matches nothing
        (["A", "Z"], [1, 1, 0]),
        (None, [4, 3, 1]),  # every event, including the unlabelled one
    ],
)
def test_count_of_an_activity_set(select: Any, expected: list[int]) -> None:
    _eq(small_index().count(select), expected)


@pytest.mark.spec("IV-A")
def test_count_of_an_event_mask() -> None:
    idx = small_index()
    _eq(idx.count(idx.valid), [3, 3, 0])
    _eq(idx.count(idx.mask("B") & (idx.timestamps >= 2 * DAY)), [2, 1, 0])
    _eq(idx.mask(["A", "C"]), [True, False, False, True, False, True, False, True])


def test_a_mask_of_the_wrong_shape_or_type_is_refused() -> None:
    idx = small_index()
    with pytest.raises(LogSchemaError, match="one entry per event"):
        idx.count(np.ones(3, dtype=bool))
    with pytest.raises(LogSchemaError, match="boolean"):
        idx.count(np.ones(8, dtype=np.int64))
    assert issubclass(LogSchemaError, WiseError)


# ------------------------------------------------------------------ first and last
@pytest.mark.spec("IV-A")
@pytest.mark.parametrize(
    ("select", "first", "last"),
    [
        ("A", [1, 7, None], [1, 7, None]),
        ("B", [2, 5, None], [2, 5, None]),
        (["A", "B"], [1, 5, None], [2, 7, None]),
        ("C", [None, None, None], [None, None, None]),  # only events without a timestamp
        (None, [1, 5, None], [2, 7, None]),
    ],
)
def test_first_and_last_timestamps_ignore_events_without_one(
    select: Any, first: list[int | None], last: list[int | None]
) -> None:
    idx = small_index()
    _eq(idx.first(select), list(_days(*first)))
    _eq(idx.last(select), list(_days(*last)))


def test_first_of_an_event_mask() -> None:
    idx = small_index()
    anchor = np.array([2 * DAY, 6 * DAY, 0], dtype=np.int64)
    after = idx.timestamps >= anchor[idx.case_codes]
    _eq(idx.first(after & idx.valid), list(_days(2, 7, None)))
    _eq(idx.last(~idx.valid), list(_days(None, None, None)))


# ------------------------------------------------------------------ aggregates
VALUES = np.array([10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 99.0])


@pytest.mark.spec("IV-A")
@pytest.mark.parametrize(
    ("select", "how", "empty", "expected"),
    [
        ("B", "sum", np.nan, [50.0, 1.0, np.nan]),
        ("B", "sum", 0.0, [50.0, 1.0, 0.0]),
        (None, "sum", 0.0, [100.0, 6.0, 99.0]),
        (None, "max", np.nan, [40.0, 3.0, 99.0]),
        ("A", "first", 0.0, [10.0, 2.0, 0.0]),
        (["A", "B"], "last", 0.0, [30.0, 2.0, 0.0]),
        (["A", "B"], "mean", np.nan, [20.0, 1.5, np.nan]),
        (["A", "B"], "min", -1.0, [10.0, 1.0, -1.0]),
    ],
)
def test_per_case_aggregate_of_an_event_column(select: Any, how: str, empty: float, expected: list[float]) -> None:
    _eq(small_index().aggregate(VALUES, select, how=how, empty=empty), expected)


def test_aggregate_skips_missing_values_and_accepts_a_series_and_a_mask() -> None:
    idx = small_index()
    values = pd.Series([10.0, np.nan, 30.0, 40.0, np.nan, 2.0, 3.0, 99.0], index=list("abcdefgh"))
    _eq(idx.aggregate(values, "B"), [30.0, 0.0, np.nan])  # a sum of missing values is 0
    _eq(idx.aggregate(values, "B", how="mean"), [30.0, np.nan, np.nan])
    _eq(idx.aggregate(values, idx.valid, how="count"), [2.0, 2.0, np.nan])
    labels = pd.Series(["x", "y", "x", None, "z", "z", "z", "w"])
    _eq(idx.aggregate(labels, None, how="nunique", empty=0.0), [2.0, 1.0, 1.0])


def test_aggregate_refuses_a_column_of_the_wrong_length() -> None:
    with pytest.raises(LogSchemaError, match="one entry per event"):
        small_index().aggregate(np.ones(3), None)


# ------------------------------------------------------------------ runs of equal timestamps
@pytest.mark.spec("IV-A")
def test_runs_of_equal_timestamps() -> None:
    idx = small_index()
    # position of the first event sharing (case, timestamp); events without a timestamp form one run per case
    _eq(idx.run_starts(), [0, 1, 1, 3, 4, 5, 5, 7])
    # tied: shares its case and a timestamp with another event; missing timestamps never tie
    _eq(idx.tied(), [False, True, True, False, False, True, True, False])
    two_missing = CaseIndex(np.array([0, 0]), _days(None, None), np.array([0, 0]), ("A",))
    _eq(two_missing.run_starts(), [0, 0])
    _eq(two_missing.tied(), [False, False])


# ------------------------------------------------------------------ timestamp conversion
@pytest.mark.spec("IV-A")
def test_timestamp_conversion_on_a_naive_index() -> None:
    idx = small_index()
    assert idx.to_timestamp(None) is None
    assert idx.to_timestamp("2024-01-02") == pd.Timestamp("2024-01-02")
    assert idx.to_timestamp(pd.Timestamp("2024-01-02 01:00", tz="Europe/Berlin")) == pd.Timestamp("2024-01-02 00:00")
    assert idx.to_ns("1970-01-02") == DAY
    with pytest.raises(LogSchemaError, match="timestamp is required"):
        idx.to_ns(None)
    out = idx.to_datetime(_days(1, None))
    assert list(out) == [pd.Timestamp("1970-01-02"), pd.NaT]
    assert out.tz is None


@pytest.mark.spec("IV-A")
def test_timestamp_conversion_on_an_aware_index() -> None:
    idx = small_index(tz="Europe/Berlin")
    t = idx.to_timestamp("2024-01-02 01:00")
    assert t == pd.Timestamp("2024-01-02 00:00", tz="UTC")
    assert str(t.tzinfo) == "Europe/Berlin"  # type: ignore[union-attr]
    assert idx.to_ns("1970-01-02 01:00") == DAY  # stored timestamps are UTC
    out = idx.to_datetime(_days(1))
    assert out[0] == pd.Timestamp("1970-01-02", tz="UTC")
    assert str(out.tz) == "Europe/Berlin"


# ------------------------------------------------------------------ immutability
@pytest.mark.parametrize("name", ["case_codes", "timestamps", "valid", "activity_codes", "starts", "sizes"])
def test_arrays_cannot_be_written_through_the_interface(name: str) -> None:
    idx = small_index()
    arr = getattr(idx, name)
    before = arr.copy()
    with pytest.raises(ValueError, match="read-only"):
        arr[0] = arr[1]
    with pytest.raises(ValueError, match="read-only"):
        arr += arr
    np.testing.assert_array_equal(getattr(idx, name), before)


def test_attributes_cannot_be_rebound() -> None:
    idx = small_index()
    with pytest.raises(AttributeError):
        idx.case_codes = np.zeros(8, dtype=np.int64)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        idx._codes = np.zeros(8, dtype=np.int64)
    with pytest.raises(AttributeError):
        idx.extra = 1  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        del idx._codes


def test_results_are_fresh_arrays_and_leave_the_index_unchanged() -> None:
    idx = small_index()
    counts = idx.count("B")
    counts[0] = 99
    _eq(idx.count("B"), [2, 1, 0])


def test_building_does_not_copy_its_inputs_and_leaves_them_writable() -> None:
    codes = np.array([0, 0, 1], dtype=np.int64)
    ts = _days(1, 2, 3)
    acts = np.array([0, 0, 0], dtype=np.int64)
    idx = CaseIndex(codes, ts, acts, ("A",))
    assert np.shares_memory(idx.case_codes, codes)
    assert np.shares_memory(idx.timestamps, ts)
    assert codes.flags.writeable
    assert ts.flags.writeable


# ------------------------------------------------------------------ order and validation
@pytest.mark.spec("IV-A")
def test_event_order_sorts_by_case_then_timestamp_then_tie_with_missing_timestamps_last() -> None:
    codes = np.array([1, 0, 0, 0, 1])
    ts = _days(3, None, 2, 2, 1)
    _eq(event_order(codes, ts), [2, 3, 1, 4, 0])
    _eq(event_order(codes, ts, tie=np.array([0, 0, 9, 1, 0])), [3, 2, 1, 4, 0])


@pytest.mark.parametrize(
    ("codes", "ts", "match"),
    [
        ([1, 0], [1, 2], "sorted by case"),
        ([0, 0], [2, 1], "sorted by timestamp"),
        ([0, 0], [None, 1], "sorted by timestamp"),
        ([0, 5], [1, 2], "case codes"),
        ([-1, 0], [1, 2], "case codes"),
    ],
)
def test_unsorted_or_out_of_range_input_is_refused(codes: list[int], ts: list[int | None], match: str) -> None:
    with pytest.raises(LogSchemaError, match=match):
        CaseIndex(np.array(codes), _days(*ts), np.array([0, 0]), ("A",), n_cases=2)


def test_inputs_of_different_lengths_are_refused() -> None:
    with pytest.raises(LogSchemaError, match="same length"):
        CaseIndex(np.array([0, 0]), _days(1), np.array([0, 0]), ("A",))
    with pytest.raises(LogSchemaError, match="activity codes"):
        CaseIndex(np.array([0]), _days(1), np.array([1]), ("A",))
