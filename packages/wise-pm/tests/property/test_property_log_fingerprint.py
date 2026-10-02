"""``EventLog.fingerprint()`` covers the consumed columns and not the input row order (decision 0001, contracts)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

import wise
from _support.strategies import ACTIVITIES, FLOWS, REGIONS, log_kwargs

# Per case the days are distinct, so the schema orders every event: (case, timestamp).
_case = st.tuples(
    st.sampled_from(FLOWS),
    st.sampled_from(REGIONS),
    st.lists(
        st.tuples(st.sampled_from(ACTIVITIES), st.integers(0, 120), st.integers(0, 400)),
        min_size=1,
        max_size=8,
        unique_by=lambda e: e[1],
    ),
)


def _frame(cases: list[tuple[str, str, list[tuple[str, int, int]]]]) -> pd.DataFrame:
    rows = [
        (f"c{i:02d}", act, day / 2, amount / 4, flow, region)
        for i, (flow, region, events) in enumerate(cases)
        for act, day, amount in events
    ]
    df = pd.DataFrame(rows, columns=["case", "activity", "day", "amount", "flow", "region"])
    df["time"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(df["day"], unit="D")
    return df


def _fingerprint(frame: pd.DataFrame) -> str:
    return wise.EventLog(frame, **log_kwargs()).fingerprint()


@pytest.mark.property
@pytest.mark.spec("decision:0001/3")
@given(cases=st.lists(_case, min_size=1, max_size=20), data=st.data())
def test_reordering_rows_the_schema_orders_keeps_the_fingerprint(cases, data):
    frame = _frame(cases)
    permutation = data.draw(st.permutations(range(len(frame))))
    assert _fingerprint(frame.iloc[list(permutation)]) == _fingerprint(frame)


@pytest.mark.property
@pytest.mark.spec("decision:0001/3")
@given(
    cases=st.lists(_case, min_size=1, max_size=20),
    column=st.sampled_from(["activity", "time", "amount", "day", "flow", "region"]),
    data=st.data(),
)
def test_changing_one_consumed_value_changes_the_fingerprint(cases, column, data):
    frame = _frame(cases)
    row = data.draw(st.integers(0, len(frame) - 1))
    changed = frame.copy()
    old: Any = frame.at[row, column]
    if column == "activity":
        new = data.draw(st.sampled_from([a for a in [*ACTIVITIES, "NEW"] if a != old]))
    elif column in ("flow", "region"):
        new = data.draw(st.sampled_from([v for v in [*FLOWS, *REGIONS, "NEW"] if v != old]))
    elif column == "time":
        # a quarter-day shift: no other event of the case lies there (they sit on half days)
        new = pd.Timestamp(old) + pd.Timedelta(hours=6)
    else:
        new = float(old) + data.draw(st.sampled_from([0.125, 1000.0]))
    changed.at[row, column] = new
    assert changed.shape == frame.shape
    assert changed["case"].equals(frame["case"])
    assert _fingerprint(changed) != _fingerprint(frame)
