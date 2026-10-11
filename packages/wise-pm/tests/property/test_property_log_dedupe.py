"""Duplicate removal keeps the same event whatever the order of the input rows (Sec. IV-A).

Each example draws cases whose events have distinct timestamps, so the schema
orders every event, and plants twins: events of the same case, activity and
time as an existing event, with another event identifier and another amount.
Under ``dedupe=True`` the survivor of each twin group is stated by
``Duplicates.DROP``: the smallest ``order_col`` value, then the smallest other
columns in column-name order (``activity, amount, case, eid, seq, time``), so
here the smallest amount among the smallest sequence numbers. Every
permutation of the rows gives the same log and the same fingerprint.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from hypothesis import given, note
from hypothesis import strategies as st

import wise

pytestmark = pytest.mark.property

COLS: dict[str, Any] = {"case_col": "case", "activity_col": "activity", "timestamp_col": "time", "event_id_col": "eid"}

_events = st.lists(st.tuples(st.sampled_from(["A", "B", "C"]), st.integers(0, 40)), min_size=1, max_size=5, unique_by=lambda e: e[1])


@st.composite
def frames(draw: st.DrawFn) -> pd.DataFrame:
    """Rows ``(case, activity, day, eid, amount, seq)``; ``eid`` is unique, amounts within a twin group differ."""
    rows: list[dict[str, Any]] = []
    for i, events in enumerate(draw(st.lists(_events, min_size=1, max_size=4))):
        for activity, day in events:
            rows.append({"case": f"c{i}", "activity": activity, "day": day, "amount": float(len(rows)), "seq": draw(st.integers(0, 2))})
    sources = draw(st.lists(st.integers(0, len(rows) - 1), min_size=1, max_size=4))
    for k, source in enumerate(sources):
        twin = {**rows[source], "amount": draw(st.integers(0, 8)) + 0.25 + 10 * k, "seq": draw(st.integers(0, 2))}
        rows.append(twin)
    frame = pd.DataFrame(rows)
    frame["eid"] = [f"e{n:02d}" for n in range(len(frame))]
    frame["time"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(frame.pop("day"), unit="D")
    return frame


def _survivors(frame: pd.DataFrame, order_col: str | None) -> list[str]:
    """The event id kept of each key group, by the stated rule, in (case, time) order."""
    by = ([order_col] if order_col else []) + ["amount"]
    ranked = frame.sort_values(by, kind="stable")
    return ranked.drop_duplicates(["case", "activity", "time"]).sort_values(["case", "time"])["eid"].tolist()


@pytest.mark.spec("IV-A")
@given(frame=frames(), order_col=st.sampled_from([None, "seq"]), data=st.data())
def test_the_survivor_and_the_fingerprint_do_not_depend_on_the_row_order(frame: pd.DataFrame, order_col: str | None, data: st.DataObject) -> None:
    permutation = data.draw(st.permutations(range(len(frame))))
    shuffled = frame.iloc[list(permutation)]
    note(f"rows:\n{shuffled}")
    log = wise.EventLog(frame, **COLS, order_col=order_col, dedupe=True)
    again = wise.EventLog(shuffled, **COLS, order_col=order_col, dedupe=True)
    assert again.events["eid"].tolist() == log.events["eid"].tolist() == _survivors(frame, order_col)
    pd.testing.assert_frame_equal(again.events, log.events, check_exact=True)
    assert again.fingerprint() == log.fingerprint()
    n_keys = len(frame.drop_duplicates(["case", "activity", "time"]))
    affected = frame[frame.duplicated(["case", "activity", "time"], keep=False)]["case"].nunique()
    assert (again.n_dropped_duplicates, again.n_cases_with_dropped_duplicates) == (len(frame) - n_keys, affected)
