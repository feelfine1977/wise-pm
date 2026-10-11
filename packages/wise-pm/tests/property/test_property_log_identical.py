"""Removing identical rows gives the log of the single rows whatever the row order (Sec. IV-A).

Each example draws distinct rows (each has its own ``seq``, the order column, so
the schema orders every event) and loads some of them more than once. Among
the distinct rows are events of one case with the same activity at the same
instant and another identifier and amount, which ``Duplicates.IDENTICAL``
keeps. Identifiers, amounts and the nested ``ctx`` cell may be missing; a copy
of a row may hold its mapping with the keys in another order and its set
built in another order, which are the same value. For every permutation of
the rows the log, its fingerprint and the counts are those of the single rows
built with ``keep``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, note
from hypothesis import strategies as st

import wise

pytestmark = pytest.mark.property

COLS: dict[str, Any] = {"case_col": "case", "activity_col": "activity", "timestamp_col": "time", "order_col": "seq"}

_nested = st.one_of(
    st.none(),
    st.dictionaries(st.sampled_from(["k", "j", "m"]), st.integers(0, 3), max_size=3),
    st.lists(st.integers(0, 3), max_size=3),
    st.frozensets(st.integers(0, 5), max_size=3).map(set),
)


def _copy(value: Any, flip: bool) -> Any:
    """``value`` as another object; a mapping or set rebuilt in the reverse order when ``flip``."""
    if isinstance(value, dict):
        return dict(reversed(list(value.items()))) if flip else dict(value)
    if isinstance(value, set):
        built: set[Any] = set()
        for item in sorted(value, reverse=flip):
            built.add(item)
        return built
    if isinstance(value, list):
        return list(value)
    return value


@st.composite
def frames(draw: st.DrawFn) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The single rows and the rows as loaded, with some of them more than once."""
    rows: list[dict[str, Any]] = []
    for c in range(draw(st.integers(1, 3))):
        for _ in range(draw(st.integers(1, 4))):
            rows.append(
                {
                    "case": f"c{c}",
                    "activity": draw(st.sampled_from(["A", "B"])),
                    "time": pd.Timestamp("2024-01-01") + pd.Timedelta(days=draw(st.integers(0, 2))),
                    "eid": draw(st.one_of(st.none(), st.sampled_from(["e1", "e2"]))),
                    "amount": draw(st.one_of(st.just(np.nan), st.sampled_from([0.5, 1.0]))),
                    "ctx": draw(_nested),
                    "seq": len(rows),
                }
            )
    loaded = []
    for row in rows:
        for _ in range(draw(st.integers(1, 3))):
            flip = draw(st.booleans())
            loaded.append({**row, "ctx": _copy(row["ctx"], flip)})
    return pd.DataFrame(rows), pd.DataFrame(loaded)


@pytest.mark.spec("IV-A")
@given(drawn=frames(), data=st.data())
def test_removing_identical_rows_gives_the_log_of_the_single_rows_in_any_row_order(
    drawn: tuple[pd.DataFrame, pd.DataFrame], data: st.DataObject
) -> None:
    single, loaded = drawn
    shuffled = loaded.iloc[list(data.draw(st.permutations(range(len(loaded)))))]
    note(f"rows:\n{shuffled}")
    log = wise.EventLog(loaded, **COLS, dedupe="identical")
    again = wise.EventLog(shuffled, **COLS, dedupe="identical")
    expected = wise.EventLog(single, **COLS)
    for built in (log, again):
        pd.testing.assert_frame_equal(built.events, expected.events, check_exact=True)
        pd.testing.assert_frame_equal(built.cases, expected.cases, check_exact=True)
        assert built.fingerprint() == expected.fingerprint()
    copies = loaded.groupby("seq")["case"].agg(["size", "first"])
    affected = copies.loc[copies["size"] > 1, "first"].nunique()
    for built in (log, again):
        assert (built.n_dropped_duplicates, built.n_cases_with_dropped_duplicates) == (len(loaded) - len(single), affected)
