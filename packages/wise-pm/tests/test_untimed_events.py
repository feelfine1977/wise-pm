"""What an event without a timestamp contributes to each kernel primitive.

``TABLE`` is the rule of ``LogSchema.untimed_events`` for every primitive of
the case index and the event log, under both policy values; the same table is
printed in ``docs/semantics/log-policies.md``. Each row has a probe that
measures the contribution on a hand-made log: case ``x`` holds three timed
events and two untimed ones (``B`` and ``D``), case ``z`` one timed and one
untimed event, and the same frame without the untimed events is the
reference for "ignored".
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import wise
from wise.log_schema import UntimedEvents

COLS: dict[str, Any] = {"case_col": "case", "activity_col": "activity", "timestamp_col": "time"}
DOC = Path(__file__).resolve().parents[3] / "docs" / "semantics" / "log-policies.md"

# primitive, legacy (default), uniform
TABLE: list[tuple[str, str, str]] = [
    ("`CaseIndex.count`, `EventLog.count`", "counted", "counted"),
    ("`EventLog.count_scoped` without `after` and `before`", "ignored", "counted"),
    ("`EventLog.count_scoped` with `after` or `before`, `count_before`, `count_after`", "ignored", "ignored"),
    ("`CaseIndex.aggregate`, `EventLog.total`", "counted", "counted"),
    ("`CaseIndex.first`, `CaseIndex.last`, `EventLog.first_ts`, `EventLog.last_ts`", "ignored", "ignored"),
    ("`EventLog.first_after` (activation and response)", "ignored", "ignored"),
    ("`EventLog.activation_lags`", "ignored", "ignored"),
    ("`CaseIndex.tied`", "never tied", "never tied"),
    ("`CaseIndex.run_starts`", "one run per case", "one run per case"),
    ("event order (`EventLog.events`, `EventLog.trace`)", "last in its case", "last in its case"),
    ("case table `n_events` and `exposure`", "counted", "counted"),
    ("case table `first_ts` and `last_ts`", "ignored", "ignored"),
    ("case attributes (first non-null value)", "read after the timed events", "read after the timed events"),
    ("`EventLog.observation_window`, `window_start`, `window_end`", "ignored", "ignored"),
]

D1, D2, D3 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")


def _frame() -> pd.DataFrame:
    """Case x: A, B, C timed plus B and D untimed; case y: A, B timed; case z: A timed, D untimed.
    The untimed events come first in the input."""
    return pd.DataFrame(
        {
            "case": ["x", "x", "x", "x", "x", "y", "y", "z", "z"],
            "activity": ["B", "D", "A", "B", "C", "A", "B", "D", "A"],
            "time": [pd.NaT, pd.NaT, D1, D2, D3, D1, D2, pd.NaT, D1],
            "amount": [5.0, 7.0, 1.0, 2.0, 3.0, 1.0, 1.0, 4.0, 1.0],
            "org": ["late", "late", None, "early", None, None, None, "late", None],
        }
    )


def _log(frame: pd.DataFrame, policy: UntimedEvents) -> wise.EventLog:
    schema = wise.LogSchema(
        **COLS, case_attributes="org", exposure_col="amount", exposure_agg="sum", missing_timestamps="keep", untimed_events=policy
    )
    return wise.EventLog(frame, schema=schema)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, pd.Series | pd.DataFrame):
        return bool(a.equals(b))
    if isinstance(a, tuple):
        return all(_same(x, y) for x, y in zip(a, b, strict=True))
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def _counted_or_ignored(f: Callable[[wise.EventLog], Any]) -> Callable[[wise.EventLog, wise.EventLog], str]:
    """'ignored' when the primitive gives the same result without the untimed events, else 'counted'."""
    return lambda kept, timed_only: "ignored" if _same(f(kept), f(timed_only)) else "counted"


def _tied(kept: wise.EventLog, _: wise.EventLog) -> str:
    untimed = ~kept.case_index.valid
    return "never tied" if not kept.case_index.tied()[untimed].any() else "tied"


def _runs(kept: wise.EventLog, _: wise.EventLog) -> str:
    idx = kept.case_index
    untimed = ~idx.valid
    starts = pd.Series(idx.run_starts()[untimed]).groupby(idx.case_codes[untimed]).nunique()
    return "one run per case" if (starts == 1).all() else "own runs"


def _order(kept: wise.EventLog, _: wise.EventLog) -> str:
    trace = kept.trace("x")
    return "last in its case" if trace["time"].isna().tolist() == [False, False, False, True, True] else "elsewhere"


def _attributes(kept: wise.EventLog, _: wise.EventLog) -> str:
    # x: a timed event says 'early', so the untimed 'late' is not taken; z: only the untimed event has a value
    org = kept.cases["org"]
    return "read after the timed events" if (org["x"], org["z"]) == ("early", "late") else "read first"


PROBES: dict[str, Callable[[wise.EventLog, wise.EventLog], str]] = {
    TABLE[0][0]: _counted_or_ignored(lambda log: (log.count(["B", "D"]), log.case_index.count(["B", "D"]))),
    TABLE[1][0]: _counted_or_ignored(lambda log: log.count_scoped(["B", "D"])),
    TABLE[2][0]: _counted_or_ignored(
        lambda log: (
            log.count_scoped(["B", "D"], after="A"),
            log.count_scoped(["B", "D"], before="C"),
            log.count_before(["B", "D"], "C"),
            log.count_after(["B", "D"], "A"),
        )
    ),
    TABLE[3][0]: _counted_or_ignored(
        lambda log: (log.total("amount", ["B", "D"]), log.case_index.aggregate(log.events["amount"], ["B", "D"]))
    ),
    TABLE[4][0]: _counted_or_ignored(
        lambda log: (
            log.first_ts(["B", "D"]),
            log.last_ts(["B", "D"]),
            log.case_index.first(["B", "D"]),
            log.case_index.last(["B", "D"]),
        )
    ),
    TABLE[5][0]: _counted_or_ignored(
        lambda log: (
            log.first_after("A", ["B", "D"]),
            log.first_after(["B", "D"], "C", activation="last"),
            log.first_after("A", ["B", "D"], response="first_overall"),
        )
    ),
    TABLE[6][0]: _counted_or_ignored(lambda log: (log.activation_lags(["B", "D"], "C"), log.activation_lags("A", ["B", "D"]))),
    TABLE[7][0]: _tied,
    TABLE[8][0]: _runs,
    TABLE[9][0]: _order,
    TABLE[10][0]: _counted_or_ignored(lambda log: (log.cases["n_events"], log.cases["exposure"])),
    TABLE[11][0]: _counted_or_ignored(lambda log: (log.cases["first_ts"], log.cases["last_ts"])),
    TABLE[12][0]: _attributes,
    TABLE[13][0]: _counted_or_ignored(lambda log: (log.observation_window(), log.window_start, log.window_end)),
}


@pytest.mark.spec("IV-A")
@pytest.mark.parametrize("policy", list(UntimedEvents))
@pytest.mark.parametrize("row", TABLE, ids=[r[0] for r in TABLE])
def test_each_primitive_follows_the_table(row: tuple[str, str, str], policy: UntimedEvents) -> None:
    primitive, legacy, uniform = row
    frame = _frame()
    kept = _log(frame, policy)
    timed_only = _log(frame[frame["time"].notna()], policy)
    assert kept.n_missing_timestamps == 3
    expected = legacy if policy is UntimedEvents.LEGACY else uniform
    assert PROBES[primitive](kept, timed_only) == expected


@pytest.mark.spec("IV-A")
def test_the_probes_tell_counted_from_ignored() -> None:
    frame = _frame()
    kept, timed_only = _log(frame, UntimedEvents.LEGACY), _log(frame[frame["time"].notna()], UntimedEvents.LEGACY)
    # hand-derived values on case x behind the words of the table
    assert kept.count(["B", "D"])["x"] == 3.0
    assert timed_only.count(["B", "D"])["x"] == 1.0
    assert kept.total("amount", ["B", "D"])["x"] == 14.0
    assert kept.cases.loc["x", "exposure"] == 18.0
    assert kept.first_ts(["B", "D"])["x"] == D2
    assert kept.count_scoped(["B", "D"])["x"] == 1.0
    assert _log(frame, UntimedEvents.UNIFORM).count_scoped(["B", "D"])["x"] == 3.0
    assert kept.observation_window() == (D1, D3)


def _doc_rows() -> list[tuple[str, str, str]]:
    text = DOC.read_text(encoding="utf-8")
    lines = text.split("<!-- untimed-events-table -->")[1].strip().splitlines()
    end = next((i for i, line in enumerate(lines) if not line.startswith("|")), len(lines))
    rows = lines[2:end]  # below the header and its rule
    out: list[tuple[str, str, str]] = []
    for line in rows:
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
        out.append((cells[0], cells[1], cells[2]))
    return out


@pytest.mark.spec("decision:0001")
@pytest.mark.skipif(not DOC.exists(), reason="the workspace documentation is not part of the package")
def test_the_documentation_prints_the_same_table() -> None:
    assert _doc_rows() == TABLE


@pytest.mark.spec("decision:0001")
def test_the_schema_documents_both_rules() -> None:
    doc = UntimedEvents.__doc__ or ""
    assert "count_scoped" in doc
    assert wise.LogSchema().untimed_events is UntimedEvents.LEGACY
