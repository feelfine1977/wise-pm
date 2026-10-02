"""``EventLog.fingerprint()``: a content fingerprint over what the log consumed (decision 0001, contracts)."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import pytest

import wise

pytestmark = pytest.mark.spec("decision:0001/3")


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": ["a", "a", "b", "b", "b"],
            "activity": ["PO", "GR", "PO", "GR", "IR"],
            "time": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-02", "2024-01-04", "2024-01-05"]),
            "amount": [10.0, 10.0, 20.0, 20.0, 5.0],
            "region": ["X", "X", "Y", "Y", "Y"],
            "note": ["n1", "n2", "n3", "n4", "n5"],
        }
    )


def _log(frame: pd.DataFrame, **kwargs: object) -> wise.EventLog:
    options: dict[str, object] = {
        "case_col": "case",
        "activity_col": "activity",
        "timestamp_col": "time",
        "case_attributes": ["region"],
        "exposure_col": "amount",
        **kwargs,
    }
    return wise.EventLog(frame, **options)  # type: ignore[arg-type]


def test_the_fingerprint_is_a_sha256_hex_digest_and_stable() -> None:
    log = _log(_frame())
    fp = log.fingerprint()
    assert re.fullmatch(r"[0-9a-f]{64}", fp)
    assert log.fingerprint() == fp
    assert _log(_frame()).fingerprint() == fp


def test_the_fingerprint_ignores_the_input_row_and_column_order() -> None:
    frame = _frame()
    shuffled = frame.iloc[[4, 2, 0, 3, 1]][["note", "time", "case", "region", "amount", "activity"]]
    assert _log(shuffled).fingerprint() == _log(frame).fingerprint()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("activity", "IR"),
        ("time", pd.Timestamp("2024-01-03 00:00:01")),
        ("amount", 10.25),
        ("region", "Z"),
        ("note", "changed"),
    ],
)
def test_changing_one_consumed_value_changes_the_fingerprint(column: str, value: Any) -> None:
    frame = _frame()
    changed = frame.copy()
    changed.loc[1, column] = value
    assert _log(changed).fingerprint() != _log(frame).fingerprint()


def test_a_column_the_log_does_not_keep_is_not_covered() -> None:
    frame = _frame()
    changed = frame.copy()
    changed.loc[1, "note"] = "changed"
    assert _log(changed, keep_columns=[]).fingerprint() == _log(frame, keep_columns=[]).fingerprint()
    assert _log(frame, keep_columns=[]).fingerprint() != _log(frame).fingerprint()


def test_the_schema_and_the_window_are_covered() -> None:
    frame = _frame()
    base = _log(frame).fingerprint()
    assert _log(frame, exposure_agg="sum").fingerprint() != base
    assert _log(frame, window=("2024-01-01", "2024-02-01")).fingerprint() != base


def test_an_attached_case_attribute_changes_the_fingerprint_and_the_old_log_keeps_its_own() -> None:
    log = _log(_frame())
    before = log.fingerprint()
    other = log.with_case_attributes({"x": pd.Series({"a": 1.0, "b": 2.0})})
    assert other.fingerprint() != before
    assert log.fingerprint() == before
    log.add_case_attribute("x", pd.Series({"a": 1.0, "b": 2.0}))
    assert log.fingerprint() == other.fingerprint()
    log.add_case_attribute("x", pd.Series({"a": 1.0, "b": 3.0}))
    assert log.fingerprint() != other.fingerprint()


def test_the_fingerprint_survives_values_that_cannot_be_hashed() -> None:
    frame = _frame()
    frame["tags"] = [["p"], ["q"], ["r"], ["s"], ["t"]]
    changed = frame.copy()
    changed["tags"] = [["p"], ["z"], ["r"], ["s"], ["t"]]
    assert _log(changed).fingerprint() != _log(frame).fingerprint()
