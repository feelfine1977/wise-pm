"""Nested cell values cannot change a log behind its fingerprint (decision 0001, contracts: an immutable log).

A log keeps its own sealed copy of every dictionary, list, set or array it is given: what it hands out
refuses to be changed, the caller's input frame no longer reaches it, and the fingerprint reads the
sealed content in a canonical form.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

import wise
from wise.errors import LogSchemaError

pytestmark = pytest.mark.spec("decision:0001/3")


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": ["a", "a", "b"],
            "activity": ["PO", "GR", "PO"],
            "time": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-02"]),
            "region": ["X", "X", "Y"],
            "payload": [{"doc": "D1", "lines": [1, 2]}, {"doc": "D2", "lines": [3]}, {"doc": "D3", "lines": []}],
            "tags": [["po", "sap"], ["gr"], []],
        }
    )


def _cells(values: list[Any]) -> pd.Series:
    return pd.Series(values, dtype=object)


def _log(frame: pd.DataFrame, **kwargs: Any) -> wise.EventLog:
    return wise.EventLog(
        frame, case_col="case", activity_col="activity", timestamp_col="time", case_attributes=["region"], **kwargs
    )


def _state(log: wise.EventLog) -> tuple[Any, ...]:
    """What a caller can read of the log, as plain values (deep copies)."""
    events = [{k: copy.deepcopy(v) for k, v in row.items()} for row in log.events.to_dict("records")]
    trace = [{k: copy.deepcopy(v) for k, v in row.items()} for row in log.trace("a").to_dict("records")]
    return events, trace, log.fingerprint()


def _first_cell(frame: pd.DataFrame, column: str) -> Any:
    return frame[column].iloc[0]


_WRITES: dict[str, Callable[[Any], None]] = {
    "dict item": lambda d: d.__setitem__("doc", "changed"),
    "dict update": lambda d: d.update(doc="changed"),
    "dict pop": lambda d: d.pop("doc"),
    "dict in-place or": lambda d: d.__ior__({"extra": 1}),
    "nested list append": lambda d: d["lines"].append(99),
    "nested list item": lambda d: d["lines"].__setitem__(0, 99),
}

_LIST_WRITES: dict[str, Callable[[Any], None]] = {
    "append": lambda xs: xs.append("x"),
    "item": lambda xs: xs.__setitem__(0, "x"),
    "in-place add": lambda xs: xs.__iadd__(["x"]),
    "clear": lambda xs: xs.clear(),
    "sort": lambda xs: xs.sort(reverse=True),
}

_SOURCES: dict[str, Callable[[wise.EventLog], pd.DataFrame]] = {
    "events": lambda log: log.events,
    "trace": lambda log: log.trace("a"),
    "to_pm4py": lambda log: log.to_pm4py(),
}


@pytest.mark.parametrize("source", sorted(_SOURCES))
@pytest.mark.parametrize("write", sorted(_WRITES))
def test_writing_into_a_dictionary_handed_out_changes_nothing_of_the_log(source: str, write: str) -> None:
    log = _log(_frame())
    before = _state(log)
    cell = _first_cell(_SOURCES[source](log), "payload")
    with pytest.raises(TypeError, match="read-only"):
        _WRITES[write](cell)
    assert _state(log) == before
    assert log.fingerprint() == _log(_frame()).fingerprint()


@pytest.mark.parametrize("source", sorted(_SOURCES))
@pytest.mark.parametrize("write", sorted(_LIST_WRITES))
def test_writing_into_a_list_handed_out_changes_nothing_of_the_log(source: str, write: str) -> None:
    log = _log(_frame())
    before = _state(log)
    cell = _first_cell(_SOURCES[source](log), "tags")
    with pytest.raises(TypeError, match="read-only"):
        _LIST_WRITES[write](cell)
    assert _state(log) == before
    assert log.fingerprint() == _log(_frame()).fingerprint()


def test_a_handed_out_value_reads_like_the_value_given() -> None:
    log = _log(_frame())
    payload, tags = log.events["payload"], log.events["tags"]
    assert payload.iloc[0] == {"doc": "D1", "lines": [1, 2]}
    assert isinstance(payload.iloc[0], dict)
    assert isinstance(payload.iloc[0]["lines"], list)
    assert tags.tolist() == [["po", "sap"], ["gr"], []]
    assert isinstance(tags.iloc[0], list)
    # a plain copy of it is the caller's own
    own = dict(payload.iloc[0])
    own["doc"] = "mine"
    assert own["doc"] == "mine"
    assert log.events["payload"].iloc[0]["doc"] == "D1"


@pytest.mark.parametrize("write", sorted(_WRITES))
def test_changing_the_input_frame_after_the_build_changes_nothing_of_the_log(write: str) -> None:
    frame = _frame()
    log = _log(frame)
    before = _state(log)
    _WRITES[write](frame["payload"].iloc[0])
    frame["tags"].iloc[0].append("x")
    assert _state(log) == before
    assert log.fingerprint() == _log(_frame()).fingerprint()


def test_the_input_frame_keeps_its_own_values() -> None:
    frame = _frame()
    original = frame["payload"].iloc[0]
    _log(frame)
    assert frame["payload"].iloc[0] is original
    assert type(original) is dict
    original["doc"] = "still writable"


def test_a_numeric_array_is_kept_read_only_and_cannot_be_made_writable() -> None:
    frame = _frame()
    frame["vector"] = _cells([np.array([1.0, 2.0]), np.array([3.0]), np.array([])])
    log = _log(frame)
    before = log.fingerprint()
    frame["vector"].iloc[0][0] = 9.0
    cell = log.events["vector"].iloc[0]
    with pytest.raises(ValueError, match="read-only"):
        cell[0] = 9.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        cell.flags.writeable = True
    np.testing.assert_array_equal(log.events["vector"].iloc[0], [1.0, 2.0])
    assert log.fingerprint() == before


def test_tuples_and_sets_are_kept_with_their_items_sealed() -> None:
    frame = _frame()
    frame["pair"] = _cells([("x", {"k": [1]}), ("y", {"k": []}), ("z", {})])
    frame["kinds"] = _cells([{"p", "q"}, set(), {"r"}])
    log = _log(frame)
    frame["pair"].iloc[0][1]["k"].append(2)
    frame["kinds"].iloc[0].add("new")
    assert log.events["pair"].iloc[0] == ("x", {"k": [1]})
    assert log.events["kinds"].iloc[0] == frozenset({"p", "q"})
    with pytest.raises(TypeError, match="read-only"):
        log.events["pair"].iloc[0][1]["k"].append(2)
    reordered = _frame()
    reordered["pair"] = _cells([("x", {"k": [1]}), ("y", {"k": []}), ("z", {})])
    reordered["kinds"] = _cells([{"q", "p"}, set(), {"r"}])
    assert _log(reordered).fingerprint() == log.fingerprint()


# ------------------------------------------------------------------ fingerprint


def test_logs_whose_nested_values_differ_only_in_key_order_have_one_fingerprint() -> None:
    frame = _frame()
    reordered = frame.copy()
    reordered["payload"] = _cells([{"lines": [1, 2], "doc": "D1"}, {"lines": [3], "doc": "D2"}, {"lines": [], "doc": "D3"}])
    assert _log(reordered).fingerprint() == _log(frame).fingerprint()


def test_logs_whose_nested_sets_were_built_in_another_order_have_one_fingerprint() -> None:
    frame = _frame()
    frame["kinds"] = _cells([{"p", "q", "r"}, set(), {"s"}])
    other = frame.copy()
    other["kinds"] = _cells([{"r", "q", "p"}, set(), {"s"}])
    assert _log(other).fingerprint() == _log(frame).fingerprint()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("payload", {"doc": "D1", "lines": [1, 3]}),  # a nested list item
        ("payload", {"doc": "D1", "lines": [2, 1]}),  # the order of a list
        ("payload", {"doc": "D1", "lines": [1, 2], "extra": None}),  # one more key
        ("payload", {"doc": "D1", "lines": (1, 2)}),  # a tuple is not a list
        ("payload", {"doc": "D1", "lines": [1.0, 2]}),  # a float is not an int
        ("payload", {"doc": "D1", "lines": ["1", 2]}),  # a string is not a number
        ("tags", ["sap", "po"]),
        ("tags", ["po"]),
    ],
)
def test_logs_that_differ_in_one_nested_value_have_different_fingerprints(column: str, value: Any) -> None:
    frame = _frame()
    changed = frame.copy()
    changed[column] = _cells([value, *frame[column].iloc[1:]])
    assert _log(changed).fingerprint() != _log(frame).fingerprint()


def test_an_attached_nested_attribute_is_sealed_and_fingerprinted_by_content() -> None:
    log = _log(_frame())
    values: dict[str, Any] = {"a": {"y": 1, "x": [2]}, "b": {"x": []}}
    log.add_case_attribute("profile", pd.Series(values))
    before = log.fingerprint()
    values["a"]["x"].append(3)
    profile: Any = log.cases["profile"]["a"]
    with pytest.raises(TypeError, match="read-only"):
        profile["x"].append(3)
    assert log.cases["profile"]["a"] == {"y": 1, "x": [2]}
    assert log.fingerprint() == before
    other = _log(_frame()).with_case_attributes({"profile": pd.Series({"a": {"x": [2], "y": 1}, "b": {"x": []}})})
    assert other.fingerprint() == before


class _Tags(list[str]):
    pass


def test_other_mappings_lists_and_immutable_scalars_are_sealed_and_fingerprinted_by_value() -> None:
    import collections
    import decimal

    frame = _frame()
    frame["extra"] = _cells(
        [
            collections.OrderedDict([("b", np.int64(1)), ("a", decimal.Decimal("2.5"))]),
            _Tags(["x"]),
            (pd.Timestamp("2024-01-01"), {"k": np.float64(0.5)}),
        ]
    )
    log = _log(frame)
    extra = log.events["extra"]
    assert type(extra.iloc[0]) is not collections.OrderedDict
    assert extra.iloc[0] == {"a": decimal.Decimal("2.5"), "b": 1}
    assert type(extra.iloc[1]) is not _Tags
    assert extra.iloc[1] == ["x"]
    with pytest.raises(TypeError, match="read-only"):
        extra.iloc[1].append("y")
    same = _frame()
    same["extra"] = _cells([{"a": decimal.Decimal("2.5"), "b": 1}, ["x"], (pd.Timestamp("2024-01-01"), {"k": 0.5})])
    assert _log(same).fingerprint() == log.fingerprint()
    other = same.copy()
    other["extra"] = _cells([{"a": decimal.Decimal("2.5"), "b": 1}, ["x"], (pd.Timestamp("2024-01-02"), {"k": 0.5})])
    assert _log(other).fingerprint() != log.fingerprint()


# ------------------------------------------------------------------ refusals


class _Mutable:
    def __init__(self) -> None:
        self.value = 1


@pytest.mark.parametrize(
    "value",
    [_Mutable(), bytearray(b"ab"), {"k": _Mutable()}, [1, [bytearray(b"x")]], np.array([{"k": 1}], dtype=object)],
    ids=["object", "bytearray", "in a dict", "in a nested list", "object array"],
)
def test_a_value_that_cannot_be_sealed_is_refused_with_the_column_named(value: Any) -> None:
    frame = _frame()
    frame["blob"] = _cells([value, None, None])
    with pytest.raises(LogSchemaError, match=r"column 'blob' holds a value of type '\w+' in case 'a'"):
        _log(frame)


def test_a_value_that_contains_itself_is_refused_with_the_column_named() -> None:
    frame = _frame()
    loop: dict[str, Any] = {}
    loop["self"] = loop
    frame["blob"] = _cells([None, None, loop])
    with pytest.raises(LogSchemaError, match=r"column 'blob' holds a value that contains itself .* in case 'b'"):
        _log(frame)


def test_a_column_the_log_does_not_keep_is_not_sealed_nor_refused() -> None:
    frame = _frame()
    frame["blob"] = _cells([_Mutable(), None, None])
    log = _log(frame, keep_columns=["payload"])
    assert "blob" not in log.events.columns


def test_a_nested_case_attribute_is_refused_with_the_column_named() -> None:
    frame = _frame()
    with pytest.raises(LogSchemaError, match=r"case attribute 'payload'"):
        wise.EventLog(frame, case_col="case", activity_col="activity", timestamp_col="time", case_attributes=["payload"])


def test_a_case_attribute_of_tuples_or_sets_is_kept() -> None:
    frame = _frame()
    frame["pair"] = _cells([("x", 1), ("x", 1), ("y", 2)])
    frame["kinds"] = _cells([{"p"}, {"p"}, {"q"}])
    log = wise.EventLog(frame, case_col="case", activity_col="activity", timestamp_col="time", case_attributes=["pair", "kinds"])
    assert log.cases["pair"].tolist() == [("x", 1), ("y", 2)]
    assert log.cases["kinds"].tolist() == [frozenset({"p"}), frozenset({"q"})]


# ------------------------------------------------------------------ plain columns


def test_plain_object_columns_keep_their_values_and_types() -> None:
    import datetime
    import decimal

    frame = _frame().drop(columns=["payload", "tags"])
    frame["mixed"] = _cells([1, "1", None])
    frame["amount"] = _cells([decimal.Decimal("1.5"), decimal.Decimal(2), None])
    frame["day"] = _cells([datetime.date(2024, 1, 1), None, datetime.date(2024, 1, 2)])
    frame["pair"] = _cells([("x", 1), ("y", 2), None])
    log = _log(frame)
    assert log.events["mixed"].tolist() == [1, "1", None]
    assert log.events["amount"].iloc[0] == decimal.Decimal("1.5")
    assert log.events["day"].iloc[0] == datetime.date(2024, 1, 1)
    assert log.events["pair"].iloc[0] == ("x", 1)
