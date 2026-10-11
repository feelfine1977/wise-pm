"""Slicing by a case attribute works for every value a log accepts, or fails with a library error (decision 0001,
section 1; paper Section IV-E).

A case attribute holds plain values, tuples or frozen sets. The ranking helpers order the slices by key; keys that
cannot be ordered against each other are refused with a :class:`~wise.errors.NormError` naming the key, never with
a bare ``TypeError``.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

import wise
from wise.errors import NormError, WiseError

pytestmark = pytest.mark.spec("IV-E.pi")

# per case A..E of the running example; E has no value where the kind allows one to be missing
_KINDS: dict[str, list[Any]] = {
    "strings": ["x", "y", "x", "y", None],
    "integers": [1, 2, 1, 2, None],
    "strings and integers": [1, "x", 1, "x", None],
    "tuples": [("a", 1), ("b", 2), ("a", 1), ("b", 2), None],
    "frozen sets": [frozenset({"p"}), frozenset({"p", "q"}), frozenset({"p"}), frozenset({"p", "q"}), None],
    "tuples of strings or integers": [("a", 1), ("a", "b"), ("a", 1), ("a", "b"), None],
    "tuples and frozen sets": [("a", 1), frozenset({"p"}), ("a", 1), frozenset({"p"}), None],
    "every kind": [("a", 1), frozenset({"p"}), "x", 1, None],
}

_UNORDERABLE = {"strings and integers", "tuples of strings or integers", "tuples and frozen sets", "every kind"}


def _result(values: list[Any]) -> wise.ScoreResult:
    log = wise.running_p2p_log()
    log = log.with_case_attributes({"context": pd.Series(values, index=log.case_ids, dtype=object)})
    return wise.score(log, wise.running_p2p_norm())


_CALLS: dict[str, Callable[[wise.ScoreResult, str], Any]] = {
    "prioritize": lambda r, v: wise.prioritize(r, "context", view=v),
    "prioritize with two keys": lambda r, v: wise.prioritize(r, ["company", "context"], view=v),
    "layer_drivers": lambda r, v: wise.layer_drivers(r, "context", view=v),
    "penalty_mass": lambda r, v: wise.penalty_mass(r, v, "context"),
    "estimate_gamma": lambda r, v: wise.estimate_gamma(r, "context", view=v),
    "validation_table": lambda r, v: wise.validation_table(r, v, "context", censored=pd.Series({"E": True})),
}


@pytest.mark.parametrize("call", _CALLS.values(), ids=_CALLS.keys())
@pytest.mark.parametrize("view", ["Finance", "Logistics"])
@pytest.mark.parametrize("values", _KINDS.values(), ids=_KINDS.keys())
@pytest.mark.parametrize("missing", [True, False], ids=["one case missing", "none missing"])
def test_slicing_by_any_accepted_case_attribute_returns_or_raises_a_library_error(
    call: Callable[[wise.ScoreResult, str], Any], view: str, values: list[Any], missing: bool
) -> None:
    result = _result(values if missing else [*values[:4], values[0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.suppress(WiseError):
            call(result, view)


@pytest.mark.parametrize("kind", sorted(_UNORDERABLE))
@pytest.mark.parametrize("view", ["Finance", "Logistics"])
def test_a_slice_key_whose_values_cannot_be_ordered_is_refused_with_the_key_named(kind: str, view: str) -> None:
    result = _result(_KINDS[kind])
    with pytest.raises(NormError, match=r"slice key 'context' holds values that cannot be ordered"):
        wise.prioritize(result, "context", view=view)
    with pytest.raises(NormError, match=r"slice key 'context' holds values that cannot be ordered"):
        wise.penalty_mass(result, view, "context")
    with pytest.raises(NormError, match=r"slice key 'context' holds values that cannot be ordered"):
        wise.layer_drivers(result, "context", view=view)


@pytest.mark.parametrize("kind", ["tuples", "frozen sets"])
@pytest.mark.parametrize("view", ["Finance", "Logistics"])
def test_tuples_and_frozen_sets_slice_like_plain_values(kind: str, view: str) -> None:
    # tuples and frozen sets stand for "x" (cases A, C) and "y" (cases B, D), E has none
    plain = wise.prioritize(_result(_KINDS["strings"]), "context", view=view).table
    table = wise.prioritize(_result(_KINDS[kind]), "context", view=view).table
    labels = dict(zip(_KINDS[kind][:2], ["x", "y"], strict=True)) | {wise.MISSING_KEY: wise.MISSING_KEY}
    renamed = table.rename(index=labels)
    pd.testing.assert_frame_equal(renamed, plain)
    assert table["n_cases"].sum() == 5
