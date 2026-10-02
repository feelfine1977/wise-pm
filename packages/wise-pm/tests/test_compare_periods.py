"""Period comparison of two backlog records (paper Sec. IV-E, governance loop).

Two backlogs are comparable only when they were ranked the same way: the same
slice keys, view, shrinkage, baseline and volume. Hand-derived frames: the
previous period scores slice ``a`` 0.25 and ``b`` 0.75, the current one ``a``
0.5 and ``c`` 0.75; against the fixed baseline 1.0 the gap of ``a`` falls from
0.75 to 0.5.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

import wise
from wise.errors import NormError


def _period(scores: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"slice": list(scores), "score": list(scores.values()), "exposure": [2.0] * len(scores)})


PREVIOUS = _period({"a": 0.25, "b": 0.75})
CURRENT = _period({"a": 0.5, "c": 0.75})


@pytest.mark.spec("IV-E")
def test_records_ranked_alike_are_compared_slice_by_slice() -> None:
    previous = wise.prioritize(PREVIOUS, "slice", baseline=1.0)
    current = wise.prioritize(CURRENT, "slice", baseline=1.0)
    out = wise.compare_periods(previous, current)
    assert list(out.columns) == [
        "n_cases_prev",
        "stable_gap_prev",
        "stable_PI_prev",
        "n_cases_now",
        "stable_gap_now",
        "stable_PI_now",
        "gap_change",
        "PI_change",
        "presence",
    ]
    assert out.index.tolist() == ["a", "c", "b"]
    assert (out.loc["a", "gap_change"], out.loc["a", "PI_change"]) == (-0.25, -0.25)
    assert out["presence"].tolist() == ["both", "new", "gone"]
    assert out[["gap_change", "PI_change"]].iloc[1:].isna().all().all()


@pytest.mark.spec("IV-E")
def test_two_periods_with_their_own_global_mean_have_the_same_specification() -> None:
    # baseline=None on both sides: the requested baseline is the same rule, the realised means differ
    out = wise.compare_periods(wise.prioritize(PREVIOUS, "slice"), wise.prioritize(CURRENT, "slice"))
    assert out.loc["a", "presence"] == "both"


DIFFERENT = pytest.mark.parametrize(
    ("field", "other"),
    [
        ("gamma", lambda: wise.prioritize(CURRENT, "slice", baseline=1.0, gamma=2.0)),
        ("baseline", lambda: wise.prioritize(CURRENT, "slice")),
        ("volume", lambda: wise.prioritize(CURRENT, "slice", baseline=1.0, volume="exposure")),
        ("by", lambda: wise.prioritize(CURRENT.rename(columns={"slice": "s2"}), "s2", baseline=1.0)),
    ],
    ids=["gamma", "baseline", "volume", "by"],
)


@pytest.mark.spec("decision:0001")
@DIFFERENT
def test_a_different_specification_is_refused_with_the_field_named(field: str, other: Callable[[], wise.Backlog]) -> None:
    previous = wise.prioritize(PREVIOUS, "slice", baseline=1.0)
    with pytest.raises(NormError, match=rf"different specifications: {field} ") as refused:
        wise.compare_periods(previous, other())
    assert f"intended=['{field}']" in str(refused.value)
    assert wise.compare_periods(previous, other(), intended=field)["presence"].tolist() == ["both", "new", "gone"]


@pytest.mark.spec("decision:0001")
def test_every_differing_field_is_named_and_must_be_intended(p2p_result: wise.ScoreResult) -> None:
    finance = wise.prioritize(p2p_result, "company", view="Finance")
    logistics = wise.prioritize(p2p_result, "company", view="Logistics", gamma=1.0)
    with pytest.raises(NormError, match=r"view 'Finance' and 'Logistics'; gamma 0.0 and 1.0") as refused:
        wise.compare_periods(finance, logistics)
    assert "intended=['view', 'gamma']" in str(refused.value)
    with pytest.raises(NormError, match=r"different specifications: view .*intended=\['view'\]"):
        wise.compare_periods(finance, logistics, intended=["gamma"])
    out = wise.compare_periods(finance, logistics, intended=["view", "gamma"])
    assert set(out.index) == {"A", "B"}


def test_intended_names_a_field_of_the_specification() -> None:
    backlog = wise.prioritize(PREVIOUS, "slice")
    with pytest.raises(NormError, match="intended"):
        wise.compare_periods(backlog, backlog, intended=["norm"])


def test_frames_carry_no_specification_and_are_compared_as_given() -> None:
    previous = wise.prioritize(PREVIOUS, "slice", baseline=1.0).table
    current = wise.prioritize(CURRENT, "slice", gamma=2.0).table
    assert wise.compare_periods(previous, current)["presence"].tolist() == ["both", "new", "gone"]
