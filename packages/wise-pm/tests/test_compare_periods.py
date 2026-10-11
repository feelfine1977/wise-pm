"""Period comparison of two backlog records (paper Sec. IV-E, governance loop).

Two backlogs are comparable only when they were ranked the same way: the same
slice keys, view, shrinkage, baseline and volume, from scores under the same
norm and scoring mode. Hand-derived frames: the
previous period scores slice ``a`` 0.25 and ``b`` 0.75, the current one ``a``
0.5 and ``c`` 0.75; against the fixed baseline 1.0 the gap of ``a`` falls from
0.75 to 0.5.
"""

from __future__ import annotations

import dataclasses
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
    assert out.attrs["intended"] == {"view": ("Finance", "Logistics"), "gamma": (0.0, 1.0)}
    assert out.attrs["statement"] == "compared across intended differences: view 'Finance' and 'Logistics'; gamma 0.0 and 1.0"


def test_intended_names_a_field_of_the_specification() -> None:
    backlog = wise.prioritize(PREVIOUS, "slice")
    for name in ("global_mean", "z", "unknown"):
        with pytest.raises(NormError, match=rf"intended names fields that are not compared: \['{name}'\]"):
            wise.compare_periods(backlog, backlog, intended=[name])


@pytest.mark.spec("decision:0001")
def test_intended_accepts_the_norm_and_the_scoring_mode() -> None:
    backlog = wise.prioritize(PREVIOUS, "slice")
    for names in ("norm", "norm_fingerprint", "mode", ["norm", "mode"]):
        out = wise.compare_periods(backlog, backlog, intended=names)
        # nothing differs, so nothing is stated
        assert "intended" not in out.attrs


def _bound_changed(norm: wise.Norm) -> wise.Norm:
    """The norm with the lag bound of c2 lowered from 10 to 5 days."""
    lag = norm.get_constraint("c2").constraint
    assert isinstance(lag, wise.Lag)
    return norm.replace_constraint("c2", constraint=dataclasses.replace(lag, delta=5.0))


@pytest.mark.spec("IV-E")
def test_a_different_norm_is_refused_and_stated_when_intended(p2p_log: wise.EventLog, p2p_norm: wise.Norm) -> None:
    changed = _bound_changed(p2p_norm)
    previous = wise.prioritize(wise.score(p2p_log, p2p_norm), "company", view="Finance")
    current = wise.prioritize(wise.score(p2p_log, changed), "company", view="Finance")
    before, after = p2p_norm.fingerprint(), changed.fingerprint()
    with pytest.raises(
        NormError, match=rf"different specifications: norm_fingerprint '{before[:12]}…' and '{after[:12]}…'"
    ) as refused:
        wise.compare_periods(previous, current)
    assert "intended=['norm_fingerprint']" in str(refused.value)
    for name in ("norm", "norm_fingerprint"):
        out = wise.compare_periods(previous, current, intended=name)
        assert out.attrs["intended"] == {"norm_fingerprint": (before, after)}
        assert out.attrs["statement"] == (
            f"compared across an intended difference: norm_fingerprint '{before[:12]}…' and '{after[:12]}…'"
        )
    # Company B holds cases C, D, E. Flat Finance scores under the norm: A .6625, B .96667, C .87, D .9, E .15
    # (mean .709833, B .64); with the lower bound C .825 and A .55, B .899167 (mean .664833, B .625).
    # The same cases show a gap change of (.664833 - .625) - (.709833 - .64) = -0.03 that is the norm's, not the process's.
    assert out.loc["B", "gap_change"] == pytest.approx(-0.03)


@pytest.mark.spec("IV-E")
def test_a_different_scoring_mode_is_refused_and_stated_when_intended(p2p_log: wise.EventLog, p2p_norm: wise.Norm) -> None:
    previous = wise.prioritize(wise.score(p2p_log, p2p_norm, mode="flat"), "company", view="Finance")
    current = wise.prioritize(wise.score(p2p_log, p2p_norm, mode="layer_balanced"), "company", view="Finance")
    with pytest.raises(NormError, match=r"different specifications: mode 'flat' and 'layer_balanced'; pass") as refused:
        wise.compare_periods(previous, current)
    assert "intended=['mode']" in str(refused.value)
    out = wise.compare_periods(previous, current, intended="mode")
    assert out.attrs["intended"] == {"mode": ("flat", "layer_balanced")}
    assert out.attrs["statement"] == "compared across an intended difference: mode 'flat' and 'layer_balanced'"
    assert out["presence"].tolist() == ["both", "both"]


@pytest.mark.spec("decision:0001")
def test_a_norm_and_a_mode_must_both_be_intended(p2p_log: wise.EventLog, p2p_norm: wise.Norm) -> None:
    previous = wise.prioritize(wise.score(p2p_log, p2p_norm, mode="flat"), "company", view="Finance")
    current = wise.prioritize(wise.score(p2p_log, _bound_changed(p2p_norm), mode="layer_balanced"), "company", view="Finance")
    with pytest.raises(
        NormError, match=r"norm_fingerprint .*; mode 'flat' and 'layer_balanced'.*intended=\['norm_fingerprint', 'mode'\]"
    ):
        wise.compare_periods(previous, current)
    with pytest.raises(NormError, match=r"different specifications: mode .*intended=\['mode'\]"):
        wise.compare_periods(previous, current, intended="norm")
    out = wise.compare_periods(previous, current, intended=["norm", "mode"])
    assert list(out.attrs["intended"]) == ["norm_fingerprint", "mode"]


@pytest.mark.spec("IV-E")
def test_records_scored_alike_state_nothing(p2p_result: wise.ScoreResult) -> None:
    backlog = wise.prioritize(p2p_result, "company", view="Finance")
    assert wise.compare_periods(backlog, backlog).attrs == {}


def test_frames_carry_no_specification_and_are_compared_as_given() -> None:
    previous = wise.prioritize(PREVIOUS, "slice", baseline=1.0).table
    current = wise.prioritize(CURRENT, "slice", gamma=2.0).table
    assert wise.compare_periods(previous, current)["presence"].tolist() == ["both", "new", "gone"]


@pytest.mark.spec("IV-E")
def test_records_with_their_keys_as_columns_are_aligned_by_slice() -> None:
    previous = wise.prioritize(PREVIOUS, "slice", baseline=1.0, as_index=False)
    current = wise.prioritize(CURRENT, "slice", baseline=1.0, as_index=False)
    out = wise.compare_periods(previous, current)
    assert out.index.tolist() == ["a", "c", "b"]
    assert out["presence"].tolist() == ["both", "new", "gone"]
    assert out.loc["a", "gap_change"] == -0.25
