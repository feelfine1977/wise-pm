"""The 0.1.0 call forms on the result of ``prioritize`` keep working, with a deprecation warning.

Decision record 0001 (section 2) keeps the consumers working through a compatibility
path until 1.0. Each test below mirrors how the workbench (backend gateway and board,
its golden test, and the analytics package) reads the backlog as a DataFrame, and
checks that the answer equals the one read from ``Backlog.table`` and ``Backlog.spec``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import wise

DEPRECATED = pytest.warns(DeprecationWarning, match=r"Backlog\.table")


@pytest.fixture
def p2p_result() -> wise.ScoreResult:
    return wise.score(wise.running_p2p_log(), wise.running_p2p_norm())


def _legacy(result: wise.ScoreResult, by: str | list[str] = "company", **kwargs: Any) -> Any:
    """The record typed as 0.1.0 callers hold it: as an untyped frame."""
    return wise.prioritize(result, by, view="Finance", **kwargs)


@pytest.mark.spec("decision:0001/2")
def test_gateway_filters_the_backlog_and_reads_its_baseline(p2p_result: wise.ScoreResult) -> None:
    backlog = _legacy(p2p_result, ["company", "vendor"], gamma=1.0, min_cases=1, z=1.96, baseline=None, volume="cases")
    with DEPRECATED:
        bl = backlog[backlog["n_cases"] >= 1].copy()
    assert isinstance(bl, pd.DataFrame)
    pd.testing.assert_frame_equal(bl, backlog.table)
    assert float(bl.attrs.get("baseline", np.nan)) == backlog.spec.global_mean
    hot = wise.hotspot_table(bl, top=12, drivers=wise.layer_drivers(p2p_result, ["company", "vendor"], view="Finance"))
    bl["hotspot_type"] = hot["hotspot"].reindex(bl.index)
    out = bl.reset_index()
    assert list(out.columns[:2]) == ["company", "vendor"]
    assert "hotspot_type" not in backlog.table.columns


@pytest.mark.spec("decision:0001/2")
def test_golden_test_reads_length_attrs_and_rows(p2p_result: wise.ScoreResult) -> None:
    backlog = _legacy(p2p_result, gamma=20.0, z=1.96)
    with DEPRECATED:
        assert len(backlog) == len(backlog.table)
    with DEPRECATED:
        assert float(backlog.attrs["baseline"]) == backlog.spec.global_mean
    with DEPRECATED:
        rows = list(backlog.iterrows())
    assert [key for key, _ in rows] == list(backlog.table.index)
    assert wise.hotspot_table(backlog, top=12).equals(wise.hotspot_table(backlog.table, top=12))


@pytest.mark.spec("decision:0001/2")
def test_board_and_analytics_read_index_columns_and_selections(p2p_result: wise.ScoreResult) -> None:
    backlog = _legacy(p2p_result, gamma=2.0)
    table = backlog.table
    with DEPRECATED:
        assert list(backlog.index) == list(table.index)
    with DEPRECATED:
        assert backlog.index[0] == table.index[0]
    with DEPRECATED:
        assert np.array_equal(backlog["n_cases"].to_numpy(dtype=int), table["n_cases"].to_numpy(dtype=int))
    with DEPRECATED:
        pd.testing.assert_frame_equal(backlog[["n_cases", "stable_PI"]], table[["n_cases", "stable_PI"]])
    with DEPRECATED:
        assert "stable_PI" in backlog.columns
    with DEPRECATED:
        pd.testing.assert_series_equal(backlog.loc["B"], table.loc["B"])
    with DEPRECATED:
        pd.testing.assert_frame_equal(backlog.reindex(["B", "A"]), table.reindex(["B", "A"]))
    with DEPRECATED:
        assert "stable_PI" in backlog
    with DEPRECATED:
        assert list(backlog) == list(table.columns)


@pytest.mark.spec("decision:0001/2")
def test_analytics_reads_the_parameters_from_attrs(p2p_result: wise.ScoreResult) -> None:
    backlog = _legacy(p2p_result, gamma=20.0, baseline=0.9, volume="cases")
    with DEPRECATED:
        attrs = backlog.attrs
    assert attrs == {"view": "Finance", "gamma": 20.0, "baseline": 0.9, "volume": "cases", "by": ["company"]}
    assert backlog.table.attrs == {}
    assert wise.top_k_overlap(backlog, backlog, k=5) == 1.0


@pytest.mark.spec("decision:0001/2")
def test_assigning_a_column_still_reaches_the_table(p2p_result: wise.ScoreResult) -> None:
    backlog = _legacy(p2p_result)
    with DEPRECATED:
        backlog["rank"] = np.arange(1, len(backlog.table) + 1)
    assert list(backlog.table["rank"]) == [1, 2]


def test_the_record_is_never_mistaken_for_an_empty_frame() -> None:
    empty = wise.prioritize(pd.DataFrame({"slice": ["a"], "score": [0.5]}), "slice", min_cases=2)
    assert empty.table.empty
    assert bool(empty) is True


def test_private_names_are_not_forwarded(p2p_result: wise.ScoreResult) -> None:
    backlog = _legacy(p2p_result)
    with pytest.raises(AttributeError):
        backlog._mgr  # noqa: B018  # private pandas internals stay private
