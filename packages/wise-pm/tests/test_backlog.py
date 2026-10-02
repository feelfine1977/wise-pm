"""The backlog record: the ranked table plus the specification it was ranked with.

Expectations are derived by hand on a three-case frame with dyadic scores, so every
number below is exact: slice ``a`` scores 0.5 and 0.75 (mean 0.625), slice ``b``
scores 0.25; the global mean is 1.5 / 3 = 0.5, so ``a`` has no gap and ``b`` has
gap 0.25 and PI 0.25. With γ = 1 the stabilised gap of ``b`` is 1/(1+1) · 0.25.
"""

from __future__ import annotations

import copy
import pickle

import pandas as pd
import pytest

import wise
from wise.errors import NormError


def _three_cases() -> pd.DataFrame:
    return pd.DataFrame({"slice": ["a", "a", "b"], "score": [0.5, 0.75, 0.25]})


@pytest.fixture
def p2p_result() -> wise.ScoreResult:
    return wise.score(wise.running_p2p_log(), wise.running_p2p_norm())


# ------------------------------------------------------------------ the record and its specification
@pytest.mark.spec("IV-E")
def test_prioritize_returns_a_record_with_the_ranked_table() -> None:
    backlog = wise.prioritize(_three_cases(), "slice", gamma=1.0)
    assert isinstance(backlog, wise.Backlog)
    table = backlog.table
    assert list(table.index) == ["b", "a"]
    assert table.index.name == "slice"
    assert list(table.columns) == [
        "n_cases",
        "mean_score",
        "volume",
        "gap",
        "PI",
        "stable_mean",
        "stable_gap",
        "stable_PI",
        "global_mean",
    ]
    assert table.loc["b", "gap"] == 0.25
    assert table.loc["b", "PI"] == 0.25
    assert table.loc["b", "stable_gap"] == 0.125
    assert table.loc["a", "gap"] == 0.0


@pytest.mark.spec("decision:0001")
def test_spec_of_a_frame_backlog_holds_the_parameters() -> None:
    backlog = wise.prioritize(_three_cases(), "slice", gamma=1.0)
    assert backlog.spec == wise.BacklogSpec(
        by=("slice",),
        view=None,
        gamma=1.0,
        volume="cases",
        baseline=None,
        global_mean=0.5,
        min_cases=1,
        z=None,
        norm_fingerprint=None,
        score_col="score",
    )


@pytest.mark.spec("decision:0001")
def test_spec_of_a_result_backlog_names_view_norm_and_every_parameter(p2p_result: wise.ScoreResult) -> None:
    backlog = wise.prioritize(p2p_result, ["company", "vendor"], view="Logistics", gamma=2, min_cases=2, z=1.96, baseline=1)
    spec = backlog.spec
    assert spec.by == ("company", "vendor")
    assert spec.view == "Logistics"
    assert (spec.gamma, spec.min_cases, spec.z, spec.baseline, spec.global_mean) == (2.0, 2, 1.96, 1.0, 1.0)
    assert spec.volume == "cases"
    assert spec.norm_fingerprint == wise.running_p2p_norm().fingerprint()
    assert spec.to_dict()["by"] == ["company", "vendor"]


def test_spec_names_the_only_view_when_none_is_given() -> None:
    norm = wise.running_p2p_norm()
    one_view = norm.replace(views=[norm.get_view("Finance")])
    result = wise.score(wise.running_p2p_log(), one_view)
    assert wise.prioritize(result, "company").spec.view == "Finance"


@pytest.mark.spec("decision:0001")
def test_spec_is_not_stored_in_the_table_attrs_and_survives_copy_and_pickle() -> None:
    backlog = wise.prioritize(_three_cases(), "slice", gamma=1.0)
    assert backlog.table.attrs == {}
    for twin in (copy.copy(backlog), copy.deepcopy(backlog), pickle.loads(pickle.dumps(backlog))):
        assert twin.spec == backlog.spec
        pd.testing.assert_frame_equal(twin.table, backlog.table)
    with pytest.raises(AttributeError):
        backlog.spec = backlog.spec  # type: ignore[misc]  # frozen record


def test_as_index_false_keeps_the_keys_as_columns() -> None:
    backlog = wise.prioritize(_three_cases(), "slice", as_index=False)
    assert list(backlog.table["slice"]) == ["b", "a"]
    assert backlog.spec.by == ("slice",)


def test_repr_names_slices_view_and_gamma() -> None:
    assert repr(wise.prioritize(_three_cases(), "slice", gamma=1.0)) == (
        "Backlog(2 slices by ['slice'], view=None, gamma=1, volume='cases', global_mean=0.5)"
    )


# ------------------------------------------------------------------ helpers read the specification
def test_layer_drivers_take_keys_and_view_from_the_record(p2p_result: wise.ScoreResult) -> None:
    backlog = wise.prioritize(p2p_result, "company", view="Finance")
    expected = wise.layer_drivers(p2p_result, "company", view="Finance")
    pd.testing.assert_frame_equal(wise.layer_drivers(p2p_result, backlog), expected)
    with pytest.raises(NormError, match="view"):
        wise.layer_drivers(p2p_result, backlog, view="Logistics")


def test_drivers_refuse_a_record_ranked_on_another_norm(p2p_result: wise.ScoreResult) -> None:
    other = wise.running_p2p_norm().replace_constraint("c1", weight=2.0)
    backlog = wise.prioritize(wise.score(wise.running_p2p_log(), other), "company", view="Finance")
    with pytest.raises(NormError, match="fingerprint"):
        wise.layer_drivers(p2p_result, backlog)
    with pytest.raises(NormError, match="fingerprint"):
        wise.constraint_drivers(p2p_result, backlog)


def test_constraint_drivers_and_penalty_mass_take_the_view_from_the_record(p2p_result: wise.ScoreResult) -> None:
    backlog = wise.prioritize(p2p_result, "vendor", view="Finance")
    pd.testing.assert_frame_equal(
        wise.constraint_drivers(p2p_result, backlog, {"company": "B"}),
        wise.constraint_drivers(p2p_result, "Finance", {"company": "B"}),
    )
    pd.testing.assert_frame_equal(
        wise.penalty_mass(p2p_result, backlog, where={"company": "B"}),
        wise.penalty_mass(p2p_result, "Finance", "vendor", where={"company": "B"}),
    )
    with pytest.raises(NormError, match="by"):
        wise.penalty_mass(p2p_result, "Finance")


def test_drivers_refuse_a_record_ranked_from_a_per_case_frame(p2p_result: wise.ScoreResult) -> None:
    backlog = wise.prioritize(p2p_result.frame("Finance"), "company")
    with pytest.raises(NormError, match="view"):
        wise.constraint_drivers(p2p_result, backlog)


def test_pareto_concentration_and_hotspots_accept_the_record(p2p_result: wise.ScoreResult) -> None:
    backlog = wise.prioritize(p2p_result, "case", view="Finance")
    pd.testing.assert_frame_equal(wise.pareto(backlog), wise.pareto(backlog.table))
    pd.testing.assert_frame_equal(wise.concentration(backlog), wise.concentration(backlog.table))
    pd.testing.assert_frame_equal(wise.hotspot_table(backlog, top=5), wise.hotspot_table(backlog.table, top=5))
    change = wise.compare_periods(backlog, backlog)
    assert (change["PI_change"] == 0.0).all()


def test_top_k_overlap_matches_slices_by_the_keys_of_the_specification() -> None:
    # Both backlogs have one slice with positive mass, at position 0, but it is a different slice.
    first = wise.prioritize(pd.DataFrame({"slice": ["x", "y"], "score": [0.0, 1.0]}), "slice", as_index=False)
    second = wise.prioritize(pd.DataFrame({"slice": ["x", "y"], "score": [1.0, 0.0]}), "slice", as_index=False)
    assert wise.top_k_overlap(first, second, k=1) == 0.0
    assert wise.top_k_overlap(first, first, k=1) == 1.0
    other_keys = wise.prioritize(pd.DataFrame({"s2": ["x", "y"], "score": [0.0, 1.0]}), "s2")
    with pytest.raises(NormError, match="slice keys"):
        wise.top_k_overlap(first, other_keys)


def test_view_agreement_ranks_every_view_with_the_record_parameters(p2p_result: wise.ScoreResult) -> None:
    backlog = wise.prioritize(p2p_result, "case", view="Finance", gamma=1.0)
    pd.testing.assert_frame_equal(
        wise.view_agreement(p2p_result, backlog, k=2), wise.view_agreement(p2p_result, "case", k=2, gamma=1.0)
    )


def test_view_agreement_ranks_every_view_with_the_record_z(p2p_result: wise.ScoreResult) -> None:
    # By vendor, V2 is the only slice below the mean in both views; with z = 0.5 its lower
    # bound stays positive in Logistics only, so the PI_lower top sets are {} and {V2}.
    backlog = wise.prioritize(p2p_result, "vendor", view="Finance", z=0.5)
    lower = wise.view_agreement(p2p_result, backlog, k=2, metric="PI_lower")
    stable = wise.view_agreement(p2p_result, backlog, k=2)
    assert lower.loc[("Finance", "Logistics"), "top2_overlap"] == 0.0
    assert stable.loc[("Finance", "Logistics"), "top2_overlap"] == 1.0
