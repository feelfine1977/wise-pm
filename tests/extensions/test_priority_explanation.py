"""Exact signed attribution and one shared comparator (stage S2, item E03).

The acceptance rows X01-X09 of the handoff live here. The two properties the
stage exists for are asserted over and over: the explanation uses the *same*
comparator, mask, keys and aggregation as the ranking, and it never invents a
reference it does not have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import wise
from wise.errors import NotScoredError
from wise.explain import BaselineError, BaselineKind, BaselineSpec, FactKind, explain_priority

TOL = {"rtol": 0, "atol": 1e-12}
LAYERS = ["completeness", "lead_times", "match", "handling", "exceptions"]
GLOBAL_MEAN_FINANCE = 0.7098333333333334


def current(result, view="Finance"):
    """The legacy comparator, named."""
    return BaselineSpec.current_population(
        baseline_id=f"current-population:{view}",
        view=view,
        scoring_mode=result.mode,
        norm_fingerprint=result.norm_fingerprint,
    )


# ---------------------------------------------------------------- X01 parity
@pytest.mark.parametrize("mode", ["flat", "layer_balanced"])
@pytest.mark.parametrize("view", ["Finance", "Logistics"])
def test_a_named_current_population_comparator_reproduces_the_legacy_output(p2p_log, p2p_norm, mode, view):
    result = wise.score(p2p_log, p2p_norm, mode=mode)
    legacy = wise.prioritize(result, "company", view=view, gamma=1.0, z=1.96)
    named = wise.prioritize(result, "company", view=view, gamma=1.0, z=1.96, baseline_spec=current(result, view))
    pd.testing.assert_frame_equal(legacy, named, check_exact=True)
    legacy_drivers = wise.layer_drivers(result, "company", view=view)
    named_drivers = wise.layer_drivers(result, "company", view=view, baseline_spec=current(result, view))
    pd.testing.assert_frame_equal(legacy_drivers, named_drivers, check_exact=True)


def test_a_call_without_a_specification_keeps_exactly_its_historical_attributes(p2p_result):
    legacy = wise.prioritize(p2p_result, "company", view="Finance", gamma=1.0)
    assert set(legacy.attrs) == {"view", "gamma", "baseline", "volume", "by"}
    named = wise.prioritize(p2p_result, "company", view="Finance", gamma=1.0, baseline_spec=current(p2p_result))
    assert named.attrs["baseline_id"] == "current-population:Finance"
    assert named.attrs["comparator"] == "current_population"
    assert named.attrs["comparator_resolved_from"] == "current_population"
    assert named.attrs["explanation_kind"] == "absolute_and_reference_contrast"
    assert named.attrs["baseline_spec"]["kind"] == "current_population"
    # layer_drivers keeps no attrs at all without a specification
    assert wise.layer_drivers(p2p_result, "company", view="Finance").attrs == {}


def test_the_positional_interface_and_the_scalar_baseline_still_work(p2p_result):
    positional = wise.prioritize(p2p_result, "company", "Finance", 1.0, "cases", 1, None, "score", None, True)
    keyword = wise.prioritize(p2p_result, "company", view="Finance", gamma=1.0)
    pd.testing.assert_frame_equal(positional, keyword, check_exact=True)
    fixed = wise.prioritize(p2p_result, "company", view="Finance", baseline=1.0)
    np.testing.assert_allclose(fixed["global_mean"].to_numpy(), [1.0, 1.0], **TOL)


def test_two_comparators_for_one_backlog_is_an_explicit_error(p2p_result):
    spec = BaselineSpec.target(0.9, baseline_id="t")
    for call in (
        lambda: wise.prioritize(p2p_result, "company", view="Finance", baseline=0.9, baseline_spec=spec),
        lambda: explain_priority(p2p_result, "company", "B", view="Finance", baseline=0.9, baseline_spec=spec),
    ):
        with pytest.raises(wise.NormError, match="two comparators for one backlog"):
            call()


def test_the_explanation_reproduces_the_backlog_row_it_explains(p2p_result):
    backlog = wise.prioritize(p2p_result, "company", view="Finance", gamma=1.0)
    packet = explain_priority(p2p_result, "company", "B", view="Finance", gamma=1.0)
    row = backlog.loc["B"]
    p = packet.priority
    assert p.n_scored == int(row["n_cases"]) == 3
    assert p.mean_score == pytest.approx(row["mean_score"], abs=1e-15)
    assert p.priority_index == pytest.approx(row["PI"], abs=1e-12)
    assert p.stable_priority_index == pytest.approx(row["stable_PI"], abs=1e-12)
    assert p.reference_score == pytest.approx(GLOBAL_MEAN_FINANCE, abs=1e-15)
    assert p.rank == 1 and p.n_groups == 2
    assert packet.explanation_kind == "absolute_and_reference_contrast"


def test_the_reconciliation_guard_actually_fires(p2p_result):
    """A tolerance of -1 makes every identity fail: the guard is not decorative."""
    with pytest.raises(BaselineError, match="the explanation and the ranking disagree"):
        explain_priority(p2p_result, "company", "B", view="Finance", atol=-1.0)


def test_an_explanation_needs_a_run_identity(p2p_result):
    frame = p2p_result.frame("Finance").reset_index()
    with pytest.raises(BaselineError, match="carries no run identity"):
        explain_priority(frame, "company", "B")


# ------------------------------------------------- X02 historical comparator
def test_a_historical_profile_makes_the_deltas_sum_to_the_same_gap(p2p_result, previous_period_result):
    spec = BaselineSpec.from_result(previous_period_result, "Finance", baseline_id="2025-Q4", population_id="2025-Q4")
    backlog = wise.prioritize(p2p_result, "company", view="Finance", gamma=1.0, baseline_spec=spec)
    packet = explain_priority(p2p_result, "company", "B", view="Finance", gamma=1.0, baseline_spec=spec)
    assert packet.baseline.kind is BaselineKind.HISTORICAL
    assert packet.reference_score == pytest.approx(spec.reference_score, abs=1e-15)
    assert packet.reference_score == pytest.approx(float(backlog.loc["B", "global_mean"]), abs=1e-15)
    deltas = [layer.delta for layer in packet.layers]
    assert sum(deltas) == pytest.approx(packet.priority.signed_gap, abs=1e-12)
    components = [layer.component for layer in packet.layers]
    assert sum(components) == pytest.approx(packet.priority.stable_priority_index, abs=1e-12)
    # and the reference really is the frozen one, not this population's mean
    assert packet.reference_score != pytest.approx(GLOBAL_MEAN_FINANCE, abs=1e-6)


def test_a_target_with_a_complete_profile_reconciles_too(p2p_result):
    profile = {"completeness": 0.02, "lead_times": 0.03, "match": 0.02, "handling": 0.01, "exceptions": 0.02}
    spec = BaselineSpec.target(None, profile, baseline_id="board-2026", view="Finance", scoring_mode="flat")
    packet = explain_priority(p2p_result, "company", "B", view="Finance", gamma=1.0, baseline_spec=spec)
    assert packet.reference_score == pytest.approx(0.90, abs=1e-12)
    assert sum(layer.delta for layer in packet.layers) == pytest.approx(packet.priority.signed_gap, abs=1e-12)
    assert sum(layer.clipped_component for layer in packet.layers) == pytest.approx(
        packet.priority.stable_priority_index, abs=1e-12
    )


def test_the_layer_deltas_come_from_layer_drivers_itself(p2p_result, previous_period_result):
    """Not recomputed: the explanation reads the same frame the user would."""
    spec = BaselineSpec.from_result(previous_period_result, "Finance")
    drivers = wise.layer_drivers(p2p_result, "company", view="Finance", baseline_spec=spec)
    packet = explain_priority(p2p_result, "company", "B", view="Finance", baseline_spec=spec)
    for layer in packet.layers:
        assert layer.delta == drivers.loc["B", f"{layer.layer}__delta"]
        assert layer.group_penalty == drivers.loc["B", layer.layer]


# ------------------------------------------------------ X03 scalar-only target
def test_a_scalar_target_ranks_but_allocates_nothing(p2p_result):
    spec = BaselineSpec.target(0.95, baseline_id="board-2026", view="Finance", scoring_mode="flat")
    backlog = wise.prioritize(p2p_result, "company", view="Finance", baseline_spec=spec)
    np.testing.assert_allclose(backlog["global_mean"].to_numpy(), [0.95, 0.95], **TOL)
    assert backlog.attrs["explanation_kind"] == "absolute_only"

    drivers = wise.layer_drivers(p2p_result, "company", view="Finance", baseline_spec=spec)
    assert drivers.attrs["reference_profile_available"] is False
    assert drivers.attrs["unavailable_reason"] == "reference_profile_unavailable"
    assert drivers.attrs["reference_layer_penalties"] is None
    # absolute layer means are retained exactly; the target deltas are null
    legacy = wise.layer_drivers(p2p_result, "company", view="Finance")
    for layer in LAYERS:
        pd.testing.assert_series_equal(drivers[layer], legacy[layer], check_exact=True)
        assert drivers[f"{layer}__delta"].isna().all()
    assert drivers["dominant_layer"].isna().all()

    packet = explain_priority(p2p_result, "company", "B", view="Finance", baseline_spec=spec)
    assert packet.explanation_kind == "absolute_only"
    assert packet.reference_layer_penalties is None
    assert all(layer.delta is None and layer.component is None for layer in packet.layers)
    assert all(layer.group_penalty is not None for layer in packet.layers)
    codes = {q.code.value for q in packet.limitations}
    assert "reference_profile_unavailable" in codes


def test_a_scalar_target_never_borrows_the_current_global_deltas(p2p_result):
    """The trap this stage exists to close: those deltas explain another comparison."""
    spec = BaselineSpec.target(0.95, baseline_id="board-2026", view="Finance", scoring_mode="flat")
    current_deltas = wise.layer_drivers(p2p_result, "company", view="Finance")
    target_drivers = wise.layer_drivers(p2p_result, "company", view="Finance", baseline_spec=spec)
    for layer in LAYERS:
        assert current_deltas[f"{layer}__delta"].notna().any()
        assert target_drivers[f"{layer}__delta"].isna().all()
    packet = explain_priority(p2p_result, "company", "B", view="Finance", baseline_spec=spec)
    for fact in packet.facts_of(FactKind.RELATIVE_PRIORITY):
        if fact.name.startswith("delta."):
            assert fact.value is None


def test_the_scalar_baseline_argument_is_treated_as_the_scalar_target_it_is(p2p_result):
    packet = explain_priority(p2p_result, "company", "B", view="Finance", baseline=1.0)
    assert packet.baseline.kind is BaselineKind.TARGET
    assert packet.baseline.scalar_only and packet.explanation_kind == "absolute_only"
    assert packet.reference_score == 1.0
    assert "reference_profile_unavailable" in {q.code.value for q in packet.limitations}


# ------------------------------------------------------------- X04 mismatch
def test_an_incompatible_comparator_is_refused_by_the_explanation_too(p2p_result):
    spec = BaselineSpec.historical(
        None,
        {"completeness": 0.05, "lead_times": 0.05},
        baseline_id="other-norm",
        view="Finance",
        scoring_mode="flat",
    )
    with pytest.raises(BaselineError, match="layer semantics"):
        explain_priority(p2p_result, "company", "B", view="Finance", baseline_spec=spec)


def test_a_partial_layer_selection_says_that_it_does_not_reconcile(p2p_result):
    packet = explain_priority(p2p_result, "company", "B", view="Finance", layers=["match", "lead_times"])
    assert [layer.layer for layer in packet.layers] == ["match", "lead_times"]
    assert sum(layer.delta for layer in packet.layers) != pytest.approx(packet.priority.signed_gap, abs=1e-6)
    assert "reference_profile_unavailable" in {q.code.value for q in packet.limitations}


# ----------------------------------------------- X05 negative layer offsets
def test_a_positive_gap_keeps_its_negative_layer_offsets(p2p_result):
    packet = explain_priority(p2p_result, "company", "B", view="Finance", gamma=1.0)
    by_layer = {layer.layer: layer for layer in packet.layers}
    assert packet.priority.signed_gap > 0
    assert by_layer["lead_times"].delta < 0 and by_layer["handling"].delta < 0
    assert by_layer["lead_times"].component < 0 and by_layer["handling"].component < 0
    # the negative offsets are part of the total: dropping them breaks the identity
    positive_only = sum(layer.component for layer in packet.layers if layer.component > 0)
    assert positive_only > packet.priority.stable_priority_index
    assert sum(layer.component for layer in packet.layers) == pytest.approx(packet.priority.stable_priority_index, abs=1e-12)
    shares = [layer.share_of_stable_pi for layer in packet.layers]
    assert sum(shares) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------- X06 zero / negative gap
def test_a_non_positive_gap_clips_the_index_and_keeps_the_contrasts(p2p_result):
    packet = explain_priority(p2p_result, "company", "A", view="Finance", gamma=1.0)
    p = packet.priority
    assert p.signed_gap < 0 and p.gap == 0.0
    assert p.priority_index == 0.0 and p.stable_priority_index == 0.0
    assert p.clipped is True
    assert all(layer.clipped_component == 0.0 for layer in packet.layers)
    # the unclipped contrasts survive the clipping and still reconcile
    assert any(layer.delta != 0 for layer in packet.layers)
    assert sum(layer.delta for layer in packet.layers) == pytest.approx(p.signed_gap, abs=1e-12)
    assert sum(layer.component for layer in packet.layers) == pytest.approx(p.volume * p.rho * p.signed_gap, abs=1e-12)
    # no percentage is formed by dividing by a zero net gap
    assert all(layer.share_of_stable_pi is None for layer in packet.layers)
    assert "non_positive_gap_clipped" in {q.code.value for q in packet.limitations}


def test_an_exactly_zero_gap_is_also_clipped_without_a_division():
    frame = pd.DataFrame({"slice": ["a", "b"], "score": [0.5, 0.5], "contrib__L": [0.5, 0.5]})
    backlog = wise.prioritize(frame, "slice")
    assert backlog["gap"].tolist() == [0.0, 0.0]
    assert backlog["PI"].tolist() == [0.0, 0.0]
    drivers = wise.layer_drivers(frame, "slice")
    assert drivers["L__delta"].tolist() == [0.0, 0.0]
    assert drivers["dominant_layer"].isna().all()


# ------------------------------------------------------------- X07 exposure
def test_exposure_scales_the_index_and_never_weights_the_mean(exposure_result):
    by_cases = wise.prioritize(exposure_result, "company", view="Finance")
    by_exposure = wise.prioritize(exposure_result, "company", view="Finance", volume="exposure")
    pd.testing.assert_series_equal(by_cases["mean_score"], by_exposure["mean_score"], check_exact=True)
    pd.testing.assert_series_equal(by_cases["gap"], by_exposure["gap"], check_exact=True)
    assert by_cases["global_mean"].iloc[0] == by_exposure["global_mean"].iloc[0]

    packet = explain_priority(exposure_result, "company", "A", view="Finance", volume="exposure")
    p = packet.priority
    assert p.volume == pytest.approx(200.0)  # company A holds two cases, each with exposure 100
    assert p.mean_score == pytest.approx(float(by_cases.loc["A", "mean_score"]), abs=1e-15)
    assert p.priority_index == pytest.approx(p.volume * p.gap, abs=1e-12)
    assert "exposure_scales_priority" in {q.code.value for q in packet.limitations}


def test_a_zero_exposure_slice_keeps_its_gap_and_loses_only_its_index(exposure_result):
    packet = explain_priority(exposure_result, "company", "B", view="Finance", volume="exposure")
    p = packet.priority
    assert p.volume == 0.0
    assert p.signed_gap > 0 and p.gap > 0
    assert p.priority_index == 0.0 and p.stable_priority_index == 0.0
    assert all(layer.component == 0.0 for layer in packet.layers)
    assert all(layer.share_of_stable_pi is None for layer in packet.layers)
    # the observed penalties are unaffected by the missing exposure
    assert sum(layer.group_penalty for layer in packet.layers) == pytest.approx(1 - p.mean_score, abs=1e-12)


def test_a_custom_volume_column_is_used_as_declared():
    frame = pd.DataFrame(
        {"slice": ["a", "a", "b"], "score": [0.2, 0.4, 0.9], "contrib__L": [0.8, 0.6, 0.1], "items": [3.0, 7.0, 2.0]}
    )
    backlog = wise.prioritize(frame, "slice", volume="items")
    assert backlog.loc["a", "volume"] == 10.0
    assert backlog.loc["a", "PI"] == pytest.approx(10.0 * backlog.loc["a", "gap"], abs=1e-12)


# ------------------------------- X08 minimum support, ties and null keys
def test_the_reference_is_computed_before_the_minimum_support_filter(p2p_result):
    everything = wise.prioritize(p2p_result, "vendor", view="Finance")
    filtered = wise.prioritize(p2p_result, "vendor", view="Finance", min_cases=3)
    assert set(everything.index) == {"V1", "V2"}
    assert len(filtered) < len(everything)
    assert filtered["global_mean"].iloc[0] == pytest.approx(GLOBAL_MEAN_FINANCE, abs=1e-15)


def test_a_slice_below_the_minimum_support_is_explained_but_not_ranked(p2p_result):
    ranked = wise.prioritize(p2p_result, "vendor", view="Finance", min_cases=3)
    assert list(ranked.index) == ["V1"], "V2 has two scored cases and is filtered out"
    packet = explain_priority(p2p_result, "vendor", "V2", view="Finance", min_cases=3)
    assert packet.priority.n_scored == 2
    assert packet.priority.rank is None
    assert packet.priority.reference_score == pytest.approx(GLOBAL_MEAN_FINANCE, abs=1e-15)
    assert "below_minimum_support" in {q.code.value for q in packet.limitations}
    denominator = packet.denominator("reference_population")
    assert denominator.count == 5


def test_null_group_keys_form_their_own_slice_and_are_matched_as_nulls(null_key_result):
    backlog = wise.prioritize(null_key_result, "company", view="Finance")
    assert backlog["n_cases"].sum() == 5
    assert any(pd.isna(key) for key in backlog.index)
    packet = explain_priority(null_key_result, "company", float("nan"), view="Finance")
    assert packet.group_label == "<null>"
    assert packet.priority.n_scored == 1
    codes = {q.code.value for q in packet.limitations}
    assert "null_group_key" in codes and "single_unit_group" in codes
    assert sum(layer.delta for layer in packet.layers) == pytest.approx(packet.priority.signed_gap, abs=1e-12)


def test_a_tie_is_broken_deterministically_and_the_top_slice_is_stable():
    frame = pd.DataFrame({"slice": ["b", "b", "a", "a", "c", "c"], "score": [0.5] * 6, "contrib__L": [0.5] * 6})
    # equal index and equal support: the key order decides, in both input orders
    assert list(wise.prioritize(frame, "slice").index) == ["a", "b", "c"]
    assert list(wise.prioritize(frame.iloc[::-1], "slice").index) == ["a", "b", "c"]
    uneven = pd.DataFrame({"slice": list("bbbaac"), "score": [0.5] * 6})
    assert list(wise.prioritize(uneven, "slice").index) == ["b", "a", "c"]


def test_the_top_ranked_slice_is_chosen_when_no_group_is_named(p2p_result):
    packet = explain_priority(p2p_result, "company", view="Finance", gamma=1.0)
    assert packet.group_key == ("B",)
    assert packet.priority.rank == 1


def test_an_unknown_slice_is_refused(p2p_result):
    with pytest.raises(BaselineError, match="no slice"):
        explain_priority(p2p_result, "company", "Z", view="Finance")
    with pytest.raises(BaselineError, match="has 2 part"):
        explain_priority(p2p_result, "company", ("A", "B"), view="Finance")


def test_a_multi_key_slice_is_addressed_by_tuple(p2p_result):
    packet = explain_priority(p2p_result, ["company", "vendor"], ("B", "V1"), view="Finance")
    assert packet.grouping == ("company", "vendor")
    assert packet.group_key == ("B", "V1")
    assert packet.group_label == "B | V1"
    assert packet.priority.n_scored == 2


def test_a_population_with_nothing_scored_raises_rather_than_reporting_zero(p2p_log, p2p_norm):
    result = wise.score(p2p_log, p2p_norm)
    result.scores["Finance"] = np.nan
    with pytest.raises(NotScoredError, match="no scored cases"):
        wise.prioritize(result, "company", view="Finance")
    with pytest.raises(NotScoredError):
        explain_priority(result, "company", "B", view="Finance")


def test_a_slice_of_one_reports_its_support_and_no_dispersion(p2p_result):
    packet = explain_priority(p2p_result, "case", "E", view="Finance", z=1.96)
    assert packet.priority.n_scored == 1
    assert packet.priority.se is None  # NaN is a status, not a number
    assert packet.priority.gap_lower == 0.0
    assert "single_unit_group" in {q.code.value for q in packet.limitations}


# ------------------------------------------- X09 different scored populations
def test_views_that_score_different_cases_are_disclosed(split_population_result):
    result = split_population_result
    assert result.scores["Left"].notna().sum() == 2
    assert result.scores["Right"].notna().sum() == 2
    assert not result.scores["Left"].notna().equals(result.scores["Right"].notna())
    packet = explain_priority(result, "team", "T1", view="Left")
    contrasts = {c.view: c for c in packet.view_contrasts}
    assert contrasts["Left"].scored_population_differs is False
    # a view in which the slice has no scored unit gets no row of zeros ...
    assert set(contrasts) == {"Left"}
    # ... but the difference is disclosed rather than silently dropped
    disclosure = [q for q in packet.limitations if q.code.value == "different_scored_populations"]
    assert len(disclosure) == 1
    assert "Right" in disclosure[0].message
    assert "no scored unit at all under Right" in disclosure[0].message


def test_a_common_population_reports_no_difference(p2p_result):
    packet = explain_priority(p2p_result, "company", "B", view="Finance")
    assert {c.view for c in packet.view_contrasts} == {"Finance", "Logistics"}
    assert all(c.scored_population_differs is False for c in packet.view_contrasts)
    assert "different_scored_populations" not in {q.code.value for q in packet.limitations}
    # each alternative view is ranked against its own current population, and says so
    for contrast in packet.view_contrasts:
        assert contrast.comparator == f"current-population:{contrast.view}"


def test_an_unknown_contrast_view_is_refused(p2p_result):
    with pytest.raises(BaselineError, match="unknown view"):
        explain_priority(p2p_result, "company", "B", view="Finance", views=["Nope"])


# ------------------------------------------------------------------ packet
def test_the_packet_separates_the_four_kinds_of_claim(p2p_result):
    packet = explain_priority(p2p_result, "company", "B", view="Finance", gamma=1.0)
    observed = packet.facts_of(FactKind.OBSERVED_ASSESSMENT)
    relative = packet.facts_of(FactKind.RELATIVE_PRIORITY)
    alternative = packet.facts_of(FactKind.ALTERNATIVE_VIEW)
    assert {f.name for f in observed} >= {"mean_score", "n_scored", "volume", "penalty.match"}
    assert {f.name for f in relative} >= {"reference_score", "signed_gap", "rho", "PI", "stable_PI", "delta.match"}
    assert {f.name for f in alternative} >= {"Logistics.mean_score", "Logistics.stable_PI"}
    assert all(f.evidence_refs for f in packet.facts)
    assert all(f.fact_id.startswith(("OBS-", "PRI-", "ALT-", "QUAL-")) for f in packet.facts)
    assert packet.fact("PRI-stable_PI").value == pytest.approx(0.15712500000000018, abs=1e-12)
    assert packet.fact("PRI-stable_PI").denominator == "volume"


def test_every_denominator_a_fact_names_exists(p2p_result):
    packet = explain_priority(p2p_result, "company", "B", view="Finance", gamma=1.0)
    names = {d.name for d in packet.denominators}
    for fact in packet.facts:
        if fact.denominator is not None:
            assert fact.denominator in names, fact.fact_id


def test_the_packet_reuses_constraint_drivers_inside_the_slice(p2p_result):
    packet = explain_priority(p2p_result, "company", "B", view="Finance")
    drivers = wise.constraint_drivers(p2p_result, "Finance", {"company": "B"})
    assert [c.constraint_id for c in packet.constraints] == list(drivers.index)
    for component in packet.constraints:
        row = drivers.loc[component.constraint_id]
        assert component.mean_penalty == pytest.approx(row["mean_penalty"], abs=1e-15)
        # an out-of-scope constraint keeps null measures, never a zero violation
        if component.share_in_scope == 0.0:
            assert component.mean_violation is None and component.share_violated is None


def test_witnesses_are_bounded_and_come_from_the_run_that_was_scored(p2p_log, p2p_norm):
    result = wise.score(p2p_log, p2p_norm, evidence="full")
    packet = explain_priority(result, "company", "B", view="Finance", evidence=result.evidence, witness_limit=2)
    assert packet.witnesses
    assert len(packet.evidence_refs) <= 3  # the aggregate reference plus at most two units
    for reference in packet.evidence_refs[1:]:
        assert reference.startswith(result.manifest.run_id)
        assert result.evidence.record(reference) is not None
    absence = [w for w in packet.witnesses if w.kind.value == "absence"]
    for witness in absence:
        assert witness.event_id is None and witness.search is not None


def test_the_run_record_of_an_explanation_is_a_backlog_record(p2p_result):
    packet = explain_priority(p2p_result, "company", "B", view="Finance", gamma=1.0, min_cases=1)
    assert packet.manifest.stage == "backlog"
    assert packet.manifest.is_complete_backlog_run
    assert packet.manifest.priority.comparator.startswith("current_population:")
    assert packet.manifest.priority.comparator_value == pytest.approx(GLOBAL_MEAN_FINANCE, abs=1e-15)
    assert packet.run["priority"]["grouping"] == ["company"]
    # the score record itself is untouched
    assert p2p_result.manifest.stage == "score"


def test_the_explanation_id_is_deterministic_for_one_run(p2p_result):
    first = explain_priority(p2p_result, "company", "B", view="Finance", gamma=1.0)
    second = explain_priority(p2p_result, "company", "B", view="Finance", gamma=1.0)
    assert first.explanation_id == second.explanation_id
    other = explain_priority(p2p_result, "company", "B", view="Finance", gamma=2.0)
    assert other.explanation_id != first.explanation_id


def test_the_layer_frame_and_fact_frame_are_tables_of_the_same_numbers(p2p_result):
    packet = explain_priority(p2p_result, "company", "B", view="Finance", gamma=1.0)
    layers = packet.layer_frame()
    assert list(layers.index) == LAYERS
    assert layers["delta"].sum() == pytest.approx(packet.priority.signed_gap, abs=1e-12)
    facts = packet.fact_frame()
    assert facts.loc["PRI-signed_gap", "value"] == pytest.approx(packet.priority.signed_gap, abs=1e-15)
    with pytest.raises(wise.errors.EvidenceError, match="unknown fact id"):
        packet.fact("PRI-nope")


# ----------------------------------------- equivalence with the base commit
BASE_COMMIT = "df5db50b839cc124b489a269894f5a2bfe7dc634"

#: The equivalence claim, split into the two legs that make it up. Both are
#: asserted below, and ``test_the_reported_equivalence_count_is_the_committed_one``
#: checks that the number the documentation reports is this number and no other.
RESULT_CONFIGURATIONS = 112
FRAME_CONFIGURATIONS = 68
BASE_COMMIT_CONFIGURATIONS = RESULT_CONFIGURATIONS + FRAME_CONFIGURATIONS


def _base_prioritization(tmp_path):
    """The unmodified ``prioritization`` module of the reviewed base commit.

    Read with ``git show`` — no state-changing command, no checkout — and
    skipped wherever the repository is not available (an installed wheel, an
    unpacked sdist). Its relative imports are rewritten to absolute ones so
    that the base module can run beside the current package.
    """
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    if not (repo / ".git").exists():
        pytest.skip("not a git checkout: the base commit is not available here")
    proc = subprocess.run(["git", "show", f"{BASE_COMMIT}:src/wise/prioritization.py"], cwd=repo, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"base commit {BASE_COMMIT[:12]} is not in this checkout")
    source = proc.stdout.replace("\nfrom .constraints import", "\nfrom wise.constraints import")
    source = source.replace("\nfrom .errors import", "\nfrom wise.errors import")
    source = source.replace("\nfrom .scoring import", "\nfrom wise.scoring import")
    path = tmp_path / "base_prioritization.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("base_prioritization", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["base_prioritization"] = module
    spec.loader.exec_module(module)
    return module


def test_the_default_path_is_byte_identical_to_the_base_commit(p2p_log, p2p_norm, tmp_path):
    """The strongest available statement of "old calls keep their values"."""
    base = _base_prioritization(tmp_path)
    checked = 0
    for mode in ("flat", "layer_balanced"):
        result = wise.score(p2p_log, p2p_norm, mode=mode)
        for view in result.views:
            for by in ("company", "vendor", "case", ["company", "vendor"]):
                for gamma in (0.0, 1.0, 20.0):
                    for baseline in (None, 0.9):
                        old = base.prioritize(result, by, view=view, gamma=gamma, z=1.96, baseline=baseline)
                        new = wise.prioritize(result, by, view=view, gamma=gamma, z=1.96, baseline=baseline)
                        pd.testing.assert_frame_equal(old, new, check_exact=True)
                        assert old.attrs == new.attrs
                        checked += 1
                old = base.layer_drivers(result, by, view=view)
                new = wise.layer_drivers(result, by, view=view)
                pd.testing.assert_frame_equal(old, new, check_exact=True)
                assert old.attrs == new.attrs == {}
                checked += 1
    assert checked == RESULT_CONFIGURATIONS  # 2 modes x 2 views x 4 groupings x (3 gammas x 2 baselines + drivers)


def test_the_untouched_helpers_are_byte_identical_to_the_base_commit(tmp_path):
    """Everything from ``constraint_drivers`` onward was not edited at all."""
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    if not (repo / ".git").exists():
        pytest.skip("not a git checkout: the base commit is not available here")
    proc = subprocess.run(["git", "show", f"{BASE_COMMIT}:src/wise/prioritization.py"], cwd=repo, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"base commit {BASE_COMMIT[:12]} is not in this checkout")
    marker = "def constraint_drivers"
    current = (repo / "src" / "wise" / "prioritization.py").read_text(encoding="utf-8")
    assert proc.stdout.split(marker, 1)[1] == current.split(marker, 1)[1]


def test_the_assessment_unit_is_declared_and_carried_through_every_number(p2p_result):
    """Support, gamma and the index are all counted in the declared unit."""
    packet = explain_priority(p2p_result, "company", "B", view="Finance", gamma=1.0, unit_type="purchase order item")
    units = {fact.name: fact.unit for fact in packet.facts}
    assert units["n_scored"] == "purchase order item"
    assert units["gamma"] == "purchase order item", "gamma is in the same unit as the support"
    assert units["stable_PI"] == "score x purchase order items"
    assert {d.unit_of_counting for d in packet.denominators} == {"purchase order item", "slices"}
    assert packet.baseline.unit_type == "purchase order item"
    assert packet.unit_type == "purchase order item"


def test_a_comparator_for_another_unit_type_is_refused(p2p_result):
    from wise.explain import BaselineSpec as Spec

    spec = Spec.target(0.9, baseline_id="objects", view="Finance", scoring_mode="flat", unit_type="invoice")
    with pytest.raises(BaselineError, match="assessment-unit type"):
        explain_priority(p2p_result, "company", "B", view="Finance", baseline_spec=spec, unit_type="case")


def _bare_backlog_frame(n_cases: int = 6_000) -> pd.DataFrame:
    """A per-case score frame of the shape ``wise_analytics.whatif`` builds.

    No :class:`~wise.ScoreResult` anywhere: slice columns, a ``score`` column
    with unscored rows, null keys, an ``exposure`` column and a second numeric
    column usable as a custom volume, plus the ``contrib__<layer>`` columns
    :func:`wise.layer_drivers` needs. This is the path CI protected nowhere.
    """
    rng = np.random.default_rng(20240907)
    companies = np.array(["A", "B", "C", "D", None], dtype=object)[rng.integers(0, 5, n_cases)]
    vendors = np.array(["V1", "V2", "V3"], dtype=object)[rng.integers(0, 3, n_cases)]
    score = rng.random(n_cases)
    score[rng.random(n_cases) < 0.07] = np.nan  # unscored rows are dropped, never zeroed
    frame = pd.DataFrame(
        {
            "company": companies,
            "vendor": vendors,
            "score": score,
            "exposure": rng.integers(0, 500, n_cases).astype(float),
            "amount": rng.random(n_cases) * 1_000.0,
        },
        index=pd.Index([f"case-{i}" for i in range(n_cases)], name="case"),
    )
    remaining = 1.0 - np.nan_to_num(score, nan=0.0)
    weights = rng.dirichlet(np.ones(len(LAYERS)), n_cases)
    for position, layer in enumerate(LAYERS):
        column = remaining * weights[:, position]
        frame[f"contrib__{layer}"] = np.where(np.isnan(score), np.nan, column)
    return frame


def test_the_bare_frame_path_is_byte_identical_to_the_base_commit(tmp_path):
    """The leg the checkpoint reported and the repository did not contain.

    ``wise_analytics.whatif`` builds frames rather than ``ScoreResult``s and
    reads ``backlog.attrs["baseline"]``, so this is the path the application
    actually depends on. It covers what the review named: the three volume
    modes, minimum support, the ``z`` parameter, the scalar baseline and both
    ``as_index`` settings, on frames with null keys and unscored rows.
    """
    base = _base_prioritization(tmp_path)
    frame = _bare_backlog_frame()
    checked = 0

    for volume in ("cases", "exposure", "amount"):
        for min_cases in (1, 3):
            for z in (None, 1.96):
                for baseline in (None, 0.9):
                    for as_index in (True, False):
                        kwargs = dict(volume=volume, min_cases=min_cases, z=z, baseline=baseline, as_index=as_index)
                        old = base.prioritize(frame, "company", **kwargs)
                        new = wise.prioritize(frame, "company", **kwargs)
                        pd.testing.assert_frame_equal(old, new, check_exact=True)
                        assert old.attrs == new.attrs
                        checked += 1

    for volume in ("cases", "exposure"):
        for min_cases in (1, 3):
            for z in (None, 1.96):
                for baseline in (None, 0.9):
                    kwargs = dict(volume=volume, min_cases=min_cases, z=z, baseline=baseline)
                    old = base.prioritize(frame, ["company", "vendor"], **kwargs)
                    new = wise.prioritize(frame, ["company", "vendor"], **kwargs)
                    pd.testing.assert_frame_equal(old, new, check_exact=True)
                    assert old.attrs == new.attrs
                    checked += 1

    for by in ("company", ["company", "vendor"]):
        for as_index in (True, False):
            old = base.layer_drivers(frame, by, as_index=as_index)
            new = wise.layer_drivers(frame, by, as_index=as_index)
            pd.testing.assert_frame_equal(old, new, check_exact=True)
            assert old.attrs == new.attrs == {}
            checked += 1

    assert checked == FRAME_CONFIGURATIONS  # 48 + 16 prioritize legs + 4 layer_drivers legs


def test_the_reported_equivalence_count_is_the_committed_one():
    """The number in the documentation is the number the tests assert.

    The review found "180 configurations byte-identical" reported while 112
    were committed. The 68 bare-frame configurations are committed now, and
    this test refuses to let the two drift apart again.
    """
    from pathlib import Path

    assert BASE_COMMIT_CONFIGURATIONS == 180
    repo = Path(__file__).resolve().parents[2]
    checkpoint = repo / "docs" / "development" / "extension-checkpoint.md"
    if not checkpoint.exists():  # pragma: no cover - absent from a wheel
        pytest.skip("development documentation is not shipped in this distribution")
    text = checkpoint.read_text(encoding="utf-8")
    assert f"**{BASE_COMMIT_CONFIGURATIONS} configurations byte-identical**" in text
    assert f"{FRAME_CONFIGURATIONS} on a bare score frame" in text
    assert f"{RESULT_CONFIGURATIONS} on `ScoreResult`s" in text
    status = (repo / "docs" / "development" / "extension-status.json").read_text(encoding="utf-8")
    assert f"{BASE_COMMIT_CONFIGURATIONS} configurations byte-identical" in status
    assert f"{FRAME_CONFIGURATIONS} on a bare score frame" in status


# ------------------------------- record qualifications reach the explanation
def _censored_slice_result():
    """One slice of four cases, three of them censored; a second, clean slice."""
    rows = []
    for case, company, days in [
        ("x1", "X", [("GR", 0), ("INV", 2)]),
        ("x2", "X", [("GR", 0)]),
        ("x3", "X", [("GR", 1)]),
        ("x4", "X", [("GR", 2)]),
        ("y1", "Y", [("GR", 0), ("INV", 1)]),
        ("y2", "Y", [("GR", 0), ("INV", 1)]),
    ]:
        for activity, day in days:
            rows.append(
                {
                    "case": case,
                    "activity": activity,
                    "time": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                    "company": company,
                }
            )
    log = wise.EventLog(
        pd.DataFrame(rows),
        case_col="case",
        activity_col="activity",
        timestamp_col="time",
        case_attributes=["company"],
        window=("2024-01-01", "2024-01-26"),
    )
    norm = wise.Norm(
        constraints=(wise.NormConstraint("c_lag", "lead_times", wise.Lag("GR", "INV", delta=10, width=20, missing_b="censor")),),
        layers=(wise.Layer("lead_times"),),
        views=(wise.View("V", constraint_weights={"c_lag": 1.0}),),
        name="censoring",
        version="1",
        scoring_mode="flat",
    )
    return wise.score(log, norm, evidence="summary")


def test_a_slice_built_on_censored_evaluations_says_so():
    """A record-scope qualification changes what an aggregate over it means."""
    from wise.evidence.models import QualificationCode

    result = _censored_slice_result()
    censored = [r for r in result.evidence.records if any(q.code is QualificationCode.LOWER_BOUND for q in r.qualifications)]
    assert len(censored) == 3, "three of the four cases of slice X are censored"

    packet = explain_priority(result, "company", "X", view="V", evidence=result.evidence)
    component = next(c for c in packet.constraints if c.constraint_id == "c_lag")
    assert component.share_evaluated == 1.0, "every check was answered ..."
    assert component.share_lower_bound == pytest.approx(0.75), "... three of the four answers are bounds"

    lower = [q for q in packet.limitations if q.code is QualificationCode.LOWER_BOUND]
    assert len(lower) == 1
    assert lower[0].scope == "group"
    assert "3 evaluations" in lower[0].message
    assert "c_lag (3)" in lower[0].message, "the affected constraint and the count are named"
    assert "bounds" in lower[0].message

    # and it reaches every rendering, because a limitation is not a JSON field
    text = wise.render_explanation(packet, "text")
    assert "lower_bound" in text
    assert "c_lag (3)" in text


def test_without_a_captured_packet_the_censored_share_is_unknown_not_zero():
    result = _censored_slice_result()
    from wise.evidence.models import QualificationCode

    packet = explain_priority(result, "company", "X", view="V")
    component = next(c for c in packet.constraints if c.constraint_id == "c_lag")
    assert component.share_lower_bound is None, "nothing was captured, so nothing is known"
    assert not [q for q in packet.limitations if q.code is QualificationCode.LOWER_BOUND]
    # the clean slice carries no censoring limitation either, with evidence supplied
    clean = explain_priority(result, "company", "Y", view="V", evidence=result.evidence)
    clean_component = next(c for c in clean.constraints if c.constraint_id == "c_lag")
    assert clean_component.share_lower_bound == 0.0
    assert not [q for q in clean.limitations if q.code is QualificationCode.LOWER_BOUND]


def test_a_vacuous_satisfaction_inside_a_slice_is_also_lifted():
    """The same lifting, for the other two record codes the review names."""
    from wise.evidence.models import QualificationCode

    rows = [
        {"case": "a", "activity": "OTHER", "time": pd.Timestamp("2024-01-01"), "company": "X", "amount": 0.0},
        {"case": "b", "activity": "GR", "time": pd.Timestamp("2024-01-01"), "company": "X", "amount": 5.0},
        {"case": "b", "activity": "INV", "time": pd.Timestamp("2024-01-02"), "company": "X", "amount": 5.0},
        {"case": "c", "activity": "GR", "time": pd.Timestamp("2024-01-01"), "company": "Y", "amount": 5.0},
        {"case": "c", "activity": "INV", "time": pd.Timestamp("2024-01-02"), "company": "Y", "amount": 4.0},
    ]
    log = wise.EventLog(
        pd.DataFrame(rows), case_col="case", activity_col="activity", timestamp_col="time", case_attributes=["company"]
    )
    norm = wise.Norm(
        constraints=(wise.NormConstraint("c_bal", "match", wise.Balance("amount", "INV", "amount", "GR", tau=0.0, width=0.5)),),
        layers=(wise.Layer("match"),),
        views=(wise.View("V", constraint_weights={"c_bal": 1.0}),),
        name="balance",
        version="1",
        scoring_mode="flat",
    )
    result = wise.score(log, norm, evidence="summary")
    packet = explain_priority(result, "company", "X", view="V", evidence=result.evidence)
    both_zero = [q for q in packet.limitations if q.code is QualificationCode.BOTH_TOTALS_ZERO]
    assert len(both_zero) == 1 and both_zero[0].scope == "group"
    assert "c_bal (1)" in both_zero[0].message
    assert "rests on an absence" in both_zero[0].message


def test_a_witness_a_bounded_capture_did_not_keep_is_disclosed(p2p_log, p2p_norm):
    """A missing witness must be a visible limitation, not a shorter list."""
    from wise.evidence import capture_evidence
    from wise.evidence.models import QualificationCode

    result = wise.score(p2p_log, p2p_norm, evidence="full")
    details = wise.evaluate_detailed(p2p_log, p2p_norm)[2]
    whole = capture_evidence(result, details=details, mode="full")
    bounded = capture_evidence(result, details=details, mode="full", units=["C"])

    complete = explain_priority(result, "company", "B", view="Finance", evidence=whole)
    assert not [q for q in complete.limitations if q.code is QualificationCode.RECORDS_TRUNCATED]

    partial = explain_priority(result, "company", "B", view="Finance", evidence=bounded)
    dropped = [q for q in partial.limitations if q.code is QualificationCode.RECORDS_TRUNCATED and q.scope == "group"]
    assert len(dropped) == 1
    assert "2 of the 3 cases selected for witnesses" in dropped[0].message
    assert "have no record in the evidence packet" in dropped[0].message
    # the run-scope statement of the same bound travels beside it, not instead of it
    assert [q.scope for q in partial.limitations if q.code is QualificationCode.RECORDS_TRUNCATED] == ["group", "run"]
    assert len(partial.evidence_refs) < len(complete.evidence_refs), "the references really are fewer"
    assert "records_truncated" in wise.render_explanation(partial, "text")
