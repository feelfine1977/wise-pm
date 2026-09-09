"""Rank sensitivity, and the dependence the resampling is over (M01).

The claim under test is narrow and checkable: **the dependence changes the
answer**. Forty-eight cases nested in four regions are not forty-eight
independent draws, and a resampling that treats them as such reports rank
intervals that are too narrow. The same backlog, resampled by region, gives
wider intervals and different verdicts — and the report says which of the two
it did, in a qualification rather than in a docstring.

The rest of the file pins the three sources apart (sampling, parameter,
construction), the comparator rule that decides whether a replicate recomputes
the reference, and the refusal to call a rank stable when a source was never
varied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import wise
from wise.errors import NormError
from wise.evaluation.sensitivity import (
    APPLICATION_BOOTSTRAP,
    INDEPENDENT_UNITS,
    Dependence,
    RankVerdict,
    combined_stability,
    construction_sensitivity,
    parameter_sensitivity,
    sampling_sensitivity,
)
from wise.evidence.models import QualificationCode
from wise.explain import BaselineSpec

GR = "Record Goods Receipt"
INV = "Record Invoice Receipt"


def nested_log() -> wise.EventLog:
    """Eight teams with a clean lag gradient, nested two per region.

    The nesting is the point: a team's six cases behave alike, and two teams
    share a region, so the number of *independent* things in this log is four,
    not forty-eight.
    """
    rows = []
    teams = [f"T{i}" for i in range(8)]
    case = 0
    for position, team in enumerate(teams):
        lag = 4 + 3 * position
        for j in range(6):
            case += 1
            name = f"c{case:03d}"
            common = {"team": team, "region": f"R{position // 2}", "flow_type": "DF2"}
            rows.append({"case": name, "activity": GR, "day": 0, **common})
            rows.append({"case": name, "activity": INV, "day": lag + (j % 3), **common})
    frame = pd.DataFrame(rows)
    frame["time"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(frame["day"], unit="D")
    return wise.EventLog(
        frame,
        case_col="case",
        activity_col="activity",
        timestamp_col="time",
        case_attributes=["team", "region", "flow_type"],
    )


def lag_norm(mode: str = "flat") -> wise.Norm:
    return wise.Norm(
        constraints=(wise.NormConstraint("c1", "lead_times", wise.Lag(GR, INV, delta=6, width=18, unit="D")),),
        layers=(wise.Layer("lead_times"),),
        views=(wise.View("Finance", constraint_weights={"c1": 1.0}),),
        scoring_mode=mode,
    )


@pytest.fixture(scope="module")
def nested():
    return wise.score(nested_log(), lag_norm())


def spans(report) -> dict[str, int]:
    return {row.label: row.rank_high - row.rank_low for row in report.slices}


# ============================================================ the dependence itself
def test_the_dependence_has_no_default_and_must_be_stated(nested):
    """M01's whole premise: the caller says what one draw is."""
    with pytest.raises(TypeError):
        sampling_sensitivity(nested, "team", view="Finance")  # type: ignore[call-arg]


def test_the_report_states_what_one_draw_is(nested):
    report = sampling_sensitivity(nested, "team", view="Finance", dependence="region", B=100, seed=5)
    assert report.dependence == Dependence(unit="region", column="region", n_draws=4, n_units=48)
    assert not report.dependence.exchangeable
    assert "one draw is one region: 4 draws over 48 scored units" in report.dependence.statement()
    assert report.dependence.statement() in report.statement()


def test_resampling_units_as_independent_draws_is_qualified_as_such(nested):
    exchangeable = sampling_sensitivity(nested, "team", view="Finance", dependence=INDEPENDENT_UNITS, B=100, seed=5)
    codes = {q.code for q in exchangeable.qualifications}
    assert QualificationCode.EXCHANGEABLE_UNITS_ASSUMED in codes
    note = next(q for q in exchangeable.qualifications if q.code is QualificationCode.EXCHANGEABLE_UNITS_ASSUMED)
    assert note.scope == "run" and "too narrow" in note.message

    clustered = sampling_sensitivity(nested, "team", view="Finance", dependence="region", B=100, seed=5)
    assert QualificationCode.EXCHANGEABLE_UNITS_ASSUMED not in {q.code for q in clustered.qualifications}


def test_the_dependence_changes_the_answer_and_not_only_the_wording(nested):
    """The finding M01 exists for: ignoring the nesting narrows every interval."""
    exchangeable = sampling_sensitivity(nested, "team", view="Finance", dependence=INDEPENDENT_UNITS, B=300, seed=11)
    clustered = sampling_sensitivity(nested, "team", view="Finance", dependence="region", B=300, seed=11)
    assert exchangeable.dependence.n_draws == 48 and clustered.dependence.n_draws == 4

    narrow, wide = spans(exchangeable), spans(clustered)
    assert sum(wide.values()) > sum(narrow.values()), "four regions cannot be as informative as forty-eight units"

    crossing = {row.label for row in clustered.slices if row.verdict == RankVerdict.MOVES_ACROSS_THE_BACKLOG.value}
    assert crossing, "with four draws some ranks must be unsafe"
    assert not [row for row in exchangeable.slices if row.verdict == RankVerdict.MOVES_ACROSS_THE_BACKLOG.value], (
        "pretending the units are independent reports every rank as safe"
    )

    # the mechanism, asserted rather than described: drawing blocks makes whole
    # blocks absent, so a per-slice interval is conditional on being present
    assert min(row.present_share for row in exchangeable.slices) > 0.95
    assert max(row.present_share for row in clustered.slices) < 0.8
    assert any("conditional on the slice being present" in line for line in clustered.does_not_cover)


def test_an_unknown_dependence_column_is_refused_by_name(nested):
    with pytest.raises(NormError, match="not on the unit table"):
        sampling_sensitivity(nested, "team", view="Finance", dependence="department", B=10)


# ================================================== rank stability, not only scores
def test_rank_stability_is_reported_and_a_stable_rank_reads_differently(nested):
    """A rank that holds and a rank that crosses the backlog are two claims."""
    report = sampling_sensitivity(nested, "team", view="Finance", dependence="region", B=300, seed=11, k=3)
    verdicts = {row.label: row.verdict for row in report.slices}
    assert RankVerdict.STABLE.value in verdicts.values()
    assert RankVerdict.MOVES_ACROSS_THE_BACKLOG.value in verdicts.values()

    worst = report.of("T7")
    assert worst.point_rank == 1 and worst.verdict == RankVerdict.STABLE.value
    assert worst.rank_low == worst.rank_high == 1
    assert worst.p_top_k == 1.0

    crossing = next(row for row in report.slices if row.verdict == RankVerdict.MOVES_ACROSS_THE_BACKLOG.value)
    assert crossing.rank_span_share >= 0.5
    codes = {q.code for q in report.qualifications}
    assert QualificationCode.RANK_MOVES_UNDER_RESAMPLING in codes


def test_the_table_carries_the_score_interval_beside_the_rank_interval(nested):
    report = sampling_sensitivity(nested, "team", view="Finance", dependence="region", B=100, seed=5)
    table = report.table()
    assert list(table.index.names) == ["team"]
    assert {"rank_low", "rank_high", "metric_low", "metric_high", "present_share", "verdict"} <= set(table.columns)
    assert (table["metric_low"] <= table["metric_high"]).all()
    assert (table["rank_low"] <= table["rank_high"]).all()


def test_every_variant_is_ranked_by_prioritize_itself(nested):
    """The point estimate is the backlog, not a re-derivation of it."""
    report = sampling_sensitivity(nested, "team", view="Finance", dependence="region", B=20, seed=5, gamma=1.0)
    backlog = wise.prioritize(nested, "team", view="Finance", gamma=1.0)
    assert [row.label for row in report.slices] == [str(k) for k in backlog.index]
    assert [row.point_metric for row in report.slices] == pytest.approx(list(backlog["stable_PI"]))


def test_the_same_seed_gives_the_same_report(nested):
    kwargs = {"view": "Finance", "dependence": "region", "B": 60, "seed": 3}
    first = sampling_sensitivity(nested, "team", **kwargs)
    second = sampling_sensitivity(nested, "team", **kwargs)
    assert first.to_dict() == second.to_dict()
    assert sampling_sensitivity(nested, "team", **{**kwargs, "seed": 4}).to_dict() != first.to_dict()


# ============================================== the comparator decides, not a keyword
def test_a_current_population_comparator_moves_with_the_replicate(nested):
    """The reference is a statistic of the units being resampled, so it is redrawn."""
    current = BaselineSpec.current_population(baseline_id="c", view="Finance")
    target = BaselineSpec.target(0.6, baseline_id="board-target", view="Finance")
    assert current.moves_with_the_population and not target.moves_with_the_population

    moving = sampling_sensitivity(nested, "team", view="Finance", dependence="region", B=200, seed=9, baseline_spec=current)
    held = sampling_sensitivity(nested, "team", view="Finance", dependence="region", B=200, seed=9, baseline_spec=target)
    assert [row.point_metric for row in moving.slices] != [row.point_metric for row in held.slices]
    assert moving.settings[0]["baseline"] is None, "recomputed inside every replicate"
    assert held.settings[0]["baseline"] == 0.6, "a target fixed earlier does not absorb this uncertainty"


def test_two_comparators_for_one_backlog_are_refused(nested):
    with pytest.raises(NormError, match="two comparators"):
        sampling_sensitivity(
            nested,
            "team",
            view="Finance",
            dependence="region",
            baseline=0.5,
            baseline_spec=BaselineSpec.target(0.6, baseline_id="t", view="Finance"),
            B=5,
        )


# ================================================================ parameter source
def test_parameter_variation_moves_the_ranking_with_the_data_held_still(nested):
    report = parameter_sensitivity(
        nested,
        "team",
        view="Finance",
        settings=[{"gamma": 0.0}, {"gamma": 1.0}, {"gamma": 20.0}, {"baseline": 0.9}, {"min_cases": 3}],
        k=3,
    )
    assert report.source == "parameter"
    assert report.n_variants == 5 and report.dependence is None
    assert len(report.settings) == 5
    assert any("sampling variability" in line for line in report.does_not_cover)


def test_a_setting_that_is_not_a_ranking_parameter_is_refused(nested):
    with pytest.raises(NormError, match="unknown prioritize setting"):
        parameter_sensitivity(nested, "team", view="Finance", settings=[{"delta": 3.0}])
    with pytest.raises(NormError, match="at least one setting"):
        parameter_sensitivity(nested, "team", view="Finance", settings=[])


# ============================================================= construction source
def test_construction_variation_compares_the_order_two_assessments_produce():
    flat = wise.score(nested_log(), lag_norm("flat"))
    balanced = wise.score(nested_log(), lag_norm("layer_balanced"))
    report = construction_sensitivity({"flat": flat, "layer_balanced": balanced}, "team", view="Finance", k=3)
    assert report.source == "construction"
    assert [s["construction"] for s in report.settings] == ["flat", "layer_balanced"]
    assert report.settings[0]["reference"] is True
    assert any("nobody built" in line for line in report.does_not_cover)

    with pytest.raises(NormError, match="unknown reference construction"):
        construction_sensitivity({"flat": flat}, "team", view="Finance", reference="native")
    with pytest.raises(NormError, match="at least one construction"):
        construction_sensitivity({}, "team", view="Finance")


# ==================================================================== combined
def test_a_stability_verdict_is_the_weakest_of_the_sources_that_were_varied(nested):
    sampling = sampling_sensitivity(nested, "team", view="Finance", dependence="region", B=300, seed=11, k=3)
    parameters = parameter_sensitivity(nested, "team", view="Finance", settings=[{"gamma": 0.0}, {"gamma": 20.0}], k=3)
    constructions = construction_sensitivity(
        {"flat": nested, "layer_balanced": wise.score(nested_log(), lag_norm("layer_balanced"))},
        "team",
        view="Finance",
        k=3,
    )
    combined = combined_stability([sampling, parameters, constructions])
    assert combined.complete
    assert QualificationCode.UNCERTAINTY_SOURCE_NOT_VARIED not in {q.code for q in combined.qualifications}

    table = combined.table()
    order = [
        RankVerdict.INSUFFICIENT_SUPPORT.value,
        RankVerdict.MOVES_ACROSS_THE_BACKLOG.value,
        RankVerdict.MOVES_WITHIN_BAND.value,
        RankVerdict.STABLE.value,
    ]
    for team, row in table.iterrows():
        seen = [row[f"verdict__{source}"] for source in ("sampling", "parameter", "construction")]
        assert row["verdict"] == min(seen, key=order.index), f"{team} was called better than its weakest source"


def test_a_verdict_from_one_source_says_which_sources_it_never_saw(nested):
    only_sampling = sampling_sensitivity(nested, "team", view="Finance", dependence="region", B=50, seed=5)
    combined = combined_stability([only_sampling])
    assert not combined.complete
    note = next(q for q in combined.qualifications if q.code is QualificationCode.UNCERTAINTY_SOURCE_NOT_VARIED)
    assert "parameter" in note.message and "construction" in note.message
    assert combined.to_dict()["sources"] == ["sampling"]


def test_reports_over_different_groupings_are_not_combined(nested):
    by_team = sampling_sensitivity(nested, "team", view="Finance", dependence="region", B=20, seed=5)
    by_region = sampling_sensitivity(nested, "region", view="Finance", dependence="region", B=20, seed=5)
    with pytest.raises(NormError, match="different groupings"):
        combined_stability([by_team, by_region])
    with pytest.raises(NormError, match="at least one report"):
        combined_stability([])


# ================================================= what the interval does not cover
def test_every_report_states_what_it_does_and_does_not_cover(nested):
    reports = [
        sampling_sensitivity(nested, "team", view="Finance", dependence="region", B=20, seed=5),
        parameter_sensitivity(nested, "team", view="Finance", settings=[{"gamma": 0.0}, {"gamma": 1.0}]),
        construction_sensitivity({"flat": nested}, "team", view="Finance"),
    ]
    for report in reports:
        assert report.covers and report.does_not_cover
        assert any("not a decision" in line for line in report.does_not_cover)
        assert any("measurement error in the log" in line for line in report.does_not_cover)
        payload = report.to_dict()
        assert payload["covers"] == list(report.covers)
        assert payload["does_not_cover"] == list(report.does_not_cover)
        assert payload["schema_version"] == "wise-sensitivity/1"


def test_the_module_points_at_the_application_bootstrap_rather_than_repeating_it():
    """The sampling source is already answered next door; that is said, not hidden."""
    import wise.evaluation.sensitivity as module

    assert APPLICATION_BOOTSTRAP == "wise_analytics.uncertainty.bootstrap_backlog"
    assert APPLICATION_BOOTSTRAP in module.__doc__
    assert "does not reimplement" in module.__doc__ or "nothing here reimplements" in module.__doc__


def test_a_bare_frame_is_refused_because_it_carries_no_run(nested):
    with pytest.raises(NormError, match="needs a scored result"):
        sampling_sensitivity(nested.frame("Finance"), "team", dependence="region", B=5)  # type: ignore[arg-type]


def test_a_degenerate_request_is_refused_rather_than_answered(nested):
    with pytest.raises(NormError, match="B must be at least 1"):
        sampling_sensitivity(nested, "team", view="Finance", dependence="region", B=0)
    with pytest.raises(NormError, match="ci must be"):
        sampling_sensitivity(nested, "team", view="Finance", dependence="region", B=5, ci=1.0)


def test_an_object_run_can_be_resampled_too():
    """The unit is not a case, and nothing in this module assumes it is."""
    from _oc_fixtures import case_shaped, case_shaped_spec, legacy_object_norm

    from wise import oc

    log = case_shaped()
    spec = case_shaped_spec()
    result = oc.score_units(log, legacy_object_norm("flat"), oc.build_units(log, spec), spec=spec)
    report = sampling_sensitivity(result, "anchor_id", view="Finance", dependence=INDEPENDENT_UNITS, B=50, seed=2)
    assert report.dependence.n_units == 5
    assert {row.label for row in report.slices} == set("ABCDE")
    assert np.isfinite([row.rank_span_share for row in report.slices]).all()
