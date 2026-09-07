"""The shared comparator specification (stage S2, item E03).

`BaselineSpec` is the object that stops one backlog resting on two
comparisons. These tests pin what it validates, what it refuses, and — just as
importantly — what it deliberately does *not* compare: two populations may
differ; two assessment semantics may not.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import wise
from wise.explain import (
    AGGREGATION,
    BASELINE_SCHEMA_VERSION,
    SCORE_CONVENTION,
    AssessmentContext,
    BaselineError,
    BaselineKind,
    BaselineSpec,
    explain_priority,
    load_baseline,
    norm_context,
    resolve_baseline,
)

LAYERS = ["completeness", "lead_times", "match", "handling", "exceptions"]
PROFILE = {"completeness": 0.04, "lead_times": 0.10, "match": 0.06, "handling": 0.02, "exceptions": 0.03}
PROFILE_SCORE = 0.75


# --------------------------------------------------------------- validation
def test_score_and_profile_must_agree_within_the_documented_tolerance():
    spec = BaselineSpec.historical(PROFILE_SCORE, PROFILE, baseline_id="2025")
    assert spec.reference_score == pytest.approx(PROFILE_SCORE, abs=1e-12)
    # inside the tolerance: accepted, and the stated score is kept
    tight = BaselineSpec.historical(PROFILE_SCORE + 5e-10, PROFILE, baseline_id="2025")
    assert tight.reference_score == pytest.approx(PROFILE_SCORE + 5e-10, abs=1e-15)
    with pytest.raises(BaselineError, match="disagree"):
        BaselineSpec.historical(0.5, PROFILE, baseline_id="2025")
    # and a caller may state the tolerance it wants
    with pytest.raises(BaselineError, match="above the tolerance"):
        BaselineSpec.historical(PROFILE_SCORE + 5e-10, PROFILE, baseline_id="2025", tolerance=0.0)


def test_the_score_is_derived_when_only_the_profile_is_given():
    spec = BaselineSpec.historical(None, PROFILE, baseline_id="2025")
    assert spec.reference_score == pytest.approx(PROFILE_SCORE, abs=1e-12)
    assert not spec.scalar_only


@pytest.mark.parametrize(
    ("score", "profile", "match"),
    [
        (float("nan"), None, "must be finite"),
        (float("inf"), None, "must be finite"),
        (1.5, None, r"outside \[0, 1\]"),
        (-0.2, None, r"outside \[0, 1\]"),
        (None, {"a": -0.1}, "non-negative"),
        (None, {"a": float("nan")}, "must be finite"),
        (None, {"a": 0.7, "b": 0.6}, "exceeds one"),
        (None, {}, "empty layer profile is not a profile"),
    ],
)
def test_rejected_scores_and_profiles(score, profile, match):
    with pytest.raises(BaselineError, match=match):
        BaselineSpec(baseline_id="b", kind=BaselineKind.TARGET, reference_score=score, reference_layer_penalties=profile)


def test_a_non_current_comparator_must_state_its_score():
    with pytest.raises(BaselineError, match="must state its reference score"):
        BaselineSpec(baseline_id="b", kind=BaselineKind.TARGET)
    with pytest.raises(BaselineError, match="must state its reference score"):
        BaselineSpec(baseline_id="b", kind=BaselineKind.HISTORICAL)
    # only the current population is resolved from the run itself
    assert BaselineSpec.current_population().reference_score is None


def test_an_unknown_layer_is_never_aligned_with_zero():
    spec = BaselineSpec.historical(None, {"a": 0.1, "b": 0.2}, baseline_id="two-layers")
    assert spec.profile_for(["a", "b"]) == {"a": 0.1, "b": 0.2}
    with pytest.raises(BaselineError, match="an unknown layer is not zero"):
        spec.profile_for(["a", "b", "c"])
    with pytest.raises(BaselineError, match="layers this assessment does not define"):
        spec.profile_for(["a"])


def test_a_scalar_only_comparator_says_so_rather_than_inventing_a_profile():
    spec = BaselineSpec.target(0.95, baseline_id="board-2026")
    assert spec.scalar_only and spec.layer_ids == ()
    assert spec.explanation_kind == "absolute_only"
    with pytest.raises(BaselineError, match="reference_profile_unavailable"):
        spec.profile_for(LAYERS)


# ------------------------------------------------------------ compatibility
def test_different_populations_are_expected_and_never_a_mismatch(p2p_result, previous_period_result):
    """The rule is about assessment semantics, not about population identity."""
    spec = BaselineSpec.from_result(previous_period_result, "Finance", kind="historical", population_id="2025-Q4")
    assert spec.population_id == "2025-Q4"
    assert spec.population_size == 3
    current = AssessmentContext.from_result(p2p_result, "Finance")
    verdict = spec.check_compatible(current)
    assert verdict.compatible and verdict.additive_attribution
    assert verdict.reasons == ()
    # the two really are different populations with different means
    assert spec.reference_score != pytest.approx(float(p2p_result.scores["Finance"].mean()))


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"view": "Logistics"}, "view: baseline 'Logistics'"),
        ({"scoring_mode": "layer_balanced"}, "normalisation"),
        ({"unit_type": "object"}, "assessment-unit type"),
        ({"score_convention": "one_minus_mean_violation"}, "score convention"),
        ({"aggregation": "exposure_weighted_mean"}, "aggregation convention"),
        ({"norm_fingerprint": "0" * 64}, "norm identity"),
        ({"calibrations": ("cal-1",)}, "calibration mapping"),
        ({"reference_layer_penalties": {"a": 0.1, "b": 0.15}}, "layer semantics"),
    ],
)
def test_every_semantic_mismatch_is_named_and_refused(p2p_result, override, match):
    fields = {
        "baseline_id": "2025",
        "kind": BaselineKind.HISTORICAL,
        "reference_score": PROFILE_SCORE,
        "reference_layer_penalties": PROFILE,
        "view": "Finance",
        "scoring_mode": "flat",
        "norm_fingerprint": p2p_result.norm_fingerprint,
    }
    fields.update(override)
    if "reference_layer_penalties" in override:
        fields["reference_score"] = None
    spec = BaselineSpec(**fields)
    # the assessment unit is declared by the caller, so the explanation's own
    # context is what a declared unit is compared against
    verdict = spec.check_compatible(AssessmentContext.from_result(p2p_result, "Finance", unit_type="case"))
    assert not verdict.compatible
    assert any(match in reason for reason in verdict.reasons), verdict.reasons
    with pytest.raises(BaselineError, match="not comparable"):
        verdict.raise_if_incompatible(spec.baseline_id)
    # and the public entry points refuse it rather than aligning anything.
    # prioritize aggregates a column of scores and declares no unit of its own,
    # so a declared unit is checked where it is declared: in explain_priority.
    if "unit_type" in override:
        with pytest.raises(BaselineError, match="not comparable"):
            explain_priority(p2p_result, "company", "B", view="Finance", baseline_spec=spec)
        return
    with pytest.raises(BaselineError, match="not comparable"):
        wise.prioritize(p2p_result, "company", view="Finance", baseline_spec=spec)
    with pytest.raises(BaselineError, match="not comparable"):
        wise.layer_drivers(p2p_result, "company", view="Finance", baseline_spec=spec)


def test_a_changed_norm_needs_an_explicit_reviewed_mapping(p2p_result):
    fields = dict(
        baseline_id="2025",
        kind=BaselineKind.HISTORICAL,
        reference_layer_penalties=PROFILE,
        view="Finance",
        scoring_mode="flat",
        norm_fingerprint="1" * 64,
    )
    refused = BaselineSpec(**fields)
    context = AssessmentContext.from_result(p2p_result, "Finance")
    assert not refused.check_compatible(context).compatible
    reviewed = BaselineSpec(**fields, reviewed_mapping={"c1": "c1", "note": "layer ids unchanged, weights re-approved"})
    verdict = reviewed.check_compatible(context)
    assert verdict.compatible
    assert any("reviewed mapping" in note for note in verdict.notes)
    # the mapping authorises the comparison; it does not hide that one happened
    backlog = wise.prioritize(p2p_result, "company", view="Finance", baseline_spec=reviewed)
    assert backlog.attrs["baseline_id"] == "2025"


def test_an_undeclared_field_is_not_assumed_to_match():
    spec = BaselineSpec.target(0.9, baseline_id="t")
    verdict = spec.check_compatible(AssessmentContext(view="Finance", scoring_mode="flat", layer_ids=tuple(LAYERS)))
    assert verdict.compatible
    assert set(verdict.undeclared) == {"view", "scoring_mode", "norm_fingerprint"}


def test_an_assessment_that_cannot_declare_a_field_is_unverifiable_not_incompatible():
    """A bare score frame knows no scoring mode; that is a limitation, not a mismatch."""
    spec = BaselineSpec.target(0.9, baseline_id="t", view="Finance", scoring_mode="flat")
    verdict = spec.check_compatible(AssessmentContext(view="Finance"))
    assert verdict.compatible
    assert set(verdict.unverifiable) == {"scoring_mode", "unit_type"}


def test_norm_context_declares_what_the_norm_knows(p2p_norm):
    context = norm_context(p2p_norm, "Finance")
    assert context.scoring_mode == "flat"
    assert list(context.layer_ids) == LAYERS
    assert context.norm_fingerprint == p2p_norm.fingerprint()
    assert context.score_convention == SCORE_CONVENTION and context.aggregation == AGGREGATION


# ---------------------------------------------------------------- resolving
def test_the_current_population_comparator_resolves_to_the_runs_own_mean(p2p_result):
    spec = BaselineSpec.current_population(view="Finance", scoring_mode="flat")
    scored = p2p_result.scores["Finance"].notna()
    profile = {layer: float(p2p_result.contributions["Finance"][layer][scored].mean()) for layer in LAYERS}
    resolved = resolve_baseline(
        spec,
        AssessmentContext.from_result(p2p_result, "Finance"),
        population_mean=float(p2p_result.scores["Finance"][scored].mean()),
        population_size=int(scored.sum()),
        population_profile=profile,
    )
    assert resolved.resolved_from == "current_population" and resolved.is_global
    assert resolved.reference_score == pytest.approx(0.7098333333333334, abs=1e-12)
    assert sum(resolved.reference_layer_penalties.values()) == pytest.approx(1 - resolved.reference_score, abs=1e-12)


def test_a_stated_scalar_never_borrows_the_current_populations_profile(p2p_result):
    spec = BaselineSpec.target(0.95, baseline_id="t", view="Finance", scoring_mode="flat")
    resolved = resolve_baseline(
        spec,
        AssessmentContext.from_result(p2p_result, "Finance"),
        population_mean=0.7098333333333334,
        population_size=5,
        population_profile={layer: 0.05 for layer in LAYERS},
    )
    assert resolved.reference_layer_penalties is None
    assert resolved.reference_score == 0.95
    assert resolved.explanation_kind == "absolute_only"
    assert not resolved.is_global
    # the run's own mean stays available, so a report can show both
    assert resolved.population_mean == pytest.approx(0.7098333333333334, abs=1e-12)
    with pytest.raises(BaselineError, match="reference_profile_unavailable"):
        resolve_baseline(
            spec,
            AssessmentContext.from_result(p2p_result, "Finance"),
            population_mean=0.7,
            population_size=5,
            require_profile=True,
        )


def test_from_result_freezes_an_identity_that_reconciles(previous_period_result):
    spec = BaselineSpec.from_result(previous_period_result, "Finance")
    assert spec.kind is BaselineKind.HISTORICAL
    assert spec.scoring_mode == "flat"
    assert spec.source_run_id == previous_period_result.manifest.run_id
    assert sum(spec.reference_layer_penalties.values()) == pytest.approx(1 - spec.reference_score, abs=1e-12)
    assert set(spec.reference_layer_penalties) == set(LAYERS)


def test_from_result_refuses_a_population_with_nothing_scored(p2p_norm, p2p_log):
    result = wise.score(p2p_log, p2p_norm)
    result.scores["Finance"] = np.nan
    with pytest.raises(BaselineError, match="no scored case"):
        BaselineSpec.from_result(result, "Finance")


# --------------------------------------------------------------- interchange
def test_round_trip_through_json_is_lossless_and_strict(tmp_path):
    spec = BaselineSpec.historical(
        None, PROFILE, baseline_id="2025", view="Finance", scoring_mode="flat", population_id="last-quarter"
    )
    text = json.dumps(spec.to_dict())
    again = BaselineSpec.from_dict(json.loads(text))
    assert again == spec
    path = tmp_path / "baseline.json"
    path.write_text(text, encoding="utf-8")
    assert load_baseline(path) == spec
    assert spec.to_dict()["schema_version"] == BASELINE_SCHEMA_VERSION


def test_a_baseline_file_is_configuration_and_is_read_strictly(tmp_path):
    path = tmp_path / "b.json"
    path.write_text(json.dumps({"baseline_id": "t", "kind": "target", "reference_score": 0.9, "rm": "-rf"}), encoding="utf-8")
    with pytest.raises(BaselineError, match="unknown keys"):
        load_baseline(path)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(BaselineError, match="must contain a JSON object"):
        load_baseline(path)
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(BaselineError, match="not valid JSON"):
        load_baseline(path)
    path.write_text(json.dumps({"baseline_id": "t", "kind": "target", "reference_score": 0.9, "schema_version": "x"}), "utf-8")
    with pytest.raises(BaselineError, match="unsupported baseline schema version"):
        load_baseline(path)


def test_the_interchange_shape_forces_absolute_only_for_a_scalar_target(p2p_log, p2p_norm):
    from wise.evidence import interchange_schema, to_interchange

    result = wise.score(p2p_log, p2p_norm, evidence="summary")
    scalar = BaselineSpec.target(0.95, baseline_id="board-2026")
    exported = to_interchange(result.evidence, view="Finance", baseline=scalar)
    assert exported["baseline"] == {"baseline_id": "board-2026", "kind": "target", "score": 0.95, "layer_profile": None}
    assert exported["explanation_kind"] == "absolute_only"

    with_profile = BaselineSpec.historical(None, PROFILE, baseline_id="2025")
    exported = to_interchange(result.evidence, view="Finance", baseline=with_profile)
    assert exported["explanation_kind"] == "absolute_and_reference_contrast"
    assert exported["baseline"]["layer_profile"] == PROFILE
    # the shipped contract's own conditional says the same thing
    schema = interchange_schema()
    assert schema["properties"]["baseline"]["properties"]["kind"]["enum"] == ["current_population", "historical", "target"]


def test_an_unresolved_current_population_comparator_cannot_be_exported():
    with pytest.raises(BaselineError, match="resolve it against a run"):
        BaselineSpec.current_population().to_interchange()
