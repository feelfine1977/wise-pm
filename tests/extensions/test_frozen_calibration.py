"""Fitted references: the default recalculation, and the frozen alternative (E01, F12)."""

from __future__ import annotations

import pandas as pd
import pytest

import wise
from wise.derive import compute_recipe
from wise.errors import EvidenceError, NormError
from wise.evidence import CalibrationRecord, apply_calibration, fit_calibration, recipe_fingerprint

RECIPE = {"name": "touch_index", "kind": "quantile_scale", "attribute": "touches", "q": 0.95}


def log_with(touches):
    log = wise.datasets.running_p2p_log()
    log.add_case_attribute("touches", pd.Series(dict(zip(list("ABCDE"), touches))))
    return log


def norm_with_recipe(recipe=RECIPE):
    return wise.Norm(
        constraints=(wise.NormConstraint("c", "L", wise.Metric("touch_index", threshold=0.5, width=0.5)),),
        layers=(wise.Layer("L"),),
        views=(wise.View("V", constraint_weights={"c": 1.0}),),
        derived_attributes=(recipe,),
    )


# ----------------------------------------------------------- the old default
def test_the_default_recipe_still_refits_on_every_log():
    """Unchanged historical behaviour: quantile_scale is relative to its log."""
    small, large = log_with([1, 2, 3, 4, 5]), log_with([10, 20, 30, 40, 50])
    assert compute_recipe(small, RECIPE).max() == pytest.approx(1.0)
    assert compute_recipe(large, RECIPE).max() == pytest.approx(1.0)
    assert compute_recipe(small, RECIPE)["A"] != pytest.approx(compute_recipe(large, RECIPE)["A"] * 10)


def test_a_degenerate_quantile_still_falls_back_to_one():
    zeros = log_with([0, 0, 0, 0, 0])
    assert list(compute_recipe(zeros, RECIPE)) == [0.0] * 5


# ------------------------------------------------------------- fit and apply
def test_a_fitted_reference_reproduces_the_default_on_its_own_population():
    log = log_with([1, 2, 3, 4, 20])
    fitted = fit_calibration(log, RECIPE)
    pd.testing.assert_series_equal(apply_calibration(log, fitted), compute_recipe(log, RECIPE))
    assert fitted.divisor == pytest.approx(16.8)
    assert fitted.parameters == {"q": 0.95}
    assert fitted.fitting_population["n_units"] == 5
    assert fitted.fitting_population["unit_type"] == "case"


def test_a_frozen_reference_is_not_recomputed_on_later_data():
    """F12: the fixed application keeps its fitted value."""
    first = log_with([1, 2, 3, 4, 20])
    fitted = fit_calibration(first, RECIPE)

    later = log_with([100, 200, 300, 400, 500])
    frozen = apply_calibration(later, fitted)
    recomputed = compute_recipe(later, RECIPE)
    assert list(frozen) == [1.0] * 5, "everything is far above the frozen divisor"
    assert recomputed.max() == pytest.approx(1.0)
    assert recomputed["A"] == pytest.approx(100 / 480.0)
    assert frozen["A"] != pytest.approx(recomputed["A"])


def test_recomputation_yields_a_different_record_with_a_different_identity():
    first, later = log_with([1, 2, 3, 4, 20]), log_with([100, 200, 300, 400, 500])
    a, b = fit_calibration(first, RECIPE), fit_calibration(later, RECIPE)
    assert a.recipe_fingerprint == b.recipe_fingerprint, "same recipe"
    assert a.divisor != b.divisor
    assert a.calibration_id != b.calibration_id, "a different fitted value is a different reference"


def test_a_changed_recipe_creates_a_new_identity_and_is_never_silently_reused():
    log = log_with([1, 2, 3, 4, 20])
    fitted = fit_calibration(log, RECIPE)
    changed = {**RECIPE, "q": 0.5}
    assert recipe_fingerprint(changed) != fitted.recipe_fingerprint
    assert not fitted.matches(changed)
    with pytest.raises(NormError, match="fitted from a different recipe"):
        compute_recipe(log, changed, calibration=fitted)


def test_only_fittable_recipe_kinds_can_be_frozen():
    log = log_with([1, 2, 3, 4, 20])
    with pytest.raises(EvidenceError, match="fits nothing"):
        fit_calibration(log, {"name": "n", "kind": "count", "activities": ["Record Goods Receipt"]})


def test_a_calibration_record_is_immutable_and_validated():
    log = log_with([1, 2, 3, 4, 20])
    fitted = fit_calibration(log, RECIPE)
    with pytest.raises(AttributeError):
        fitted.divisor = 1.0
    with pytest.raises(EvidenceError, match="finite and > 0"):
        CalibrationRecord(
            recipe_name="x",
            kind="quantile_scale",
            attribute="touches",
            divisor=0.0,
            parameters={},
            recipe=RECIPE,
            recipe_fingerprint="f",
            fitting_population={},
        )


# ------------------------------------------------------------------ scoring
def test_score_applies_a_frozen_reference_and_records_it():
    first = log_with([1, 2, 3, 4, 20])
    fitted = fit_calibration(first, RECIPE)
    norm = norm_with_recipe()

    later = log_with([100, 200, 300, 400, 500])
    frozen_run = wise.score(later, norm, calibrations=[fitted], evidence="summary")
    assert list(frozen_run.log.cases["touch_index"]) == [1.0] * 5
    recorded = frozen_run.manifest.calibrations
    assert len(recorded) == 1
    assert recorded[0]["calibration_id"] == fitted.calibration_id
    assert recorded[0]["divisor"] == pytest.approx(16.8)

    default_run = wise.score(log_with([100, 200, 300, 400, 500]), norm)
    assert default_run.manifest.calibrations == ()
    assert default_run.log.cases["touch_index"]["A"] == pytest.approx(100 / 480.0)
    assert frozen_run.scores["V"]["A"] != pytest.approx(default_run.scores["V"]["A"])


def test_a_frozen_and_a_recomputed_run_are_different_configurations():
    first = log_with([1, 2, 3, 4, 20])
    fitted = fit_calibration(first, RECIPE)
    norm = norm_with_recipe()
    frozen = wise.score(log_with([1, 2, 3, 4, 20]), norm, calibrations=[fitted]).manifest
    plain = wise.score(log_with([1, 2, 3, 4, 20]), norm).manifest
    assert frozen.config_fingerprint() != plain.config_fingerprint()


def test_calibrations_for_unknown_or_underived_attributes_are_refused():
    log = log_with([1, 2, 3, 4, 20])
    fitted = fit_calibration(log, RECIPE)
    other = norm_with_recipe({**RECIPE, "name": "something_else"})
    with pytest.raises(EvidenceError, match="does not derive"):
        wise.score(log, other, calibrations=[fitted])
    with pytest.raises(EvidenceError, match="needs derive=True"):
        wise.score(log, norm_with_recipe(), calibrations=[fitted], derive=False)
    with pytest.raises(EvidenceError, match="two calibrations"):
        wise.score(log, norm_with_recipe(), calibrations=[fitted, fitted])
    with pytest.raises(EvidenceError, match="CalibrationRecord"):
        wise.score(log, norm_with_recipe(), calibrations=["not a record"])


def test_a_frozen_column_is_recomputed_by_a_later_uncalibrated_call():
    """The recipe cache must not hand a frozen column to an ordinary call."""
    log = log_with([100, 200, 300, 400, 500])
    fitted = fit_calibration(log_with([1, 2, 3, 4, 20]), RECIPE)
    norm = norm_with_recipe()
    wise.score(log, norm, calibrations=[fitted])
    assert list(log.cases["touch_index"]) == [1.0] * 5
    wise.score(log, norm)
    assert log.cases["touch_index"]["A"] == pytest.approx(100 / 480.0)


def test_the_record_exports_as_plain_json_types():
    fitted = fit_calibration(log_with([1, 2, 3, 4, 20]), RECIPE)
    payload = fitted.to_dict()
    assert payload["kind"] == "quantile_scale"
    assert payload["divisor"] == pytest.approx(16.8)
    assert payload["fitting_population"]["n_observed"] == 5
    assert payload["calibration_id"].startswith("cal-")
    assert pd.Timestamp(payload["fitted_at"]) is not None
