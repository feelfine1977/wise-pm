"""Properties that must hold for any small log and any valid norm (Hypothesis)."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

import wise
from _support.builders import make_log
from _support.strategies import VIEW_BY_CONSTRAINT, VIEWS, event_logs, log_kwargs, norms

MODES = ["flat", "layer_balanced"]


def _finite(frame: pd.DataFrame) -> np.ndarray:
    values = frame.to_numpy(dtype=float)
    return values[~np.isnan(values)]


def _scaled_view(norm: wise.Norm, factor: float) -> wise.Norm:
    weights = norm.get_view(VIEW_BY_CONSTRAINT).constraint_weights
    assert weights is not None
    scaled = wise.View(VIEW_BY_CONSTRAINT, constraint_weights={k: factor * v for k, v in weights.items()})
    return norm.replace(views=tuple(scaled if v.name == VIEW_BY_CONSTRAINT else v for v in norm.views))


# ------------------------------------------------------------------ 1. violations
@pytest.mark.property
@pytest.mark.spec("IV-B.sat")
@given(log=event_logs(), norm=norms())
def test_every_violation_is_nan_or_within_unit_interval(log, norm):
    finite = _finite(wise.score(log, norm).violations)
    assert ((finite >= 0.0) & (finite <= 1.0)).all()


# ---------------------------------------------------------------------- 2. scores
@pytest.mark.property
@pytest.mark.spec("IV-D.score")
@pytest.mark.parametrize("mode", MODES)
@given(log=event_logs(), norm=norms())
def test_every_score_is_nan_or_within_unit_interval(mode, log, norm):
    finite = _finite(wise.score(log, norm, mode=mode).scores)
    assert ((finite >= -1e-12) & (finite <= 1.0 + 1e-12)).all()


@pytest.mark.property
@pytest.mark.spec("IV-D.decomposition")
@pytest.mark.parametrize("mode", MODES)
@given(log=event_logs(), norm=norms())
def test_layer_contributions_decompose_the_penalty_exactly(mode, log, norm):
    assert wise.score(log, norm, mode=mode).check_decomposition(1e-9) < 1e-9


# -------------------------------------------------------------- 3. scale freedom
@pytest.mark.property
@pytest.mark.spec("IV-C.views")
@given(log=event_logs(), norm=norms(), factor=st.floats(0.01, 100.0, allow_nan=False, allow_infinity=False))
def test_scaling_constraint_view_weights_leaves_scores_unchanged(log, norm, factor):
    expected = wise.score(log, norm).scores
    actual = wise.score(log, _scaled_view(norm, factor)).scores
    pd.testing.assert_frame_equal(actual, expected, rtol=0.0, atol=1e-12)


# ------------------------------------------------------- 4. permutation invariance
@pytest.mark.property
@pytest.mark.spec("IV-A.log")
@given(log=event_logs(), norm=norms(), seed=st.integers(0, 2**16))
def test_shuffling_event_rows_leaves_violations_and_scores_unchanged(log, norm, seed):
    shuffled = wise.EventLog(log.events.sample(frac=1, random_state=seed), **log_kwargs())
    expected, actual = wise.score(log, norm), wise.score(shuffled, norm)
    pd.testing.assert_frame_equal(actual.violations, expected.violations, check_exact=True)
    pd.testing.assert_frame_equal(actual.scores, expected.scores, check_exact=True)


# ------------------------------------------------------ 5. modes that coincide
@pytest.mark.property
@pytest.mark.spec("IV-D.modes")
@given(log=event_logs(), norm=norms())
def test_modes_coincide_on_fully_evaluated_cases_when_applicability_is_empty(log, norm):
    unrestricted = norm.map_constraints(lambda c: c.replace(applicability={}))
    flat = wise.score(log, unrestricted, mode="flat")
    balanced = wise.score(log, unrestricted, mode="layer_balanced")
    assert flat.in_scope.all().all()
    fully_evaluated = flat.applicable.all(axis=1)
    assume(fully_evaluated.any())
    pd.testing.assert_frame_equal(balanced.scores[fully_evaluated], flat.scores[fully_evaluated], rtol=0.0, atol=1e-12)


@pytest.mark.property
@pytest.mark.spec("IV-D.modes")
@given(log=event_logs(), norm=norms(one_per_layer=True))
def test_modes_coincide_when_every_layer_holds_one_constraint(log, norm):
    flat = wise.score(log, norm, mode="flat").scores
    balanced = wise.score(log, norm, mode="layer_balanced").scores
    pd.testing.assert_frame_equal(balanced, flat, rtol=0.0, atol=1e-12)


# ----------------------------------------------------------- 6. prioritisation
@pytest.mark.property
@pytest.mark.spec("IV-E.pi")
@pytest.mark.parametrize("view", VIEWS)
@given(log=event_logs(), norm=norms())
def test_backlog_case_counts_times_means_sum_to_the_scored_total(view, log, norm):
    result = wise.score(log, norm)
    assume(result.scores[view].notna().any())
    backlog = wise.prioritize(result, "region", view=view)
    assert (backlog["n_cases"] * backlog["mean_score"]).sum() == pytest.approx(result.scores[view].dropna().sum())
    assert backlog["n_cases"].sum() == result.scores[view].notna().sum()


@pytest.mark.property
@pytest.mark.spec("IV-E.pi")
@pytest.mark.parametrize("view", VIEWS)
@given(log=event_logs(), norm=norms())
def test_priority_index_is_never_negative(view, log, norm):
    result = wise.score(log, norm)
    assume(result.scores[view].notna().any())
    backlog = wise.prioritize(result, "region", view=view)
    assert (backlog["PI"] >= 0).all()
    assert (backlog["stable_PI"] >= 0).all()


@pytest.mark.property
@pytest.mark.spec("IV-E.shrinkage")
@pytest.mark.parametrize("view", VIEWS)
@given(log=event_logs(), norm=norms(), gamma=st.floats(1e-3, 1e3, allow_nan=False, allow_infinity=False))
def test_shrinkage_never_raises_the_priority_index(view, log, norm, gamma):
    result = wise.score(log, norm)
    assume(result.scores[view].notna().any())
    backlog = wise.prioritize(result, "region", view=view, gamma=gamma)
    assert (backlog["stable_PI"] <= backlog["PI"] + 1e-12).all()
    assert (backlog["stable_gap"] <= backlog["gap"] + 1e-12).all()


# ---------------------------------------------------------- 7. JSON round trip
@pytest.mark.property
@pytest.mark.spec("IV-C.norm")
@given(log=event_logs(), norm=norms())
def test_norm_json_round_trip_reproduces_every_result_frame(log, norm):
    reloaded = wise.Norm.loads(norm.dumps())
    expected, actual = wise.score(log, norm), wise.score(log, reloaded)
    pd.testing.assert_frame_equal(actual.violations, expected.violations, check_exact=True)
    pd.testing.assert_frame_equal(actual.in_scope, expected.in_scope, check_exact=True)
    pd.testing.assert_frame_equal(actual.scores, expected.scores, check_exact=True)
    for view in expected.views:
        pd.testing.assert_frame_equal(actual.contributions[view], expected.contributions[view], check_exact=True)


@pytest.mark.property
@pytest.mark.spec("IV-C.norm")
@given(norm=norms())
def test_norm_fingerprint_is_stable_across_json_round_trip(norm):
    reloaded = wise.Norm.loads(norm.dumps())
    assert reloaded.fingerprint() == norm.fingerprint()
    assert reloaded.to_dict() == norm.to_dict()


# ----------------------------------------------------------------- 8. unscored
@pytest.mark.property
@pytest.mark.spec("IV-D.unscored")
@pytest.mark.parametrize("view", VIEWS)
@given(log=event_logs(), norm=norms())
def test_case_is_unscored_iff_no_evaluated_constraint_has_positive_weight(view, log, norm):
    result = wise.score(log, norm)
    positive = norm.weight_vector(view).to_numpy() > 0
    none_evaluated = (result.applicable.to_numpy() & positive).sum(axis=1) == 0
    np.testing.assert_array_equal(result.scores[view].isna().to_numpy(), none_evaluated)


# -------------------------------------------------------- 9. effective weights
@pytest.mark.property
@pytest.mark.spec("IV-D.weights")
@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("mode", MODES)
@given(log=event_logs(), norm=norms())
def test_effective_weights_sum_to_one_on_scored_cases(mode, view, log, norm):
    result = wise.score(log, norm, mode=mode)
    scored = result.scores[view].notna()
    row_sums = result.effective_weights(view)[scored].sum(axis=1)
    np.testing.assert_allclose(row_sums.to_numpy(), 1.0, rtol=0.0, atol=1e-12)


@pytest.mark.property
@pytest.mark.spec("IV-D.weights")
@pytest.mark.parametrize("view", VIEWS)
@given(log=event_logs(), norm=norms())
def test_effective_weights_are_zero_outside_the_scope(view, log, norm):
    result = wise.score(log, norm)
    weights = result.effective_weights(view).to_numpy()
    assert (weights[~result.in_scope.to_numpy()] == 0.0).all()


# ------------------------------------------------- 10. missing attribute scope
MISSING_ATTRIBUTE_RULES: dict[str, dict[str, Any]] = {
    "in": {"attr": "flow_type", "in": ["X"]},
    "eq": {"attr": "flow_type", "eq": "X"},
    "gt": {"attr": "amount", "gt": 0},
    "gte": {"attr": "amount", "gte": 5},
    "lt": {"attr": "amount", "lt": 10},
    "lte": {"attr": "amount", "lte": 5},
}


@pytest.mark.spec("IV-C.applicability")
@pytest.mark.parametrize("op", sorted(MISSING_ATTRIBUTE_RULES))
def test_case_with_missing_attribute_is_out_of_scope(op):
    log = make_log([("c1", "A", 0, 5.0, "X"), ("c2", "A", 0, None, None)], attrs=("flow_type", "amount"))
    nc = wise.NormConstraint("c", "L", wise.Presence("A"), applicability=MISSING_ATTRIBUTE_RULES[op])
    assert nc.applies_to(log.cases, log).tolist() == [True, False]
