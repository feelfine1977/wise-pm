"""Scoring defects from the ledger (C3, C10, C13) and the scoring behaviour that must stay.

Regression tests pin the desired behaviour and are ``xfail(strict=True)`` until the
fix lands; characterisation tests are plain tests that already pass.
"""

import pandas as pd
import pytest

import wise
from wise.errors import NormError, WiseError

P2P_CONSTRAINT_IDS = ["c1", "c2", "c3", "c4", "c5", "c6"]
COUNT_GR_RECIPE = {"name": "n_gr", "kind": "count", "activities": ["Record Goods Receipt"]}


def norm_with_recipe_n_gr() -> wise.Norm:
    """One Metric constraint on the derived attribute ``n_gr`` (see test_scoring.py)."""
    return wise.Norm(
        (wise.NormConstraint("m", "L", wise.Metric("n_gr", threshold=2, width=3)),),
        (wise.Layer("L"),),
        (wise.View("v", constraint_weights={"m": 1}),),
        derived_attributes=(COUNT_GR_RECIPE,),
    )


# ------------------------------------------------------------------ regression
@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="C3a: score() adds the derived column to log.cases and appends it to log.case_attributes",
)
def test_score_leaves_log_columns_and_case_attributes_unchanged(p2p_log):
    columns_before = list(p2p_log.cases.columns)
    attributes_before = list(p2p_log.case_attributes)

    wise.score(p2p_log, norm_with_recipe_n_gr())

    assert list(p2p_log.cases.columns) == columns_before
    assert list(p2p_log.case_attributes) == attributes_before


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="C3b: score() silently overwrites a user case attribute that shares a recipe name",
)
def test_user_case_attribute_sharing_recipe_name_is_not_silently_overwritten(p2p_log):
    p2p_log.add_case_attribute("n_gr", pd.Series(99.0, index=p2p_log.case_ids))

    try:
        wise.score(p2p_log, norm_with_recipe_n_gr())
    except NormError:
        pass  # refusing the clash is the other acceptable resolution
    else:
        assert (p2p_log.cases["n_gr"] == 99.0).all()


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="C10: check_decomposition raises AssertionError instead of a WiseError subclass",
)
def test_check_decomposition_on_tampered_result_raises_wise_error(p2p_result):
    p2p_result.contributions["Finance"].loc["A", "lead_times"] += 0.5

    with pytest.raises(WiseError):
        p2p_result.check_decomposition()


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="C13: ScoreResult.mode defaults to 'flat' while Norm.scoring_mode defaults to 'layer_balanced'",
)
def test_score_result_mode_default_equals_norm_default_scoring_mode():
    default = wise.ScoreResult.__dataclass_fields__["mode"].default

    assert default == wise.norm.DEFAULT_SCORING_MODE


# ------------------------------------------------------------- characterisation
@pytest.mark.spec("IV-D")
def test_scoring_is_deterministic_across_calls(p2p_log, p2p_norm):
    first = wise.score(p2p_log, p2p_norm)
    second = wise.score(p2p_log, p2p_norm)

    pd.testing.assert_frame_equal(first.violations, second.violations)
    pd.testing.assert_frame_equal(first.scores, second.scores)
    assert first.views == second.views
    for view in first.views:
        pd.testing.assert_frame_equal(first.contributions[view], second.contributions[view])


@pytest.mark.spec("IV-D")
@pytest.mark.parametrize("cid", P2P_CONSTRAINT_IDS)
def test_evaluate_constraint_equals_violation_matrix_column(p2p_log, p2p_norm, cid):
    single = wise.evaluate_constraint(p2p_log, p2p_norm.get_constraint(cid))
    column = wise.violation_matrix(p2p_log, p2p_norm)[cid]

    pd.testing.assert_series_equal(single, column)


@pytest.mark.spec("IV-D")
@pytest.mark.parametrize("mode", ["flat", "layer_balanced"])
@pytest.mark.parametrize("view", ["Finance", "Logistics"])
def test_effective_weights_of_scored_cases_sum_to_one(p2p_log, p2p_norm, mode, view):
    res = wise.score(p2p_log, p2p_norm, mode=mode)
    scored = res.scores[view].notna()
    assert scored.any()

    row_sums = res.effective_weights(view)[scored].sum(axis=1)

    assert row_sums.to_numpy() == pytest.approx(1.0)


@pytest.mark.spec("IV-D")
def test_unscored_cases_have_nan_contributions_in_every_layer(p2p_log, p2p_norm):
    only_company_a = p2p_norm.map_constraints(lambda c: c.replace(applicability={"company": ["A"]}))
    res = wise.score(p2p_log, only_company_a)
    unscored = res.scores["Finance"].isna()
    assert unscored.tolist() == [False, False, True, True, True]

    for view in res.views:
        assert list(res.contributions[view].columns) == p2p_norm.layer_ids
        assert res.contributions[view][unscored].isna().all().all()
        assert res.contributions[view][~unscored].notna().all().all()


def test_result_cases_is_a_copy_of_log_cases(p2p_log, p2p_result):
    assert p2p_result.log is p2p_log
    companies_before = p2p_log.cases["company"].tolist()

    p2p_result.cases["company"] = "tampered"
    p2p_result.cases["extra"] = 1

    assert p2p_log.cases["company"].tolist() == companies_before
    assert "extra" not in p2p_log.cases.columns
