"""Case scoring against the paper's Table VI and the layer-decomposition identity."""

import numpy as np
import pandas as pd
import pytest
from conftest import make_log

import wise
from wise.errors import NormError, NotScoredError

TABLE_VI = {
    "A": (0.6625, 0.8875),
    "B": (0.9667, 0.7000),
    "C": (0.8700, 0.9675),
    "D": (0.9000, 0.9000),
    "E": (0.1500, 0.5500),
}


def test_table_vi_scores(p2p_result):
    for case, (fin, log_) in TABLE_VI.items():
        assert p2p_result.scores.loc[case, "Finance"] == pytest.approx(fin, abs=5e-5)
        assert p2p_result.scores.loc[case, "Logistics"] == pytest.approx(log_, abs=5e-5)


def test_table_vi_nonzero_violations(p2p_result):
    V = p2p_result.violations
    assert V.loc["A", "c2"] == pytest.approx(0.75)
    assert V.loc["B", "c5"] == pytest.approx(2 / 3)
    assert V.loc["C", "c3"] == pytest.approx(0.65)
    assert V.loc["D", "c6"] == 1.0
    assert V.loc["E", "c1"] == 1.0 and V.loc["E", "c2"] == 1.0 and V.loc["E", "c3"] == 1.0
    assert V["c4"].isna().all() and not p2p_result.in_scope["c4"].any()
    assert V.drop(columns="c4").fillna(0.0).to_numpy().sum() == pytest.approx(0.75 + 2 / 3 + 0.65 + 1 + 3)


def test_layer_decomposition_exact(p2p_result):
    assert p2p_result.check_decomposition(atol=1e-12) < 1e-12
    contrib = p2p_result.contributions["Finance"]
    assert contrib.loc["A", "lead_times"] == pytest.approx(0.45 * 0.75)
    assert contrib.loc["A"].drop("lead_times").sum() == 0.0
    pen = p2p_result.penalties("Logistics").sum(axis=1)
    pd.testing.assert_series_equal(pen, 1 - p2p_result.scores["Logistics"], check_names=False)


def test_effective_weights_renormalise_over_applicable(p2p_result):
    w = p2p_result.effective_weights("Finance").loc["A"]
    assert w["c4"] == 0.0 and w.sum() == pytest.approx(1.0) and w["c2"] == pytest.approx(0.45)


def test_unscored_cases_are_nan(p2p_log, p2p_norm):
    norm = p2p_norm.map_constraints(lambda c: c.replace(applicability={"flow_type": ["DF1"]}))
    res = wise.score(p2p_log, norm)
    assert res.scores.isna().all().all()
    assert res.contributions["Finance"].isna().all().all()
    with pytest.raises(NotScoredError):
        wise.prioritize(res, ["flow_type"], view="Finance")


def test_scoring_modes(p2p_log, p2p_norm):
    full = p2p_norm.map_constraints(lambda c: c.replace(applicability={}))
    flat = wise.score(p2p_log, full).scores
    bal = wise.score(p2p_log, full, mode="layer_balanced").scores
    pd.testing.assert_frame_equal(flat, bal)  # coincide under full applicability
    res_flat = wise.score(p2p_log, p2p_norm)
    res_bal = wise.score(p2p_log, p2p_norm, mode="layer_balanced")
    res_bal.check_decomposition()
    assert res_flat.scores.loc["C", "Finance"] == pytest.approx(0.87, abs=1e-6)
    assert res_bal.scores.loc["C", "Finance"] == pytest.approx(1 - 0.25 / 1.05 * 0.65, abs=1e-6)
    stored = p2p_norm.replace(scoring_mode="layer_balanced")
    assert stored.scoring_mode == "layer_balanced" and p2p_norm.scoring_mode == "flat"
    assert wise.Norm(p2p_norm.constraints, p2p_norm.layers, p2p_norm.views).scoring_mode == "layer_balanced"
    # the running example under the layer-balanced rule
    lb = res_bal.scores.round(4)
    assert lb.loc["A", "Finance"] == pytest.approx(1 - 0.45 / 1.05 * 0.75, abs=1e-4)
    pd.testing.assert_frame_equal(wise.score(p2p_log, stored).scores, res_bal.scores)
    assert wise.Norm.loads(stored.dumps()).scoring_mode == "layer_balanced"
    with pytest.raises(NormError):
        wise.score(p2p_log, p2p_norm, mode="magic")
    with pytest.raises(NormError):
        p2p_norm.replace(scoring_mode="magic")


def test_frame_summary_and_repr(p2p_result):
    f = p2p_result.frame("Finance")
    assert {"score", "contrib__completeness", "company"} <= set(f.columns)
    wide = p2p_result.frame()
    assert "score__Logistics" in wide.columns and "contrib__Finance__match" in wide.columns
    s = p2p_result.summary()
    assert s.loc["Finance", "n_scored"] == 5
    assert s.loc["Finance", "mean_score"] == pytest.approx(np.mean([v[0] for v in TABLE_VI.values()]), abs=1e-4)
    assert p2p_result.applicability_density() == pytest.approx(5 / 6)
    assert p2p_result.applicability_density(scope=True) == pytest.approx(5 / 6)
    assert "ScoreResult(5 cases" in repr(p2p_result)
    assert p2p_result.norm_fingerprint == wise.running_p2p_norm().fingerprint()
    with pytest.raises(NormError):
        p2p_result.frame("Nope")


def test_scope_vs_evaluable_distinguished():
    rows = [("a", "GR", 0, 1, "DF1"), ("a", "INV", 3, 1, "DF1"), ("b", "GR", 0, 1, "DF1"), ("c", "X", 0, 1, "DF2")]
    log = make_log(rows)
    norm = wise.Norm(
        [
            wise.NormConstraint(
                "lag", "L", wise.Lag("GR", "INV", delta=1, width=1, missing_b="skip"), applicability={"flow_type": ["DF1"]}
            )
        ],
        [wise.Layer("L")],
        [wise.View("v", constraint_weights={"lag": 1})],
    )
    res = wise.score(log, norm)
    assert res.in_scope["lag"].tolist() == [True, True, False]
    assert res.applicable["lag"].tolist() == [True, False, False]  # b is skipped for case b
    assert res.applicability_density(scope=True) == pytest.approx(2 / 3)
    assert res.applicability_density() == pytest.approx(1 / 3)


def test_worst_cases_and_trace(p2p_result):
    w = p2p_result.worst_cases("Finance", n=2)
    assert list(w.index) == ["E", "A"]
    w_b = p2p_result.worst_cases("Finance", where={"company": "B"})
    assert list(w_b.index) == ["E", "C", "D"]
    assert len(p2p_result.trace("B")) == 7
    assert p2p_result.worst_cases("Finance", where={"case": "C"}).index.tolist() == ["C"]


def test_views_argument_and_validation(p2p_log, p2p_norm):
    res = wise.score(p2p_log, p2p_norm, views="Finance")
    assert res.views == ["Finance"]
    assert wise.score(p2p_log, p2p_norm, views=["Finance", "Finance"]).frame().shape[1] == p2p_log.cases.shape[1] + 6
    with pytest.raises(NormError):
        wise.score(p2p_log, p2p_norm, views=["Nope"])


def test_two_stage_weights_match_flat_raw_weights(p2p_norm):
    cons = [c.replace(weight=(0.8 if c.id == "c3" else 0.2 if c.id == "c4" else 1.0)) for c in p2p_norm.constraints]
    view = wise.View(
        "Two", layer_weights={"completeness": 0.2, "lead_times": 0.45, "match": 0.25, "handling": 0.05, "exceptions": 0.05}
    )
    n = wise.Norm(cons, p2p_norm.layers, [view])
    w = n.raw_weights("Two")
    assert w["c3"] == pytest.approx(0.25 * 0.8) and w["c4"] == pytest.approx(0.25 * 0.2)
    assert n.layer_weight_table().loc["match", "Two"] == pytest.approx(0.25)
    assert n.weight_vector("Two").index.tolist() == n.constraint_ids


def test_derived_attributes_run_before_scoring(p2p_log):
    norm = wise.Norm(
        [wise.NormConstraint("m", "L", wise.Metric("n_gr", threshold=2, width=3))],
        [wise.Layer("L")],
        [wise.View("v", constraint_weights={"m": 1})],
        derived_attributes=[{"name": "n_gr", "kind": "count", "activities": ["Record Goods Receipt"]}],
    )
    res = wise.score(p2p_log, norm)
    assert res.scores.loc["B", "v"] == pytest.approx(1 - 2 / 3)
    assert "n_gr" in p2p_log.cases.columns
