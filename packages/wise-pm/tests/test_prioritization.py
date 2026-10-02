"""Priority Index, shrinkage, exposure weighting, and drill-down helpers."""

import numpy as np
import pandas as pd
import pytest

import wise
from wise.errors import NormError, NotScoredError


def _frame(n_slices=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_slices):
        n = [240, 30, 5][s]
        mean = [0.80, 0.70, 0.60][s]
        for _ in range(n):
            rows.append({"slice": f"s{s}", "score": float(np.clip(rng.normal(mean, 0.02), 0, 1)), "exposure": 10.0 * (s + 1)})
    return pd.DataFrame(rows)


@pytest.mark.paper
@pytest.mark.spec("IV-E.pi")
def test_pi_illustration_from_paper():
    # μ̄ = 0.94, μ_s = 0.80, n_s = 240 → PI = 33.6
    df = pd.DataFrame({"slice": ["big"] * 240 + ["rest"] * 760, "score": [0.80] * 240 + [0.94 + 0.14 * 240 / 760] * 760})
    b = wise.prioritize(df, "slice").table
    assert b.index.name == "slice"
    assert b.loc["big", "global_mean"] == pytest.approx(0.94)
    assert b.loc["big", "gap"] == pytest.approx(0.14)
    assert b.loc["big", "PI"] == pytest.approx(33.6)
    assert b.loc["rest", "PI"] == 0.0
    spec = wise.prioritize(df, "slice").spec
    assert spec.gamma == 0.0
    assert spec.by == ("slice",)
    flat = wise.prioritize(df, ["slice"], as_index=False).table
    assert "slice" in flat.columns


@pytest.mark.paper
@pytest.mark.spec("IV-E.shrinkage")
def test_shrinkage_closed_form():
    df = _frame()
    raw = wise.prioritize(df, ["slice"]).table
    stab = wise.prioritize(df, ["slice"], gamma=30).table
    for s in raw.index:
        n = raw.loc[s, "n_cases"]
        assert stab.loc[s, "stable_gap"] == pytest.approx(n / (n + 30) * raw.loc[s, "gap"])
        assert stab.loc[s, "stable_PI"] == pytest.approx(n * n / (n + 30) * raw.loc[s, "gap"])
    pd.testing.assert_series_equal(raw["stable_PI"], raw["PI"], check_names=False)
    for bad in (-1, np.nan, None):
        with pytest.raises(NormError):
            wise.prioritize(df, ["slice"], gamma=bad)


@pytest.mark.spec("IV-E.exposure")
def test_exposure_weighted_pi_and_min_cases():
    df = _frame()
    b = wise.prioritize(df, ["slice"], volume="exposure").table
    assert b.loc["s1", "volume"] == pytest.approx(30 * 20.0)
    assert b.loc["s1", "PI"] == pytest.approx(600 * b.loc["s1", "gap"])
    assert set(wise.prioritize(df, ["slice"], min_cases=10).table.index) == {"s0", "s1"}
    with pytest.raises(NormError):
        wise.prioritize(df, ["slice"], volume="nope")


def test_conservative_lower_bound_and_singletons():
    df = pd.concat([_frame(), pd.DataFrame([{"slice": "solo", "score": 0.1, "exposure": 1.0}])], ignore_index=True)
    b = wise.prioritize(df, ["slice"], z=1.96).table
    assert (b["PI_lower"] <= b["stable_PI"] + 1e-12).all()
    assert (b.loc[["s0", "s1", "s2"], "se"] > 0).all()
    assert np.isnan(b.loc["solo", "se"])
    assert b.loc["solo", "PI_lower"] == 0.0
    assert b.loc["solo", "PI"] > 0


def test_deterministic_tie_break():
    df = pd.DataFrame({"slice": list("bbbaac"), "score": [0.5] * 6})  # every slice at the global mean → all PI 0
    b1 = wise.prioritize(df, ["slice"]).table
    b2 = wise.prioritize(df.iloc[::-1], ["slice"]).table
    assert list(b1.index) == ["b", "a", "c"] == list(b2.index)  # more cases first, then key order


def test_pareto_and_concentration():
    b = wise.prioritize(_frame(), ["slice"]).table
    p = wise.pareto(b)
    assert p["cum_share"].iloc[-1] == pytest.approx(1.0)
    assert list(p["rank"]) == [1, 2, 3]
    assert p.index.name == "slice"
    c = wise.concentration(b, thresholds=(0.5, 1.0))
    assert c.loc[1.0, "top_k"] <= 3
    assert c.loc[0.5, "top_k"] >= 1
    with pytest.raises(NormError):
        wise.pareto(b, metric="nope")


@pytest.mark.spec("IV-E.gamma")
def test_estimate_gamma_moments():
    rng = np.random.default_rng(1)
    sigma, tau = 0.02, 0.08
    means = rng.normal(0.7, tau, size=40)
    rows = [{"slice": f"s{i}", "score": rng.normal(m, sigma)} for i, m in enumerate(means) for _ in range(50)]
    g = wise.estimate_gamma(pd.DataFrame(rows), ["slice"])
    assert g == pytest.approx(sigma**2 / tau**2, rel=0.5)
    same = pd.DataFrame({"slice": ["a"] * 50 + ["b"] * 50, "score": [0.5] * 100})
    assert wise.estimate_gamma(same, ["slice"]) == float("inf")
    with pytest.raises(NotScoredError):
        wise.estimate_gamma(pd.DataFrame({"slice": ["a"], "score": [0.5]}), ["slice"])


def test_running_example_backlog_and_drivers(p2p_result):
    b = wise.prioritize(p2p_result, ["company"], view="Finance").table
    assert b.loc["B", "PI"] > 0
    assert b.loc["A", "PI"] == 0.0
    ld = wise.layer_drivers(p2p_result, ["company"], view="Finance")
    assert ld.loc["B", "completeness__delta"] > 0
    assert ld.loc["B", "dominant_layer"] == "match"
    assert ld.loc["A", "dominant_layer"] == "lead_times"
    cd = wise.constraint_drivers(p2p_result, "Finance", {"company": "B"})
    assert cd.index[0] in {"c1", "c2", "c3"}
    assert cd.loc["c4", "share_in_scope"] == 0.0
    assert cd.loc["c4", "share_evaluated"] == 0.0
    assert wise.constraint_drivers(p2p_result, "Finance").loc["c6", "share_violated"] == pytest.approx(0.2)
    with pytest.raises(NormError):
        wise.constraint_drivers(p2p_result, "Finance", {"nope": 1})


def test_view_agreement_overlap_and_methods(p2p_result):
    va = wise.view_agreement(p2p_result, "case", k=2)
    assert va.index.tolist() == [("Finance", "Logistics")]
    # Finance worst two: E, A ; Logistics worst two: E, B → Jaccard 1/3
    assert va["top2_overlap"].iloc[0] == pytest.approx(1 / 3)
    pytest.importorskip("scipy")
    sp = wise.view_agreement(p2p_result, "case", k=2, method="spearman")
    assert -1 <= sp["score_correlation"].iloc[0] <= 1
    a = wise.prioritize(p2p_result, ["case"], view="Finance").table
    assert wise.top_k_overlap(a, a, k=3) == 1.0
    # slices without priority mass are not part of the top-k
    zero = pd.DataFrame({"slice": list("xyzw"), "score": [0.9, 0.9, 0.9, 0.1]})
    b1 = wise.prioritize(zero, "slice").table
    b2 = wise.prioritize(zero.iloc[::-1], "slice").table
    assert wise.top_k_overlap(b1, b2, k=3) == 1.0


def test_hotspot_table_and_penalty_mass(p2p_result):
    b = wise.prioritize(p2p_result, ["case"], view="Finance").table
    h = wise.hotspot_table(b, top=5, drivers=wise.layer_drivers(p2p_result, ["case"], view="Finance"))
    assert set(h["hotspot"]) <= {"reservoir", "severity", "mechanism"}
    assert "dominant_layer" in h.columns
    df = _frame()
    hb = wise.hotspot_table(wise.prioritize(df, "slice").table)
    assert "s0" not in hb.index  # above the baseline: no priority
    assert hb.loc["s1", "hotspot"] == "reservoir"
    assert hb.loc["s2", "hotspot"] == "severity"
    pm = wise.penalty_mass(p2p_result, "Finance", "vendor")
    assert pm.index.tolist() == ["V2", "V1"]
    assert pm["cum_share"].iloc[-1] == pytest.approx(1.0)
    pm_b = wise.penalty_mass(p2p_result, "Finance", "vendor", where={"company": "B"})
    assert pm_b["n_cases"].sum() == 3


def test_baseline_and_compare_periods():
    df = _frame()
    b = wise.prioritize(df, ["slice"], baseline=1.0).table
    assert (b["global_mean"] == 1.0).all()
    assert (b["PI"] > 0).all()
    now = wise.prioritize(df.assign(score=df["score"] + 0.05), ["slice"], baseline=1.0).table
    cmp_ = wise.compare_periods(b, now)
    assert (cmp_["gap_change"] < 0).all()
    assert set(cmp_.columns) >= {"stable_PI_prev", "stable_PI_now", "PI_change"}


def test_slice_by_case_id_index(p2p_result):
    b = wise.prioritize(p2p_result, "case", view="Finance").table
    assert b.loc["E", "PI"] > b.loc["A", "PI"] > 0
    assert b.loc["B", "PI"] == 0.0
    # a frame whose case id is both index and column
    f = p2p_result.frame("Finance").assign(case=lambda d: d.index)
    assert wise.prioritize(f, "case").table.loc["E", "PI"] > 0
    # the case id listed as a case attribute
    log = wise.EventLog(
        wise.running_p2p_events(),
        case_col="case",
        activity_col="activity",
        timestamp_col="time",
        case_attributes=["case", "company", "flow_type"],
    )
    res = wise.score(log, wise.running_p2p_norm())
    assert wise.penalty_mass(res, "Finance", "case").loc["E", "penalty_mass"] == pytest.approx(0.85)
    assert wise.prioritize(res, "case", view="Finance").table.loc["E", "PI"] > 0
    assert wise.layer_drivers(res, "case", view="Finance").loc["E", "dominant_layer"] == "lead_times"
    tab = wise.validation_table(res, "Finance", "case", censored=pd.Series({"E": True}))
    assert tab.loc["E", "censored_share"] == 1.0
    with pytest.raises(NormError):
        wise.prioritize(res, "case", view="Finance", gamma="five")
    assert len(wise.concentration(wise.prioritize(res, "case", view="Finance").table, thresholds=0.8)) == 1
    assert "match" in wise.layer_drivers(res, "case", view="Finance", layers="match").columns


# ------------------------------------------------------------------ slices of cases without a key
@pytest.mark.spec("IV-E")
@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (["b", None, "a"], ["a", "b", wise.MISSING_KEY]),
        ([2, None, 1], [1.0, 2.0, wise.MISSING_KEY]),
        (pd.Categorical(["a", None, "z"], categories=["z", "a"]), ["z", "a", wise.MISSING_KEY]),
    ],
    ids=["strings", "numbers", "categories-in-their-order"],
)
def test_the_missing_slice_ranks_after_every_keyed_slice_on_a_tie(keys, expected):
    # every slice scores the global mean: PI 0 and one case each, so key order decides
    backlog = wise.prioritize(pd.DataFrame({"slice": keys, "score": [0.5] * 3}), "slice").table
    assert backlog.index.tolist() == expected


@pytest.mark.spec("IV-E")
def test_layer_drivers_label_and_order_the_missing_slice_last():
    df = pd.DataFrame({"slice": [2, None, 1], "score": [0.5] * 3, "contrib__L": [0.25, 0.5, 0.75]})
    drivers = wise.layer_drivers(df, "slice")
    assert drivers.index.tolist() == [1.0, 2.0, wise.MISSING_KEY]
    assert drivers.loc[wise.MISSING_KEY, "L"] == 0.5


def test_a_key_value_equal_to_the_missing_label_is_refused_when_keys_are_missing():
    df = pd.DataFrame({"slice": ["(missing)", None], "score": [0.5, 0.5]})
    with pytest.raises(NormError, match="slice"):
        wise.prioritize(df, "slice")
    assert wise.prioritize(df.dropna(), "slice").table.index.tolist() == ["(missing)"]


@pytest.mark.spec("IV-E")
def test_case_slices_gives_each_case_its_slice_and_score(p2p_result):
    slices = wise.case_slices(p2p_result, ["company", "vendor"], view="Finance")
    assert slices.index.equals(p2p_result.cases.index)
    assert list(slices.columns) == ["company", "vendor", "score"]
    assert slices.loc["E"].tolist() == ["B", "V2", p2p_result.scores["Finance"]["E"]]
    backlog = wise.prioritize(p2p_result, "case", view="Logistics")
    by_record = wise.case_slices(p2p_result, backlog)
    assert list(by_record.columns) == ["case", "score"]
    pd.testing.assert_series_equal(by_record["score"], p2p_result.scores["Logistics"], check_names=False)
    frame = pd.DataFrame({"slice": ["a", None], "s": [0.5, None]}, index=["x", "y"])
    slices = wise.case_slices(frame, "slice", score_col="s")
    assert slices.index.tolist() == ["x", "y"]
    assert slices["slice"].tolist() == ["a", "(missing)"]
    assert slices["s"].iloc[0] == 0.5
    assert pd.isna(slices["s"].iloc[1])


@pytest.mark.spec("IV-E")
def test_case_slices_with_a_numeric_key_can_be_ranked_again(p2p_result):
    # the labelled numeric key mixes numbers and the label; ranking it again must order it as prioritize did
    df = pd.DataFrame({"slice": [2, None, 1], "score": [0.5] * 3})
    slices = wise.case_slices(df, "slice")
    assert wise.prioritize(slices, "slice").table.index.tolist() == [1.0, 2.0, wise.MISSING_KEY]


@pytest.mark.spec("IV-E")
def test_diagnostics_label_a_missing_numeric_key(p2p_log, p2p_norm):
    events = p2p_log.events
    events["plant"] = events["case"].map({"A": 1, "B": 2, "C": 1, "D": 2})
    log = wise.EventLog(
        events,
        case_col="case",
        activity_col="activity",
        timestamp_col="time",
        case_attributes=["company", "vendor", "flow_type", "plant"],
    )
    result = wise.score(log, p2p_norm)
    # Finance scores: A 0.6625 and C 0.87 (plant 1), B 29/30 and D 0.9 (plant 2), E 0.15 (no plant);
    # excluding A keeps E in the missing slice, re-ranked against the same baseline: its gap is retained whole
    retained = wise.gap_retained(result, "Finance", "plant", exclude=pd.Series({"A": True}))
    assert retained.index.tolist() == [wise.MISSING_KEY, 1.0, 2.0]
    assert retained["n_kept"].tolist() == [1, 1, 2]
    assert retained.loc[wise.MISSING_KEY, "retained"] == 1.0
    table = wise.validation_table(result, "Finance", "plant", censored=pd.Series({"A": True}))
    assert table.loc[wise.MISSING_KEY, "censored_share"] == 0.0
    assert table.loc[1.0, "censored_share"] == 0.5
