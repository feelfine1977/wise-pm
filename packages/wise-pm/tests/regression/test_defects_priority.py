"""Defect ledger C9 in prioritisation and diagnostics, plus characterisation
of the priority algebra that must survive the fix.

Regression tests pin the *desired* behaviour and are strict xfails until the
fix lands; characterisation tests pass today and must keep passing.
"""

import numpy as np
import pandas as pd
import pytest

import wise
from _support.builders import make_log
from wise.errors import WiseError

# ---------------------------------------------------------------------------
# C9: ``where={attr: [values]}`` must mean membership, not element-wise equality
# ---------------------------------------------------------------------------
# In the running example company A owns cases A, B and company B owns C, D, E.
COMPANY_MEMBERSHIP = pytest.mark.parametrize(
    ("companies", "expected_cases"),
    [(["A", "B"], {"A", "B", "C", "D", "E"}), (["B"], {"C", "D", "E"})],
    ids=["two-companies", "one-company-as-list"],
)


def _scalar_where(companies):
    """The equivalent filter expressed without a list (None when it covers every company)."""
    return None if len(companies) == 2 else {"company": companies[0]}


@pytest.mark.regression
@pytest.mark.xfail(strict=True, raises=ValueError, reason="C29: where with a list compares element-wise ('Lengths must match')")
@COMPANY_MEMBERSHIP
def test_worst_cases_where_list_selects_cases_of_member_companies(p2p_result, companies, expected_cases):
    worst = p2p_result.worst_cases("Finance", where={"company": companies})
    assert set(worst.index) == expected_cases
    assert worst["score"].is_monotonic_increasing


@pytest.mark.regression
@pytest.mark.xfail(strict=True, raises=ValueError, reason="C29: where with a list compares element-wise ('Lengths must match')")
@COMPANY_MEMBERSHIP
def test_constraint_drivers_where_list_equals_union_of_member_companies(p2p_result, companies, expected_cases):
    drivers = wise.constraint_drivers(p2p_result, "Finance", {"company": companies})
    expected = wise.constraint_drivers(p2p_result, "Finance", _scalar_where(companies))
    pd.testing.assert_frame_equal(drivers, expected)
    in_scope_of_members = p2p_result.in_scope.loc[sorted(expected_cases)].mean()
    pd.testing.assert_series_equal(drivers["share_in_scope"].sort_index(), in_scope_of_members.sort_index(), check_names=False)


@pytest.mark.regression
@pytest.mark.xfail(strict=True, raises=ValueError, reason="C29: where with a list compares element-wise ('Lengths must match')")
@COMPANY_MEMBERSHIP
def test_penalty_mass_where_list_equals_union_of_member_companies(p2p_result, companies, expected_cases):
    pm = wise.penalty_mass(p2p_result, "Finance", "vendor", where={"company": companies})
    expected = wise.penalty_mass(p2p_result, "Finance", "vendor", where=_scalar_where(companies))
    pd.testing.assert_frame_equal(pm, expected)
    assert pm["n_cases"].sum() == len(expected_cases)
    assert pm["cum_share"].iloc[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# C9: frames of the wrong shape must be rejected with a library error
# ---------------------------------------------------------------------------


@pytest.mark.regression
# C9 closed 2026-09-27: every public entry point raises a WiseError subclass (decision record 0001, error contract)
def test_hotspot_table_rejects_penalty_mass_frame_with_wise_error(p2p_result):
    not_a_backlog = wise.penalty_mass(p2p_result, "Finance", "vendor")
    with pytest.raises(WiseError):
        wise.hotspot_table(not_a_backlog)


@pytest.mark.regression
# C9 closed 2026-09-27: every public entry point raises a WiseError subclass (decision record 0001, error contract)
def test_compare_periods_rejects_non_backlog_frames_with_wise_error():
    not_a_backlog = pd.DataFrame({"x": [1]})
    with pytest.raises(WiseError):
        wise.compare_periods(not_a_backlog, not_a_backlog.copy())


# ---------------------------------------------------------------------------
# C9: timestamp_outliers must accept a half-open window
# ---------------------------------------------------------------------------


@pytest.mark.regression
# C9 closed 2026-09-27: every public entry point raises a WiseError subclass (decision record 0001, error contract)
@pytest.mark.parametrize(
    ("window", "expected_outlier_days"),
    [((None, "2024-02-01"), {40}), (("2024-01-05", None), {0})],
    ids=["open-start", "open-end"],
)
def test_timestamp_outliers_accepts_half_open_window(window, expected_outlier_days):
    rows = [("a", "X", 0, 1, "F"), ("a", "Y", 5, 1, "F"), ("b", "X", 10, 1, "F"), ("b", "Y", 40, 1, "F")]
    log = make_log(rows)
    out = wise.timestamp_outliers(log, window=window)
    assert out.dtype == bool
    assert len(out) == len(log.events)
    flagged_days = (log.events.loc[out.to_numpy(), "time"] - pd.Timestamp("2024-01-01")).dt.days
    assert set(flagged_days) == expected_outlier_days


# ---------------------------------------------------------------------------
# Characterisation: the priority algebra that must not change
# ---------------------------------------------------------------------------


def _random_frame(seed: int, n_slices: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 40, size=n_slices)
    means = rng.uniform(0.3, 1.0, size=n_slices)
    rows = [
        {"slice": f"s{i}", "score": float(np.clip(rng.normal(m, 0.1), 0, 1))}
        for i, (n, m) in enumerate(zip(sizes, means))
        for _ in range(n)
    ]
    return pd.DataFrame(rows)


@pytest.mark.spec("IV-E")
def test_prioritize_sorts_by_stable_pi_then_n_cases_then_key():
    # baseline 0.5 and dyadic scores keep every gap exact, so the ties are real ties:
    # p: PI 1.0 | r: PI 0.5 (n=2) | q: PI 0.5 (n=1) | b, a, c: PI 0 with n=3, 2, 2 | z: PI 0, n=1
    scores = {"p": [0.0, 0.0], "q": [0.0], "r": [0.25, 0.25], "a": [0.5, 0.5], "b": [0.5] * 3, "c": [0.5, 0.5], "z": [0.75]}
    df = pd.DataFrame([{"slice": s, "score": v} for s, vals in scores.items() for v in vals])
    backlog = wise.prioritize(df, "slice", baseline=0.5)
    assert list(backlog.index) == ["p", "r", "q", "b", "a", "c", "z"]
    assert backlog.loc["r", "stable_PI"] == backlog.loc["q", "stable_PI"] == 0.5


@pytest.mark.spec("IV-E")
@pytest.mark.parametrize(
    ("by", "view"),
    [("company", "Finance"), ("vendor", "Logistics"), ("case", "Finance"), (["company", "vendor"], "Logistics")],
    ids=["company", "vendor", "case", "company-vendor"],
)
def test_pi_is_non_negative_for_every_slice_of_the_running_example(p2p_result, by, view):
    backlog = wise.prioritize(p2p_result, by, view=view, gamma=1.0)
    assert (backlog["PI"] >= 0).all()
    assert (backlog["stable_PI"] >= 0).all()
    above_baseline = backlog["mean_score"] >= backlog["global_mean"]
    assert (backlog.loc[above_baseline, "PI"] == 0.0).all()


@pytest.mark.spec("IV-E")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pi_is_non_negative_for_every_slice_of_random_frames(seed):
    df = _random_frame(seed)
    backlog = wise.prioritize(df, "slice", gamma=5.0)
    assert (backlog["PI"] >= 0).all()
    assert (backlog["stable_PI"] >= 0).all()
    assert (backlog["gap"] == (backlog["global_mean"] - backlog["mean_score"]).clip(lower=0.0)).all()


@pytest.mark.spec("IV-E")
def test_prioritize_with_fixed_baseline_reports_it_as_global_mean(p2p_result):
    backlog = wise.prioritize(p2p_result, "company", view="Finance", baseline=1.0)
    assert (backlog["global_mean"] == 1.0).all()
    assert backlog.attrs["baseline"] == 1.0
    pd.testing.assert_series_equal(backlog["gap"], 1.0 - backlog["mean_score"], check_names=False)


@pytest.mark.parametrize("k", [1, 2, 20])
@pytest.mark.parametrize("by", ["case", ["company", "vendor"]], ids=["single-key", "two-keys"])
def test_top_k_overlap_of_backlog_with_itself_is_one(p2p_result, by, k):
    backlog = wise.prioritize(p2p_result, by, view="Finance")
    assert (backlog["stable_PI"] > 0).any()
    assert wise.top_k_overlap(backlog, backlog, k=k) == 1.0


def test_compare_periods_of_backlog_with_itself_reports_no_change(p2p_result):
    backlog = wise.prioritize(p2p_result, "company", view="Finance")
    change = wise.compare_periods(backlog, backlog)
    assert set(change.index) == set(backlog.index)
    assert (change["PI_change"] == 0.0).all()
    assert (change["gap_change"] == 0.0).all()
    pd.testing.assert_series_equal(change["stable_PI_now"], change["stable_PI_prev"], check_names=False)


@pytest.mark.spec("IV-E")
@pytest.mark.parametrize(
    "per_slice_scores",
    [[0.5, 0.5, 0.5, 0.5], [0.25, 0.75, 0.25, 0.75]],
    ids=["constant-scores", "equal-means-with-spread"],
)
def test_estimate_gamma_is_infinite_when_slice_means_coincide(per_slice_scores):
    df = pd.DataFrame({"slice": ["a"] * 4 + ["b"] * 4 + ["c"] * 4, "score": per_slice_scores * 3})
    assert wise.estimate_gamma(df, "slice") == float("inf")


# ---------------------------------------------------------------- C24: period comparison invents zeros


@pytest.mark.regression
@pytest.mark.xfail(strict=True, raises=AssertionError, reason="C24: compare_periods zero-fills slices absent from one period")
def test_compare_periods_keeps_absent_slices_unknown():
    previous = pd.DataFrame(
        {"n_cases": [10, 5], "stable_gap": [0.2, 0.1], "stable_PI": [2.0, 0.5]}, index=pd.Index(["a", "b"], name="slice")
    )
    current = pd.DataFrame({"n_cases": [12], "stable_gap": [0.1], "stable_PI": [1.2]}, index=pd.Index(["a"], name="slice"))
    out = wise.compare_periods(previous, current)
    assert pd.isna(out.loc["b", "gap_change"]), (
        "a slice that disappeared has no measurable change; it must not read as an improvement of 0.1"
    )
    assert pd.isna(out.loc["b", "PI_change"])


# ---------------------------------------------------------------- C25: negation admits unknown attributes


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="C25: the 'not' combinator brings a case with a missing attribute into scope"
)
def test_negated_rule_keeps_missing_attribute_out_of_scope():
    df = pd.DataFrame(
        {
            "case": [1, 1, 2, 2],
            "activity": ["A", "B", "A", "B"],
            "time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03"]),
            "ft": ["X", None, None, None],
        }
    )
    log = wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="time", case_attributes=["ft"])
    rule = {"not": {"attr": "ft", "eq": "X"}}
    applies = wise.NormConstraint("c", "L", wise.Presence("B"), applicability=rule).applies_to(log.cases).tolist()
    assert applies == [False, False], "case 2 has no ft value; negating eq must not make it eligible"


# ---------------------------------------------------------------- C26–C28: ranking parameters accepted unchecked


def _two_slices() -> pd.DataFrame:
    return pd.DataFrame({"slice": ["a"] * 3 + ["b"] * 2, "score": [0.5, 0.6, 0.7, 0.2, 0.3]})


@pytest.mark.regression
# C26 closed 2026-09-27: input refusal, no valid number changes (docs/semantics/parity.md)
def test_nan_baseline_is_rejected():
    with pytest.raises(WiseError):
        wise.prioritize(_two_slices(), "slice", baseline=float("nan"))


@pytest.mark.regression
# C27 closed 2026-09-27: input refusal, no valid number changes (docs/semantics/parity.md)
def test_negative_z_is_rejected():
    with pytest.raises(WiseError):
        wise.prioritize(_two_slices(), "slice", z=-2)


@pytest.mark.regression
# C28 closed 2026-09-27: input refusal, no valid number changes (docs/semantics/parity.md)
def test_fractional_min_cases_is_rejected():
    with pytest.raises(WiseError):
        wise.prioritize(_two_slices(), "slice", min_cases=2.9)  # type: ignore[arg-type]  # C28: deliberate wrong type
