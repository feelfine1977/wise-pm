"""Contract tests for transport into the public native API."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import wise

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_assessments import carrier_log, prepare_native, weighted_coverage, write_prepared
from run_native_parity import require_same_labels, residual, string_index


def example_assessments():
    index = pd.Index(["complete", "partial", "outside", "unscored"], name="case_id")
    severity = pd.DataFrame(
        {"a": [1.0, 1.0, np.nan, np.nan], "b": [0.0, np.nan, 0.0, np.nan], "c": [0.0, 0.0, 1.0, np.nan]}, index=index
    )
    scope = pd.DataFrame(True, index=index, columns=severity.columns)
    scope.loc["outside", "a"] = False
    groups = pd.Series(["x", "x", "y", "y"], index=index)
    return severity, scope, {"a": "L1", "b": "L1", "c": "L2"}, {"Equal": {"a": 1.0, "b": 1.0, "c": 1.0}}, groups


def test_scope_missingness_modes_and_roundtrip(tmp_path):
    severity, scope, layers, weights, groups = example_assessments()
    log, norm, cases = prepare_native(severity, scope, layers, weights, groups, name="contract")
    loaded = write_prepared(tmp_path, cases, norm)
    assert norm.fingerprint() == loaded.fingerprint()
    result = wise.score(carrier_log(pd.read_parquet(tmp_path / "assessment_features.parquet")), loaded)
    pd.testing.assert_frame_equal(result.in_scope, scope.reindex(result.in_scope.index))
    assert np.isnan(result.scores.loc["unscored", "Equal"])
    assert result.scores.loc["complete", "Equal"] == pytest.approx(2 / 3)
    assert result.scores.loc["partial", "Equal"] == pytest.approx(0.5)
    assert weighted_coverage(result, "Equal")["partial"] == pytest.approx(2 / 3)
    assert weighted_coverage(result, "Equal")["outside"] == 1.0
    layered = wise.score(log, norm, mode="layer_balanced")
    assert layered.scores.loc["partial", "Equal"] == pytest.approx(1 / 3)
    backlog = wise.prioritize(result, by="assessment_group", view="Equal", gamma=20)
    assert int(backlog.n_cases.sum()) == 3
    assert result.check_decomposition() < 1e-12


def test_finite_outside_scope_is_rejected():
    severity, scope, layers, weights, groups = example_assessments()
    severity.loc["outside", "a"] = 0
    with pytest.raises(ValueError, match="Out-of-scope"):
        prepare_native(severity, scope, layers, weights, groups, name="invalid")


@pytest.mark.parametrize("value", [np.inf, -0.1, 1.1])
def test_nonfinite_or_out_of_range_severity_is_rejected(value):
    severity, scope, layers, weights, groups = example_assessments()
    severity.iloc[0, 0] = value
    with pytest.raises(ValueError, match="Finite severity"):
        prepare_native(severity, scope, layers, weights, groups, name="invalid")


def test_misalignment_and_nonboolean_scope_are_rejected():
    severity, scope, layers, weights, groups = example_assessments()
    with pytest.raises(ValueError, match="exactly the severity index"):
        prepare_native(severity, scope.iloc[::-1], layers, weights, groups, name="invalid")
    with pytest.raises(ValueError, match="boolean"):
        prepare_native(severity, scope.astype(int), layers, weights, groups, name="invalid")


def test_zero_view_weight_makes_missingness_view_specific():
    severity, scope, layers, weights, groups = example_assessments()
    weights["OnlyB"] = {"a": 0.0, "b": 1.0, "c": 0.0}
    log, norm, _ = prepare_native(severity, scope, layers, weights, groups, name="zero-weight")
    result = wise.score(log, norm)
    assert np.isnan(result.scores.loc["partial", "OnlyB"])
    assert result.scores.loc["outside", "OnlyB"] == 1.0
    assert weighted_coverage(result, "OnlyB")["partial"] == 0.0


def test_native_layer_drivers_retain_negative_offsets():
    index = pd.Index(["a1", "a2", "b1"], name="case_id")
    severity = pd.DataFrame({"c1": [1.0, 1.0, 0.0], "c2": [0.0, 0.0, 0.8]}, index=index)
    scope = pd.DataFrame(True, index=index, columns=severity.columns)
    layers = {"c1": "L1", "c2": "L2"}
    views = {"Equal": {"c1": 1.0, "c2": 1.0}}
    groups = pd.Series(["A", "A", "B"], index=index)
    log, norm, _ = prepare_native(severity, scope, layers, views, groups, name="offsets")
    result = wise.score(log, norm)
    drivers = wise.layer_drivers(result, by="assessment_group", view="Equal")
    signed = drivers.loc["A", ["L1__delta", "L2__delta"]] * drivers.loc["A", "n_cases"]
    assert signed["L1__delta"] == pytest.approx(1 / 3)
    assert signed["L2__delta"] == pytest.approx(-4 / 15)
    priority = wise.prioritize(result, by="assessment_group", view="Equal").loc["A", "PI"]
    assert max(float(signed.sum()), 0) == pytest.approx(priority)
    assert signed.clip(lower=0).sum() > priority


@pytest.mark.parametrize("extra_axis", ["rows", "columns"])
def test_parity_rejects_extra_expected_identity(extra_axis):
    actual = pd.DataFrame({"c1": [0.2]}, index=["case1"])
    expected = actual.copy()
    if extra_axis == "rows":
        expected.loc["case2"] = 0.3
    else:
        expected["c2"] = 0.3
    with pytest.raises(AssertionError, match="exact identity mismatch"):
        residual(actual, expected)


def test_parity_rejects_duplicate_and_mismatched_scope_identities():
    with pytest.raises(AssertionError, match="duplicate identities"):
        residual(pd.Series([0.2], index=["case1"]), pd.Series([0.2, 0.2], index=["case1", "case1"]))
    with pytest.raises(AssertionError, match="exact identity mismatch"):
        require_same_labels(pd.Index(["case1"]), pd.Index(["case1", "extra"]), "scope rows")


def test_string_conversion_rejects_missing_and_literal_nan_collision():
    with pytest.raises(ValueError, match="merge distinct"):
        string_index(pd.Index([np.nan, "nan"], dtype=object))
    with pytest.raises(AssertionError, match="exact identity mismatch"):
        residual(pd.Series([0.2], index=pd.Index([np.nan])), pd.Series([0.2], index=["nan"]))


def test_parity_allows_only_order_changes():
    actual = pd.DataFrame({"c1": [0.2, np.nan], "c2": [0.1, 0.3]}, index=["a", "b"])
    expected = actual.loc[["b", "a"], ["c2", "c1"]]
    assert residual(actual, expected) == 0
