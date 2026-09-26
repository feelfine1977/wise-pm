"""Defects C4, C5, C7, C9, C12, C15 and C16 in the norm layer (``wise.norm``, ``wise.derive``).

Regression tests pin the desired behaviour and are ``xfail(strict=True)``
until the fix lands; characterisation tests guard behaviour that is already
right. See ``01-findings-classic-main.md`` and decisions 2 and 4 of 2026-09-26.
"""

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

import wise
from _support.builders import make_log
from wise.derive import validate_recipe
from wise.errors import NormError

#: Smallest valid norm in dictionary form; tests override one key at a time.
BASE: dict[str, Any] = {
    "layers": [{"id": "L"}],
    "views": [{"name": "v", "constraint_weights": {"c": 1}}],
    "constraints": [{"id": "c", "layer": "L", "type": "presence", "params": {"activity": "A"}}],
}
LAG_RECIPE = {"name": "lag_ab", "kind": "lag", "a": ["A"], "b": ["B"]}
QUANTILE_RECIPE = {"name": "scaled", "kind": "quantile_scale", "attribute": "amount", "q": 0.95}


@pytest.fixture
def missing_attr_log():
    """Two cases: case 1 has ``ft="X"`` and ``val=5``; case 2 has both attributes missing."""
    df = pd.DataFrame(
        {
            "case": [1, 1, 2, 2],
            "activity": ["A", "B", "A", "B"],
            "time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "ft": ["X", "X", None, None],
            "val": [5.0, 5.0, np.nan, np.nan],
        }
    )
    return wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="time", case_attributes=["ft", "val"])


def _applies(log, rule):
    return wise.NormConstraint("c", "L", wise.Presence("A"), applicability=rule).applies_to(log.cases).tolist()


def _norm_with_recipe(recipe):
    return wise.Norm(
        (wise.NormConstraint("m", "L", wise.Metric(recipe["name"], 1, 1)),),
        (wise.Layer("L"),),
        (wise.View("v", constraint_weights={"m": 1}),),
        derived_attributes=(recipe,),
    )


# --- C4: applicability scope of a case whose attribute is missing ------------------------


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="C4: ne/not_in put a case with a missing attribute in scope (NaN != v is True)",
)
@pytest.mark.parametrize("rule", [{"attr": "ft", "ne": "X"}, {"attr": "ft", "not_in": ["X"]}], ids=["ne", "not_in"])
def test_missing_attribute_is_out_of_scope_under_negated_operators(missing_attr_log, rule):
    assert _applies(missing_attr_log, rule) == [False, False]


@pytest.mark.spec("IV-C")
@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        pytest.param({"attr": "ft", "eq": "X"}, [True, False], id="eq"),
        pytest.param({"attr": "ft", "in": ["X"]}, [True, False], id="in"),
        pytest.param({"attr": "val", "gt": 1}, [True, False], id="gt"),
        pytest.param({"attr": "val", "gte": 5}, [True, False], id="gte"),
        pytest.param({"attr": "val", "lt": 10}, [True, False], id="lt"),
        pytest.param({"attr": "val", "lte": 5}, [True, False], id="lte"),
        pytest.param({"attr": "ft", "isna": True}, [False, True], id="isna"),
        pytest.param({"attr": "ft", "notna": True}, [True, False], id="notna"),
    ],
)
def test_missing_attribute_scope_under_remaining_leaf_operators(missing_attr_log, rule, expected):
    assert _applies(missing_attr_log, rule) == expected


# --- C5: recipe options are not validated -------------------------------------------------


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="C5: validate_recipe accepts unknown activation/response/agg values and unknown keys",
)
@pytest.mark.parametrize(
    "recipe",
    [
        pytest.param({**LAG_RECIPE, "activation": "each"}, id="lag-activation-each"),
        pytest.param({**LAG_RECIPE, "activation": "bogus"}, id="lag-activation-bogus"),
        pytest.param({**LAG_RECIPE, "response": "bogus"}, id="lag-response-bogus"),
        pytest.param({"name": "amt", "kind": "agg", "column": "amount", "agg": "nonsense"}, id="agg-nonsense"),
        pytest.param({"name": "amt", "kind": "agg", "column": "amount", "agg": "describe"}, id="agg-describe"),
        pytest.param({"name": "n", "kind": "count", "activities": ["A"], "activitis": ["A"]}, id="unknown-key"),
    ],
)
def test_validate_recipe_rejects_bad_options(recipe):
    with pytest.raises(NormError):
        validate_recipe(recipe)


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="C5: a lag recipe with activation 'each' silently runs as 'last' in EventLog.derive",
)
def test_derive_rejects_lag_recipe_with_unknown_activation(missing_attr_log):
    with pytest.raises(NormError):
        missing_attr_log.derive([{**LAG_RECIPE, "activation": "each"}])


# --- C7: nested containers of frozen objects are mutable ----------------------------------


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="C7: View.constraint_weights is a plain dict, so item assignment succeeds silently",
)
def test_view_constraint_weights_cannot_be_assigned_in_place():
    v = wise.View("F", constraint_weights={"c1": 1.0})
    assert v.constraint_weights is not None
    with pytest.raises(TypeError):
        v.constraint_weights["c1"] = -1
    assert dict(v.constraint_weights) == {"c1": 1.0}


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="C7: NormConstraint.applicability holds plain lists, so clear() succeeds silently",
)
def test_norm_constraint_applicability_values_cannot_be_cleared_in_place():
    nc = wise.NormConstraint("c", "L", wise.Presence("A"), applicability={"flow_type": ["DF1"]})
    # A frozen value is a tuple (AttributeError on clear) or a frozen list (TypeError).
    with pytest.raises((TypeError, AttributeError)):
        nc.applicability["flow_type"].clear()
    assert list(nc.applicability["flow_type"]) == ["DF1"]


# --- C9: serialisation leaks non-Wise exceptions -----------------------------------------


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="C9: Norm.dumps() raises a bare TypeError for ndarray metadata",
)
def test_dumps_with_ndarray_metadata_serialises_or_raises_norm_error(p2p_norm):
    n = p2p_norm.replace(metadata={"a": np.arange(3)})
    try:
        text = n.dumps()
    except NormError:
        return  # rejecting unserialisable metadata up front is acceptable too
    assert json.loads(text)["metadata"] == {"a": [0, 1, 2]}


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="C9: Norm.dumps() raises a bare TypeError for pandas Timedelta metadata",
)
def test_dumps_with_timedelta_metadata_serialises_or_raises_norm_error(p2p_norm):
    n = p2p_norm.replace(metadata={"sla": pd.Timedelta("1D")})
    try:
        text = n.dumps()
    except NormError:
        return  # rejecting unserialisable metadata up front is acceptable too
    assert isinstance(json.loads(text)["metadata"]["sla"], str)


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    raises=json.JSONDecodeError,
    reason="C9: Norm.from_json leaks json.JSONDecodeError for malformed text",
)
def test_from_json_wraps_malformed_text_in_norm_error():
    with pytest.raises(NormError):
        wise.Norm.from_json("  {not json")


# --- C12: fingerprint covers free-form metadata ------------------------------------------


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="C12: fingerprint() hashes to_dict() including metadata, so provenance changes it",
)
def test_fingerprint_is_unchanged_when_only_metadata_changes(p2p_norm):
    assert p2p_norm.replace(metadata={"note": "x"}).fingerprint() == p2p_norm.fingerprint()


def test_fingerprint_changes_with_within_layer_weight(p2p_norm):
    assert p2p_norm.replace_constraint("c1", weight=2.0).fingerprint() != p2p_norm.fingerprint()


def test_fingerprint_changes_with_view_weight(p2p_norm):
    finance = p2p_norm.get_view("Finance")
    heavier = wise.View("Finance", constraint_weights={**finance.constraint_weights, "c1": 0.4})
    edited = p2p_norm.replace(views=tuple(heavier if v.name == "Finance" else v for v in p2p_norm.views))
    assert edited.fingerprint() != p2p_norm.fingerprint()


# --- C15 (decision 4): quantile_scale must carry a frozen divisor -------------------------


@pytest.mark.regression
@pytest.mark.spec("decision:2026-09-26/4")
@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="C15: quantile_scale without a stored divisor is accepted and normalises by the scored log",
)
def test_quantile_scale_without_divisor_is_rejected():
    with pytest.raises(NormError):
        validate_recipe(QUANTILE_RECIPE)


@pytest.mark.spec("decision:2026-09-26/4")
def test_quantile_scale_with_stored_divisor_is_accepted_and_round_trips():
    recipe = {**QUANTILE_RECIPE, "divisor": 12.5}
    validate_recipe(recipe)
    norm = _norm_with_recipe(recipe)
    assert wise.Norm.loads(norm.dumps()).derived_attributes[0]["divisor"] == 12.5


@pytest.mark.regression
@pytest.mark.spec("decision:2026-09-26/4")
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="C15: quantile_scale divides by the scored log's quantile instead of the stored divisor",
)
def test_quantile_scale_divides_by_stored_divisor():
    log = make_log([(c, "A", 0, float(c), "F") for c in (1, 2, 3, 4)], attrs=("amount",))
    log.derive([{**QUANTILE_RECIPE, "divisor": 2.0}])
    assert log.cases["scaled"].tolist() == pytest.approx([0.5, 1.0, 1.0, 1.0])


# --- C16 (decision 2): eval recipes are removed -----------------------------------------


@pytest.mark.regression
@pytest.mark.spec("decision:2026-09-26/2")
@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="C16: eval recipes execute arbitrary pandas expressions from a norm file",
)
def test_eval_recipe_is_rejected():
    with pytest.raises(NormError):
        validate_recipe({"name": "twice", "kind": "eval", "expr": "n_events * 2"})


# --- characterisation: loading, round trip and editing already behave --------------------


def test_unknown_top_level_key_raises_norm_error():
    with pytest.raises(NormError, match="unknown top-level"):
        wise.Norm.from_dict({**BASE, "extra": 1})


def test_unknown_constraint_key_raises_norm_error():
    with pytest.raises(NormError, match="unknown keys"):
        wise.Norm.from_dict({**BASE, "constraints": [{**BASE["constraints"][0], "applicabilty": {}}]})


@pytest.mark.parametrize("schema_version", [99, "abc"], ids=["newer", "non-integer"])
def test_unsupported_schema_version_raises_norm_error(schema_version):
    with pytest.raises(NormError):
        wise.Norm.from_dict({**BASE, "schema_version": schema_version})


def test_json_string_round_trip_preserves_dict_and_fingerprint(p2p_norm):
    again = wise.Norm.loads(p2p_norm.dumps())
    assert again.to_dict() == p2p_norm.to_dict()
    assert again.fingerprint() == p2p_norm.fingerprint()


def test_replace_constraint_returns_new_norm_and_keeps_original(p2p_norm):
    edited = p2p_norm.replace_constraint("c4", applicability={})
    assert edited is not p2p_norm
    assert edited.get_constraint("c4").applicability == {}
    assert p2p_norm.get_constraint("c4").applicability == {"flow_type": ["DF1"]}
