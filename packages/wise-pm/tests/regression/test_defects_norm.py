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
from wise.selection import THREE_VALUED

# Schema version 2 documents are read here; each read announces the 0.1.0 semantics (LegacySemanticsWarning).
pytestmark = pytest.mark.filterwarnings("ignore::wise.LegacySemanticsWarning")

#: Smallest valid norm in dictionary form; tests override one key at a time.
BASE: dict[str, Any] = {
    "schema_version": 2,
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
@pytest.mark.spec("decision:0001/3")
# C4 closed 2026-10-02: under schema version 3 applicability is three-valued (decision 0001, section 3): a case whose
# attribute is missing is unknown under ne/not_in, is not in scope and is left out of the denominator. Schema
# version 2 keeps the 0.1.0 scope under the legacy profile (section 4).
@pytest.mark.parametrize("rule", [{"attr": "ft", "ne": "X"}, {"attr": "ft", "not_in": ["X"]}], ids=["ne", "not_in"])
def test_missing_attribute_is_unknown_under_negated_operators(missing_attr_log, rule):
    nc = wise.NormConstraint("c", "L", wise.Presence("A"), applicability=rule)
    assert nc.scope(missing_attr_log.cases, missing=THREE_VALUED).tolist() == [False, pd.NA]
    assert nc.applies_to(missing_attr_log.cases, missing=THREE_VALUED).tolist() == [False, False]
    constraints = [{**BASE["constraints"][0], "applicability": rule}]
    corrected = wise.Norm.from_dict({**BASE, "schema_version": 3, "missing_values": "three_valued", "constraints": constraints})
    result = wise.score(missing_attr_log, corrected)
    assert result.in_scope["c"].tolist() == [False, False]
    assert wise.evaluate(corrected.get_constraint("c"), missing_attr_log, semantics=corrected.semantics).reason.astype(
        str
    ).tolist() == ["out_of_scope", "scope_unknown"]
    # c is the only constraint: case 1 is out of its scope and case 2 has no denominator left, so both are unscored
    assert result.scores["v"].isna().tolist() == [True, True]
    legacy = wise.Norm.from_dict({**BASE, "constraints": constraints})
    assert wise.score(missing_attr_log, legacy).scores["v"].tolist() == pytest.approx([np.nan, 1.0], nan_ok=True)


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
@pytest.mark.spec("decision:0001/3")
# C5 closed 2026-10-02: a recipe is parsed strictly into its typed record (decision 0001, section 3)
@pytest.mark.parametrize(
    ("recipe", "key"),
    [
        pytest.param({**LAG_RECIPE, "activation": "each"}, "activation", id="lag-activation-each"),
        pytest.param({**LAG_RECIPE, "activation": "bogus"}, "activation", id="lag-activation-bogus"),
        pytest.param({**LAG_RECIPE, "response": "bogus"}, "response", id="lag-response-bogus"),
        pytest.param({"name": "amt", "kind": "agg", "column": "amount", "agg": "nonsense"}, "agg", id="agg-nonsense"),
        pytest.param({"name": "amt", "kind": "agg", "column": "amount", "agg": "describe"}, "agg", id="agg-describe"),
        pytest.param({"name": "n", "kind": "count", "activities": ["A"], "activitis": ["A"]}, "activitis", id="unknown-key"),
    ],
)
def test_validate_recipe_rejects_bad_options(recipe, key):
    with pytest.raises(NormError, match=rf"derived attribute '{recipe['name']}' \({recipe['kind']}\): (unknown key )?'?{key}"):
        validate_recipe(recipe)
    with pytest.raises(NormError, match=rf"'{recipe['name']}'.*{key}"):
        wise.Norm.from_dict({**BASE, "derived_attributes": [recipe]})


@pytest.mark.regression
@pytest.mark.spec("decision:0001/3")
# C5 closed 2026-10-02: the lag recipe gives one lag per case, so activation 'each' is refused instead of read as 'last'
def test_derive_rejects_lag_recipe_with_unknown_activation(missing_attr_log):
    with pytest.raises(
        NormError, match=r"derived attribute 'lag_ab' \(lag\): activation must be one of \['first', 'last'\], got 'each'"
    ):
        missing_attr_log.derive([{**LAG_RECIPE, "activation": "each"}])
    assert "lag_ab" not in missing_attr_log.cases.columns


# --- C7: nested containers of frozen objects are read-only --------------------------------


@pytest.mark.regression
@pytest.mark.spec("IV-C")
# C7 closed 2026-10-02 (Sec. IV-C: the norm is a versioned artefact, fixed once built): a view's weights are read-only
def test_view_constraint_weights_cannot_be_assigned_in_place():
    v = wise.View("F", constraint_weights={"c1": 1.0})
    assert v.constraint_weights is not None
    with pytest.raises(TypeError):
        v.constraint_weights["c1"] = -1
    assert dict(v.constraint_weights) == {"c1": 1.0}


@pytest.mark.regression
@pytest.mark.spec("IV-C")
# C7 closed 2026-10-02 (Sec. IV-C): the values of an applicability rule are read-only, so the rule cannot change
def test_norm_constraint_applicability_values_cannot_be_cleared_in_place():
    nc = wise.NormConstraint("c", "L", wise.Presence("A"), applicability={"flow_type": ["DF1"]})
    # A frozen value is a tuple (AttributeError on clear) or a frozen list (TypeError).
    with pytest.raises((TypeError, AttributeError)):
        nc.applicability["flow_type"].clear()
    assert list(nc.applicability["flow_type"]) == ["DF1"]
    cases = pd.DataFrame({"flow_type": ["DF1", "DF2"]}, index=["a", "b"])
    assert nc.applies_to(cases).tolist() == [True, False]


# --- C9: serialisation leaks non-Wise exceptions -----------------------------------------


@pytest.mark.regression
# C9 closed 2026-09-27: every public entry point raises a WiseError subclass (decision record 0001, error contract)
def test_dumps_with_ndarray_metadata_serialises_or_raises_norm_error(p2p_norm):
    n = p2p_norm.replace(metadata={"a": np.arange(3)})
    try:
        text = n.dumps()
    except NormError:
        return  # rejecting unserialisable metadata up front is acceptable too
    assert json.loads(text)["metadata"] == {"a": [0, 1, 2]}


@pytest.mark.regression
# C9 closed 2026-09-27: every public entry point raises a WiseError subclass (decision record 0001, error contract)
def test_dumps_with_timedelta_metadata_serialises_or_raises_norm_error(p2p_norm):
    n = p2p_norm.replace(metadata={"sla": pd.Timedelta("1D")})
    try:
        text = n.dumps()
    except NormError:
        return  # rejecting unserialisable metadata up front is acceptable too
    assert isinstance(json.loads(text)["metadata"]["sla"], str)


@pytest.mark.regression
# C9 closed 2026-09-27: every public entry point raises a WiseError subclass (decision record 0001, error contract)
def test_from_json_wraps_malformed_text_in_norm_error():
    with pytest.raises(NormError):
        wise.Norm.from_json("  {not json")


# --- C12: fingerprint leaves out free-form metadata and descriptions ----------------------


@pytest.mark.regression
@pytest.mark.spec("decision:0001")
# C12 closed 2026-10-02 (decision record 0001: a result carries the identity of its norm): the fingerprint covers what
# the norm measures, not its free-form metadata and descriptions
def test_fingerprint_is_unchanged_when_only_metadata_changes(p2p_norm):
    assert p2p_norm.replace(metadata={"note": "x"}).fingerprint() == p2p_norm.fingerprint()
    assert p2p_norm.replace(description="annotated").fingerprint() == p2p_norm.fingerprint()
    assert p2p_norm.replace_constraint("c1", description="annotated").fingerprint() == p2p_norm.fingerprint()


def test_fingerprint_changes_with_within_layer_weight(p2p_norm):
    assert p2p_norm.replace_constraint("c1", weight=2.0).fingerprint() != p2p_norm.fingerprint()


def test_fingerprint_changes_with_view_weight(p2p_norm):
    finance = p2p_norm.get_view("Finance")
    heavier = wise.View("Finance", constraint_weights={**finance.constraint_weights, "c1": 0.4})
    edited = p2p_norm.replace(views=tuple(heavier if v.name == "Finance" else v for v in p2p_norm.views))
    assert edited.fingerprint() != p2p_norm.fingerprint()


# --- C15 (decision 0001, section 3): quantile_scale carries a frozen calibration record ----------------------------


def _fitted_record(divisor: float) -> dict[str, Any]:
    return wise.Calibration.fitted(
        0.95, divisor, log_fingerprint="0" * 64, size=4, window=("2024-01-01T00:00:00", "2024-01-01T00:00:00")
    ).to_dict()


# --- C20: evaluation and validation logic existed twice ---------------------------------


def _count_calls(monkeypatch, module, name, calls):
    original = getattr(module, name)

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, counted)


@pytest.mark.regression
@pytest.mark.spec("decision:0001")
# C20 closed 2026-10-02 (decision record 0001: one owner per computation): the norm's structure is checked once per
# construction, Norm.validate() runs that same check, and scoring does not check the norm again
def test_the_norm_is_checked_once_and_validate_runs_the_same_check(monkeypatch):
    import wise.norm

    checks: list[Any] = []
    rules: list[Any] = []
    _count_calls(monkeypatch, wise.norm, "_check_structure", checks)
    _count_calls(monkeypatch, wise.norm, "parse", rules)
    norm = wise.Norm.from_dict(BASE)
    assert (len(checks), len(rules)) == (1, 1)  # one norm, one constraint with one rule
    norm.validate()
    assert (len(checks), len(rules)) == (2, 1)
    wise.score(make_log([("1", "A", 0, 1.0, "DF1")]), norm)
    assert (len(checks), len(rules)) == (2, 1)


@pytest.mark.regression
@pytest.mark.spec("decision:0001")
# C20 closed 2026-10-02 (decision record 0001): the lag recipe and the Lag constraint pair activations with responses
# through the one rule of the Lag family
def test_the_lag_recipe_and_the_lag_constraint_share_one_pairing_rule(monkeypatch, missing_attr_log):
    from wise.families import lag as lag_family

    calls: list[Any] = []
    _count_calls(monkeypatch, lag_family, "pairs", calls)
    missing_attr_log.derive([LAG_RECIPE])
    wise.evaluate(wise.Lag("A", "B", delta=1), missing_attr_log)
    assert [args[1:3] for args in calls] == [(("A",), ("B",)), (("A",), ("B",))]


@pytest.mark.regression
@pytest.mark.spec("decision:0001/4")
# C15 closed 2026-10-02: under schema version 2 a quantile_scale recipe without a fitted record is read and divides
# by the quantile of the scored cases, as 0.1.0 did; refusing an unfitted record belongs to schema version 3
def test_quantile_scale_without_a_record_divides_by_the_scored_quantile_under_schema_2():
    validate_recipe(QUANTILE_RECIPE)
    log = make_log([(c, "A", 0, float(c), "F") for c in (1, 2, 3, 4)], attrs=("amount",))
    log.derive([QUANTILE_RECIPE])
    # the 0.95 quantile of 1, 2, 3, 4 is 3 + 0.85 = 3.85
    assert log.cases["scaled"].tolist() == pytest.approx([1 / 3.85, 2 / 3.85, 3 / 3.85, 1.0], rel=1e-12)


@pytest.mark.spec("decision:0001/3")
def test_quantile_scale_with_stored_divisor_is_accepted_and_round_trips():
    recipe = {**QUANTILE_RECIPE, "calibration": _fitted_record(12.5)}
    validate_recipe(recipe)
    norm = _norm_with_recipe(recipe)
    again = wise.Norm.loads(norm.dumps())
    assert again.derived_attributes[0]["calibration"]["divisor"] == 12.5
    assert again.recipes[0].to_dict()["calibration"]["divisor"] == 12.5


@pytest.mark.regression
@pytest.mark.spec("decision:0001/3")
# C15 closed 2026-10-02: with a fitted record the divisor is the record's, so the value of a case does not depend
# on which other cases are scored
def test_quantile_scale_divides_by_stored_divisor():
    recipe = {**QUANTILE_RECIPE, "calibration": _fitted_record(2.0)}
    log = make_log([(c, "A", 0, float(c), "F") for c in (1, 2, 3, 4)], attrs=("amount",))
    log.derive([recipe])
    assert log.cases["scaled"].tolist() == [0.5, 1.0, 1.0, 1.0]
    alone = make_log([(1, "A", 0, 1.0, "F")], attrs=("amount",))
    alone.derive([recipe])
    assert alone.cases["scaled"].tolist() == [0.5]


@pytest.mark.regression
@pytest.mark.spec("decision:0001/3")
def test_a_fitted_case_value_does_not_depend_on_the_other_scored_cases():
    log = make_log([(c, "A", c, float(c), "F") for c in (1, 2, 3, 4)], attrs=("amount",))
    norm = wise.fit_calibration(_norm_with_recipe(QUANTILE_RECIPE), log)
    assert norm.recipes[0].to_dict()["calibration"]["divisor"] == pytest.approx(3.85, abs=1e-12)
    events = log.events
    whole = wise.score(log, norm).cases["scaled"]
    for subset in ([1], [2, 4], [1, 3, 4], [4]):
        part = make_log([(c, "A", c, float(c), "F") for c in subset], attrs=("amount",))
        assert len(part.events) == events["case"].isin(subset).sum()
        assert wise.score(part, norm).cases["scaled"].tolist() == whole.loc[subset].tolist()


# --- C16 (decision 0001, section 4): eval recipes name one case-table column -----------------------------------------


@pytest.mark.regression
@pytest.mark.spec("decision:0001/4")
# C16 closed 2026-10-02: under schema version 2 an eval recipe may name one existing case-table column and is
# read as that column; any other expression is refused at load with the recipe named, and none is evaluated
def test_eval_recipe_is_rejected():
    with pytest.raises(
        NormError, match=r"derived attribute 'twice' \(eval\): expr 'n_events \* 2' is not a single case-table column"
    ):
        validate_recipe({"name": "twice", "kind": "eval", "expr": "n_events * 2"})
    with pytest.raises(NormError, match="'twice'"):
        wise.Norm.from_dict({**BASE, "derived_attributes": [{"name": "twice", "kind": "eval", "expr": "n_events * 2"}]})


@pytest.mark.regression
@pytest.mark.spec("decision:0001/4")
def test_eval_recipe_of_one_case_table_column_reads_that_column(missing_attr_log):
    missing_attr_log.derive([{"name": "total", "kind": "eval", "expr": "n_events"}])
    assert missing_attr_log.cases["total"].tolist() == [2, 2]


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
