"""Norm construction, validation, immutability, and serialisation."""

import json
import pathlib

import numpy as np
import pandas as pd
import pytest
from conftest import make_log

import wise
from wise.errors import NormError


def test_json_roundtrip(tmp_path, p2p_norm, p2p_log):
    path = tmp_path / "norm.json"
    p2p_norm.dump(path)
    again = wise.Norm.load(path)
    assert again.to_dict() == p2p_norm.to_dict() and again.fingerprint() == p2p_norm.fingerprint()
    assert wise.Norm.from_json(p2p_norm.to_json()).constraint_ids == p2p_norm.constraint_ids
    pd.testing.assert_frame_equal(wise.score(p2p_log, p2p_norm).scores, wise.score(p2p_log, again).scores)
    d = json.loads(p2p_norm.dumps())
    assert d["schema_version"] == wise.SCHEMA_VERSION and d["scoring_mode"] == "flat"


def test_to_dict_is_a_copy_and_numpy_safe(p2p_norm):
    d = p2p_norm.to_dict()
    d["constraints"][3]["applicability"]["flow_type"].append("ZZZ")
    assert p2p_norm.get_constraint("c4").applicability == {"flow_type": ["DF1"]}
    nc = wise.NormConstraint("x", "completeness", wise.Presence("A"), applicability={"k": list(np.array([1, 2]))})
    n = p2p_norm.replace(constraints=(*p2p_norm.constraints, nc), metadata={"date": pd.Timestamp("2026-01-01")})
    assert json.loads(n.dumps())["constraints"][-1]["applicability"] == {"k": [1, 2]}


def test_from_dict_accepts_mapping_style_layers_and_views():
    d = {
        "name": "mini",
        "layers": {"L1": {"name": "Closure"}, "L2": "Timing"},
        "views": {"Finance": {"layer_weights": {"L1": 0.7, "L2": 0.3}}},
        "constraints": [
            {"id": "c1", "layer": "L1", "type": "pres", "params": {"activity": "INV"}, "weight": 2},
            {"id": "c2", "layer": "L1", "type": "excl", "params": {"activity": "CINV"}},
            {"id": "c3", "layer": "L2", "type": "lag", "params": {"a": "GR", "b": "INV", "delta": 10, "width": 20}},
        ],
    }
    norm = wise.Norm.from_dict(d)
    assert norm.layer_ids == ["L1", "L2"]
    w = norm.raw_weights("Finance")
    assert w["c1"] == pytest.approx(0.7 * 2 / 3) and w["c2"] == pytest.approx(0.7 / 3) and w["c3"] == pytest.approx(0.3)


def test_strict_loading():
    base = {
        "layers": [{"id": "L"}],
        "views": [{"name": "v", "constraint_weights": {"c": 1}}],
        "constraints": [{"id": "c", "layer": "L", "type": "presence", "params": {"activity": "A"}}],
    }
    wise.Norm.from_dict({**base, "schema_version": 2})
    with pytest.raises(NormError, match="newer"):
        wise.Norm.from_dict({**base, "schema_version": 99})
    with pytest.raises(NormError, match="unknown keys"):
        wise.Norm.from_dict({**base, "constraints": [{**base["constraints"][0], "applicabilty": {}}]})
    with pytest.raises(NormError, match="unknown top-level"):
        wise.Norm.from_dict({**base, "extra": 1})
    with pytest.raises(NormError):
        wise.Norm.from_dict({**base, "constraints": [{**base["constraints"][0], "params": {"activity": "A", "m": "x"}}]})
    with pytest.raises(NormError, match="missing 'type'"):
        wise.Norm.from_dict({**base, "constraints": [{"id": "c", "layer": "L", "params": {}}]})


def test_validation_errors(p2p_norm):
    c, layers, views = p2p_norm.constraints, p2p_norm.layers, p2p_norm.views
    with pytest.raises(NormError, match="duplicate constraint"):
        wise.Norm((*c, c[0]), layers, views)
    with pytest.raises(NormError, match="unknown constraints"):
        wise.Norm(c, layers, [wise.View("X", constraint_weights={"zz": 1})])
    with pytest.raises(NormError, match="all constraint weights are zero"):
        wise.Norm(c, layers, [wise.View("X", layer_weights={"completeness": 0.0})])
    with pytest.raises(NormError):
        wise.View("bad", layer_weights={"a": 1}, constraint_weights={"c": 1})
    with pytest.raises(NormError):
        wise.View("neg", constraint_weights={"c1": -1})
    with pytest.raises(NormError):
        wise.View("empty", layer_weights={})
    with pytest.raises(NormError):
        wise.View("str", constraint_weights={"c1": "x"})
    with pytest.raises(NormError):
        wise.NormConstraint("x", "completeness", wise.Presence("A"), weight=-1)
    with pytest.raises(NormError, match="unknown layer"):
        wise.Norm([c[0].replace(layer="ghost"), *list(c[1:])], layers, views)
    with pytest.raises(NormError):
        wise.Norm((), layers, views)
    with pytest.raises(NormError):
        wise.Norm(c, layers, ())
    with pytest.raises(NormError):
        wise.NormConstraint("x", "L", "not a constraint")  # type: ignore[arg-type]


def test_nested_mutation_is_caught_before_scoring(p2p_norm, p2p_log):
    norm = wise.running_p2p_norm()
    norm.views[0].constraint_weights["c2"] = -5  # in-place edit of a nested mapping
    with pytest.raises(NormError, match="finite and >= 0"):
        wise.score(p2p_log, norm)
    norm = wise.running_p2p_norm()
    norm.constraints[3].applicability["flow_type"].clear()
    with pytest.raises(NormError):
        wise.score(p2p_log, norm)


def test_readme_norm_example_loads():
    text = (pathlib.Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    block = text.split("```json\n", 1)[1].split("```", 1)[0]
    norm = wise.Norm.loads(block)
    assert norm.constraint_ids == ["c1", "c2"] and norm.view_names == ["Finance", "Logistics"]


def test_malformed_norm_dicts_raise_norm_error():
    for bad in (
        {
            "views": ["Finance"],
            "layers": [],
            "constraints": [{"id": "c", "layer": "L", "type": "presence", "params": {"activity": "A"}}],
        },
        {
            "views": [{"name": "v", "constraint_weights": {"c": 1}}],
            "layers": ["L"],
            "constraints": [{"id": "c", "layer": "L", "type": "presence", "params": {"activity": "A"}}],
        },
        {
            "views": [{"name": "v", "constraint_weights": {"c": 1}}],
            "layers": [{"id": "L"}],
            "constraints": [{"layer": "L", "type": "presence", "params": {"activity": "A"}}],
        },
        [],
    ):
        with pytest.raises(NormError):
            wise.Norm.from_dict(bad)  # type: ignore[arg-type]
    with pytest.raises(NormError):
        wise.Presence(5)  # type: ignore[arg-type]
    with pytest.raises(NormError):
        wise.Lag("A", "B", delta="five")  # type: ignore[arg-type]


def test_immutability_and_editing(p2p_norm):
    with pytest.raises(AttributeError):
        p2p_norm.constraints[0].layer = "ghost"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        p2p_norm.name = "x"  # type: ignore[misc]
    n2 = p2p_norm.replace_constraint("c4", applicability={})
    assert n2.get_constraint("c4").applicability == {} and p2p_norm.get_constraint("c4").applicability
    assert n2.fingerprint() != p2p_norm.fingerprint()
    with pytest.raises(NormError):
        p2p_norm.replace_constraint("nope", weight=2)
    assert p2p_norm.get_constraint("c1").type == "presence"
    assert "Norm(" in repr(p2p_norm)


def test_applicability_rules():
    rows = [("a", "GR", 0, 5, "DF1"), ("b", "INV", 0, 50, "DF2"), ("c", "GR", 0, 500, "DF2")]
    log = make_log(rows, exposure_col="amount")
    cases = log.cases

    def app(rule):
        return wise.NormConstraint("c", "L", wise.Presence("GR"), applicability=rule).applies_to(cases, log).tolist()

    assert app({"flow_type": ["DF1"]}) == [True, False, False]
    assert app({"all": [{"attr": "flow_type", "in": ["DF2"]}, {"attr": "exposure", "gte": 100}]}) == [False, False, True]
    assert app({"any": [{"attr": "flow_type", "eq": "DF1"}, {"has": "INV"}]}) == [True, True, False]
    assert app({"not": {"attr": "flow_type", "in": ["DF1"]}}) == [False, True, True]
    assert app({"lacks": ["GR"]}) == [False, True, False]
    assert app({"attr": "exposure", "lt": 10}) == [True, False, False]
    for bad in ({"all": []}, {"attr": "x"}, {"attr": "x", "in": [1], "eq": 2}, {"has": "A", "attr": "x"}, {"flow_type": []}):
        with pytest.raises(NormError):
            wise.NormConstraint("c", "L", wise.Presence("GR"), applicability=bad)
    with pytest.raises(NormError):
        wise.NormConstraint("c", "L", wise.Presence("GR"), applicability={"has": "INV"}).applies_to(cases)


def test_check_against_log(p2p_norm, p2p_log):
    assert p2p_norm.check(p2p_log) == []
    bad = p2p_norm.replace_constraint("c1", constraint=wise.Presence("Clear invoice"))  # typo
    issues = bad.check(p2p_log)
    assert any("never occurs" in i for i in issues)
    metric = p2p_norm.replace_constraint("c6", constraint=wise.Metric("touches", 1, 1))
    assert any("neither a case attribute nor derived" in i for i in metric.check(p2p_log))
    rule = p2p_norm.replace_constraint("c6", applicability={"has": ["Clear invoice"]})
    assert any("'Clear invoice' never occurs" in i for i in rule.check(p2p_log))
    recipe = p2p_norm.replace(derived_attributes=[{"name": "x", "kind": "count", "activities": ["Clear invoice"]}])
    assert any("derived attribute 'x'" in i for i in recipe.check(p2p_log))


def test_describe_and_tables(p2p_norm):
    desc = p2p_norm.describe()
    assert list(desc.index) == p2p_norm.constraint_ids
    assert desc.loc["c4", "applicability"] == {"flow_type": ["DF1"]} and desc.loc["c1", "applicability"] == "always"
    assert p2p_norm.activities() == sorted({wise.datasets.GR, wise.datasets.INV, wise.datasets.CINV, wise.datasets.PO})
    assert p2p_norm.applicability_attributes() == ["flow_type"] and p2p_norm.attributes() == ["flow_type"]
    assert p2p_norm.layer_weight_table()["Finance"].sum() == pytest.approx(1.05)


def test_recipes_validation():
    with pytest.raises(NormError):
        wise.Norm(
            [wise.NormConstraint("m", "L", wise.Metric("x", 1, 1))],
            [wise.Layer("L")],
            [wise.View("v", constraint_weights={"m": 1})],
            derived_attributes=[{"name": "x", "kind": "bogus"}],
        )
    with pytest.raises(NormError, match="duplicate derived"):
        wise.Norm(
            [wise.NormConstraint("m", "L", wise.Metric("x", 1, 1))],
            [wise.Layer("L")],
            [wise.View("v", constraint_weights={"m": 1})],
            derived_attributes=[
                {"name": "x", "kind": "eval", "expr": "n_events"},
                {"name": "x", "kind": "eval", "expr": "n_events"},
            ],
        )
