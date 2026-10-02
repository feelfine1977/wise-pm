"""The norm document format (``wise.norm_document``): reading, writing, the schema version, the fingerprint, and
the model behind it (one structural check, read-only values).

Expectations are written out by hand from the small document ``MINI`` below.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import pickle
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import wise
import wise.norm
from wise import norm_document
from wise._frozen import FrozenDict, FrozenList, freeze, thaw
from wise.errors import NormError

PKG = Path(__file__).resolve().parents[1]
SHIPPED = [
    PKG / "examples" / "bpic19_norm.json",
    PKG / "examples" / "running_p2p_norm.json",
    PKG / "tests" / "data" / "golden" / "norm.json",
    *sorted((PKG / "evaluation" / "norms" / "native").glob("*.json")),
]

#: One layer, one view, two constraints, one recipe; every free-form field filled in.
MINI: dict[str, Any] = {
    "schema_version": 2,
    "name": "mini",
    "version": "3",
    "description": "a small norm",
    "scoring_mode": "flat",
    "metadata": {"owner": "finance", "tags": ["draft"]},
    "layers": [{"id": "L", "name": "Closure", "description": "is the case closed"}],
    "views": [{"name": "v", "description": "the only view", "constraint_weights": {"c": 1, "m": 3}}],
    "derived_attributes": [{"name": "n_a", "kind": "count", "activities": ["A"]}],
    "constraints": [
        {
            "id": "c",
            "layer": "L",
            "type": "presence",
            "params": {"activity": "A"},
            "applicability": {"flow_type": ["DF1", "DF2"]},
            "description": "A happens",
        },
        {"id": "m", "layer": "L", "type": "metric", "params": {"attribute": "n_a", "threshold": 1, "width": 1}},
    ],
}

#: The canonical JSON of MINI's identity, written by hand: sorted keys, no whitespace, every option of the written
#: document, and no metadata or description at any level.
MINI_IDENTITY = (
    '{"constraints":['
    '{"applicability":{"flow_type":["DF1","DF2"]},"id":"c","layer":"L",'
    '"params":{"activity":["A"],"m":1},"type":"presence","weight":1.0},'
    '{"applicability":{},"id":"m","layer":"L",'
    '"params":{"attribute":"n_a","direction":"high","threshold":1.0,"width":1.0},"type":"metric","weight":1.0}],'
    '"derived_attributes":[{"activities":["A"],"kind":"count","name":"n_a"}],'
    '"layers":[{"id":"L","name":"Closure"}],'
    '"name":"mini","schema_version":2,"scoring_mode":"flat","version":"3",'
    '"views":[{"constraint_weights":{"c":1.0,"m":3.0},"name":"v"}]}'
)


def mini(**changes: Any) -> dict[str, Any]:
    return {**copy.deepcopy(MINI), **changes}


# ----------------------------------------------------------------------------- reading and writing
@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.name)
def test_shipped_norms_round_trip_byte_identically_through_the_module_and_the_norm(path: Path, tmp_path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    norm = norm_document.load(path)
    assert norm_document.dumps(norm) + "\n" == text
    assert norm_document.dump(norm, tmp_path / "a.json").read_bytes() == path.read_bytes()
    assert wise.Norm.load(path).dump(tmp_path / "b.json").read_bytes() == path.read_bytes()
    assert wise.Norm.loads(text).dumps() + "\n" == text
    assert wise.Norm.from_json(path).to_json(tmp_path / "c.json") + "\n" == text
    assert (tmp_path / "c.json").read_bytes() == path.read_bytes()
    assert wise.Norm.from_json(text) == norm
    assert wise.Norm.from_dict(json.loads(text)).to_dict() == json.loads(text)


def test_the_norm_methods_give_what_the_module_gives() -> None:
    norm = norm_document.read(MINI)
    assert wise.Norm.from_dict(MINI) == norm
    assert norm.to_dict() == norm_document.write(norm) == MINI | {
        "views": [{"name": "v", "description": "the only view", "constraint_weights": {"c": 1.0, "m": 3.0}}],
        "constraints": [
            {**MINI["constraints"][0], "params": {"activity": ["A"], "m": 1}, "weight": 1.0},
            {
                **MINI["constraints"][1],
                "params": {"attribute": "n_a", "threshold": 1.0, "width": 1.0, "direction": "high"},
                "weight": 1.0,
                "applicability": {},
                "description": "",
            },
        ],
    }
    assert norm.dumps(indent=1) == norm_document.dumps(norm, indent=1)
    assert norm.fingerprint() == norm_document.fingerprint(norm)
    assert norm.constraints[0].to_dict() == norm_document.write_constraint(norm.constraints[0])
    assert norm.constraints[0].to_dict() == norm.to_dict()["constraints"][0]


def test_norm_py_holds_the_model_only() -> None:
    """The format's keys, JSON handling and hashing live in the document module; ``norm.py`` delegates."""
    source = inspect.getsource(wise.norm)
    tree = ast.parse(source)
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not {"json", "hashlib"} & imported
    for name in ("_TOP_KEYS", "_CONSTRAINT_KEYS", "_parse_json", "_json_default", "_from_dict"):
        assert not hasattr(wise.norm, name)
    for method in ("load", "loads", "dump", "dumps", "from_dict", "to_dict", "to_json", "fingerprint"):
        body = textwrap.dedent(inspect.getsource(getattr(wise.Norm, method)))
        assert "norm_document." in body, method


def test_the_written_document_is_independent_of_the_norm() -> None:
    norm = norm_document.read(MINI)
    doc = norm.to_dict()
    assert type(doc["metadata"]) is dict and type(doc["metadata"]["tags"]) is list
    assert type(doc["constraints"][0]["applicability"]["flow_type"]) is list
    assert type(doc["derived_attributes"][0]) is dict and type(doc["derived_attributes"][0]["activities"]) is list
    doc["metadata"]["tags"].append("x")
    doc["constraints"][0]["applicability"]["flow_type"].append("ZZ")
    doc["derived_attributes"][0]["activities"].append("B")
    assert norm.to_dict() == norm_document.read(MINI).to_dict()


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ([], "a norm must be a mapping, got list"),
        (mini(extra=1), r"unknown top-level keys in norm: \['extra'\]"),
        (mini(views=[{"name": "v", "weights": {}}]), r"view 'v': unknown keys \['weights'\]"),
        (mini(constraints=[{"id": "c", "layer": "L", "params": {}}]), "constraint 'c': missing 'type'"),
        (mini(constraints=[{"id": "c", "type": "presence", "params": {"activity": "A"}}]), "constraint 'c': missing 'layer'"),
        (mini(constraints=[{"layer": "L", "type": "presence", "params": {"activity": "A"}}]), "malformed norm: 'id'"),
        (mini(layers=[{"name": "no id"}]), "malformed norm: 'id'"),
    ],
)
def test_malformed_documents_are_refused(document: Any, message: str) -> None:
    with pytest.raises(NormError, match=message):
        norm_document.read(document)
    with pytest.raises(NormError, match=message):
        wise.Norm.from_dict(document)


def test_text_that_is_not_json_names_its_source(tmp_path: Path) -> None:
    with pytest.raises(NormError, match=r"norm text is not valid JSON: .* \(line 1, column 2\)"):
        wise.Norm.loads("{oops")
    bad = tmp_path / "bad.json"
    bad.write_text("[1,", encoding="utf-8")
    with pytest.raises(NormError, match=r"bad\.json is not valid JSON"):
        wise.Norm.load(bad)


def test_metadata_that_json_cannot_hold_is_refused_when_written() -> None:
    norm = norm_document.read(MINI).replace(metadata={"when": object()})
    with pytest.raises(NormError, match="metadata holds a value of type object"):
        norm.dumps()


# ----------------------------------------------------------------------------- the schema version
@pytest.mark.spec("decision:0001/4")
def test_the_constructor_defaults_to_the_current_schema_version(p2p_norm: wise.Norm) -> None:
    assert wise.SCHEMA_VERSION == 2
    assert p2p_norm.schema_version == wise.SCHEMA_VERSION
    assert "schema_version" in {f.name for f in __import__("dataclasses").fields(wise.Norm)}
    assert p2p_norm.to_dict()["schema_version"] == 2


@pytest.mark.spec("decision:0001/4")
def test_reading_a_document_requires_its_schema_version() -> None:
    document = mini()
    del document["schema_version"]
    with pytest.raises(NormError, match=r"norm document has no 'schema_version'; give one of \[1, 2\] \(current: 2\)"):
        wise.Norm.from_dict(document)
    with pytest.raises(NormError, match="no 'schema_version'"):
        wise.Norm.loads(json.dumps(document))


@pytest.mark.spec("decision:0001/4")
@pytest.mark.parametrize(
    ("value", "message"),
    [
        (3, "newer than supported"),
        (99, "newer than supported"),
        (0, "is not a known schema version"),
        (-1, "is not a known schema version"),
        ("abc", "is not an integer"),
        (True, "is not an integer"),
        (2.0, "is not an integer"),
        (None, "is not an integer"),
    ],
)
def test_an_unknown_schema_version_is_refused(value: Any, message: str) -> None:
    with pytest.raises(NormError, match=message):
        wise.Norm.from_dict(mini(schema_version=value))


@pytest.mark.spec("decision:0001/4")
@pytest.mark.parametrize(("value", "carried"), [(1, 1), (2, 2), ("1", 1), ("2", 2)])
def test_a_norm_carries_the_schema_version_it_was_read_with_and_writes_it_back(value: Any, carried: int) -> None:
    norm = wise.Norm.from_dict(mini(schema_version=value))
    assert norm.schema_version == carried
    assert norm.to_dict()["schema_version"] == carried
    assert json.loads(norm.dumps())["schema_version"] == carried
    assert wise.Norm.loads(norm.dumps()).schema_version == carried


@pytest.mark.spec("decision:0001/4")
def test_replace_and_its_variants_keep_the_schema_version() -> None:
    v1 = wise.Norm.from_dict(mini(schema_version=1))
    assert v1.replace(name="renamed").schema_version == 1
    assert v1.replace_constraint("c", weight=2.0).schema_version == 1
    assert v1.map_constraints(lambda c: c.replace(description="x")).schema_version == 1
    assert v1.replace(schema_version=1).schema_version == 1
    with pytest.raises(NormError, match=r"replace keeps the schema version \(1\); .* only by migration, not to 2"):
        v1.replace(schema_version=2)


@pytest.mark.spec("decision:0001/4")
@pytest.mark.parametrize("value", [0, 3, True, "2", None])
def test_the_constructor_refuses_an_unknown_schema_version(p2p_norm: wise.Norm, value: Any) -> None:
    with pytest.raises(NormError, match=r"schema_version must be one of \[1, 2\]"):
        wise.Norm(p2p_norm.constraints, p2p_norm.layers, p2p_norm.views, schema_version=value)


def test_a_schema_version_without_an_adapter_is_refused() -> None:
    with pytest.raises(NormError, match="no document format for norm schema_version 7"):
        norm_document.adapter(7)
    assert norm_document.adapter(1) is norm_document.adapter(2) is norm_document.V2


# ----------------------------------------------------------------------------- one structural check
def _counting(monkeypatch: pytest.MonkeyPatch, module: Any, name: str) -> list[Any]:
    calls: list[Any] = []
    original = getattr(module, name)

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, counted)
    return calls


def test_the_structure_is_checked_once_per_construction(monkeypatch: pytest.MonkeyPatch, p2p_log: wise.EventLog) -> None:
    checks = _counting(monkeypatch, wise.norm, "_check_structure")
    rules = _counting(monkeypatch, wise.norm, "parse")
    norm = wise.Norm.from_dict(MINI)
    assert len(checks) == 1
    assert len(rules) == 2  # one parse per constraint, none by the norm
    norm.validate()
    assert len(checks) == 2  # validate() repeats the same check
    assert len(rules) == 2
    edited = norm.replace(name="again")
    assert len(checks) == 3
    assert len(rules) == 2  # the constraints are reused as they are
    p2p = wise.running_p2p_norm()
    before = len(checks)
    wise.score(p2p_log, p2p)
    assert len(checks) == before  # scoring does not check a norm again
    assert edited.name == "again"


def test_validate_reports_what_construction_reports(p2p_norm: wise.Norm) -> None:
    p2p_norm.validate()
    # A norm cannot be changed after construction, so the re-check can only fail if the object was forced.
    object.__setattr__(p2p_norm, "scoring_mode", "bogus")
    with pytest.raises(NormError, match="scoring_mode must be one of"):
        p2p_norm.validate()


def test_the_norm_validation_does_not_repeat_the_checks_of_its_parts() -> None:
    """Weights and rules are checked by the view and the constraint that hold them, not again by the norm."""
    source = textwrap.dedent(inspect.getsource(wise.norm._check_structure))
    assert "isfinite" not in source
    assert "parse(" not in source
    assert textwrap.dedent(inspect.getsource(wise.Norm.validate)).count("_check_structure(self)") == 1


# ----------------------------------------------------------------------------- the fingerprint
@pytest.mark.spec("decision:0001")
def test_the_fingerprint_is_the_hash_of_the_document_without_free_form_parts() -> None:
    norm = wise.Norm.from_dict(MINI)
    assert norm.fingerprint() == hashlib.sha256(MINI_IDENTITY.encode("utf-8")).hexdigest()


def _strip(document: dict[str, Any]) -> dict[str, Any]:
    out = {**document, "metadata": {}, "description": ""}
    for part in ("layers", "views", "constraints"):
        out[part] = [{k: v for k, v in e.items() if k != "description"} for e in document[part]]
    return out


@pytest.mark.spec("decision:0001")
@pytest.mark.parametrize(
    ("change", "same"),
    [
        ({"metadata": {}}, True),
        ({"metadata": {"owner": "audit", "run": {"id": 7}}}, True),
        ({"description": "another text"}, True),
        ({"layers": [{**MINI["layers"][0], "description": "x"}]}, True),
        ({"views": [{**MINI["views"][0], "description": "x"}]}, True),
        ({"constraints": [{**MINI["constraints"][0], "description": "x"}, MINI["constraints"][1]]}, True),
        ({"name": "other"}, False),
        ({"version": "4"}, False),
        ({"scoring_mode": "layer_balanced"}, False),
        ({"schema_version": 1}, False),
        ({"layers": [{**MINI["layers"][0], "name": "Closed"}]}, False),
        ({"views": [{**MINI["views"][0], "constraint_weights": {"c": 1, "m": 4}}]}, False),
        ({"constraints": [{**MINI["constraints"][0], "weight": 2}, MINI["constraints"][1]]}, False),
        ({"constraints": [{**MINI["constraints"][0], "applicability": {}}, MINI["constraints"][1]]}, False),
        ({"constraints": [{**MINI["constraints"][0], "params": {"activity": "B"}}, MINI["constraints"][1]]}, False),
        ({"derived_attributes": [{"name": "n_a", "kind": "count", "activities": ["B"]}]}, False),
    ],
)
def test_the_fingerprint_changes_with_what_the_norm_measures_only(change: dict[str, Any], same: bool) -> None:
    base = wise.Norm.from_dict(MINI).fingerprint()
    changed = wise.Norm.from_dict(mini(**change)).fingerprint()
    assert (changed == base) is same
    assert wise.Norm.from_dict(_strip(mini(**change))).fingerprint() == changed


@pytest.mark.spec("decision:0001")
def test_the_fingerprints_of_the_shipped_norms() -> None:
    """The declared delta of the fingerprint definition (CHANGELOG, ``Fixed: C12``)."""
    expected = {
        "bpic19_norm.json": "a284ebdb4a732e4029771220726cce2a3a4b9ed96e0daeec66d90e8b6b759fde",
        "running_p2p_norm.json": "68e8462a1d19d8076dcb50376b4390720489116c92c8e24f72c41263a0f9006e",
        "norm.json": "5ffdbef31bf546d8d89d6fbb8f5f572e74084799149eb9976b4fb9b3e0d52f47",
    }
    for path in SHIPPED[:3]:
        assert wise.Norm.load(path).fingerprint() == expected[path.name], path.name
    assert wise.running_p2p_norm().fingerprint() == expected["running_p2p_norm.json"]


# ----------------------------------------------------------------------------- read-only values (C7)
@pytest.mark.spec("IV-C")
def test_nothing_read_from_a_norm_can_be_written_into() -> None:
    norm = wise.Norm.from_dict(MINI)
    before = norm.to_dict()
    fingerprint = norm.fingerprint()
    writes: list[Callable[[], object]] = [
        lambda: norm.metadata.__setitem__("owner", "x"),
        lambda: norm.metadata["tags"].append("x"),
        lambda: norm.metadata.update(owner="x"),
        lambda: norm.metadata.pop("owner"),
        lambda: norm.metadata.clear(),
        lambda: norm.derived_attributes[0].__setitem__("kind", "nunique"),
        lambda: norm.derived_attributes[0]["activities"].append("B"),
        lambda: norm.views[0].weights.__setitem__("c", 9),
        lambda: norm.views[0].weights.setdefault("z", 1.0),
        lambda: norm.views[0].weights.__delitem__("c"),
        lambda: norm.constraints[0].applicability.__setitem__("flow_type", []),
        lambda: norm.constraints[0].applicability["flow_type"].__setitem__(0, "DF9"),
        lambda: norm.constraints[0].applicability["flow_type"].extend(["DF9"]),
        lambda: norm.constraints[0].applicability["flow_type"].sort(),
        lambda: norm.constraints[0].applicability["flow_type"].pop(),
        lambda: norm.constraints[0]._selection.document["flow_type"].insert(0, "DF9"),
    ]
    for write in writes:
        with pytest.raises(TypeError, match="read-only"):
            write()
    layer_of = norm.layer_of
    layer_of["c"] = "elsewhere"
    described = norm.describe()
    described.loc["c", "weight"] = 9
    assert norm.layer_of["c"] == "L"
    assert norm.to_dict() == before
    assert norm.fingerprint() == fingerprint


@pytest.mark.spec("IV-C")
def test_the_inputs_of_a_norm_are_copied_on_construction() -> None:
    meta: dict[str, Any] = {"tags": ["a"], "array": np.arange(3)}
    recipe: dict[str, Any] = {"name": "n_a", "kind": "count", "activities": ["A"]}
    weights = {"c": 1.0}
    rule: dict[str, Any] = {"flow_type": ["DF1"]}
    nc = wise.NormConstraint("c", "L", wise.Presence("A"), applicability=rule)
    norm = wise.Norm((nc,), (wise.Layer("L"),), (wise.View("v", constraint_weights=weights),), derived_attributes=(recipe,), metadata=meta)
    meta["tags"].append("b")
    meta["array"][0] = 99
    recipe["activities"].append("B")
    weights["c"] = 0.0
    rule["flow_type"].append("DF2")
    assert norm.metadata["tags"] == ["a"]
    assert norm.metadata["array"].tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="read-only"):
        norm.metadata["array"][0] = 5
    assert norm.derived_attributes[0]["activities"] == ["A"]
    assert norm.recipes[0].activities == ("A",)  # type: ignore[attr-defined]
    assert norm.views[0].constraint_weights == {"c": 1.0}
    assert norm.constraints[0].applicability == {"flow_type": ["DF1"]}


@pytest.mark.spec("IV-C")
def test_read_only_values_still_behave_as_the_document_they_came_from(p2p_norm: wise.Norm) -> None:
    nc = p2p_norm.get_constraint("c4")
    assert isinstance(nc.applicability, dict) and isinstance(nc.applicability["flow_type"], list)
    assert nc.applicability == {"flow_type": ["DF1"]}
    assert json.dumps(nc.applicability) == '{"flow_type": ["DF1"]}'
    assert repr(nc.applicability) == "{'flow_type': ['DF1']}"
    plain = dict(nc.applicability)
    plain["flow_type"] = ["DF2"]
    assert nc.applicability == {"flow_type": ["DF1"]}
    assert list(nc.applicability["flow_type"]) + ["DF2"] == ["DF1", "DF2"]
    assert copy.copy(nc.applicability) is nc.applicability
    assert copy.deepcopy(p2p_norm) == p2p_norm
    assert pickle.loads(pickle.dumps(p2p_norm)) == p2p_norm
    edited = p2p_norm.replace_constraint("c4", applicability={**nc.applicability, "vendor": ["V1"]})
    assert edited.get_constraint("c4").applicability == {"flow_type": ["DF1"], "vendor": ["V1"]}


def test_freeze_and_thaw_are_inverse_on_documents() -> None:
    document = {"a": [1, {"b": (2, [3])}], "c": {"d": None}, "s": {1, 2}}
    frozen = freeze(document)
    assert isinstance(frozen, FrozenDict) and isinstance(frozen["a"], FrozenList)
    assert isinstance(frozen["a"][1]["b"], tuple) and isinstance(frozen["a"][1]["b"][1], FrozenList)
    assert frozen["s"] == frozenset({1, 2})
    assert freeze(frozen) is frozen
    thawed = thaw(frozen)
    assert thawed == {"a": [1, {"b": (2, [3])}], "c": {"d": None}, "s": frozenset({1, 2})}
    assert type(thawed["a"]) is list and type(thawed["a"][1]["b"][1]) is list
    for method, args in [("__iadd__", ([1],)), ("__imul__", (2,)), ("remove", (1,)), ("reverse", ()), ("__delitem__", (0,))]:
        with pytest.raises(TypeError, match="'FrozenList' is read-only"):
            getattr(frozen["a"], method)(*args)
    for method, args in [("__ior__", ({},)), ("popitem", ())]:
        with pytest.raises(TypeError, match="'FrozenDict' is read-only"):
            getattr(frozen, method)(*args)


def test_a_parsed_where_clause_is_read_only() -> None:
    norm = wise.Norm.from_dict(
        mini(derived_attributes=[{"name": "n_a", "kind": "count_events", "where": {"column": "activity", "in": ["A"]}}])
    )
    where = norm.recipes[0].where  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="read-only"):
        where.rule.value.append("B")
    assert norm.recipes[0].to_dict()["where"] == {"column": "activity", "in": ["A"]}
    assert type(norm.recipes[0].to_dict()["where"]) is dict


def test_the_series_a_norm_returns_are_new_each_time(p2p_norm: wise.Norm) -> None:
    assert p2p_norm.layer_of is not p2p_norm.layer_of
    pd.testing.assert_series_equal(p2p_norm.layer_of, p2p_norm.layer_of)
