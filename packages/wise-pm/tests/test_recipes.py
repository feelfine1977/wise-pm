"""Typed derived-attribute recipes (``wise.recipes``): parsing, stated inputs, values, refusals.

Values are hand-derived from the running example (``wise.running_p2p_log()``):

    case  events                                        GR   amounts (non-null)
    A     PO, GR 01-01, IR 01-26, Clear 01-31           1    100, 100
    B     PO, GR 01-01..01-04 (4), IR 01-09, Clear      4    25 ×4, 100
    C     PO, GR 01-01, IR 01-08, Clear 01-10           1    100, 82
    D     PO, GR 01-01, IR 01-06, Cancel IR 01-07       1    100, 100
    E     PO, GR 01-01                                  1    100
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import wise
import wise.norm
from wise.errors import LogSchemaError, NormError
from wise.family import Inputs
from wise.recipes import (
    KINDS,
    RECIPES,
    AggRecipe,
    ColumnRecipe,
    CountEventsRecipe,
    CountRecipe,
    CvRecipe,
    IndicatorTimesRecipe,
    LagRecipe,
    NuniqueRecipe,
    QuantileScaleRecipe,
    RatioRecipe,
    Recipe,
    parse_recipe,
)

GR, IR, PO = "Record Goods Receipt", "Record Invoice Receipt", "Create Purchase Order Item"
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def log() -> wise.EventLog:
    return wise.running_p2p_log()


def _norm(*recipes: dict[str, Any]) -> wise.Norm:
    return wise.Norm(
        (wise.NormConstraint("c", "L", wise.Presence(GR)),),
        (wise.Layer("L"),),
        (wise.View("v", constraint_weights={"c": 1}),),
        derived_attributes=recipes,
    )


# ------------------------------------------------------------------ one record per kind, with its inputs
#: (document, record type, stated inputs)
TRUTH_TABLE: list[tuple[dict[str, Any], type[Recipe], Inputs]] = [
    ({"name": "n", "kind": "count", "activities": GR}, CountRecipe, Inputs(activities=(GR,))),
    (
        {"name": "n", "kind": "count", "activities": [GR], "after": IR, "before": ["Clear Invoice"]},
        CountRecipe,
        Inputs(activities=(GR, IR, "Clear Invoice")),
    ),
    (
        {"name": "n", "kind": "count_events", "where": {"column": "org:resource", "regex": "^user"}},
        CountEventsRecipe,
        Inputs(event_columns=("org:resource",)),
    ),
    (
        {"name": "n", "kind": "count_events", "where": {"all": [{"attr": "activity", "eq": GR}, {"lacks": "Clear Invoice"}]}},
        CountEventsRecipe,
        Inputs(activities=("Clear Invoice",), event_columns=("activity",)),
    ),
    (
        {"name": "n", "kind": "nunique", "column": "vendor", "where": {"column": "activity", "in": [GR]}},
        NuniqueRecipe,
        Inputs(event_columns=("vendor", "activity")),
    ),
    ({"name": "n", "kind": "agg", "column": "amount", "agg": "max"}, AggRecipe, Inputs(event_columns=("amount",))),
    ({"name": "n", "kind": "cv", "column": "amount", "eps": 0.5}, CvRecipe, Inputs(event_columns=("amount",))),
    ({"name": "n", "kind": "lag", "a": GR, "b": [IR]}, LagRecipe, Inputs(activities=(GR, IR))),
    (
        {"name": "n", "kind": "ratio", "numerator": "x", "denominator": "n_events"},
        RatioRecipe,
        Inputs(case_attributes=("x", "n_events")),
    ),
    ({"name": "n", "kind": "quantile_scale", "attribute": "x"}, QuantileScaleRecipe, Inputs(case_attributes=("x",))),
    (
        {"name": "n", "kind": "indicator_times", "activities": ["Cancel Invoice Receipt"], "attribute": "x"},
        IndicatorTimesRecipe,
        Inputs(activities=("Cancel Invoice Receipt",), case_attributes=("x",)),
    ),
    ({"name": "n", "kind": "column", "attribute": "n_events"}, ColumnRecipe, Inputs(case_attributes=("n_events",))),
    ({"name": "n", "kind": "eval", "expr": "n_events"}, ColumnRecipe, Inputs(case_attributes=("n_events",))),
]


@pytest.mark.spec("decision:0001/3")
@pytest.mark.parametrize(
    ("document", "record", "inputs"), TRUTH_TABLE, ids=[f"{r[0]['kind']}-{i}" for i, r in enumerate(TRUTH_TABLE)]
)
def test_each_kind_parses_into_its_record_and_states_its_inputs(
    document: dict[str, Any], record: type[Recipe], inputs: Inputs
) -> None:
    parsed = parse_recipe(document)
    assert type(parsed) is record
    assert parsed.name == "n"
    assert parsed.requires() == inputs


def test_every_kind_of_0_1_0_has_a_record() -> None:
    kinds_0_1_0 = {"count", "count_events", "nunique", "agg", "cv", "lag", "ratio", "quantile_scale", "indicator_times", "eval"}
    assert kinds_0_1_0 <= set(KINDS)
    assert set(KINDS) == {*RECIPES, "eval"}


def test_records_are_frozen_and_options_take_their_0_1_0_defaults() -> None:
    lag = parse_recipe({"name": "n", "kind": "lag", "a": GR, "b": IR})
    assert lag == LagRecipe("n", (GR,), (IR,), unit="D", activation="first", response="first_after", allow_negative=False)
    assert parse_recipe({"name": "n", "kind": "quantile_scale", "attribute": "x"}) == QuantileScaleRecipe("n", "x", q=0.95)
    assert parse_recipe({"name": "n", "kind": "cv", "column": "amount"}) == CvRecipe("n", "amount", eps=1e-9)
    with pytest.raises(AttributeError):
        lag.unit = "h"  # type: ignore[misc]


# ------------------------------------------------------------------ values on the running example
VALUES: list[tuple[dict[str, Any], list[float]]] = [
    ({"kind": "count", "activities": [GR]}, [1, 4, 1, 1, 1]),
    # GR strictly after the first IR: none; E has no IR, so the scope is undefined
    ({"kind": "count", "activities": GR, "after": IR}, [0, 0, 0, 0, np.nan]),
    ({"kind": "count_events", "where": {"column": "activity", "in": [GR]}}, [1, 4, 1, 1, 1]),
    ({"kind": "count_events", "where": None}, [4, 7, 4, 4, 2]),
    ({"kind": "nunique", "column": "activity"}, [4, 4, 4, 4, 2]),
    ({"kind": "agg", "column": "amount", "agg": "sum"}, [200, 200, 182, 200, 100]),
    ({"kind": "agg", "column": "amount", "agg": "count", "where": {"column": "activity", "eq": GR}}, [1, 4, 1, 1, 1]),
    # A and D: 100, 100 → std 0; E: one value → std NaN, read as 0
    ({"kind": "cv", "column": "amount"}, [0, np.std([25, 25, 25, 25, 100], ddof=1) / 40, np.std([100, 82], ddof=1) / 91, 0, 0]),
    # first GR → first IR at or after it, in days: A 01-01→01-26, B 01-01→01-09, C →01-08, D →01-06, E none
    ({"kind": "lag", "a": GR, "b": IR}, [25, 8, 7, 5, np.nan]),
    ({"kind": "lag", "a": GR, "b": IR, "activation": "last"}, [25, 5, 7, 5, np.nan]),
    ({"kind": "lag", "a": GR, "b": IR, "unit": "h"}, [600, 192, 168, 120, np.nan]),
    # first IR overall before the first PO: b before a, NaN unless allowed
    ({"kind": "lag", "a": IR, "b": PO, "response": "first_overall"}, [np.nan] * 5),
    ({"kind": "lag", "a": IR, "b": PO, "response": "first_overall", "allow_negative": True}, [-26, -9, -8, -6, np.nan]),
    ({"kind": "ratio", "numerator": "n_events", "denominator": "n_events"}, [1, 1, 1, 1, 1]),
    # median of n_events (4, 7, 4, 4, 2) is 4
    ({"kind": "quantile_scale", "attribute": "n_events", "q": 0.5}, [1, 1, 1, 1, 0.5]),
    ({"kind": "indicator_times", "activities": ["Cancel Invoice Receipt"], "attribute": "n_events"}, [0, 0, 0, 4, 0]),
    ({"kind": "column", "attribute": "n_events"}, [4, 7, 4, 4, 2]),
    ({"kind": "eval", "expr": "n_events"}, [4, 7, 4, 4, 2]),
    ({"kind": "eval", "expr": " `n_events` "}, [4, 7, 4, 4, 2]),
]


@pytest.mark.spec("decision:0001/3")
@pytest.mark.parametrize(("document", "expected"), VALUES, ids=[json.dumps(v[0])[:60] for v in VALUES])
def test_records_compute_hand_derived_values(log: wise.EventLog, document: dict[str, Any], expected: list[float]) -> None:
    record = parse_recipe({"name": "x", **document})
    values = record.compute(log)
    assert values.index.tolist() == ["A", "B", "C", "D", "E"]
    np.testing.assert_allclose(values.to_numpy(dtype=float), expected, rtol=1e-12, equal_nan=True)
    log.derive([{"name": "x", **document}])
    np.testing.assert_allclose(log.cases["x"].to_numpy(dtype=float), expected, rtol=1e-12, equal_nan=True)


def test_a_column_recipe_needs_an_existing_case_table_column(log: wise.EventLog) -> None:
    with pytest.raises(NormError, match=r"derived attribute 'x' \(column\): 'nope' is not a column of the case table"):
        parse_recipe({"name": "x", "kind": "column", "attribute": "nope"}).compute(log)
    with pytest.raises(NormError, match=r"derived attribute 'x' \(column\): 'nope'"):
        log.derive([{"name": "x", "kind": "eval", "expr": "nope"}])


def test_a_column_recipe_is_a_number_per_case(log: wise.EventLog) -> None:
    log.add_case_attribute("label", pd.Series(["1.5", "x", "2", None, "3"], index=log.case_ids))
    values = parse_recipe({"name": "y", "kind": "column", "attribute": "label"}).compute(log)
    np.testing.assert_allclose(values.to_numpy(dtype=float), [1.5, np.nan, 2, np.nan, 3], equal_nan=True)


# ------------------------------------------------------------------ refusals when the norm is built
LAG = {"name": "lag_ab", "kind": "lag", "a": [GR], "b": [IR]}

#: (document, the key the refusal names); the recipe is always named.
REFUSALS: list[tuple[dict[str, Any], str]] = [
    # unknown keys
    ({"name": "n", "kind": "count", "activities": [GR], "activitis": [GR]}, "activitis"),
    ({**LAG, "missing": "skip"}, "missing"),
    ({"name": "n", "kind": "agg", "column": "amount", "agg": "sum", "exposure_agg": "max"}, "exposure_agg"),
    ({"name": "n", "kind": "eval", "expr": "n_events", "column": "x"}, "column"),
    # unknown option values
    ({**LAG, "activation": "each"}, "activation"),
    ({**LAG, "activation": "bogus"}, "activation"),
    ({**LAG, "response": "bogus"}, "response"),
    ({**LAG, "unit": "fortnight"}, "unit"),
    ({"name": "n", "kind": "agg", "column": "amount", "agg": "nonsense"}, "agg"),
    ({"name": "n", "kind": "agg", "column": "amount", "agg": "describe"}, "agg"),
    ({"name": "n", "kind": "quantile_scale", "attribute": "x", "q": 1.5}, "q"),
    ({"name": "n", "kind": "count_events", "where": {"column": "activity"}}, "where"),
    ({"name": "n", "kind": "count_events", "where": {"attr": "activity", "approx": 1}}, "where"),
    # wrong types
    ({**LAG, "allow_negative": "false"}, "allow_negative"),
    ({**LAG, "a": 3}, "a"),
    ({**LAG, "b": []}, "b"),
    ({"name": "n", "kind": "count", "activities": [GR, 1]}, "activities"),
    ({"name": "n", "kind": "count", "activities": {"label": GR}}, "activities"),
    ({"name": "n", "kind": "quantile_scale", "attribute": "x", "q": "0.95"}, "q"),
    ({"name": "n", "kind": "quantile_scale", "attribute": "x", "q": True}, "q"),
    ({"name": "n", "kind": "quantile_scale", "attribute": "x", "divisor": 0}, "divisor"),
    ({"name": "n", "kind": "quantile_scale", "attribute": "x", "calibration": {"status": "fitted"}}, "calibration"),
    ({"name": "n", "kind": "cv", "column": "amount", "eps": None}, "eps"),
    ({"name": "n", "kind": "nunique", "column": 3}, "column"),
    ({"name": "n", "kind": "ratio", "numerator": "x", "denominator": ""}, "denominator"),
    ({"name": "n", "kind": "count_events", "where": ["activity"]}, "where"),
    ({"name": "n", "kind": "eval", "expr": 3}, "expr"),
]


@pytest.mark.spec("decision:0001/3")
@pytest.mark.parametrize(("document", "key"), REFUSALS, ids=[f"{d['kind']}-{k}-{i}" for i, (d, k) in enumerate(REFUSALS)])
def test_malformed_recipes_are_refused_when_the_norm_is_built(document: dict[str, Any], key: str) -> None:
    with pytest.raises(
        NormError, match=rf"derived attribute '{document['name']}' \({document['kind']}\): (unknown key )?'?{key}\b"
    ):
        _norm(document)
    with pytest.raises(NormError, match=rf"'{document['name']}'"):
        wise.Norm.from_dict({**_norm().to_dict(), "derived_attributes": [document]})


@pytest.mark.parametrize(
    ("document", "match"),
    [
        ({"kind": "count", "activities": [GR]}, "without name"),
        ({"name": 7, "kind": "count", "activities": [GR]}, "name must be a string"),
        ({"name": "n", "kind": "nope"}, "derived attribute 'n': unknown kind 'nope'"),
        ({"name": "n", "kind": "lag", "a": GR}, r"derived attribute 'n' \(lag\): missing \['b'\]"),
        (["name", "n"], "must be a mapping, got list"),
    ],
    ids=["no name", "name not a string", "unknown kind", "missing key", "not a mapping"],
)
def test_recipes_without_a_name_a_known_kind_or_their_keys_are_refused(document: Any, match: str) -> None:
    with pytest.raises(NormError, match=match):
        _norm(document)


def test_an_unknown_key_is_refused_with_the_allowed_keys() -> None:
    with pytest.raises(NormError, match=r"allowed \['a', 'activation', 'allow_negative', 'b', 'response', 'unit'\]"):
        parse_recipe({**LAG, "activations": "first"})


# ------------------------------------------------------------------ eval (decision 0001)
@pytest.mark.spec("decision:0001/4")
@pytest.mark.parametrize(
    "expr",
    ["n_events * 2", "manual_touches / n_events", "0", "n_events.sum()", "`a` + `b`", "__import__('os')", "True", "", "lambda"],
)
def test_an_eval_recipe_other_than_one_column_is_refused_at_load(expr: str) -> None:
    document = {"name": "twice", "kind": "eval", "expr": expr}
    with pytest.raises(NormError, match=r"derived attribute 'twice' \(eval\): expr .* is not a single case-table column"):
        wise.Norm.from_dict({**_norm().to_dict(), "derived_attributes": [document]})


@pytest.mark.spec("decision:0001/4")
def test_no_expression_is_evaluated(log: wise.EventLog, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an expression was evaluated")

    monkeypatch.setattr(pd.DataFrame, "eval", forbidden)
    monkeypatch.setattr(pd, "eval", forbidden)
    norm = _norm({"name": "total_events", "kind": "eval", "expr": "n_events"})
    assert norm.recipes == (ColumnRecipe("total_events", "n_events"),)
    result = wise.score(log, norm)
    assert result.cases["total_events"].tolist() == [4, 7, 4, 4, 2]


# ------------------------------------------------------------------ the norm asks the record
def test_the_norm_carries_its_records_and_keeps_the_document_form() -> None:
    documents = (
        {"name": "n_gr", "kind": "count", "activities": GR},
        {"name": "share", "kind": "ratio", "numerator": "n_gr", "denominator": "n_events"},
    )
    norm = _norm(*documents)
    assert norm.recipes == (CountRecipe("n_gr", (GR,)), RatioRecipe("share", "n_gr", "n_events"))
    assert norm.derived_attributes == documents
    assert norm.to_dict()["derived_attributes"] == list(documents)


def test_duplicate_recipe_names_are_refused() -> None:
    with pytest.raises(NormError, match="duplicate derived attribute 'n'"):
        _norm({"name": "n", "kind": "count", "activities": GR}, {"name": "n", "kind": "column", "attribute": "n_events"})


def test_the_check_reports_the_activities_every_record_states(log: wise.EventLog) -> None:
    norm = _norm(
        {"name": "n", "kind": "count", "activities": ["Nope"], "after": GR, "before": "Never"},
        {"name": "l", "kind": "lag", "a": "Ghost", "b": IR},
        {"name": "w", "kind": "count_events", "where": {"has": "Phantom"}},
        {"name": "i", "kind": "indicator_times", "activities": GR, "attribute": "n"},
    )
    assert norm.check(log) == [
        "derived attribute 'n': activity 'Nope' never occurs in the log",
        "derived attribute 'n': activity 'Never' never occurs in the log",
        "derived attribute 'l': activity 'Ghost' never occurs in the log",
        "derived attribute 'w': activity 'Phantom' never occurs in the log",
    ]


def _recipe_keys() -> set[str]:
    keys = {"name", "kind", "expr"}
    for cls in RECIPES.values():
        keys.update(cls.required, cls.optional)
    return keys


def test_the_norm_validation_and_check_name_no_recipe_key() -> None:
    for method in (wise.Norm.validate, wise.Norm.check):
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        formatted = {id(v) for node in ast.walk(tree) if isinstance(node, ast.JoinedStr) for v in node.values}
        strings = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in formatted
        }
        docstring = ast.get_docstring(tree.body[0])  # type: ignore[arg-type]
        assert not (strings - {docstring}) & _recipe_keys(), method.__name__


def test_the_norm_module_does_not_import_the_recipe_evaluator() -> None:
    tree = ast.parse(Path(wise.norm.__file__).read_text(encoding="utf-8"))
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "derive" not in modules


# ------------------------------------------------------------------ the document form is unchanged
@pytest.mark.parametrize("path", [EXAMPLES / "bpic19_norm.json", EXAMPLES / "running_p2p_norm.json"], ids=lambda p: p.name)
def test_the_shipped_norms_load_and_round_trip_byte_identically(path: Path, tmp_path: Path) -> None:
    norm = wise.Norm.load(path)
    assert norm.dump(tmp_path / "again.json").read_bytes() == path.read_bytes()
    assert [r.name for r in norm.recipes] == [d["name"] for d in norm.derived_attributes]


def test_the_bpic19_eval_recipe_is_its_column() -> None:
    norm = wise.Norm.load(EXAMPLES / "bpic19_norm.json")
    total = next(r for r in norm.recipes if r.name == "total_events")
    assert total == ColumnRecipe("total_events", "n_events")
    assert next(d for d in norm.derived_attributes if d["name"] == "total_events") == {
        "name": "total_events",
        "kind": "eval",
        "expr": "n_events",
    }


@pytest.mark.parametrize(
    "document",
    [
        *(d for d, _, _ in TRUTH_TABLE),
        {
            "name": "n",
            "kind": "quantile_scale",
            "attribute": "x",
            "calibration": {
                "status": "fitted",
                "quantile": 0.95,
                "divisor": 12.5,
                "log_fingerprint": "f" * 64,
                "size": 3,
                "window": ["2024-01-01T00:00:00", None],
            },
        },
    ],
    ids=[*(f"{r[0]['kind']}-{i}" for i, r in enumerate(TRUTH_TABLE)), "quantile_scale-calibration"],
)
def test_the_canonical_document_of_a_record_parses_back_into_it(document: dict[str, Any]) -> None:
    record = parse_recipe(document)
    assert parse_recipe(record.to_dict()) == record


def test_a_quantile_scale_divisor_of_null_is_no_divisor(log: wise.EventLog) -> None:
    # 0.1.0 read "divisor": null as absent; n_events (4, 7, 4, 4, 2) over its median 4
    record = parse_recipe({"name": "x", "kind": "quantile_scale", "attribute": "n_events", "q": 0.5, "divisor": None})
    assert record == QuantileScaleRecipe("x", "n_events", q=0.5)
    assert record.compute(log).tolist() == [1, 1, 1, 1, 0.5]


def test_equivalent_documents_share_one_record_and_reuse_it(log: wise.EventLog) -> None:
    assert parse_recipe({"name": "n", "kind": "count", "activities": GR}) == parse_recipe(
        {"name": "n", "kind": "count", "activities": [GR], "after": None}
    )
    log.derive([{"name": "n", "kind": "count", "activities": GR}])
    result = wise.score(log, _norm({"name": "n", "kind": "count", "activities": [GR]}))
    assert result.cases["n"].tolist() == [1, 4, 1, 1, 1]


def test_a_malformed_recipe_leaves_the_log_unchanged(log: wise.EventLog) -> None:
    before = log.cases.columns.tolist()
    with pytest.raises(NormError, match="activation"):
        log.derive([{"name": "ok", "kind": "count", "activities": GR}, {**LAG, "activation": "each"}])
    assert log.cases.columns.tolist() == before


def test_an_unknown_event_column_is_refused_when_computed(log: wise.EventLog) -> None:
    record = parse_recipe({"name": "n", "kind": "agg", "column": "nope", "agg": "sum"})
    with pytest.raises(LogSchemaError, match="unknown event column 'nope'"):
        record.compute(log)


def test_quantile_scale_divides_by_one_when_the_quantile_is_not_positive(log: wise.EventLog) -> None:
    # Cancel Invoice Receipt occurs in D only: the median count is 0, so the counts are divided by 1
    log.derive([{"name": "cancels", "kind": "count", "activities": "Cancel Invoice Receipt"}])
    values = parse_recipe({"name": "x", "kind": "quantile_scale", "attribute": "cancels", "q": 0.5}).compute(log)
    assert values.tolist() == [0, 0, 0, 1, 0]
