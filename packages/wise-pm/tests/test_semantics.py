"""The semantics profile and the schema v3 document (decision 0001, sections 3 and 4).

Schema version 2 documents evaluate under the 0.1.0 semantics and announce it
with one ``LegacySemanticsWarning`` per norm read; schema version 3 documents
evaluate under the corrected semantics: three-valued applicability, explicit
lag bounds, fitted calibration records, no ``eval``. The truth tables below
are derived by hand from the three-case log ``scope_log``.
"""

from __future__ import annotations

import ast
import copy
import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import wise
from wise import norm_document
from wise.errors import NormError
from wise.family import EVALUATED, OUT_OF_SCOPE, SCOPE_UNKNOWN
from wise.selection import LEGACY, THREE_VALUED
from wise.semantics import CORRECTED_SEMANTICS, LEGACY_SEMANTICS, LegacySemanticsWarning, semantics_of

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[1]
SRC = PKG / "src" / "wise"


@pytest.fixture
def scope_log() -> wise.EventLog:
    """Case 1: ``ft="X"``, ``val=5``, A then B. Case 2: ``ft="Y"``, ``val=1``, A only. Case 3: no ``ft``, no ``val``, A only."""
    df = pd.DataFrame(
        {
            "case": ["1", "1", "2", "3"],
            "activity": ["A", "B", "A", "A"],
            "time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-01"]),
            "ft": ["X", "X", "Y", None],
            "val": [5.0, 5.0, 1.0, np.nan],
        }
    )
    return wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="time", case_attributes=["ft", "val"])


def scope_document(version: int, rule: Any) -> dict[str, Any]:
    """Two equally weighted constraints: ``c`` (B present) under ``rule``, ``d`` (A present) always."""
    document: dict[str, Any] = {
        "schema_version": version,
        "name": "scope",
        "scoring_mode": "flat",
        "layers": [{"id": "L"}],
        "views": [{"name": "v", "constraint_weights": {"c": 1, "d": 1}}],
        "constraints": [
            {"id": "c", "layer": "L", "type": "presence", "params": {"activity": "B"}, "applicability": rule},
            {"id": "d", "layer": "L", "type": "presence", "params": {"activity": "A"}},
        ],
    }
    if version == 3:
        document["missing_values"] = "three_valued"
    return document


def read_legacy(document: dict[str, Any]) -> wise.Norm:
    """Read a schema version 1 or 2 document, which announces the 0.1.0 semantics once."""
    with pytest.warns(LegacySemanticsWarning):
        return wise.Norm.from_dict(document)


def v3(**changes: Any) -> dict[str, Any]:
    document = scope_document(3, {})
    document.update(copy.deepcopy(changes))
    return document


LAG = {"id": "g", "layer": "L", "type": "lag", "params": {"a": "A", "b": "B"}}


def with_lag(version: int, params: dict[str, Any]) -> dict[str, Any]:
    document = scope_document(version, {})
    document["constraints"].append({**LAG, "params": params})
    return document


# ----------------------------------------------------------------------------- the profile
@pytest.mark.spec("decision:0001/4")
def test_the_schema_version_names_the_profile() -> None:
    assert semantics_of(1) is semantics_of(2) is LEGACY_SEMANTICS
    assert semantics_of(3) is CORRECTED_SEMANTICS
    assert (LEGACY_SEMANTICS.missing_values, LEGACY_SEMANTICS.fitted_calibration) == (LEGACY, False)
    assert (CORRECTED_SEMANTICS.missing_values, CORRECTED_SEMANTICS.fitted_calibration) == (THREE_VALUED, True)
    with pytest.raises(NormError, match=r"no semantics profile for norm schema_version 4; known: \[1, 2, 3\]"):
        semantics_of(4)
    with pytest.raises(NormError, match="missing-value policy must be one of"):
        wise.Semantics("other", "unknown", fitted_calibration=False)  # type: ignore[arg-type]


@pytest.mark.spec("decision:0001/4")
def test_a_norm_states_its_profile_by_its_schema_version(p2p_norm: wise.Norm) -> None:
    assert p2p_norm.semantics is LEGACY_SEMANTICS
    assert read_legacy(scope_document(2, {})).semantics is LEGACY_SEMANTICS
    assert wise.Norm.from_dict(scope_document(3, {})).semantics is CORRECTED_SEMANTICS
    assert wise.Norm(p2p_norm.constraints, p2p_norm.layers, p2p_norm.views, schema_version=3).semantics is CORRECTED_SEMANTICS


# ----------------------------------------------------------------------------- the legacy warning
def _caught(read: Any) -> tuple[Any, list[warnings.WarningMessage]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = read()
    return out, caught


@pytest.mark.spec("decision:0001/4")
@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("how", ["read", "from_dict", "loads", "load", "from_json text", "from_json path"])
def test_reading_a_legacy_document_warns_exactly_once(version: int, how: str, tmp_path: Path) -> None:
    document = scope_document(version, {})
    text = json.dumps(document)
    path = tmp_path / "norm.json"
    path.write_text(text, encoding="utf-8")
    readers = {
        "read": lambda: norm_document.read(document),
        "from_dict": lambda: wise.Norm.from_dict(document),
        "loads": lambda: wise.Norm.loads(text),
        "load": lambda: wise.Norm.load(path),
        "from_json text": lambda: wise.Norm.from_json(text),
        "from_json path": lambda: wise.Norm.from_json(path),
    }
    norm, caught = _caught(readers[how])
    assert [w.category for w in caught] == [LegacySemanticsWarning]
    assert str(caught[0].message).startswith(
        f"norm 'scope' has schema_version {version} and is evaluated under the 0.1.0 semantics"
    )
    assert norm.schema_version == version


@pytest.mark.spec("decision:0001/4")
def test_each_legacy_norm_read_warns_once_and_nothing_else_does(p2p_log: wise.EventLog, tmp_path: Path) -> None:
    assert issubclass(wise.LegacySemanticsWarning, FutureWarning)
    norms, caught = _caught(lambda: [wise.Norm.from_dict(scope_document(2, {})) for _ in range(3)])
    assert [w.category for w in caught] == [LegacySemanticsWarning] * 3
    legacy = norms[0]

    def everything_else() -> None:
        wise.Norm.from_dict(scope_document(3, {}))  # schema version 3
        built = wise.running_p2p_norm()  # built in Python, not read
        wise.score(p2p_log, built)
        legacy.to_dict()
        legacy.dumps()
        legacy.dump(tmp_path / "out.json")
        legacy.fingerprint()
        legacy.replace(name="other")

    _, quiet = _caught(everything_else)
    assert quiet == []


@pytest.mark.spec("decision:0001/4")
def test_the_tiers_that_read_legacy_documents_filter_the_warning_by_name() -> None:
    """The golden, paper (Section V) and compatibility tiers keep their numbers and filter the warning explicitly."""
    marker = 'pytest.mark.filterwarnings("ignore::wise.LegacySemanticsWarning")'
    for module in ("tests/golden/test_pipeline_golden.py", "tests/golden/test_schema_v3_golden.py", "tests/test_bpic19.py"):
        assert marker in (PKG / module).read_text(encoding="utf-8"), module
    gate = (REPO / "scripts" / "compat_gate.sh").read_text(encoding="utf-8")
    assert 'LEGACY_FILTER="ignore::wise.LegacySemanticsWarning"' in gate
    assert gate.count('-W "$LEGACY_FILTER"') == 2  # both pytest commands of the gate


# ----------------------------------------------------------------------------- three-valued applicability
#: (rule, scope under schema 2, scope under schema 3) for cases 1, 2, 3; ``None`` is unknown.
SCOPES = [
    pytest.param({"attr": "ft", "ne": "X"}, [False, True, True], [False, True, None], id="ne"),
    pytest.param({"attr": "ft", "not_in": ["X"]}, [False, True, True], [False, True, None], id="not_in"),
    pytest.param({"not": {"attr": "ft", "eq": "X"}}, [False, True, True], [False, True, None], id="not-eq"),
    pytest.param({"attr": "ft", "eq": "Y"}, [False, True, False], [False, True, None], id="eq"),
    pytest.param({"attr": "ft", "in": ["Y"]}, [False, True, False], [False, True, None], id="in"),
    pytest.param({"ft": ["Y"]}, [False, True, False], [False, True, None], id="short-form"),
    pytest.param({"attr": "ft", "regex": "^Y$"}, [False, True, False], [False, True, None], id="regex"),
    pytest.param({"attr": "ft", "isna": True}, [False, False, True], [False, False, None], id="isna"),
    pytest.param({"attr": "ft", "notna": True}, [True, True, False], [True, True, None], id="notna"),
    pytest.param({"attr": "val", "gt": 2}, [True, False, False], [True, False, None], id="gt"),
    pytest.param({"attr": "val", "gte": 1}, [True, True, False], [True, True, None], id="gte"),
    pytest.param({"attr": "val", "lt": 2}, [False, True, False], [False, True, None], id="lt"),
    pytest.param({"attr": "val", "lte": 5}, [True, True, False], [True, True, None], id="lte"),
    pytest.param({"not": {"attr": "val", "gt": 2}}, [False, True, True], [False, True, None], id="not-gt"),
    pytest.param({"any": [{"attr": "ft", "eq": "Y"}, {"has": ["B"]}]}, [True, True, False], [True, True, None], id="any-unknown"),
    pytest.param({"any": [{"attr": "ft", "eq": "Y"}, {"has": ["A"]}]}, [True, True, True], [True, True, True], id="any-true"),
    pytest.param(
        {"all": [{"attr": "ft", "ne": "X"}, {"lacks": ["A"]}]}, [False, False, False], [False, False, False], id="all-false"
    ),
    pytest.param(
        {"not": {"any": [{"attr": "ft", "eq": "X"}, {"lacks": ["A"]}]}}, [False, True, True], [False, True, None], id="not-any"
    ),
]
#: ν of ``c`` (B present) per case when in scope: case 1 has B, cases 2 and 3 lack it.
NU_IN_SCOPE = [0.0, 1.0, 1.0]


def _expected(scope: list[bool | None]) -> tuple[list[float], list[str], list[bool], list[float]]:
    """Value, reason and in-scope flag of ``c``, and the flat score, per case.

    ``d`` is satisfied by every case, so a case with ``c`` in scope scores ``1 − ν_c / 2`` and any other ``1``:
    out of scope and unknown cases leave ``c`` out of the denominator.
    """
    values = [nu if s else np.nan for s, nu in zip(scope, NU_IN_SCOPE, strict=True)]
    reasons = [EVALUATED if s else (SCOPE_UNKNOWN if s is None else OUT_OF_SCOPE) for s in scope]
    scores = [1.0 - nu / 2 if s else 1.0 for s, nu in zip(scope, NU_IN_SCOPE, strict=True)]
    return values, reasons, [bool(s) for s in scope], scores


@pytest.mark.spec("decision:0001/3")
@pytest.mark.parametrize(("rule", "legacy", "three_valued"), SCOPES)
def test_applicability_is_three_valued_under_schema_3(
    scope_log: wise.EventLog, rule: Any, legacy: list[bool | None], three_valued: list[bool | None]
) -> None:
    for norm, scope in (
        (read_legacy(scope_document(2, rule)), legacy),
        (wise.Norm.from_dict(scope_document(3, rule)), three_valued),
    ):
        values, reasons, in_scope, scores = _expected(scope)
        ev = wise.evaluate(norm.get_constraint("c"), scope_log, semantics=norm.semantics)
        assert ev.value.tolist() == pytest.approx(values, nan_ok=True)
        assert ev.reason.astype(str).tolist() == reasons
        result = wise.score(scope_log, norm)
        assert result.violations["c"].tolist() == pytest.approx(values, nan_ok=True)
        assert result.in_scope["c"].tolist() == in_scope
        assert result.scores["v"].tolist() == pytest.approx(scores)
        assert result.effective_weights("v")["c"].tolist() == pytest.approx([0.5 if s else 0.0 for s in scope])


@pytest.mark.spec("decision:0001/3")
def test_the_scope_of_a_constraint_names_the_unknown_cases(scope_log: wise.EventLog) -> None:
    nc = wise.NormConstraint("c", "L", wise.Presence("B"), applicability={"attr": "ft", "ne": "X"})
    assert nc.scope(scope_log.cases, missing=THREE_VALUED).tolist() == [False, True, pd.NA]
    assert nc.scope(scope_log.cases).tolist() == [False, True, True]
    assert nc.applies_to(scope_log.cases, missing=THREE_VALUED).tolist() == [False, True, False]
    assert nc.applies_to(scope_log.cases).tolist() == [False, True, True]
    always = wise.NormConstraint("d", "L", wise.Presence("A"))
    assert always.scope(scope_log.cases, missing=THREE_VALUED).tolist() == [True, True, True]
    assert always.applies_to(scope_log.cases, missing=THREE_VALUED).tolist() == [True, True, True]


@pytest.mark.spec("decision:0001/4")
def test_evaluation_receives_the_profile_as_an_argument(scope_log: wise.EventLog) -> None:
    """The same constraint, evaluated under two profiles in turn, follows the profile it is given each time."""
    rule = {"attr": "ft", "ne": "X"}
    nc = wise.NormConstraint("c", "L", wise.Presence("B"), applicability=rule)
    legacy, corrected = read_legacy(scope_document(2, rule)), wise.Norm.from_dict(scope_document(3, rule))
    for _ in range(2):
        assert wise.evaluate_constraint(scope_log, nc).tolist() == [pytest.approx(np.nan, nan_ok=True), 1.0, 1.0]
        assert wise.evaluate(nc, scope_log, semantics=CORRECTED_SEMANTICS).reason.astype(str).tolist() == [
            OUT_OF_SCOPE,
            EVALUATED,
            SCOPE_UNKNOWN,
        ]
        assert wise.score(scope_log, corrected).scores["v"].tolist() == [1.0, 0.5, 1.0]
        assert wise.score(scope_log, legacy).scores["v"].tolist() == [1.0, 0.5, 0.5]
        assert wise.violation_matrix(scope_log, corrected)["c"].isna().tolist() == [True, False, True]


def test_no_module_keeps_the_profile_in_global_state() -> None:
    """Nothing in the library assigns a module-level profile or uses ``global``; the profile travels as a value."""
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not any(isinstance(node, ast.Global) for node in ast.walk(tree)), path
        assert "contextvars" not in path.read_text(encoding="utf-8"), path
    semantics_module = (SRC / "semantics.py").read_text(encoding="utf-8")
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "semantics.py":
            continue
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign | ast.AnnAssign):
                assert "Semantics(" not in ast.unparse(node), (path, ast.unparse(node))
    assert "LEGACY_SEMANTICS = Semantics(" in semantics_module


# ----------------------------------------------------------------------------- the schema v3 document
@pytest.mark.spec("decision:0001/3")
def test_a_v3_document_is_written_with_its_missing_value_policy() -> None:
    document = v3(constraints=[{"id": "d", "layer": "L", "type": "presence", "params": {"activity": "A"}}])
    document["views"] = [{"name": "v", "constraint_weights": {"d": 1}}]
    norm = wise.Norm.from_dict(document)
    assert norm.schema_version == 3
    assert norm.to_dict() == {
        "schema_version": 3,
        "name": "scope",
        "version": "1",
        "description": "",
        "scoring_mode": "flat",
        "missing_values": "three_valued",
        "metadata": {},
        "layers": [{"id": "L", "name": "", "description": ""}],
        "views": [{"name": "v", "description": "", "constraint_weights": {"d": 1.0}}],
        "derived_attributes": [],
        "constraints": [
            {
                "id": "d",
                "layer": "L",
                "type": "presence",
                "params": {"activity": ["A"], "m": 1},
                "weight": 1.0,
                "applicability": {},
                "description": "",
            }
        ],
    }
    legacy_keys = list(norm_document.v2.write(norm))
    assert list(norm.to_dict()) == [*legacy_keys[:5], "missing_values", *legacy_keys[5:]]
    assert wise.Norm.loads(norm.dumps()).dumps() == norm.dumps()


@pytest.mark.spec("decision:0001/3")
@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param({"missing_values": None}, r"missing_values must be one of \['three_valued'\], got None", id="null"),
        pytest.param(
            {"missing_values": "legacy"}, r"missing_values must be one of \['three_valued'\], got 'legacy'", id="legacy"
        ),
        pytest.param({"missing_value": "three_valued"}, r"unknown top-level keys in norm: \['missing_value'\]", id="typo"),
    ],
)
def test_the_missing_value_policy_is_a_named_field_of_the_v3_document(change: dict[str, Any], message: str) -> None:
    with pytest.raises(NormError, match=message):
        wise.Norm.from_dict(v3(**change))
    absent = v3()
    del absent["missing_values"]
    with pytest.raises(
        NormError, match="a schema version 3 norm names its missing-value policy: 'missing_values': 'three_valued'"
    ):
        wise.Norm.from_dict(absent)
    with pytest.raises(NormError, match=r"unknown top-level keys in norm: \['missing_values'\]"):
        wise.Norm.from_dict({**scope_document(2, {}), "missing_values": "three_valued"})


@pytest.mark.spec("decision:0001/3")
@pytest.mark.parametrize(
    ("params", "message"),
    [
        pytest.param({"a": "A", "b": "B"}, "give 'delta' (and 'width') or 'unbounded': true", id="bare"),
        pytest.param({"a": "A", "b": "B", "width": 5}, "give 'delta' (and 'width') or 'unbounded': true", id="width-only"),
        pytest.param({"a": "A", "b": "B", "delta": None}, "give 'delta' (and 'width') or 'unbounded': true", id="delta-null"),
        pytest.param({"a": "A", "b": "B", "unbounded": False}, "give 'delta' (and 'width') or 'unbounded': true", id="false"),
        pytest.param(
            {"a": "A", "b": "B", "unbounded": True, "delta": 3}, "an unbounded lag has no delta, got delta=3", id="both"
        ),
        pytest.param({"a": "A", "b": "B", "unbounded": "yes"}, "unbounded must be true or false, got 'yes'", id="not-bool"),
    ],
)
def test_a_v3_lag_states_its_bound(params: dict[str, Any], message: str) -> None:
    with pytest.raises(NormError, match=r"constraint 'g' \(lag\): .*" + re.escape(message)):
        wise.Norm.from_dict(with_lag(3, params))


@pytest.mark.spec("decision:0001/3")
def test_a_v3_lag_with_a_bound_or_unbounded_is_read_and_written_back() -> None:
    bounded = wise.Norm.from_dict(with_lag(3, {"a": "A", "b": "B", "delta": 0, "width": 0}))
    lag = bounded.get_constraint("g").constraint
    assert isinstance(lag, wise.Lag)
    assert (lag.delta, lag.width) == (0.0, 0.0)
    assert "unbounded" not in bounded.to_dict()["constraints"][-1]["params"]
    unbounded = wise.Norm.from_dict(with_lag(3, {"a": "A", "b": "B", "unbounded": True}))
    lag = unbounded.get_constraint("g").constraint
    assert isinstance(lag, wise.Lag)
    assert lag.delta is None
    params = unbounded.to_dict()["constraints"][-1]["params"]
    assert params["unbounded"] is True
    assert "delta" not in params
    assert list(params)[:3] == ["a", "b", "unbounded"]
    assert wise.Norm.loads(unbounded.dumps()).dumps() == unbounded.dumps()
    assert unbounded.get_constraint("g").to_dict()["params"]["delta"] is None  # the default writer is schema 2
    assert norm_document.write_constraint(unbounded.get_constraint("g"), 3)["params"]["unbounded"] is True


@pytest.mark.spec("decision:0001/4")
def test_a_v2_lag_keeps_its_0_1_0_reading() -> None:
    bare = read_legacy(with_lag(2, {"a": "A", "b": "B"})).get_constraint("g").constraint
    assert isinstance(bare, wise.Lag)
    assert (bare.delta, bare.width) == (0.0, 0.0)
    unbounded = read_legacy(with_lag(2, {"a": "A", "b": "B", "delta": None})).get_constraint("g").constraint
    assert isinstance(unbounded, wise.Lag)
    assert unbounded.delta is None


UNFITTED = {"status": "unfitted", "quantile": 0.5}
SCALED = {"name": "scaled", "kind": "quantile_scale", "attribute": "val", "q": 0.5}


def with_recipe(version: int, recipe: dict[str, Any]) -> dict[str, Any]:
    document = scope_document(version, {})
    document["derived_attributes"] = [recipe]
    document["constraints"].append(
        {"id": "m", "layer": "L", "type": "metric", "params": {"attribute": recipe["name"], "threshold": 1}}
    )
    return document


@pytest.mark.spec("decision:0001/3")
def test_an_unfitted_calibration_is_admitted_in_v3_and_refused_by_scoring(scope_log: wise.EventLog) -> None:
    norm = wise.Norm.from_dict(with_recipe(3, {**SCALED, "calibration": UNFITTED}))
    assert norm.recipes[0].calibration == wise.Calibration(0.5)  # type: ignore[attr-defined]
    assert wise.Norm.loads(norm.dumps()).dumps() == norm.dumps()
    with pytest.raises(
        NormError,
        match=r"derived attribute 'scaled' \(quantile_scale\) has an unfitted calibration record; under the corrected "
        r"semantics a population-dependent recipe is computed only with a fitted record \(fit it with wise.fit_calibration\)",
    ):
        wise.score(scope_log, norm)
    fitted = wise.fit_calibration(norm, scope_log)
    assert fitted.schema_version == 3
    # val is 5 and 1 on cases 1 and 2: median 3; case 3 has none
    assert fitted.recipes[0].calibration.divisor == 3.0  # type: ignore[attr-defined]
    result = wise.score(scope_log, fitted)
    assert result.cases["scaled"].tolist() == pytest.approx([1.0, 1 / 3, np.nan], nan_ok=True)
    # an attribute attached beforehand (EventLog.derive divides by the population quantile) is refused all the same
    assert scope_log.derive(norm.derived_attributes) == ["scaled"]
    with pytest.raises(NormError, match=r"'scaled' \(quantile_scale\) has an unfitted calibration record"):
        wise.score(scope_log, norm, derive=False)


@pytest.mark.spec("decision:0001/3")
def test_a_v3_quantile_scale_carries_its_record(scope_log: wise.EventLog) -> None:
    with pytest.raises(
        NormError,
        match=r"derived attribute 'scaled' \(quantile_scale\): a schema version 3 recipe carries its calibration record",
    ):
        wise.Norm.from_dict(with_recipe(3, SCALED))
    built = wise.Norm.from_dict(with_recipe(3, {**SCALED, "calibration": UNFITTED})).replace(derived_attributes=(SCALED,))
    with pytest.raises(NormError, match=r"'scaled' \(quantile_scale\) has no calibration record; under the corrected semantics"):
        wise.score(scope_log, built)


@pytest.mark.spec("decision:0001/4")
def test_under_schema_2_an_unfitted_record_divides_by_the_scored_quantile(scope_log: wise.EventLog) -> None:
    result = wise.score(scope_log, read_legacy(with_recipe(2, {**SCALED, "calibration": UNFITTED})))
    assert result.cases["scaled"].tolist() == pytest.approx([1.0, 1 / 3, np.nan], nan_ok=True)
    assert result.calibrations["scaled"].divisor == 3.0


EVAL = {"name": "n", "kind": "eval", "expr": "n_events"}


@pytest.mark.spec("decision:0001/3")
def test_an_eval_recipe_is_refused_under_schema_3(p2p_norm: wise.Norm) -> None:
    message = r"derived attribute 'n': schema version 3 has no eval recipe; name the case-table column with a column recipe"
    with pytest.raises(NormError, match=message):
        wise.Norm.from_dict(with_recipe(3, EVAL))
    with pytest.raises(NormError, match=message):
        wise.Norm(p2p_norm.constraints, p2p_norm.layers, p2p_norm.views, derived_attributes=(EVAL,), schema_version=3)
    assert read_legacy(with_recipe(2, EVAL)).recipes[0].kind == "column"
    column = wise.Norm.from_dict(with_recipe(3, {"name": "n", "kind": "column", "attribute": "n_events"}))
    assert column.recipes[0].kind == "column"


@pytest.mark.spec("decision:0001/4")
def test_v2_and_v3_copies_of_a_norm_have_different_fingerprints() -> None:
    legacy, corrected = read_legacy(scope_document(2, {})), wise.Norm.from_dict(scope_document(3, {}))
    assert legacy.fingerprint() != corrected.fingerprint()
    assert norm_document.V3.identity(corrected.to_dict())["missing_values"] == "three_valued"


# ----------------------------------------------------------------------------- the scoring mode of a result
@pytest.mark.spec("IV-D")
@pytest.mark.parametrize("mode", ["flat", "layer_balanced"])
def test_a_result_built_without_a_mode_takes_the_norm_scoring_mode(p2p_result: wise.ScoreResult, mode: str) -> None:
    norm = p2p_result.norm.replace(scoring_mode=mode)
    shell = wise.ScoreResult(
        norm=norm,
        cases=p2p_result.cases,
        violations=p2p_result.violations,
        in_scope=p2p_result.in_scope,
        scores=p2p_result.scores,
        contributions=p2p_result.contributions,
        _weights=p2p_result._weights,
    )
    assert shell.mode == mode
    expected = wise.score(p2p_result.log, norm, mode=mode)  # type: ignore[arg-type]
    pd.testing.assert_frame_equal(shell.effective_weights("Finance"), expected.effective_weights("Finance"))
    assert wise.score(p2p_result.log, norm).mode == mode  # type: ignore[arg-type]
