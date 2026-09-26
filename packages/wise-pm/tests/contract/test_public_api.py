"""Contract tests: the public API of ``wise`` is frozen by ``api_snapshot.json``.

``scripts/snapshot_api.py`` rebuilds the snapshot from the installed package;
these tests compare it with the stored file. Allowed without a snapshot
refresh: a new keyword-only parameter with a default appended to a function
or method, and a new public name (both are reported in the test's report
section). Everything else fails and names the symbol, shows old vs new, and
tells how to refresh. The last group of tests proves that rule on synthetic
snapshots, so it is exercised even while the real API is unchanged.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "snapshot_api.py"
SNAPSHOT_PATH = Path(__file__).with_name("api_snapshot.json")


def _load_script() -> ModuleType:
    name = "wise_snapshot_api"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None, SCRIPT_PATH
    assert spec.loader is not None, SCRIPT_PATH
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(module)
    return module


snapshot_api = _load_script()
REQUIRED = snapshot_api.REQUIRED


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def stored_text() -> str:
    return SNAPSHOT_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def stored(stored_text: str) -> dict[str, Any]:
    return json.loads(stored_text)


@pytest.fixture(scope="module")
def current() -> dict[str, Any]:
    return snapshot_api.build_snapshot()


@pytest.fixture(scope="module")
def diff(stored: dict[str, Any], current: dict[str, Any]) -> Any:
    return snapshot_api.compare_snapshots(stored, current)


def _assert_clean(request: pytest.FixtureRequest, diff: Any, failures: list[str]) -> None:
    if diff.additions:
        request.node.add_report_section("call", "contract additions (allowed)", "\n".join(diff.additions))
    assert not failures, "\n".join([*failures, snapshot_api.refresh_hint()])


# ------------------------------------------------------------------ the real API against the file
@pytest.mark.spec("contract")
def test_public_names(request: pytest.FixtureRequest, diff: Any) -> None:
    _assert_clean(request, diff, diff.names)


@pytest.mark.spec("contract")
def test_signatures(request: pytest.FixtureRequest, diff: Any) -> None:
    _assert_clean(request, diff, diff.signatures)


@pytest.mark.spec("contract")
def test_output_columns(request: pytest.FixtureRequest, diff: Any) -> None:
    _assert_clean(request, diff, diff.outputs)


@pytest.mark.spec("contract")
def test_dataclass_field_order(request: pytest.FixtureRequest, diff: Any) -> None:
    _assert_clean(request, diff, diff.fields)


@pytest.mark.spec("contract")
def test_snapshot_file_is_canonical(stored_text: str, stored: dict[str, Any]) -> None:
    assert snapshot_api.dumps(stored) == stored_text, (
        f"{SNAPSHOT_PATH.name} is not in canonical form (sorted keys, indent 2, trailing newline); {snapshot_api.refresh_hint()}"
    )
    assert set(stored) == {"outputs", "symbols"}
    assert stored["symbols"], "empty symbol table"


# ------------------------------------------------------------------ self-test of the rule on synthetic snapshots
def _param(name: str, kind: str = "POSITIONAL_OR_KEYWORD", default: str = REQUIRED, annotation: str = "int") -> dict[str, str]:
    return {"name": name, "kind": kind, "default": default, "annotation": annotation}


BASE_PARAMS = [
    _param("data", annotation="ScoreResult"),
    _param("by", annotation="str"),
    _param("gamma", default="0.0", annotation="float"),
]
BASE_FIELDS = ["id", "layer", "weight"]
BASE_COLUMNS = ["n_cases", "mean_score", "gap", "PI"]


def _synthetic(
    params: list[dict[str, str]] = BASE_PARAMS,
    fields: list[str] = BASE_FIELDS,
    columns: list[str] = BASE_COLUMNS,
    extra_symbols: dict[str, Any] | None = None,
) -> dict[str, Any]:
    init = {"parameters": [_param(f, default="1") for f in fields], "returns": "None"}
    symbols: dict[str, Any] = {
        "wise.prioritize": {"kind": "function", "parameters": params, "returns": "pd.DataFrame"},
        "wise.NormConstraint": {
            "kind": "class",
            "bases": ["object"],
            "dataclass_fields": fields,
            "init": init,
            "members": {"to_dict": {"kind": "method", "parameters": [], "returns": "dict[str, Any]"}},
            "class_constants": {},
        },
        "wise.SCHEMA_VERSION": {"kind": "constant", "type": "int", "value": "2"},
    }
    symbols.update(extra_symbols or {})
    return {"symbols": symbols, "outputs": {"prioritize": {"columns": columns, "index": ["company"]}}}


@pytest.mark.spec("contract")
def test_rule_allows_appended_keyword_only_parameter_with_default() -> None:
    new = _synthetic(params=[*BASE_PARAMS, _param("baseline", kind="KEYWORD_ONLY", default="None", annotation="float | None")])
    diff = snapshot_api.compare_snapshots(_synthetic(), new)
    assert diff.ok, diff.failures
    assert diff.additions == ["wise.prioritize: new keyword-only parameter baseline: float | None = None"]


@pytest.mark.spec("contract")
@pytest.mark.parametrize(
    "extra",
    [
        pytest.param(_param("baseline", kind="KEYWORD_ONLY"), id="keyword-only-without-default"),
        pytest.param(_param("baseline", default="None"), id="positional-or-keyword-with-default"),
    ],
)
def test_rule_rejects_other_appended_parameters(extra: dict[str, str]) -> None:
    diff = snapshot_api.compare_snapshots(_synthetic(), _synthetic(params=[*BASE_PARAMS, extra]))
    assert len(diff.signatures) == 1
    assert not diff.additions
    assert "wise.prioritize" in diff.signatures[0]
    assert "'baseline'" in diff.signatures[0]


@pytest.mark.spec("contract")
def test_rule_rejects_renamed_parameter() -> None:
    renamed = [BASE_PARAMS[0], BASE_PARAMS[1], _param("shrinkage", default="0.0", annotation="float")]
    diff = snapshot_api.compare_snapshots(_synthetic(), _synthetic(params=renamed))
    assert not diff.ok
    assert len(diff.signatures) == 1
    message = diff.signatures[0]
    assert message.startswith("wise.prioritize: parameter 'gamma' at position 2 is now 'shrinkage'")
    assert "old: (data: ScoreResult, by: str, gamma: float = 0.0)" in message
    assert "new: (data: ScoreResult, by: str, shrinkage: float = 0.0)" in message


@pytest.mark.spec("contract")
@pytest.mark.parametrize(
    ("params", "expected"),
    [
        pytest.param(BASE_PARAMS[:2], "removed parameter(s) ['gamma']", id="removed"),
        pytest.param([BASE_PARAMS[1], BASE_PARAMS[0], BASE_PARAMS[2]], "renamed or reordered", id="reordered"),
        pytest.param(
            [*BASE_PARAMS[:2], _param("gamma", default="1.0", annotation="float")], "changed default: 0.0 -> 1.0", id="default"
        ),
        pytest.param(
            [*BASE_PARAMS[:2], _param("gamma", kind="KEYWORD_ONLY", default="0.0", annotation="float")], "changed kind", id="kind"
        ),
    ],
)
def test_rule_rejects_parameter_changes(params: list[dict[str, str]], expected: str) -> None:
    diff = snapshot_api.compare_snapshots(_synthetic(), _synthetic(params=params))
    assert any(expected in message and message.startswith("wise.prioritize:") for message in diff.signatures), diff.signatures


@pytest.mark.spec("contract")
def test_rule_rejects_removed_output_column() -> None:
    diff = snapshot_api.compare_snapshots(_synthetic(), _synthetic(columns=["n_cases", "mean_score", "PI"]))
    assert diff.outputs == [
        "prioritize: output columns changed\n"
        "  old: {'columns': ['n_cases', 'mean_score', 'gap', 'PI'], 'index': ['company']}\n"
        "  new: {'columns': ['n_cases', 'mean_score', 'PI'], 'index': ['company']}"
    ]
    assert not diff.names
    assert not diff.signatures
    assert not diff.fields


@pytest.mark.spec("contract")
def test_rule_rejects_added_output_column_and_index_change() -> None:
    added = snapshot_api.compare_snapshots(_synthetic(), _synthetic(columns=[*BASE_COLUMNS, "se"]))
    assert len(added.outputs) == 1
    assert "prioritize" in added.outputs[0]
    reindexed = _synthetic()
    reindexed["outputs"]["prioritize"]["index"] = [None]
    assert len(snapshot_api.compare_snapshots(_synthetic(), reindexed).outputs) == 1


@pytest.mark.spec("contract")
def test_rule_on_public_names() -> None:
    removed = _synthetic()
    del removed["symbols"]["wise.SCHEMA_VERSION"]
    diff = snapshot_api.compare_snapshots(_synthetic(), removed)
    assert diff.names == ["wise.SCHEMA_VERSION: removed public name (constant)"]

    new_name: dict[str, Any] = {
        "wise.Backlog": {
            "kind": "class",
            "bases": ["object"],
            "dataclass_fields": None,
            "init": None,
            "members": {},
            "class_constants": {},
        }
    }
    diff = snapshot_api.compare_snapshots(_synthetic(), _synthetic(extra_symbols=new_name))
    assert diff.ok
    assert diff.additions == ["wise.Backlog: new public name (class)"]

    changed_kind = _synthetic()
    changed_kind["symbols"]["wise.SCHEMA_VERSION"] = {"kind": "function", "parameters": [], "returns": "int"}
    diff = snapshot_api.compare_snapshots(_synthetic(), changed_kind)
    assert diff.names == ["wise.SCHEMA_VERSION: kind changed constant -> function"]


@pytest.mark.spec("contract")
def test_rule_on_dataclass_fields() -> None:
    reordered = snapshot_api.compare_snapshots(_synthetic(), _synthetic(fields=["layer", "id", "weight"]))
    assert len(reordered.fields) == 1
    assert reordered.fields[0].startswith("wise.NormConstraint: dataclass fields removed, renamed or reordered")
    assert "old: ['id', 'layer', 'weight']" in reordered.fields[0]
    assert "new: ['layer', 'id', 'weight']" in reordered.fields[0]

    removed = snapshot_api.compare_snapshots(_synthetic(), _synthetic(fields=["id", "layer"]))
    assert len(removed.fields) == 1
    assert removed.signatures

    appended = snapshot_api.compare_snapshots(_synthetic(), _synthetic(fields=[*BASE_FIELDS, "description"]))
    assert not appended.fields
    assert "wise.NormConstraint: new dataclass field(s) ['description']" in appended.additions
    assert appended.signatures, "the appended field reached __init__ as a positional parameter, which the signature rule rejects"


@pytest.mark.spec("contract")
def test_rule_on_class_members() -> None:
    lost_method = _synthetic()
    lost_method["symbols"]["wise.NormConstraint"]["members"] = {}
    diff = snapshot_api.compare_snapshots(_synthetic(), lost_method)
    assert diff.names == ["wise.NormConstraint.to_dict: removed public member (method)"]

    new_base = _synthetic()
    new_base["symbols"]["wise.NormConstraint"]["bases"] = ["wise.norm.Frozen"]
    diff = snapshot_api.compare_snapshots(_synthetic(), new_base)
    assert len(diff.names) == 1
    assert diff.names[0].startswith("wise.NormConstraint: base classes changed")


@pytest.mark.spec("contract")
def test_failure_message_tells_how_to_refresh() -> None:
    hint = snapshot_api.refresh_hint()
    assert hint.startswith("record the change under Unreleased in CHANGELOG.md and refresh with  ")
    assert hint.endswith(" scripts/snapshot_api.py --write")
    assert sys.executable in hint
