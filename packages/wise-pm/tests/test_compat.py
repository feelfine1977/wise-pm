"""The compatibility module: every changed 0.1.0 name and every private a consumer touches is accounted for.

Decision record 0001, section 2: ``wise.compat`` serves the names and privates the
consumers use today, with deprecation warnings, until 1.0. The 0.1.0 interface is
the shape record of the ``v0.1.0`` source under ``tests/data/compat`` (see
``_support/api_shapes``); the consumer's use is a scan of the workbench checkout
(``_support/consumer_usage``), taken live when the checkout is present and from
the committed record otherwise.
"""

from __future__ import annotations

import dataclasses
import json
import os
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from _support.api_shapes import Shape, canonical, compatible, shapes
from _support.consumer_usage import BASELINE, DATA, RECORD, Usage, member_names, scan

import wise
import wise.compat
import wise.norm
from wise.compat import PATHS
from wise.log import EventLog

OLD: dict[str, Shape] = json.loads(BASELINE.read_text(encoding="utf-8"))
OLD_OUTPUTS: dict[str, Any] = json.loads((DATA / "outputs_0_1_0.json").read_text(encoding="utf-8"))
SNAPSHOT = Path(__file__).resolve().parent / "contract" / "api_snapshot.json"
#: The workbench checkout; the compatibility gate (scripts/compat_gate.sh) looks for it next to the repository.
WORKBENCH = Path(os.environ.get("WISE_WORKBENCH", Path(__file__).resolve().parents[4] / "wise-workbench"))


@pytest.fixture(scope="module")
def new() -> dict[str, Shape]:
    return shapes()


def _warned_once(call: Callable[[], Any], replacement: str) -> Any:
    """``call()``'s value; it must emit exactly one warning, a DeprecationWarning naming ``replacement``."""
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        value = call()
    messages = [(w.category, str(w.message)) for w in seen]
    assert len(messages) == 1, messages
    assert messages[0][0] is DeprecationWarning
    assert replacement in messages[0][1]
    assert "removed in 1.0" in messages[0][1]
    return value


# ------------------------------------------------------------------ the deprecated paths
def _prioritize() -> None:
    result = wise.score(wise.running_p2p_log(), wise.running_p2p_norm())
    backlog = wise.prioritize(result, by="company", view="Finance", gamma=1.0)
    replacement = PATHS["wise.prioritization.prioritize"].replacement
    legacy: Any = backlog  # the 0.1.0 frame form is hidden from type checkers
    frame = _warned_once(lambda: wise.compat.prioritize(result, by="company", view="Finance", gamma=1.0), replacement)
    pd.testing.assert_frame_equal(frame, backlog.table, check_exact=True)
    assert frame.attrs == {
        "view": "Finance",
        "gamma": 1.0,
        "baseline": backlog.spec.global_mean,
        "volume": "cases",
        "by": ["company"],
    }
    assert backlog.table.attrs == {}
    # the record used as the 0.1.0 frame: one warning per use, same values
    assert _warned_once(lambda: len(legacy), "Backlog.table") == len(backlog.table)
    column = _warned_once(lambda: legacy["stable_PI"], "Backlog.table")
    pd.testing.assert_series_equal(column, backlog.table["stable_PI"], check_exact=True)
    assert _warned_once(lambda: legacy.attrs, "Backlog.spec") == frame.attrs


def _norm_from_dict() -> None:
    document = wise.running_p2p_norm().to_dict()
    assert document.pop("schema_version") == 2
    with pytest.raises(wise.NormError, match="schema_version"):
        wise.Norm.from_dict(document)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        norm = wise.compat.norm_from_dict(document)
    deprecations = [w for w in seen if w.category is DeprecationWarning]
    assert len(deprecations) == 1
    assert PATHS["wise.norm.Norm.from_dict"].replacement in str(deprecations[0].message)
    # the reader announces the legacy semantics of a version 2 norm, as for any other version 2 document
    assert [w.category for w in seen if w not in deprecations] == [wise.LegacySemanticsWarning]
    with pytest.warns(wise.LegacySemanticsWarning):
        stated = wise.Norm.from_dict({**document, "schema_version": 2})
    assert norm.schema_version == 2
    assert norm.fingerprint() == stated.fingerprint()
    v3 = _warned_once(lambda: wise.compat.norm_from_dict(wise.migrate(stated).document), "schema_version")
    assert v3.schema_version == 3


def _rule_attributes() -> None:
    rule = {"all": [{"attr": "flow_type", "eq": "DF1"}, {"not": {"attr": "amount", "lt": 1}}]}
    replacement = "wise.selection.parse(rule).columns"
    assert _warned_once(lambda: wise.compat._rule_attributes(rule), replacement) == {"flow_type", "amount"}
    assert _warned_once(lambda: wise.norm._rule_attributes(rule), replacement) == {"flow_type", "amount"}


def _to_ts() -> None:
    log = wise.running_p2p_log()
    value = _warned_once(lambda: wise.compat.EventLog._to_ts(log, "2024-03-01"), "EventLog.to_timestamp")
    assert value == log.to_timestamp("2024-03-01") == pd.Timestamp("2024-03-01")


def _index_array(name: str, replacement: str) -> Callable[[], None]:
    def drive() -> None:
        log = wise.running_p2p_log()
        array = _warned_once(lambda: getattr(log, name), f"EventLog.case_index.{replacement}")
        np.testing.assert_array_equal(array, getattr(log.case_index, replacement))

    return drive


def _assign_cases() -> None:
    log, norm = wise.running_p2p_log(), wise.running_p2p_norm()
    before = wise.score(log, norm).scores

    def assign() -> None:
        log.cases = log.cases

    _warned_once(assign, "EventLog.with_case_attributes")
    pd.testing.assert_frame_equal(wise.score(log, norm).scores, before, check_exact=True)


DRIVERS: dict[str, Callable[[], None]] = {
    "wise.prioritization.prioritize": _prioritize,
    "wise.norm.Norm.from_dict": _norm_from_dict,
    "wise.norm._rule_attributes": _rule_attributes,
    "wise.log.EventLog._to_ts": _to_ts,
    "wise.log.EventLog._codes": _index_array("_codes", "case_codes"),
    "wise.log.EventLog._ts": _index_array("_ts", "timestamps"),
    "wise.log.EventLog._valid": _index_array("_valid", "valid"),
    "wise.log.EventLog._starts": _index_array("_starts", "starts"),
    "wise.log.EventLog.cases": _assign_cases,
}


@pytest.mark.spec("decision:0001/2")
def test_every_deprecated_path_has_a_driver_below() -> None:
    assert set(DRIVERS) == {name for name, path in PATHS.items() if path.status == "deprecated"}


@pytest.mark.spec("decision:0001/2")
@pytest.mark.parametrize("name", sorted(DRIVERS))
def test_a_deprecated_path_warns_once_naming_its_replacement_and_returns_its_value(name: str) -> None:
    DRIVERS[name]()


@pytest.mark.spec("decision:0001/2")
def test_the_kept_private_builds_a_result_as_in_0_1_0_without_a_warning() -> None:
    result = wise.score(wise.running_p2p_log(), wise.running_p2p_norm())
    detached = wise.compat.ScoreResult(
        norm=result.norm,
        cases=result.cases,
        violations=result.violations,
        in_scope=result.in_scope,
        scores=result.scores,
        contributions=result.contributions,
        mode=result.mode,
        norm_fingerprint=result.norm_fingerprint,
        _weights=result._weights,
    )
    pd.testing.assert_frame_equal(detached.effective_weights("Finance"), result.effective_weights("Finance"), check_exact=True)
    replaced = dataclasses.replace(result, log=None)
    assert replaced.log is None
    pd.testing.assert_frame_equal(replaced.penalties("Finance"), result.penalties("Finance"), check_exact=True)
    assert [p.status for n, p in PATHS.items() if n.endswith("._weights")] == ["kept"]


@pytest.mark.spec("decision:0001/2")
@pytest.mark.parametrize("name", sorted(PATHS))
def test_each_listed_name_is_a_0_1_0_name_and_is_reachable_through_the_module(name: str) -> None:
    assert name in OLD, name
    path = PATHS[name]
    assert path.replacement
    assert (path.status == "changed") == (not path.served_by)
    if path.status == "changed":
        assert path.note
        return
    module, _, member = name.rpartition(".")
    owner = module.rpartition(".")[2]
    if OLD.get(module, {}).get("kind") == "class":  # a member: served by the object, reachable through the class
        cls = getattr(wise.compat, owner)
        assert hasattr(cls, member) or member in {f.name for f in dataclasses.fields(cls)}
    for where in path.served_by:
        if where.startswith("wise.compat."):
            assert callable(getattr(wise.compat, where.removeprefix("wise.compat.")))


# ------------------------------------------------------------------ coverage of the 0.1.0 interface
def _uncovered_public(new: dict[str, Shape], baseline: dict[str, Shape] = OLD) -> list[str]:
    missing = []
    for name, old in sorted(baseline.items()):
        if not old.get("public") or old["kind"] == "alias" or name in PATHS:
            continue
        current = canonical(new, name)
        now = new.get(current) if current else None
        if not compatible(old, now):
            missing.append(name)
        elif now is not None and old["kind"] == "attribute" and now["kind"] == "property" and not now["settable"]:
            missing.append(name)  # readable as before, but no longer assignable
    return missing


@pytest.mark.spec("decision:0001/2")
def test_every_0_1_0_public_name_whose_location_or_type_changed_is_listed(new: dict[str, Shape]) -> None:
    assert _uncovered_public(new) == []


@pytest.mark.spec("decision:0001/2")
def test_every_0_1_0_output_table_whose_columns_changed_is_listed() -> None:
    outputs = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["outputs"]
    changed = {name for name, columns in OLD_OUTPUTS.items() if outputs.get(name) != columns}
    assert changed == {"compare_periods"}
    assert {f"wise.prioritization.{name}" for name in changed} <= set(PATHS)


def test_the_coverage_check_sees_a_changed_signature_and_a_removed_name(new: dict[str, Shape]) -> None:
    old = dict(OLD)
    old["wise.prioritization.pareto"] = {**OLD["wise.prioritization.pareto"], "returns": "pd.Series"}
    old["wise.constraints.gone"] = {"kind": "function", "params": [], "returns": "None", "public": True}
    assert _uncovered_public(new, old) == ["wise.constraints.gone", "wise.prioritization.pareto"]


# ------------------------------------------------------------------ coverage of the consumer's use
def _owners(member: str) -> list[str]:
    """The 0.1.0 class members called ``member``."""
    return sorted(n for n in OLD if n.rpartition(".")[2] == member and OLD.get(n.rpartition(".")[0], {}).get("kind") == "class")


def _private(name: str) -> bool:
    return any(part.startswith("_") and not part.startswith("__") for part in name.split("."))


def uncovered(usage: Usage, new: dict[str, Shape], paths: Mapping[str, object] = PATHS) -> list[str]:
    """Each name or member use of the consumer that neither works unchanged nor is listed in ``PATHS``."""
    out = []
    for name in sorted(usage.names):
        old_name = canonical(OLD, name)
        if old_name is None:
            if canonical(new, name) is None and name not in {"wise.__version__"} and not _is_module(name):
                out.append(f"{name}: neither a 0.1.0 nor a current name")
            continue
        if old_name in paths:
            continue
        current = canonical(new, name)
        if _private(old_name) or not compatible(OLD[old_name], new.get(current) if current else None):
            out.append(f"{name}: changed and not in wise.compat.PATHS")
    for member, contexts in sorted(usage.members.items()):
        for owner in _owners(member):
            if owner in paths:
                continue
            old = OLD[owner]
            current = canonical(new, owner)
            now = new.get(current) if current else None
            if _private(owner):
                out.append(f"{owner} ({', '.join(sorted(contexts))}): a private not in wise.compat.PATHS")
                continue
            if "read" in contexts and not compatible(old, now):
                out.append(f"{owner} (read): changed and not in wise.compat.PATHS")
            became_property = old["kind"] in {"attribute", "field"} and now is not None and now["kind"] == "property"
            if "assign" in contexts and (now is None or (became_property and not now["settable"])):
                out.append(f"{owner} (assign): no longer assignable and not in wise.compat.PATHS")
            if "write" in contexts and became_property:
                out.append(f"{owner} (write): writes into a copy and not in wise.compat.PATHS")
    return out


def _is_module(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


def _consumer_usage() -> Usage:
    if WORKBENCH.is_dir():
        return scan(WORKBENCH, member_names(OLD))
    return Usage.from_dict(json.loads(RECORD.read_text(encoding="utf-8")))


@pytest.mark.spec("decision:0001/2")
def test_every_name_and_private_the_consumer_uses_is_covered(new: dict[str, Shape]) -> None:
    usage = _consumer_usage()
    assert {"wise.norm._rule_attributes", "wise.prioritize", "wise.Norm.from_dict"} <= set(usage.names)
    assert {"_codes", "_to_ts", "_weights", "cases", "events"} <= set(usage.members)
    assert uncovered(usage, new) == []


@pytest.mark.spec("decision:0001/2")
def test_the_committed_record_is_the_scan_of_the_checkout() -> None:
    if not WORKBENCH.is_dir():
        pytest.skip(f"no workbench checkout at {WORKBENCH} (set WISE_WORKBENCH); the committed record is checked instead")
    recorded = json.loads(RECORD.read_text(encoding="utf-8"))
    assert scan(WORKBENCH, member_names(OLD)).to_dict() == recorded, (
        "the workbench changed: refresh the record with tests/_support/consumer_usage.py and review wise.compat.PATHS"
    )


SOURCE = '''
import wise
from wise.norm import _rule_attributes
from wise import compare_periods as cp

log = wise.EventLog(frame)
log._codes
log.events["alias"] = log.events["amount"]
log.cases = log.cases
log.case_attributes = []
self._codes = None
wise.ScoreResult(_weights={})
wise.not_a_name
cp(a, b)
'''


def test_the_consumer_check_reports_each_kind_of_uncovered_use(tmp_path: Path, new: dict[str, Shape]) -> None:
    (tmp_path / "consumer.py").write_text(SOURCE, encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "ignored.py").write_text("log._starts\n", encoding="utf-8")
    usage = scan(tmp_path, member_names(OLD))
    assert usage.names.keys() == {
        "wise.EventLog",
        "wise.ScoreResult",
        "wise.compare_periods",
        "wise.norm._rule_attributes",
        "wise.not_a_name",
    }
    assert {m: sorted(c) for m, c in usage.members.items() if m in {"_codes", "_starts", "_weights", "case_attributes"}} == {
        "_codes": ["read"],
        "_weights": ["keyword"],
        "case_attributes": ["assign"],
    }
    assert dict(usage.members["events"]) == {"read": {"consumer.py"}, "write": {"consumer.py"}}
    assert dict(usage.members["cases"]) == {"assign": {"consumer.py"}, "read": {"consumer.py"}}
    assert uncovered(usage, new) == ["wise.not_a_name: neither a 0.1.0 nor a current name"]
    assert uncovered(usage, new, paths={}) == [
        "wise.ScoreResult: changed and not in wise.compat.PATHS",
        "wise.norm._rule_attributes: changed and not in wise.compat.PATHS",
        "wise.not_a_name: neither a 0.1.0 nor a current name",
        "wise.log.EventLog._codes (read): a private not in wise.compat.PATHS",
        "wise.scoring.ScoreResult._weights (keyword): a private not in wise.compat.PATHS",
        "wise.log.EventLog.case_attributes (assign): no longer assignable and not in wise.compat.PATHS",
        "wise.log.EventLog.events (write): writes into a copy and not in wise.compat.PATHS",
    ]


def test_the_record_round_trips() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert Usage.from_dict(record).to_dict() == record


def test_reading_an_attribute_that_became_a_property_is_unchanged() -> None:
    log = wise.running_p2p_log()
    assert isinstance(EventLog.events, property)
    assert isinstance(log.events, pd.DataFrame)
    assert isinstance(log.case_attributes, list)
