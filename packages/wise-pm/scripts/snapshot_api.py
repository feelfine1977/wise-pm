"""Snapshot the public API contract of ``wise`` (signatures and output columns).

The snapshot pins what downstream code (the workbench, ``wise-analytics``,
``process-knowledge``) can rely on:

* every name in ``wise.__all__`` plus the module-level names imported directly
  (``wise.errors.*``, ``wise.constraints.in_units`` / ``constraint_from_dict``,
  ``wise.norm.SCORING_MODES`` / ``DEFAULT_SCORING_MODE``): kind, signature
  (functions), ``__init__`` and public members, dataclass field order and
  bases (classes), repr (constants), public members (modules);
* the column and index names of every table the running example produces.

``tests/contract/test_public_api.py`` rebuilds the snapshot in memory and
compares it with ``tests/contract/api_snapshot.json`` under one rule: a new
keyword-only parameter with a default appended to a function or method is
allowed, a new public name is allowed but reported, anything else fails.

Usage::

    python scripts/snapshot_api.py            # print the snapshot as JSON
    python scripts/snapshot_api.py --write    # refresh tests/contract/api_snapshot.json
    python scripts/snapshot_api.py --check    # exit 1 when the stored file is out of date

Two values are normalised so the snapshot survives a release: a default or
constant equal to ``wise.__version__`` is written as ``"<wise.__version__>"``.
Nothing else is rewritten; there are no timestamps.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import importlib
import inspect
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

import wise

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = REPO_ROOT / "tests" / "contract" / "api_snapshot.json"

REQUIRED = "<required>"
VERSION_PLACEHOLDER = "<wise.__version__>"
PACKAGE = wise.__name__

#: Module-level names the downstream workbench imports directly, outside ``wise.__all__``.
EXTRA_SYMBOLS: dict[str, tuple[str, ...]] = {
    "wise.errors": ("WiseError", "NormError", "LogSchemaError", "NotScoredError"),
    "wise.constraints": ("in_units", "constraint_from_dict"),
    "wise.norm": ("SCORING_MODES", "DEFAULT_SCORING_MODE"),
}

Snapshot = dict[str, Any]
Params = list[dict[str, str]]


# ------------------------------------------------------------------ symbols
def _annotation(ann: Any) -> str:
    if ann is inspect.Parameter.empty:
        return ""
    return ann if isinstance(ann, str) else inspect.formatannotation(ann)


def _value_repr(value: Any) -> str:
    if isinstance(value, str) and value == wise.__version__:
        return VERSION_PLACEHOLDER
    return repr(value)


def _parameters(sig: inspect.Signature, *, drop_first: bool) -> Params:
    params = list(sig.parameters.values())[1 if drop_first else 0 :]
    return [
        {
            "name": p.name,
            "kind": p.kind.name,
            "default": REQUIRED if p.default is inspect.Parameter.empty else _value_repr(p.default),
            "annotation": _annotation(p.annotation),
        }
        for p in params
    ]


def _callable_entry(func: Callable[..., Any], *, drop_first: bool = False) -> dict[str, Any]:
    sig = inspect.signature(func)
    return {"parameters": _parameters(sig, drop_first=drop_first), "returns": _annotation(sig.return_annotation)}


def _qualified(cls: type) -> str:
    return cls.__name__ if cls.__module__ == "builtins" else f"{cls.__module__}.{cls.__qualname__}"


def _member_entry(cls: type, name: str, raw: Any) -> dict[str, Any] | None:
    """Describe one public attribute found in a class ``__dict__``; ``None`` for plain values."""
    if isinstance(raw, classmethod):
        return {"kind": "classmethod", **_callable_entry(getattr(cls, name))}
    if isinstance(raw, staticmethod):
        return {"kind": "staticmethod", **_callable_entry(raw.__func__)}
    if isinstance(raw, property):
        entry: dict[str, Any] = {"kind": "property", "settable": raw.fset is not None}
        if raw.fget is not None:
            entry["returns"] = _annotation(inspect.signature(raw.fget).return_annotation)
        return entry
    if isinstance(raw, functools.cached_property):
        return {"kind": "cached_property", "returns": _annotation(inspect.signature(raw.func).return_annotation)}
    if inspect.isfunction(raw):
        return {"kind": "method", **_callable_entry(raw, drop_first=True)}
    return None


def _class_entry(cls: type) -> dict[str, Any]:
    is_dc = dataclasses.is_dataclass(cls)
    field_names = [f.name for f in dataclasses.fields(cls)] if is_dc else None
    members: dict[str, Any] = {}
    constants: dict[str, str] = {}
    # Only attributes defined inside this package: builtin members (``Exception.add_note``,
    # ``with_traceback``) differ between Python versions and are not part of the contract.
    for klass in reversed(cls.__mro__):
        if not klass.__module__.startswith(PACKAGE):
            continue
        for name, raw in vars(klass).items():
            if name.startswith("_") or (field_names is not None and name in field_names):
                continue
            entry = _member_entry(cls, name, raw)
            if entry is not None:
                members[name] = entry
                constants.pop(name, None)
            else:
                constants[name] = _value_repr(raw)
                members.pop(name, None)
    init_fn = inspect.getattr_static(cls, "__init__")
    init = _callable_entry(init_fn, drop_first=True) if inspect.isfunction(init_fn) else None
    return {
        "kind": "class",
        "bases": [_qualified(b) for b in cls.__bases__],
        "dataclass_fields": field_names,
        "init": init,
        "members": members,
        "class_constants": constants,
    }


def _module_entry(module: Any) -> dict[str, Any]:
    members: dict[str, str] = {}
    for name, obj in vars(module).items():
        if name.startswith("_") or inspect.ismodule(obj):
            continue
        if inspect.isclass(obj) or inspect.isfunction(obj):
            if obj.__module__ == module.__name__:
                members[name] = "class" if inspect.isclass(obj) else "function"
        elif name.isupper():
            members[name] = "constant"
    return {"kind": "module", "members": members}


def describe_symbol(obj: Any) -> dict[str, Any]:
    """Contract entry for one public object."""
    if inspect.ismodule(obj):
        return _module_entry(obj)
    if inspect.isclass(obj):
        return _class_entry(obj)
    if inspect.isroutine(obj):
        return {"kind": "function", **_callable_entry(obj)}
    return {"kind": "constant", "type": type(obj).__name__, "value": _value_repr(obj)}


def symbol_table() -> dict[str, dict[str, Any]]:
    """``{dotted import path: entry}`` for ``wise.__all__`` and :data:`EXTRA_SYMBOLS`."""
    table = {f"{PACKAGE}.{name}": describe_symbol(getattr(wise, name)) for name in wise.__all__}
    for module_name, names in EXTRA_SYMBOLS.items():
        module = importlib.import_module(module_name)
        for name in names:
            table[f"{module_name}.{name}"] = describe_symbol(getattr(module, name))
    return table


# ------------------------------------------------------------------ outputs
def _table(df: pd.DataFrame) -> dict[str, Any]:
    return {"columns": [str(c) for c in df.columns], "index": [None if n is None else str(n) for n in df.index.names]}


def _series(s: pd.Series, *, values: bool = False) -> dict[str, Any]:
    entry: dict[str, Any] = {"name": None if s.name is None else str(s.name)}
    if values:
        entry["index_values"] = [str(v) for v in s.index]
    return entry


def output_tables() -> dict[str, dict[str, Any]]:
    """Column and index names of every table the running example produces."""
    log = wise.running_p2p_log()
    norm = wise.running_p2p_norm()
    result = wise.score(log, norm)
    view = result.views[0]
    backlog = wise.prioritize(result, by="company", view=view, z=1.96)
    drivers = wise.layer_drivers(result, by="company", view=view)
    censored = wise.right_censored(log, closure="Clear Invoice")
    replication = wise.event_replication(log)
    gamma = wise.estimate_gamma(result, by="company", view=view)
    overlap = wise.top_k_overlap(backlog, backlog)
    return {
        "EventLog.validate": _series(log.validate(), values=True),
        "Norm.describe": _table(norm.describe()),
        "Norm.layer_weight_table": _table(norm.layer_weight_table()),
        "Norm.weight_table": _table(norm.weight_table()),
        "ScoreResult.effective_weights": _table(result.effective_weights(view)),
        "ScoreResult.frame": _table(result.frame()),
        "ScoreResult.frame[view]": _table(result.frame(view)),
        "ScoreResult.penalties": _table(result.penalties(view)),
        "ScoreResult.summary": _table(result.summary()),
        "ScoreResult.worst_cases": _table(result.worst_cases(view)),
        "compare_periods": _table(wise.compare_periods(backlog, backlog)),
        "concentration": _table(wise.concentration(backlog)),
        "constraint_drivers": _table(wise.constraint_drivers(result, view, {"company": "B"})),
        "cross_case_replication": _series(wise.cross_case_replication(log, "vendor")),
        "estimate_gamma": {"type": type(gamma).__name__},
        "event_replication": _table(replication),
        "gap_retained": _table(wise.gap_retained(result, view, by="company", exclude=censored)),
        "hotspot_table": _table(wise.hotspot_table(backlog, drivers=drivers)),
        "layer_drivers": _table(drivers),
        "left_truncated": _series(wise.left_truncated(log, opening="Create Purchase Order Item")),
        "pareto": _table(wise.pareto(backlog)),
        "penalty_mass": _table(wise.penalty_mass(result, view, by="vendor")),
        "prioritize": _table(backlog),
        "right_censored": _series(censored),
        "timestamp_outliers": _series(wise.timestamp_outliers(log)),
        "top_k_overlap": {"type": type(overlap).__name__},
        "validation_table": _table(wise.validation_table(result, view, by="company", censored=censored, replication=replication)),
        "view_agreement": _table(wise.view_agreement(result, by="company")),
    }


def build_snapshot() -> Snapshot:
    return {"outputs": output_tables(), "symbols": symbol_table()}


def dumps(snapshot: Mapping[str, Any]) -> str:
    """Canonical text: sorted keys, two-space indent, trailing newline."""
    return json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def load_snapshot(path: Path = SNAPSHOT_PATH) -> Snapshot:
    return json.loads(path.read_text(encoding="utf-8"))


def count_columns(snapshot: Mapping[str, Any]) -> int:
    return sum(len(entry.get("columns", ())) for entry in snapshot["outputs"].values())


# ------------------------------------------------------------------ comparison
@dataclass
class ContractDiff:
    """Result of :func:`compare_snapshots`, one list per contract test."""

    names: list[str] = field(default_factory=list)
    signatures: list[str] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    additions: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[str]:
        return [*self.names, *self.signatures, *self.fields, *self.outputs]

    @property
    def ok(self) -> bool:
        return not self.failures


def _fmt_param(p: Mapping[str, str]) -> str:
    text = p["name"]
    if p["kind"] == "VAR_POSITIONAL":
        text = "*" + text
    elif p["kind"] == "VAR_KEYWORD":
        text = "**" + text
    if p["annotation"]:
        text += f": {p['annotation']}"
    if p["default"] != REQUIRED:
        text += f" = {p['default']}"
    return text


def _fmt_params(params: list[Mapping[str, str]]) -> str:
    return "(" + ", ".join(_fmt_param(p) for p in params) + ")"


def _split_var_keyword(params: Params) -> tuple[Params, dict[str, str] | None]:
    if params and params[-1]["kind"] == "VAR_KEYWORD":
        return params[:-1], params[-1]
    return params, None


def _compare_signature(symbol: str, old: Mapping[str, Any], new: Mapping[str, Any], diff: ContractDiff) -> None:
    old_params, old_var = _split_var_keyword(list(old["parameters"]))
    new_params, new_var = _split_var_keyword(list(new["parameters"]))
    old_text, new_text = _fmt_params(old["parameters"]), _fmt_params(new["parameters"])
    problems: list[str] = []
    if old_var != new_var:
        problems.append("the **kwargs parameter changed")
    for position, (o, n) in enumerate(zip(old_params, new_params)):
        if o == n:
            continue
        if o["name"] != n["name"]:
            problems.append(f"parameter {o['name']!r} at position {position} is now {n['name']!r} (renamed or reordered)")
        else:
            changed = [k for k in ("kind", "default", "annotation") if o[k] != n[k]]
            problems.extend(f"parameter {o['name']!r} changed {k}: {o[k]} -> {n[k]}" for k in changed)
    if len(new_params) < len(old_params):
        removed = [p["name"] for p in old_params[len(new_params) :]]
        problems.append(f"removed parameter(s) {removed}")
    for extra in new_params[len(old_params) :]:
        if extra["kind"] != "KEYWORD_ONLY" or extra["default"] == REQUIRED:
            problems.append(f"new parameter {extra['name']!r} is not keyword-only with a default")
        else:
            diff.additions.append(f"{symbol}: new keyword-only parameter {_fmt_param(extra)}")
    if old.get("returns", "") != new.get("returns", ""):
        problems.append(f"return annotation changed: {old.get('returns')!r} -> {new.get('returns')!r}")
    for problem in problems:
        diff.signatures.append(f"{symbol}: {problem}\n  old: {old_text}\n  new: {new_text}")


def _compare_members(symbol: str, old: Mapping[str, Any], new: Mapping[str, Any], diff: ContractDiff) -> None:
    for name in sorted(set(old) - set(new)):
        diff.names.append(f"{symbol}.{name}: removed public member ({old[name]['kind']})")
    for name in sorted(set(new) - set(old)):
        diff.additions.append(f"{symbol}.{name}: new public member ({new[name]['kind']})")
    for name in sorted(set(old) & set(new)):
        o, n = old[name], new[name]
        if o["kind"] != n["kind"]:
            diff.names.append(f"{symbol}.{name}: kind changed {o['kind']} -> {n['kind']}")
        elif "parameters" in o:
            _compare_signature(f"{symbol}.{name}", o, n, diff)
        elif o != n:
            diff.signatures.append(f"{symbol}.{name}: {o['kind']} changed\n  old: {o}\n  new: {n}")


def _compare_fields(symbol: str, old: list[str] | None, new: list[str] | None, diff: ContractDiff) -> None:
    if old == new:
        return
    if old is None or new is None:
        diff.fields.append(f"{symbol}: dataclass status changed\n  old: {old}\n  new: {new}")
    elif new[: len(old)] != old:
        diff.fields.append(f"{symbol}: dataclass fields removed, renamed or reordered\n  old: {old}\n  new: {new}")
    else:
        diff.additions.append(f"{symbol}: new dataclass field(s) {new[len(old) :]}")


def _compare_class(symbol: str, old: Mapping[str, Any], new: Mapping[str, Any], diff: ContractDiff) -> None:
    if old["bases"] != new["bases"]:
        diff.names.append(f"{symbol}: base classes changed\n  old: {old['bases']}\n  new: {new['bases']}")
    _compare_fields(symbol, old.get("dataclass_fields"), new.get("dataclass_fields"), diff)
    if (old["init"] is None) != (new["init"] is None):
        diff.signatures.append(f"{symbol}.__init__: presence changed\n  old: {old['init']}\n  new: {new['init']}")
    elif old["init"] is not None:
        _compare_signature(f"{symbol}.__init__", old["init"], new["init"], diff)
    _compare_members(symbol, old["members"], new["members"], diff)
    _compare_constants(symbol, old["class_constants"], new["class_constants"], diff)


def _compare_constants(symbol: str, old: Mapping[str, str], new: Mapping[str, str], diff: ContractDiff) -> None:
    for name in sorted(set(old) - set(new)):
        diff.names.append(f"{symbol}.{name}: removed class constant")
    for name in sorted(set(new) - set(old)):
        diff.additions.append(f"{symbol}.{name}: new class constant = {new[name]}")
    for name in sorted(set(old) & set(new)):
        if old[name] != new[name]:
            diff.signatures.append(f"{symbol}.{name}: value changed\n  old: {old[name]}\n  new: {new[name]}")


def _compare_module(symbol: str, old: Mapping[str, str], new: Mapping[str, str], diff: ContractDiff) -> None:
    for name in sorted(set(old) - set(new)):
        diff.names.append(f"{symbol}.{name}: removed module member ({old[name]})")
    for name in sorted(set(new) - set(old)):
        diff.additions.append(f"{symbol}.{name}: new module member ({new[name]})")
    for name in sorted(set(old) & set(new)):
        if old[name] != new[name]:
            diff.names.append(f"{symbol}.{name}: kind changed {old[name]} -> {new[name]}")


def compare_snapshots(stored: Mapping[str, Any], current: Mapping[str, Any]) -> ContractDiff:
    """Compare two snapshots; see the module docstring for the rule."""
    diff = ContractDiff()
    old_symbols, new_symbols = stored["symbols"], current["symbols"]
    for symbol in sorted(set(old_symbols) - set(new_symbols)):
        diff.names.append(f"{symbol}: removed public name ({old_symbols[symbol]['kind']})")
    for symbol in sorted(set(new_symbols) - set(old_symbols)):
        diff.additions.append(f"{symbol}: new public name ({new_symbols[symbol]['kind']})")
    for symbol in sorted(set(old_symbols) & set(new_symbols)):
        old, new = old_symbols[symbol], new_symbols[symbol]
        if old["kind"] != new["kind"]:
            diff.names.append(f"{symbol}: kind changed {old['kind']} -> {new['kind']}")
        elif old["kind"] == "function":
            _compare_signature(symbol, old, new, diff)
        elif old["kind"] == "class":
            _compare_class(symbol, old, new, diff)
        elif old["kind"] == "module":
            _compare_module(symbol, old["members"], new["members"], diff)
        elif old != new:
            diff.signatures.append(f"{symbol}: constant changed\n  old: {old['value']}\n  new: {new['value']}")

    old_outputs, new_outputs = stored["outputs"], current["outputs"]
    for name in sorted(set(old_outputs) - set(new_outputs)):
        diff.outputs.append(f"{name}: output table no longer captured\n  old: {old_outputs[name]}")
    for name in sorted(set(new_outputs) - set(old_outputs)):
        diff.additions.append(f"{name}: new output table {new_outputs[name]}")
    for name in sorted(set(old_outputs) & set(new_outputs)):
        if old_outputs[name] != new_outputs[name]:
            diff.outputs.append(f"{name}: output columns changed\n  old: {old_outputs[name]}\n  new: {new_outputs[name]}")
    return diff


# ------------------------------------------------------------------ CLI
def refresh_hint(python: str = sys.executable) -> str:
    return f"record the change under Unreleased in CHANGELOG.md and refresh with  {python} scripts/snapshot_api.py --write"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true", help=f"write the snapshot to {SNAPSHOT_PATH.relative_to(REPO_ROOT)}")
    group.add_argument("--check", action="store_true", help="compare with the stored snapshot; exit 1 on a contract change")
    args = parser.parse_args(argv)
    snapshot = build_snapshot()
    text = dumps(snapshot)
    summary = f"{len(snapshot['symbols'])} symbols, {len(snapshot['outputs'])} output tables, {count_columns(snapshot)} columns"
    if args.write:
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {SNAPSHOT_PATH.relative_to(REPO_ROOT)}: {summary}")
        return 0
    if args.check:
        if not SNAPSHOT_PATH.exists():
            print(f"no snapshot at {SNAPSHOT_PATH}; {refresh_hint()}")
            return 1
        diff = compare_snapshots(load_snapshot(), snapshot)
        for line in diff.additions:
            print(f"allowed addition: {line}")
        for line in diff.failures:
            print(f"CONTRACT CHANGE: {line}")
        if diff.failures:
            print(refresh_hint())
            return 1
        print(f"contract unchanged: {summary}")
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
