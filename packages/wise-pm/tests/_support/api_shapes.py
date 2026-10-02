"""The shape of every name the ``wise`` package defines, for comparing two versions of it.

A shape is a JSON-ready record of what a caller relies on: the kind of the name, a
function's parameters and return annotation, a class's bases and constructor, a
member's kind and annotation. :func:`shapes` records every module-level name and
every class member of the importable ``wise`` package, public and private;
:func:`compatible` says whether a caller of the old shape keeps working with the
new one. The record of 0.1.0 under ``tests/data/compat`` was written by running this
module against the source of the ``v0.1.0`` tag::

    git archive v0.1.0 src | tar -x -C /tmp/v010
    PYTHONPATH=/tmp/v010/src python packages/wise-pm/tests/_support/api_shapes.py \\
        > packages/wise-pm/tests/data/compat/api_0_1_0.json

It imports nothing but the standard library and ``wise``, so it runs against either tree.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import importlib
import inspect
import json
import pkgutil
import sys
import textwrap
from typing import Any

Shape = dict[str, Any]

REQUIRED = "<required>"


def _signature(obj: Any) -> Shape:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):  # pragma: no cover - builtins without a signature
        return {}
    params = [
        [
            p.name,
            p.kind.name,
            REQUIRED if p.default is p.empty else repr(p.default),
            None if p.annotation is p.empty else str(p.annotation),
        ]
        for p in sig.parameters.values()
    ]
    returns = None if sig.return_annotation is sig.empty else str(sig.return_annotation)
    return {"params": params, "returns": returns}


def _member(cls: type, name: str, raw: Any) -> Shape:
    if isinstance(raw, property):
        returns = _signature(raw.fget).get("returns") if raw.fget else None
        return {"kind": "property", "returns": returns, "settable": raw.fset is not None}
    if isinstance(raw, functools.cached_property):
        return {"kind": "property", "returns": _signature(raw.func).get("returns"), "settable": False}
    if isinstance(raw, (classmethod, staticmethod)):
        return {"kind": type(raw).__name__, **_signature(getattr(cls, name))}
    if inspect.isfunction(raw):
        return {"kind": "method", **_signature(raw)}
    return {"kind": "class attribute", "type": type(raw).__name__}


def _instance_attributes(cls: type) -> set[str]:
    """Names a class's own methods assign on ``self``."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    except (OSError, TypeError):  # pragma: no cover - classes without source
        return set()
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }


def _class(qualified: str, cls: type) -> dict[str, Shape]:
    fields = {f.name: str(f.type) for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else {}
    init = _signature(cls.__init__) if fields or "__init__" in vars(cls) else {}  # type: ignore[misc]
    out: dict[str, Shape] = {qualified: {"kind": "class", "bases": [b.__name__ for b in cls.__bases__], "init": init}}
    for klass in cls.__mro__:
        if klass.__module__.split(".")[0] != "wise":
            continue
        for name, raw in vars(klass).items():
            if name.startswith("__") or f"{qualified}.{name}" in out:
                continue
            if name in fields and not isinstance(raw, (property, functools.cached_property)) and not callable(raw):
                continue  # a dataclass field's default; recorded as the field below
            out[f"{qualified}.{name}"] = _member(cls, name, raw)
        for name in _instance_attributes(klass):
            out.setdefault(f"{qualified}.{name}", {"kind": "attribute"})
    for name, annotation in fields.items():
        out[f"{qualified}.{name}"] = {"kind": "field", "type": annotation}
    return out


def _public_names(module: Any, defined: set[str]) -> set[str]:
    declared = getattr(module, "__all__", None)
    if declared is not None:
        return set(declared)
    return {n for n in defined if not n.startswith("_")}


def shapes() -> dict[str, Shape]:
    """Every name of the importable ``wise`` package, keyed by its dotted path.

    A name another module of the package defines is recorded as an ``alias`` of its
    definition. ``public`` marks the names a module declares in ``__all__`` (or, for a
    module without ``__all__``, the names without a leading underscore that it defines
    itself), and the public members of those classes.
    """
    import wise

    out: dict[str, Shape] = {}
    modules = ["wise", *sorted(m.name for m in pkgutil.walk_packages(wise.__path__, "wise."))]
    for module_name in modules:
        module = importlib.import_module(module_name)
        names = {n: obj for n, obj in vars(module).items() if not n.startswith("__") and not inspect.ismodule(obj)}
        defined: set[str] = set()
        for name, obj in names.items():
            qualified = f"{module_name}.{name}"
            home = getattr(obj, "__module__", None)
            if (inspect.isclass(obj) or inspect.isfunction(obj)) and home != module_name:
                if home and home.split(".")[0] == "wise":
                    out[qualified] = {"kind": "alias", "of": f"{home}.{obj.__qualname__}"}
                continue
            if inspect.isclass(obj):
                defined.add(name)
                out.update(_class(qualified, obj))
            elif inspect.isfunction(obj):
                defined.add(name)
                out[qualified] = {"kind": "function", **_signature(obj)}
            elif not callable(obj):
                defined.add(name)
                out[qualified] = {"kind": "constant", "type": type(obj).__name__}
        for name in _public_names(module, defined):
            qualified = f"{module_name}.{name}"
            if qualified in out:
                out[qualified]["public"] = True
    for qualified, shape in list(out.items()):
        owner, _, member = qualified.rpartition(".")
        if out.get(owner, {}).get("kind") == "class" and out[owner].get("public") and not member.startswith("_"):
            shape["public"] = True
    return out


def canonical(table: dict[str, Shape], name: str) -> str | None:
    """The dotted path of the definition ``name`` refers to in ``table`` (following aliases)."""
    seen = set()
    while name in table and table[name].get("kind") == "alias" and name not in seen:
        seen.add(name)
        name = table[name]["of"]
    if name in table:
        return name
    owner, _, member = name.rpartition(".")
    resolved = canonical(table, owner) if owner and owner != name else None
    if resolved and resolved != owner and f"{resolved}.{member}" in table:
        return f"{resolved}.{member}"
    return None


def _parameters_compatible(old: list[list[Any]], new: list[list[Any]]) -> bool:
    """Every old call still binds: old parameters keep name, kind, position and default (a required
    one may gain a default); new parameters are optional. Annotations may widen."""
    new_by_name = {p[0]: (i, p) for i, p in enumerate(new)}
    for i, (name, kind, default, _annotation) in enumerate(old):
        if name not in new_by_name:
            return False
        j, (_, new_kind, new_default, _) = new_by_name[name]
        if new_kind != kind or (kind.startswith("POSITIONAL") and j != i):
            return False
        if default != REQUIRED and new_default != default:
            return False
    old_names = {p[0] for p in old}
    return all(p[2] != REQUIRED or p[1].startswith("VAR") for p in new if p[0] not in old_names)


def compatible(old: Shape, new: Shape | None) -> bool:
    """Whether code written against ``old`` works unchanged against ``new``.

    A plain attribute that became a property is compatible for reading (the reader's
    view of it); assigning to it is judged separately by the caller.
    """
    if new is None:
        return False
    kind, new_kind = old["kind"], new["kind"]
    if kind == "attribute":
        return new_kind in {"attribute", "property", "field"}
    if kind != new_kind:
        return False
    if kind in {"function", "method", "classmethod", "staticmethod"}:
        return old.get("returns") == new.get("returns") and _parameters_compatible(old["params"], new["params"])
    if kind == "class":
        if old["bases"] != new["bases"]:
            return False
        if not old["init"]:
            return True
        return bool(new["init"]) and _parameters_compatible(old["init"]["params"], new["init"]["params"])
    if kind == "property":
        return old["returns"] == new["returns"] and (new["settable"] or not old["settable"])
    if kind == "field":
        return old["type"] == new["type"]
    return old.get("type") == new.get("type")


if __name__ == "__main__":
    json.dump(shapes(), sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
