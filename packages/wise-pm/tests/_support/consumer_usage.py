"""What a consumer's Python sources use of ``wise``: the names they import and the members they touch.

:func:`scan` reads every ``*.py`` file under a checkout (skipping virtual environments,
build output and vendored trees) without importing it, and reports

* every dotted name rooted at the ``wise`` package that the sources import or read,
  through ``import wise``, ``import wise.x as y``, ``from wise.x import y`` or
  ``importlib.import_module("wise")`` (``wise.Norm.from_dict`` is reported whole);
* every attribute of an object, from a given set of member names, with how it is used:
  ``read``; ``assign`` (``obj.name = ...``); ``write`` (a write into the object it
  returns: ``obj.name[k] = ...``, ``obj.name.loc[...] = ...``, ``obj.name[k] += ...``);
  ``keyword`` (a private name passed as ``_name=...`` to a call, such as a private
  dataclass field of a constructor). Attributes of ``self`` and ``cls``
  are the consumer's own and are not reported.

The record under ``tests/data/compat/consumer_usage.json`` is the scan of the workbench
checkout the compatibility module was checked against; refresh it with::

    uv run --frozen --no-sync python packages/wise-pm/tests/_support/consumer_usage.py /path/to/wise-workbench
"""

from __future__ import annotations

import ast
import json
import os
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

SKIPPED_DIRECTORIES = frozenset(
    {".git", ".venv", "venv", "node_modules", "build", "dist", "vendor", "site-packages", "__pycache__", ".mypy_cache"}
)
CONTEXTS = ("read", "assign", "write", "keyword")
_INDEXERS = frozenset({"loc", "iloc", "at", "iat"})


def python_sources(root: Path) -> Iterator[Path]:
    """Every ``*.py`` file under ``root`` outside the skipped directories, in path order."""
    found = []
    for directory, subdirectories, files in os.walk(root):
        subdirectories[:] = [d for d in subdirectories if d not in SKIPPED_DIRECTORIES]
        found += [Path(directory) / f for f in files if f.endswith(".py")]
    yield from sorted(found)


def _wise_aliases(tree: ast.Module) -> dict[str, str]:
    """Local names bound to something in the ``wise`` package, with the dotted name they stand for."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "wise":
                    aliases[a.asname or "wise"] = a.name if a.asname else "wise"
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").split(".")[0] == "wise":
            for a in node.names:
                aliases[a.asname or a.name] = f"{node.module}.{a.name}"
        elif (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute | ast.Name)
            and (node.value.func.attr if isinstance(node.value.func, ast.Attribute) else node.value.func.id) == "import_module"
            and node.value.args
            and isinstance(node.value.args[0], ast.Constant)
            and str(node.value.args[0].value).split(".")[0] == "wise"
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    aliases[target.id] = str(node.value.args[0].value)
    return aliases


def _chain(node: ast.expr) -> tuple[ast.expr, list[str]]:
    """The root of an attribute chain and the attribute names from the root outwards."""
    names: list[str] = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    return node, names[::-1]


def _written_attribute(target: ast.expr) -> ast.Attribute | None:
    """For ``obj.name[k] = ...`` or ``obj.name.loc[k] = ...``, the ``obj.name`` node."""
    if not isinstance(target, ast.Subscript):
        return None
    node = target.value
    while isinstance(node, ast.Subscript) or (isinstance(node, ast.Attribute) and node.attr in _INDEXERS):
        node = node.value
    return node if isinstance(node, ast.Attribute) else None


class Usage:
    """Names and member uses found by :func:`scan`, each with the files they occur in."""

    def __init__(self) -> None:
        self.names: defaultdict[str, set[str]] = defaultdict(set)
        self.members: defaultdict[str, defaultdict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    def to_dict(self) -> dict[str, object]:
        return {
            "names": {n: sorted(files) for n, files in sorted(self.names.items())},
            "members": {m: {c: sorted(files) for c, files in sorted(uses.items())} for m, uses in sorted(self.members.items())},
        }

    @classmethod
    def from_dict(cls, record: dict[str, dict[str, object]]) -> Usage:
        usage = cls()
        for name, files in record["names"].items():
            usage.names[name] = set(files)  # type: ignore[call-overload]
        for member, uses in record["members"].items():
            for context, files in uses.items():  # type: ignore[attr-defined]
                usage.members[member][context] = set(files)
        return usage


def _scan_file(usage: Usage, tree: ast.Module, where: str, members: frozenset[str]) -> None:
    aliases = _wise_aliases(tree)
    for name in aliases.values():
        if name != "wise":
            usage.names[name].add(where)
    seen: set[int] = set()
    for node in ast.walk(tree):
        targets: Iterable[ast.expr] = ()
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AugAssign | ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Attribute) and target.attr in members:
                root, _ = _chain(target)
                if not (isinstance(root, ast.Name) and root.id in {"self", "cls"}):
                    usage.members[target.attr]["assign"].add(where)
                    seen.add(id(target))
            written = _written_attribute(target)
            if written is not None and written.attr in members:
                root, _ = _chain(written)
                if not (isinstance(root, ast.Name) and root.id in {"self", "cls"}):
                    usage.members[written.attr]["write"].add(where)
        if isinstance(node, ast.Call):
            for kw in node.keywords:  # a private constructor field passed by name; public keywords are parameters
                if kw.arg in members and kw.arg.startswith("_"):
                    usage.members[kw.arg]["keyword"].add(where)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or id(node) in seen:
            continue
        root, names = _chain(node)
        if isinstance(root, ast.Name) and root.id in aliases:
            if node is _outermost(tree, node):
                usage.names[".".join([aliases[root.id], *names])].add(where)
            continue
        if isinstance(root, ast.Name) and root.id in {"self", "cls"} and len(names) == 1:
            continue
        if node.attr in members and isinstance(node.ctx, ast.Load):
            usage.members[node.attr]["read"].add(where)


def _outermost(tree: ast.Module, node: ast.Attribute) -> ast.Attribute:
    parent = _parents(tree).get(id(node))
    while isinstance(parent, ast.Attribute):
        node = parent
        parent = _parents(tree).get(id(node))
    return node


_PARENTS: dict[int, dict[int, ast.AST]] = {}


def _parents(tree: ast.Module) -> dict[int, ast.AST]:
    if id(tree) not in _PARENTS:
        _PARENTS.clear()
        _PARENTS[id(tree)] = {id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    return _PARENTS[id(tree)]


def scan(root: Path, members: Iterable[str]) -> Usage:
    """Scan every Python source under ``root``; ``members`` are the attribute names to report."""
    usage = Usage()
    wanted = frozenset(members)
    for path in python_sources(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        _scan_file(usage, tree, path.relative_to(root).as_posix(), wanted)
    return usage


DATA = Path(__file__).resolve().parents[1] / "data" / "compat"
BASELINE = DATA / "api_0_1_0.json"
RECORD = DATA / "consumer_usage.json"


def member_names(baseline: dict[str, dict[str, object]]) -> set[str]:
    """Names of the members of the classes in a shape table (see ``api_shapes``)."""
    classes = {name for name, shape in baseline.items() if shape.get("kind") == "class"}
    return {name.rpartition(".")[2] for name in baseline if name.rpartition(".")[0] in classes}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    usage = scan(Path(argv[1]), member_names(baseline))
    RECORD.write_text(json.dumps(usage.to_dict(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
