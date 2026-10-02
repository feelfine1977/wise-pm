"""Migration of a norm document from schema version 1 or 2 to schema version 3 (decision 0001, sections 3 and 4).

:func:`migrate` is the only operation that changes the schema version of a
norm. It rewrites the document, not the model: every part of the document it
does not rewrite is copied as it was, so a number keeps its exact value and
its JSON form (``10`` stays ``10``, it does not become ``10.0``). The
rewrites are those decision 0001 prescribes:

* ``missing_values``: the missing-value policy of applicability is added as
  the named field ``"missing_values": "three_valued"``;
* ``eval_column``: an ``eval`` recipe over a bare case-table column becomes a
  ``column`` recipe; any other ``eval`` stops the migration with the recipe
  named, since schema version 3 has no ``eval`` recipe;
* ``calibration``: a ``quantile_scale`` recipe without a calibration record
  gets an unfitted one at its quantile;
* ``lag_bound``: a ``lag`` without ``delta`` gets ``"delta": 0`` (and
  ``"width": 0`` where it has none), the bound 0.1.0 read it with, written
  explicitly and recommended for review;
* ``lag_unbounded``: a ``lag`` with ``"delta": null`` is written
  ``"unbounded": true``.

Each rewrite is one :class:`Rewrite` of the :class:`MigrationReport`, in
document order; its ``corrections`` name the ledger entries whose correction
can change the rows a migrated norm scores. The report is plain JSON
(:meth:`MigrationReport.to_json`) and the same document always gives the
same report.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .._frozen import freeze, thaw
from ..errors import NormError
from ..norm import Norm
from ..recipes import ColumnRecipe, QuantileScaleRecipe, parse_recipe
from ..selection import THREE_VALUED
from . import adapter, schema_version_of, v2, v3

#: The schema version :func:`migrate` writes.
TARGET_VERSION = 3
#: The rules of the migration; a report lists its rewrites in document order, not by rule.
RULES: tuple[str, ...] = ("missing_values", "eval_column", "calibration", "lag_bound", "lag_unbounded")


@dataclass(frozen=True)
class Rewrite:
    """One rewrite of a migration: at ``path`` (a JSON pointer into the document), the keys ``before`` became
    the keys ``after``; a key that was absent is not in ``before``."""

    rule: str
    path: str
    #: The recipe name, constraint id or norm name the rewrite belongs to.
    subject: str
    before: Mapping[str, Any]
    after: Mapping[str, Any]
    note: str
    #: The ledger entries whose correction can change the scored rows of the rewritten part.
    corrections: tuple[str, ...] = ()
    recommendation: str | None = None

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise NormError(f"unknown migration rule {self.rule!r}; known {list(RULES)}")
        object.__setattr__(self, "before", freeze(dict(self.before)))
        object.__setattr__(self, "after", freeze(dict(self.after)))
        object.__setattr__(self, "corrections", tuple(self.corrections))

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "path": self.path,
            "subject": self.subject,
            "before": thaw(self.before),
            "after": thaw(self.after),
            "note": self.note,
            "corrections": list(self.corrections),
            "recommendation": self.recommendation,
        }


@dataclass(frozen=True)
class MigrationReport:
    """Every rewrite of one migration, in document order."""

    norm: str
    source_version: int
    target_version: int
    rewrites: tuple[Rewrite, ...] = field(default=())

    @property
    def corrections(self) -> tuple[str, ...]:
        """The ledger entries named by the rewrites, each once, in order of first mention."""
        return tuple(dict.fromkeys(c for r in self.rewrites for c in r.corrections))

    def to_dict(self) -> dict[str, Any]:
        return {
            "norm": self.norm,
            "source_version": self.source_version,
            "target_version": self.target_version,
            "rewrites": [r.to_dict() for r in self.rewrites],
        }

    def to_json(self, indent: int | None = 2) -> str:
        """The report as JSON text with sorted keys: the same report always gives the same text."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> MigrationReport:
        """The report a :meth:`to_dict` document describes."""
        return cls(
            norm=str(document["norm"]),
            source_version=int(document["source_version"]),
            target_version=int(document["target_version"]),
            rewrites=tuple(
                Rewrite(
                    rule=r["rule"],
                    path=r["path"],
                    subject=r["subject"],
                    before=r["before"],
                    after=r["after"],
                    note=r["note"],
                    corrections=tuple(r["corrections"]),
                    recommendation=r["recommendation"],
                )
                for r in document["rewrites"]
            ),
        )


@dataclass(frozen=True)
class Migration:
    """The result of :func:`migrate`: the migrated document (read-only), the norm it describes and the report."""

    document: Mapping[str, Any]
    norm: Norm
    report: MigrationReport

    def dumps(self, indent: int = 2) -> str:
        """The migrated document as JSON text, laid out as :func:`wise.norm_document.dumps` lays out a norm."""
        return json.dumps(self.document, indent=indent, ensure_ascii=False)


def migrate(source: Norm | Mapping[str, Any]) -> Migration:
    """Migrate a norm, or its document, from schema version 1 or 2 to schema version 3.

    A norm is migrated through its document. A document without a supported
    ``schema_version`` is refused with the supported versions listed, a
    document that is not a valid schema version 1 or 2 norm with the reason,
    one that is at schema version 3 already, and one with an ``eval`` recipe
    that is not a bare case-table column, with the recipe named. Reading the
    source does not emit the :class:`~wise.semantics.LegacySemanticsWarning`.
    """
    document = adapter(source.schema_version).write(source) if isinstance(source, Norm) else source
    if not isinstance(document, Mapping):
        raise NormError(f"a norm must be a mapping, got {type(document).__name__}")
    version = schema_version_of(document)
    name = str(document.get("name", "norm"))
    if version not in v2.VERSIONS:
        raise NormError(
            f"norm {name!r} has schema_version {version} already; migrate() takes schema version "
            f"{' or '.join(map(str, v2.VERSIONS))} to {TARGET_VERSION}"
        )
    for recipe in _recipes(document):
        if recipe.get("kind") == "eval":
            _eval_column(recipe, name)
    legacy = _read(document, version)
    out: dict[str, Any] = copy.deepcopy(thaw(dict(document)))
    rewrites = [*_top(out, name), *_derived_attributes(out, legacy), *_constraints(out)]
    migrated = _read(out, TARGET_VERSION)
    return Migration(freeze(out), migrated, MigrationReport(name, version, TARGET_VERSION, tuple(rewrites)))


# ----------------------------------------------------------------------------- rules
def _read(document: Mapping[str, Any], version: int) -> Norm:
    """The norm of a document, read by the adapter of its version (which announces no semantics profile)."""
    try:
        return adapter(version).read(document, version)
    except NormError:
        raise
    except (TypeError, KeyError, AttributeError, ValueError) as exc:
        raise NormError(f"malformed norm: {exc}") from exc


def _recipes(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [r for r in document.get("derived_attributes", []) or [] if isinstance(r, Mapping)]


def _eval_column(recipe: Mapping[str, Any], norm: str) -> str:
    try:
        record = parse_recipe(recipe)
    except NormError as exc:
        raise NormError(
            f"cannot migrate norm {norm!r} to schema version {TARGET_VERSION}: {exc}; schema version "
            f"{TARGET_VERSION} has no eval recipe; rewrite {recipe.get('name')!r} as a typed recipe"
        ) from None
    if not isinstance(record, ColumnRecipe):  # pragma: no cover - an eval recipe is always read as a column recipe
        raise NormError(f"derived attribute {recipe.get('name')!r} (eval) is not read as a column recipe")
    return record.attribute


def _top(out: dict[str, Any], name: str) -> list[Rewrite]:
    after = "scoring_mode" if "scoring_mode" in out else "schema_version"
    items = list(out.items())
    out.clear()
    for key, value in items:
        out[key] = TARGET_VERSION if key == "schema_version" else value
        if key == after:
            out["missing_values"] = THREE_VALUED
    return [
        Rewrite(
            "missing_values",
            "",
            name,
            {},
            {"missing_values": THREE_VALUED},
            "the missing-value policy of applicability is named: a case whose attribute is missing is unknown under "
            "every operator and left out of the denominator",
            ("C4", "C25"),
        )
    ]


def _derived_attributes(out: dict[str, Any], legacy: Norm) -> list[Rewrite]:
    rewrites: list[Rewrite] = []
    recipes = out.get("derived_attributes", []) or []
    for i, (recipe, record) in enumerate(zip(recipes, legacy.recipes, strict=True)):
        path = f"/derived_attributes/{i}"
        if recipe.get("kind") == "eval":
            column = _eval_column(recipe, legacy.name)
            before = {"kind": "eval", "expr": recipe["expr"]}
            recipes[i] = {("attribute" if k == "expr" else k): v for k, v in recipe.items()}
            recipes[i].update(kind="column", attribute=column)
            rewrites.append(
                Rewrite(
                    "eval_column",
                    path,
                    recipe["name"],
                    before,
                    {"kind": "column", "attribute": column},
                    f"an eval recipe over the bare case-table column {column!r} becomes a column recipe",
                    ("C16",),
                )
            )
        elif isinstance(record, QuantileScaleRecipe) and recipe.get("calibration") is None:
            before = {"calibration": None} if "calibration" in recipe else {}
            record_ = {"status": "unfitted", "quantile": record.q}
            recipe["calibration"] = record_
            rewrites.append(
                Rewrite(
                    "calibration",
                    path,
                    recipe["name"],
                    before,
                    {"calibration": record_},
                    "the quantile_scale recipe carries an unfitted calibration record",
                    ("C15",),
                    "fit the record on the reference log (wise.fit_calibration) before scoring: scoring refuses an "
                    "unfitted record, and 0.1.0 divided by the quantile of whatever cases were scored",
                )
            )
    return rewrites


def _constraints(out: dict[str, Any]) -> list[Rewrite]:
    rewrites: list[Rewrite] = []
    for i, entry in enumerate(out.get("constraints", []) or []):
        if not v3.is_lag(entry):
            continue
        params: dict[str, Any] = entry.get("params") or {}
        path = f"/constraints/{i}/params"
        if "delta" not in params:
            after = {"delta": 0, **({} if "width" in params else {"width": 0})}
            params.update(after)
            entry["params"] = params
            rewrites.append(
                Rewrite(
                    "lag_bound",
                    path,
                    str(entry["id"]),
                    {},
                    after,
                    "the lag had no bound; the bound 0.1.0 read it with, delta 0 and width 0, is written explicitly",
                    ("C2",),
                    "review the bound: with delta 0 and width 0 any delay between the activities is a full violation; "
                    "state the intended delta and width, or 'unbounded': true",
                )
            )
        elif params["delta"] is None:
            entry["params"] = {("unbounded" if k == "delta" else k): (True if k == "delta" else v) for k, v in params.items()}
            rewrites.append(
                Rewrite(
                    "lag_unbounded",
                    path,
                    str(entry["id"]),
                    {"delta": None},
                    {"unbounded": True},
                    "the lag without a time bound ('delta': null) is written 'unbounded': true",
                )
            )
    return rewrites


__all__ = ["RULES", "TARGET_VERSION", "Migration", "MigrationReport", "Rewrite", "migrate"]
