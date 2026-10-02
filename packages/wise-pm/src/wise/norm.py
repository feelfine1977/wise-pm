"""Process norm ``N = (C, Λ, lay)``, views, and applicability (paper Sec. IV-C).

A :class:`Norm` holds the constraint catalogue, the layer partition, the
stakeholder views, per-constraint applicability conditions, the scoring
mode, and optional recipes for derived case attributes. It is a versioned
artefact: its objects are frozen, their inputs are copied on construction
into read-only containers (a value read from a norm cannot be written into),
its structure is checked once, on construction, and it carries the schema
version of the document format it belongs to.

This module holds the model only. The document format (reading and writing
JSON documents, the fingerprint) is :mod:`wise.norm_document`;
:meth:`Norm.load`, :meth:`Norm.dump` and the other serialisation methods
delegate to it, and both shipped norms round-trip through it byte for byte.

Views
-----
A view ``p`` is a non-negative raw weight vector ``w_c^(p)`` over the shared
constraint set, given either directly (``constraint_weights``) or through the
paper's optional two-stage rule ``w_c = a_λ · b_c / Σ_{d∈C_λ} b_d`` with
``layer_weights`` and each constraint's within-layer ``weight``. Only relative
magnitudes matter; scoring renormalises over the applicable constraints of
each case.

Applicability
-------------
``C_app(σ)`` is stated per constraint as a selection rule over the case table
(short form or rule form, see :mod:`wise.selection`). Scoring evaluates it
under the missing-value policy of the norm's semantics profile
(:attr:`Norm.semantics`, :mod:`wise.semantics`): the 0.1.0 policy for schema
versions 1 and 2, three-valued for schema version 3.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd

from ._frozen import freeze
from ._json import jsonable as _jsonable
from ._version import __version__
from .constraints import (
    CONSTRAINT_TYPES,
    TYPE_ALIASES,
    Constraint,
)
from .errors import NormError
from .family import Inputs, family_of
from .recipes import Recipe, parse_recipe
from .selection import LEGACY, CaseTable, MissingPolicy, Selection, parse
from .semantics import Semantics, semantics_of

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from .log import EventLog

#: The schema version a norm built in Python carries by default.
SCHEMA_VERSION = 2
#: The schema versions a norm can carry: 1 and 2 in the 0.1.0 layout (:mod:`wise.norm_document.v2`), evaluated under
#: the 0.1.0 semantics, and 3 (:mod:`wise.norm_document.v3`), evaluated under the corrected semantics.
SCHEMA_VERSIONS = (1, 2, 3)
SCORING_MODES = ("layer_balanced", "flat")
DEFAULT_SCORING_MODE = "layer_balanced"


# ----------------------------------------------------------------------------- layers, views
@dataclass(frozen=True)
class Layer:
    """A business-facing group of constraints (``λ ∈ Λ``)."""

    id: str
    name: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not str(self.id):
            raise NormError("layer id must be a non-empty string")
        object.__setattr__(self, "id", str(self.id))


@dataclass(frozen=True)
class View:
    """A stakeholder perspective: raw weights over the shared constraint set.

    Give exactly one of ``constraint_weights`` (``{constraint_id: w}``) or
    ``layer_weights`` (``{layer_id: a_λ}``, combined with the constraints'
    within-layer weights).
    """

    name: str
    layer_weights: dict[str, float] | None = None
    constraint_weights: dict[str, float] | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not str(self.name):
            raise NormError("view name must be a non-empty string")
        if (self.layer_weights is None) == (self.constraint_weights is None):
            raise NormError(f"view {self.name!r}: give exactly one of layer_weights or constraint_weights")
        which = "layer_weights" if self.layer_weights is not None else "constraint_weights"
        raw = getattr(self, which)
        clean: dict[str, float] = {}
        for k, v in dict(raw).items():
            try:
                fv = float(v)
            except (TypeError, ValueError) as exc:
                raise NormError(f"view {self.name!r}: weight {k}={v!r} is not numeric") from exc
            if not np.isfinite(fv) or fv < 0:
                raise NormError(f"view {self.name!r}: weight {k}={v!r} must be finite and >= 0")
            clean[str(k)] = fv
        if not clean:
            raise NormError(f"view {self.name!r}: {which} is empty")
        object.__setattr__(self, which, freeze(clean))

    @property
    def weights(self) -> dict[str, float]:
        return self.layer_weights if self.layer_weights is not None else self.constraint_weights  # type: ignore[return-value]


# ----------------------------------------------------------------------------- constraints
@dataclass(frozen=True)
class NormConstraint:
    """One catalogue entry: a constraint, its layer, within-layer weight, and applicability."""

    id: str
    layer: str
    constraint: Constraint
    weight: float = 1.0
    applicability: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    if TYPE_CHECKING:  # pragma: no cover
        _selection: ClassVar[Selection]  # set per instance in __post_init__, not a dataclass field

    def __post_init__(self) -> None:
        if not str(self.id):
            raise NormError("constraint id must be a non-empty string")
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "layer", str(self.layer))
        if not isinstance(self.constraint, Constraint):
            raise NormError(f"constraint {self.id!r}: constraint must be a wise Constraint, got {type(self.constraint).__name__}")
        try:
            w = float(self.weight)
        except (TypeError, ValueError) as exc:
            raise NormError(f"constraint {self.id!r}: weight {self.weight!r} is not numeric") from exc
        if not np.isfinite(w) or w < 0:
            raise NormError(f"constraint {self.id!r}: within-layer weight must be finite and >= 0")
        object.__setattr__(self, "weight", w)
        selection = parse(self.applicability, what="applicability")
        object.__setattr__(self, "applicability", selection.document)
        object.__setattr__(self, "_selection", selection)

    def applies_to(self, cases: pd.DataFrame, log: EventLog | None = None, *, missing: MissingPolicy = LEGACY) -> pd.Series:
        """Boolean mask over ``cases`` (one row per case) for ``C_app``.

        ``missing`` is the missing-value policy (:mod:`wise.selection`); under
        ``"three_valued"`` a case whose scope is unknown is not in the mask
        (:meth:`scope` tells it apart from a case out of scope).
        """
        if not self.applicability:
            return pd.Series(True, index=cases.index)
        if missing == LEGACY:
            return self.scope(cases, log, missing=missing).astype(bool)
        return self.scope(cases, log, missing=missing).fillna(False).astype(bool)

    def scope(self, cases: pd.DataFrame, log: EventLog | None = None, *, missing: MissingPolicy = LEGACY) -> pd.Series:
        """Nullable boolean Series over ``cases``: in scope, out of scope, ``<NA>`` unknown under ``missing``."""
        if not self.applicability:
            return pd.Series(True, index=cases.index, dtype="boolean")
        for attr in self._selection.columns:
            if attr not in cases.columns:
                raise NormError(
                    f"constraint {self.id!r} is conditioned on case attribute {attr!r}, which is missing from the case table"
                )
        if self._selection.activities and log is None:
            raise NormError(f"constraint {self.id!r}: 'has'/'lacks' applicability needs the event log")
        return self._selection.evaluate(CaseTable(cases, log), missing=missing)

    def replace(self, **changes: Any) -> NormConstraint:
        """A copy with the given fields changed (constraints are immutable)."""
        return dataclasses.replace(self, **changes)

    @property
    def type(self) -> str:
        return self.constraint.type

    def to_dict(self) -> dict[str, Any]:
        """The constraint's entry in a norm document (:func:`wise.norm_document.write_constraint`)."""
        from . import norm_document

        return norm_document.write_constraint(self)


def _inputs(nc: NormConstraint) -> Inputs:
    """What a constraint reads from a log, as its family states it (applicability not included)."""
    needs: Inputs = family_of(nc.constraint).requires(nc.constraint)
    return needs


# ----------------------------------------------------------------------------- norm
@dataclass(frozen=True)
class Norm:
    """The versioned process norm: constraints, layers, views, scoring mode, recipes."""

    constraints: tuple[NormConstraint, ...]
    layers: tuple[Layer, ...]
    views: tuple[View, ...]
    name: str = "norm"
    version: str = "1"
    description: str = ""
    scoring_mode: str = DEFAULT_SCORING_MODE
    derived_attributes: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    #: The schema version of the document format; kept by ``replace`` and written back by ``to_dict``.
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    if TYPE_CHECKING:  # pragma: no cover
        _recipes: ClassVar[tuple[Recipe, ...]]  # set per instance in __post_init__, not a dataclass field

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "views", tuple(self.views))
        documents = tuple(self.derived_attributes)
        object.__setattr__(self, "_recipes", tuple(parse_recipe(r) for r in documents))
        object.__setattr__(self, "derived_attributes", tuple(freeze(dict(r)) for r in documents))
        object.__setattr__(self, "metadata", freeze(dict(self.metadata)))
        object.__setattr__(self, "version", str(self.version))
        _check_structure(self)

    # ------------------------------------------------------------ lookups
    @property
    def semantics(self) -> Semantics:
        """The semantics profile the norm is evaluated under, given by its schema version (:mod:`wise.semantics`)."""
        return semantics_of(self.schema_version)

    @property
    def recipes(self) -> tuple[Recipe, ...]:
        """The derived attributes as typed records (:mod:`wise.recipes`), in the order of ``derived_attributes``."""
        return self._recipes

    @property
    def constraint_ids(self) -> list[str]:
        return [c.id for c in self.constraints]

    @property
    def layer_ids(self) -> list[str]:
        return [layer.id for layer in self.layers]

    @property
    def view_names(self) -> list[str]:
        return [v.name for v in self.views]

    @cached_property
    def _constraints_by_id(self) -> dict[str, NormConstraint]:
        return {c.id: c for c in self.constraints}

    @cached_property
    def _views_by_name(self) -> dict[str, View]:
        return {v.name: v for v in self.views}

    def get_view(self, name: str) -> View:
        try:
            return self._views_by_name[name]
        except KeyError:
            raise NormError(f"unknown view {name!r}; available: {self.view_names}") from None

    def get_constraint(self, cid: str) -> NormConstraint:
        try:
            return self._constraints_by_id[cid]
        except KeyError:
            raise NormError(f"unknown constraint {cid!r}") from None

    @cached_property
    def _layer_of(self) -> pd.Series:
        return pd.Series({c.id: c.layer for c in self.constraints}, name="layer", dtype=object)

    @property
    def layer_of(self) -> pd.Series:
        """``lay: C → Λ`` as a Series indexed by constraint id (a copy on every call)."""
        return self._layer_of.copy()

    def constraints_in_layer(self, layer_id: str) -> list[NormConstraint]:
        return [c for c in self.constraints if c.layer == layer_id]

    def activities(self) -> list[str]:
        acts: set[str] = set()
        for c in self.constraints:
            acts.update(_inputs(c).activities)
        return sorted(acts)

    def attributes(self) -> list[str]:
        """Case attributes referenced by applicability rules and read by constraints (e.g. Metric)."""
        attrs: set[str] = set()
        for c in self.constraints:
            attrs.update(c._selection.columns)
            attrs.update(_inputs(c).case_attributes)
        return sorted(attrs)

    def applicability_attributes(self) -> list[str]:
        attrs: set[str] = set()
        for c in self.constraints:
            attrs.update(c._selection.columns)
        return sorted(attrs)

    # ---------------------------------------------------------- validation
    def validate(self) -> None:
        """Re-run the structural check of construction; raises :class:`NormError`.

        A norm is checked once, when it is built, and cannot change afterwards, so this only repeats that check.
        """
        _check_structure(self)

    def check(self, log: EventLog) -> list[str]:
        """Check the norm against a log. Returns human-readable issues (empty
        list = fine); nothing is raised.

        Reports activities that never occur in the log, case attributes that
        are missing (unless a recipe derives them), and event columns that a
        constraint reads but the log lacks (e.g. balance attributes). What a
        constraint reads is stated by its family (:mod:`wise.family`), what a
        recipe reads by its record (:mod:`wise.recipes`).
        """
        issues: list[str] = []
        labels = set(log.activity_labels)
        derived = {r.name for r in self.recipes}
        for r in self.recipes:
            for a in r.requires().activities:
                if a not in labels:
                    issues.append(f"derived attribute {r.name!r}: activity {a!r} never occurs in the log")
        case_columns = log.cases.columns
        event_columns = log.events.columns
        for c in self.constraints:
            needs = _inputs(c)
            for a in needs.activities + c._selection.activities:
                if a not in labels:
                    issues.append(f"constraint {c.id!r}: activity {a!r} never occurs in the log")
            for attr in c._selection.columns:
                if attr not in case_columns and attr not in derived:
                    issues.append(f"constraint {c.id!r}: applicability attribute {attr!r} is not a case attribute")
            for attr in needs.case_attributes:
                if attr not in case_columns and attr not in derived:
                    issues.append(f"constraint {c.id!r}: {c.type} attribute {attr!r} is neither a case attribute nor derived")
            for attr in needs.event_columns:
                if attr not in event_columns:
                    issues.append(f"constraint {c.id!r}: {c.type} attribute {attr!r} is not an event column")
        return issues

    # ------------------------------------------------------------- weights
    def _raw_weights(self, v: View) -> dict[str, float]:
        if v.constraint_weights is not None:
            return {c.id: float(v.constraint_weights.get(c.id, 0.0)) for c in self.constraints}
        out: dict[str, float] = {}
        for layer in self.layer_ids:
            members = self.constraints_in_layer(layer)
            total_b = sum(c.weight for c in members)
            a = float(v.layer_weights.get(layer, 0.0))  # type: ignore[union-attr]
            for c in members:
                out[c.id] = a * ((c.weight / total_b) if total_b > 0 else 0.0)
        return out

    def raw_weights(self, view: str | View) -> dict[str, float]:
        """Raw constraint weights ``w_c^(p)`` for a view (paper Sec. IV-C)."""
        v = view if isinstance(view, View) else self.get_view(view)
        return self._raw_weights(v)

    def weight_vector(self, view: str | View) -> pd.Series:
        """Raw weights as a Series in constraint order."""
        w = self.raw_weights(view)
        return pd.Series(
            [w[c] for c in self.constraint_ids], index=self.constraint_ids, name=view if isinstance(view, str) else view.name
        )

    def weight_table(self) -> pd.DataFrame:
        """Constraints × views table of raw weights plus layer, for review."""
        tab = pd.DataFrame({v.name: self.raw_weights(v) for v in self.views}).reindex(self.constraint_ids)
        tab.insert(0, "layer", self.layer_of)
        tab.index.name = "constraint"
        return tab

    def layer_weight_table(self) -> pd.DataFrame:
        """Layers × views table of ``a_λ^(p)`` (sum of raw weights per layer)."""
        tab = self.weight_table()
        return tab.groupby("layer")[self.view_names].sum().reindex(self.layer_ids)

    # ---------------------------------------------------------- editing
    def replace(self, **changes: Any) -> Norm:
        """A copy of the norm with the given top-level fields changed; it keeps the schema version."""
        if "schema_version" in changes and changes["schema_version"] != self.schema_version:
            raise NormError(
                f"replace keeps the schema version ({self.schema_version}); "
                f"a norm changes its schema version only by migration, not to {changes['schema_version']!r}"
            )
        return dataclasses.replace(self, **changes)

    def replace_constraint(self, cid: str, **changes: Any) -> Norm:
        """A copy with one constraint's fields changed (e.g. ``applicability={}``)."""
        self.get_constraint(cid)
        return self.replace(constraints=tuple(c.replace(**changes) if c.id == cid else c for c in self.constraints))

    def map_constraints(self, fn: Any) -> Norm:
        """A copy with every constraint replaced by ``fn(constraint)``."""
        return self.replace(constraints=tuple(fn(c) for c in self.constraints))

    # ----------------------------------------------------------- serialise
    # The document format lives in :mod:`wise.norm_document`; these methods delegate to it.
    def to_dict(self) -> dict[str, Any]:
        """The norm document (plain containers, independent of the norm)."""
        from . import norm_document

        return norm_document.write(self)

    def dumps(self, indent: int = 2) -> str:
        from . import norm_document

        return norm_document.dumps(self, indent)

    def dump(self, path: str | Path, indent: int = 2) -> Path:
        from . import norm_document

        return norm_document.dump(self, path, indent)

    @classmethod
    def loads(cls, text: str) -> Norm:
        from . import norm_document

        return norm_document.loads(text)

    @classmethod
    def load(cls, path: str | Path) -> Norm:
        from . import norm_document

        return norm_document.load(path)

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        """JSON text; also written to ``path`` if given (alias of dumps/dump)."""
        from . import norm_document

        if path is not None:
            norm_document.dump(self, path, indent)
        return norm_document.dumps(self, indent)

    @classmethod
    def from_json(cls, source: str | Path) -> Norm:
        """Load from a file path or a JSON string (alias of load/loads)."""
        s = str(source)
        if s.lstrip().startswith("{"):
            return cls.loads(s)
        return cls.load(source)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Norm:
        """Build from a norm document, the dictionary form of :meth:`to_dict`.

        The document must give its ``schema_version``. Unknown keys and
        malformed structures raise :class:`NormError` so that a typo cannot
        silently change the meaning of a norm.
        """
        from . import norm_document

        return norm_document.read(d)

    def fingerprint(self) -> str:
        """SHA-256 of the norm's document without metadata and descriptions (:func:`wise.norm_document.fingerprint`)."""
        from . import norm_document

        return norm_document.fingerprint(self)

    def describe(self) -> pd.DataFrame:
        """One row per constraint: layer, type, params, weight, applicability."""
        rows = []
        for c in self.constraints:
            rows.append(
                {
                    "id": c.id,
                    "layer": c.layer,
                    "type": c.constraint.type,
                    "params": _jsonable(c.constraint.params()),
                    "weight": c.weight,
                    "applicability": c.applicability or "always",
                    "description": c.description,
                }
            )
        return pd.DataFrame(rows).set_index("id")

    def __repr__(self) -> str:
        return (
            f"Norm({self.name!r} v{self.version}: {len(self.constraints)} constraints, "
            f"{len(self.layers)} layers, views={self.view_names}, scoring_mode={self.scoring_mode!r})"
        )


def _check_structure(norm: Norm) -> None:
    """The structural check of a norm, run once on construction (and by :meth:`Norm.validate`).

    Each layer, view and constraint checks its own fields when it is built; this checks what only the norm as a
    whole can: unique ids and names, references between constraints, layers and views, a non-zero weight in every
    view, the scoring mode, the schema version, the names of the derived attributes and, from schema version 3 on,
    that no recipe is an ``eval`` (decision 0001, section 3).
    """
    ids = norm.constraint_ids
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise NormError(f"duplicate constraint ids: {dup}")
    if not norm.constraints:
        raise NormError("a norm needs at least one constraint")
    layer_ids = norm.layer_ids
    if len(set(layer_ids)) != len(layer_ids):
        raise NormError("duplicate layer ids")
    for c in norm.constraints:
        if c.layer not in layer_ids:
            raise NormError(f"constraint {c.id!r} refers to unknown layer {c.layer!r}; layers: {layer_ids}")
    names = norm.view_names
    if len(set(names)) != len(names):
        raise NormError("duplicate view names")
    if not norm.views:
        raise NormError("a norm needs at least one view")
    for v in norm.views:
        if v.layer_weights is not None:
            unknown = set(v.layer_weights) - set(layer_ids)
            if unknown:
                raise NormError(f"view {v.name!r}: unknown layers {sorted(unknown)}")
        else:
            unknown = set(v.constraint_weights or {}) - set(ids)
            if unknown:
                raise NormError(f"view {v.name!r}: unknown constraints {sorted(unknown)}")
        if not any(w > 0 for w in norm.raw_weights(v).values()):
            raise NormError(f"view {v.name!r}: all constraint weights are zero")
    if norm.scoring_mode not in SCORING_MODES:
        raise NormError(f"scoring_mode must be one of {SCORING_MODES}, got {norm.scoring_mode!r}")
    if type(norm.schema_version) is not int or norm.schema_version not in SCHEMA_VERSIONS:
        raise NormError(f"schema_version must be one of {list(SCHEMA_VERSIONS)}, got {norm.schema_version!r}")
    seen: set[str] = set()
    for r in norm.recipes:
        if r.name in seen:
            raise NormError(f"duplicate derived attribute {r.name!r}")
        seen.add(r.name)
    if norm.schema_version >= 3:
        for document in norm.derived_attributes:
            if document.get("kind") == "eval":
                raise NormError(
                    f"derived attribute {document.get('name')!r}: schema version {norm.schema_version} has no eval recipe; "
                    "name the case-table column with a column recipe instead"
                )


def _rule_attributes(rule: Mapping[str, Any]) -> set[str]:
    """Case attributes an applicability rule reads (kept for 0.1.0 callers that import it from here; removed in 1.0)."""
    warnings.warn(
        "wise.norm._rule_attributes is deprecated and will be removed in 1.0; use wise.selection.parse(rule).columns",
        DeprecationWarning,
        stacklevel=2,
    )
    return set(parse(rule, what="applicability").columns)


__all__ = [
    "CONSTRAINT_TYPES",
    "SCHEMA_VERSION",
    "SCHEMA_VERSIONS",
    "SCORING_MODES",
    "TYPE_ALIASES",
    "Layer",
    "Norm",
    "NormConstraint",
    "View",
    "__version__",
]
