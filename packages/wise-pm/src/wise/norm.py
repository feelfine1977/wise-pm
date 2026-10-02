"""Process norm ``N = (C, Λ, lay)``, views, and applicability (paper Sec. IV-C).

A :class:`Norm` holds the constraint catalogue, the layer partition, the
stakeholder views, per-constraint applicability conditions, the scoring
mode, and optional recipes for derived case attributes. It is a versioned
artefact: its objects are frozen, their inputs are copied on construction,
the norm is validated on construction and again before every scoring run,
and :meth:`Norm.dump` / :meth:`Norm.load` round-trip it losslessly so it can
be reviewed, diffed, and reused.

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
under the 0.1.0 missing-value policy.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd

from ._json import jsonable as _jsonable
from ._version import __version__
from .constraints import (
    CONSTRAINT_TYPES,
    TYPE_ALIASES,
    Constraint,
    constraint_from_dict,
)
from .errors import NormError
from .family import Inputs, family_of
from .recipes import Recipe, parse_recipe
from .selection import LEGACY, CaseTable, Selection, parse

if TYPE_CHECKING:  # pragma: no cover
    from .log import EventLog

SCHEMA_VERSION = 2
SCORING_MODES = ("layer_balanced", "flat")
DEFAULT_SCORING_MODE = "layer_balanced"


def _json_default(obj: Any) -> Any:
    out = _jsonable(obj)
    if out is obj:
        raise NormError(f"norm metadata holds a value of type {type(obj).__name__} that cannot be serialised to JSON")
    return out


def _parse_json(text: str, source: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise NormError(f"{source} is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})") from exc


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
        object.__setattr__(self, which, clean)

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

    def applies_to(self, cases: pd.DataFrame, log: EventLog | None = None) -> pd.Series:
        """Boolean mask over ``cases`` (one row per case) for ``C_app``."""
        if not self.applicability:
            return pd.Series(True, index=cases.index)
        for attr in self._selection.columns:
            if attr not in cases.columns:
                raise NormError(
                    f"constraint {self.id!r} is conditioned on case attribute {attr!r}, which is missing from the case table"
                )
        if self._selection.activities and log is None:
            raise NormError(f"constraint {self.id!r}: 'has'/'lacks' applicability needs the event log")
        return self._selection.evaluate(CaseTable(cases, log), missing=LEGACY).astype(bool)

    def replace(self, **changes: Any) -> NormConstraint:
        """A copy with the given fields changed (constraints are immutable)."""
        return dataclasses.replace(self, **changes)

    @property
    def type(self) -> str:
        return self.constraint.type

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "layer": self.layer,
            "type": self.constraint.type,
            "params": _jsonable(self.constraint.params()),
            "weight": self.weight,
            "applicability": copy.deepcopy(self.applicability),
            "description": self.description,
        }


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

    if TYPE_CHECKING:  # pragma: no cover
        _recipes: ClassVar[tuple[Recipe, ...]]  # set per instance in __post_init__, not a dataclass field

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "views", tuple(self.views))
        documents = tuple(self.derived_attributes)
        object.__setattr__(self, "_recipes", tuple(parse_recipe(r) for r in documents))
        object.__setattr__(self, "derived_attributes", tuple(copy.deepcopy(dict(r)) for r in documents))
        object.__setattr__(self, "metadata", copy.deepcopy(dict(self.metadata)))
        object.__setattr__(self, "version", str(self.version))
        self.validate()

    # ------------------------------------------------------------ lookups
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
    def layer_of(self) -> pd.Series:
        """``lay: C → Λ`` as a Series indexed by constraint id."""
        return pd.Series({c.id: c.layer for c in self.constraints}, name="layer", dtype=object)

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
        """Structural validation; raises :class:`NormError`."""
        ids = self.constraint_ids
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise NormError(f"duplicate constraint ids: {dup}")
        if not self.constraints:
            raise NormError("a norm needs at least one constraint")
        layer_ids = self.layer_ids
        if len(set(layer_ids)) != len(layer_ids):
            raise NormError("duplicate layer ids")
        for c in self.constraints:
            if c.layer not in layer_ids:
                raise NormError(f"constraint {c.id!r} refers to unknown layer {c.layer!r}; layers: {layer_ids}")
        names = self.view_names
        if len(set(names)) != len(names):
            raise NormError("duplicate view names")
        if not self.views:
            raise NormError("a norm needs at least one view")
        for v in self.views:
            for k, w in v.weights.items():
                if not isinstance(w, int | float) or not np.isfinite(w) or w < 0:
                    raise NormError(f"view {v.name!r}: weight {k}={w!r} must be finite and >= 0")
            if v.layer_weights is not None:
                unknown = set(v.layer_weights) - set(layer_ids)
                if unknown:
                    raise NormError(f"view {v.name!r}: unknown layers {sorted(unknown)}")
            else:
                unknown = set(v.constraint_weights or {}) - set(ids)
                if unknown:
                    raise NormError(f"view {v.name!r}: unknown constraints {sorted(unknown)}")
            if not any(w > 0 for w in self._raw_weights(v).values()):
                raise NormError(f"view {v.name!r}: all constraint weights are zero")
        for c in self.constraints:
            if not np.isfinite(c.weight) or c.weight < 0:
                raise NormError(f"constraint {c.id!r}: within-layer weight must be finite and >= 0")
            parse(c.applicability, what="applicability")
        if self.scoring_mode not in SCORING_MODES:
            raise NormError(f"scoring_mode must be one of {SCORING_MODES}, got {self.scoring_mode!r}")
        seen: set[str] = set()
        for r in self.recipes:
            if r.name in seen:
                raise NormError(f"duplicate derived attribute {r.name!r}")
            seen.add(r.name)

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
        """A copy of the norm with the given top-level fields changed."""
        return dataclasses.replace(self, **changes)

    def replace_constraint(self, cid: str, **changes: Any) -> Norm:
        """A copy with one constraint's fields changed (e.g. ``applicability={}``)."""
        self.get_constraint(cid)
        return self.replace(constraints=tuple(c.replace(**changes) if c.id == cid else c for c in self.constraints))

    def map_constraints(self, fn: Any) -> Norm:
        """A copy with every constraint replaced by ``fn(constraint)``."""
        return self.replace(constraints=tuple(fn(c) for c in self.constraints))

    # ----------------------------------------------------------- serialise
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "scoring_mode": self.scoring_mode,
            "metadata": _jsonable(self.metadata),
            "layers": [{"id": layer.id, "name": layer.name, "description": layer.description} for layer in self.layers],
            "views": [
                {
                    "name": v.name,
                    "description": v.description,
                    **({"layer_weights": dict(v.layer_weights)} if v.layer_weights is not None else {}),
                    **({"constraint_weights": dict(v.constraint_weights)} if v.constraint_weights is not None else {}),
                }
                for v in self.views
            ],
            "derived_attributes": [copy.deepcopy(r) for r in self.derived_attributes],
            "constraints": [c.to_dict() for c in self.constraints],
        }

    def dumps(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=_json_default)

    def dump(self, path: str | Path, indent: int = 2) -> Path:
        p = Path(path)
        p.write_text(self.dumps(indent) + "\n", encoding="utf-8")
        return p

    @classmethod
    def loads(cls, text: str) -> Norm:
        return cls.from_dict(_parse_json(text, "norm text"))

    @classmethod
    def load(cls, path: str | Path) -> Norm:
        return cls.from_dict(_parse_json(Path(path).read_text(encoding="utf-8"), str(path)))

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        """JSON text; also written to ``path`` if given (alias of dumps/dump)."""
        text = self.dumps(indent)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, source: str | Path) -> Norm:
        """Load from a file path or a JSON string (alias of load/loads)."""
        s = str(source)
        if s.lstrip().startswith("{"):
            return cls.loads(s)
        return cls.load(source)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Norm:
        """Build from the dictionary form of :meth:`to_dict`.

        Unknown keys and malformed structures raise :class:`NormError` so
        that a typo cannot silently change the meaning of a norm.
        """
        try:
            return cls._from_dict(d)
        except NormError:
            raise
        except (TypeError, KeyError, AttributeError, ValueError) as exc:
            raise NormError(f"malformed norm: {exc}") from exc

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Norm:
        if not isinstance(d, Mapping):
            raise NormError(f"a norm must be a mapping, got {type(d).__name__}")
        version = int(d.get("schema_version", SCHEMA_VERSION))
        if version > SCHEMA_VERSION:
            raise NormError(f"norm schema_version {version} is newer than supported ({SCHEMA_VERSION})")
        unknown = set(d) - _TOP_KEYS
        if unknown:
            raise NormError(f"unknown top-level keys in norm: {sorted(unknown)}")

        layers_raw = d.get("layers", [])
        if isinstance(layers_raw, Mapping):
            layers_raw = [{"id": k, **(v if isinstance(v, Mapping) else {"name": str(v)})} for k, v in layers_raw.items()]
        layers = [
            Layer(id=str(layer["id"]), name=str(layer.get("name", "")), description=str(layer.get("description", "")))
            for layer in layers_raw
        ]

        views_raw = d.get("views", [])
        if isinstance(views_raw, Mapping):
            views_raw = [{"name": k, **v} for k, v in views_raw.items()]
        views = []
        for v in views_raw:
            unknown_v = set(v) - {"name", "description", "layer_weights", "constraint_weights"}
            if unknown_v:
                raise NormError(f"view {v.get('name')!r}: unknown keys {sorted(unknown_v)}")
            views.append(
                View(
                    name=str(v["name"]),
                    layer_weights=v.get("layer_weights"),
                    constraint_weights=v.get("constraint_weights"),
                    description=str(v.get("description", "")),
                )
            )

        constraints = []
        for c in d.get("constraints", []):
            unknown_c = set(c) - _CONSTRAINT_KEYS
            if unknown_c:
                raise NormError(
                    f"constraint {c.get('id')!r}: unknown keys {sorted(unknown_c)}; allowed {sorted(_CONSTRAINT_KEYS)}"
                )
            if "type" not in c:
                raise NormError(f"constraint {c.get('id')!r}: missing 'type'")
            if "layer" not in c:
                raise NormError(f"constraint {c.get('id')!r}: missing 'layer'")
            constraints.append(
                NormConstraint(
                    id=str(c["id"]),
                    layer=str(c["layer"]),
                    constraint=constraint_from_dict(str(c["type"]), c.get("params", {}) or {}),
                    weight=c.get("weight", 1.0),
                    applicability=c.get("applicability", {}) or {},
                    description=str(c.get("description", "")),
                )
            )
        if not layers:
            layers = [Layer(id=lid) for lid in dict.fromkeys(c.layer for c in constraints)]
        return cls(
            constraints=tuple(constraints),
            layers=tuple(layers),
            views=tuple(views),
            name=str(d.get("name", "norm")),
            version=str(d.get("version", "1")),
            description=str(d.get("description", "")),
            scoring_mode=str(d.get("scoring_mode", DEFAULT_SCORING_MODE)),
            derived_attributes=tuple(d.get("derived_attributes", []) or []),
            metadata=dict(d.get("metadata", {}) or {}),
        )

    def fingerprint(self) -> str:
        """SHA-256 of the canonical JSON form (provenance for score results)."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_json_default)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

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


_TOP_KEYS = {
    "schema_version",
    "name",
    "version",
    "description",
    "scoring_mode",
    "metadata",
    "layers",
    "views",
    "constraints",
    "derived_attributes",
}
_CONSTRAINT_KEYS = {"id", "layer", "type", "params", "weight", "applicability", "description"}


def _rule_attributes(rule: Mapping[str, Any]) -> set[str]:
    """Case attributes an applicability rule reads (kept for consumers that import it from here)."""
    return set(parse(rule, what="applicability").columns)


__all__ = [
    "CONSTRAINT_TYPES",
    "SCHEMA_VERSION",
    "SCORING_MODES",
    "TYPE_ALIASES",
    "Layer",
    "Norm",
    "NormConstraint",
    "View",
    "__version__",
]
