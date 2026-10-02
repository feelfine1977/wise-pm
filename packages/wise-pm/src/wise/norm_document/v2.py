"""Schema v2 adapter: the JSON layout of 0.1.0 (also read for schema version 1).

The document is an object with the top-level keys ``schema_version``,
``name``, ``version``, ``description``, ``scoring_mode``, ``metadata``,
``layers``, ``views``, ``derived_attributes`` and ``constraints``. Layers and
views may also be given as mappings keyed by id and name; without
``layers`` the layers are those named by the constraints, in order of first
use. Unknown keys are refused so that a typo cannot silently change the
meaning of a norm. Everything else (weights, rules, recipes, the cross
references) is checked once, by the model the document is read into.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .._frozen import thaw
from .._json import jsonable
from ..constraints import constraint_from_dict
from ..errors import NormError
from ..norm import DEFAULT_SCORING_MODE, Layer, Norm, NormConstraint, View

VERSIONS = (1, 2)

TOP_KEYS = frozenset(
    {
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
)
VIEW_KEYS = frozenset({"name", "description", "layer_weights", "constraint_weights"})
CONSTRAINT_KEYS = frozenset({"id", "layer", "type", "params", "weight", "applicability", "description"})
#: Free-form text and annotations: written to the document, left out of its fingerprint.
FREE_FORM_KEYS = frozenset({"metadata", "description"})


# ----------------------------------------------------------------------------- read
def read(document: Mapping[str, Any], schema_version: int) -> Norm:
    """The norm a schema v2 document describes, carrying ``schema_version``."""
    unknown = set(document) - TOP_KEYS
    if unknown:
        raise NormError(f"unknown top-level keys in norm: {sorted(unknown)}")
    layers = tuple(_layers(document.get("layers", [])))
    views = tuple(_view(v) for v in _views(document.get("views", [])))
    constraints = tuple(_constraint(c) for c in document.get("constraints", []))
    if not layers:
        layers = tuple(Layer(id=lid) for lid in dict.fromkeys(c.layer for c in constraints))
    return Norm(
        constraints=constraints,
        layers=layers,
        views=views,
        name=str(document.get("name", "norm")),
        version=str(document.get("version", "1")),
        description=str(document.get("description", "")),
        scoring_mode=str(document.get("scoring_mode", DEFAULT_SCORING_MODE)),
        derived_attributes=tuple(document.get("derived_attributes", []) or []),
        metadata=dict(document.get("metadata", {}) or {}),
        schema_version=schema_version,
    )


def _layers(raw: Any) -> list[Layer]:
    if isinstance(raw, Mapping):
        raw = [{"id": k, **(v if isinstance(v, Mapping) else {"name": str(v)})} for k, v in raw.items()]
    return [Layer(id=str(lay["id"]), name=str(lay.get("name", "")), description=str(lay.get("description", ""))) for lay in raw]


def _views(raw: Any) -> list[Any]:
    if isinstance(raw, Mapping):
        return [{"name": k, **v} for k, v in raw.items()]
    return list(raw)


def _view(v: Mapping[str, Any]) -> View:
    unknown = set(v) - VIEW_KEYS
    if unknown:
        raise NormError(f"view {v.get('name')!r}: unknown keys {sorted(unknown)}")
    return View(
        name=str(v["name"]),
        layer_weights=v.get("layer_weights"),
        constraint_weights=v.get("constraint_weights"),
        description=str(v.get("description", "")),
    )


def _constraint(c: Mapping[str, Any]) -> NormConstraint:
    unknown = set(c) - CONSTRAINT_KEYS
    if unknown:
        raise NormError(f"constraint {c.get('id')!r}: unknown keys {sorted(unknown)}; allowed {sorted(CONSTRAINT_KEYS)}")
    if "type" not in c:
        raise NormError(f"constraint {c.get('id')!r}: missing 'type'")
    if "layer" not in c:
        raise NormError(f"constraint {c.get('id')!r}: missing 'layer'")
    return NormConstraint(
        id=str(c["id"]),
        layer=str(c["layer"]),
        constraint=constraint_from_dict(str(c["type"]), c.get("params", {}) or {}),
        weight=c.get("weight", 1.0),
        applicability=c.get("applicability", {}) or {},
        description=str(c.get("description", "")),
    )


# ----------------------------------------------------------------------------- write
def write(norm: Norm) -> dict[str, Any]:
    """The schema v2 document of ``norm``: plain containers, independent of the norm."""
    return {
        "schema_version": norm.schema_version,
        "name": norm.name,
        "version": norm.version,
        "description": norm.description,
        "scoring_mode": norm.scoring_mode,
        "metadata": jsonable(norm.metadata),
        "layers": [{"id": layer.id, "name": layer.name, "description": layer.description} for layer in norm.layers],
        "views": [
            {
                "name": v.name,
                "description": v.description,
                **({"layer_weights": dict(v.layer_weights)} if v.layer_weights is not None else {}),
                **({"constraint_weights": dict(v.constraint_weights)} if v.constraint_weights is not None else {}),
            }
            for v in norm.views
        ],
        "derived_attributes": [thaw(r) for r in norm.derived_attributes],
        "constraints": [write_constraint(c) for c in norm.constraints],
    }


def write_constraint(nc: NormConstraint) -> dict[str, Any]:
    """One entry of the ``constraints`` list."""
    return {
        "id": nc.id,
        "layer": nc.layer,
        "type": nc.constraint.type,
        "params": jsonable(nc.constraint.params()),
        "weight": nc.weight,
        "applicability": thaw(nc.applicability),
        "description": nc.description,
    }


def identity(document: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a written document that defines what the norm measures: the document without the free-form
    ``metadata`` and the ``description`` of the norm, its layers, views and constraints."""
    out = {k: v for k, v in document.items() if k not in FREE_FORM_KEYS}
    for part in ("layers", "views", "constraints"):
        out[part] = [{k: v for k, v in entry.items() if k not in FREE_FORM_KEYS} for entry in document[part]]
    return out
