"""Schema v3 adapter: the layout of schema version 2 with the corrected semantics (decision 0001, section 3).

A schema v3 document has the keys of a schema v2 document
(:mod:`wise.norm_document.v2`) and differs from it in four places:

* ``missing_values`` names the missing-value policy of applicability. It is
  required, and ``"three_valued"`` is its only value: a case whose attribute
  is missing is unknown under every operator, negations included, and is
  left out of the constraint's denominator.
* A ``lag`` constraint states its bound: ``delta`` (with ``width``) or
  ``"unbounded": true`` without ``delta``. A lag with neither, or with
  ``"delta": null``, is refused; schema v2 read a missing ``delta`` as
  ``0`` and ``null`` as unbounded.
* A ``quantile_scale`` recipe carries its ``calibration`` record, fitted or
  unfitted; scoring refuses an unfitted one (:mod:`wise.semantics`).
* There is no ``eval`` recipe (the norm refuses it, :mod:`wise.norm`).

Everything else is read and written as in schema version 2.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..constraints import TYPE_ALIASES
from ..errors import NormError
from ..norm import Norm, NormConstraint
from ..selection import THREE_VALUED
from . import v2

VERSIONS = (3,)

#: The values of the ``missing_values`` field.
MISSING_VALUES: tuple[str, ...] = (THREE_VALUED,)
TOP_KEYS = v2.TOP_KEYS | {"missing_values"}
FREE_FORM_KEYS = v2.FREE_FORM_KEYS


# ----------------------------------------------------------------------------- read
def read(document: Mapping[str, Any], schema_version: int) -> Norm:
    """The norm a schema v3 document describes, carrying ``schema_version``."""
    unknown = set(document) - TOP_KEYS
    if unknown:
        raise NormError(f"unknown top-level keys in norm: {sorted(unknown)}")
    if "missing_values" not in document:
        raise NormError(
            f"a schema version {schema_version} norm names its missing-value policy: 'missing_values': {THREE_VALUED!r}"
        )
    if document["missing_values"] not in MISSING_VALUES:
        raise NormError(f"missing_values must be one of {list(MISSING_VALUES)}, got {document['missing_values']!r}")
    for recipe in document.get("derived_attributes", []) or []:
        if isinstance(recipe, Mapping) and recipe.get("kind") == "quantile_scale" and recipe.get("calibration") is None:
            raise NormError(
                f"derived attribute {recipe.get('name')!r} (quantile_scale): a schema version {schema_version} recipe carries "
                "its calibration record; write {'status': 'unfitted', 'quantile': q} until it is fitted (wise.fit_calibration)"
            )
    legacy = {k: v for k, v in document.items() if k != "missing_values"}
    legacy["constraints"] = [_read_constraint(c) for c in document.get("constraints", [])]
    return v2.read(legacy, schema_version)


def _is_lag(entry: Mapping[str, Any]) -> bool:
    type_ = entry.get("type")
    return isinstance(type_, str) and TYPE_ALIASES.get(type_, type_) == "lag"


def _read_constraint(entry: Any) -> Any:
    if not isinstance(entry, Mapping) or not _is_lag(entry):
        return entry
    params = dict(entry.get("params", {}) or {})
    where = f"constraint {entry.get('id')!r} (lag)"
    if "unbounded" in params:
        unbounded = params.pop("unbounded")
        if not isinstance(unbounded, bool):
            raise NormError(f"{where}: unbounded must be true or false, got {unbounded!r}")
        if unbounded:
            if "delta" in params:
                raise NormError(f"{where}: an unbounded lag has no delta, got delta={params['delta']!r}")
            return {**entry, "params": {**params, "delta": None}}
    if params.get("delta") is None:
        raise NormError(f"{where}: a lag states its bound in schema version 3; give 'delta' (and 'width') or 'unbounded': true")
    return {**entry, "params": params}


# ----------------------------------------------------------------------------- write
def write(norm: Norm) -> dict[str, Any]:
    """The schema v3 document of ``norm``: the schema v2 layout with ``missing_values`` and explicit lag bounds."""
    out: dict[str, Any] = {}
    for key, value in v2.write(norm).items():
        out[key] = value
        if key == "scoring_mode":
            out["missing_values"] = norm.semantics.missing_values
    out["constraints"] = [write_constraint(c) for c in norm.constraints]
    return out


def write_constraint(nc: NormConstraint) -> dict[str, Any]:
    """One entry of the ``constraints`` list; an unbounded lag is written ``"unbounded": true``."""
    entry = v2.write_constraint(nc)
    if _is_lag(entry) and entry["params"].get("delta") is None:
        entry["params"] = {
            ("unbounded" if k == "delta" else k): (True if k == "delta" else v) for k, v in entry["params"].items()
        }
    return entry


def identity(document: Mapping[str, Any]) -> dict[str, Any]:
    """The written document without its free-form parts, as for schema version 2."""
    return v2.identity(document)
