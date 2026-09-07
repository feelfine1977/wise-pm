"""Explicit capture and application of a fitted reference.

``derive.py``'s ``quantile_scale`` recipe divides an attribute by a quantile
**of the log it is applied to**. That is the right default for a single run and
the wrong thing for a comparison: applied to next month's data it silently
refits, so two runs that look like they share a scale do not.

This module leaves that default untouched and adds the explicit path:

>>> import wise
>>> from wise.evidence import fit_calibration, apply_calibration
>>> log = wise.datasets.running_p2p_log()
>>> log.add_case_attribute("touches", [1.0, 2.0, 3.0, 4.0, 20.0])
>>> recipe = {"name": "touch_index", "kind": "quantile_scale", "attribute": "touches", "q": 0.95}
>>> fitted = fit_calibration(log, recipe)
>>> round(fitted.divisor, 3)
16.8
>>> float(apply_calibration(log, fitted).max())   # frozen: not refitted on this log
1.0

A record is immutable and carries its own identity. Changing the recipe, the
quantile or the fitting population produces a different
:attr:`CalibrationRecord.calibration_id`, so a frozen reference can never be
confused with a recomputed one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..derive import _quantile_divisor, validate_recipe
from ..errors import EvidenceError

if TYPE_CHECKING:  # pragma: no cover
    from ..log import EventLog

#: Recipe kinds that fit something from the data and can therefore be frozen.
FITTABLE_KINDS = ("quantile_scale",)


def recipe_fingerprint(recipe: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical recipe. A changed recipe is a changed identity."""
    canonical = json.dumps(dict(recipe), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CalibrationRecord:
    """A reference fitted once, kept as a value, and applied without refitting."""

    recipe_name: str
    kind: str
    attribute: str
    divisor: float
    parameters: dict[str, Any]
    recipe: dict[str, Any]
    recipe_fingerprint: str
    fitting_population: dict[str, Any]
    unit: str = "attribute_unit"
    fitted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        object.__setattr__(self, "recipe", dict(self.recipe))
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "fitting_population", dict(self.fitting_population))
        divisor = float(self.divisor)
        if not (divisor > 0) or divisor != divisor or divisor == float("inf"):
            raise EvidenceError(f"calibration {self.recipe_name!r}: divisor must be finite and > 0, got {self.divisor!r}")
        object.__setattr__(self, "divisor", divisor)

    @property
    def calibration_id(self) -> str:
        """Identity of *this* fitted reference: recipe, parameters, value, population."""
        payload = {
            "recipe_fingerprint": self.recipe_fingerprint,
            "kind": self.kind,
            "attribute": self.attribute,
            "parameters": self.parameters,
            "divisor": self.divisor,
            "fitting_population": self.fitting_population,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
        return "cal-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]

    def matches(self, recipe: Mapping[str, Any]) -> bool:
        """Whether this record was fitted from exactly this recipe."""
        return recipe_fingerprint(recipe) == self.recipe_fingerprint

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "recipe_name": self.recipe_name,
            "kind": self.kind,
            "attribute": self.attribute,
            "divisor": self.divisor,
            "unit": self.unit,
            "parameters": dict(self.parameters),
            "recipe_fingerprint": self.recipe_fingerprint,
            "fitting_population": dict(self.fitting_population),
            "fitted_at": self.fitted_at,
        }

    def __repr__(self) -> str:
        return f"CalibrationRecord({self.recipe_name!r}, {self.kind}, divisor={self.divisor!r}, id={self.calibration_id})"


def fit_calibration(
    log: EventLog,
    recipe: Mapping[str, Any],
    *,
    unit_type: str = "case",
    source: str | None = None,
) -> CalibrationRecord:
    """Fit a reference on ``log`` and freeze it.

    The value is computed by the same code the ordinary recipe uses, so a
    freshly fitted calibration reproduces the default behaviour exactly.
    """
    validate_recipe(recipe)
    kind = str(recipe["kind"])
    if kind not in FITTABLE_KINDS:
        raise EvidenceError(f"recipe kind {kind!r} fits nothing; fittable kinds are {list(FITTABLE_KINDS)}")
    attribute = str(recipe["attribute"])
    q = float(recipe.get("q", 0.95))
    values = log.attribute(attribute)
    divisor = _quantile_divisor(values, q)
    population = {
        "unit_type": unit_type,
        "n_units": len(values),
        "n_observed": int(values.notna().sum()),
        "source": source,
        "snapshot_fingerprint": log.snapshot().fingerprint,
    }
    return CalibrationRecord(
        recipe_name=str(recipe["name"]),
        kind=kind,
        attribute=attribute,
        divisor=divisor,
        parameters={"q": q},
        recipe=dict(recipe),
        recipe_fingerprint=recipe_fingerprint(recipe),
        fitting_population=population,
    )


def apply_calibration(log: EventLog, record: CalibrationRecord, *, attach: bool = False) -> pd.Series:
    """Apply a frozen reference to ``log`` — never refitting it.

    With ``attach=True`` the result is also added to the log as a case
    attribute under the recipe's name.
    """
    from ..derive import compute_recipe

    values = compute_recipe(log, record.recipe, calibration=record)
    if attach:
        log.add_case_attribute(record.recipe_name, values)
        log._recipe_cache[record.recipe_name] = f"calibrated:{record.calibration_id}"
    return values


def as_calibration_map(
    calibrations: Iterable[CalibrationRecord] | Mapping[str, CalibrationRecord] | None,
) -> dict[str, CalibrationRecord]:
    """Normalise the ``calibrations=`` argument to ``{recipe name: record}``."""
    if calibrations is None:
        return {}
    if isinstance(calibrations, Mapping):
        items = list(calibrations.values())
    else:
        items = list(calibrations)
    out: dict[str, CalibrationRecord] = {}
    for record in items:
        if not isinstance(record, CalibrationRecord):
            raise EvidenceError(f"calibrations must be CalibrationRecord instances, got {type(record).__name__}")
        if record.recipe_name in out:
            raise EvidenceError(f"two calibrations for derived attribute {record.recipe_name!r}")
        out[record.recipe_name] = record
    return out
