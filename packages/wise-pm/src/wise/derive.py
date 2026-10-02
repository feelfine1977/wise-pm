"""Derived case attributes: computing a norm's recipes on a log.

Engineered signals (manual-touch counts, distinct resources, value
volatility, indicator × exposure products) are part of a norm's semantics.
A norm lists their definitions under ``derived_attributes``, so the norm file
is self-contained. Each definition is read into a typed record of
:mod:`wise.recipes`, which lists the kinds and their keys.
:func:`wise.score` computes them for its run with :func:`derived_log` and
leaves the scored log alone; :meth:`wise.EventLog.derive` attaches them to a
log explicitly (:func:`apply_recipes`). Both remember which recipe produced
an attribute, so that an unchanged recipe is not computed again.
:func:`fit_calibration` fits the calibration records of a norm's
population-dependent recipes on a log, and :func:`realised_calibrations`
states the records a run computed with.
"""

from __future__ import annotations

import dataclasses
import json
import weakref
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import pandas as pd

from .calibration import Calibration
from .errors import NormError
from .recipes import KINDS as KINDS
from .recipes import QuantileScaleRecipe, Recipe, as_recipe, parse_recipe

if TYPE_CHECKING:  # pragma: no cover
    from .log import EventLog
    from .norm import Norm

# Per log, the recipe that produced each derived attribute, so that an unchanged recipe is reused.
_produced: weakref.WeakKeyDictionary[EventLog, dict[str, str]] = weakref.WeakKeyDictionary()
# Per log, the attributes derived_log computed on it, keyed by the recipes up to and including theirs.
_computed: weakref.WeakKeyDictionary[EventLog, dict[str, pd.Series]] = weakref.WeakKeyDictionary()


def validate_recipe(recipe: Mapping[str, Any]) -> None:
    """Raise :class:`NormError` if the recipe document is malformed (see :func:`wise.recipes.parse_recipe`)."""
    parse_recipe(recipe)


def compute_recipe(log: EventLog, recipe: Recipe | Mapping[str, Any]) -> pd.Series:
    """Evaluate one recipe (a record or its document) on a log → Series aligned to ``log.cases``."""
    return as_recipe(recipe).compute(log)


def _key(recipe: Recipe) -> str:
    return json.dumps(recipe.to_dict(), sort_keys=True, default=str)


def forget(log: EventLog, names: Iterable[str] | None = None) -> None:
    """Drop what this module remembers about ``log`` after its case table
    changed: the attributes computed for runs, and the record of the recipe
    that produced ``names`` (every name when ``None``)."""
    _computed.pop(log, None)
    produced = _produced.get(log)
    if produced is not None:
        for name in list(produced) if names is None else names:
            produced.pop(name, None)


def derived_log(log: EventLog, recipes: Iterable[Recipe | Mapping[str, Any]]) -> EventLog:
    """The log to evaluate a norm on: ``log`` with the recipes' attributes
    (computed in order; later recipes may use earlier ones).

    ``log`` is not changed. When it already holds every attribute, it is
    returned itself; otherwise the result is a new log that shares its events
    and case index (:meth:`EventLog.with_case_attributes`). An attribute
    ``log`` already has is used when :meth:`EventLog.derive` produced it with
    the same recipe; any other case attribute of the same name raises
    :class:`NormError`, so a recipe never replaces the user's data. Computed
    attributes are remembered for ``log`` until its case table changes.
    """
    records = [as_recipe(r) for r in recipes]
    produced = _produced.get(log, {})
    existing = set(log.cases.columns)
    memo = _computed.setdefault(log, {})
    out, chain = log, ""
    for r in records:
        name, key = r.name, _key(r)
        chain += key + "\n"
        if name in existing:
            if produced.get(name) == key:
                continue
            raise NormError(
                f"derived attribute {name!r} would replace the case attribute {name!r} of the log, which this "
                "recipe did not produce; rename the recipe or the attribute"
            )
        values = memo.get(chain)
        if values is None:
            values = memo[chain] = compute_recipe(out, r)
        out = out.with_case_attributes({name: values})
    return out


def apply_recipes(log: EventLog, recipes: Iterable[Recipe | Mapping[str, Any]], *, overwrite: bool = True) -> list[str]:
    """Compute and attach all recipes in order (later recipes may use earlier ones).

    Changes ``log`` (see :meth:`EventLog.derive`). With ``overwrite=False`` an
    attribute is kept when it was produced by the same recipe before; a
    changed recipe is always recomputed. Every recipe is parsed before the
    first is computed, so a malformed one leaves the log as it was.
    """
    records = [as_recipe(r) for r in recipes]
    names: list[str] = []
    for r in records:
        name = r.name
        key = _key(r)
        produced = _produced.setdefault(log, {})
        if name in log.cases.columns and not overwrite and produced.get(name) == key:
            names.append(name)
            continue
        log.add_case_attribute(name, compute_recipe(log, r))
        produced[name] = key
        names.append(name)
    return names


def fit_calibration(norm: Norm, log: EventLog) -> Norm:
    """The norm with a fitted calibration record (:class:`~wise.calibration.Calibration`)
    on every ``quantile_scale`` recipe, fitted on ``log`` (decision 0001, section 3).

    Each divisor is the recipe's quantile over the cases of ``log``, with the
    attributes of the earlier recipes computed as :func:`derived_log` does
    (an earlier ``quantile_scale`` with the record just fitted). The records
    state ``log.fingerprint()``, the number of cases whose value entered the
    quantile and the window of ``log``. A record that was there is replaced.
    The recipe documents keep their other keys; neither ``norm`` nor ``log``
    is changed. Scoring with the fitted norm divides by the stored divisors,
    whatever cases are scored; on ``log`` itself it gives the values of a
    run without records.
    """
    fingerprint = log.fingerprint()
    records: list[Recipe] = []
    documents: list[dict[str, Any]] = []
    for record, document in zip(norm.recipes, norm.derived_attributes, strict=True):
        if isinstance(record, QuantileScaleRecipe):
            unfitted = dataclasses.replace(record, calibration=None)
            fitted = unfitted.calibrate(derived_log(log, records), log_fingerprint=fingerprint)
            record = dataclasses.replace(record, calibration=fitted)
            document = {k: v for k, v in document.items() if k != "divisor"}
            document["calibration"] = fitted.to_dict()
        records.append(record)
        documents.append(document)
    return norm.replace(derived_attributes=tuple(documents))


def realised_calibrations(scored: EventLog, log: EventLog, recipes: Iterable[Recipe]) -> dict[str, Calibration]:
    """Per ``quantile_scale`` recipe, the record it computes with on ``log``
    (``scored`` with the attributes of the recipes, as :func:`derived_log`
    gives it): its fitted record, or the record of the divisor realised on
    ``log``, stating the fingerprint of ``scored``."""
    population = [r for r in recipes if isinstance(r, QuantileScaleRecipe)]
    if not population:
        return {}
    fingerprint = scored.fingerprint()
    return {r.name: r.calibrate(log, log_fingerprint=fingerprint) for r in population}
