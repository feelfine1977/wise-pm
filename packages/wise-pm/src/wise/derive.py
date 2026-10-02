"""Declarative recipes for derived case attributes.

Engineered signals (manual-touch counts, distinct resources, value
volatility, indicator × exposure products) are part of a norm's semantics.
A norm lists their definitions under ``derived_attributes``, so the norm file
is self-contained. :func:`wise.score` computes them for its run with
:func:`derived_log` and leaves the scored log alone;
:meth:`wise.EventLog.derive` attaches them to a log explicitly.

Each recipe is a mapping with a ``name`` and a ``kind``:

``count``
    ``{"kind": "count", "activities": [...], "after": [...]?, "before": [...]?}``
    — occurrences per case, optionally scoped.
``count_events``
    ``{"kind": "count_events", "where": WHERE}`` — events matching a
    condition on an event column.
``nunique``
    ``{"kind": "nunique", "column": c, "where": WHERE?}`` — distinct values
    of an event column per case.
``agg``
    ``{"kind": "agg", "column": c, "agg": "sum|mean|max|min|std|first|last",
    "where": WHERE?}`` — aggregate of a numeric event column.
``cv``
    ``{"kind": "cv", "column": c}`` — coefficient of variation ``std / mean``.
``lag``
    ``{"kind": "lag", "a": [...], "b": [...], "unit": "D", "activation":
    "first", "response": "first_after"}`` — time from ``a`` to ``b`` (NaN when
    undefined), for use with a :class:`~wise.Metric` constraint. The pairing
    is the one of :class:`~wise.Lag` (:func:`wise.families.lag.pairs`). With
    ``response: "first_overall"`` a ``b`` before ``a`` gives NaN unless
    ``"allow_negative": true``.
``ratio``
    ``{"kind": "ratio", "numerator": attr, "denominator": attr}``.
``quantile_scale``
    ``{"kind": "quantile_scale", "attribute": attr, "q": 0.95}`` — attribute
    divided by its ``q`` quantile, clipped to ``[0, 1]``.
``indicator_times``
    ``{"kind": "indicator_times", "activities": [...], "attribute": attr}`` —
    ``1[cnt > 0] · attribute``.
``eval``
    ``{"kind": "eval", "expr": "manual_touches / n_events"}`` — a pandas
    expression over the case table.

``WHERE`` is ``{"column": c, "in": [...]}``, ``{"column": c, "regex": r}``,
``{"column": c, "eq": v}`` or ``{"column": c, "not_in": [...]}``, or a rule of
the selection grammar over event columns (:mod:`wise.selection`), e.g.
``{"all": [{"attr": "activity", "eq": "Record Goods Receipt"}, {"lacks": "Clear Invoice"}]}``.
It is evaluated over the event table under the legacy missing-value policy.
"""

from __future__ import annotations

import json
import weakref
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .constraints import _UNITS, as_labels
from .errors import LogSchemaError, NormError
from .families import lag as lag_family
from .selection import LEGACY, EventTable, parse_recipe_where

if TYPE_CHECKING:  # pragma: no cover
    from .log import EventLog

# Per log, the recipe that produced each derived attribute, so that an unchanged recipe is reused.
_produced: weakref.WeakKeyDictionary[EventLog, dict[str, str]] = weakref.WeakKeyDictionary()
# Per log, the attributes derived_log computed on it, keyed by the recipes up to and including theirs.
_computed: weakref.WeakKeyDictionary[EventLog, dict[str, pd.Series]] = weakref.WeakKeyDictionary()

KINDS = ("count", "count_events", "nunique", "agg", "cv", "lag", "ratio", "quantile_scale", "indicator_times", "eval")


def _where_mask(log: EventLog, where: Mapping[str, Any] | None) -> np.ndarray:
    if not where:
        return np.ones(len(log.events), dtype=bool)
    columns = log.events.columns
    if isinstance(where, Mapping) and "column" in where and where["column"] not in columns:
        raise LogSchemaError(f"recipe where-clause refers to unknown event column {where['column']!r}")
    selection = parse_recipe_where(where)
    for col in selection.columns:
        if col not in columns:
            raise LogSchemaError(f"recipe where-clause refers to unknown event column {col!r}")
    mask = selection.evaluate(EventTable(log), missing=LEGACY)
    return mask.fillna(False).to_numpy(dtype=bool)


def validate_recipe(recipe: Mapping[str, Any]) -> None:
    """Raise :class:`NormError` if the recipe is malformed."""
    if "name" not in recipe or not str(recipe["name"]):
        raise NormError(f"derived attribute recipe without name: {dict(recipe)}")
    kind = recipe.get("kind")
    if kind not in KINDS:
        raise NormError(f"derived attribute {recipe['name']!r}: unknown kind {kind!r}; known {KINDS}")
    needs = {
        "count": ("activities",),
        "count_events": ("where",),
        "nunique": ("column",),
        "agg": ("column", "agg"),
        "cv": ("column",),
        "lag": ("a", "b"),
        "ratio": ("numerator", "denominator"),
        "quantile_scale": ("attribute",),
        "indicator_times": ("activities", "attribute"),
        "eval": ("expr",),
    }[kind]
    missing = [k for k in needs if k not in recipe]
    if missing:
        raise NormError(f"derived attribute {recipe['name']!r} ({kind}): missing {missing}")


def compute_recipe(log: EventLog, recipe: Mapping[str, Any]) -> pd.Series:
    """Evaluate one recipe on a log → Series aligned to ``log.cases``."""
    validate_recipe(recipe)
    kind = recipe["kind"]
    idx = log.case_index

    def per_case(values: np.ndarray) -> pd.Series:
        return pd.Series(values, index=log.case_ids)

    if kind == "count":
        return log.count_scoped(as_labels(recipe["activities"]), after=recipe.get("after"), before=recipe.get("before"))
    if kind == "count_events":
        return per_case(idx.count(_where_mask(log, recipe["where"])).astype(float))
    if kind == "nunique":
        m = _where_mask(log, recipe.get("where"))
        col = recipe["column"]
        if col not in log.events.columns:
            raise LogSchemaError(f"recipe {recipe['name']!r}: unknown event column {col!r}")
        return per_case(idx.aggregate(log.events[col], m, how="nunique", empty=0).astype(float))
    if kind == "agg":
        m = _where_mask(log, recipe.get("where"))
        col = recipe["column"]
        if col not in log.events.columns:
            raise LogSchemaError(f"recipe {recipe['name']!r}: unknown event column {col!r}")
        vals = pd.to_numeric(log.events[col], errors="coerce")
        # pandas, not the index's "sum", so that a sum keeps pandas' compensated arithmetic
        s = vals[m].groupby(idx.case_codes[m]).agg(recipe["agg"])
        return per_case(s.reindex(range(idx.n_cases)).to_numpy(dtype=float))
    if kind == "cv":
        col = recipe["column"]
        if col not in log.events.columns:
            raise LogSchemaError(f"recipe {recipe['name']!r}: unknown event column {col!r}")
        vals = pd.to_numeric(log.events[col], errors="coerce")
        std = pd.Series(idx.aggregate(vals, how="std")).fillna(0.0)
        mean = pd.Series(idx.aggregate(vals, how="mean"))
        eps = float(recipe.get("eps", 1e-9))
        return per_case((std / mean.abs().clip(lower=eps)).to_numpy(dtype=float))
    if kind == "lag":
        unit = str(recipe.get("unit", "D"))
        if unit not in _UNITS:
            raise NormError(f"derived attribute {recipe['name']!r}: unknown unit {unit!r}")
        # as in 0.1.0, any activation other than "first" is the last a, any response other than "first_overall" the first b after it
        pairing = lag_family.pairs(
            log,
            recipe["a"],
            recipe["b"],
            activation="first" if recipe.get("activation", "first") == "first" else "last",
            response="first_overall" if recipe.get("response", "first_after") == "first_overall" else "first_after",
            unit=unit,
            negative=bool(recipe.get("allow_negative", False)),
        )
        return per_case(pairing.per_case())
    if kind == "ratio":
        num = log.attribute(recipe["numerator"])
        den = log.attribute(recipe["denominator"])
        return (num / den.replace(0, np.nan)).astype(float)
    if kind == "quantile_scale":
        x = log.attribute(recipe["attribute"])
        q = float(x.quantile(float(recipe.get("q", 0.95))))
        if not np.isfinite(q) or q <= 0:
            q = 1.0
        return (x / q).clip(0.0, 1.0)
    if kind == "indicator_times":
        ind = (log.count(as_labels(recipe["activities"])) > 0).astype(float)
        return ind * log.attribute(recipe["attribute"]).fillna(0.0)
    if kind == "eval":
        try:
            out = log.cases.eval(recipe["expr"])
        except Exception as exc:  # pragma: no cover - pandas raises many types
            raise NormError(f"derived attribute {recipe['name']!r}: cannot evaluate {recipe['expr']!r}: {exc}") from exc
        return pd.to_numeric(pd.Series(out, index=log.case_ids), errors="coerce")
    raise NormError(f"unknown recipe kind {kind!r}")  # pragma: no cover


def _key(recipe: Mapping[str, Any]) -> str:
    return json.dumps(dict(recipe), sort_keys=True, default=str)


def forget(log: EventLog, names: Iterable[str] | None = None) -> None:
    """Drop what this module remembers about ``log`` after its case table
    changed: the attributes computed for runs, and the record of the recipe
    that produced ``names`` (every name when ``None``)."""
    _computed.pop(log, None)
    produced = _produced.get(log)
    if produced is not None:
        for name in list(produced) if names is None else names:
            produced.pop(name, None)


def derived_log(log: EventLog, recipes: Iterable[Mapping[str, Any]]) -> EventLog:
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
    produced = _produced.get(log, {})
    existing = set(log.cases.columns)
    memo = _computed.setdefault(log, {})
    out, chain = log, ""
    for r in recipes:
        validate_recipe(r)
        name, key = str(r["name"]), _key(r)
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


def apply_recipes(log: EventLog, recipes: Iterable[Mapping[str, Any]], *, overwrite: bool = True) -> list[str]:
    """Compute and attach all recipes in order (later recipes may use earlier ones).

    Changes ``log`` (see :meth:`EventLog.derive`). With ``overwrite=False`` an
    attribute is kept when it was produced by the same recipe before; a
    changed recipe is always recomputed.
    """
    names: list[str] = []
    for r in recipes:
        validate_recipe(r)
        name = str(r["name"])
        key = _key(r)
        produced = _produced.setdefault(log, {})
        if name in log.cases.columns and not overwrite and produced.get(name) == key:
            names.append(name)
            continue
        log.add_case_attribute(name, compute_recipe(log, r))
        produced[name] = key
        names.append(name)
    return names
