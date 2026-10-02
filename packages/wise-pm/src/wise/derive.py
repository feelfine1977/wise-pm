"""Declarative recipes for derived case attributes.

Engineered signals (manual-touch counts, distinct resources, value
volatility, indicator × exposure products) are part of a norm's semantics.
A norm lists their definitions under ``derived_attributes`` and
:meth:`wise.EventLog.derive` computes them before scoring, so the norm file
is self-contained.

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
    undefined), for use with a :class:`~wise.Metric` constraint. With
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
``{"column": c, "eq": v}`` or ``{"column": c, "not_in": [...]}``.
"""

from __future__ import annotations

import json
import warnings
import weakref
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .constraints import _UNITS, as_labels, in_units
from .errors import LogSchemaError, NormError

if TYPE_CHECKING:  # pragma: no cover
    from .log import EventLog

# Per log, the recipe that produced each derived attribute, so that an unchanged recipe is reused.
_produced: weakref.WeakKeyDictionary[EventLog, dict[str, str]] = weakref.WeakKeyDictionary()

KINDS = ("count", "count_events", "nunique", "agg", "cv", "lag", "ratio", "quantile_scale", "indicator_times", "eval")


def _where_mask(log: EventLog, where: Mapping[str, Any] | None) -> np.ndarray:
    if not where:
        return np.ones(len(log.events), dtype=bool)
    col = where.get("column")
    if col is None or col not in log.events.columns:
        raise LogSchemaError(f"recipe where-clause refers to unknown event column {col!r}")
    s = log.events[col]
    if "in" in where:
        return s.isin(list(where["in"])).to_numpy()
    if "not_in" in where:
        return (~s.isin(list(where["not_in"]))).to_numpy()
    if "regex" in where:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # patterns with groups are fine here
            return s.astype(str).str.contains(where["regex"], regex=True, na=False).to_numpy()
    if "eq" in where:
        return (s == where["eq"]).to_numpy()
    raise NormError(f"recipe where-clause needs one of in/not_in/regex/eq: {dict(where)}")


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
        t_a, t_b = log.first_after(
            recipe["a"], recipe["b"], activation=recipe.get("activation", "first"), response=recipe.get("response", "first_after")
        )
        unit = str(recipe.get("unit", "D"))
        if unit not in _UNITS:
            raise NormError(f"derived attribute {recipe['name']!r}: unknown unit {unit!r}")
        lag = in_units(t_b - t_a, unit)
        if recipe.get("response", "first_after") == "first_overall" and not recipe.get("allow_negative", False):
            lag = lag.where(lag >= 0)
        return lag
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


def apply_recipes(log: EventLog, recipes: Iterable[Mapping[str, Any]], *, overwrite: bool = True) -> list[str]:
    """Compute and attach all recipes in order (later recipes may use earlier ones).

    With ``overwrite=False`` an attribute is kept when it was produced by the
    same recipe before; a changed recipe is always recomputed.
    """
    names: list[str] = []
    for r in recipes:
        validate_recipe(r)
        name = str(r["name"])
        key = json.dumps(dict(r), sort_keys=True, default=str)
        produced = _produced.setdefault(log, {})
        if name in log.cases.columns and not overwrite and produced.get(name) == key:
            names.append(name)
            continue
        log.add_case_attribute(name, compute_recipe(log, r))
        produced[name] = key
        names.append(name)
    return names
