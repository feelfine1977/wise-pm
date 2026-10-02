"""The 0.1.0 row filters, frozen as an oracle for the selection grammar.

:func:`legacy_recipe_mask` is the where-clause of a derived-attribute recipe
and :func:`legacy_where_mask` the ``where=`` filter of results, drivers and
penalty mass, both as they stood before the selection module evaluated them.
Do not edit them to make a comparison pass: they are the reference.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from wise.errors import LogSchemaError, NormError


def legacy_recipe_mask(events: pd.DataFrame, where: Mapping[str, Any] | None) -> np.ndarray:
    if not where:
        return np.ones(len(events), dtype=bool)
    col = where.get("column")
    if col is None or col not in events.columns:
        raise LogSchemaError(f"recipe where-clause refers to unknown event column {col!r}")
    s = events[col]
    if "in" in where:
        return s.isin(list(where["in"])).to_numpy()
    if "not_in" in where:
        return (~s.isin(list(where["not_in"]))).to_numpy()
    if "regex" in where:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return s.astype(str).str.contains(where["regex"], regex=True, na=False).to_numpy()
    if "eq" in where:
        return (s == where["eq"]).to_numpy()
    raise NormError(f"recipe where-clause needs one of in/not_in/regex/eq: {dict(where)}")


def legacy_where_mask(cases: pd.DataFrame, where: dict[str, Any] | pd.Series | None) -> pd.Series:
    mask = pd.Series(True, index=cases.index)
    if where is None:
        return mask
    if isinstance(where, dict):
        for k, v in where.items():
            if k in cases.columns:
                mask &= cases[k] == v
            elif k == cases.index.name:
                mask &= pd.Series(cases.index == v, index=cases.index)
            else:
                raise NormError(f"unknown case attribute {k!r} in where; known: {list(cases.columns)}")
        return mask
    return mask & where.reindex(cases.index, fill_value=False).astype(bool)
