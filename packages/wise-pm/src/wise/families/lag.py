"""Lag ``(lag, a, b, δ, Δ)``: ``ν = sat(t_b − t_a; δ, Δ)`` (paper Sec. IV-B).

The options of :class:`~wise.constraints.Lag` choose the activation (first,
last or each ``a``), the response (first ``b`` at or after the activation, or
the first ``b`` overall) and what a missing endpoint means. A case left
without a value by ``missing_a="skip"`` or ``missing_b="skip"`` is not
evaluable (:data:`~wise.family.NOT_EVALUABLE`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..constraints import Lag, in_units, sat
from ..family import Evaluation, Family, Inputs

if TYPE_CHECKING:  # pragma: no cover
    from ..log import EventLog


def requires(c: Lag) -> Inputs:
    return Inputs(activities=c.activities())


def _lag_missing(v: pd.Series, lag: pd.Series, t_a: pd.Series, c: Lag, log: EventLog) -> pd.Series:
    a_missing = t_a.isna()
    b_missing = (~a_missing) & lag.isna()
    if c.missing_a == "violate":
        v = v.where(~a_missing, 1.0)
    if c.missing_b == "violate":
        v = v.where(~b_missing, 1.0)
    elif c.missing_b == "censor" and c.delta is not None:
        end = log.censoring_end(tolerance=(c.delta + c.width) * c.timedelta_unit)
        so_far = in_units(end - t_a, c.unit)
        v = v.where(~b_missing, sat(so_far, c.delta, c.width))
    return v


def _values(c: Lag, log: EventLog) -> pd.Series:
    if c.activation in ("first", "last"):
        t_a, t_b = log.first_after(c.a, c.b, activation=c.activation, response=c.response)
        lag = in_units(t_b - t_a, c.unit)
        if c.response == "first_overall":
            lag = lag.where(lag >= 0)  # b before the activation counts as missing
        v = sat(lag, c.delta, c.width) if c.delta is not None else lag * 0.0
        return _lag_missing(v, lag, t_a, c, log)

    # activation == "each": one violation per a-event, averaged per case
    pairs = log.activation_lags(c.a, c.b)
    t_a_case = log.first_ts(c.a)
    if pairs.empty:
        v = pd.Series(np.nan, index=log.case_ids)
        return _lag_missing(v, v, t_a_case, c, log)
    lag = in_units(pairs["t_b"] - pairs["t_a"], c.unit)
    vi = sat(lag, c.delta, c.width) if c.delta is not None else lag * 0.0
    missing = lag.isna()
    if c.missing_b == "violate":
        vi = vi.where(~missing, 1.0)
    elif c.missing_b == "censor" and c.delta is not None:
        end = pd.Timestamp(log.censoring_end(tolerance=(c.delta + c.width) * c.timedelta_unit))
        end = end.tz_convert("UTC").tz_localize(None) if end.tzinfo is not None else end
        so_far = in_units(end - pairs["t_a"], c.unit)
        vi = vi.where(~missing, sat(so_far, c.delta, c.width))
    per_case = vi.groupby(pairs["code"].to_numpy()).mean()
    v = pd.Series(per_case.reindex(range(len(log))).to_numpy(), index=log.case_ids)
    a_missing = t_a_case.isna()
    if c.missing_a == "violate":
        v = v.where(~a_missing, 1.0)
    return v


def evaluate(c: Lag, log: EventLog) -> Evaluation:
    return Evaluation.of(_values(c, log), log.case_ids)


FAMILY = Family(
    name="lag",
    constraint=Lag,
    requires=requires,
    evaluate=evaluate,
    numbers=frozenset({"delta", "width"}),
)
