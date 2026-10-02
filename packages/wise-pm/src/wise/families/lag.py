"""Lag ``(lag, a, b, δ, Δ)``: ``ν = sat(t_b − t_a; δ, Δ)`` (paper Sec. IV-B).

The options of :class:`~wise.constraints.Lag` choose the activation (first,
last or each ``a``), the response (first ``b`` at or after the activation, or
the first ``b`` overall) and what a missing endpoint means. The pairing of
activations with responses is :func:`pairs`; the ``lag`` recipe of
:mod:`wise.derive` uses the same rule.

Reasons: a case without an activation under ``missing_a="skip"`` is
:data:`~wise.family.MISSING_ACTIVATION`, one whose activations all lack a
response under ``missing_b="skip"`` is :data:`~wise.family.MISSING_RESPONSE`,
and one that has a value because an open activation was scored by its age
(``missing_b="censor"``) is :data:`~wise.family.CENSORED`. A missing endpoint
under ``"violate"`` gives ``ν = 1`` and is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..constraints import _UNITS, Activities, Lag, in_units, sat
from ..errors import NormError
from ..family import CENSORED, MISSING_ACTIVATION, MISSING_RESPONSE, Evaluation, Family, Inputs

if TYPE_CHECKING:  # pragma: no cover
    from ..log import EventLog


@dataclass(frozen=True, eq=False)
class Pairs:
    """The activations of a log, each paired with its response.

    One row per activation: ``case`` is the position of its case in the log,
    ``t_a`` its timestamp and ``lag`` the time to its response in the unit of
    the pairing, NaN where no response is paired with it. With activation
    ``"first"`` or ``"last"`` a case has at most one row; with ``"each"`` one
    row per ``a`` event. A case without a row has no activation.
    """

    case: NDArray[np.intp]
    t_a: pd.Series
    lag: pd.Series
    n_cases: int
    one_per_case: bool

    def activated(self) -> NDArray[np.bool_]:
        """Boolean per case: the case has an activation."""
        out = np.zeros(self.n_cases, dtype=bool)
        out[self.case] = True
        return out

    def per_case(self) -> NDArray[np.float64]:
        """The lag of each case, NaN without an activation or a response (activation first or last)."""
        if not self.one_per_case:
            raise NormError("activation 'each' gives one lag per activation, not one lag per case")
        out = np.full(self.n_cases, np.nan)
        out[self.case] = self.lag.to_numpy(dtype=float)
        return out


def pairs(
    log: EventLog,
    a: Activities,
    b: Activities,
    *,
    activation: Literal["first", "last", "each"] = "first",
    response: Literal["first_after", "first_overall"] = "first_after",
    unit: str = "D",
    negative: bool = False,
) -> Pairs:
    """Pair the activations ``a`` of every case with their response ``b``.

    ``activation`` is the first, the last or each ``a`` event with a
    timestamp. ``response="first_after"`` is the first ``b`` at or after the
    activation (an event at the same instant counts, whatever its row
    order); ``"first_overall"`` is the first ``b`` of the case, and one before
    the activation counts as no response unless ``negative=True`` keeps its
    negative lag. Lags are expressed in ``unit``.
    """
    if activation not in ("first", "last", "each"):
        raise NormError(f"lag: activation must be 'first', 'last' or 'each', got {activation!r}")
    if response not in ("first_after", "first_overall"):
        raise NormError(f"lag: response must be 'first_after' or 'first_overall', got {response!r}")
    if unit not in _UNITS:
        raise NormError(f"lag: unknown unit {unit!r}")
    if activation == "each":
        if response != "first_after":
            raise NormError("lag: activation='each' requires response='first_after'")
        rows = log.activation_lags(a, b)
        lag = in_units(rows["t_b"] - rows["t_a"], unit)
        return Pairs(rows["code"].to_numpy(dtype=np.intp), rows["t_a"], lag, len(log), one_per_case=False)
    t_a, t_b = log.first_after(a, b, activation=activation, response=response)
    lag = in_units(t_b - t_a, unit)
    if response == "first_overall" and not negative:
        lag = lag.where(lag >= 0)  # b before the activation counts as missing
    has = t_a.notna().to_numpy()
    return Pairs(
        np.flatnonzero(has).astype(np.intp),
        t_a[has].reset_index(drop=True),
        lag[has].reset_index(drop=True),
        len(log),
        one_per_case=True,
    )


def requires(c: Lag) -> Inputs:
    return Inputs(activities=c.activities())


def evaluate(c: Lag, log: EventLog) -> Evaluation:
    p = pairs(log, c.a, c.b, activation=c.activation, response=c.response, unit=c.unit)
    v = sat(p.lag, c.delta, c.width) if c.delta is not None else p.lag * 0.0
    open_ = p.lag.isna()
    censored = np.zeros(len(v), dtype=bool)
    if c.missing_b == "violate":
        v = v.where(~open_, 1.0)
    elif c.missing_b == "censor" and c.delta is not None:
        end = pd.Timestamp(log.censoring_end(tolerance=(c.delta + c.width) * c.timedelta_unit))
        if end.tzinfo is not None and getattr(p.t_a.dt, "tz", None) is None:
            end = end.tz_convert("UTC").tz_localize(None)  # activation_lags gives naive UTC timestamps
        so_far = in_units(end - p.t_a, c.unit)
        v = v.where(~open_, sat(so_far, c.delta, c.width))
        censored = open_.to_numpy()

    if p.one_per_case:
        values = np.full(p.n_cases, np.nan)
        values[p.case] = v.to_numpy(dtype=float)
    else:  # one violation per activation, averaged per case
        values = v.groupby(p.case).mean().reindex(range(p.n_cases)).to_numpy(dtype=float, copy=True)
    activated = p.activated()
    if c.missing_a == "violate":
        values[~activated] = 1.0
    censored_case = np.zeros(p.n_cases, dtype=bool)
    censored_case[p.case[censored]] = True
    return Evaluation.of(
        values,
        log.case_ids,
        why=((~activated, MISSING_ACTIVATION), (activated, MISSING_RESPONSE), (censored_case, CENSORED)),
    )


FAMILY = Family(
    name="lag",
    constraint=Lag,
    requires=requires,
    evaluate=evaluate,
    numbers=frozenset({"delta", "width"}),
)
