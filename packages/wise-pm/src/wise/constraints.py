"""Constraint catalogue: bounded violation signals ``ν_c(σ) ∈ [0, 1]``.

Implements Section IV-B of the paper. Every constraint maps a case ``σ`` to a
violation in ``[0, 1]``, where 0 means satisfied and 1 means maximally
violated under the declared cap. Graded constraints share the
threshold–saturation rule::

    sat(z; ϑ, W) = 0                     if z ≤ ϑ
                 = min((z − ϑ) / W, 1)   if z > ϑ

Constraint objects are frozen dataclasses holding parameters only; they are
evaluated on an :class:`~wise.log.EventLog` by :mod:`wise.scoring`.

Activity parameters accept one label or a list of labels. A list is treated
as *one* merged activity: counts are summed and first timestamps are taken
as the minimum over the alternatives. This covers logs where the same
business step is recorded under several labels.

The paper's five types are :class:`Presence`, :class:`Lag`, :class:`Balance`,
:class:`Singularity`, and :class:`Exclusion`. Two further types extend the
catalogue: :class:`Precedence` (no ``b`` before the first ``a``) and
:class:`Metric` (the saturation rule on a numeric case attribute).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd

from .errors import NormError

Labels = tuple[str, ...]
Activities = str | Sequence[str]

_UNITS: dict[str, str] = {
    "D": "D",
    "d": "D",
    "h": "h",
    "H": "h",
    "min": "min",
    "T": "min",
    "s": "s",
    "S": "s",
    "ms": "ms",
    "W": "W",
    "w": "W",
}


def as_labels(x: str | Iterable[str] | None, *, what: str = "activities") -> Labels:
    """Normalise one label or an iterable of labels to a tuple of strings.

    A bare string is one label (it is *not* iterated character by character).
    """
    if x is None:
        raise NormError(f"{what} must be given")
    if isinstance(x, str):
        return (x,)
    try:
        out = tuple(str(v) for v in x)
    except TypeError as exc:
        raise NormError(f"{what} must be a label or a list of labels, got {x!r}") from exc
    if not out:
        raise NormError(f"empty {what}")
    return out


def as_list(x: Any, *, what: str = "value") -> list[Any]:
    """A bare string or scalar becomes a one-element list; iterables become lists."""
    if x is None:
        return []
    if isinstance(x, str | bytes) or not isinstance(x, Iterable):
        return [x]
    return list(x)


def sat(z: Any, threshold: float, width: float) -> Any:
    """Threshold–saturation rule ``sat(z; ϑ, W)`` (paper Sec. IV-B).

    Works on scalars, numpy arrays, and pandas Series; NaN stays NaN. With
    ``width <= 0`` the rule degenerates to a step: 0 up to and including the
    threshold, 1 above it.

    >>> sat(8, 10, 20), sat(15, 10, 20), sat(20, 10, 20), sat(30, 10, 20)
    (0.0, 0.25, 0.5, 1.0)
    """
    if isinstance(z, pd.Series):
        ser = pd.to_numeric(z, errors="coerce").astype(float)
        res = (ser > threshold).astype(float) if width <= 0 else ((ser - threshold) / width).clip(0.0, 1.0)
        return res.where(ser.notna(), np.nan)
    arr = np.asarray(z, dtype=float)
    out = (arr > threshold).astype(float) if width <= 0 else np.clip((arr - threshold) / width, 0.0, 1.0)
    out = np.where(np.isnan(arr), np.nan, out)
    return float(out) if np.ndim(out) == 0 else out


def _opt(x: Activities | None) -> Labels:
    return () if x is None else as_labels(x)


def in_units(delta: pd.Series, unit: str) -> pd.Series:
    """Express a Series of time differences in ``unit`` (NaT → NaN)."""
    return delta.dt.total_seconds() / pd.Timedelta(f"1{_UNITS[unit]}").total_seconds()


def _plain(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def _num(name: str, value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise NormError(f"{name} must be a number, got {value!r}") from exc


def _check_nonneg(name: str, value: Any) -> None:
    if value is None or not np.isfinite(_num(name, value)) or _num(name, value) < 0:
        raise NormError(f"{name} must be a finite number >= 0, got {value!r}")


def _check_pos(name: str, value: Any) -> None:
    if value is None or not np.isfinite(_num(name, value)) or _num(name, value) <= 0:
        raise NormError(f"{name} must be a finite number > 0, got {value!r}")


@dataclass(frozen=True)
class Constraint:
    """Base class. Subclasses define ``type`` and their parameters."""

    type: ClassVar[str] = "abstract"
    _hidden: ClassVar[frozenset[str]] = frozenset()

    def params(self) -> dict[str, Any]:
        """Parameters as a JSON-friendly mapping (tuples become lists)."""
        return {f.name: _plain(getattr(self, f.name)) for f in fields(self) if f.name not in self._hidden}

    def activities(self) -> Labels:
        """All activity labels referenced by this constraint."""
        return ()

    def describe(self) -> str:
        return f"{self.type}({', '.join(f'{k}={v!r}' for k, v in self.params().items())})"


@dataclass(frozen=True)
class Presence(Constraint):
    """``(pres, a, m)``: activity ``a`` must occur at least ``m`` times.

    ``ν = 1 − min(cnt(a, σ) / m, 1)``; for ``m = 1`` an indicator.
    """

    type: ClassVar[str] = "presence"
    activity: Activities
    m: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "activity", as_labels(self.activity, what="activity"))
        if _num("presence: m", self.m) != int(_num("presence: m", self.m)) or self.m < 1:
            raise NormError(f"presence: m must be an integer >= 1, got {self.m!r}")
        object.__setattr__(self, "m", int(self.m))

    def activities(self) -> Labels:
        return as_labels(self.activity)


@dataclass(frozen=True)
class Exclusion(Constraint):
    """``(excl, a)``: activity ``a`` must not occur. ``ν = 1[cnt(a, σ) > 0]``.

    ``after`` / ``before`` restrict the count to events strictly after the
    first occurrence of ``after`` and/or strictly before the first occurrence
    of ``before`` (e.g. "no price change after the goods receipt"). When an
    anchor activity is absent from a case the constraint is not applicable
    to it.
    """

    type: ClassVar[str] = "exclusion"
    activity: Activities
    after: Activities | None = None
    before: Activities | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "activity", as_labels(self.activity, what="activity"))
        for f in ("after", "before"):
            v = getattr(self, f)
            object.__setattr__(self, f, None if v is None else as_labels(v, what=f))

    def activities(self) -> Labels:
        return as_labels(self.activity) + _opt(self.after) + _opt(self.before)


@dataclass(frozen=True)
class Singularity(Constraint):
    """``(sing, a, k, K)``: at most ``k`` occurrences tolerated.

    ``ν = min(max(0, cnt(a, σ) − k) / K, 1)``, i.e. ``sat(cnt; k, K)``.
    ``after`` / ``before`` scope the count as in :class:`Exclusion`.
    """

    type: ClassVar[str] = "singularity"
    activity: Activities
    k: int = 1
    K: float = 1.0
    after: Activities | None = None
    before: Activities | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "activity", as_labels(self.activity, what="activity"))
        if _num("singularity: k", self.k) != int(_num("singularity: k", self.k)) or self.k < 0:
            raise NormError(f"singularity: k must be an integer >= 0, got {self.k!r}")
        object.__setattr__(self, "k", int(self.k))
        _check_pos("singularity: K", self.K)
        object.__setattr__(self, "K", float(self.K))
        for f in ("after", "before"):
            v = getattr(self, f)
            object.__setattr__(self, f, None if v is None else as_labels(v, what=f))

    def activities(self) -> Labels:
        return as_labels(self.activity) + _opt(self.after) + _opt(self.before)


@dataclass(frozen=True)
class Lag(Constraint):
    """``(lag, a, b, δ, Δ)``: after ``a``, ``b`` should follow within ``delta``
    time units; the violation saturates at ``delta + width``.

    Paper semantics (the defaults): ``t_a = t1(a, σ)`` (first ``a``),
    ``t_b = min{ts(e) | act(e) = b, ts(e) ≥ t_a}`` (first ``b`` at or after
    it), ``ν = sat(t_b − t_a; δ, Δ)``.

    Parameters
    ----------
    delta, width, unit
        Threshold and saturation width in ``unit`` (``"D"`` days, ``"h"``,
        ``"min"``, ``"s"``, ``"ms"``, ``"W"``). ``delta=None`` removes the
        time bound: the constraint only requires that some ``b`` follows
        ``a``.
    missing_a
        ``"violate"`` (default): ν = 1 when ``a`` never occurs. ``"skip"``:
        the constraint is not applicable to such cases, the expectation
        being vacuous when its activation is absent.
    missing_b
        ``"violate"`` (default): ν = 1 when ``a`` occurs but no ``b`` follows
        it. ``"skip"``: not applicable, so that only a presence constraint on
        ``b`` accounts for the missing milestone. ``"censor"``: score the lag
        observed so far, ``sat(window_end − t_a; δ, Δ)``, treating an open
        item by its age.
    activation
        ``"first"`` (default): measure from the first ``a``. ``"last"``: from
        the last ``a``. ``"each"``: from every ``a`` to the next ``b``, then
        average the per-activation violations.
    response
        ``"first_after"`` (default): the first ``b`` at or after the
        activation. ``"first_overall"``: the first ``b`` in the case; if it
        precedes the activation the response counts as missing.
    """

    type: ClassVar[str] = "lag"
    _hidden: ClassVar[frozenset[str]] = frozenset({"missing"})
    a: Activities
    b: Activities
    delta: float | None = 0.0
    width: float = 0.0
    unit: str = "D"
    missing_a: Literal["violate", "skip"] = "violate"
    missing_b: Literal["violate", "skip", "censor"] = "violate"
    activation: Literal["first", "last", "each"] = "first"
    response: Literal["first_after", "first_overall"] = "first_after"
    missing: Literal["violate", "skip"] | None = None  # shorthand setting both endpoints

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", as_labels(self.a, what="a"))
        object.__setattr__(self, "b", as_labels(self.b, what="b"))
        if self.delta is not None:
            _check_nonneg("lag: delta", self.delta)
            object.__setattr__(self, "delta", float(self.delta))
        _check_nonneg("lag: width", self.width)
        object.__setattr__(self, "width", float(self.width))
        if self.unit not in _UNITS:
            raise NormError(f"lag: unit must be one of {sorted(set(_UNITS.values()))}, got {self.unit!r}")
        object.__setattr__(self, "unit", _UNITS[self.unit])
        if self.missing is not None:
            if self.missing not in ("violate", "skip"):
                raise NormError("lag: missing must be 'violate' or 'skip'")
            object.__setattr__(self, "missing_a", self.missing)
            object.__setattr__(self, "missing_b", self.missing)
            object.__setattr__(self, "missing", None)
        if self.missing_a not in ("skip", "violate"):
            raise NormError("lag: missing_a must be 'skip' or 'violate'")
        if self.missing_b not in ("violate", "skip", "censor"):
            raise NormError("lag: missing_b must be 'violate', 'skip' or 'censor'")
        if self.activation not in ("first", "last", "each"):
            raise NormError("lag: activation must be 'first', 'last' or 'each'")
        if self.response not in ("first_after", "first_overall"):
            raise NormError("lag: response must be 'first_after' or 'first_overall'")
        if self.activation == "each" and self.response != "first_after":
            raise NormError("lag: activation='each' requires response='first_after'")
        if self.missing_b == "censor" and self.delta is None:
            raise NormError("lag: missing_b='censor' needs a time bound (delta)")
        if self.missing_b == "censor" and self.response != "first_after":
            raise NormError("lag: missing_b='censor' requires response='first_after'")

    def activities(self) -> Labels:
        return as_labels(self.a) + as_labels(self.b)

    @property
    def timedelta_unit(self) -> pd.Timedelta:
        return pd.Timedelta(f"1{self.unit}")


@dataclass(frozen=True)
class Precedence(Constraint):
    """DECLARE precedence: ``b`` must not occur before the first ``a``.

    ``ν = sat(#{b events with ts < t1(a)}; k, K)`` — with the defaults one
    premature ``b`` is a full violation. Covers expectations such as "in a
    3-way match the invoice must not precede the goods receipt".

    ``missing_a="skip"`` (default) marks cases without ``a`` as not
    applicable, ``"violate"`` gives ν = 1. ``missing_b="satisfy"`` (default)
    treats cases without any ``b`` as satisfied (vacuous), ``"skip"`` as not
    applicable.
    """

    type: ClassVar[str] = "precedence"
    a: Activities
    b: Activities
    k: int = 0
    K: float = 1.0
    missing_a: Literal["skip", "violate"] = "skip"
    missing_b: Literal["satisfy", "skip"] = "satisfy"

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", as_labels(self.a, what="a"))
        object.__setattr__(self, "b", as_labels(self.b, what="b"))
        if _num("precedence: k", self.k) != int(_num("precedence: k", self.k)) or self.k < 0:
            raise NormError(f"precedence: k must be an integer >= 0, got {self.k!r}")
        object.__setattr__(self, "k", int(self.k))
        _check_pos("precedence: K", self.K)
        object.__setattr__(self, "K", float(self.K))
        if self.missing_a not in ("skip", "violate"):
            raise NormError("precedence: missing_a must be 'skip' or 'violate'")
        if self.missing_b not in ("satisfy", "skip"):
            raise NormError("precedence: missing_b must be 'satisfy' or 'skip'")

    def activities(self) -> Labels:
        return as_labels(self.a) + as_labels(self.b)


@dataclass(frozen=True)
class Balance(Constraint):
    """``(bal, α_x, A_x, α_y, A_y, τ, Γ)``: two case-level totals must agree.

    ``tot_x = agg(attr_x over events with act ∈ A_x)`` (likewise ``tot_y``),
    ``d = |tot_x − tot_y| / max(tot_x, tot_y, ε)``, ``ν = sat(d; τ, Γ)``.
    The paper assumes non-negative attribute values; scoring raises if a
    total is negative (net credit notes first, or take absolute values).

    ``agg`` is ``"sum"`` (paper), or ``"first"``, ``"last"``, ``"max"``,
    ``"min"``, ``"mean"`` for attributes that are already cumulative.
    Sums over empty sets are 0, so a missing side yields ``d = 1`` unless
    both sides are empty (then ``d = 0``).
    """

    type: ClassVar[str] = "balance"
    attr_x: str
    activities_x: Activities
    attr_y: str
    activities_y: Activities
    tau: float = 0.0
    width: float = 1.0
    eps: float = 1e-9
    agg: Literal["sum", "first", "last", "max", "min", "mean"] = "sum"

    def __post_init__(self) -> None:
        object.__setattr__(self, "activities_x", as_labels(self.activities_x, what="activities_x"))
        object.__setattr__(self, "activities_y", as_labels(self.activities_y, what="activities_y"))
        if not (0.0 <= float(self.tau) <= 1.0):
            raise NormError(f"balance: tau must be in [0, 1], got {self.tau!r}")
        object.__setattr__(self, "tau", float(self.tau))
        _check_nonneg("balance: width", self.width)
        object.__setattr__(self, "width", float(self.width))
        _check_pos("balance: eps", self.eps)
        if self.agg not in ("sum", "first", "last", "max", "min", "mean"):
            raise NormError(f"balance: unknown agg {self.agg!r}")

    def activities(self) -> Labels:
        return as_labels(self.activities_x) + as_labels(self.activities_y)


@dataclass(frozen=True)
class Metric(Constraint):
    """Extension: the saturation rule on a numeric case attribute.

    ``ν = sat(att(σ); ϑ, W)`` (``direction="high"``) or
    ``sat(ϑ − att(σ); 0, W)`` (``direction="low"``, penalising small values).
    Used for engineered signals such as manual-touch counts. A case whose
    attribute is missing is not evaluable (NaN).
    """

    type: ClassVar[str] = "metric"
    attribute: str
    threshold: float = 0.0
    width: float = 1.0
    direction: Literal["high", "low"] = "high"

    def __post_init__(self) -> None:
        if not self.attribute:
            raise NormError("metric: attribute must be given")
        if self.threshold is None or not np.isfinite(_num("metric: threshold", self.threshold)):
            raise NormError("metric: threshold must be finite")
        object.__setattr__(self, "threshold", float(self.threshold))
        _check_nonneg("metric: width", self.width)
        object.__setattr__(self, "width", float(self.width))
        if self.direction not in ("high", "low"):
            raise NormError("metric: direction must be 'high' or 'low'")


CONSTRAINT_TYPES: dict[str, type[Constraint]] = {
    cls.type: cls for cls in (Presence, Exclusion, Singularity, Lag, Precedence, Balance, Metric)
}

#: Short type names accepted in norm files (the paper's vocabulary).
TYPE_ALIASES: dict[str, str] = {
    "pres": "presence",
    "excl": "exclusion",
    "sing": "singularity",
    "bal": "balance",
    "order": "precedence",
}

_NUMERIC_PARAMS = {"m", "k", "K", "delta", "width", "tau", "eps", "threshold"}


def constraint_from_dict(type_name: str, params: Mapping[str, Any]) -> Constraint:
    """Build a constraint from its type name and a parameter mapping.

    Unknown types and unknown parameters raise :class:`NormError`; numeric
    parameters given as strings are coerced.
    """
    key = TYPE_ALIASES.get(type_name, type_name)
    if key not in CONSTRAINT_TYPES:
        raise NormError(f"unknown constraint type {type_name!r}; known: {sorted(CONSTRAINT_TYPES)}")
    cls = CONSTRAINT_TYPES[key]
    allowed = {f.name for f in fields(cls)}
    unknown = set(params) - allowed
    if unknown:
        raise NormError(f"{key}: unknown parameters {sorted(unknown)}; allowed {sorted(allowed)}")
    clean: dict[str, Any] = {}
    for k, v in params.items():
        if k in _NUMERIC_PARAMS and isinstance(v, str):
            try:
                v = float(v)
            except ValueError as exc:
                raise NormError(f"{key}: parameter {k}={v!r} is not numeric") from exc
        if k in ("m", "k") and v is not None:
            v = int(v)
        clean[k] = v
    return cls(**clean)
