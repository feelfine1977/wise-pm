"""The constraint-family interface and the registry that maps type names to families.

A *family* is one kind of constraint (``presence``, ``lag``, ...). It knows
three things about its constraints, and nothing outside its module needs to
know them:

* how to **parse** the parameters of a norm document into a constraint
  (:meth:`Family.parse`); the constraint's constructor refuses what is not
  valid, so the same rule holds for a document and for Python code;
* which **inputs** a constraint needs from a log (:class:`Inputs`: the
  activities, the case attributes and the event columns it reads), which the
  norm's readiness check asks for;
* how to **evaluate** a constraint on a log to one value per case with the
  reason for it (:class:`Evaluation`).

The registry holds one family per type name (and its short aliases). Adding
a family takes one module that defines its parameter class and its
:class:`Family`, and one call to :func:`register`; scoring and the norm find
it here. The built-in families live in :mod:`wise.families`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import pandas as pd

from .errors import NormError

if TYPE_CHECKING:  # pragma: no cover
    from .constraints import Constraint
    from .log import EventLog

C = TypeVar("C", bound="Constraint")

#: The case was evaluated; its value is a number in ``[0, 1]``.
EVALUATED = "evaluated"
#: The constraint does not apply to the case (``σ ∉ C_app``).
OUT_OF_SCOPE = "out_of_scope"
#: The case is in scope but has no value, for a reason the family does not name.
NOT_EVALUABLE = "not_evaluable"
#: An anchor activity is absent: ``after``/``before`` of Exclusion and Singularity, ``a`` of Precedence.
MISSING_ANCHOR = "missing_anchor"
#: The constrained activity is absent and the constraint skips such cases (``b`` of Precedence).
MISSING_ACTIVITY = "missing_activity"
#: The case attribute a constraint reads is missing or not numeric (Metric).
MISSING_ATTRIBUTE = "missing_attribute"

#: Every reason an :class:`Evaluation` can carry.
REASONS: tuple[str, ...] = (EVALUATED, OUT_OF_SCOPE, NOT_EVALUABLE, MISSING_ANCHOR, MISSING_ACTIVITY, MISSING_ATTRIBUTE)


@dataclass(frozen=True)
class Inputs:
    """What a constraint reads from a log."""

    activities: tuple[str, ...] = ()
    case_attributes: tuple[str, ...] = ()
    event_columns: tuple[str, ...] = ()


@dataclass(frozen=True, eq=False)
class Evaluation:
    """One constraint evaluated on every case of a log.

    ``value`` is ``ν_c(σ)`` as a float Series over the case ids, NaN where the
    case has no value. ``reason`` says why, per case: :data:`EVALUATED` where
    there is a value, :data:`OUT_OF_SCOPE` where the constraint does not
    apply, otherwise the reason the case is not evaluable (one of
    :data:`REASONS`). A value is present exactly where the reason is
    :data:`EVALUATED`.
    """

    value: pd.Series
    reason: pd.Series

    def __post_init__(self) -> None:
        if not self.value.index.equals(self.reason.index):
            raise NormError("an evaluation's values and reasons must share one index")
        if not np.array_equal(self.value.notna().to_numpy(), (self.reason == EVALUATED).to_numpy()):
            raise NormError(f"an evaluation has a value exactly where its reason is {EVALUATED!r}")

    @classmethod
    def _consistent(cls, value: pd.Series, reason: pd.Series) -> Evaluation:
        # built from an evaluation that passed the checks, by an operation that keeps them
        ev = object.__new__(cls)
        object.__setattr__(ev, "value", value)
        object.__setattr__(ev, "reason", reason)
        return ev

    @classmethod
    def of(
        cls,
        values: Any,
        index: pd.Index,
        *,
        why: Iterable[tuple[Any, str]] = (),
    ) -> Evaluation:
        """Build an evaluation from per-case values in case order.

        Missing values get :data:`NOT_EVALUABLE` unless a ``(mask, reason)``
        pair of ``why`` covers the case; the first pair that covers it wins.
        The reasons are a categorical Series over :data:`REASONS` (and any
        further reason a family names).
        """
        value = pd.Series(np.asarray(values, dtype=float), index=index)
        missing = np.isnan(value.to_numpy())
        categories = list(REASONS)
        codes = np.zeros(len(value), dtype=np.int8)
        codes[missing] = categories.index(NOT_EVALUABLE)
        open_ = missing.copy()
        for mask, code in why:
            if code not in categories:
                categories.append(code)
            hit = open_ & np.asarray(mask, dtype=bool)
            codes[hit] = categories.index(code)
            open_ &= ~hit
        reason = pd.Series(pd.Categorical.from_codes(codes, categories=pd.Index(categories)), index=index)
        return cls._consistent(value, reason)

    @property
    def in_scope(self) -> pd.Series:
        """Boolean per case: the constraint applies (``σ ∈ C_app``)."""
        return self.reason != OUT_OF_SCOPE

    @property
    def evaluated(self) -> pd.Series:
        """Boolean per case: the case has a value."""
        return self.value.notna()

    def within(self, scope: pd.Series) -> Evaluation:
        """The evaluation restricted to the cases where ``scope`` is true; the others are out of scope."""
        return self._consistent(self.value.where(scope, np.nan), self.reason.where(scope, OUT_OF_SCOPE))

    def rename(self, name: Any) -> Evaluation:
        """The same evaluation with both Series named ``name``."""
        return self._consistent(self.value.rename(name), self.reason.rename(name))


@dataclass(frozen=True, eq=False)
class Family(Generic[C]):
    """One constraint family.

    Parameters
    ----------
    name
        The type name in norm documents; equal to ``constraint.type``.
    constraint
        The frozen parameter class; its constructor validates the parameters.
    requires
        ``requires(constraint) -> Inputs``.
    evaluate
        ``evaluate(constraint, log) -> Evaluation`` over every case of the
        log, without applicability (scoring adds that).
    numbers
        Parameters that a document may give as numeric strings; they are
        converted to ``float`` before the constructor sees them.
    aliases
        Further type names accepted in documents.
    """

    name: str
    constraint: type[C]
    requires: Callable[[C], Inputs]
    evaluate: Callable[[C, EventLog], Evaluation]
    numbers: frozenset[str] = frozenset()
    aliases: tuple[str, ...] = ()

    def parse(self, params: Mapping[str, Any]) -> C:
        """Build a constraint from the parameter mapping of a norm document.

        Unknown parameters and non-numeric strings for numeric parameters
        raise :class:`NormError`; every other check is the constructor's.
        """
        allowed = {f.name for f in fields(self.constraint)}
        unknown = set(params) - allowed
        if unknown:
            raise NormError(f"{self.name}: unknown parameters {sorted(unknown)}; allowed {sorted(allowed)}")
        clean: dict[str, Any] = {}
        for k, v in params.items():
            if k in self.numbers and isinstance(v, str):
                try:
                    v = float(v)
                except ValueError as exc:
                    raise NormError(f"{self.name}: parameter {k}={v!r} is not numeric") from exc
            clean[k] = v
        build: Callable[..., C] = self.constraint
        return build(**clean)


# ------------------------------------------------------------------ registry
_FAMILIES: dict[str, Family[Any]] = {}
#: Type name → parameter class of every registered family, in registration order.
CONSTRAINT_TYPES: dict[str, type[Constraint]] = {}
#: Short type name → type name, for every registered alias.
TYPE_ALIASES: dict[str, str] = {}


def register(family: Family[Any]) -> Family[Any]:
    """Add a family to the registry; its name and aliases must be new."""
    if family.constraint.type != family.name:
        raise NormError(f"family {family.name!r}: its constraint class has type {family.constraint.type!r}")
    for key in (family.name, *family.aliases):
        if key in _FAMILIES or key in TYPE_ALIASES:
            raise NormError(f"constraint type {key!r} is already registered")
    _FAMILIES[family.name] = family
    CONSTRAINT_TYPES[family.name] = family.constraint
    for alias in family.aliases:
        TYPE_ALIASES[alias] = family.name
    return family


def unregister(name: str) -> Family[Any]:
    """Remove a family (by type name) and its aliases from the registry."""
    fam = family(name)
    del _FAMILIES[fam.name]
    del CONSTRAINT_TYPES[fam.name]
    for alias in fam.aliases:
        del TYPE_ALIASES[alias]
    return fam


def family(type_name: str) -> Family[Any]:
    """The family registered under a type name or alias."""
    key = TYPE_ALIASES.get(type_name, type_name)
    try:
        return _FAMILIES[key]
    except KeyError:
        raise NormError(f"unknown constraint type {type_name!r}; known: {sorted(CONSTRAINT_TYPES)}") from None


def family_of(constraint: Constraint) -> Family[Any]:
    """The family a constraint belongs to."""
    fam = _FAMILIES.get(constraint.type)
    if fam is None or not isinstance(constraint, fam.constraint):
        raise NormError(f"{type(constraint).__name__} does not belong to a registered constraint family")
    return fam


def families() -> tuple[Family[Any], ...]:
    """Every registered family, in registration order."""
    return tuple(_FAMILIES.values())


__all__ = [
    "CONSTRAINT_TYPES",
    "EVALUATED",
    "MISSING_ACTIVITY",
    "MISSING_ANCHOR",
    "MISSING_ATTRIBUTE",
    "NOT_EVALUABLE",
    "OUT_OF_SCOPE",
    "REASONS",
    "TYPE_ALIASES",
    "Evaluation",
    "Family",
    "Inputs",
    "families",
    "family",
    "family_of",
    "register",
    "unregister",
]
