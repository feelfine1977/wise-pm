"""Derived-attribute recipes as typed records.

A norm lists the definitions of its engineered case attributes under
``derived_attributes``, each a mapping with a ``name`` and a ``kind``.
:func:`parse_recipe` reads such a mapping strictly into one record per kind: a
key the kind does not know, an option value it does not know, or a value of
the wrong type is refused with a :class:`~wise.errors.NormError` that names
the recipe and the key. A record states what it reads from a log
(:meth:`Recipe.requires`, the :class:`~wise.family.Inputs` of the constraint
families) and computes one value per case through the log's kernel
(:meth:`Recipe.compute`).

Kinds and their keys (``?`` marks an optional key):

``count`` — :class:`CountRecipe`
    ``activities``, ``after?``, ``before?``: occurrences per case, optionally
    restricted to events after the first ``after`` / before the first
    ``before`` (NaN where that anchor is absent).
``count_events`` — :class:`CountEventsRecipe`
    ``where``: events matching a where-clause.
``nunique`` — :class:`NuniqueRecipe`
    ``column``, ``where?``: distinct values of an event column per case.
``agg`` — :class:`AggRecipe`
    ``column``, ``agg``, ``where?``: an aggregate of a numeric event column;
    ``agg`` is one of :data:`AGGREGATIONS`.
``cv`` — :class:`CvRecipe`
    ``column``, ``eps?``: coefficient of variation ``std / max(|mean|, eps)``.
``lag`` — :class:`LagRecipe`
    ``a``, ``b``, ``unit?`` (``"D"``), ``activation?`` (``"first"`` or
    ``"last"``), ``response?`` (``"first_after"`` or ``"first_overall"``),
    ``allow_negative?`` (``false``): time from ``a`` to ``b``, NaN when
    undefined, paired as :class:`~wise.Lag` pairs (:func:`wise.families.lag.pairs`).
``ratio`` — :class:`RatioRecipe`
    ``numerator``, ``denominator``: a quotient of case attributes, NaN where
    the denominator is 0.
``quantile_scale`` — :class:`QuantileScaleRecipe`
    ``attribute``, ``q?`` (0.95), ``divisor?`` (``null``): the attribute divided by its
    ``q`` quantile over the scored cases, clipped to ``[0, 1]``. ``divisor``
    is carried with the recipe and not used.
``indicator_times`` — :class:`IndicatorTimesRecipe`
    ``activities``, ``attribute``: ``1[cnt > 0] · attribute``.
``column`` — :class:`ColumnRecipe`
    ``attribute``: an existing case-table column, as a number.
``eval`` (schema version 2)
    ``expr``: accepted only when the expression is a single case-table column
    name (``n_events`` or ```name with spaces```) and then read as that
    ``column`` recipe (decision 0001). Any other expression is refused; no
    expression is evaluated.

A where-clause is ``{"column": c, op: v}`` with ``op`` one of ``in``,
``not_in``, ``regex``, ``eq``, or a rule of the selection grammar over event
columns (:func:`wise.selection.parse_recipe_where`). It is evaluated over the
event table under the legacy missing-value policy.
"""

from __future__ import annotations

import keyword
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .constraints import TIME_UNITS, Labels
from .errors import LogSchemaError, NormError
from .families import lag as lag_family
from .family import Inputs
from .log_schema import ExposureAggregation
from .selection import LEGACY, EventTable, Selection, parse_recipe_where

if TYPE_CHECKING:  # pragma: no cover
    from .log import EventLog

#: The aggregations of an ``agg`` recipe: the pandas reductions that give one number per case.
AGGREGATIONS: tuple[str, ...] = tuple(a.value for a in ExposureAggregation)
ACTIVATIONS: tuple[str, ...] = ("first", "last")
RESPONSES: tuple[str, ...] = ("first_after", "first_overall")

_EVAL_COLUMN = re.compile(r"`([^`]+)`")


# ----------------------------------------------------------------------------- reading a document
class _Reader:
    """Reads the keys of one recipe document; every refusal names the recipe and the key."""

    def __init__(self, document: Mapping[str, Any], name: str, kind: str) -> None:
        self.document = document
        self.name = name
        self.kind = kind

    def refuse(self, key: str, problem: str) -> NormError:
        return NormError(f"derived attribute {self.name!r} ({self.kind}): {key} {problem}")

    def text(self, key: str) -> str:
        value = self.document[key]
        if not isinstance(value, str) or not value:
            raise self.refuse(key, f"must be a non-empty string, got {value!r}")
        return value

    def labels(self, key: str) -> Labels:
        value = self.document[key]
        if isinstance(value, str):
            return (value,)
        if isinstance(value, Iterable) and not isinstance(value, Mapping | bytes):
            out = tuple(value)
            if out and all(isinstance(v, str) for v in out):
                return cast(Labels, out)
        raise self.refuse(key, f"must be an activity label or a non-empty list of labels, got {value!r}")

    def optional_labels(self, key: str) -> Labels | None:
        return None if self.document.get(key) is None else self.labels(key)

    def number(self, key: str, default: float) -> float:
        if key not in self.document:
            return default
        value = self.document[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise self.refuse(key, f"must be a finite number, got {value!r}")
        return float(value)

    def flag(self, key: str, default: bool) -> bool:
        value = self.document.get(key, default)
        if not isinstance(value, bool):
            raise self.refuse(key, f"must be true or false, got {value!r}")
        return value

    def choice(self, key: str, options: tuple[str, ...], default: str | None = None) -> str:
        value = self.document.get(key, default)
        if value not in options:
            raise self.refuse(key, f"must be one of {list(options)}, got {value!r}")
        return cast(str, value)

    def where(self, key: str) -> Selection:
        return parse_recipe_where(self.document.get(key), what=f"derived attribute {self.name!r} ({self.kind}): {key}")


# ----------------------------------------------------------------------------- records
@dataclass(frozen=True)
class Recipe(ABC):
    """A derived case attribute: its name and the definition of its values."""

    name: str

    kind: ClassVar[str]
    #: The keys a document of this kind must give, and the ones it may give.
    required: ClassVar[tuple[str, ...]]
    optional: ClassVar[tuple[str, ...]] = ()

    @classmethod
    @abstractmethod
    def _read(cls, doc: _Reader) -> Recipe:
        """Build the record from the keys of a document (presence of the required keys is checked)."""

    @abstractmethod
    def requires(self) -> Inputs:
        """What the recipe reads from a log: activities, case attributes, event columns."""

    @abstractmethod
    def compute(self, log: EventLog) -> pd.Series:
        """One value per case of ``log``, indexed by case id."""

    @abstractmethod
    def _fields(self) -> dict[str, Any]: ...

    def to_dict(self) -> dict[str, Any]:
        """The canonical document of the record (labels as lists, every option written)."""
        return {"name": self.name, "kind": self.kind, **self._fields()}

    def _undefined(self, problem: str) -> NormError:
        return NormError(f"derived attribute {self.name!r} ({self.kind}): {problem}")


def _per_case(log: EventLog, values: NDArray[Any]) -> pd.Series:
    return pd.Series(values, index=log.case_ids)


def _mask(log: EventLog, where: Selection) -> NDArray[np.bool_]:
    if where.rule is None:
        return np.ones(len(log.events), dtype=bool)
    columns = log.events.columns
    for col in where.columns:
        if col not in columns:
            raise LogSchemaError(f"recipe where-clause refers to unknown event column {col!r}")
    mask = where.evaluate(EventTable(log), missing=LEGACY)
    return mask.fillna(False).to_numpy(dtype=bool)


def _event_column(recipe: Recipe, log: EventLog, column: str) -> pd.Series:
    events = log.events
    if column not in events.columns:
        raise LogSchemaError(f"recipe {recipe.name!r}: unknown event column {column!r}")
    return events[column]


def _where_document(where: Selection) -> dict[str, Any] | None:
    return None if where.rule is None else where.document


@dataclass(frozen=True)
class CountRecipe(Recipe):
    activities: Labels
    after: Labels | None = None
    before: Labels | None = None

    kind: ClassVar[str] = "count"
    required: ClassVar[tuple[str, ...]] = ("activities",)
    optional: ClassVar[tuple[str, ...]] = ("after", "before")

    @classmethod
    def _read(cls, doc: _Reader) -> Recipe:
        return cls(doc.name, doc.labels("activities"), doc.optional_labels("after"), doc.optional_labels("before"))

    def requires(self) -> Inputs:
        return Inputs(activities=self.activities + (self.after or ()) + (self.before or ()))

    def compute(self, log: EventLog) -> pd.Series:
        return log.count_scoped(self.activities, after=self.after, before=self.before)

    def _fields(self) -> dict[str, Any]:
        return {
            "activities": list(self.activities),
            "after": None if self.after is None else list(self.after),
            "before": None if self.before is None else list(self.before),
        }


@dataclass(frozen=True)
class CountEventsRecipe(Recipe):
    where: Selection

    kind: ClassVar[str] = "count_events"
    required: ClassVar[tuple[str, ...]] = ("where",)

    @classmethod
    def _read(cls, doc: _Reader) -> Recipe:
        return cls(doc.name, doc.where("where"))

    def requires(self) -> Inputs:
        return Inputs(activities=self.where.activities, event_columns=self.where.columns)

    def compute(self, log: EventLog) -> pd.Series:
        return _per_case(log, log.case_index.count(_mask(log, self.where)).astype(float))

    def _fields(self) -> dict[str, Any]:
        return {"where": _where_document(self.where)}


@dataclass(frozen=True)
class NuniqueRecipe(Recipe):
    column: str
    where: Selection

    kind: ClassVar[str] = "nunique"
    required: ClassVar[tuple[str, ...]] = ("column",)
    optional: ClassVar[tuple[str, ...]] = ("where",)

    @classmethod
    def _read(cls, doc: _Reader) -> Recipe:
        return cls(doc.name, doc.text("column"), doc.where("where"))

    def requires(self) -> Inputs:
        return Inputs(activities=self.where.activities, event_columns=tuple(dict.fromkeys((self.column, *self.where.columns))))

    def compute(self, log: EventLog) -> pd.Series:
        m = _mask(log, self.where)
        values = _event_column(self, log, self.column)
        return _per_case(log, log.case_index.aggregate(values, m, how="nunique", empty=0).astype(float))

    def _fields(self) -> dict[str, Any]:
        return {"column": self.column, "where": _where_document(self.where)}


@dataclass(frozen=True)
class AggRecipe(Recipe):
    column: str
    agg: str
    where: Selection

    kind: ClassVar[str] = "agg"
    required: ClassVar[tuple[str, ...]] = ("column", "agg")
    optional: ClassVar[tuple[str, ...]] = ("where",)

    @classmethod
    def _read(cls, doc: _Reader) -> Recipe:
        return cls(doc.name, doc.text("column"), doc.choice("agg", AGGREGATIONS), doc.where("where"))

    def requires(self) -> Inputs:
        return Inputs(activities=self.where.activities, event_columns=tuple(dict.fromkeys((self.column, *self.where.columns))))

    def compute(self, log: EventLog) -> pd.Series:
        idx = log.case_index
        m = _mask(log, self.where)
        vals = pd.to_numeric(_event_column(self, log, self.column), errors="coerce")
        # pandas, not the index's "sum", so that a sum keeps pandas' compensated arithmetic
        s = vals[m].groupby(idx.case_codes[m]).agg(self.agg)
        return _per_case(log, s.reindex(range(idx.n_cases)).to_numpy(dtype=float))

    def _fields(self) -> dict[str, Any]:
        return {"column": self.column, "agg": self.agg, "where": _where_document(self.where)}


@dataclass(frozen=True)
class CvRecipe(Recipe):
    column: str
    eps: float = 1e-9

    kind: ClassVar[str] = "cv"
    required: ClassVar[tuple[str, ...]] = ("column",)
    optional: ClassVar[tuple[str, ...]] = ("eps",)

    @classmethod
    def _read(cls, doc: _Reader) -> Recipe:
        return cls(doc.name, doc.text("column"), doc.number("eps", 1e-9))

    def requires(self) -> Inputs:
        return Inputs(event_columns=(self.column,))

    def compute(self, log: EventLog) -> pd.Series:
        idx = log.case_index
        vals = pd.to_numeric(_event_column(self, log, self.column), errors="coerce")
        std = pd.Series(idx.aggregate(vals, how="std")).fillna(0.0)
        mean = pd.Series(idx.aggregate(vals, how="mean"))
        return _per_case(log, (std / mean.abs().clip(lower=self.eps)).to_numpy(dtype=float))

    def _fields(self) -> dict[str, Any]:
        return {"column": self.column, "eps": self.eps}


@dataclass(frozen=True)
class LagRecipe(Recipe):
    a: Labels
    b: Labels
    unit: str = "D"
    activation: Literal["first", "last"] = "first"
    response: Literal["first_after", "first_overall"] = "first_after"
    allow_negative: bool = False

    kind: ClassVar[str] = "lag"
    required: ClassVar[tuple[str, ...]] = ("a", "b")
    optional: ClassVar[tuple[str, ...]] = ("unit", "activation", "response", "allow_negative")

    @classmethod
    def _read(cls, doc: _Reader) -> Recipe:
        return cls(
            doc.name,
            doc.labels("a"),
            doc.labels("b"),
            unit=doc.choice("unit", tuple(TIME_UNITS), "D"),
            activation=cast(Literal["first", "last"], doc.choice("activation", ACTIVATIONS, "first")),
            response=cast(Literal["first_after", "first_overall"], doc.choice("response", RESPONSES, "first_after")),
            allow_negative=doc.flag("allow_negative", False),
        )

    def requires(self) -> Inputs:
        return Inputs(activities=self.a + self.b)

    def compute(self, log: EventLog) -> pd.Series:
        pairing = lag_family.pairs(
            log,
            self.a,
            self.b,
            activation=self.activation,
            response=self.response,
            unit=self.unit,
            negative=self.allow_negative,
        )
        return _per_case(log, pairing.per_case())

    def _fields(self) -> dict[str, Any]:
        return {
            "a": list(self.a),
            "b": list(self.b),
            "unit": self.unit,
            "activation": self.activation,
            "response": self.response,
            "allow_negative": self.allow_negative,
        }


@dataclass(frozen=True)
class RatioRecipe(Recipe):
    numerator: str
    denominator: str

    kind: ClassVar[str] = "ratio"
    required: ClassVar[tuple[str, ...]] = ("numerator", "denominator")

    @classmethod
    def _read(cls, doc: _Reader) -> Recipe:
        return cls(doc.name, doc.text("numerator"), doc.text("denominator"))

    def requires(self) -> Inputs:
        return Inputs(case_attributes=tuple(dict.fromkeys((self.numerator, self.denominator))))

    def compute(self, log: EventLog) -> pd.Series:
        num = log.attribute(self.numerator)
        den = log.attribute(self.denominator)
        return (num / den.replace(0, np.nan)).astype(float)

    def _fields(self) -> dict[str, Any]:
        return {"numerator": self.numerator, "denominator": self.denominator}


@dataclass(frozen=True)
class QuantileScaleRecipe(Recipe):
    attribute: str
    q: float = 0.95
    divisor: float | None = None

    kind: ClassVar[str] = "quantile_scale"
    required: ClassVar[tuple[str, ...]] = ("attribute",)
    optional: ClassVar[tuple[str, ...]] = ("q", "divisor")

    @classmethod
    def _read(cls, doc: _Reader) -> Recipe:
        q = doc.number("q", 0.95)
        if not 0.0 <= q <= 1.0:
            raise doc.refuse("q", f"must be between 0 and 1, got {q!r}")
        divisor = None if doc.document.get("divisor") is None else doc.number("divisor", math.nan)
        if divisor is not None and divisor <= 0:
            raise doc.refuse("divisor", f"must be > 0, got {divisor!r}")
        return cls(doc.name, doc.text("attribute"), q, divisor)

    def requires(self) -> Inputs:
        return Inputs(case_attributes=(self.attribute,))

    def compute(self, log: EventLog) -> pd.Series:
        x = log.attribute(self.attribute)
        q = float(x.quantile(self.q))
        if not np.isfinite(q) or q <= 0:
            q = 1.0
        return (x / q).clip(0.0, 1.0)

    def _fields(self) -> dict[str, Any]:
        return {"attribute": self.attribute, "q": self.q, "divisor": self.divisor}


@dataclass(frozen=True)
class IndicatorTimesRecipe(Recipe):
    activities: Labels
    attribute: str

    kind: ClassVar[str] = "indicator_times"
    required: ClassVar[tuple[str, ...]] = ("activities", "attribute")

    @classmethod
    def _read(cls, doc: _Reader) -> Recipe:
        return cls(doc.name, doc.labels("activities"), doc.text("attribute"))

    def requires(self) -> Inputs:
        return Inputs(activities=self.activities, case_attributes=(self.attribute,))

    def compute(self, log: EventLog) -> pd.Series:
        ind = (log.count(self.activities) > 0).astype(float)
        return ind * log.attribute(self.attribute).fillna(0.0)

    def _fields(self) -> dict[str, Any]:
        return {"activities": list(self.activities), "attribute": self.attribute}


@dataclass(frozen=True)
class ColumnRecipe(Recipe):
    """An existing case-table column, as a number (non-numeric values become NaN)."""

    attribute: str

    kind: ClassVar[str] = "column"
    required: ClassVar[tuple[str, ...]] = ("attribute",)

    @classmethod
    def _read(cls, doc: _Reader) -> Recipe:
        return cls(doc.name, doc.text("attribute"))

    @classmethod
    def _read_eval(cls, doc: _Reader) -> ColumnRecipe:
        expr = doc.document["expr"]
        column = _single_column(expr) if isinstance(expr, str) else None
        if column is None:
            raise doc.refuse(
                "expr",
                f"{expr!r} is not a single case-table column; an eval recipe may only name one column "
                "(decision 0001), and no expression is evaluated",
            )
        return cls(doc.name, column)

    def requires(self) -> Inputs:
        return Inputs(case_attributes=(self.attribute,))

    def compute(self, log: EventLog) -> pd.Series:
        try:
            return log.attribute(self.attribute)
        except LogSchemaError:
            raise self._undefined(f"{self.attribute!r} is not a column of the case table") from None

    def _fields(self) -> dict[str, Any]:
        return {"attribute": self.attribute}


def _single_column(expr: str) -> str | None:
    """The column an eval expression names when it is a bare column name, else ``None``."""
    text = expr.strip()
    quoted = _EVAL_COLUMN.fullmatch(text)
    if quoted:
        return quoted.group(1)
    if text.isidentifier() and not keyword.iskeyword(text):
        return text
    return None


#: Record class per kind of a document.
RECIPES: dict[str, type[Recipe]] = {
    cls.kind: cls
    for cls in (
        CountRecipe,
        CountEventsRecipe,
        NuniqueRecipe,
        AggRecipe,
        CvRecipe,
        LagRecipe,
        RatioRecipe,
        QuantileScaleRecipe,
        IndicatorTimesRecipe,
        ColumnRecipe,
    )
}
#: Every kind a document may name; ``eval`` is read as a ``column`` recipe.
KINDS: tuple[str, ...] = (*RECIPES, "eval")
_EVAL_KEYS = ("expr",)


def parse_recipe(document: Mapping[str, Any]) -> Recipe:
    """Read one recipe document into its typed record; refuse it with :class:`NormError` if malformed."""
    if not isinstance(document, Mapping):
        raise NormError(f"a derived attribute recipe must be a mapping, got {type(document).__name__}")
    name = document.get("name")
    if name is None or name == "":
        raise NormError(f"derived attribute recipe without name: {dict(document)}")
    if not isinstance(name, str):
        raise NormError(f"derived attribute recipe name must be a string, got {name!r}")
    kind = document.get("kind")
    if kind not in KINDS:
        raise NormError(f"derived attribute {name!r}: unknown kind {kind!r}; known {KINDS}")
    cls = RECIPES["column" if kind == "eval" else kind]
    required = _EVAL_KEYS if kind == "eval" else cls.required
    optional = () if kind == "eval" else cls.optional
    missing = [k for k in required if k not in document]
    if missing:
        raise NormError(f"derived attribute {name!r} ({kind}): missing {missing}")
    unknown = [k for k in document if k not in ("name", "kind", *required, *optional)]
    if unknown:
        raise NormError(
            f"derived attribute {name!r} ({kind}): unknown key {unknown[0]!r}; allowed {sorted((*required, *optional))}"
        )
    doc = _Reader(document, name, kind)
    return ColumnRecipe._read_eval(doc) if kind == "eval" else cls._read(doc)


def as_recipe(recipe: Recipe | Mapping[str, Any]) -> Recipe:
    """The record itself, or the record a document is parsed into."""
    return recipe if isinstance(recipe, Recipe) else parse_recipe(recipe)


__all__ = [
    "ACTIVATIONS",
    "AGGREGATIONS",
    "KINDS",
    "RECIPES",
    "RESPONSES",
    "AggRecipe",
    "ColumnRecipe",
    "CountEventsRecipe",
    "CountRecipe",
    "CvRecipe",
    "IndicatorTimesRecipe",
    "LagRecipe",
    "NuniqueRecipe",
    "QuantileScaleRecipe",
    "RatioRecipe",
    "Recipe",
    "as_recipe",
    "parse_recipe",
]
