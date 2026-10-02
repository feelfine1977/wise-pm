"""Row selection: one rule grammar, parsed once, evaluated three-valued over a table.

A rule is written in one of two forms. The short form maps attributes to
allowed values, combined with AND::

    {"flow_type": ["DF1", "DF2"], "document_type": ["Standard PO"]}

The rule form allows ``all`` / ``any`` / ``not`` combinations of leaves::

    {"all": [{"attr": "flow_type", "in": ["DF1"]},
             {"attr": "exposure", "gte": 1000},
             {"has": ["Record Goods Receipt"]},
             {"not": {"attr": "vendor", "in": ["V0"]}}]}

Leaf operators: ``in``, ``not_in``, ``eq``, ``ne``, ``gt``, ``gte``, ``lt``,
``lte``, ``isna``, ``notna``, ``regex`` on an attribute; ``has`` / ``lacks`` on
activity occurrence in the row's case. ``regex`` searches the value's text
anywhere (``re.search``), as the recipe where-clauses of 0.1.0 did.

:func:`parse` turns a rule into a :class:`Selection`: a small tree that states
the columns and activities it needs. :meth:`Selection.evaluate` evaluates the
tree over a table, given through one of two adapters: :class:`CaseTable` (one
row per case, attributes are case-table columns) and :class:`EventTable` (one
row per event, attributes are event columns). The result is a nullable boolean
Series: ``True`` in scope, ``False`` out of scope, ``<NA>`` unknown.

Two callers write rules in their own 0.1.0 shapes, and both are read into the
same tree. :func:`parse_recipe_where` reads the where-clause of a
derived-attribute recipe, ``{"column": c, op: v}`` with ``op`` one of ``in``,
``not_in``, ``regex``, ``eq`` (evaluated over the event table).
:func:`parse_where` reads the ``where=`` filter of results, drivers and
penalty mass, where a scalar value means equality and a list means
membership; :func:`select_cases` turns it into a plain case mask. Both also
accept the rule form.

Missing values
--------------
The missing-value policy is an explicit argument of every evaluation.

``"legacy"`` reproduces 0.1.0 operator by operator: a missing attribute is out
of scope under ``in``, ``eq``, ``gt``, ``gte``, ``lt``, ``lte`` and ``notna``
and in scope under ``not_in``, ``ne`` and ``isna``; ``not`` negates that. It
yields unknown only where a pandas comparison itself returns ``<NA>`` (a column
of a nullable dtype holding ``pd.NA`` under ``eq``, ``ne`` and their
negations), which 0.1.0 could not evaluate at all.

``"three_valued"`` (decision 0001, section 3): a row whose attribute is missing
is unknown under every attribute operator, and ``not`` keeps it unknown.
``all`` and ``any`` combine by Kleene logic: ``all`` is out of scope when one
part is out of scope, else unknown when one part is unknown; ``any`` is in
scope when one part is in scope, else unknown when one part is unknown.
``has`` / ``lacks`` observe activities, not attributes, and are never unknown.
"""

from __future__ import annotations

import copy
import re
import warnings
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias

import pandas as pd

from ._json import jsonable
from .constraints import as_labels
from .errors import NormError

if TYPE_CHECKING:  # pragma: no cover
    from .log import EventLog

MissingPolicy: TypeAlias = Literal["legacy", "three_valued"]
LEGACY: MissingPolicy = "legacy"
THREE_VALUED: MissingPolicy = "three_valued"
MISSING_POLICIES: tuple[MissingPolicy, ...] = (LEGACY, THREE_VALUED)

LEAF_OPERATORS = ("in", "not_in", "eq", "ne", "gt", "gte", "lt", "lte", "isna", "notna", "regex")
RECIPE_WHERE_OPERATORS = ("in", "not_in", "regex", "eq")
_RULE_KEYS = {"all", "any", "not", "attr", "has", "lacks"}


# ----------------------------------------------------------------------------- tree
@dataclass(frozen=True)
class Compare:
    """Leaf ``{"attr": attr, op: value}`` on one attribute."""

    attr: str
    op: str
    value: Any


@dataclass(frozen=True)
class Occurs:
    """Leaf ``{"has": activities}`` (``present``) or ``{"lacks": activities}``."""

    activities: tuple[str, ...]
    present: bool


@dataclass(frozen=True)
class Not:
    rule: Rule


@dataclass(frozen=True)
class AllOf:
    rules: tuple[Rule, ...]


@dataclass(frozen=True)
class AnyOf:
    rules: tuple[Rule, ...]


Rule: TypeAlias = Compare | Occurs | Not | AllOf | AnyOf


def _walk(rule: Rule) -> Iterator[Rule]:
    yield rule
    if isinstance(rule, Not):
        yield from _walk(rule.rule)
    elif isinstance(rule, AllOf | AnyOf):
        for r in rule.rules:
            yield from _walk(r)


# ----------------------------------------------------------------------------- tables
class Table(Protocol):
    """What a selection reads from the rows it evaluates."""

    @property
    def index(self) -> pd.Index: ...

    def column(self, name: str) -> pd.Series: ...

    def occurs(self, activities: tuple[str, ...]) -> pd.Series: ...


class CaseTable:
    """One row per case: attributes are columns of ``cases``; ``has`` / ``lacks`` need ``log``.

    With ``case_ids=True`` the name of the index reads the case ids when no column has that name.
    """

    def __init__(self, cases: pd.DataFrame, log: EventLog | None = None, *, case_ids: bool = False) -> None:
        self.cases = cases
        self.log = log
        self.case_ids = case_ids

    @property
    def index(self) -> pd.Index:
        return self.cases.index

    def column(self, name: str) -> pd.Series:
        if name not in self.cases.columns:
            if self.case_ids and name == self.cases.index.name:
                return pd.Series(self.cases.index, index=self.cases.index)
            raise NormError(f"selection refers to case attribute {name!r}, which is missing from the case table")
        return self.cases[name]

    def occurs(self, activities: tuple[str, ...]) -> pd.Series:
        if self.log is None:
            raise NormError("'has'/'lacks' selection needs the event log")
        return (self.log.count(activities) > 0).reindex(self.cases.index).fillna(False).astype(bool)


class EventTable:
    """One row per event of ``log``: attributes are event columns; ``has`` / ``lacks`` test the event's case."""

    def __init__(self, log: EventLog) -> None:
        self.log = log

    @property
    def index(self) -> pd.Index:
        return self.log.events.index

    def column(self, name: str) -> pd.Series:
        if name not in self.log.events.columns:
            raise NormError(f"selection refers to event column {name!r}, which is missing from the event table")
        return self.log.events[name]

    def occurs(self, activities: tuple[str, ...]) -> pd.Series:
        present = self.log.count(activities) > 0
        return self.log.events[self.log.case_col].isin(present.index[present.to_numpy()])


# ----------------------------------------------------------------------------- selection
@dataclass(frozen=True)
class Selection:
    """A parsed rule. ``rule is None`` selects every row; ``document`` is the normalised source."""

    rule: Rule | None
    document: dict[str, Any]

    @property
    def columns(self) -> tuple[str, ...]:
        """Attributes the rule reads, in order of first use."""
        if self.rule is None:
            return ()
        return tuple(dict.fromkeys(r.attr for r in _walk(self.rule) if isinstance(r, Compare)))

    @property
    def activities(self) -> tuple[str, ...]:
        """Activities whose occurrence the rule tests, in order of first use."""
        if self.rule is None:
            return ()
        return tuple(dict.fromkeys(a for r in _walk(self.rule) if isinstance(r, Occurs) for a in r.activities))

    def evaluate(self, table: Table, *, missing: MissingPolicy) -> pd.Series:
        """Nullable boolean Series over ``table.index``: in scope, out of scope, ``<NA>`` unknown."""
        if missing not in MISSING_POLICIES:
            raise NormError(f"missing-value policy must be one of {MISSING_POLICIES}, got {missing!r}")
        if self.rule is None:
            return pd.Series(True, index=table.index, dtype="boolean")
        return _evaluate(self.rule, table, missing)


def _evaluate(rule: Rule, table: Table, missing: MissingPolicy) -> pd.Series:
    if isinstance(rule, AllOf):
        out = pd.Series(True, index=table.index, dtype="boolean")
        for r in rule.rules:
            out &= _evaluate(r, table, missing)
        return out
    if isinstance(rule, AnyOf):
        out = pd.Series(False, index=table.index, dtype="boolean")
        for r in rule.rules:
            out |= _evaluate(r, table, missing)
        return out
    if isinstance(rule, Not):
        return ~_evaluate(rule.rule, table, missing)
    if isinstance(rule, Occurs):
        present = table.occurs(rule.activities)
        return (present if rule.present else ~present).astype("boolean")
    s = table.column(rule.attr)
    known = _compare(s, rule.op, rule.value).astype("boolean")
    if missing == THREE_VALUED:
        known = known.mask(s.isna().to_numpy(), pd.NA)
    return known


def _compare(s: pd.Series, op: str, v: Any) -> pd.Series:
    """The 0.1.0 leaf: a plain boolean Series, missing values resolved per operator."""
    if op == "in":
        return s.isin(list(v))
    if op == "not_in":
        return ~s.isin(list(v))
    out: pd.Series
    if op == "eq":
        out = s == v
        return out
    if op == "ne":
        out = s != v
        return out
    if op == "isna":
        return s.isna() if v else s.notna()
    if op == "notna":
        return s.notna() if v else s.isna()
    if op == "regex":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # patterns with groups are fine here
            return s.astype(str).str.contains(v, regex=True, na=False)
    num = pd.to_numeric(s, errors="coerce")
    compared: dict[str, pd.Series] = {"gt": num > v, "gte": num >= v, "lt": num < v, "lte": num <= v}
    return compared[op].fillna(False)


# ----------------------------------------------------------------------------- parsing
def parse(rule: Any, *, what: str = "selection") -> Selection:
    """Parse the short form or the rule form; raise :class:`NormError` naming ``what`` if malformed.

    ``None`` and ``{}`` select every row.
    """
    if rule is None:
        return Selection(None, {})
    if not isinstance(rule, Mapping):
        raise NormError(f"{what} must be a mapping, got {type(rule).__name__}")
    if not rule:
        return Selection(None, {})
    if set(rule) & _RULE_KEYS:
        return Selection(_parse_rule(rule, what), copy.deepcopy(dict(rule)))
    short: dict[str, Any] = {}
    leaves: list[Rule] = []
    for attr, values in rule.items():
        if isinstance(values, str | bytes) or not isinstance(values, Iterable):
            values = [values]
        short[str(attr)] = [jsonable(v) for v in values]
        if not short[str(attr)]:
            raise NormError(f"{what} for {attr!r} is an empty list (nothing would apply)")
        leaves.append(Compare(str(attr), "in", list(short[str(attr)])))
    return Selection(AllOf(tuple(leaves)), short)


def _parse_rule(rule: Any, what: str) -> Rule:
    if not isinstance(rule, Mapping):
        raise NormError(f"{what} rule must be a mapping, got {rule!r}")
    keys = set(rule)
    if "all" in keys or "any" in keys:
        if len(keys) != 1:
            raise NormError(f"{what} rule {dict(rule)}: 'all'/'any' must be the only key")
        items = rule["all"] if "all" in keys else rule["any"]
        if not isinstance(items, list) or not items:
            raise NormError(f"{what} 'all'/'any' must be a non-empty list of rules")
        parts = tuple(_parse_rule(r, what) for r in items)
        return AllOf(parts) if "all" in keys else AnyOf(parts)
    if "not" in keys:
        if len(keys) != 1:
            raise NormError(f"{what} 'not' must be the only key")
        return Not(_parse_rule(rule["not"], what))
    if "has" in keys or "lacks" in keys:
        if len(keys) != 1:
            raise NormError(f"{what} 'has'/'lacks' must be the only key")
        return Occurs(as_labels(rule["has"] if "has" in keys else rule["lacks"]), "has" in keys)
    if "attr" in keys:
        ops = keys - {"attr"}
        if len(ops) != 1 or not ops <= set(LEAF_OPERATORS):
            raise NormError(f"{what} leaf {dict(rule)} needs exactly one operator from {sorted(LEAF_OPERATORS)}")
        (op,) = ops
        if op == "regex":
            _check_pattern(rule[op], what)
        return Compare(str(rule["attr"]), op, copy.deepcopy(rule[op]))
    raise NormError(f"{what} rule {dict(rule)} is neither a short form nor a rule")


def _check_pattern(pattern: Any, what: str) -> None:
    if not isinstance(pattern, str):
        raise NormError(f"{what} regex must be a string, got {pattern!r}")
    try:
        re.compile(pattern)
    except re.error as e:
        raise NormError(f"{what} regex {pattern!r} is not a valid pattern: {e}") from None


def parse_recipe_where(where: Any, *, what: str = "recipe where-clause") -> Selection:
    """Parse the where-clause of a derived-attribute recipe.

    ``{"column": c, op: v}`` is the 0.1.0 clause: the first of ``in``, ``not_in``,
    ``regex``, ``eq`` present is the operator and other keys are ignored, so every
    clause 0.1.0 accepted selects the same rows. Without ``column`` the clause is
    read as :func:`parse` reads a rule. ``None`` and ``{}`` select every row.
    """
    if not where:
        return Selection(None, {})
    if not isinstance(where, Mapping) or "column" not in where:
        return parse(where, what=what)
    op = next((o for o in RECIPE_WHERE_OPERATORS if o in where), None)
    if op is None:
        raise NormError(f"{what} needs one of in/not_in/regex/eq: {dict(where)}")
    return Selection(_parse_rule({"attr": where["column"], op: where[op]}, what), copy.deepcopy(dict(where)))


def parse_where(where: Any, *, what: str = "where") -> Selection:
    """Parse the ``where=`` filter of results, drivers and penalty mass.

    A mapping without rule keys maps attributes to values, combined with AND: a
    scalar means equality (``eq``, as in 0.1.0) and a list, tuple or set means
    membership (``in``). Otherwise the mapping is read as :func:`parse` reads a
    rule. ``None`` and ``{}`` select every row.
    """
    if where is None:
        return Selection(None, {})
    if not isinstance(where, Mapping):
        raise NormError(f"{what} must be a mapping, a boolean Series or None, got {type(where).__name__}")
    if not where or set(where) & _RULE_KEYS:
        return parse(where, what=what)
    leaves: list[Rule] = []
    for attr, value in where.items():
        if isinstance(value, str | bytes) or not isinstance(value, Iterable):
            leaves.append(Compare(str(attr), "eq", copy.deepcopy(value)))
            continue
        values = list(value)
        if not values:
            raise NormError(f"{what} for {attr!r} is an empty list (nothing would be selected)")
        leaves.append(Compare(str(attr), "in", copy.deepcopy(values)))
    document = {"all": [{"attr": leaf.attr, leaf.op: jsonable(leaf.value)} for leaf in leaves if isinstance(leaf, Compare)]}
    return Selection(AllOf(tuple(leaves)), document)


def select_cases(cases: pd.DataFrame, where: Any, *, log: EventLog | None = None) -> pd.Series:
    """Plain boolean mask over ``cases.index`` for a ``where=`` filter.

    ``None`` selects every case. A boolean Series is aligned to the cases, and a
    case it does not list is not selected. A mapping is parsed with
    :func:`parse_where`; the name of the case index reads the case ids, and
    ``has`` / ``lacks`` need ``log``. It is evaluated under the legacy policy,
    and a row whose value is unknown is not selected.
    """
    if isinstance(where, pd.Series):
        return where.reindex(cases.index, fill_value=False).astype(bool)
    selection = parse_where(where)
    for name in selection.columns:
        if name not in cases.columns and name != cases.index.name:
            raise NormError(f"unknown case attribute {name!r} in where; known: {list(cases.columns)}")
    out = selection.evaluate(CaseTable(cases, log, case_ids=True), missing=LEGACY)
    return out.fillna(False).astype(bool)


__all__ = [
    "LEAF_OPERATORS",
    "LEGACY",
    "MISSING_POLICIES",
    "RECIPE_WHERE_OPERATORS",
    "THREE_VALUED",
    "AllOf",
    "AnyOf",
    "CaseTable",
    "Compare",
    "EventTable",
    "MissingPolicy",
    "Not",
    "Occurs",
    "Rule",
    "Selection",
    "Table",
    "parse",
    "parse_recipe_where",
    "parse_where",
    "select_cases",
]
