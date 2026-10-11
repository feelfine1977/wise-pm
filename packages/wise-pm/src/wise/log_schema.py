"""The log schema and the parse step in front of the event log (paper Sec. IV-A).

Building a log is separate from querying it. A :class:`LogSchema` names the
columns of an event frame by their role and carries every policy of building
a log as an enumerated value with a documented default; no policy is inferred
from the data. :func:`parse` turns an event frame and a schema into a
:class:`ParsedLog`: the events in index order, the immutable
:class:`~wise.case_index.CaseIndex` and the case table.
:class:`~wise.log.EventLog` is the façade over both.

Input that cannot form a log is refused with a :class:`~wise.errors.LogSchemaError`
that names the problem: a missing column, a frame without rows, a null case id,
a null lifecycle transition, unparseable timestamps under the ``"raise"``
policy, negative exposure, a log whose filters leave no event, and case
attributes that vary within a case under the ``"strict"`` policy.

Two rules of the method's input are stated here once, as policies: what an
event without a timestamp contributes to each primitive
(:class:`UntimedEvents`; the table is in ``docs/semantics/log-policies.md``),
and how one value per case is taken from a case attribute recorded on events
(:class:`CaseAttributeValues`).
"""

from __future__ import annotations

import datetime
import decimal
import fractions
import gc
import hashlib
import json
import uuid
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, fields
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal, TypeVar, cast

import numpy as np
import pandas as pd

from ._frozen import FrozenDict, FrozenList
from .case_index import CaseIndex, event_order
from .constraints import as_list
from .errors import LogSchemaError

CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
TIMESTAMP_COL = "time:timestamp"

_RESERVED = {"n_events", "first_ts", "last_ts", "exposure", "score"}
_DIGEST_FORMAT = "wise.LogSchema/1"


def check_attribute_name(name: str) -> None:
    """Refuse a case attribute name that collides with a computed column."""
    if name in _RESERVED or name.startswith(("contrib__", "score__")):
        raise LogSchemaError(
            f"{name!r} is reserved for computed columns; rename the attribute "
            f"(reserved: {sorted(_RESERVED)}, prefixes 'contrib__' and 'score__')"
        )


# ------------------------------------------------------------------ policies
class MissingTimestamps(str, Enum):
    """What to do with events whose timestamp is null or unparseable."""

    RAISE = "raise"
    """Refuse the log (default)."""
    DROP = "drop"
    """Remove those events; a case that loses all its events is not a case."""
    KEEP = "keep"
    """Keep them without a timestamp; what they contribute to each primitive
    is the rule of :class:`UntimedEvents`."""


class UntimedEvents(str, Enum):
    """What an event kept without a timestamp contributes to the primitives.

    Under both values such an event never defines a time: it is ignored by
    first and last timestamps, lags, time-scoped counts (``count_scoped`` with
    ``after`` or ``before``, ``count_before``, ``count_after``), the case's
    ``first_ts``/``last_ts`` and the observation window; it never ties with
    another event and sorts last in its case. It is counted by
    ``EventLog.count`` (and so by presence, exclusion and singularity without
    a scope), by totals and aggregates, ``n_events`` and exposure. The values
    differ in one primitive only, ``EventLog.count_scoped`` without a scope,
    which the ``count`` recipe uses.
    """

    LEGACY = "legacy"
    """The 0.1.0 rule (default): ``count_scoped`` ignores untimed events even
    without a scope, so a ``count`` recipe can disagree with a presence
    constraint on the same activity."""
    UNIFORM = "uniform"
    """One rule for every primitive: an untimed event counts wherever no time
    is compared, so ``count_scoped`` without a scope equals ``count``."""


class Duplicates(str, Enum):
    """What to do with events equal in case, activity and timestamp."""

    KEEP = "keep"
    """Keep every event (default)."""
    DROP = "drop"
    """Keep the first of each group of equal events."""


class TimeZone(str, Enum):
    """The time zone of the parsed timestamps."""

    AS_PARSED = "as_parsed"
    """Naive timestamps stay naive, aware ones keep their zone (default);
    strings with mixed offsets are refused."""
    UTC = "utc"
    """Convert every timestamp to UTC; naive ones are read as UTC."""


class DateOrder(str, Enum):
    """How ambiguous date strings such as ``05/01/2024`` are read."""

    MONTH_FIRST = "month_first"
    """Month before day (default; pandas ``dayfirst=False``)."""
    DAY_FIRST = "day_first"
    """Day before month (pandas ``dayfirst=True``)."""


class ExposureAggregation(str, Enum):
    """How the exposure column is aggregated per case (a pandas aggregation).

    ``MAX`` (default) suits cumulative amounts, ``SUM`` per-event amounts and
    ``FIRST`` constant attributes.
    """

    MAX = "max"
    SUM = "sum"
    FIRST = "first"
    LAST = "last"
    MIN = "min"
    MEAN = "mean"
    MEDIAN = "median"
    PROD = "prod"
    STD = "std"
    VAR = "var"
    SEM = "sem"
    SKEW = "skew"
    COUNT = "count"
    NUNIQUE = "nunique"
    SIZE = "size"
    ANY = "any"
    ALL = "all"


class CaseAttributeValues(str, Enum):
    """How the value of a case attribute is taken from the events of a case."""

    FIRST_NON_NULL = "first_non_null"
    """The first non-null value in event order (default). The attributes that
    vary within a case are recorded with their number of cases
    (:attr:`ParsedLog.varying_case_attributes`) and reported by
    :meth:`EventLog.validate <wise.log.EventLog.validate>`."""
    STRICT = "strict"
    """The single non-null value of the case; a log in which an attribute
    takes two values within one case is refused with the attributes named."""


_E = TypeVar("_E", bound=Enum)


def _policy(name: str, enum: type[_E], value: Any) -> _E:
    if isinstance(value, enum):
        return value
    try:
        return enum(value)
    except ValueError:
        allowed = [m.value for m in enum]
        raise LogSchemaError(f"{name} must be one of {allowed}, got {value!r}") from None


# ------------------------------------------------------------------ schema
@dataclass(frozen=True, init=False)
class LogSchema:
    """The column roles and policies that turn an event frame into a log.

    Column roles
        ``case_col``, ``activity_col``, ``timestamp_col`` (pm4py names by
        default); ``case_attributes`` (case-level columns, in case-table
        order); ``exposure_col`` (numeric business exposure ``exp(σ) ≥ 0``);
        ``order_col`` (orders events sharing a timestamp; without it the input
        row order does); ``event_id_col`` (identifies the source event for the
        replication diagnostics); ``lifecycle_col`` with ``keep_transitions``
        (the transitions kept, compared case-insensitively, default
        ``("complete",)``; a null transition is refused); ``keep_columns``
        (restricts the event table to these columns plus the roles above;
        ``None`` keeps every column); ``timestamp_format`` (a
        :func:`pandas.to_datetime` format for string timestamps; ``None``
        lets pandas read each value).

    Policies, each an enumerated value
        ``missing_timestamps`` (:class:`MissingTimestamps`, default ``'raise'``);
        ``duplicates`` (:class:`Duplicates`, default ``'keep'``);
        ``time_zone`` (:class:`TimeZone`, default ``'as_parsed'``);
        ``date_order`` (:class:`DateOrder`, default ``'month_first'``);
        ``exposure_agg`` (:class:`ExposureAggregation`, default ``'max'``);
        ``case_attribute_values`` (:class:`CaseAttributeValues`, default ``'first_non_null'``);
        ``untimed_events`` (:class:`UntimedEvents`, default ``'legacy'``).

    A policy may be given as its enum member or its value. Column lists accept
    a bare string; transitions are stored lower-cased, sorted and without
    repeats, and the kept columns sorted, so two schemas that build the same
    log are equal. :attr:`digest` identifies the schema.
    """

    case_col: str = CASE_COL
    activity_col: str = ACTIVITY_COL
    timestamp_col: str = TIMESTAMP_COL
    case_attributes: tuple[str, ...] = ()
    exposure_col: str | None = None
    order_col: str | None = None
    event_id_col: str | None = None
    lifecycle_col: str | None = None
    keep_transitions: tuple[str, ...] = ("complete",)
    keep_columns: tuple[str, ...] | None = None
    timestamp_format: str | None = None
    missing_timestamps: MissingTimestamps = MissingTimestamps.RAISE
    duplicates: Duplicates = Duplicates.KEEP
    time_zone: TimeZone = TimeZone.AS_PARSED
    date_order: DateOrder = DateOrder.MONTH_FIRST
    exposure_agg: ExposureAggregation = ExposureAggregation.MAX
    case_attribute_values: CaseAttributeValues = CaseAttributeValues.FIRST_NON_NULL
    untimed_events: UntimedEvents = UntimedEvents.LEGACY

    def __init__(
        self,
        *,
        case_col: str = CASE_COL,
        activity_col: str = ACTIVITY_COL,
        timestamp_col: str = TIMESTAMP_COL,
        case_attributes: str | Iterable[str] = (),
        exposure_col: str | None = None,
        order_col: str | None = None,
        event_id_col: str | None = None,
        lifecycle_col: str | None = None,
        keep_transitions: str | Iterable[str] = ("complete",),
        keep_columns: str | Iterable[str] | None = None,
        timestamp_format: str | None = None,
        missing_timestamps: MissingTimestamps | Literal["raise", "drop", "keep"] = MissingTimestamps.RAISE,
        duplicates: Duplicates | Literal["keep", "drop"] = Duplicates.KEEP,
        time_zone: TimeZone | Literal["as_parsed", "utc"] = TimeZone.AS_PARSED,
        date_order: DateOrder | Literal["month_first", "day_first"] = DateOrder.MONTH_FIRST,
        exposure_agg: ExposureAggregation | str = ExposureAggregation.MAX,
        case_attribute_values: CaseAttributeValues | Literal["first_non_null", "strict"] = CaseAttributeValues.FIRST_NON_NULL,
        untimed_events: UntimedEvents | Literal["legacy", "uniform"] = UntimedEvents.LEGACY,
    ) -> None:
        attributes = tuple(as_list(case_attributes))
        for attr in attributes:
            check_attribute_name(attr)
        set_ = object.__setattr__
        set_(self, "case_col", case_col)
        set_(self, "activity_col", activity_col)
        set_(self, "timestamp_col", timestamp_col)
        set_(self, "case_attributes", attributes)
        set_(self, "exposure_col", exposure_col)
        set_(self, "order_col", order_col)
        set_(self, "event_id_col", event_id_col)
        set_(self, "lifecycle_col", lifecycle_col)
        set_(self, "keep_transitions", tuple(sorted({str(t).lower() for t in as_list(keep_transitions)})))
        set_(self, "keep_columns", None if keep_columns is None else tuple(sorted(set(as_list(keep_columns)), key=str)))
        set_(self, "timestamp_format", timestamp_format)
        set_(self, "missing_timestamps", _policy("missing_timestamps", MissingTimestamps, missing_timestamps))
        set_(self, "duplicates", _policy("duplicates", Duplicates, duplicates))
        set_(self, "time_zone", _policy("time_zone", TimeZone, time_zone))
        set_(self, "date_order", _policy("date_order", DateOrder, date_order))
        set_(self, "exposure_agg", _policy("exposure_agg", ExposureAggregation, exposure_agg))
        set_(self, "case_attribute_values", _policy("case_attribute_values", CaseAttributeValues, case_attribute_values))
        set_(self, "untimed_events", _policy("untimed_events", UntimedEvents, untimed_events))

    @classmethod
    def from_options(
        cls,
        *,
        case_col: str = CASE_COL,
        activity_col: str = ACTIVITY_COL,
        timestamp_col: str = TIMESTAMP_COL,
        case_attributes: Iterable[str] = (),
        exposure_col: str | None = None,
        exposure_agg: str = "max",
        order_col: str | None = None,
        event_id_col: str | None = None,
        lifecycle_col: str | None = None,
        keep_transitions: Iterable[str] = ("complete",),
        utc: bool = False,
        timestamp_format: str | None = None,
        dayfirst: bool = False,
        missing_timestamps: Literal["raise", "drop", "keep"] = "raise",
        keep_columns: Iterable[str] | None = None,
        dedupe: bool = False,
    ) -> LogSchema:
        """The schema for the column and policy options of :class:`~wise.log.EventLog`:
        ``utc`` selects :attr:`TimeZone.UTC`, ``dayfirst`` :attr:`DateOrder.DAY_FIRST`
        and ``dedupe`` :attr:`Duplicates.DROP`; the other options map by name."""
        return cls(
            case_col=case_col,
            activity_col=activity_col,
            timestamp_col=timestamp_col,
            case_attributes=case_attributes,
            exposure_col=exposure_col,
            order_col=order_col,
            event_id_col=event_id_col,
            lifecycle_col=lifecycle_col,
            keep_transitions=keep_transitions,
            keep_columns=keep_columns,
            timestamp_format=timestamp_format,
            missing_timestamps=missing_timestamps,
            duplicates=Duplicates.DROP if dedupe else Duplicates.KEEP,
            time_zone=TimeZone.UTC if utc else TimeZone.AS_PARSED,
            date_order=DateOrder.DAY_FIRST if dayfirst else DateOrder.MONTH_FIRST,
            exposure_agg=exposure_agg,
        )

    @property
    def roles(self) -> tuple[str, ...]:
        """The columns an event frame must have: every named role."""
        named = (self.exposure_col, self.order_col, self.event_id_col, self.lifecycle_col)
        return (self.case_col, self.activity_col, self.timestamp_col, *self.case_attributes, *(c for c in named if c is not None))

    @property
    def digest(self) -> str:
        """SHA-256 over the column roles and policies; equal for equal schemas
        and different as soon as one role or policy differs."""
        payload: dict[str, Any] = {"format": _DIGEST_FORMAT}
        for f in fields(self):
            value = getattr(self, f.name)
            payload[f.name] = value.value if isinstance(value, Enum) else value
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ sealed values
# Values a cell keeps as they are: none of them can be changed in place.
_IMMUTABLE = (
    str,
    bytes,
    bool,
    int,
    float,
    complex,
    decimal.Decimal,
    fractions.Fraction,
    datetime.date,
    datetime.time,
    datetime.timedelta,
    uuid.UUID,
    Enum,
    np.number,
    np.bool_,
    np.datetime64,
    np.timedelta64,
    np.str_,
    np.bytes_,
    pd.Period,
    pd.Interval,
    type(None),
    type(pd.NaT),
    type(pd.NA),
)
# What pandas infers for an object column all of whose values are of one immutable kind; such a column is not
# looked at cell by cell (``mixed`` and the other answers can hide a container).
_PLAIN_KINDS = frozenset(
    {
        "empty",
        "string",
        "bytes",
        "integer",
        "floating",
        "mixed-integer-float",
        "decimal",
        "complex",
        "boolean",
        "datetime64",
        "datetime",
        "date",
        "timedelta64",
        "timedelta",
        "time",
        "period",
    }
)


# The exact types most cells hold, checked before anything else.
_PLAIN = frozenset({str, int, float, bool, type(None)})


class _Unsealable(Exception):
    def __init__(self, value: object) -> None:
        self.value = value


def _seal(value: Any) -> Any:
    """``value`` as an independent copy that cannot be changed in place; mappings become
    :class:`~wise._frozen.FrozenDict`, lists :class:`~wise._frozen.FrozenList`, sets ``frozenset``, tuples
    tuples of sealed items and numeric arrays (not their subclasses) read-only arrays over immutable bytes. Raises
    :class:`_Unsealable` for any other value that is not immutable."""
    kind = type(value)
    if kind in _PLAIN:
        return value
    if kind is dict or kind is FrozenDict:
        return FrozenDict({k if type(k) in _PLAIN else _seal(k): v if type(v) in _PLAIN else _seal(v) for k, v in value.items()})
    if kind is list or kind is FrozenList:
        return FrozenList([v if type(v) in _PLAIN else _seal(v) for v in value])
    if kind is tuple:
        return tuple(v if type(v) in _PLAIN else _seal(v) for v in value)
    if isinstance(value, _IMMUTABLE):
        return value
    if isinstance(value, Mapping):
        return FrozenDict({_seal(k): _seal(v) for k, v in value.items()})
    if isinstance(value, list):
        return FrozenList([_seal(v) for v in value])
    if isinstance(value, set | frozenset):
        return frozenset(_seal(v) for v in value)
    if type(value) is np.ndarray and value.dtype.kind in "biufcmMSU":  # a subclass (masked, matrix) would lose its kind
        # backed by a bytes object, so that not even ``flags.writeable = True`` makes it writable again
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    raise _Unsealable(value)


def _hashable(value: Any) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _is_nested(value: Any) -> bool:
    if type(value) is tuple:
        return any(_is_nested(v) for v in value)
    return isinstance(value, Mapping | list | set | frozenset | np.ndarray)


def nested_cells(column: pd.Series) -> bool:
    """Whether a cell of ``column`` holds a mapping, list, set or array, at any depth of a tuple."""
    if column.dtype != object or pd.api.types.infer_dtype(column, skipna=True) in _PLAIN_KINDS:
        return False
    return any(_is_nested(v) for v in column.to_numpy())


def sealed(column: pd.Series, where: Callable[[int], str] | None = None) -> pd.Series:
    """``column`` with every cell sealed (``column`` itself when no cell holds a container).

    An object column whose cells are of one immutable kind is returned as it is without looking at the cells.
    A value that is neither immutable nor a mapping, list, tuple, set or numeric array, at any depth, is
    refused with a :class:`~wise.errors.LogSchemaError` naming the column and where the value is: ``where``
    of its position, by default its row label.
    """
    if column.dtype != object or pd.api.types.infer_dtype(column, skipna=True) in _PLAIN_KINDS:
        return column
    values = column.to_numpy()
    out = np.empty(len(values), dtype=object)
    changed = False
    collecting = gc.isenabled()
    gc.disable()  # the copies hold no cycles; scanning millions of new containers would double the time
    try:
        for i, value in enumerate(values):
            try:
                out[i] = _seal(value)
            except (_Unsealable, RecursionError) as exc:
                at = where(i) if where is not None else f"row {column.index[i]!r}"
                what = (
                    f"a value of type {type(exc.value).__name__!r}"
                    if isinstance(exc, _Unsealable)
                    else "a value that contains itself or is nested too deeply"
                )
                raise LogSchemaError(
                    f"column {column.name!r} holds {what} in {at} that the log cannot keep unchanged; convert it to "
                    "a plain value, a dict, list, tuple, set or numeric array, or leave the column out with "
                    "keep_columns"
                ) from None
            changed = changed or out[i] is not value
    finally:
        if collecting:
            gc.enable()
    return pd.Series(out, index=column.index, name=column.name) if changed else column


def _canonical(value: Any) -> str:
    # of a sealed value (see ``_seal``); ``repr`` keeps 1, 1.0, True, "1" and None apart, and quotes the separators a string may hold
    kind = type(value)
    if kind in _PLAIN:
        return repr(value)
    if kind is FrozenDict or kind is dict:
        items = [
            (repr(k) if type(k) in _PLAIN else _canonical(k)) + ":" + (repr(v) if type(v) in _PLAIN else _canonical(v))
            for k, v in value.items()
        ]
        items.sort()
        return "{" + ",".join(items) + "}"
    if kind is FrozenList or kind is list:
        return "[" + ",".join([repr(v) if type(v) in _PLAIN else _canonical(v) for v in value]) + "]"
    if kind is tuple:
        return "(" + ",".join([repr(v) if type(v) in _PLAIN else _canonical(v) for v in value]) + ")"
    if isinstance(value, frozenset):
        return "<" + ",".join(sorted(_canonical(v) for v in value)) + ">"
    if isinstance(value, np.ndarray):
        return f"array({value.dtype.str},{value.shape},{value.tobytes().hex()})"
    if isinstance(value, np.generic):
        return _canonical(value.item())
    return repr(value)


def canonical(column: pd.Series) -> pd.Series:
    """The text of each cell of a column with nested values, the same for equal values whatever the order of
    the keys of a mapping or the items of a set; ``1``, ``1.0``, ``True`` and ``"1"`` stay apart."""
    return pd.Series([_canonical(v) for v in column.to_numpy()], index=column.index, dtype=object)


# ------------------------------------------------------------------ parse
@dataclass(frozen=True)
class ParsedLog:
    """What :func:`parse` builds: the events sorted by ``(case, timestamp, order)``,
    the case index over them, and the case table (``n_events``, ``first_ts``,
    ``last_ts``, the case attributes and ``exposure``) indexed by case id.
    ``varying_case_attributes`` maps each case attribute that takes more than
    one non-null value within a case to the number of such cases (read-only,
    in case-table order)."""

    schema: LogSchema
    events: pd.DataFrame
    index: CaseIndex
    cases: pd.DataFrame
    n_missing_timestamps: int
    n_null_activities: int
    varying_case_attributes: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}), kw_only=True)

    @property
    def tz(self) -> Any:
        """Time zone of the timestamps, ``None`` when they are naive."""
        return self.index.tz


def _refuse_null_transitions(events: pd.DataFrame, column: str) -> None:
    null = events[column].isna().to_numpy()
    n = int(null.sum())
    if n:
        first = int(np.flatnonzero(null)[0])
        raise LogSchemaError(
            f"lifecycle column {column!r} has {n} null transition{'s' if n > 1 else ''} (first at row {first}); "
            "fill or drop them first, or leave lifecycle_col unset"
        )


def _parse_timestamps(raw: pd.Series, schema: LogSchema) -> pd.Series:
    utc = schema.time_zone is TimeZone.UTC
    if pd.api.types.is_datetime64_any_dtype(raw):
        if not utc:
            return raw
        return cast("pd.Series", raw.dt.tz_convert("UTC") if raw.dt.tz is not None else raw.dt.tz_localize("UTC"))
    dayfirst = schema.date_order is DateOrder.DAY_FIRST
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", UserWarning)
        ts = pd.to_datetime(raw, errors="coerce", utc=utc, format=schema.timestamp_format, dayfirst=dayfirst)
        retry = ts.isna() & raw.notna()
        if retry.any() and pd.api.types.is_datetime64_any_dtype(ts):
            again = pd.to_datetime(raw[retry], errors="coerce", utc=utc, format="mixed", dayfirst=dayfirst)
            if pd.api.types.is_datetime64_any_dtype(again) and (again.dt.tz is None) == (ts.dt.tz is None):
                ts = ts.copy()
                ts[retry] = again
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise LogSchemaError(f"timestamp column {schema.timestamp_col!r} has mixed time zones or types; pass utc=True")
    return cast("pd.Series", ts)


def parse(events: pd.DataFrame, schema: LogSchema) -> ParsedLog:
    """Build the events, the case index and the case table of ``events`` under ``schema``.

    The lifecycle filter and the duplicate policy apply first, then case ids
    are factorised (sorted), timestamps parsed under the time-zone, date-order
    and missing-timestamp policies, and the events sorted by
    ``(case, timestamp, order_col or input row)``. ``events`` is not modified.

    Raises :class:`~wise.errors.LogSchemaError` for a missing column, a frame
    without rows, a null lifecycle transition, a null case id, a log whose
    lifecycle or missing-timestamp filter removes every event, null or
    unparseable timestamps under ``missing_timestamps='raise'``, mixed time
    zones without ``time_zone='utc'``, negative case exposure and, under
    ``case_attribute_values='strict'``, case attributes that vary within a case.
    """
    required = list(schema.roles)
    missing = [c for c in required if c not in events.columns]
    if missing:
        raise LogSchemaError(
            f"event log is missing columns {missing}; available: {list(events.columns)[:30]}"
            + (" ..." if events.shape[1] > 30 else "")
        )
    if len(events) == 0:
        raise LogSchemaError("the event frame has no rows; an event log needs at least one event")
    case_col, activity_col, timestamp_col = schema.case_col, schema.activity_col, schema.timestamp_col

    df = events
    if schema.lifecycle_col is not None:
        _refuse_null_transitions(df, schema.lifecycle_col)
        keep = set(schema.keep_transitions)
        df = df[df[schema.lifecycle_col].astype(str).str.lower().isin(keep)]
        if len(df) == 0:
            found = sorted({str(t).lower() for t in events[schema.lifecycle_col].unique()[:10]})
            raise LogSchemaError(
                f"no event of lifecycle column {schema.lifecycle_col!r} has a transition in {list(schema.keep_transitions)} "
                f"(found: {found}); set keep_transitions to the transitions the log records"
            )
    if schema.duplicates is Duplicates.DROP:
        df = df.drop_duplicates(subset=[case_col, activity_col, timestamp_col])

    # ---- case ids -------------------------------------------------------------------
    case_series = df[case_col]
    n_null_case = int(case_series.isna().sum())
    if n_null_case:
        raise LogSchemaError(f"{n_null_case} events have a null case id; drop or fill them first")
    if isinstance(case_series.dtype, pd.CategoricalDtype):
        case_series = case_series.astype(case_series.cat.categories.dtype)
    codes, uniques = pd.factorize(case_series, sort=True)
    codes = codes.astype(np.int64)

    # ---- timestamps -----------------------------------------------------------------
    ts_raw = df[timestamp_col]
    ts = _parse_timestamps(ts_raw, schema)
    n_missing_ts = int(ts.isna().sum())
    if n_missing_ts and schema.missing_timestamps is MissingTimestamps.RAISE:
        first_bad = ts_raw[ts.isna()].iloc[0]
        raise LogSchemaError(
            f"{n_missing_ts} timestamps are null or unparseable (first: {first_bad!r}); "
            "pass missing_timestamps='drop' or 'keep', or utc=True for mixed offsets"
        )
    if n_missing_ts and schema.missing_timestamps is MissingTimestamps.DROP:
        if n_missing_ts == len(ts):
            raise LogSchemaError(
                f"every event has a null or unparseable timestamp in column {timestamp_col!r}; "
                "missing_timestamps='drop' would leave no event"
            )
        keep_mask = ts.notna().to_numpy()
        df, ts, codes = df[keep_mask], ts[keep_mask], codes[keep_mask]
        codes, uniques = pd.factorize(pd.Series(uniques[codes]), sort=True)
        codes = codes.astype(np.int64)
        n_missing_ts = 0
    tz = ts.dt.tz
    ts_int = (
        ts.to_numpy("datetime64[ns]").view("int64")
        if tz is None
        else ts.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]").view("int64")
    )

    # ---- order: (case, timestamp, tie-break) ------------------------------------------
    tie = pd.factorize(df[schema.order_col], sort=True)[0] if schema.order_col is not None else None
    order = event_order(codes, ts_int, tie)
    n_cases = len(uniques)

    cols = list(df.columns)
    if schema.keep_columns is not None:
        wanted = set(required) | set(schema.keep_columns)
        cols = [c for c in cols if c in wanted]
    ev = df.iloc[order][cols].reset_index(drop=True)
    ev[timestamp_col] = ts.to_numpy()[order] if tz is None else ts.iloc[order].reset_index(drop=True)
    for name in ev.columns:
        kept = ev[name]
        column = sealed(kept, lambda i: f"case {uniques[codes[order[i]]]!r}")
        if column is not kept:
            if name in schema.case_attributes and not all(_hashable(v) for v in column.to_numpy()):
                raise LogSchemaError(
                    f"case attribute {name!r} holds dicts, lists or arrays; a case attribute takes one plain "
                    "value per case, so flatten it into plain columns first"
                )
            ev[name] = column

    act = ev[activity_col]
    n_null_activities = int(act.isna().sum())
    if isinstance(act.dtype, pd.CategoricalDtype):
        act_codes = act.cat.codes.to_numpy().astype(np.int64)
        act_labels = pd.Index(act.cat.categories.astype(str))
    else:
        act_codes, act_uniques = pd.factorize(act, sort=False)
        act_codes = act_codes.astype(np.int64)
        act_labels = pd.Index(pd.Index(act_uniques).astype(str))
    sorted_codes, sorted_ts = codes[order], ts_int[order]
    sorted_codes.flags.writeable = sorted_ts.flags.writeable = False  # the index keeps views of them
    index = CaseIndex(sorted_codes, sorted_ts, act_codes, act_labels, n_cases=n_cases, tz=tz)

    # ---- case table -----------------------------------------------------------------
    cases = pd.DataFrame(index=pd.Index(uniques, name=case_col))
    cases["n_events"] = index.count()
    cases["first_ts"] = index.to_datetime(index.first())
    cases["last_ts"] = index.to_datetime(index.last())
    varying: dict[str, int] = {}
    for attr in schema.case_attributes:
        by_case = ev[attr].groupby(index.case_codes)
        n_varying = int((by_case.nunique(dropna=True) > 1).sum())
        if n_varying:
            varying[attr] = n_varying
        cases[attr] = by_case.first().reindex(range(n_cases)).to_numpy()
    if varying and schema.case_attribute_values is CaseAttributeValues.STRICT:
        named = ", ".join(f"{attr!r} ({n} case{'s' if n > 1 else ''})" for attr, n in varying.items())
        raise LogSchemaError(
            f"case attributes vary within a case: {named}; clean them first, or set "
            "case_attribute_values='first_non_null' to take the first non-null value"
        )
    if schema.exposure_col is not None:
        vals = pd.to_numeric(ev[schema.exposure_col], errors="coerce")
        per_case = vals.groupby(index.case_codes).agg(schema.exposure_agg.value)
        exp = per_case.reindex(range(n_cases)).fillna(0.0).to_numpy()
        if (exp < 0).any():
            raise LogSchemaError(
                f"exposure column {schema.exposure_col!r} yields negative case exposure for "
                f"{int((exp < 0).sum())} cases; the paper assumes exp(σ) >= 0"
            )
        cases["exposure"] = exp.astype(float)
    return ParsedLog(schema, ev, index, cases, n_missing_ts, n_null_activities, varying_case_attributes=MappingProxyType(varying))
