"""The case index: the immutable per-case structure behind the event log (paper Sec. IV-A).

A :class:`CaseIndex` holds one entry per event, sorted by ``(case, timestamp,
tie-break)``: the integer case code, the timestamp as int64 nanoseconds since
the epoch in UTC (:data:`NAT` for an event without a timestamp, sorted last
within its case), and the activity code. Every primitive the constraint
semantics are written in is a numpy operation on those arrays:

* counts for an activity set or an event mask    → :meth:`CaseIndex.count`
* first and last timestamps                      → :meth:`CaseIndex.first`, :meth:`CaseIndex.last`
* the per-case aggregate of an event column      → :meth:`CaseIndex.aggregate`
* runs of equal timestamps                       → :meth:`CaseIndex.run_starts`, :meth:`CaseIndex.tied`
* timestamp conversion                           → :meth:`CaseIndex.to_timestamp`, :meth:`CaseIndex.to_ns`,
  :meth:`CaseIndex.to_datetime`

The arrays are read-only and the object cannot be rebound; results are fresh
arrays indexed by case code. Events without a timestamp count, but never
define a first or last timestamp and never tie.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .errors import LogSchemaError

NAT: Final[int] = int(np.iinfo(np.int64).min)
"""Timestamp of an event without one (the int64 view of ``NaT``)."""

_LATEST: Final[int] = int(np.iinfo(np.int64).max)

Codes = NDArray[np.int64]
Mask = NDArray[np.bool_]
Select = str | Iterable[str] | Mask | pd.Series | None
"""An activity label, an iterable of labels, a boolean mask over the events (array or
Series, by position), or ``None`` for every event."""


def event_order(case_codes: ArrayLike, timestamps: ArrayLike, tie: ArrayLike | None = None) -> NDArray[np.intp]:
    """Positions that sort events by case, then timestamp (missing last), then ``tie``
    (default: the input order)."""
    codes = np.asarray(case_codes, dtype=np.int64)
    ts = np.asarray(timestamps, dtype=np.int64)
    sort_ts = np.where(ts == NAT, _LATEST, ts)
    tie_key = np.arange(len(codes)) if tie is None else np.asarray(tie)
    order: NDArray[np.intp] = np.lexsort((tie_key, sort_ts, codes))
    return order


def _frozen(arr: NDArray[Any]) -> NDArray[Any]:
    """A read-only view of an array the caller owns."""
    view = arr.view()
    view.flags.writeable = False
    return view


def _sealed(arr: NDArray[Any]) -> NDArray[Any]:
    """An array the index built itself, made read-only in place so no writable alias remains."""
    arr.flags.writeable = False
    return arr


class CaseIndex:
    """Immutable per-case index over events sorted by ``(case, timestamp)``.

    Parameters
    ----------
    case_codes
        Case code per event in ``[0, n_cases)``, non-decreasing.
    timestamps
        int64 nanoseconds since the epoch in UTC, :data:`NAT` where an event
        has none; non-decreasing within a case with missing ones last.
    activity_codes
        Position of the event's label in ``activity_labels``; negative for an
        event without a label.
    activity_labels
        The activity vocabulary.
    n_cases
        Number of cases (default: the largest case code plus one).
    tz
        Time zone of the source timestamps, used by the conversions.

    The arrays are taken over without a copy: the index keeps read-only views
    of them, so the caller must not modify them afterwards (or passes them
    read-only). The arrays the index derives are read-only themselves.
    """

    __slots__ = ("_act_codes", "_act_labels", "_codes", "_sizes", "_starts", "_ts", "_tz", "_valid")

    _codes: Codes
    _ts: Codes
    _valid: Mask
    _act_codes: Codes
    _act_labels: pd.Index
    _starts: Codes
    _sizes: Codes
    _tz: Any

    def __init__(
        self,
        case_codes: ArrayLike,
        timestamps: ArrayLike,
        activity_codes: ArrayLike,
        activity_labels: Iterable[str] | pd.Index,
        *,
        n_cases: int | None = None,
        tz: Any = None,
    ) -> None:
        codes = np.asarray(case_codes, dtype=np.int64)
        ts = np.asarray(timestamps, dtype=np.int64)
        acts = np.asarray(activity_codes, dtype=np.int64)
        labels = (
            activity_labels
            if isinstance(activity_labels, pd.Index)
            else pd.Index([str(v) for v in activity_labels], dtype=object)
        )
        if not (codes.ndim == ts.ndim == acts.ndim == 1 and len(codes) == len(ts) == len(acts)):
            raise LogSchemaError("case codes, timestamps and activity codes must be one-dimensional and of the same length")
        n = int(codes.max()) + 1 if n_cases is None and len(codes) else int(n_cases or 0)
        if n < 0:
            raise LogSchemaError(f"the number of cases must not be negative, got {n}")
        if len(codes) and (codes[0] < 0 or codes[-1] >= n):
            raise LogSchemaError(f"case codes must lie in [0, {n})")
        same_case = codes[1:] == codes[:-1]
        if not (same_case | (codes[1:] > codes[:-1])).all():
            raise LogSchemaError("events must be sorted by case")
        valid = ts != NAT
        sort_ts = np.where(valid, ts, _LATEST)
        if (same_case & (sort_ts[1:] < sort_ts[:-1])).any():
            raise LogSchemaError("events must be sorted by timestamp within a case, events without a timestamp last")
        if len(acts) and acts.max() >= len(labels):
            raise LogSchemaError(f"activity codes must lie below the number of activity labels ({len(labels)})")
        set_ = object.__setattr__
        set_(self, "_codes", _frozen(codes))
        set_(self, "_ts", _frozen(ts))
        set_(self, "_valid", _sealed(valid))
        set_(self, "_act_codes", _sealed(np.where(acts < 0, len(labels), acts)))
        set_(self, "_act_labels", labels)
        set_(self, "_starts", _sealed(np.searchsorted(codes, np.arange(n)).astype(np.int64)))
        set_(self, "_sizes", _sealed(np.bincount(codes, minlength=n).astype(np.int64)))
        set_(self, "_tz", tz)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"CaseIndex is immutable; cannot set {name!r}")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"CaseIndex is immutable; cannot delete {name!r}")

    def __repr__(self) -> str:
        return f"CaseIndex({self.n_events:,} events, {self.n_cases:,} cases, {len(self._act_labels)} activities)"

    # ------------------------------------------------------------------ structure
    @property
    def n_cases(self) -> int:
        return len(self._starts)

    @property
    def n_events(self) -> int:
        return len(self._codes)

    @property
    def case_codes(self) -> Codes:
        """Case code per event (read-only)."""
        return self._codes

    @property
    def timestamps(self) -> Codes:
        """int64 nanoseconds in UTC per event, :data:`NAT` where missing (read-only)."""
        return self._ts

    @property
    def valid(self) -> Mask:
        """Whether the event has a timestamp (read-only)."""
        return self._valid

    @property
    def activity_codes(self) -> Codes:
        """Activity code per event; ``len(activity_labels)`` for an event without a label (read-only)."""
        return self._act_codes

    @property
    def activity_labels(self) -> tuple[str, ...]:
        return tuple(self._act_labels)

    @property
    def starts(self) -> Codes:
        """Position of each case's first event (read-only)."""
        return self._starts

    @property
    def sizes(self) -> Codes:
        """Number of events per case (read-only)."""
        return self._sizes

    @property
    def tz(self) -> Any:
        return self._tz

    # ------------------------------------------------------------------ selection
    def mask(self, activities: str | Iterable[str]) -> Mask:
        """Events whose activity is one of ``activities`` (unknown labels match nothing)."""
        labels = [activities] if isinstance(activities, str) else list(activities)
        if not all(isinstance(a, str) for a in labels):
            raise LogSchemaError("activities must be labels (strings); pass an event mask as a boolean array")
        lut = np.zeros(len(self._act_labels) + 1, dtype=bool)
        idx = self._act_labels.get_indexer(pd.Index(labels, dtype=object))
        lut[idx[idx >= 0]] = True
        out: Mask = lut[self._act_codes]
        return out

    def _select(self, select: Select) -> Mask | None:
        if select is None:
            return None
        if isinstance(select, pd.Series):
            select = select.to_numpy()
        if isinstance(select, np.ndarray):
            if select.dtype != np.bool_:
                raise LogSchemaError(f"an event mask must be boolean, got dtype {select.dtype}")
            if select.shape != self._codes.shape:
                raise LogSchemaError(f"an event mask needs one entry per event ({self.n_events}), got shape {select.shape}")
            return select
        return self.mask(select)

    # ------------------------------------------------------------------ primitives
    def count(self, select: Select = None) -> Codes:
        """Number of selected events per case, with or without a timestamp."""
        m = self._select(select)
        codes = self._codes if m is None else self._codes[m]
        out: Codes = np.bincount(codes, minlength=self.n_cases).astype(np.int64)
        return out

    def first(self, select: Select = None) -> Codes:
        """Timestamp of the first selected event with a timestamp per case, :data:`NAT` if none."""
        m = self._select(select)
        m = self._valid if m is None else m & self._valid
        c, t = self._codes[m], self._ts[m]
        out = np.full(self.n_cases, NAT, dtype=np.int64)
        if len(c):
            head = np.r_[True, c[1:] != c[:-1]]
            out[c[head]] = t[head]
        return out

    def last(self, select: Select = None) -> Codes:
        """Timestamp of the last selected event with a timestamp per case, :data:`NAT` if none."""
        m = self._select(select)
        m = self._valid if m is None else m & self._valid
        c, t = self._codes[m], self._ts[m]
        out = np.full(self.n_cases, NAT, dtype=np.int64)
        if len(c):
            tail = np.r_[c[1:] != c[:-1], True]
            out[c[tail]] = t[tail]
        return out

    def aggregate(
        self, values: ArrayLike | pd.Series, select: Select = None, *, how: str = "sum", empty: float = np.nan
    ) -> NDArray[Any]:
        """Per-case aggregate of an event column over the selected events.

        ``values`` holds one entry per event in index order. ``how`` is
        ``"sum"`` (missing values count as 0, accumulated in event order,
        always float) or any pandas group aggregation name (missing values
        skipped, the aggregation's own dtype while every case has a selected
        event, float otherwise). Cases without a selected event get ``empty``.
        """
        if not isinstance(values, pd.Series):
            values = np.asarray(values)
        if len(values) != self.n_events:
            raise LogSchemaError(f"an event column needs one entry per event ({self.n_events}), got {len(values)}")
        m = self._select(select)
        codes = self._codes if m is None else self._codes[m]
        n = self.n_cases
        if how == "sum":
            v = np.asarray(values, dtype=np.float64)
            v = v if m is None else v[m]
            out: NDArray[Any] = np.bincount(codes, weights=np.where(np.isnan(v), 0.0, v), minlength=n)
        else:
            series = values if isinstance(values, pd.Series) else pd.Series(values, copy=False)
            picked = series if m is None else series[m]
            out = picked.groupby(codes).agg(how).reindex(range(n)).to_numpy(copy=True)
        none = np.bincount(codes, minlength=n) == 0
        if none.any():
            out[none] = empty
        return out

    def run_starts(self) -> NDArray[np.intp]:
        """Per event, the position of the first event of its run of equal
        ``(case, timestamp)``; the events of a case without a timestamp form one run."""
        codes, ts = self._codes, self._ts
        new_run = np.r_[True, (codes[1:] != codes[:-1]) | (ts[1:] != ts[:-1])]
        out: NDArray[np.intp] = np.maximum.accumulate(np.where(new_run, np.arange(len(codes)), 0))
        return out

    def tied(self) -> Mask:
        """Events that share their case and their timestamp with another event."""
        codes, ts = self._codes, self._ts
        same_prev = (codes[1:] == codes[:-1]) & (ts[1:] == ts[:-1]) & self._valid[1:]
        out = np.zeros(len(codes), dtype=bool)
        out[1:] |= same_prev
        out[:-1] |= same_prev
        return out

    # ------------------------------------------------------------------ timestamps
    def to_timestamp(self, value: Any) -> pd.Timestamp | None:
        """``value`` as a :class:`pandas.Timestamp` comparable with the log's
        timestamps: naive values are localised to the log's time zone, aware
        values are converted to naive UTC on a naive log. ``None`` stays ``None``."""
        if value is None:
            return None
        t = pd.Timestamp(value)
        if self._tz is not None and t.tzinfo is None:
            t = t.tz_localize(self._tz)
        elif self._tz is None and t.tzinfo is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        return t

    def to_ns(self, value: Any) -> int:
        """``value`` as int64 nanoseconds in UTC, comparable with :attr:`timestamps`."""
        t = self.to_timestamp(value)
        if t is None:
            raise LogSchemaError("a timestamp is required, got None")
        if self._tz is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        return int(np.datetime64(t, "ns").view("int64"))

    def to_datetime(self, values: ArrayLike) -> pd.DatetimeIndex:
        """int64 nanoseconds (:data:`NAT` where missing) as datetimes in the log's time zone."""
        idx = pd.DatetimeIndex(np.asarray(values, dtype=np.int64).view("datetime64[ns]"))
        if self._tz is not None:
            idx = idx.tz_localize("UTC").tz_convert(self._tz)
        return idx
