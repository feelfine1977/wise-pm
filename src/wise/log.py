"""Event log wrapper and vectorised case-level primitives (paper Sec. IV-A).

:class:`EventLog` validates a case-based log held in a pandas DataFrame,
parses timestamps, fixes the case notion, and exposes the primitives the
constraint semantics are written in:

* ``cnt(a, σ)``                  → :meth:`EventLog.count`
* ``t1(a, σ)``                   → :meth:`EventLog.first_ts`
* first ``b`` after (first) ``a`` → :meth:`EventLog.first_after`
* ``tot_α(A, σ)``                → :meth:`EventLog.total`
* ``att_β(σ)``, ``exp(σ)``       → :attr:`EventLog.cases`

Internally the log is stored as integer case codes and activity codes sorted
by ``(case, timestamp, order)``; every primitive is a numpy operation on
those codes, so a log with a few million events scores in about a second.

Column names default to the pm4py / XES conventions ``case:concept:name``,
``concept:name`` and ``time:timestamp``.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Iterable, Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

from .constraints import Labels, as_labels, as_list
from .errors import LogSchemaError, StaleEvidenceError

_INT_MIN = np.iinfo(np.int64).min
_INT_MAX = np.iinfo(np.int64).max

CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
TIMESTAMP_COL = "time:timestamp"

#: Prefix of a witness reference that is **local to a snapshot**: the position
#: of a row in this log's sorted event table. It is deliberately not a source
#: system event id; see :class:`LogSnapshot`.
SNAPSHOT_LOCAL_PREFIX = "snapshot-local:row="

_RESERVED = {"n_events", "first_ts", "last_ts", "exposure", "score"}


def _check_attribute_name(name: str) -> None:
    if name in _RESERVED or name.startswith(("contrib__", "score__")):
        raise LogSchemaError(
            f"{name!r} is reserved for computed columns; rename the attribute "
            f"(reserved: {sorted(_RESERVED)}, prefixes 'contrib__' and 'score__')"
        )


class EventLog:
    """A case-based event log.

    Parameters
    ----------
    events
        One row per event.
    case_col, activity_col, timestamp_col
        Column names (pm4py defaults).
    case_attributes
        Case-level attribute columns (company, spend area, flow type, ...).
        The first non-null value per case is used; :meth:`validate` reports
        attributes that vary within a case. These become the slice keys and
        the applicability attributes.
    exposure_col, exposure_agg
        Optional numeric column recording business exposure ``exp(σ) ≥ 0``,
        aggregated per case with ``exposure_agg`` (``"max"`` for cumulative
        amounts, ``"sum"`` for per-event amounts, ``"first"`` for constant
        attributes; any pandas aggregation name). Cases without a numeric
        value get exposure 0.
    order_col
        Column that orders events sharing a timestamp (an event id or a
        sequence number); it determines the order of traces and of
        derived counts. Constraint semantics compare timestamps, so events
        with equal timestamps are simultaneous for them. Default: the input
        row order.
    event_id_col
        Column identifying the source event, used by diagnostics to detect
        events replicated across cases.
    lifecycle_col, keep_transitions
        If the log has lifecycle transitions, keep only the listed ones
        (default ``("complete",)``, compared case-insensitively) so that
        start/complete pairs are not counted twice.
    utc
        Convert timestamps to UTC (required for strings with mixed offsets).
    timestamp_format, dayfirst
        Passed to :func:`pandas.to_datetime` when the column holds strings
        (e.g. ``timestamp_format="%d-%m-%Y %H:%M:%S"`` or ``dayfirst=True``).
    missing_timestamps
        ``"raise"`` (default) if any timestamp is null or unparseable,
        ``"drop"`` those events, or ``"keep"`` them with NaT. Kept events
        count for plain presence, exclusion and singularity constraints;
        they never define a lag and are ignored by time-scoped counts.
    window
        Optional ``(start, end)`` of the observation window. Used by the
        censoring diagnostics and by ``Lag(missing_b="censor")``. When not
        given, :meth:`observation_window` derives a robust one from quantiles.
    keep_columns
        Restrict :attr:`events` to these columns (plus the core ones) to
        save memory on wide logs.
    dedupe
        Drop exact duplicate events (same case, activity, timestamp).
    """

    def __init__(
        self,
        events: pd.DataFrame,
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
        window: tuple[Any, Any] | None = None,
        keep_columns: Iterable[str] | None = None,
        dedupe: bool = False,
    ):
        case_attributes = as_list(case_attributes)
        keep_columns = None if keep_columns is None else as_list(keep_columns)
        required = [case_col, activity_col, timestamp_col, *case_attributes]
        for c in (exposure_col, order_col, event_id_col, lifecycle_col):
            if c is not None:
                required.append(c)
        missing = [c for c in required if c not in events.columns]
        if missing:
            raise LogSchemaError(
                f"event log is missing columns {missing}; available: {list(events.columns)[:30]}"
                + (" ..." if events.shape[1] > 30 else "")
            )
        for attr in case_attributes:
            _check_attribute_name(attr)
        if missing_timestamps not in ("raise", "drop", "keep"):
            raise ValueError("missing_timestamps must be 'raise', 'drop' or 'keep'")

        df = events
        if lifecycle_col is not None:
            keep = {str(t).lower() for t in as_list(keep_transitions)}
            df = df[df[lifecycle_col].astype(str).str.lower().isin(keep)]
        if dedupe:
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
        if pd.api.types.is_datetime64_any_dtype(ts_raw):
            ts = ts_raw if not utc else ts_raw.dt.tz_convert("UTC") if ts_raw.dt.tz is not None else ts_raw.dt.tz_localize("UTC")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                warnings.simplefilter("ignore", UserWarning)
                ts = pd.to_datetime(ts_raw, errors="coerce", utc=utc, format=timestamp_format, dayfirst=dayfirst)
                retry = ts.isna() & ts_raw.notna()
                if retry.any() and pd.api.types.is_datetime64_any_dtype(ts):
                    again = pd.to_datetime(ts_raw[retry], errors="coerce", utc=utc, format="mixed", dayfirst=dayfirst)
                    if pd.api.types.is_datetime64_any_dtype(again) and (again.dt.tz is None) == (ts.dt.tz is None):
                        ts = ts.copy()
                        ts[retry] = again
            if not pd.api.types.is_datetime64_any_dtype(ts):
                raise LogSchemaError(f"timestamp column {timestamp_col!r} has mixed time zones or types; pass utc=True")
        n_missing_ts = int(ts.isna().sum())
        if n_missing_ts and missing_timestamps == "raise":
            first_bad = ts_raw[ts.isna()].iloc[0]
            raise LogSchemaError(
                f"{n_missing_ts} timestamps are null or unparseable (first: {first_bad!r}); "
                "pass missing_timestamps='drop' or 'keep', or utc=True for mixed offsets"
            )
        if n_missing_ts and missing_timestamps == "drop":
            keep_mask = ts.notna().to_numpy()
            df, ts, codes = df[keep_mask], ts[keep_mask], codes[keep_mask]
            codes, uniques = pd.factorize(pd.Series(uniques[codes]), sort=True)
            codes = codes.astype(np.int64)
            n_missing_ts = 0
        self.tz = ts.dt.tz
        ts_int = (
            ts.to_numpy("datetime64[ns]").view("int64")
            if self.tz is None
            else ts.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]").view("int64")
        )
        valid = ts_int != _INT_MIN
        sort_ts = np.where(valid, ts_int, _INT_MAX)

        # ---- order: (case, timestamp, tie-break) ------------------------------------------
        if order_col is not None:
            tie = pd.factorize(df[order_col], sort=True)[0]
        else:
            tie = np.arange(len(df))
        order = np.lexsort((tie, sort_ts, codes))

        self._codes = codes[order]
        self._ts = ts_int[order]
        self._valid = valid[order]
        n_cases = len(uniques)
        self._n_cases = n_cases
        self._starts = np.searchsorted(self._codes, np.arange(n_cases))
        counts = np.bincount(self._codes, minlength=n_cases)

        cols = list(df.columns)
        if keep_columns is not None:
            wanted = set(required) | set(keep_columns)
            cols = [c for c in cols if c in wanted]
        ev = df.iloc[order][cols].reset_index(drop=True)
        ev[timestamp_col] = ts.to_numpy()[order] if self.tz is None else ts.iloc[order].reset_index(drop=True)
        self.events = ev

        act = ev[activity_col]
        self.n_null_activities = int(act.isna().sum())
        if isinstance(act.dtype, pd.CategoricalDtype):
            act_codes = act.cat.codes.to_numpy().astype(np.int64)
            act_labels = pd.Index(act.cat.categories.astype(str))
        else:
            act_codes, act_uniques = pd.factorize(act, sort=False)
            act_codes = act_codes.astype(np.int64)
            act_labels = pd.Index(pd.Index(act_uniques).astype(str))
        self._act_codes = np.where(act_codes < 0, len(act_labels), act_codes)
        self._act_labels = act_labels

        self.case_col, self.activity_col, self.timestamp_col = case_col, activity_col, timestamp_col
        self.case_attributes = case_attributes
        self.exposure_col, self.order_col, self.event_id_col = exposure_col, order_col, event_id_col
        self.n_missing_timestamps = n_missing_ts

        # ---- case table -----------------------------------------------------------------
        index = pd.Index(uniques, name=case_col)
        cases = pd.DataFrame(index=index)
        cases["n_events"] = counts
        cases["first_ts"] = self._dt_values(self._first_from_mask(self._valid))
        cases["last_ts"] = self._dt_values(self._last_from_mask(self._valid))
        for attr in case_attributes:
            cases[attr] = ev[attr].groupby(self._codes).first().reindex(range(n_cases)).to_numpy()
        if exposure_col is not None:
            vals = pd.to_numeric(ev[exposure_col], errors="coerce")
            exp = vals.groupby(self._codes).agg(exposure_agg).reindex(range(n_cases)).fillna(0.0).to_numpy()
            if (exp < 0).any():
                raise LogSchemaError(
                    f"exposure column {exposure_col!r} yields negative case exposure for "
                    f"{int((exp < 0).sum())} cases; the paper assumes exp(σ) >= 0"
                )
            cases["exposure"] = exp.astype(float)
        self.cases = cases

        self.window: tuple[pd.Timestamp | None, pd.Timestamp | None] | None = None
        if window is not None:
            start, end = (self._to_ts(window[0]), self._to_ts(window[1]))
            if start is not None and end is not None and start > end:
                raise ValueError("window start must not be after window end")
            self.window = (start, end)

        self._count_cache: dict[Labels, pd.Series] = {}
        self._first_cache: dict[Labels, pd.Series] = {}
        self._last_cache: dict[Labels, pd.Series] = {}
        self._recipe_cache: dict[str, str] = {}

        # The preparation options exactly as they were supplied, for the run
        # record (:mod:`wise.evidence.manifest`). Recorded, never re-applied.
        self._options: dict[str, Any] = {
            "case_col": case_col,
            "activity_col": activity_col,
            "timestamp_col": timestamp_col,
            "case_attributes": list(case_attributes),
            "exposure_col": exposure_col,
            "exposure_agg": exposure_agg if exposure_col is not None else None,
            "order_col": order_col,
            "event_id_col": event_id_col,
            "lifecycle_col": lifecycle_col,
            "keep_transitions": [str(t) for t in as_list(keep_transitions)] if lifecycle_col is not None else None,
            "utc": bool(utc),
            "timestamp_format": timestamp_format,
            "dayfirst": bool(dayfirst),
            "missing_timestamps": missing_timestamps,
            "window": [None if t is None else t.isoformat() for t in self.window] if self.window is not None else None,
            "keep_columns": None if keep_columns is None else list(keep_columns),
            "dedupe": bool(dedupe),
            "tie_order_policy": f"order_col:{order_col}" if order_col is not None else "input_row_order",
            "timezone": str(self.tz) if self.tz is not None else None,
        }

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_csv(cls, path: str, *, read_kwargs: Mapping[str, Any] | None = None, **kwargs: Any) -> EventLog:
        """Read a CSV (or parquet, by extension) and build the log."""
        p = str(path)
        if p.lower().endswith((".parquet", ".pq")):
            df = pd.read_parquet(p, **(read_kwargs or {}))
        else:
            df = pd.read_csv(p, **{"low_memory": False, **(read_kwargs or {})})
        return cls(df, **kwargs)

    @classmethod
    def from_pm4py(cls, obj: Any, **kwargs: Any) -> EventLog:
        """Build from a pm4py DataFrame or EventLog object (pm4py optional)."""
        if isinstance(obj, pd.DataFrame):
            return cls(obj, **kwargs)
        try:
            import pm4py  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("pm4py is required for EventLog.from_pm4py on non-DataFrame input") from exc
        return cls(pm4py.convert_to_dataframe(obj), **kwargs)

    @classmethod
    def from_xes(cls, path: str, **kwargs: Any) -> EventLog:
        """Read an XES file with pm4py (optional dependency)."""
        try:
            import pm4py  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("pm4py is required for EventLog.from_xes; pip install 'wise-pm[pm4py]'") from exc
        return cls(pm4py.read_xes(str(path)), **kwargs)

    def to_pm4py(self) -> pd.DataFrame:
        """The events with pm4py's standard column names."""
        return self.events.rename(
            columns={self.case_col: CASE_COL, self.activity_col: ACTIVITY_COL, self.timestamp_col: TIMESTAMP_COL}
        )

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return self._n_cases

    def __repr__(self) -> str:
        return (
            f"EventLog({len(self.events):,} events, {len(self):,} cases, "
            f"{len(self._act_labels)} activities, attributes={self.case_attributes})"
        )

    @property
    def case_ids(self) -> pd.Index:
        return self.cases.index

    @property
    def activity_labels(self) -> list[str]:
        return list(self._act_labels)

    @property
    def window_end(self) -> pd.Timestamp:
        """End of the observation window: the explicit ``window`` end if given,
        else the last timestamp in the log (which may be an outlier; see
        :meth:`observation_window`)."""
        if self.window is not None and self.window[1] is not None:
            return self.window[1]
        return pd.Timestamp(self.events[self.timestamp_col].max())

    @property
    def window_start(self) -> pd.Timestamp:
        if self.window is not None and self.window[0] is not None:
            return self.window[0]
        return pd.Timestamp(self.events[self.timestamp_col].min())

    def observation_window(self, q: float = 0.001) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Observation window ``(start, end)``.

        The explicit ``window`` if one was given; otherwise the ``q`` quantile
        of case starts and the ``1 − q`` quantile of case ends, taken at
        actual case timestamps, which trims placeholder dates at the
        extremes. When ``q`` would trim less than half a case the plain
        minimum and maximum are returned.
        """
        if self.window is not None and None not in self.window:
            return self.window  # type: ignore[return-value]
        first, last = self.cases["first_ts"].dropna(), self.cases["last_ts"].dropna()
        if len(last) == 0:
            return pd.NaT, pd.NaT  # type: ignore[return-value]
        if q * (len(last) - 1) < 0.5:
            return pd.Timestamp(first.min()), pd.Timestamp(last.max())
        start = first.quantile(q, interpolation="higher")
        end = last.quantile(1 - q, interpolation="lower")
        return pd.Timestamp(start), pd.Timestamp(end)

    def censoring_end(self, tolerance: pd.Timedelta | None = None, q: float = 0.001) -> pd.Timestamp:
        """End of the observation window used for censoring: the explicit
        window end if set, else the robust end of :meth:`observation_window`.
        Warns when the last raw timestamp lies more than ``tolerance`` beyond
        that end."""
        if self.window is not None and self.window[1] is not None:
            return self.window[1]
        end = self.observation_window(q)[1]
        raw_max = pd.Timestamp(self.events[self.timestamp_col].max())
        if tolerance is not None and raw_max - end > tolerance:
            warnings.warn(
                f"log ends at {raw_max} but the observation window ends at {end}; using the latter. "
                "Pass EventLog(window=...) to set the window explicitly.",
                stacklevel=3,
            )
        return end

    def _to_ts(self, value: Any) -> pd.Timestamp | None:
        if value is None:
            return None
        t = pd.Timestamp(value)
        if self.tz is not None and t.tzinfo is None:
            t = t.tz_localize(self.tz)
        elif self.tz is None and t.tzinfo is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        return t

    def _ts_to_int(self, t: pd.Timestamp) -> int:
        t = self._to_ts(t)  # type: ignore[assignment]
        if self.tz is not None:
            t = t.tz_convert("UTC").tz_localize(None)  # type: ignore[union-attr]
        return int(np.datetime64(t, "ns").view("int64"))  # type: ignore[arg-type]

    def _dt_values(self, values: np.ndarray) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex(values.view("datetime64[ns]"))
        if self.tz is not None:
            idx = idx.tz_localize("UTC").tz_convert(self.tz)
        return idx

    def _series_dt(self, values: np.ndarray) -> pd.Series:
        return pd.Series(self._dt_values(values), index=self.case_ids)

    def _series(self, values: np.ndarray) -> pd.Series:
        return pd.Series(values, index=self.case_ids)

    # ------------------------------------------------------------------ masks
    def _mask(self, labels: Labels) -> np.ndarray:
        lut = np.zeros(len(self._act_labels) + 1, dtype=bool)
        idx = self._act_labels.get_indexer(list(labels))
        lut[idx[idx >= 0]] = True
        return lut[self._act_codes]

    def _first_from_mask(self, mask: np.ndarray) -> np.ndarray:
        """Timestamp (int64 ns, NaT = int64 min) of the first masked event per case."""
        c, t = self._codes[mask], self._ts[mask]
        out = np.full(len(self), _INT_MIN, dtype=np.int64)
        if len(c):
            first = np.r_[True, c[1:] != c[:-1]]
            out[c[first]] = t[first]
        return out

    def _last_from_mask(self, mask: np.ndarray) -> np.ndarray:
        m = mask & self._valid
        c, t = self._codes[m], self._ts[m]
        out = np.full(len(self), _INT_MIN, dtype=np.int64)
        if len(c):
            last = np.r_[c[1:] != c[:-1], True]
            out[c[last]] = t[last]
        return out

    # ------------------------------------------------------------------ primitives
    def count(self, activities: str | Iterable[str]) -> pd.Series:
        """``cnt(a, σ)``: occurrences of any of ``activities`` per case."""
        labels = as_labels(activities)
        key = tuple(sorted(labels))
        if key not in self._count_cache:
            cnt = np.bincount(self._codes[self._mask(labels)], minlength=len(self)).astype(float)
            self._count_cache[key] = self._series(cnt)
        return self._count_cache[key]

    def first_ts(self, activities: str | Iterable[str]) -> pd.Series:
        """``t1(a, σ)``: timestamp of the first occurrence, NaT if none."""
        labels = as_labels(activities)
        key = tuple(sorted(labels))
        if key not in self._first_cache:
            self._first_cache[key] = self._series_dt(self._first_from_mask(self._mask(labels)))
        return self._first_cache[key]

    def last_ts(self, activities: str | Iterable[str]) -> pd.Series:
        """Timestamp of the last occurrence, NaT if none."""
        labels = as_labels(activities)
        key = tuple(sorted(labels))
        if key not in self._last_cache:
            self._last_cache[key] = self._series_dt(self._last_from_mask(self._mask(labels)))
        return self._last_cache[key]

    def first_after(
        self,
        a: str | Iterable[str],
        b: str | Iterable[str],
        *,
        activation: Literal["first", "last"] = "first",
        response: Literal["first_after", "first_overall"] = "first_after",
    ) -> tuple[pd.Series, pd.Series]:
        """``t_a`` (first or last ``a``) and ``t_b`` (first ``b`` at or after it).

        With ``response="first_overall"`` ``t_b`` is the first ``b`` in the
        case regardless of order. ``t_b`` is NaT when undefined.
        """
        la, lb = as_labels(a, what="a"), as_labels(b, what="b")
        t_a_int = self._first_from_mask(self._mask(la)) if activation == "first" else self._last_from_mask(self._mask(la))
        if response == "first_overall":
            return self._series_dt(t_a_int), self.first_ts(lb)
        anchor = np.where(t_a_int == _INT_MIN, _INT_MAX, t_a_int)
        mask_b = self._mask(lb) & self._valid & (self._ts >= anchor[self._codes])
        return self._series_dt(t_a_int), self._series_dt(self._first_from_mask(mask_b))

    def activation_lags(self, a: str | Iterable[str], b: str | Iterable[str]) -> pd.DataFrame:
        """One row per ``a`` event: its case code, timestamp, and the timestamp
        of the first ``b`` event at or after it in the same case (NaT if none)."""
        la, lb = as_labels(a, what="a"), as_labels(b, what="b")
        a_pos = np.flatnonzero(self._mask(la) & self._valid)
        b_pos = np.flatnonzero(self._mask(lb) & self._valid)
        # first position of the run of events sharing (case, timestamp) with each a event
        new_run = np.r_[True, (self._codes[1:] != self._codes[:-1]) | (self._ts[1:] != self._ts[:-1])]
        run_start = np.maximum.accumulate(np.where(new_run, np.arange(len(self._codes)), 0))
        j = np.searchsorted(b_pos, run_start[a_pos])
        has = j < len(b_pos)
        bp = b_pos[np.minimum(j, max(len(b_pos) - 1, 0))] if len(b_pos) else np.zeros(len(a_pos), dtype=int)
        ok = has & (len(b_pos) > 0) & (self._codes[bp] == self._codes[a_pos])
        t_b = np.where(ok, self._ts[bp], _INT_MIN)
        out = pd.DataFrame(
            {"code": self._codes[a_pos], "t_a": self._ts[a_pos].view("datetime64[ns]"), "t_b": t_b.view("datetime64[ns]")}
        )
        return out

    def count_scoped(
        self,
        activities: str | Iterable[str],
        *,
        after: str | Iterable[str] | None = None,
        before: str | Iterable[str] | None = None,
    ) -> pd.Series:
        """Count of ``activities`` per case, restricted to events strictly after
        the first ``after`` activity and/or strictly before the first ``before``
        activity. NaN where an anchor activity is absent."""
        mask = self._mask(as_labels(activities)) & self._valid
        undefined = np.zeros(len(self), dtype=bool)
        if after is not None:
            t_after = self._first_from_mask(self._mask(as_labels(after, what="after")))
            undefined |= t_after == _INT_MIN
            mask &= self._ts > np.where(t_after == _INT_MIN, _INT_MAX, t_after)[self._codes]
        if before is not None:
            t_before = self._first_from_mask(self._mask(as_labels(before, what="before")))
            undefined |= t_before == _INT_MIN
            mask &= self._ts < np.where(t_before == _INT_MIN, _INT_MIN, t_before)[self._codes]
        cnt = np.bincount(self._codes[mask], minlength=len(self)).astype(float)
        cnt[undefined] = np.nan
        return self._series(cnt)

    def count_before(self, b: str | Iterable[str], a: str | Iterable[str]) -> pd.Series:
        """Number of ``b`` events strictly before the first ``a``; NaN if no ``a``."""
        return self.count_scoped(b, before=a)

    def count_after(self, b: str | Iterable[str], a: str | Iterable[str]) -> pd.Series:
        """Number of ``b`` events strictly after the first ``a``; NaN if no ``a``."""
        return self.count_scoped(b, after=a)

    def total(
        self,
        attribute: str,
        activities: str | Iterable[str],
        agg: Literal["sum", "first", "last", "max", "min", "mean"] = "sum",
    ) -> pd.Series:
        """``tot_α(A, σ)``: aggregate of ``attribute`` over events with act ∈ A.

        Sums over empty sets are 0; other aggregations give 0 as well when
        no event matches. Non-numeric values count as 0.
        """
        if attribute not in self.events.columns:
            raise LogSchemaError(f"event attribute {attribute!r} not in log")
        mask = self._mask(as_labels(activities))
        vals = pd.to_numeric(self.events[attribute], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if agg == "sum":
            return self._series(np.bincount(self._codes[mask], weights=vals[mask], minlength=len(self)))
        s = pd.Series(vals[mask]).groupby(self._codes[mask]).agg(agg)
        return self._series(s.reindex(range(len(self))).fillna(0.0).to_numpy())

    def attribute(self, name: str) -> pd.Series:
        """Numeric case attribute from :attr:`cases`, NaN where missing."""
        if name not in self.cases.columns:
            raise LogSchemaError(
                f"case attribute {name!r} not available; pass it via case_attributes, "
                f"derive it with EventLog.derive, or add it with add_case_attribute"
            )
        return pd.to_numeric(self.cases[name], errors="coerce")

    def add_case_attribute(self, name: str, values: pd.Series | Mapping[Any, Any] | np.ndarray) -> None:
        """Attach an engineered case-level attribute (aligned on case id)."""
        _check_attribute_name(name)
        if isinstance(values, Mapping):
            values = pd.Series(values)
        if isinstance(values, pd.Series):
            if not values.index.isin(self.case_ids).any() and len(values):
                raise LogSchemaError(f"values for {name!r} are not indexed by case id")
            aligned = values.reindex(self.case_ids)
        else:
            arr = np.asarray(values)
            if len(arr) != len(self):
                raise LogSchemaError(f"values for {name!r} must have one entry per case ({len(self)})")
            aligned = pd.Series(arr, index=self.case_ids)
        self.cases[name] = aligned
        if name not in self.case_attributes:
            self.case_attributes.append(name)

    def derive(
        self,
        recipes: Iterable[Mapping[str, Any]],
        *,
        overwrite: bool = True,
        calibrations: Mapping[str, Any] | None = None,
    ) -> list[str]:
        """Compute derived case attributes from declarative recipes (see
        :mod:`wise.derive`) and attach them. Returns the attribute names.

        ``calibrations`` maps an attribute name to a frozen
        :class:`wise.evidence.CalibrationRecord`; those recipes then apply the
        fitted reference instead of refitting one on this log.

        This modifies the log in place — that is the historical behaviour, and
        :func:`wise.score` uses it by default. Evidence capture records that
        the input was modified.
        """
        from .derive import apply_recipes

        return apply_recipes(self, recipes, overwrite=overwrite, calibrations=calibrations)

    def trace(self, case_id: Any) -> pd.DataFrame:
        """Events of one case in timestamp order (for drill-down)."""
        pos = self.case_ids.get_loc(case_id)
        if not isinstance(pos, int | np.integer):
            raise LogSchemaError(f"case id {case_id!r} is not unique")
        i = int(pos)
        start = int(self._starts[i])
        n = int(self.cases["n_events"].to_numpy()[i])
        return self.events.iloc[start : start + n]

    # ------------------------------------------------------------------ provenance
    def options(self) -> dict[str, Any]:
        """The preparation options this log was built with, as plain JSON types.

        Includes the column mapping, the lifecycle filter, the timestamp
        policy, the observation window, ``dedupe`` and the tie-order policy
        (``"order_col:<name>"`` or ``"input_row_order"``). These are recorded
        for the run manifest; reading them changes nothing.
        """
        return {k: (list(v) if isinstance(v, list) else v) for k, v in self._options.items()}

    def snapshot(self, *, freeze: bool = False, scope: Literal["structure", "cases", "events"] = "cases") -> LogSnapshot:
        """A witness interface over this log, protected against later mutation.

        With ``freeze=False`` (default) the snapshot keeps a fingerprint of the
        log and re-checks it before materialising any witness, raising
        :class:`~wise.errors.StaleEvidenceError` when the log has changed.
        With ``freeze=True`` it copies the columns needed for witnesses, so it
        stays valid whatever happens to the log afterwards.

        ``scope="cases"`` (default) fingerprints the case table — which is
        what :meth:`add_case_attribute` and :meth:`derive` change — together
        with the structure of the event table; ``scope="events"`` additionally
        fingerprints **every column** of the event table, at the cost of
        hashing every event, so a relabelled activity, a shifted timestamp or a
        changed amount all invalidate it; ``scope="structure"`` hashes no
        content at all and records only the preparation options, the shapes and
        the column names.

        :attr:`LogSnapshot.content_hashed_tables` says which tables a snapshot's
        identity covers, and :attr:`LogSnapshot.content_hashed` is ``True`` only
        when every one of them is covered — which is ``"events"`` alone. Evidence
        capture uses ``"events"`` for exactly that reason: a witness served after
        a changed amount would be the evidence of a different score.
        """
        return LogSnapshot(self, freeze=freeze, scope=scope)

    # ------------------------------------------------------------------ validation
    def validate(self, q: float = 0.001) -> pd.Series:
        """Data-quality summary of the log as a Series of counts and shares.

        Reports missing timestamps, timestamp outliers outside the robust
        observation window, null activity labels, timestamp ties within a
        case, case attributes that vary within a case, and exposure issues.
        Nothing here raises; read it before trusting a score.
        """
        start, end = self.observation_window(q)
        ts = self.events[self.timestamp_col]
        outliers = int(((ts < start) | (ts > end)).sum())
        same_prev = (self._codes[1:] == self._codes[:-1]) & (self._ts[1:] == self._ts[:-1]) & self._valid[1:]
        tied = np.zeros(len(self._codes), dtype=bool)
        tied[1:] |= same_prev
        tied[:-1] |= same_prev
        cases_with_ties = len(np.unique(self._codes[tied]))
        report: dict[str, Any] = {
            "n_events": len(self.events),
            "n_cases": len(self),
            "n_activities": len(self._act_labels),
            "window_start": start,
            "window_end": end,
            "raw_max_timestamp": ts.max(),
            "missing_timestamps": self.n_missing_timestamps,
            "timestamp_outliers": outliers,
            "null_activity_labels": self.n_null_activities,
            "tied_events_share": float(tied.mean()) if len(tied) else 0.0,
            "cases_with_ties_share": cases_with_ties / max(len(self), 1),
        }
        for attr in self.case_attributes:
            if attr in self.events.columns:
                n_var = int((self.events[attr].groupby(self._codes).nunique(dropna=True) > 1).sum())
                if n_var:
                    report[f"varying_within_case[{attr}]"] = n_var
        if "exposure" in self.cases.columns:
            report["zero_exposure_cases"] = int((self.cases["exposure"] == 0).sum())
        return pd.Series(report, name="log_validation")


# ---------------------------------------------------------------------- snapshots
def _frame_digest(frame: pd.DataFrame) -> str | None:
    """SHA-256 over pandas' row hashes, or ``None`` if a column is unhashable."""
    try:
        values = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="uint64")
    except TypeError:  # pragma: no cover - unhashable object column
        return None
    return hashlib.sha256(values.tobytes()).hexdigest()


def _iso(value: Any) -> str | None:
    ts = pd.Timestamp(value)
    return None if pd.isna(ts) else str(ts.isoformat())


class LogSnapshot:
    """Bounded witness access to an :class:`EventLog`, guarded against mutation.

    A score result keeps the *live* log, and both :meth:`EventLog.derive` and
    :meth:`EventLog.add_case_attribute` change it in place. A snapshot records
    what the log looked like when evidence was captured, so that a witness
    materialised later is either the original one (``freeze=True``) or refused
    (:class:`~wise.errors.StaleEvidenceError`).

    Event references are source event ids when ``event_id_col`` was configured
    (:attr:`source_identity_available` is then ``True``); otherwise they are
    positions in this snapshot's sorted event table, prefixed with
    ``"snapshot-local:row="``. A snapshot-local reference identifies a row here
    and nowhere else — it is never presented as a source system event id.
    """

    def __init__(self, log: EventLog, *, freeze: bool = False, scope: Literal["structure", "cases", "events"] = "cases"):
        if scope not in ("structure", "cases", "events"):
            raise ValueError("scope must be 'structure', 'cases' or 'events'")
        self.scope = scope
        self.frozen = bool(freeze)
        self.options = log.options()
        self.case_col, self.activity_col, self.timestamp_col = log.case_col, log.activity_col, log.timestamp_col
        self.event_id_col = log.event_id_col
        self.source_identity_available = log.event_id_col is not None
        self.n_events = len(log.events)
        self.n_cases = len(log)
        self.timezone = str(log.tz) if log.tz is not None else None
        self._columns = [log.activity_col, log.timestamp_col] + ([log.event_id_col] if log.event_id_col else [])
        # The identity of the event table covers *every* event column, not only
        # the three a witness prints: a Balance or a Metric reads an amount, and
        # a changed amount is a changed input even when the trace is untouched.
        self._digest_columns = [str(c) for c in log.events.columns]
        self._case_index = log.case_ids.copy()
        self._starts = np.asarray(log._starts, dtype=np.int64).copy()
        self._counts = log.cases["n_events"].to_numpy(dtype=np.int64).copy()
        self._first = log.cases["first_ts"].copy()
        self._last = log.cases["last_ts"].copy()
        self.fingerprint, self.content_hashed_tables = self._fingerprint(log)
        #: ``True`` only when **every** table is content-hashed. A ``"cases"``
        #: snapshot hashes the case table and not the events, so it is ``False``
        #: there: saying otherwise would claim an identity the run never verified.
        self.content_hashed = all(self.content_hashed_tables.values())
        self._log: EventLog | None = None if self.frozen else log
        self._events: pd.DataFrame | None = log.events[self._columns].copy() if self.frozen else None
        self._cache: tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray | None] | None = None

    def _fingerprint(self, log: EventLog) -> tuple[str, dict[str, bool]]:
        """The snapshot digest, and which tables its content actually covers."""
        cases_digest = None if self.scope == "structure" else _frame_digest(log.cases)
        parts: dict[str, Any] = {
            "options": log.options(),
            "n_events": len(log.events),
            "n_cases": len(log),
            "event_columns": [str(c) for c in log.events.columns],
            "case_columns": [str(c) for c in log.cases.columns],
            "recipes": dict(log._recipe_cache),
            "cases_digest": cases_digest,
        }
        covered = {"cases": self.scope != "structure" and cases_digest is not None, "events": False}
        if self.scope == "events":
            columns = [c for c in self._digest_columns if c in log.events.columns]
            events_digest = _frame_digest(log.events[columns])
            parts["events_digest"] = events_digest
            covered["events"] = events_digest is not None
        canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), covered

    def __repr__(self) -> str:
        kind = "frozen" if self.frozen else f"fingerprinted[{self.scope}]"
        return f"LogSnapshot({self.n_events:,} events, {self.n_cases:,} cases, {kind}, {self.fingerprint[:12]}…)"

    def check_fresh(self) -> None:
        """Raise :class:`~wise.errors.StaleEvidenceError` if the log changed.

        A frozen snapshot always passes: it no longer refers to the log.
        """
        if self._log is None:
            return
        current, _ = self._fingerprint(self._log)
        if current != self.fingerprint:
            raise StaleEvidenceError(
                "the event log changed after this evidence was captured "
                f"(snapshot {self.fingerprint[:12]}…, log now {current[:12]}…); "
                "re-score, or capture with EventLog.snapshot(freeze=True) to keep witnesses available"
            )

    # ------------------------------------------------------------------ access
    def _frame(self) -> pd.DataFrame:
        if self._events is not None:
            return self._events
        log = self._log
        if log is None:  # pragma: no cover - a frozen snapshot returned above
            raise StaleEvidenceError("this snapshot has no log to read witnesses from")
        return log.events

    def _arrays(self) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray | None]:
        """Witness columns as arrays, converted once.

        Freshness is checked at the entry points (:meth:`check_fresh`), not per
        lookup: hashing the case table for every witness would make evidence
        cost more than scoring.
        """
        if self._cache is None:
            frame = self._frame()
            activities = frame[self.activity_col].astype(str).to_numpy(dtype=object)
            stamps = pd.DatetimeIndex(pd.to_datetime(frame[self.timestamp_col]))
            naive = stamps.tz_convert("UTC").tz_localize(None) if stamps.tz is not None else stamps
            i8 = naive.to_numpy("datetime64[ns]").view("int64")
            ids = None if self.event_id_col is None else frame[self.event_id_col].astype(str).to_numpy(dtype=object)
            self._cache = (activities, stamps, i8, ids)
        return self._cache

    def _slice(self, unit_id: Any) -> tuple[int, int]:
        pos = self._case_index.get_loc(unit_id)
        if not isinstance(pos, int | np.integer):
            raise LogSchemaError(f"case id {unit_id!r} is not unique in this snapshot")
        i = int(pos)
        return int(self._starts[i]), int(self._counts[i])

    def n_events_of(self, unit_id: Any) -> int:
        """Number of events of one unit in this snapshot."""
        return self._slice(unit_id)[1]

    def bounds_of(self, unit_id: Any) -> tuple[str | None, str | None]:
        """First and last event timestamp of one unit, ISO-8601 or ``None``."""
        pos = int(self._case_index.get_loc(unit_id))  # type: ignore[arg-type]
        return _iso(self._first.iloc[pos]), _iso(self._last.iloc[pos])

    def event_refs(
        self,
        unit_id: Any,
        activities: Iterable[str] | None = None,
        *,
        since: Any = None,
        until: Any = None,
        limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Witness rows of one unit, and how many matched in total.

        Returns ``(rows, n_matched)`` where each row carries ``reference``,
        ``event_id`` (``None`` without a configured ``event_id_col``),
        ``activity`` and an ISO-8601 ``timestamp``. ``limit`` bounds the rows
        returned, never ``n_matched``, so a truncated display keeps the exact
        total.

        Call :meth:`check_fresh` once before a batch of lookups: this method
        reads the snapshot's cached columns and does not re-verify the log.
        """
        start, n = self._slice(unit_id)
        if n == 0:
            return [], 0
        acts, stamps, stamps_i8, ids = self._arrays()
        block = slice(start, start + n)
        mask = np.ones(n, dtype=bool)
        if activities is not None:
            wanted = np.asarray(sorted({str(a) for a in activities}), dtype=object)
            mask &= np.isin(acts[block], wanted)
        if since is not None or until is not None:
            values = stamps_i8[block]
            valid = values != _INT_MIN
            if since is not None:
                mask &= valid & (values >= self._to_i8(since))
            if until is not None:
                mask &= valid & (values <= self._to_i8(until))
        positions = np.flatnonzero(mask)
        n_matched = int(positions.size)
        if limit is not None:
            positions = positions[: max(int(limit), 0)]
        rows: list[dict[str, Any]] = []
        for offset in positions:
            row = start + int(offset)
            rows.append(
                {
                    "reference": f"{SNAPSHOT_LOCAL_PREFIX}{row}",
                    "event_id": None if ids is None else str(ids[row]),
                    "activity": str(acts[row]),
                    "timestamp": _iso(stamps[row]),
                }
            )
        return rows, n_matched

    def _to_i8(self, value: Any) -> int:
        """A timestamp as int64 nanoseconds, in the same convention as the arrays."""
        ts = pd.Timestamp(value)
        if self.timezone is not None and ts.tzinfo is None:
            ts = ts.tz_localize(self.timezone)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return int(np.datetime64(ts, "ns").view("int64"))
