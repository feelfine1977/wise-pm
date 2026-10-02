"""Event log wrapper and vectorised case-level primitives (paper Sec. IV-A).

:class:`EventLog` validates a case-based log held in a pandas DataFrame,
parses timestamps, fixes the case notion, and exposes the primitives the
constraint semantics are written in:

* ``cnt(a, σ)``                  → :meth:`EventLog.count`
* ``t1(a, σ)``                   → :meth:`EventLog.first_ts`
* first ``b`` after (first) ``a`` → :meth:`EventLog.first_after`
* ``tot_α(A, σ)``                → :meth:`EventLog.total`
* ``att_β(σ)``, ``exp(σ)``       → :attr:`EventLog.cases`

Internally the log is held by an immutable :class:`~wise.case_index.CaseIndex`
(:attr:`EventLog.case_index`): integer case codes and activity codes sorted
by ``(case, timestamp, order)``; every primitive is a numpy operation on
those codes, so a log with a few million events scores in about a second.

Column names default to the pm4py / XES conventions ``case:concept:name``,
``concept:name`` and ``time:timestamp``.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

from .case_index import NAT, CaseIndex, event_order
from .constraints import Labels, as_labels, as_list
from .errors import LogSchemaError

_INT_MAX = np.iinfo(np.int64).max

CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
TIMESTAMP_COL = "time:timestamp"

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
            raise LogSchemaError(f"missing_timestamps must be 'raise', 'drop' or 'keep', got {missing_timestamps!r}")

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

        # ---- order: (case, timestamp, tie-break) ------------------------------------------
        tie = pd.factorize(df[order_col], sort=True)[0] if order_col is not None else None
        order = event_order(codes, ts_int, tie)
        n_cases = len(uniques)

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
        self._index = CaseIndex(codes[order], ts_int[order], act_codes, act_labels, n_cases=n_cases, tz=self.tz)

        self.case_col, self.activity_col, self.timestamp_col = case_col, activity_col, timestamp_col
        self.case_attributes = case_attributes
        self.exposure_col, self.order_col, self.event_id_col = exposure_col, order_col, event_id_col
        self.n_missing_timestamps = n_missing_ts

        # ---- case table -----------------------------------------------------------------
        index = pd.Index(uniques, name=case_col)
        cases = pd.DataFrame(index=index)
        cases["n_events"] = self._index.count()
        cases["first_ts"] = self._index.to_datetime(self._index.first())
        cases["last_ts"] = self._index.to_datetime(self._index.last())
        for attr in case_attributes:
            cases[attr] = ev[attr].groupby(self._codes).first().reindex(range(n_cases)).to_numpy()
        if exposure_col is not None:
            vals = pd.to_numeric(ev[exposure_col], errors="coerce")
            try:
                per_case = vals.groupby(self._codes).agg(exposure_agg)
            except (AttributeError, TypeError, ValueError) as exc:
                raise LogSchemaError(f"exposure_agg must be a pandas aggregation name, got {exposure_agg!r}") from exc
            exp = per_case.reindex(range(n_cases)).fillna(0.0).to_numpy()
            if (exp < 0).any():
                raise LogSchemaError(
                    f"exposure column {exposure_col!r} yields negative case exposure for "
                    f"{int((exp < 0).sum())} cases; the paper assumes exp(σ) >= 0"
                )
            cases["exposure"] = exp.astype(float)
        self.cases = cases

        self.window: tuple[pd.Timestamp | None, pd.Timestamp | None] | None = None
        if window is not None:
            start, end = (self.to_timestamp(window[0]), self.to_timestamp(window[1]))
            if start is not None and end is not None and start > end:
                raise LogSchemaError(f"window start must not be after window end, got {window!r}")
            self.window = (start, end)

        self._count_cache: dict[Labels, pd.Series] = {}
        self._first_cache: dict[Labels, pd.Series] = {}
        self._last_cache: dict[Labels, pd.Series] = {}
        self._recipe_cache: dict[str, str] = {}

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
        return self._index.n_cases

    def __repr__(self) -> str:
        return (
            f"EventLog({len(self.events):,} events, {len(self):,} cases, "
            f"{len(self._index.activity_labels)} activities, attributes={self.case_attributes})"
        )

    @property
    def case_index(self) -> CaseIndex:
        """The immutable per-case index the primitives are computed on; its
        events are in the row order of :attr:`events`."""
        return self._index

    # Read-only views of the index, kept for callers of 0.1.0.
    @property
    def _codes(self) -> np.ndarray:
        return self._index.case_codes

    @property
    def _ts(self) -> np.ndarray:
        return self._index.timestamps

    @property
    def _valid(self) -> np.ndarray:
        return self._index.valid

    @property
    def _starts(self) -> np.ndarray:
        return self._index.starts

    @property
    def case_ids(self) -> pd.Index:
        return self.cases.index

    @property
    def activity_labels(self) -> list[str]:
        return list(self._index.activity_labels)

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
        return pd.Timestamp(start), pd.Timestamp(end)  # type: ignore[arg-type]  # pandas-stubs types quantile() as floating

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

    def to_timestamp(self, value: Any) -> pd.Timestamp | None:
        """``value`` as a :class:`pandas.Timestamp` comparable with the log's
        timestamps: a naive value is localised to the log's time zone, an
        aware value on a naive log is converted to naive UTC. ``None`` stays
        ``None``."""
        return self._index.to_timestamp(value)

    def _to_ts(self, value: Any) -> pd.Timestamp | None:
        warnings.warn(
            "EventLog._to_ts is deprecated and will be removed in 1.0; use EventLog.to_timestamp",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_timestamp(value)

    def _series_dt(self, values: np.ndarray) -> pd.Series:
        return pd.Series(self._index.to_datetime(values), index=self.case_ids)

    def _series(self, values: np.ndarray) -> pd.Series:
        return pd.Series(values, index=self.case_ids)

    # ------------------------------------------------------------------ primitives
    def count(self, activities: str | Iterable[str]) -> pd.Series:
        """``cnt(a, σ)``: occurrences of any of ``activities`` per case."""
        labels = as_labels(activities)
        key = tuple(sorted(labels))
        if key not in self._count_cache:
            self._count_cache[key] = self._series(self._index.count(labels).astype(float))
        return self._count_cache[key]

    def first_ts(self, activities: str | Iterable[str]) -> pd.Series:
        """``t1(a, σ)``: timestamp of the first occurrence, NaT if none."""
        labels = as_labels(activities)
        key = tuple(sorted(labels))
        if key not in self._first_cache:
            self._first_cache[key] = self._series_dt(self._index.first(labels))
        return self._first_cache[key]

    def last_ts(self, activities: str | Iterable[str]) -> pd.Series:
        """Timestamp of the last occurrence, NaT if none."""
        labels = as_labels(activities)
        key = tuple(sorted(labels))
        if key not in self._last_cache:
            self._last_cache[key] = self._series_dt(self._index.last(labels))
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
        idx = self._index
        la, lb = as_labels(a, what="a"), as_labels(b, what="b")
        t_a_int = idx.first(la) if activation == "first" else idx.last(la)
        if response == "first_overall":
            return self._series_dt(t_a_int), self.first_ts(lb)
        anchor = np.where(t_a_int == NAT, _INT_MAX, t_a_int)
        mask_b = idx.mask(lb) & (idx.timestamps >= anchor[idx.case_codes])
        return self._series_dt(t_a_int), self._series_dt(idx.first(mask_b))

    def activation_lags(self, a: str | Iterable[str], b: str | Iterable[str]) -> pd.DataFrame:
        """One row per ``a`` event: its case code, timestamp, and the timestamp
        of the first ``b`` event at or after it in the same case (NaT if none)."""
        idx = self._index
        codes, ts = idx.case_codes, idx.timestamps
        la, lb = as_labels(a, what="a"), as_labels(b, what="b")
        a_pos = np.flatnonzero(idx.mask(la) & idx.valid)
        b_pos = np.flatnonzero(idx.mask(lb) & idx.valid)
        # first position of the run of events sharing (case, timestamp) with each a event
        j = np.searchsorted(b_pos, idx.run_starts()[a_pos])
        has = j < len(b_pos)
        bp = b_pos[np.minimum(j, max(len(b_pos) - 1, 0))] if len(b_pos) else np.zeros(len(a_pos), dtype=int)
        ok = has & (len(b_pos) > 0) & (codes[bp] == codes[a_pos])
        t_b = np.where(ok, ts[bp], NAT)
        out = pd.DataFrame({"code": codes[a_pos], "t_a": ts[a_pos].view("datetime64[ns]"), "t_b": t_b.view("datetime64[ns]")})
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
        idx = self._index
        codes, ts = idx.case_codes, idx.timestamps
        mask = idx.mask(as_labels(activities)) & idx.valid
        undefined = np.zeros(len(self), dtype=bool)
        if after is not None:
            t_after = idx.first(as_labels(after, what="after"))
            undefined |= t_after == NAT
            mask &= ts > np.where(t_after == NAT, _INT_MAX, t_after)[codes]
        if before is not None:
            t_before = idx.first(as_labels(before, what="before"))
            undefined |= t_before == NAT
            mask &= ts < np.where(t_before == NAT, NAT, t_before)[codes]
        cnt = idx.count(mask).astype(float)
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
        vals = pd.to_numeric(self.events[attribute], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        out = self._index.aggregate(vals, as_labels(activities), how=agg, empty=0.0)
        if agg != "sum":
            out[np.isnan(out)] = 0.0
        return self._series(out)

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

    def derive(self, recipes: Iterable[Mapping[str, Any]], *, overwrite: bool = True) -> list[str]:
        """Compute derived case attributes from declarative recipes (see
        :mod:`wise.derive`) and attach them. Returns the attribute names."""
        from .derive import apply_recipes

        return apply_recipes(self, recipes, overwrite=overwrite)

    def trace(self, case_id: Any) -> pd.DataFrame:
        """Events of one case in timestamp order (for drill-down)."""
        pos = self.case_ids.get_loc(case_id)
        if not isinstance(pos, int | np.integer):
            raise LogSchemaError(f"case id {case_id!r} is not unique")
        i = int(pos)
        start = int(self._starts[i])
        n = int(self.cases["n_events"].to_numpy()[i])
        return self.events.iloc[start : start + n]

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
        tied = self._index.tied()
        cases_with_ties = len(np.unique(self._codes[tied]))
        report: dict[str, Any] = {
            "n_events": len(self.events),
            "n_cases": len(self),
            "n_activities": len(self._index.activity_labels),
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
