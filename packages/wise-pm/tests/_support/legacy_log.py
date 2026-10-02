"""The 0.1.0 event-log construction, frozen as an oracle for the parse step.

:func:`legacy_build` is the body of ``EventLog.__init__`` as it stood before the
schema and the parse step were put in front of it, minus the observation
window (which the parse step does not touch). It returns the pieces the log
is made of, so a test can compare them with what ``EventLog`` builds today.
Do not edit it to make a comparison pass: it is the reference.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from wise.case_index import CaseIndex, event_order
from wise.constraints import as_list
from wise.errors import LogSchemaError

_RESERVED = {"n_events", "first_ts", "last_ts", "exposure", "score"}


@dataclass
class LegacyLog:
    events: pd.DataFrame
    cases: pd.DataFrame
    index: CaseIndex
    tz: Any
    n_missing_timestamps: int
    n_null_activities: int


def legacy_build(
    events: pd.DataFrame,
    *,
    case_col: str = "case:concept:name",
    activity_col: str = "concept:name",
    timestamp_col: str = "time:timestamp",
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
    missing_timestamps: str = "raise",
    keep_columns: Iterable[str] | None = None,
    dedupe: bool = False,
) -> LegacyLog:
    case_attributes = as_list(case_attributes)
    keep_columns = None if keep_columns is None else as_list(keep_columns)
    required = [case_col, activity_col, timestamp_col, *case_attributes]
    for c in (exposure_col, order_col, event_id_col, lifecycle_col):
        if c is not None:
            required.append(c)
    missing = [c for c in required if c not in events.columns]
    if missing:
        raise LogSchemaError(f"event log is missing columns {missing}")
    for attr in case_attributes:
        if attr in _RESERVED or attr.startswith(("contrib__", "score__")):
            raise LogSchemaError(f"{attr!r} is reserved")
    if missing_timestamps not in ("raise", "drop", "keep"):
        raise LogSchemaError(f"missing_timestamps must be 'raise', 'drop' or 'keep', got {missing_timestamps!r}")

    df = events
    if lifecycle_col is not None:
        keep = {str(t).lower() for t in as_list(keep_transitions)}
        df = df[df[lifecycle_col].astype(str).str.lower().isin(keep)]
    if dedupe:
        df = df.drop_duplicates(subset=[case_col, activity_col, timestamp_col])

    case_series = df[case_col]
    n_null_case = int(case_series.isna().sum())
    if n_null_case:
        raise LogSchemaError(f"{n_null_case} events have a null case id; drop or fill them first")
    if isinstance(case_series.dtype, pd.CategoricalDtype):
        case_series = case_series.astype(case_series.cat.categories.dtype)
    codes, uniques = pd.factorize(case_series, sort=True)
    codes = codes.astype(np.int64)

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
        raise LogSchemaError(f"{n_missing_ts} timestamps are null or unparseable")
    if n_missing_ts and missing_timestamps == "drop":
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

    tie = pd.factorize(df[order_col], sort=True)[0] if order_col is not None else None
    order = event_order(codes, ts_int, tie)
    n_cases = len(uniques)

    cols = list(df.columns)
    if keep_columns is not None:
        wanted = set(required) | set(keep_columns)
        cols = [c for c in cols if c in wanted]
    ev = df.iloc[order][cols].reset_index(drop=True)
    ev[timestamp_col] = ts.to_numpy()[order] if tz is None else ts.iloc[order].reset_index(drop=True)

    act = ev[activity_col]
    n_null_activities = int(act.isna().sum())
    if isinstance(act.dtype, pd.CategoricalDtype):
        act_codes = act.cat.codes.to_numpy().astype(np.int64)
        act_labels = pd.Index(act.cat.categories.astype(str))
    else:
        act_codes, act_uniques = pd.factorize(act, sort=False)
        act_codes = act_codes.astype(np.int64)
        act_labels = pd.Index(pd.Index(act_uniques).astype(str))
    index = CaseIndex(codes[order], ts_int[order], act_codes, act_labels, n_cases=n_cases, tz=tz)

    cases = pd.DataFrame(index=pd.Index(uniques, name=case_col))
    cases["n_events"] = index.count()
    cases["first_ts"] = index.to_datetime(index.first())
    cases["last_ts"] = index.to_datetime(index.last())
    for attr in case_attributes:
        cases[attr] = ev[attr].groupby(index.case_codes).first().reindex(range(n_cases)).to_numpy()
    if exposure_col is not None:
        vals = pd.to_numeric(ev[exposure_col], errors="coerce")
        try:
            per_case = vals.groupby(index.case_codes).agg(exposure_agg)
        except (AttributeError, TypeError, ValueError) as exc:
            raise LogSchemaError(f"exposure_agg must be a pandas aggregation name, got {exposure_agg!r}") from exc
        exp = per_case.reindex(range(n_cases)).fillna(0.0).to_numpy()
        if (exp < 0).any():
            raise LogSchemaError(f"exposure column {exposure_col!r} yields negative case exposure")
        cases["exposure"] = exp.astype(float)
    return LegacyLog(ev, cases, index, tz, n_missing_ts, n_null_activities)
