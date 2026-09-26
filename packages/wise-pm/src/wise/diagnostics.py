"""Validation diagnostics for the governance loop (paper Sec. V-C, Table XII).

These helpers separate likely process signals from observation-window and
logging artefacts *before* action hypotheses are stated:

* :func:`observation_window`, :func:`timestamp_outliers` — a robust window
  that ignores sentinel dates,
* :func:`right_censored`, :func:`left_truncated` — cases that look open (or
  incomplete) only because the log ends (or starts),
* :func:`event_replication` — events sharing one timestamp inside a case,
* :func:`cross_case_replication` — the same source event copied into several
  cases (e.g. header-level postings replicated per item),
* :func:`gap_retained`, :func:`validation_table` — how much of a slice's
  stabilised gap survives when flagged cases are excluded.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .constraints import as_labels, as_list
from .errors import LogSchemaError, NotScoredError
from .log import EventLog
from .prioritization import _frame, _keys, prioritize
from .scoring import ScoreResult


def observation_window(log: EventLog, q: float = 0.001) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Robust ``(start, end)`` of the observation window (see
    :meth:`EventLog.observation_window`)."""
    return log.observation_window(q)


def timestamp_outliers(log: EventLog, window: tuple[Any, Any] | None = None, q: float = 0.001) -> pd.Series:
    """Boolean per event: timestamp outside the observation window."""
    start, end = (log._to_ts(window[0]), log._to_ts(window[1])) if window is not None else log.observation_window(q)
    ts = log.events[log.timestamp_col]
    return ((ts < start) | (ts > end)).rename("timestamp_outlier")


def _resolve_end(log: EventLog, window_end: Any, window: pd.Timedelta, q: float) -> pd.Timestamp:
    if window_end is not None:
        return log._to_ts(window_end)  # type: ignore[return-value]
    return log.censoring_end(tolerance=window, q=q)


def right_censored(
    log: EventLog,
    closure: str | Sequence[str],
    window: str | pd.Timedelta = "60D",
    opened_by: str | Sequence[str] | None = None,
    window_end: Any = None,
    q: float = 0.001,
) -> pd.Series:
    """Boolean per case: still open and active within ``window`` of the end.

    A case is right-censored if it lacks every ``closure`` activity, its last
    event lies within ``window`` before ``window_end`` (or after it), and, if
    ``opened_by`` is given, at least one of those activities occurred (only
    invoice-bearing cases can be open invoices).

    ``window_end`` defaults to the log's explicit window end, else to the
    end of :func:`observation_window` (with a warning when the last raw
    timestamp lies more than ``window`` beyond it, as with placeholder dates
    far outside the extraction period).
    """
    win = pd.Timedelta(window)
    end = _resolve_end(log, window_end, win, q)
    horizon = end - win
    has_closure = log.count(as_labels(closure, what="closure")) > 0
    active_late = log.cases["last_ts"] >= horizon
    flag = (~has_closure) & active_late
    if opened_by is not None:
        flag &= log.count(as_labels(opened_by, what="opened_by")) > 0
    return flag.rename("right_censored")


def left_truncated(
    log: EventLog,
    opening: str | Sequence[str],
    window: str | pd.Timedelta = "60D",
    window_start: Any = None,
    q: float = 0.001,
) -> pd.Series:
    """Boolean per case: lacks every ``opening`` activity and starts within
    ``window`` after the window start — its beginning probably lies before
    the log."""
    win = pd.Timedelta(window)
    start = (
        log._to_ts(window_start)
        if window_start is not None
        else (log.window[0] if log.window and log.window[0] is not None else log.observation_window(q)[0])
    )
    has_opening = log.count(as_labels(opening, what="opening")) > 0
    early = log.cases["first_ts"] <= start + win  # type: ignore[operator]
    return ((~has_opening) & early).rename("left_truncated")


def event_replication(log: EventLog) -> pd.DataFrame:
    """Per case: ``n_events``, ``n_distinct_ts``, ``replication_ratio``
    (events per distinct timestamp) and ``replicated_share`` (share of events
    whose timestamp is shared with another event of the same case)."""
    codes, ts, valid = log._codes, log._ts, log._valid
    n = len(log)
    same_prev = (codes[1:] == codes[:-1]) & (ts[1:] == ts[:-1]) & valid[1:] & valid[:-1]
    tied = np.zeros(len(codes), dtype=bool)
    tied[1:] |= same_prev
    tied[:-1] |= same_prev
    n_events = np.bincount(codes, minlength=n).astype(float)
    n_valid = np.bincount(codes[valid], minlength=n).astype(float)
    n_dup = np.bincount(codes[1:][same_prev], minlength=n).astype(float)
    distinct = n_valid - n_dup
    replicated = np.bincount(codes[tied], minlength=n).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(distinct > 0, n_valid / distinct, np.nan)
        share = np.where(n_events > 0, replicated / n_events, np.nan)
    return pd.DataFrame(
        {
            "n_events": n_events.astype(int),
            "n_distinct_ts": distinct.astype(int),
            "replication_ratio": ratio,
            "replicated_share": share,
        },
        index=log.case_ids,
    )


def cross_case_replication(
    log: EventLog,
    group_col: str,
    keys: Sequence[str] | None = None,
) -> pd.Series:
    """Share of a case's events that have an identical copy (same ``keys``,
    default activity and timestamp, plus the event id if configured) in
    *another* case of the same ``group_col`` (e.g. a purchasing document).

    High shares mean header-level events were replicated into item-level
    traces, which inflates counts and multiplies one deviation by the
    number of items when rolling up to the group.
    """
    if group_col not in log.events.columns:
        raise LogSchemaError(f"group column {group_col!r} not in log")
    keys = as_list(keys) if keys is not None else [log.activity_col, log.timestamp_col]
    if log.event_id_col is not None and log.event_id_col not in keys:
        keys.append(log.event_id_col)
    ev = log.events
    n_cases_per_key = ev.groupby([group_col, *keys], dropna=False, observed=True)[log.case_col].transform("nunique")
    replicated = (n_cases_per_key > 1).to_numpy()
    share = np.bincount(log._codes[replicated], minlength=len(log)) / np.maximum(np.bincount(log._codes, minlength=len(log)), 1)
    return pd.Series(share, index=log.case_ids, name="cross_case_replicated_share")


def gap_retained(
    result: ScoreResult,
    view: str,
    by: str | Sequence[str],
    exclude: pd.Series,
    gamma: float = 0.0,
) -> pd.DataFrame:
    """Recompute each slice's stabilised gap without the ``exclude``-flagged
    cases, against the *same* baseline, and report the retained share.

    Returns slices × ``[n_cases, n_kept, stable_gap, stable_gap_kept,
    retained]``. ``retained`` can exceed 1 when the excluded cases were a
    slice's best ones; a slice whose gap collapses was a window artefact.
    """
    by = _keys(by)
    full = prioritize(result, by, view=view, gamma=gamma)
    baseline = float(full.attrs["baseline"])
    frame, score_col = _frame(result, view, "score", by)
    keep = ~exclude.reindex(result.cases.index, fill_value=False).astype(bool).to_numpy()
    if not (keep & frame[score_col].notna().to_numpy()).any():
        raise NotScoredError("every scored case is excluded")
    kept = prioritize(frame[keep], by, gamma=gamma, baseline=baseline, score_col=score_col)
    out = pd.DataFrame(
        {
            "n_cases": full["n_cases"],
            "n_kept": kept["n_cases"].reindex(full.index).fillna(0).astype(int),
            "stable_gap": full["stable_gap"],
            "stable_gap_kept": kept["stable_gap"].reindex(full.index).fillna(0.0),
        }
    )
    out["retained"] = np.where(out["stable_gap"] > 0, out["stable_gap_kept"] / out["stable_gap"], np.nan)
    return out.sort_values("stable_gap", ascending=False, kind="mergesort")


def validation_table(
    result: ScoreResult,
    view: str,
    by: str | Sequence[str],
    censored: pd.Series | None = None,
    replication: pd.DataFrame | pd.Series | None = None,
    gamma: float = 0.0,
    ratio_flag: float = 2.0,
    top: int | None = None,
) -> pd.DataFrame:
    """The paper's Table XII per slice: censored share, replicated share,
    gap retained without censored cases, and a reading.

    ``censored`` is a boolean per case (e.g. :func:`right_censored`);
    ``replication`` is :func:`event_replication`'s frame (cases with
    ``replication_ratio > ratio_flag`` count as replicated) or a boolean
    Series (e.g. from :func:`cross_case_replication` thresholded).
    """
    by = _keys(by)
    full = prioritize(result, by, view=view, gamma=gamma)
    frame, score_col = _frame(result, view, "score", by)
    scored = frame[score_col].notna().to_numpy()
    out = pd.DataFrame({"n_cases": full["n_cases"], "stable_gap": full["stable_gap"], "stable_PI": full["stable_PI"]})
    if censored is not None:
        c = censored.reindex(result.cases.index, fill_value=False).astype(bool)
        flags = frame[scored].assign(_c=c.to_numpy()[scored])
        out["censored_share"] = flags.groupby(by, dropna=False, observed=True)["_c"].mean()
        gr = gap_retained(result, view, by, exclude=c, gamma=gamma)
        out["stable_gap_kept"] = gr["stable_gap_kept"]
        out["retained"] = gr["retained"]
    if replication is not None:
        r = replication["replication_ratio"] > ratio_flag if isinstance(replication, pd.DataFrame) else replication.astype(bool)
        r = r.reindex(result.cases.index, fill_value=False)
        flags = frame[scored].assign(_r=r.to_numpy()[scored])
        out["replicated_share"] = flags.groupby(by, dropna=False, observed=True)["_r"].mean()

    def reading(row: pd.Series) -> str:
        notes = []
        if "retained" in row and pd.notna(row["retained"]) and row["retained"] < 0.5:
            notes.append("gap collapses without censored cases: window artefact")
        if "replicated_share" in row and pd.notna(row["replicated_share"]) and row["replicated_share"] >= 0.5:
            notes.append("high event replication: verify logging before acting")
        return "; ".join(notes) if notes else "stable signal"

    out["reading"] = out.apply(reading, axis=1)
    out = out.sort_values("stable_PI", ascending=False, kind="mergesort")
    return out.head(top) if top else out
