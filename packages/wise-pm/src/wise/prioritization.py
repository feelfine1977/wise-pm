"""Phase 3: slice-first prioritisation and the Priority Index (paper Sec. IV-E).

For a view ``p`` and slicing function ``g`` (a list of case-attribute
columns):

* slice mean ``μ_s``, global mean ``μ̄`` (unweighted over scored cases),
* Priority Index ``PI_s = n_s · (μ̄ − μ_s)_+`` (or ``E_s`` for exposure),
* shrinkage-stabilised mean ``μ̃_s = n_s/(n_s+γ) μ_s + γ/(n_s+γ) μ̄`` and the
  stabilised index ``PĨ_s = v_s · (μ̄ − μ̃_s)_+``.

Because ``μ̄ − μ̃_s = n_s/(n_s+γ) · (μ̄ − μ_s)``, the stabilised gap is the raw
gap scaled by ``n_s/(n_s+γ)``: a slice with ``n_s = γ`` keeps half of its
observed gap. ``γ`` is therefore in units of cases and should be chosen
relative to typical slice size; :func:`estimate_gamma` offers a data-driven
diagnostic.

:func:`prioritize` returns a :class:`~wise.backlog.Backlog`: the ranked table
plus the specification it was ranked with. The helpers that work on a backlog
take that record and read the slice keys, view and parameters from its
specification. All aggregations return DataFrames indexed by the slice key(s)
so they can be joined with ``.loc``; pass ``as_index=False`` to get the keys as
columns. A case without a value for a slice key falls in the slice labelled
:data:`MISSING_KEY`, which ranks after every keyed slice on a tie.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Literal, TypeVar

import numpy as np
import pandas as pd

from .backlog import Backlog, BacklogSpec
from .constraints import as_list
from .errors import NormError, NotScoredError
from .scoring import ScoreResult
from .selection import select_cases


def _keys(by: str | Sequence[str]) -> list[str]:
    return [by] if isinstance(by, str) else list(by)


def _frame(data: ScoreResult | pd.DataFrame, view: str | None, score_col: str, by: Sequence[str]) -> tuple[pd.DataFrame, str]:
    if isinstance(data, ScoreResult):
        if view is None:
            if len(data.views) != 1:
                raise NormError(f"specify view=...; available: {data.views}")
            view = data.views[0]
        df, score_col = data.frame(view), "score"
    else:
        if score_col not in data.columns:
            raise NormError(f"score column {score_col!r} not in frame")
        df = data
    index_names = [n for n in df.index.names if n]
    clash = [n for n in index_names if n in df.columns]
    if any(c not in df.columns and c in index_names for c in by):
        df = df.drop(columns=clash).reset_index() if clash else df.reset_index()
    elif any(c in clash for c in by):
        df = df.reset_index(drop=True)
    missing = [c for c in by if c not in df.columns]
    if missing:
        raise NormError(f"slice columns not in frame: {missing}")
    return df, score_col


MISSING_KEY = "(missing)"
"""The label of a slice key a case has no value for; a slice key never becomes a NaN index entry (C23)."""


def _label_missing(df: pd.DataFrame, by: Sequence[str]) -> tuple[pd.DataFrame, bool]:
    """``df`` with every missing value of a slice key replaced by :data:`MISSING_KEY`, and whether a key holds the label.

    A categorical key gains the label as its last category; another key column becomes ``object``
    unless it holds strings. A key that already has the value :data:`MISSING_KEY` next to missing
    values is refused, since the two slices could not be told apart. A key labelled before (from
    :func:`case_slices`) counts as labelled when it mixes the label with non-string values, which
    only the key-function sort can order; a string key holding the value is sorted as before.
    """
    labelled: dict[str, pd.Series] = {}
    relabelled = False
    for col in by:
        values = df[col]
        if not values.isna().any():
            if values.dtype == object and (values == MISSING_KEY).any():
                relabelled |= not all(isinstance(v, str) for v in values)
            continue
        categorical = isinstance(values.dtype, pd.CategoricalDtype)
        if (MISSING_KEY in values.cat.categories) if categorical else (values == MISSING_KEY).any():
            raise NormError(f"slice column {col!r} has missing values and the value {MISSING_KEY!r}, which labels them")
        if categorical:
            labelled[col] = values.cat.add_categories([MISSING_KEY]).fillna(MISSING_KEY)
        else:
            base = values if pd.api.types.is_string_dtype(values.dtype) else values.astype(object)
            labelled[col] = base.where(values.notna(), MISSING_KEY)
    return (df.assign(**labelled), True) if labelled else (df, relabelled)


def _missing_last(level: pd.Index) -> pd.Index:
    """Sort key of one slice-key level: key order, the missing label last (categories keep their order)."""
    if isinstance(level.dtype, pd.CategoricalDtype):
        return level
    return pd.Index([(isinstance(v, str) and v == MISSING_KEY, v) for v in level], tupleize_cols=False)


_Table = TypeVar("_Table", pd.DataFrame, "pd.Series[Any]")


def _sort_slices(frame: _Table, labelled: bool) -> _Table:
    """``frame`` in slice-key order; with a labelled missing key that slice comes last, as NaN did."""
    return frame.sort_index(key=_missing_last) if labelled else frame.sort_index()


BACKLOG_COLUMNS: tuple[str, ...] = ("n_cases", "stable_gap", "stable_PI")
"""Columns every backlog frame carries; helpers that take a backlog refuse frames without them (C9)."""


def _check_min_cases(min_cases: Any) -> int:
    """``min_cases`` is a non-negative integer; a fractional value is refused instead of truncated (C28)."""
    if isinstance(min_cases, bool) or not isinstance(min_cases, int | float | np.integer | np.floating):
        raise NormError(f"min_cases must be a non-negative integer, got {min_cases!r}")
    value = float(min_cases)
    if not np.isfinite(value) or value < 0 or value != int(value):
        raise NormError(f"min_cases must be a non-negative integer, got {min_cases!r}")
    return int(value)


def _check_z(z: Any) -> float | None:
    """``z`` is a finite multiplier >= 0; a negative value would turn ``PI_lower`` into an upper bound (C27)."""
    if z is None:
        return None
    try:
        value = float(z)
    except (TypeError, ValueError) as exc:
        raise NormError(f"z must be a finite number >= 0, got {z!r}") from exc
    if not np.isfinite(value) or value < 0:
        raise NormError(f"z must be a finite number >= 0, got {z!r}")
    return value


def _check_baseline(baseline: Any) -> float | None:
    """``baseline`` is a finite reference score; NaN would make every gap and priority undefined (C26)."""
    if baseline is None:
        return None
    try:
        value = float(baseline)
    except (TypeError, ValueError) as exc:
        raise NormError(f"baseline must be a finite number, got {baseline!r}") from exc
    if not np.isfinite(value):
        raise NormError(f"baseline must be a finite number, got {baseline!r}")
    return value


def _check_backlog(frame: Backlog | pd.DataFrame, what: str) -> pd.DataFrame:
    """The table of a backlog record; a frame is checked for the backlog columns (C9).

    A :class:`Backlog` is a backlog by construction, so its table is taken as it is.
    """
    if isinstance(frame, Backlog):
        return frame.table
    if not isinstance(frame, pd.DataFrame):
        raise NormError(f"{what} must be a backlog DataFrame from prioritize(), got {type(frame).__name__}")
    missing = [c for c in BACKLOG_COLUMNS if c not in frame.columns]
    if missing:
        raise NormError(f"{what} is not a backlog from prioritize(): missing columns {missing}")
    return frame


def _table(backlog: Backlog | pd.DataFrame) -> tuple[pd.DataFrame, BacklogSpec | None]:
    """The ranked table and, for a record, its specification."""
    if isinstance(backlog, Backlog):
        return backlog.table, backlog.spec
    return backlog, None


def _from_spec(result: ScoreResult | pd.DataFrame, backlog: Backlog, view: str | None) -> tuple[list[str], str | None]:
    """Slice keys and view of a backlog record, checked against the scores they are used with.

    A record ranked on another norm, or a view given that is not the record's, is refused.
    """
    spec = backlog.spec
    if view is not None and spec.view is not None and view != spec.view:
        raise NormError(f"view {view!r} differs from the backlog's view {spec.view!r}")
    if (
        isinstance(result, ScoreResult)
        and spec.norm_fingerprint is not None
        and result.norm_fingerprint
        and spec.norm_fingerprint != result.norm_fingerprint
    ):
        raise NormError(
            f"the backlog was ranked on another norm (fingerprint {spec.norm_fingerprint[:12]}…, "
            f"result {result.norm_fingerprint[:12]}…)"
        )
    return list(spec.by), view if spec.view is None else spec.view


def _view_of(result: ScoreResult, view: str | Backlog) -> str:
    """The view argument of a driver helper: a view name or a backlog record ranked on ``result``."""
    if not isinstance(view, Backlog):
        return view
    _, name = _from_spec(result, view, None)
    if name is None:
        raise NormError("the backlog was ranked from a per-case frame and names no view; pass the view name")
    return name


def prioritize(
    data: ScoreResult | pd.DataFrame,
    by: str | Sequence[str],
    view: str | None = None,
    gamma: float = 0.0,
    volume: str = "cases",
    min_cases: int = 1,
    z: float | None = None,
    score_col: str = "score",
    baseline: float | None = None,
    as_index: bool = True,
) -> Backlog:
    """Aggregate scored cases into a ranked slice backlog.

    Parameters
    ----------
    data
        A :class:`~wise.scoring.ScoreResult` (then ``view`` selects the score)
        or a per-case DataFrame with the slice columns and ``score_col``.
    by
        Slice key column(s), e.g. ``["company", "spend_area"]``. A single
        column such as the purchasing document gives the document backlog;
        the case id (the frame's index) is allowed too.
    gamma
        Shrinkage constant ``γ ≥ 0``; 0 disables stabilisation.
    volume
        ``"cases"`` (``v_s = n_s``), ``"exposure"`` (``v_s = E_s``, needs an
        ``exposure`` column), or the name of a numeric per-case column to sum.
    min_cases
        Slices with fewer scored cases are dropped from the ranking (after
        the global mean is computed).
    z
        If given, add ``se`` (standard error of the slice mean),
        ``gap_lower = (stable_gap − z·se)_+`` and ``PI_lower``. Slices with a
        single case get ``gap_lower = 0``.
    baseline
        Reference score ``μ̄``. Default: the global mean over scored cases
        (paper). Pass a fixed target (last period's mean, or 1.0) to make
        gaps comparable across periods in the governance loop.

    Returns
    -------
    A :class:`~wise.backlog.Backlog`. Its ``table`` is a DataFrame indexed by
    ``by`` (unless ``as_index=False``), sorted by ``stable_PI`` (ties: larger
    ``n_cases`` first, then key order, the :data:`MISSING_KEY` slice last),
    with ``n_cases, mean_score, volume, [exposure], gap, PI, stable_mean,
    stable_gap, stable_PI, [se, gap_lower, PI_lower], global_mean``. Its ``spec`` holds the slice keys, the view,
    ``gamma``, ``volume``, the requested ``baseline`` and the realised global
    mean, ``min_cases``, ``z``, the score column and the norm fingerprint.
    It aggregates scores and carries no reason per case and constraint; :meth:`ScoreResult.verdicts` does.
    """
    by = _keys(by)
    try:
        gamma = float(gamma)
    except (TypeError, ValueError) as exc:
        raise NormError(f"gamma must be a finite number >= 0, got {gamma!r}") from exc
    if not np.isfinite(gamma) or gamma < 0:
        raise NormError(f"gamma must be a finite number >= 0, got {gamma!r}")
    min_cases = _check_min_cases(min_cases)
    z_value = _check_z(z)
    baseline_value = _check_baseline(baseline)
    df, score_col = _frame(data, view, score_col, by)
    df, labelled = _label_missing(df, by)
    if isinstance(data, ScoreResult):
        view = view if view is not None else data.views[0]
        fingerprint: str | None = data.norm_fingerprint or None
    else:
        view, fingerprint = None, None
    d = df.dropna(subset=[score_col])
    if d.empty:
        raise NotScoredError("no scored cases to aggregate")
    mu_bar = float(d[score_col].mean()) if baseline_value is None else baseline_value

    vol_col = None if volume == "cases" else ("exposure" if volume == "exposure" else volume)
    if vol_col is not None and vol_col not in d.columns:
        raise NormError(f"volume column {vol_col!r} not in frame")

    g = d.groupby(by, dropna=False, observed=True, sort=True)
    agg = g[score_col].agg(n_cases="size", mean_score="mean", sd="std")
    agg["volume"] = g[vol_col].sum().astype(float) if vol_col is not None else agg["n_cases"].astype(float)
    if "exposure" in d.columns:
        agg["exposure"] = g["exposure"].sum().astype(float)
    if vol_col is not None and (agg["volume"] < 0).any():
        raise NormError(f"volume column {vol_col!r} sums to a negative value for some slices")

    n = agg["n_cases"].astype(float)
    agg["gap"] = (mu_bar - agg["mean_score"]).clip(lower=0.0)
    agg["PI"] = agg["volume"] * agg["gap"]
    shrink = n / (n + float(gamma)) if gamma > 0 else pd.Series(1.0, index=agg.index)
    agg["stable_mean"] = shrink * agg["mean_score"] + (1.0 - shrink) * mu_bar
    agg["stable_gap"] = (mu_bar - agg["stable_mean"]).clip(lower=0.0)
    agg["stable_PI"] = agg["volume"] * agg["stable_gap"]
    if z_value is not None:
        se = agg["sd"] / np.sqrt(n)
        agg["se"] = se
        agg["gap_lower"] = (agg["stable_gap"] - z_value * se.fillna(np.inf)).clip(lower=0.0)
        agg["PI_lower"] = agg["volume"] * agg["gap_lower"]
    agg = agg.drop(columns=["sd"])
    agg["global_mean"] = mu_bar

    agg = agg[agg["n_cases"] >= min_cases]
    agg = _sort_slices(agg, labelled).sort_values(["stable_PI", "n_cases"], ascending=[False, False], kind="mergesort")
    spec = BacklogSpec(
        by=tuple(by),
        view=view,
        gamma=float(gamma),
        volume=volume,
        baseline=baseline_value,
        global_mean=mu_bar,
        min_cases=min_cases,
        z=z_value,
        norm_fingerprint=fingerprint,
        score_col=score_col,
    )
    return Backlog(agg if as_index else agg.reset_index(), spec)


def layer_drivers(
    data: ScoreResult | pd.DataFrame,
    by: str | Sequence[str] | Backlog,
    view: str | None = None,
    layers: Sequence[str] | None = None,
    as_index: bool = True,
) -> pd.DataFrame:
    """Mean layer contribution ``Δ_λ`` per slice and its delta to the global mean.

    ``by`` is the slice key(s) or a backlog record from :func:`prioritize`,
    whose slice keys and view are used (a record ranked on another norm is
    refused). Returns a DataFrame indexed by slice with ``n_cases``, per layer
    ``<layer>`` (mean contribution) and ``<layer>__delta`` (minus the global
    mean), and ``dominant_layer`` (the largest positive delta). Positive
    deltas mark mechanisms more pronounced in the slice than in the log.
    It aggregates scores and carries no reason per case and constraint; :meth:`ScoreResult.verdicts` does.
    """
    by, view = _from_spec(data, by, view) if isinstance(by, Backlog) else (_keys(by), view)
    df, score_col = _frame(data, view, "score", by)
    df, labelled = _label_missing(df, by)
    prefix = "contrib__"
    if isinstance(data, ScoreResult):
        layer_cols = [f"{prefix}{layer}" for layer in (as_list(layers) or data.norm.layer_ids)]
    else:
        layer_cols = [c for c in df.columns if c.startswith(prefix)]
    if not layer_cols:
        raise NormError("no contrib__<layer> columns in frame")
    d = df.dropna(subset=[score_col])
    if d.empty:
        raise NotScoredError("no scored cases to aggregate")
    global_mean = d[layer_cols].mean()
    g = d.groupby(by, dropna=False, observed=True, sort=True)
    out = g[layer_cols].mean()
    out.insert(0, "n_cases", g.size())
    if labelled:
        out = _sort_slices(out, labelled)
    deltas = out[layer_cols] - global_mean
    for col in layer_cols:
        out[f"{col[len(prefix) :]}__delta"] = deltas[col]
    out = out.rename(columns={c: c[len(prefix) :] for c in layer_cols})
    delta_cols = [f"{c[len(prefix) :]}__delta" for c in layer_cols]
    best = out[delta_cols].idxmax(axis=1).astype(str).str.replace("__delta", "", regex=False)
    out["dominant_layer"] = best.where(out[delta_cols].max(axis=1) > 0, None)
    return out if as_index else out.reset_index()


def constraint_drivers(
    result: ScoreResult,
    view: str | Backlog,
    where: dict[str, Any] | pd.Series | None = None,
) -> pd.DataFrame:
    """Which constraints carry the penalty inside a slice (or the whole log)?

    ``view`` is a view name or a backlog record from :func:`prioritize` ranked
    on ``result`` (its view is used). ``where`` is ``{attribute: value}`` (equality) or ``{attribute: [values]}``
    (membership), the case id allowed under the name of the case index, a rule
    of the selection grammar (:mod:`wise.selection`), or a boolean mask over
    cases. Returns one row per constraint with the mean effective
    penalty ``w̃ ν``, the mean violation over evaluated cases, the share of
    evaluated cases with a non-zero violation, the share of cases in scope,
    and the share actually evaluated.
    The shares do not say why a case is not in scope or not evaluated
    (unknown scope, a missing anchor, ...); :meth:`ScoreResult.verdicts` carries
    the reason per case and constraint.
    """
    view = _view_of(result, view)
    mask = select_cases(result.cases, where, log=result.log) & result.scores[view].notna()
    V = result.violations[mask]
    S = result.in_scope[mask]
    pen = result.penalties(view)[mask]
    rows = []
    for nc in result.norm.constraints:
        v = V[nc.id]
        ev = v.notna()
        rows.append(
            {
                "constraint": nc.id,
                "layer": nc.layer,
                "type": nc.constraint.type,
                "mean_penalty": float(pen[nc.id].mean()) if len(pen) else np.nan,
                "mean_violation": float(v[ev].mean()) if ev.any() else np.nan,
                "share_violated": float((v[ev] > 0).mean()) if ev.any() else np.nan,
                "share_in_scope": float(S[nc.id].mean()) if len(S) else np.nan,
                "share_evaluated": float(ev.mean()) if len(ev) else np.nan,
                "description": nc.description,
            }
        )
    out = pd.DataFrame(rows).set_index("constraint")
    return out.sort_values(["mean_penalty", "mean_violation"], ascending=False, kind="mergesort")


def penalty_mass(
    result: ScoreResult,
    view: str | Backlog,
    by: str | Sequence[str] | None = None,
    where: dict[str, Any] | pd.Series | None = None,
    as_index: bool = True,
) -> pd.DataFrame:
    """Sum of case penalties ``1 − S`` per slice (optionally within a filter),
    with ``share``, ``cum_share`` and ``rank`` — the vendor Pareto of the paper.
    ``view`` is a view name or a backlog record ranked on ``result``, whose
    view and (unless ``by`` is given) slice keys are used. ``where`` filters
    cases as in :func:`constraint_drivers`.
    It aggregates scores and carries no reason per case and constraint; :meth:`ScoreResult.verdicts` does.
    """
    if isinstance(view, Backlog) and by is None:
        by = list(view.spec.by)
    view = _view_of(result, view)
    if by is None:
        raise NormError("penalty_mass needs the slice keys: pass by=... or a backlog record as view")
    by = _keys(by)
    f, score_col = _frame(result, view, "score", by)
    f, labelled = _label_missing(f, by)
    mask = select_cases(result.cases, where, log=result.log).to_numpy() & f[score_col].notna().to_numpy()
    d = f[mask]
    if d.empty:
        raise NotScoredError("no scored cases match the filter")
    d = d.assign(_penalty=1.0 - d[score_col])
    g = d.groupby(by, dropna=False, observed=True, sort=True)
    out = pd.DataFrame({"n_cases": g.size(), "penalty_mass": g["_penalty"].sum(), "mean_penalty": g["_penalty"].mean()})
    out = _sort_slices(out, labelled).sort_values(["penalty_mass", "n_cases"], ascending=False, kind="mergesort")
    total = float(out["penalty_mass"].sum())
    out["share"] = out["penalty_mass"] / total if total > 0 else 0.0
    out["cum_share"] = out["share"].cumsum()
    out["rank"] = np.arange(1, len(out) + 1)
    return out if as_index else out.reset_index()


def case_slices(
    data: ScoreResult | pd.DataFrame,
    by: str | Sequence[str] | Backlog,
    view: str | None = None,
    score_col: str = "score",
) -> pd.DataFrame:
    """The slice each case falls in, as :func:`prioritize` forms it, and the case's score.

    ``by`` is the slice key(s) or a backlog record, whose slice keys, view and
    score column are used (a record ranked on another norm is refused).
    Returns one row per case, indexed like the cases of ``data``, with the key
    columns (a missing key as :data:`MISSING_KEY`) and the score column
    (``"score"`` for a result; missing for an unscored case).
    It carries no reason per case and constraint; :meth:`ScoreResult.verdicts` does.
    """
    if isinstance(by, Backlog):
        score_col = by.spec.score_col
        by, view = _from_spec(data, by, view)
    else:
        by = _keys(by)
    df, score_col = _frame(data, view, score_col, by)
    df, _ = _label_missing(df, by)
    out = df[[*by, score_col]]
    out.index = data.cases.index if isinstance(data, ScoreResult) else data.index
    return out


def pareto(backlog: Backlog | pd.DataFrame, metric: str = "stable_PI") -> pd.DataFrame:
    """Add ``rank``, ``share`` and ``cum_share`` of ``metric`` mass (index kept)."""
    backlog, _ = _table(backlog)
    if metric not in backlog.columns:
        raise NormError(f"metric {metric!r} not in backlog")
    out = backlog.sort_values(metric, ascending=False, kind="mergesort").copy()
    total = float(out[metric].sum())
    out["rank"] = np.arange(1, len(out) + 1)
    out["share"] = out[metric] / total if total > 0 else 0.0
    out["cum_share"] = out["share"].cumsum()
    return out


def concentration(
    backlog: Backlog | pd.DataFrame, thresholds: float | Sequence[float] = (0.8, 0.95), metric: str = "stable_PI"
) -> pd.DataFrame:
    """How many top-ranked slices carry ``threshold`` of total ``metric`` mass?"""
    p = pareto(backlog, metric)
    rows = []
    for t in as_list(thresholds):
        k = min(int(np.searchsorted(p["cum_share"].to_numpy(), t) + 1), len(p)) if len(p) else 0
        rows.append({"threshold": t, "top_k": k, "share_of_slices": (k / len(p)) if len(p) else np.nan})
    return pd.DataFrame(rows).set_index("threshold")


def _top_keys(backlog: pd.DataFrame, k: int, metric: str, by: Sequence[str] | None) -> set[tuple[str, ...]]:
    pos = backlog[backlog[metric] > 0].sort_values(metric, ascending=False, kind="mergesort").head(k)
    if by:
        return set(map(tuple, pos.reset_index()[list(by)].astype(str).to_numpy()))
    idx = pos.index
    return {tuple(map(str, key if isinstance(key, tuple) else (key,))) for key in idx}


def top_k_overlap(
    backlog_a: Backlog | pd.DataFrame,
    backlog_b: Backlog | pd.DataFrame,
    k: int = 20,
    metric: str = "stable_PI",
    by: Sequence[str] | None = None,
) -> float:
    """Jaccard overlap of the top-``k`` positive-mass slices of two backlogs.

    Slices are matched on ``by``, by default on the slice keys of the backlog
    records (two records sliced by different keys are refused) or else on the
    index of the frames."""
    table_a, spec_a = _table(backlog_a)
    table_b, spec_b = _table(backlog_b)
    if by is None and spec_a is not None and spec_b is not None:
        if spec_a.by != spec_b.by:
            raise NormError(f"the backlogs have different slice keys: {list(spec_a.by)} and {list(spec_b.by)}")
        by = list(spec_a.by)
    a, b = _top_keys(table_a, k, metric, by), _top_keys(table_b, k, metric, by)
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def view_agreement(
    result: ScoreResult,
    by: str | Sequence[str] | Backlog,
    k: int = 20,
    gamma: float = 0.0,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    metric: str = "stable_PI",
) -> pd.DataFrame:
    """Pairwise top-``k`` Jaccard overlap of backlogs and case-score correlation
    across views. ``by`` is the slice key(s) or a backlog record ranked on
    ``result``; with a record every view is ranked with the record's slice
    keys, ``gamma``, ``volume``, ``baseline``, ``min_cases`` and ``z`` (so
    ``metric="PI_lower"`` works; ``gamma`` is then ignored). ``method`` is ``"pearson"`` (default), ``"spearman"`` or
    ``"kendall"``; the latter two need SciPy (``pip install "wise-pm[stats]"``).
    It aggregates scores and carries no reason per case and constraint; :meth:`ScoreResult.verdicts` does.
    """
    if method not in ("pearson", "spearman", "kendall"):
        raise NormError(f"method must be 'pearson', 'spearman' or 'kendall', got {method!r}")
    if isinstance(by, Backlog):
        spec = by.spec
        keys, _ = _from_spec(result, by, None)
        backlogs = {
            v: prioritize(
                result,
                keys,
                view=v,
                gamma=spec.gamma,
                volume=spec.volume,
                min_cases=spec.min_cases,
                z=spec.z,
                baseline=spec.baseline,
            )
            for v in result.views
        }
    else:
        keys = _keys(by)
        backlogs = {v: prioritize(result, keys, view=v, gamma=gamma) for v in result.views}
    rows = []
    views = result.views
    for i, a in enumerate(views):
        for b in views[i + 1 :]:
            rows.append(
                {
                    "view_a": a,
                    "view_b": b,
                    f"top{k}_overlap": top_k_overlap(backlogs[a].table, backlogs[b].table, k, metric),
                    "score_correlation": _corr(result.scores[a], result.scores[b], method),
                }
            )
    return pd.DataFrame(rows).set_index(["view_a", "view_b"])


def _corr(a: pd.Series, b: pd.Series, method: Literal["pearson", "spearman", "kendall"]) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(a.corr(b, method=method))


def hotspot_table(backlog: Backlog | pd.DataFrame, top: int = 12, drivers: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach the paper's hotspot typology to the top of a backlog.

    Among the top-``top`` slices with positive stable PI, the slice with the
    highest volume rank relative to its gap rank is the *reservoir*
    (priority from scale), the mirrored extreme is the *severity* hotspot
    (priority from a large gap at small volume), and the rest are
    *mechanism* hotspots. If ``drivers`` (from :func:`layer_drivers`, same
    index) is given, ``dominant_layer`` is joined in.
    """
    table = _check_backlog(backlog, "backlog")
    pos = table[table["stable_PI"] > 0].head(top).copy()
    pos["hotspot"] = "mechanism"
    if len(pos) >= 2:
        vr = pos["n_cases"].rank(method="average")
        gr = pos["stable_gap"].rank(method="average")
        res_score, sev_score = vr - gr, gr - vr
        i_res = res_score.idxmax()
        if float(res_score.max()) > 0:  # the value at idxmax
            pos.loc[i_res, "hotspot"] = "reservoir"
        sev_order = sev_score.drop(index=i_res).sort_values(ascending=False)
        if len(sev_order) and sev_order.iloc[0] > 0:
            pos.loc[sev_order.index[0], "hotspot"] = "severity"
    if drivers is not None and "dominant_layer" in drivers.columns:
        pos["dominant_layer"] = drivers["dominant_layer"].reindex(pos.index)
    return pos


COMPARED_FIELDS: tuple[str, ...] = ("by", "view", "gamma", "baseline", "volume")
"""The fields of :class:`~wise.backlog.BacklogSpec` two compared periods must share."""


def compare_periods(
    previous: Backlog | pd.DataFrame,
    current: Backlog | pd.DataFrame,
    *,
    intended: str | Sequence[str] = (),
) -> pd.DataFrame:
    """Join two backlogs and report the change in gap and PI per slice.

    Two backlog records must have been ranked alike: their specifications may
    not differ in slice keys (``by``), ``view``, ``gamma``, the requested
    ``baseline`` or ``volume``, unless each differing field is named in
    ``intended``; otherwise :class:`~wise.errors.NormError` names the fields.
    A record is aligned on its slice keys, also when ranked with
    ``as_index=False``. Frames carry no specification and are compared as
    given, on their index. Use a common
    ``baseline`` in :func:`prioritize` to compare gaps against one reference.

    ``presence`` is ``"both"``, ``"new"`` (ranked in ``current`` only) or
    ``"gone"`` (ranked in ``previous`` only). A slice ranked in one period only
    has no measured change: its ``gap_change`` and ``PI_change`` are missing,
    like the columns of the period it is absent from (C24).
    """
    accepted = _keys(intended)
    unknown = [f for f in accepted if f not in COMPARED_FIELDS]
    if unknown:
        raise NormError(f"intended names fields that are not compared: {unknown}; compared are {list(COMPARED_FIELDS)}")
    if isinstance(previous, Backlog) and isinstance(current, Backlog):
        _check_comparable(previous.spec, current.spec, accepted)
    previous = _by_slice(previous, "previous")
    current = _by_slice(current, "current")
    cols = list(BACKLOG_COLUMNS)
    p = previous[cols].add_suffix("_prev")
    c = current[cols].add_suffix("_now")
    out = p.join(c, how="outer")
    out["gap_change"] = out["stable_gap_now"] - out["stable_gap_prev"]
    out["PI_change"] = out["stable_PI_now"] - out["stable_PI_prev"]
    in_prev, in_now = out["n_cases_prev"].notna().to_numpy(), out["n_cases_now"].notna().to_numpy()
    out["presence"] = np.select([in_prev & in_now, in_now], ["both", "new"], "gone")
    return out.sort_values("stable_PI_now", ascending=False, kind="mergesort")


def _by_slice(backlog: Backlog | pd.DataFrame, what: str) -> pd.DataFrame:
    """The table of a backlog indexed by its slice keys; a record ranked with ``as_index=False`` is re-indexed."""
    table = _check_backlog(backlog, what)
    if isinstance(backlog, Backlog):
        keys = list(backlog.spec.by)
        if list(table.index.names) != keys and all(k in table.columns for k in keys):
            return table.set_index(keys)
    return table


def _check_comparable(previous: BacklogSpec, current: BacklogSpec, intended: Sequence[str]) -> None:
    """Refuse two specifications that differ in a compared field not named in ``intended``."""

    def shown(value: Any) -> Any:
        return list(value) if isinstance(value, tuple) else value

    refused = [f for f in COMPARED_FIELDS if getattr(previous, f) != getattr(current, f) and f not in intended]
    if refused:
        differences = "; ".join(f"{f} {shown(getattr(previous, f))!r} and {shown(getattr(current, f))!r}" for f in refused)
        raise NormError(
            f"the periods were ranked with different specifications: {differences}; "
            f"pass intended={refused!r} if the difference is intended"
        )


def estimate_gamma(
    data: ScoreResult | pd.DataFrame,
    by: str | Sequence[str],
    view: str | None = None,
    score_col: str = "score",
) -> float:
    """Method-of-moments estimate of the shrinkage constant ``γ = σ²/τ²``.

    Under a normal–normal random-effects model with within-slice variance
    ``σ²`` and between-slice variance ``τ²``, the posterior slice mean is
    exactly the stabilised mean with ``γ = σ²/τ²``. ``σ²`` is the pooled
    within-slice variance and ``τ²`` the variance of slice means minus the
    sampling noise ``σ² · mean(1/n_s)``. Returns ``inf`` when slices do not
    differ beyond noise. Treat the value as a diagnostic: it depends on the
    norm and on the slice granularity.
    """
    by = _keys(by)
    df, score_col = _frame(data, view, score_col, by)
    d = df.dropna(subset=[score_col])
    stats = d.groupby(by, dropna=False, observed=True)[score_col].agg(["size", "mean", "var"])
    stats = stats[stats["size"] >= 2]
    if len(stats) < 2:
        raise NotScoredError("estimate_gamma needs at least two slices with two or more scored cases")
    dof = stats["size"] - 1
    sigma2 = float((stats["var"] * dof).sum() / dof.sum())
    tau2 = float(stats["mean"].var() - sigma2 * (1.0 / stats["size"]).mean())
    if tau2 <= 0:
        return float("inf")
    return sigma2 / tau2
