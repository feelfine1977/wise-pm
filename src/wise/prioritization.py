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

All aggregations return DataFrames indexed by the slice key(s) so they can
be joined with ``.loc``; pass ``as_index=False`` to get the keys as columns.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd

from .constraints import as_list
from .errors import NormError, NotScoredError
from .scoring import ScoreResult, _where_mask

if TYPE_CHECKING:  # pragma: no cover
    from .explain.baseline import AssessmentContext, BaselineSpec, ResolvedBaseline
    from .oc.evaluation import OCScoreResult

#: The prefix of the per-layer contribution columns of
#: :meth:`wise.ScoreResult.frame`.
_CONTRIB = "contrib__"

#: Keys a ranking copies from the frame it ranked. A frame carries its own
#: warnings — that its rows pool several assessment unit types, say — and this
#: is the one point where they would otherwise be dropped, because ``agg`` is a
#: fresh frame built by ``groupby``.
CARRIED_FRAME_ATTRS = ("heterogeneous_unit_types",)


def _keys(by: str | Sequence[str]) -> list[str]:
    return [by] if isinstance(by, str) else list(by)


def _is_scored_result(data: Any) -> bool:
    """Whether ``data`` is a scored result rather than a bare per-unit frame.

    Structural, not nominal. :class:`~wise.scoring.ScoreResult` and
    :class:`wise.oc.evaluation.OCScoreResult` are deliberately different types
    with no common base — one promises a case :class:`~wise.log.EventLog`, the
    other an object log — but both carry the four things a ranking needs: the
    views, the wide per-unit frame, the norm's layers and the mode. Naming the
    second one here would pull the whole object model into ``import wise``.
    """
    if isinstance(data, pd.DataFrame):
        return False
    return isinstance(data, ScoreResult) or all(hasattr(data, name) for name in ("views", "frame", "norm", "mode", "scores"))


def _context(
    data: ScoreResult | OCScoreResult | pd.DataFrame,
    view: str | None,
    *,
    layer_ids: Sequence[str] = (),
    unit_type: str | None = None,
) -> AssessmentContext:
    """What this assessment declares about itself, for comparator checking.

    A :class:`~wise.scoring.ScoreResult` knows its mode, its layers and its
    norm; a bare score frame knows none of them, and neither knows what the
    scored rows are meant to *count* — the assessment unit is declared by the
    caller, in :func:`~wise.explain.explain_priority`. Missing declarations are
    reported as *unverifiable* rather than assumed to match.
    """
    from .explain.baseline import AssessmentContext

    if _is_scored_result(data):
        return AssessmentContext.from_result(cast("ScoreResult | OCScoreResult", data), view, unit_type=unit_type)
    return AssessmentContext(view=view, unit_type=unit_type, layer_ids=tuple(layer_ids))


def _resolve(
    spec: BaselineSpec,
    data: ScoreResult | OCScoreResult | pd.DataFrame,
    view: str | None,
    *,
    population_mean: float,
    population_size: int,
    population_profile: dict[str, float] | None = None,
    layer_ids: Sequence[str] = (),
) -> ResolvedBaseline:
    from .explain.baseline import resolve_baseline

    return resolve_baseline(
        spec,
        _context(data, view, layer_ids=layer_ids),
        population_mean=population_mean,
        population_size=population_size,
        population_profile=population_profile,
    )


def _reject_two_comparators(baseline: float | None, baseline_spec: BaselineSpec | None) -> None:
    if baseline is not None and baseline_spec is not None:
        raise NormError(
            "baseline= and baseline_spec= are two comparators for one backlog: pass the scalar for the "
            "historical interface, or the specification, which also carries the layer profile the "
            "explanation needs — not both"
        )


def _frame(
    data: ScoreResult | OCScoreResult | pd.DataFrame, view: str | None, score_col: str, by: Sequence[str]
) -> tuple[pd.DataFrame, str]:
    if _is_scored_result(data):
        scored = cast("ScoreResult | OCScoreResult", data)
        if view is None:
            if len(scored.views) != 1:
                raise NormError(f"specify view=...; available: {scored.views}")
            view = scored.views[0]
        df, score_col = scored.frame(view), "score"
    else:
        frame = cast("pd.DataFrame", data)
        if score_col not in frame.columns:
            raise NormError(f"score column {score_col!r} not in frame")
        df = frame
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


def prioritize(
    data: ScoreResult | OCScoreResult | pd.DataFrame,
    by: str | Sequence[str],
    view: str | None = None,
    gamma: float = 0.0,
    volume: str = "cases",
    min_cases: int = 1,
    z: float | None = None,
    score_col: str = "score",
    baseline: float | None = None,
    as_index: bool = True,
    *,
    baseline_spec: BaselineSpec | None = None,
) -> pd.DataFrame:
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
    baseline_spec
        A :class:`~wise.explain.BaselineSpec` — the *shared* comparator, which
        also carries the layer profile, the view, the normalisation and the
        layer meanings, so that :func:`layer_drivers` and
        :func:`~wise.explain.explain_priority` explain the very comparison
        this ranking used. Keyword-only, and mutually exclusive with
        ``baseline``: two comparators for one backlog is the error this
        argument exists to prevent.

    Returns
    -------
    DataFrame indexed by ``by`` (unless ``as_index=False``), sorted by
    ``stable_PI`` (ties: larger ``n_cases`` first, then key order), with
    ``n_cases, volume, [exposure], mean_score, gap, PI, stable_mean,
    stable_gap, stable_PI, [se, gap_lower, PI_lower], global_mean`` and the
    parameters in ``DataFrame.attrs``.

    Notes
    -----
    The ``global_mean`` column keeps its historical name and position, and it
    holds whatever reference was used: with ``baseline`` or ``baseline_spec``
    that reference **need not be global**. The comparator's identity and kind
    are recorded in ``DataFrame.attrs`` under ``baseline_id`` and
    ``comparator`` whenever a specification was supplied; a call without one
    keeps exactly its historical attributes.
    """
    by = _keys(by)
    _reject_two_comparators(baseline, baseline_spec)
    try:
        gamma = float(gamma)
    except (TypeError, ValueError) as exc:
        raise NormError(f"gamma must be a finite number >= 0, got {gamma!r}") from exc
    if not np.isfinite(gamma) or gamma < 0:
        raise NormError(f"gamma must be a finite number >= 0, got {gamma!r}")
    df, score_col = _frame(data, view, score_col, by)
    d = df.dropna(subset=[score_col])
    if d.empty:
        raise NotScoredError("no scored cases to aggregate")
    # the population mean is computed over every scored case, before the
    # minimum-support filter below removes small slices from the ranking
    resolved = None
    if baseline_spec is None:
        mu_bar = float(d[score_col].mean()) if baseline is None else float(baseline)
    else:
        resolved = _resolve(baseline_spec, data, view, population_mean=float(d[score_col].mean()), population_size=len(d))
        mu_bar = resolved.reference_score

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
    if z is not None:
        se = agg["sd"] / np.sqrt(n)
        agg["se"] = se
        agg["gap_lower"] = (agg["stable_gap"] - float(z) * se.fillna(np.inf)).clip(lower=0.0)
        agg["PI_lower"] = agg["volume"] * agg["gap_lower"]
    agg = agg.drop(columns=["sd"])
    agg["global_mean"] = mu_bar

    agg = agg[agg["n_cases"] >= int(min_cases)]
    agg = agg.sort_index().sort_values(["stable_PI", "n_cases"], ascending=[False, False], kind="mergesort")
    agg.attrs.update({"view": view, "gamma": float(gamma), "baseline": mu_bar, "volume": volume, "by": by})
    # `agg` is a fresh frame, so a warning the ranked frame carried about
    # itself — that its rows pool several assessment unit types, say — stops
    # here unless it is copied on. The backlog is the artefact a reader
    # receives, and it is the last place that can say the volumes it summed
    # are not one volume.
    for name in CARRIED_FRAME_ATTRS:
        if name in df.attrs:
            agg.attrs[name] = df.attrs[name]
    if resolved is not None:
        # only a call that supplied a specification gains metadata: a historical
        # call keeps exactly the attributes it always had
        agg.attrs.update(
            {
                "baseline_id": resolved.spec.baseline_id,
                "comparator": resolved.spec.kind.value,
                "comparator_resolved_from": resolved.resolved_from,
                # what the comparator supports; prioritize itself resolves only
                # the scalar, and layer_drivers resolves the profile
                "explanation_kind": resolved.spec.explanation_kind,
                "baseline_spec": resolved.spec.to_dict(),
            }
        )
    return agg if as_index else agg.reset_index()


def layer_drivers(
    data: ScoreResult | OCScoreResult | pd.DataFrame,
    by: str | Sequence[str],
    view: str | None = None,
    layers: Sequence[str] | None = None,
    as_index: bool = True,
    *,
    baseline_spec: BaselineSpec | None = None,
) -> pd.DataFrame:
    """Mean layer contribution ``Δ_λ`` per slice and its delta to the comparator.

    Returns a DataFrame indexed by slice with ``n_cases``, per layer
    ``<layer>`` (mean contribution) and ``<layer>__delta`` (minus the
    comparator's layer penalty), and ``dominant_layer`` (the largest positive
    delta). Positive deltas mark mechanisms more pronounced in the slice than
    in the comparator.

    Without ``baseline_spec`` the comparator is the current scored population,
    exactly as before. With one, it is the shared specification
    :func:`prioritize` ranked against, so the two no longer describe different
    comparisons of one backlog:

    * a compatible **complete profile** gives signed deltas that sum to the
      same score gap the priority used;
    * a **scalar-only** comparator keeps the absolute layer means, sets every
      ``<layer>__delta`` to null and ``dominant_layer`` to ``None``, and
      reports ``attrs["reference_profile_available"] = False`` with
      ``attrs["unavailable_reason"] = "reference_profile_unavailable"``. The
      current population's profile is *not* substituted: it would explain a
      comparison nobody made;
    * an **incompatible** comparator raises
      :class:`~wise.explain.BaselineError` naming every mismatch.
    """
    by = _keys(by)
    df, score_col = _frame(data, view, "score", by)
    prefix = _CONTRIB
    if _is_scored_result(data):
        norm = cast("ScoreResult | OCScoreResult", data).norm
        layer_cols = [f"{prefix}{layer}" for layer in (as_list(layers) or norm.layer_ids)]
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
    layer_names = [c[len(prefix) :] for c in layer_cols]
    resolved = None
    reference = global_mean
    if baseline_spec is not None:
        resolved = _resolve(
            baseline_spec,
            data,
            view,
            population_mean=float(d[score_col].mean()),
            population_size=len(d),
            population_profile={name: float(global_mean[col]) for name, col in zip(layer_names, layer_cols)},
            layer_ids=layer_names,
        )
        profile = resolved.reference_layer_penalties
        reference = (
            pd.Series(np.nan, index=layer_cols)
            if profile is None
            else pd.Series([profile[name] for name in layer_names], index=layer_cols)
        )
    deltas = out[layer_cols] - reference
    for col in layer_cols:
        out[f"{col[len(prefix) :]}__delta"] = deltas[col]
    out = out.rename(columns={c: c[len(prefix) :] for c in layer_cols})
    delta_cols = [f"{name}__delta" for name in layer_names]
    if resolved is not None and resolved.reference_layer_penalties is None:
        # nulls, not zeros, and no dominant layer: nothing was contrasted
        out["dominant_layer"] = pd.Series([None] * len(out), index=out.index, dtype=object)
    else:
        best = out[delta_cols].idxmax(axis=1).astype(str).str.replace("__delta", "", regex=False)
        out["dominant_layer"] = best.where(out[delta_cols].max(axis=1) > 0, None)
    if resolved is not None:
        available = resolved.reference_layer_penalties is not None
        out.attrs.update(
            {
                "view": view,
                "by": by,
                "baseline_id": resolved.spec.baseline_id,
                "comparator": resolved.spec.kind.value,
                "comparator_resolved_from": resolved.resolved_from,
                "reference_score": resolved.reference_score,
                "reference_layer_penalties": resolved.reference_layer_penalties,
                "reference_profile_available": available,
                "explanation_kind": resolved.explanation_kind,
                "unavailable_reason": None if available else "reference_profile_unavailable",
                "baseline_spec": resolved.spec.to_dict(),
            }
        )
    return out if as_index else out.reset_index()


def constraint_drivers(
    result: ScoreResult | OCScoreResult,
    view: str,
    where: dict[str, Any] | pd.Series | None = None,
) -> pd.DataFrame:
    """Which constraints carry the penalty inside a slice (or the whole log)?

    ``where`` is ``{attribute: value}`` (the case id is allowed) or a boolean
    mask over cases. Returns one row per constraint with the mean effective
    penalty ``w̃ ν``, the mean violation over evaluated cases, the share of
    evaluated cases with a non-zero violation, the share of cases in scope,
    and the share actually evaluated.
    """
    mask = _where_mask(result.unit_table, where) & result.scores[view].notna()
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
                "type": nc.type,
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
    view: str,
    by: str | Sequence[str],
    where: dict[str, Any] | pd.Series | None = None,
    as_index: bool = True,
) -> pd.DataFrame:
    """Sum of case penalties ``1 − S`` per slice (optionally within a filter),
    with ``share``, ``cum_share`` and ``rank`` — the vendor Pareto of the paper."""
    by = _keys(by)
    f, score_col = _frame(result, view, "score", by)
    mask = _where_mask(result.cases, where).to_numpy() & f[score_col].notna().to_numpy()
    d = f[mask]
    if d.empty:
        raise NotScoredError("no scored cases match the filter")
    d = d.assign(_penalty=1.0 - d[score_col])
    g = d.groupby(by, dropna=False, observed=True, sort=True)
    out = pd.DataFrame({"n_cases": g.size(), "penalty_mass": g["_penalty"].sum(), "mean_penalty": g["_penalty"].mean()})
    out = out.sort_index().sort_values(["penalty_mass", "n_cases"], ascending=False, kind="mergesort")
    total = float(out["penalty_mass"].sum())
    out["share"] = out["penalty_mass"] / total if total > 0 else 0.0
    out["cum_share"] = out["share"].cumsum()
    out["rank"] = np.arange(1, len(out) + 1)
    return out if as_index else out.reset_index()


def pareto(backlog: pd.DataFrame, metric: str = "stable_PI") -> pd.DataFrame:
    """Add ``rank``, ``share`` and ``cum_share`` of ``metric`` mass (index kept)."""
    if metric not in backlog.columns:
        raise NormError(f"metric {metric!r} not in backlog")
    out = backlog.sort_values(metric, ascending=False, kind="mergesort").copy()
    total = float(out[metric].sum())
    out["rank"] = np.arange(1, len(out) + 1)
    out["share"] = out[metric] / total if total > 0 else 0.0
    out["cum_share"] = out["share"].cumsum()
    return out


def concentration(
    backlog: pd.DataFrame, thresholds: float | Sequence[float] = (0.8, 0.95), metric: str = "stable_PI"
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
    backlog_a: pd.DataFrame,
    backlog_b: pd.DataFrame,
    k: int = 20,
    metric: str = "stable_PI",
    by: Sequence[str] | None = None,
) -> float:
    """Jaccard overlap of the top-``k`` positive-mass slices of two backlogs."""
    a, b = _top_keys(backlog_a, k, metric, by), _top_keys(backlog_b, k, metric, by)
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def view_agreement(
    result: ScoreResult,
    by: str | Sequence[str],
    k: int = 20,
    gamma: float = 0.0,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    metric: str = "stable_PI",
) -> pd.DataFrame:
    """Pairwise top-``k`` Jaccard overlap of backlogs and case-score correlation
    across views. ``method`` is ``"pearson"`` (default), ``"spearman"`` or
    ``"kendall"``; the latter two need SciPy (``pip install "wise-pm[stats]"``)."""
    by = _keys(by)
    if method not in ("pearson", "spearman", "kendall"):
        raise NormError(f"method must be 'pearson', 'spearman' or 'kendall', got {method!r}")
    backlogs = {v: prioritize(result, by, view=v, gamma=gamma) for v in result.views}
    rows = []
    views = result.views
    for i, a in enumerate(views):
        for b in views[i + 1 :]:
            rows.append(
                {
                    "view_a": a,
                    "view_b": b,
                    f"top{k}_overlap": top_k_overlap(backlogs[a], backlogs[b], k, metric),
                    "score_correlation": _corr(result.scores[a], result.scores[b], method),
                }
            )
    return pd.DataFrame(rows).set_index(["view_a", "view_b"])


def _corr(a: pd.Series, b: pd.Series, method: Literal["pearson", "spearman", "kendall"]) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(a.corr(b, method=method))


def hotspot_table(backlog: pd.DataFrame, top: int = 12, drivers: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach the paper's hotspot typology to the top of a backlog.

    Among the top-``top`` slices with positive stable PI, the slice with the
    highest volume rank relative to its gap rank is the *reservoir*
    (priority from scale), the mirrored extreme is the *severity* hotspot
    (priority from a large gap at small volume), and the rest are
    *mechanism* hotspots. If ``drivers`` (from :func:`layer_drivers`, same
    index) is given, ``dominant_layer`` is joined in.
    """
    pos = backlog[backlog["stable_PI"] > 0].head(top).copy()
    pos["hotspot"] = "mechanism"
    if len(pos) >= 2:
        vr = pos["n_cases"].rank(method="average")
        gr = pos["stable_gap"].rank(method="average")
        res_score, sev_score = vr - gr, gr - vr
        i_res = res_score.idxmax()
        if res_score.loc[i_res] > 0:
            pos.loc[i_res, "hotspot"] = "reservoir"
        sev_order = sev_score.drop(index=i_res).sort_values(ascending=False)
        if len(sev_order) and sev_order.iloc[0] > 0:
            pos.loc[sev_order.index[0], "hotspot"] = "severity"
    if drivers is not None and "dominant_layer" in drivers.columns:
        pos["dominant_layer"] = drivers["dominant_layer"].reindex(pos.index)
    return pos


def compare_periods(previous: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Join two backlogs (same ``by``) and report the change in gap and PI per
    slice. Use a common ``baseline`` in :func:`prioritize` for comparability."""
    cols = ["n_cases", "stable_gap", "stable_PI"]
    p = previous[cols].add_suffix("_prev")
    c = current[cols].add_suffix("_now")
    out = p.join(c, how="outer")
    out["gap_change"] = out["stable_gap_now"].fillna(0) - out["stable_gap_prev"].fillna(0)
    out["PI_change"] = out["stable_PI_now"].fillna(0) - out["stable_PI_prev"].fillna(0)
    return out.sort_values("stable_PI_now", ascending=False, kind="mergesort")


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
