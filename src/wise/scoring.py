"""Phase 2: constraint evaluation and view-specific case scoring (paper Sec. IV-D).

Given an :class:`~wise.log.EventLog` and a :class:`~wise.norm.Norm`, this
module produces

* the violation matrix ``V`` (cases × constraints) with ``ν_c(σ) ∈ [0, 1]``
  and ``NaN`` where the constraint is not applicable (``σ ∉ C_app``) or not
  evaluable (a skipped lag endpoint, a missing metric attribute),
* the applicability-aware case score per view::

      S^(p)(σ) = 1 − Σ_{c∈C_app(σ)} w_c^(p) ν_c(σ) / Σ_{c∈C_app(σ)} w_c^(p)

  (``NaN`` when no positively weighted constraint applies: the case is
  *unscored* and excluded from aggregation), and
* the layer contributions ``Δ_λ^(p)(σ) = Σ_{c∈C_λ} w̃_c,σ ν_c(σ)`` with the
  per-case effective weights ``w̃``, so that ``1 − S = Σ_λ Δ_λ`` exactly.

Two scoring modes exist. ``"layer_balanced"`` (default) first averages the
applicable constraints within each layer with their within-layer weights,
then averages the applicable layers with the layer weights, so that every
applicable layer keeps its full weight. ``"flat"`` renormalises the raw
constraint weights over the applicable set. The two coincide when all
constraints of a layer apply to a case. The mode is part of the norm
(``Norm.scoring_mode``) and can be overridden per call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from ._version import __version__
from .constraints import Balance, Exclusion, Lag, Metric, Precedence, Presence, Singularity, in_units, sat
from .errors import NormError, NotScoredError
from .log import EventLog
from .norm import SCORING_MODES, Norm, NormConstraint


# ------------------------------------------------------------------ signals
def _lag_missing(v: pd.Series, lag: pd.Series, t_a: pd.Series, c: Lag, log: EventLog) -> pd.Series:
    a_missing = t_a.isna()
    b_missing = (~a_missing) & lag.isna()
    if c.missing_a == "violate":
        v = v.where(~a_missing, 1.0)
    if c.missing_b == "violate":
        v = v.where(~b_missing, 1.0)
    elif c.missing_b == "censor" and c.delta is not None:
        end = log.censoring_end(tolerance=(c.delta + c.width) * c.timedelta_unit)
        so_far = in_units(end - t_a, c.unit)
        v = v.where(~b_missing, sat(so_far, c.delta, c.width))
    return v


def _evaluate_lag(log: EventLog, c: Lag) -> pd.Series:
    if c.activation in ("first", "last"):
        activation = cast(Literal["first", "last"], c.activation)
        t_a, t_b = log.first_after(c.a, c.b, activation=activation, response=c.response)
        lag = in_units(t_b - t_a, c.unit)
        if c.response == "first_overall":
            lag = lag.where(lag >= 0)  # b before the activation counts as missing
        v = sat(lag, c.delta, c.width) if c.delta is not None else lag * 0.0
        return _lag_missing(v, lag, t_a, c, log)

    # activation == "each": one violation per a-event, averaged per case
    pairs = log.activation_lags(c.a, c.b)
    t_a_case = log.first_ts(c.a)
    if pairs.empty:
        v = pd.Series(np.nan, index=log.case_ids)
        return _lag_missing(v, v, t_a_case, c, log)
    lag = in_units(pairs["t_b"] - pairs["t_a"], c.unit)
    vi = sat(lag, c.delta, c.width) if c.delta is not None else lag * 0.0
    missing = lag.isna()
    if c.missing_b == "violate":
        vi = vi.where(~missing, 1.0)
    elif c.missing_b == "censor" and c.delta is not None:
        end = pd.Timestamp(log.censoring_end(tolerance=(c.delta + c.width) * c.timedelta_unit))
        end = end.tz_convert("UTC").tz_localize(None) if end.tzinfo is not None else end
        so_far = in_units(end - pairs["t_a"], c.unit)
        vi = vi.where(~missing, sat(so_far, c.delta, c.width))
    per_case = vi.groupby(pairs["code"].to_numpy()).mean()
    v = pd.Series(per_case.reindex(range(len(log))).to_numpy(), index=log.case_ids)
    a_missing = t_a_case.isna()
    if c.missing_a == "violate":
        v = v.where(~a_missing, 1.0)
    return v


def _evaluate(log: EventLog, nc: NormConstraint) -> pd.Series:
    c = nc.constraint
    if isinstance(c, Presence):
        cnt = log.count(c.activity)
        return pd.Series(1.0 - np.minimum(cnt.to_numpy() / c.m, 1.0), index=log.case_ids)
    if isinstance(c, Exclusion):
        cnt = log.count_scoped(c.activity, after=c.after, before=c.before) if (c.after or c.before) else log.count(c.activity)
        return (cnt > 0).astype(float).where(cnt.notna())
    if isinstance(c, Singularity):
        cnt = log.count_scoped(c.activity, after=c.after, before=c.before) if (c.after or c.before) else log.count(c.activity)
        return sat(cnt, c.k, c.K)
    if isinstance(c, Lag):
        return _evaluate_lag(log, c)
    if isinstance(c, Precedence):
        n_before = log.count_before(c.b, c.a)
        v = sat(n_before, c.k, c.K)
        if c.missing_a == "violate":
            v = v.where(n_before.notna(), 1.0)
        if c.missing_b == "skip":
            v = v.where(log.count(c.b) > 0)
        return v
    if isinstance(c, Balance):
        tot_x = log.total(c.attr_x, c.activities_x, agg=c.agg)
        tot_y = log.total(c.attr_y, c.activities_y, agg=c.agg)
        if (tot_x < 0).any() or (tot_y < 0).any():
            raise NormError(
                f"constraint {nc.id!r}: balance totals must be non-negative (paper Sec. IV-A); "
                "net credit notes first or take absolute values"
            )
        denom = np.maximum(np.maximum(tot_x.to_numpy(), tot_y.to_numpy()), c.eps)
        d = pd.Series((tot_x - tot_y).abs().to_numpy() / denom, index=log.case_ids)
        return sat(d, c.tau, c.width)
    if isinstance(c, Metric):
        x = log.attribute(c.attribute)
        return sat(x, c.threshold, c.width) if c.direction == "high" else sat(c.threshold - x, 0.0, c.width)
    raise TypeError(f"unsupported constraint {type(c).__name__}")  # pragma: no cover


def evaluate_constraint(log: EventLog, nc: NormConstraint) -> pd.Series:
    """Evaluate one norm constraint on every case → ``ν_c(σ)``.

    Returns a float Series aligned to ``log.cases``; ``NaN`` marks cases to
    which the constraint does not apply or on which it is not evaluable.
    """
    v = pd.Series(np.asarray(_evaluate(log, nc), dtype=float), index=log.case_ids, name=nc.id)
    if nc.applicability:
        v = v.where(nc.applies_to(log.cases, log), np.nan)
    return v


def violation_matrix(log: EventLog, norm: Norm, *, return_scope: bool = False) -> Any:
    """``V``: cases × constraints (``ν_c(σ)``, NaN if not applicable/evaluable).

    With ``return_scope=True`` also return the boolean scope matrix ``S``
    (``σ ∈ C_app``), which distinguishes "out of scope" from "not evaluable".
    """
    cols: dict[str, pd.Series] = {}
    scope: dict[str, pd.Series] = {}
    for nc in norm.constraints:
        v = pd.Series(np.asarray(_evaluate(log, nc), dtype=float), index=log.case_ids, name=nc.id)
        s = nc.applies_to(log.cases, log) if nc.applicability else pd.Series(True, index=log.case_ids)
        cols[nc.id] = v.where(s, np.nan)
        scope[nc.id] = s
    V = pd.DataFrame(cols, index=log.case_ids)
    if return_scope:
        return V, pd.DataFrame(scope, index=log.case_ids)
    return V


# ------------------------------------------------------------------- scores
@dataclass(eq=False)
class ScoreResult:
    """Output of :func:`score`.

    Attributes
    ----------
    cases
        Case attributes (index = case id), incl. ``exposure`` if provided.
    violations
        ``V`` matrix, NaN where not applicable or not evaluable.
    in_scope
        Boolean cases × constraints: ``σ ∈ C_app`` (scope only).
    scores
        Cases × views, ``S^(p)(σ)``; NaN = unscored.
    contributions
        ``{view: cases × layers}`` with ``Δ_λ^(p)(σ)``.
    log
        The scored :class:`EventLog` (for drill-down to traces).
    """

    norm: Norm
    cases: pd.DataFrame
    violations: pd.DataFrame
    in_scope: pd.DataFrame
    scores: pd.DataFrame
    contributions: dict[str, pd.DataFrame]
    mode: str = "flat"
    log: EventLog | None = field(default=None, repr=False)
    norm_fingerprint: str = ""
    wise_version: str = __version__
    _weights: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _eff_cache: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        return (
            f"ScoreResult({len(self.scores):,} cases × {self.violations.shape[1]} constraints, "
            f"views={self.views}, mode={self.mode!r}, applicability={self.applicability_density():.3f})"
        )

    @property
    def applicable(self) -> pd.DataFrame:
        """Boolean cases × constraints: evaluated (in scope and evaluable)."""
        return self.violations.notna()

    @property
    def views(self) -> list[str]:
        return list(self.scores.columns)

    def effective_weights(self, view: str) -> pd.DataFrame:
        """Per-case effective weights ``w̃_c,σ`` (cases × constraints); rows of
        scored cases sum to 1."""
        if view not in self._eff_cache:
            M = self.violations.notna().to_numpy(dtype=float)
            w_eff = _effective_weights(M, self._weights[view], self.norm, self.mode)
            self._eff_cache[view] = pd.DataFrame(
                np.nan_to_num(w_eff), index=self.violations.index, columns=self.violations.columns
            )
        return self._eff_cache[view]

    def penalties(self, view: str) -> pd.DataFrame:
        """Per-constraint penalty ``w̃_c,σ ν_c(σ)`` (cases × constraints); sums
        to ``1 − S^(p)(σ)`` per case."""
        return self.effective_weights(view) * self.violations.fillna(0.0)

    def frame(self, view: str | None = None) -> pd.DataFrame:
        """Wide per-case table for aggregation and drill-down.

        Columns: case attributes, ``exposure`` (if any), then for the chosen
        view ``score`` and ``contrib__<layer>``; with ``view=None`` every view
        gets ``score__<view>`` and ``contrib__<view>__<layer>``.
        """
        if view is not None:
            if view not in self.contributions:
                raise NormError(f"unknown view {view!r}; available: {self.views}")
            extra = pd.concat([self.scores[view].rename("score"), self.contributions[view].add_prefix("contrib__")], axis=1)
            return pd.concat([self.cases, extra], axis=1)
        parts: list[pd.Series | pd.DataFrame] = []
        for v in self.views:
            parts.append(self.scores[v].rename(f"score__{v}"))
            parts.append(self.contributions[v].add_prefix(f"contrib__{v}__"))
        return pd.concat([self.cases, *parts], axis=1) if parts else self.cases.copy()

    def summary(self) -> pd.DataFrame:
        """Per view: scored cases, mean score ``μ̄^(p)``, mean layer contributions."""
        rows = []
        for v in self.views:
            s = self.scores[v]
            row: dict[str, Any] = {"view": v, "n_scored": int(s.notna().sum()), "mean_score": float(s.mean())}
            for layer in self.norm.layer_ids:
                row[f"contrib__{layer}"] = float(self.contributions[v][layer][s.notna()].mean())
            rows.append(row)
        return pd.DataFrame(rows).set_index("view")

    def applicability_density(self, scope: bool = False) -> float:
        """Average share of constraints per case that were evaluated
        (``scope=False``, default) or that are in scope (``scope=True``)."""
        m = self.in_scope if scope else self.applicable
        return float(m.mean(axis=1).mean()) if len(m) else float("nan")

    def check_decomposition(self, atol: float = 1e-9) -> float:
        """Max |Σ_λ Δ_λ − (1 − S)| over scored cases; raises if above ``atol``."""
        worst = 0.0
        for v in self.views:
            s = self.scores[v]
            lhs = self.contributions[v].sum(axis=1)[s.notna()]
            rhs = (1.0 - s)[s.notna()]
            worst = max(worst, float((lhs - rhs).abs().max()) if len(lhs) else 0.0)
        if worst > atol:
            raise AssertionError(f"layer decomposition violated: max error {worst:.3e}")
        return worst

    def worst_cases(self, view: str, n: int = 20, where: dict[str, Any] | pd.Series | None = None) -> pd.DataFrame:
        """The ``n`` lowest-scoring cases (optionally within a slice) with their
        layer contributions."""
        f = self.frame(view)
        mask = _where_mask(self.cases, where)
        f = f[mask & f["score"].notna()]
        return f.sort_values("score", kind="mergesort").head(n)

    def trace(self, case_id: Any) -> pd.DataFrame:
        """Events of one case (requires the log to be attached)."""
        if self.log is None:
            raise NotScoredError("no event log attached to this result")
        return self.log.trace(case_id)


def _where_mask(cases: pd.DataFrame, where: dict[str, Any] | pd.Series | None) -> pd.Series:
    mask = pd.Series(True, index=cases.index)
    if where is None:
        return mask
    if isinstance(where, dict):
        for k, v in where.items():
            if k in cases.columns:
                mask &= cases[k] == v
            elif k == cases.index.name:
                mask &= pd.Series(cases.index == v, index=cases.index)
            else:
                raise NormError(f"unknown case attribute {k!r} in where; known: {list(cases.columns)}")
        return mask
    return mask & where.reindex(cases.index, fill_value=False).astype(bool)


def _effective_weights(M: np.ndarray, w: np.ndarray, norm: Norm, mode: str) -> np.ndarray:
    if mode == "flat":
        w_app = M @ w
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(w_app[:, None] > 0, M * w[None, :] / w_app[:, None], np.nan)
    return _layer_balanced_weights(M, w, norm)


def _layer_balanced_weights(M: np.ndarray, w: np.ndarray, norm: Norm) -> np.ndarray:
    n = M.shape[0]
    cids = norm.constraint_ids
    layer_of = norm.layer_of
    w_eff = np.zeros_like(M)
    a_app = np.zeros(n)
    for layer in norm.layer_ids:
        idx = np.array([i for i, cid in enumerate(cids) if layer_of[cid] == layer], dtype=int)
        a = float(w[idx].sum()) if len(idx) else 0.0
        if len(idx) == 0 or a <= 0:
            continue
        b_app = M[:, idx] * w[idx][None, :]
        b_sum = b_app.sum(axis=1)
        layer_app = b_sum > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            w_eff[:, idx] = np.where(b_sum[:, None] > 0, b_app / b_sum[:, None], 0.0) * a
        a_app += layer_app * a
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a_app[:, None] > 0, w_eff / a_app[:, None], np.nan)


def score(
    log: EventLog,
    norm: Norm,
    views: Sequence[str] | str | None = None,
    *,
    mode: str | None = None,
    derive: bool = True,
) -> ScoreResult:
    """Score every case of ``log`` against ``norm`` under one or more views.

    Parameters
    ----------
    views
        View names to score; default all views of the norm.
    mode
        ``"layer_balanced"`` or ``"flat"``; default ``norm.scoring_mode``.
    derive
        Compute the norm's ``derived_attributes`` on the log first. A recipe
        that was already computed on this log is reused; a changed recipe
        is recomputed and replaces the attribute of the same name.
    """
    norm.validate()
    views = [views] if isinstance(views, str) else list(dict.fromkeys(views or norm.view_names))
    for v in views:
        norm.get_view(v)
    mode = mode or norm.scoring_mode
    if mode not in SCORING_MODES:
        raise NormError(f"mode must be one of {SCORING_MODES}, got {mode!r}")
    if derive and norm.derived_attributes:
        log.derive(norm.derived_attributes, overwrite=False)

    V, S = violation_matrix(log, norm, return_scope=True)
    M = V.notna().to_numpy(dtype=float)
    V0 = V.fillna(0.0).to_numpy(dtype=float)
    cids = list(V.columns)
    layer_ids = norm.layer_ids
    onehot = np.array([[1.0 if norm.layer_of[cid] == layer else 0.0 for layer in layer_ids] for cid in cids])

    score_cols: list[np.ndarray] = []
    contributions: dict[str, pd.DataFrame] = {}
    weights: dict[str, np.ndarray] = {}
    for view in views:
        w = norm.weight_vector(view).to_numpy(dtype=float)
        weights[view] = w
        w_eff = _effective_weights(M, w, norm, mode)
        penalty_c = np.nan_to_num(w_eff) * V0
        unscored = np.all(np.isnan(w_eff), axis=1)
        s = 1.0 - penalty_c.sum(axis=1)
        s[unscored] = np.nan
        score_cols.append(s)
        contrib = penalty_c @ onehot
        contrib[unscored, :] = np.nan
        contributions[view] = pd.DataFrame(contrib, index=V.index, columns=layer_ids)

    scores = pd.DataFrame(np.column_stack(score_cols) if score_cols else np.empty((len(V), 0)), index=V.index, columns=views)
    return ScoreResult(
        norm=norm,
        cases=log.cases.copy(),
        violations=V,
        in_scope=S,
        scores=scores,
        contributions=contributions,
        mode=mode,
        log=log,
        norm_fingerprint=norm.fingerprint(),
        _weights=weights,
    )
