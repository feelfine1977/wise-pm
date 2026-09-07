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
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd

from ._aggregation import LayerAssignment, aggregate, effective_weights
from ._version import __version__
from .constraints import Balance, Exclusion, Lag, Metric, Precedence, Presence, Singularity, in_units, sat
from .errors import EvidenceUnavailableError, NormError, NotScoredError
from .log import EventLog
from .norm import SCORING_MODES, Norm, NormConstraint

if TYPE_CHECKING:  # pragma: no cover
    from .evidence.calibration import CalibrationRecord
    from .evidence.manifest import RunManifest
    from .evidence.models import EvidencePacket

#: Capture modes accepted by :func:`score`. ``"none"`` is the default and keeps
#: the historical code path; ``"summary"`` records one evidence row per check
#: with its measurements; ``"full"`` additionally materialises bounded witnesses.
EVIDENCE_MODES = ("none", "summary", "full")

# Reason codes recorded by the detailed evaluation path. These strings are the
# values of :class:`wise.evidence.models.ReasonCode`; ``scoring`` keeps them as
# plain strings so that it never has to import the evidence package, and
# ``tests/extensions/test_evidence_capture.py`` pins the correspondence.
_OBSERVED = "observed"
_SATISFIED_VACUOUSLY = "satisfied_vacuously"
_VIOLATION_MISSING_A = "policy_violation_missing_activation"
_VIOLATION_MISSING_B = "policy_violation_missing_response"
_OPEN_WINDOW = "open_observation_window"
_SKIP_MISSING_A = "skipped_missing_activation"
_SKIP_MISSING_B = "skipped_missing_response"
_SKIP_MISSING_ANCHOR = "skipped_missing_anchor"
_MISSING_ATTRIBUTE = "missing_attribute"


@dataclass
class _Detail:
    """Primitive measurements and reason codes behind one constraint's ``ν``.

    Produced by the *same* evaluation code that produces the violation
    Series, so an evidence row can never disagree with the number it explains.
    """

    measurements: list[tuple[str, pd.Series, str, str]] = field(default_factory=list)
    reasons: pd.Series | None = None
    horizon: pd.Timestamp | None = None

    def add(self, name: str, values: pd.Series, unit: str, kind: str) -> None:
        self.measurements.append((name, values, unit, kind))


# ------------------------------------------------------------------ signals
def _lag_missing(v: pd.Series, lag: pd.Series, t_a: pd.Series, c: Lag, log: EventLog, detail: _Detail | None = None) -> pd.Series:
    a_missing = t_a.isna()
    b_missing = (~a_missing) & lag.isna()
    if detail is not None:
        detail.reasons = pd.Series(_OBSERVED, index=v.index, dtype=object)
        detail.reasons = detail.reasons.where(~a_missing, _VIOLATION_MISSING_A if c.missing_a == "violate" else _SKIP_MISSING_A)
    if c.missing_a == "violate":
        v = v.where(~a_missing, 1.0)
    if c.missing_b == "violate":
        v = v.where(~b_missing, 1.0)
    elif c.missing_b == "censor" and c.delta is not None:
        end = log.censoring_end(tolerance=(c.delta + c.width) * c.timedelta_unit)
        so_far = in_units(end - t_a, c.unit)
        v = v.where(~b_missing, sat(so_far, c.delta, c.width))
        if detail is not None:
            detail.horizon = pd.Timestamp(end)
            detail.add("elapsed_at_horizon", so_far.where(b_missing), c.unit, "duration")
    if detail is not None and detail.reasons is not None:
        missing_b_reason = {"violate": _VIOLATION_MISSING_B, "skip": _SKIP_MISSING_B, "censor": _OPEN_WINDOW}[c.missing_b]
        if c.missing_b == "censor" and c.delta is None:  # pragma: no cover - rejected by Lag validation
            missing_b_reason = _SKIP_MISSING_B
        detail.reasons = detail.reasons.where(~b_missing, missing_b_reason)
    return v


def _evaluate_lag(log: EventLog, c: Lag, detail: _Detail | None = None) -> pd.Series:
    if c.activation in ("first", "last"):
        activation = cast(Literal["first", "last"], c.activation)
        t_a, t_b = log.first_after(c.a, c.b, activation=activation, response=c.response)
        lag = in_units(t_b - t_a, c.unit)
        if c.response == "first_overall":
            lag = lag.where(lag >= 0)  # b before the activation counts as missing
        v = sat(lag, c.delta, c.width) if c.delta is not None else lag * 0.0
        if detail is not None:
            detail.add("activation_timestamp", t_a, "timestamp", "timestamp")
            detail.add("response_timestamp", t_b.where(lag.notna()), "timestamp", "timestamp")
            detail.add("lag", lag, c.unit, "duration")
        return _lag_missing(v, lag, t_a, c, log, detail)

    # activation == "each": one violation per a-event, averaged per case
    pairs = log.activation_lags(c.a, c.b)
    t_a_case = log.first_ts(c.a)
    if pairs.empty:
        v = pd.Series(np.nan, index=log.case_ids)
        if detail is not None:
            detail.add("activations", pd.Series(0.0, index=log.case_ids), "events", "count")
        return _lag_missing(v, v, t_a_case, c, log, detail)
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
        if detail is not None:
            detail.horizon = pd.Timestamp(end)
    per_case = vi.groupby(pairs["code"].to_numpy()).mean()
    v = pd.Series(per_case.reindex(range(len(log))).to_numpy(), index=log.case_ids)
    a_missing = t_a_case.isna()
    if c.missing_a == "violate":
        v = v.where(~a_missing, 1.0)
    if detail is not None:
        codes = pairs["code"].to_numpy()
        n = len(log)
        responded = np.bincount(codes[~missing.to_numpy()], minlength=n).astype(float)
        detail.add("activations", _per_case(np.bincount(codes, minlength=n).astype(float), log), "events", "count")
        detail.add("responses_observed", _per_case(responded, log), "events", "count")
        mean_lag = lag.groupby(codes).mean().reindex(range(n)).to_numpy()
        detail.add("mean_lag", _per_case(mean_lag, log), c.unit, "duration")
        reasons = pd.Series(_OBSERVED, index=log.case_ids, dtype=object)
        reasons = reasons.where(~a_missing, _VIOLATION_MISSING_A if c.missing_a == "violate" else _SKIP_MISSING_A)
        reasons = reasons.where(~((~a_missing) & v.isna()), _SKIP_MISSING_B)
        if c.missing_b == "censor":
            open_items = _per_case(np.bincount(codes[missing.to_numpy()], minlength=n) > 0, log)
            reasons = reasons.where(~(open_items & v.notna() & ~a_missing), _OPEN_WINDOW)
        detail.reasons = reasons
    return v


def _per_case(values: np.ndarray, log: EventLog) -> pd.Series:
    return pd.Series(values, index=log.case_ids)


def _evaluate(log: EventLog, nc: NormConstraint, detail: _Detail | None = None) -> pd.Series:
    """``ν_c`` for every case; with ``detail`` also the primitives behind it.

    ``detail`` only *adds* recording to the existing computation: the arithmetic
    that produces the returned Series is the same in both cases, which is why
    capturing evidence cannot move a number.
    """
    c = nc.constraint
    if isinstance(c, Presence):
        cnt = log.count(c.activity)
        if detail is not None:
            detail.add("occurrences", cnt, "events", "count")
            detail.reasons = pd.Series(_OBSERVED, index=log.case_ids, dtype=object)
        return pd.Series(1.0 - np.minimum(cnt.to_numpy() / c.m, 1.0), index=log.case_ids)
    if isinstance(c, Exclusion):
        cnt = log.count_scoped(c.activity, after=c.after, before=c.before) if (c.after or c.before) else log.count(c.activity)
        if detail is not None:
            detail.add("occurrences", cnt, "events", "count")
            detail.reasons = pd.Series(_OBSERVED, index=log.case_ids, dtype=object).where(cnt.notna(), _SKIP_MISSING_ANCHOR)
        return (cnt > 0).astype(float).where(cnt.notna())
    if isinstance(c, Singularity):
        cnt = log.count_scoped(c.activity, after=c.after, before=c.before) if (c.after or c.before) else log.count(c.activity)
        if detail is not None:
            detail.add("occurrences", cnt, "events", "count")
            detail.reasons = pd.Series(_OBSERVED, index=log.case_ids, dtype=object).where(cnt.notna(), _SKIP_MISSING_ANCHOR)
        return sat(cnt, c.k, c.K)
    if isinstance(c, Lag):
        return _evaluate_lag(log, c, detail)
    if isinstance(c, Precedence):
        n_before = log.count_before(c.b, c.a)
        n_b = log.count(c.b)
        v = sat(n_before, c.k, c.K)
        if detail is not None:
            detail.add("b_before_activation", n_before, "events", "count")
            detail.add("b_total", n_b, "events", "count")
            detail.add("activation_timestamp", log.first_ts(c.a), "timestamp", "timestamp")
            # the reasons follow the same order as the value assignments below:
            # a vacuous satisfaction first, then the missing activation, then the
            # skipped response, which is the last thing to remove a value
            no_b = n_b <= 0
            reasons = pd.Series(_OBSERVED, index=log.case_ids, dtype=object)
            if c.missing_b == "satisfy":
                reasons = reasons.where(~no_b, _SATISFIED_VACUOUSLY)
            reasons = reasons.where(n_before.notna(), _VIOLATION_MISSING_A if c.missing_a == "violate" else _SKIP_MISSING_A)
            if c.missing_b == "skip":
                reasons = reasons.where(~no_b, _SKIP_MISSING_B)
            detail.reasons = reasons
        if c.missing_a == "violate":
            v = v.where(n_before.notna(), 1.0)
        if c.missing_b == "skip":
            v = v.where(n_b > 0)
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
        if detail is not None:
            detail.add("total_x", tot_x, c.attr_x, "amount")
            detail.add("total_y", tot_y, c.attr_y, "amount")
            detail.add("relative_mismatch", d, "ratio", "ratio")
            detail.reasons = pd.Series(_OBSERVED, index=log.case_ids, dtype=object)
        return sat(d, c.tau, c.width)
    if isinstance(c, Metric):
        x = log.attribute(c.attribute)
        if detail is not None:
            detail.add(c.attribute, x, c.attribute, "index")
            detail.reasons = pd.Series(_OBSERVED, index=log.case_ids, dtype=object).where(x.notna(), _MISSING_ATTRIBUTE)
        return sat(x, c.threshold, c.width) if c.direction == "high" else sat(c.threshold - x, 0.0, c.width)
    raise TypeError(f"unsupported constraint {type(c).__name__}")  # pragma: no cover


def evaluate_detailed(log: EventLog, norm: Norm) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, _Detail]]:
    """``V``, the scope matrix and the primitives behind every constraint.

    The same numbers as :func:`violation_matrix` — they come from the same
    code — plus, per constraint, the raw measurements and the reason each
    case was or was not evaluated. Used by
    :func:`wise.evidence.capture_evidence`; the two-item contract of
    ``violation_matrix(..., return_scope=True)`` is untouched.
    """
    cols: dict[str, pd.Series] = {}
    scope: dict[str, pd.Series] = {}
    details: dict[str, _Detail] = {}
    for nc in norm.constraints:
        detail = _Detail()
        v = pd.Series(np.asarray(_evaluate(log, nc, detail), dtype=float), index=log.case_ids, name=nc.id)
        s = nc.applies_to(log.cases, log) if nc.applicability else pd.Series(True, index=log.case_ids)
        cols[nc.id] = v.where(s, np.nan)
        scope[nc.id] = s
        details[nc.id] = detail
    return pd.DataFrame(cols, index=log.case_ids), pd.DataFrame(scope, index=log.case_ids), details


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
    manifest: RunManifest | None = field(default=None, repr=False)
    evidence: EvidencePacket | None = field(default=None, repr=False)

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
        """Events of one case (requires the log to be attached).

        This reads the log **as it is now**. It is a drill-down, not the
        evidence of this score: if the log changed since, so does the answer.
        Use :meth:`evidence_frame` or :attr:`evidence` for what was actually
        observed when the score was computed.
        """
        if self.log is None:
            raise NotScoredError("no event log attached to this result")
        return self.log.trace(case_id)

    def evidence_frame(self, view: str | None = None, *, unit_id: Any = None, constraint_id: str | None = None) -> pd.DataFrame:
        """Long-format evidence: one row per check on one case.

        Columns: the identities, ``in_scope``, ``evaluable``, ``reason_code``,
        ``violation``, one ``m__<name>`` and ``m__<name>__unit`` pair per raw
        measurement, and the witness counts. With ``view`` given, the
        view-dependent ``effective_weight``, ``penalty`` and ``scored``
        annotations are joined on.

        Raises :class:`~wise.errors.EvidenceUnavailableError` when the run did
        not capture evidence. The frame is never reconstructed from the current
        log: an approximation computed after the fact is not the evidence of
        this run.
        """
        if self.evidence is None:
            raise EvidenceUnavailableError(
                "this result carries no evidence: it was scored with evidence='none'. "
                "Re-run score(log, norm, evidence='summary') — evidence cannot be reconstructed "
                "afterwards from a log that may since have changed."
            )
        from .evidence.capture import evidence_frame

        return evidence_frame(self.evidence, view=view, unit_id=unit_id, constraint_id=constraint_id)


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


def _layers(norm: Norm) -> LayerAssignment:
    """The norm's checks and layers as the kernel's plain-data assignment.

    Both orders come from the norm — ``constraint_ids`` is the column order of
    ``V`` and of every weight vector, ``layer_ids`` the column order of the
    contributions — so the kernel reproduces the historical arithmetic exactly.
    """
    return LayerAssignment.from_mapping(norm.constraint_ids, norm.layer_of.to_dict(), norm.layer_ids)


def _effective_weights(M: np.ndarray, w: np.ndarray, norm: Norm, mode: str) -> np.ndarray:
    """Per-case effective weights; delegated to the shared kernel.

    Kept as a private function of this module with its historical signature
    because it is the seam the stage-0 regression contract pins. The
    arithmetic now lives in :mod:`wise._aggregation`, which the object-centric
    evaluator uses too, so the two paths cannot drift apart.
    """
    return effective_weights(M, w, _layers(norm), mode)


def _layer_balanced_weights(M: np.ndarray, w: np.ndarray, norm: Norm) -> np.ndarray:
    return effective_weights(M, w, _layers(norm), "layer_balanced")


def score(
    log: EventLog,
    norm: Norm,
    views: Sequence[str] | str | None = None,
    *,
    mode: str | None = None,
    derive: bool = True,
    evidence: str = "none",
    calibrations: Sequence[CalibrationRecord] | dict[str, CalibrationRecord] | None = None,
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
        is recomputed and replaces the attribute of the same name. This
        modifies the log in place; the run record says so.
    evidence
        ``"none"`` (default, the historical path and its performance),
        ``"summary"`` (one evidence row per check, with its raw measurements
        and reason code) or ``"full"`` (also materialise bounded witnesses).
        The scores are identical in all three cases.
    calibrations
        Frozen :class:`wise.evidence.CalibrationRecord` references applied to
        the matching derived attributes instead of refitting them on this log.

    The result always carries a :attr:`ScoreResult.manifest` describing the
    run — the mode actually used, the views, the input identity, the
    preparation options and the observation policy. It is a *score* record:
    the grouping and comparator of a backlog are not known here, so
    :meth:`wise.evidence.RunManifest.finalize` produces that record separately.
    """
    norm.validate()
    views = [views] if isinstance(views, str) else list(dict.fromkeys(views or norm.view_names))
    for v in views:
        norm.get_view(v)
    requested_mode = mode
    mode = mode or norm.scoring_mode
    if mode not in SCORING_MODES:
        raise NormError(f"mode must be one of {SCORING_MODES}, got {mode!r}")
    if evidence not in EVIDENCE_MODES:
        raise NormError(f"evidence must be one of {EVIDENCE_MODES}, got {evidence!r}")

    from .evidence.capture import build_score_manifest, capture_evidence, resolve_calibrations

    frozen = resolve_calibrations(norm, calibrations, derive=derive)
    derived_names: list[str] = []
    if derive and norm.derived_attributes:
        derived_names = log.derive(norm.derived_attributes, overwrite=False, calibrations=frozen)

    details: dict[str, _Detail] | None = None
    if evidence == "none":
        V, S = violation_matrix(log, norm, return_scope=True)
    else:
        V, S, details = evaluate_detailed(log, norm)
    layers = LayerAssignment(tuple(V.columns), tuple(norm.layer_ids), tuple(norm.layer_of[cid] for cid in V.columns))
    Vm = V.to_numpy(dtype=float)
    row_ids = tuple(V.index)

    score_cols: list[np.ndarray] = []
    contributions: dict[str, pd.DataFrame] = {}
    weights: dict[str, np.ndarray] = {}
    for view in views:
        w = norm.weight_vector(view).to_numpy(dtype=float)
        weights[view] = w
        out = aggregate(Vm, w, layers, mode, row_ids=row_ids)
        score_cols.append(out.scores)
        contributions[view] = pd.DataFrame(out.contributions, index=V.index, columns=norm.layer_ids)

    scores = pd.DataFrame(np.column_stack(score_cols) if score_cols else np.empty((len(V), 0)), index=V.index, columns=views)
    result = ScoreResult(
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
    # one snapshot for the run record and the evidence, so that the packet's
    # witnesses provably belong to the run the manifest describes. A capturing
    # run hashes the event table too: a witness is evidence of *this* score, and
    # a relabelled activity or a changed amount makes it evidence of another one.
    snapshot = log.snapshot(scope="structure" if details is None else "events")
    result.manifest = build_score_manifest(
        log,
        norm,
        views=views,
        mode=mode,
        mode_source="call_override" if requested_mode is not None else "norm_default",
        details=details,
        calibrations=frozen,
        derived_names=derived_names,
        snapshot=snapshot,
    )
    if details is not None:
        result.evidence = capture_evidence(result, details=details, mode=evidence, snapshot=snapshot)
    return result
