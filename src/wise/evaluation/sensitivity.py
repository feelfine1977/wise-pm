"""Rank sensitivity: what moves a backlog, and what an interval does not cover.

A rank is a claim about *order*, and the ordinary way of qualifying it — a
percentile interval around each slice's score — answers a different question.
Two slices can have wide, overlapping score intervals and never change places,
and two slices with narrow intervals can swap on every replicate. This module
reports the rank.

**Three sources, reported separately, because they are separate claims.**

:func:`sampling_sensitivity`
    the units could have been a different sample. Resampling, with the
    dependence *declared*: cases nested in a purchasing document, invoices
    billed by one vendor and obligations of one order are not independent
    draws, and a resampling that pretends they are reports an interval that is
    too narrow by a factor nobody states. ``dependence=`` has no default.
:func:`parameter_sensitivity`
    the method has settings somebody chose — the shrinkage constant, the
    minimum support, the comparator, the volume. Re-rank under a grid of them.
:func:`construction_sensitivity`
    the assessment itself was constructed: a scoring mode, a view, a unit type,
    a representation. Re-rank across runs that differ in that choice.

:func:`combined_stability` puts the three beside each other and takes the worst
verdict per slice — and says so when a source was never varied, because a
stability badge that has only seen one source is a badge about one source.

**What this is not.** It is not the application's bootstrap. WISE Workbench's
``wise_analytics.uncertainty.bootstrap_backlog`` already resamples cases (with
an optional ``cluster=``), reports percentile intervals for the gap, the
stabilised gap, ``PI`` and ``stable_PI``, rank intervals, ``P(top-k)`` and a
stability badge, vectorised over replicates. Where the sampling source is
concerned it answers more than this module does and answers it faster, and
nothing here reimplements its badge or its interval table. What this module
adds is the part a bootstrap cannot reach:

* the dependence is a **required, reported object** rather than an optional
  keyword with a warning, and an interval taken over exchangeable units says so
  in a qualification instead of in a docstring;
* the same rank machinery is applied to **parameter** and **construction**
  variation, which resampling the data cannot express at all;
* one **combined** verdict that refuses to call a rank stable when only one of
  the three was varied.

Every variant is ranked by :func:`wise.prioritize` itself. The resampled
ranking is therefore *the* ranking rather than a re-derivation of it, at the
cost of one call per variant; for a large log and many replicates, the
vectorised application instrument is the right tool and this one is the
statement of what the numbers mean.

Nothing here is imported by ``import wise``.

>>> import wise
>>> from wise.evaluation.sensitivity import sampling_sensitivity
>>> result = wise.score(wise.running_p2p_log(), wise.running_p2p_norm())
>>> report = sampling_sensitivity(result, "company", view="Finance", dependence="vendor", B=50, seed=7)
>>> report.dependence.unit
'vendor'
>>> report.source
'sampling'
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..errors import NormError
from ..evidence.models import Qualification, QualificationCode
from ..prioritization import _is_scored_result, _keys, prioritize

if TYPE_CHECKING:  # pragma: no cover
    from ..explain.baseline import BaselineSpec
    from ..oc.evaluation import OCScoreResult
    from ..scoring import ScoreResult

#: Version of the sensitivity contract exported by :meth:`SensitivityReport.to_dict`.
SENSITIVITY_SCHEMA_VERSION = "wise-sensitivity/1"

#: The value of ``dependence=`` that declares the assessment units to be
#: independent draws. It is spelled out rather than defaulted, because it is an
#: assumption about the process and almost never a true one.
INDEPENDENT_UNITS = "unit"

#: The application-side instrument this module deliberately does not duplicate.
APPLICATION_BOOTSTRAP = "wise_analytics.uncertainty.bootstrap_backlog"


class UncertaintySource(str, Enum):
    """Which of the three questions a report answers."""

    #: the units could have been a different sample of the same process
    SAMPLING = "sampling"
    #: the method's settings could have been chosen differently
    PARAMETER = "parameter"
    #: the assessment itself could have been constructed differently
    CONSTRUCTION = "construction"


class RankVerdict(str, Enum):
    """What a slice's rank did across the variants. Four different claims."""

    #: the slice keeps its position, and its band is a small part of the backlog
    STABLE = "rank_stable"
    #: it moves, but inside a band well short of half the backlog
    MOVES_WITHIN_BAND = "rank_moves_within_band"
    #: it moves across half the backlog or more: its position is a reading of
    #: one variant, not a property of the slice
    MOVES_ACROSS_THE_BACKLOG = "rank_moves_across_the_backlog"
    #: too few units, or the slice was absent from too many variants, for any
    #: of the above to mean anything
    INSUFFICIENT_SUPPORT = "insufficient_support"


@dataclass(frozen=True)
class Dependence:
    """What one draw is, stated rather than assumed.

    ``unit`` is either :data:`INDEPENDENT_UNITS` — the assessment units are
    resampled one at a time — or the name of the attribute whose blocks are
    resampled whole. ``n_draws`` is how many things are actually drawn, and it
    is the number the width of every interval below really rests on: a hundred
    invoices from four vendors, resampled by vendor, is four draws.

    >>> Dependence(INDEPENDENT_UNITS, None, n_draws=100, n_units=100).exchangeable
    True
    >>> Dependence("vendor", "vendor", n_draws=4, n_units=100).statement()
    'one draw is one vendor: 4 draws over 100 scored units (25.0 units per draw)'
    """

    unit: str
    column: str | None
    n_draws: int
    n_units: int

    @property
    def exchangeable(self) -> bool:
        """Whether this resampling treats the units themselves as the draws."""
        return self.column is None

    def statement(self) -> str:
        """The sentence a report has to carry."""
        if self.exchangeable:
            return (
                f"one draw is one assessment unit: {self.n_draws} draws over {self.n_units} scored units, "
                "which assumes they are independent"
            )
        per = self.n_units / self.n_draws if self.n_draws else float("nan")
        return f"one draw is one {self.unit}: {self.n_draws} draws over {self.n_units} scored units ({per:.1f} units per draw)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "column": self.column,
            "n_draws": int(self.n_draws),
            "n_units": int(self.n_units),
            "exchangeable": self.exchangeable,
            "statement": self.statement(),
        }


@dataclass(frozen=True)
class SliceStability:
    """What one slice's rank did, and how much of the backlog it covered."""

    key: tuple[Any, ...]
    label: str
    n_units: int
    point_rank: int
    point_metric: float
    rank_low: int
    rank_high: int
    rank_span_share: float
    present_share: float
    p_top_k: float
    metric_low: float
    metric_high: float
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": list(self.key),
            "label": self.label,
            "n_units": int(self.n_units),
            "point_rank": int(self.point_rank),
            "point_metric": float(self.point_metric),
            "rank_low": int(self.rank_low),
            "rank_high": int(self.rank_high),
            "rank_span_share": float(self.rank_span_share),
            "present_share": float(self.present_share),
            "p_top_k": float(self.p_top_k),
            "metric_low": float(self.metric_low),
            "metric_high": float(self.metric_high),
            "verdict": self.verdict,
        }


@dataclass(frozen=True)
class SensitivityReport:
    """One source of uncertainty, reported over the ranks it moves.

    ``covers`` and ``does_not_cover`` are part of the result, not of the prose
    around it: an interval whose scope is not written down is read as covering
    everything.
    """

    source: str
    view: str | None
    grouping: tuple[str, ...]
    metric: str
    k: int
    ci: float
    n_variants: int
    n_slices: int
    slices: tuple[SliceStability, ...] = ()
    dependence: Dependence | None = None
    settings: tuple[dict[str, Any], ...] = ()
    covers: tuple[str, ...] = ()
    does_not_cover: tuple[str, ...] = ()
    qualifications: tuple[Qualification, ...] = ()
    schema_version: str = SENSITIVITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "slices", tuple(self.slices))
        object.__setattr__(self, "settings", tuple(dict(s) for s in self.settings))
        object.__setattr__(self, "covers", tuple(str(c) for c in self.covers))
        object.__setattr__(self, "does_not_cover", tuple(str(c) for c in self.does_not_cover))
        object.__setattr__(self, "qualifications", tuple(self.qualifications))

    def __repr__(self) -> str:
        counts = self.verdict_counts()
        return (
            f"SensitivityReport({self.source!r}, {self.n_slices} slices x {self.n_variants} variants, "
            f"stable={counts.get(RankVerdict.STABLE.value, 0)}, "
            f"moves_across={counts.get(RankVerdict.MOVES_ACROSS_THE_BACKLOG.value, 0)})"
        )

    def of(self, key: Any) -> SliceStability:
        """One slice's stability by key (a scalar for a single grouping column)."""
        wanted = key if isinstance(key, tuple) else (key,)
        for row in self.slices:
            if row.key == wanted:
                return row
        raise NormError(f"no slice {wanted!r} in this report; it holds {[s.key for s in self.slices]}")

    def verdict_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in self.slices:
            out[row.verdict] = out.get(row.verdict, 0) + 1
        return out

    def table(self) -> pd.DataFrame:
        """One row per slice, indexed by the grouping key(s)."""
        if not self.slices:
            return pd.DataFrame(columns=["n_units", "point_rank", "rank_low", "rank_high", "verdict"])
        frame = pd.DataFrame([row.to_dict() for row in self.slices])
        keys = frame.pop("key")
        index = pd.MultiIndex.from_tuples([tuple(k) for k in keys], names=list(self.grouping))
        if len(self.grouping) == 1:
            index = pd.Index([k[0] for k in keys], name=self.grouping[0])
        frame.index = index
        return frame

    def statement(self) -> str:
        """One paragraph: what was varied, what moved, and what is not covered."""
        counts = self.verdict_counts()
        head = (
            f"{self.source} sensitivity of the backlog by {', '.join(self.grouping)}"
            + (f" under view {self.view}" if self.view else "")
            + f": {self.n_slices} slices re-ranked over {self.n_variants} variants."
        )
        if self.dependence is not None:
            head += " Resampling: " + self.dependence.statement() + "."
        badges = ", ".join(f"{n} {name}" for name, n in sorted(counts.items()))
        return (
            f"{head} Verdicts: {badges or 'none'}. "
            f"Covers: {'; '.join(self.covers)}. Does not cover: {'; '.join(self.does_not_cover)}."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "view": self.view,
            "grouping": list(self.grouping),
            "metric": self.metric,
            "k": int(self.k),
            "ci": float(self.ci),
            "n_variants": int(self.n_variants),
            "n_slices": int(self.n_slices),
            "dependence": None if self.dependence is None else self.dependence.to_dict(),
            "settings": [dict(s) for s in self.settings],
            "slices": [row.to_dict() for row in self.slices],
            "covers": list(self.covers),
            "does_not_cover": list(self.does_not_cover),
            "qualifications": [q.to_dict() for q in self.qualifications],
            "statement": self.statement(),
        }


# --------------------------------------------------------------------- internals
def _scored_frame(
    result: ScoreResult | OCScoreResult, view: str | None, by: Sequence[str], dependence_column: str | None
) -> tuple[pd.DataFrame, str]:
    """The wide per-unit frame, restricted to scored units, with the keys present."""
    if not _is_scored_result(result):
        raise NormError(
            "rank sensitivity needs a scored result (a ScoreResult or an OCScoreResult): "
            "a bare frame carries no view, no norm and no identity to resample under"
        )
    views = list(result.views)
    if view is None:
        if len(views) != 1:
            raise NormError(f"specify view=...; available: {views}")
        view = views[0]
    if view not in views:
        raise NormError(f"unknown view {view!r}; available: {views}")
    frame = result.frame(view)
    wanted = [*by] + ([dependence_column] if dependence_column else [])
    if any(c not in frame.columns for c in wanted):
        frame = frame.reset_index()
    missing = [c for c in wanted if c not in frame.columns]
    if missing:
        raise NormError(f"column(s) not on the unit table: {missing}")
    return frame[frame["score"].notna()].copy(), view


def _key_tuples(frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    if isinstance(frame.index, pd.MultiIndex):
        return [tuple(k) for k in frame.index]
    return [(k,) for k in frame.index]


def _label(key: Sequence[Any]) -> str:
    return " | ".join("null" if k is None or (isinstance(k, float) and math.isnan(k)) else str(k) for k in key)


def _ranks_of(backlog: pd.DataFrame, metric: str) -> dict[tuple[Any, ...], tuple[int, float]]:
    """1-based position in a ranked backlog, with the metric that put it there."""
    out: dict[tuple[Any, ...], tuple[int, float]] = {}
    values = backlog[metric].to_numpy(dtype=float)
    for position, key in enumerate(_key_tuples(backlog), start=1):
        out[key] = (position, float(values[position - 1]))
    return out


def _rank_backlog(frame: pd.DataFrame, by: Sequence[str], **kwargs: Any) -> pd.DataFrame:
    """One ranking, by :func:`wise.prioritize` itself — never a re-derivation."""
    return prioritize(frame, list(by), score_col="score", **kwargs)


def _summarise(
    *,
    source: UncertaintySource,
    point: pd.DataFrame,
    variants: Sequence[Mapping[tuple[Any, ...], tuple[int, float]]],
    view: str | None,
    by: Sequence[str],
    metric: str,
    k: int,
    ci: float,
    min_support: int,
    dependence: Dependence | None,
    settings: Sequence[Mapping[str, Any]],
    covers: Sequence[str],
    does_not_cover: Sequence[str],
    extra_qualifications: Sequence[Qualification] = (),
) -> SensitivityReport:
    """Turn one point estimate and a bag of re-rankings into the report."""
    point_ranks = _ranks_of(point, metric)
    n_slices = len(point_ranks)
    supports = point["n_cases"].to_numpy(dtype=float)
    lo_q, hi_q = 100.0 * (1.0 - ci) / 2.0, 100.0 * (1.0 - (1.0 - ci) / 2.0)
    band = max(1, math.ceil(0.1 * n_slices))
    rows: list[SliceStability] = []
    for position, key in enumerate(_key_tuples(point)):
        seen = [variant[key] for variant in variants if key in variant]
        ranks = np.array([r for r, _ in seen], dtype=float)
        values = np.array([v for _, v in seen], dtype=float)
        present = len(seen) / len(variants) if variants else 0.0
        point_rank, point_metric = point_ranks[key]
        if len(seen):
            rank_low, rank_high = (round(float(x)) for x in np.percentile(ranks, [lo_q, hi_q]))
            metric_low, metric_high = (float(x) for x in np.percentile(values, [lo_q, hi_q]))
            inside = float(np.mean((ranks <= k) & (values > 0)))
        else:  # pragma: no cover - a slice of the point estimate absent everywhere
            rank_low = rank_high = point_rank
            metric_low = metric_high = point_metric
            inside = 0.0
        span = rank_high - rank_low
        span_share = span / max(n_slices - 1, 1)
        stays = inside if point_rank <= k else 1.0 - inside
        # the rank span decides. p_top_k can only *demote* a slice that has a
        # priority to lose: a slice whose stabilised index is zero is not
        # "unstable" for staying out of the top k, it simply has nothing there
        if supports[position] < min_support or present < 0.5:
            verdict = RankVerdict.INSUFFICIENT_SUPPORT
        elif span_share >= 0.5:
            verdict = RankVerdict.MOVES_ACROSS_THE_BACKLOG
        elif span > band or (point_metric > 0 and stays < 0.8):
            verdict = RankVerdict.MOVES_WITHIN_BAND
        else:
            verdict = RankVerdict.STABLE
        rows.append(
            SliceStability(
                key=key,
                label=_label(key),
                n_units=int(supports[position]),
                point_rank=point_rank,
                point_metric=point_metric,
                rank_low=rank_low,
                rank_high=rank_high,
                rank_span_share=float(span_share),
                present_share=float(present),
                p_top_k=float(inside),
                metric_low=metric_low,
                metric_high=metric_high,
                verdict=verdict.value,
            )
        )
    qualifications = list(extra_qualifications)
    unstable = [row for row in rows if row.verdict == RankVerdict.MOVES_ACROSS_THE_BACKLOG.value]
    if unstable:
        qualifications.append(
            Qualification(
                QualificationCode.RANK_MOVES_UNDER_RESAMPLING,
                f"{len(unstable)} of {n_slices} slices change rank across at least half of this backlog under "
                f"{source.value} variation ("
                + ", ".join(f"{row.label}: {row.rank_low}-{row.rank_high}" for row in unstable[:5])
                + "); for those slices the reported position is a reading of one variant, not a property of the slice",
                scope="run",
            )
        )
    return SensitivityReport(
        source=source.value,
        view=view,
        grouping=tuple(by),
        metric=metric,
        k=int(k),
        ci=float(ci),
        n_variants=len(variants),
        n_slices=n_slices,
        slices=tuple(rows),
        dependence=dependence,
        settings=tuple(dict(s) for s in settings),
        covers=tuple(covers),
        does_not_cover=tuple(does_not_cover),
        qualifications=tuple(qualifications),
    )


_NEVER_COVERED = (
    "measurement error in the log itself, and anything a filter removed before scoring",
    "the norm's thresholds, weights and layer partition, unless they were varied as parameters",
    "whether the ranked slice is the one worth acting on: an interval is not a decision, "
    "and none of this is a probability that a slice is 'really' first",
)


# ----------------------------------------------------------------- sampling (M01)
def sampling_sensitivity(
    result: ScoreResult | OCScoreResult,
    by: str | Sequence[str],
    view: str | None = None,
    *,
    dependence: str,
    gamma: float = 0.0,
    volume: str = "cases",
    baseline: float | None = None,
    baseline_spec: BaselineSpec | None = None,
    min_cases: int = 1,
    B: int = 200,
    seed: int = 0,
    k: int = 5,
    ci: float = 0.90,
    min_support: int = 1,
) -> SensitivityReport:
    """Re-rank the backlog over ``B`` resamples of the declared dependence unit.

    ``dependence`` has no default and that is the point of the function. Pass
    the name of the attribute whose blocks move together — the purchasing
    document, the vendor, the order, whatever nests the units — and whole
    blocks are drawn with replacement. Pass :data:`INDEPENDENT_UNITS` to draw
    the units themselves, which is a claim about the process and is recorded as
    an ``exchangeable_units_assumed`` qualification, not waved through.

    Whether the comparator is recomputed inside each replicate is decided by
    the comparator, not by a keyword:
    :attr:`~wise.explain.BaselineSpec.moves_with_the_population` is true for a
    current-population comparator, which is a statistic of the units being
    resampled, and false for a historical or target one, which was fixed
    earlier and must not absorb this uncertainty. A scalar ``baseline=`` is a
    fixed target and is held.

    Every replicate is ranked by :func:`wise.prioritize`, so the resampled
    ranking is the ranking. For a large log, ``B`` calls is the wrong shape and
    the application's vectorised bootstrap (see the module docstring) is the
    right instrument; this one exists to state what the interval means.
    """
    by = _keys(by)
    if B < 1:
        raise NormError(f"B must be at least 1, got {B!r}")
    if not 0.0 < ci < 1.0:
        raise NormError(f"ci must be strictly between 0 and 1, got {ci!r}")
    column = None if str(dependence) == INDEPENDENT_UNITS else str(dependence)
    frame, view = _scored_frame(result, view, by, column)
    if frame.empty:
        raise NormError("no scored units to resample")

    if column is None:
        blocks = [np.array([i]) for i in range(len(frame))]
        draw_name = INDEPENDENT_UNITS
    else:
        codes = pd.factorize(frame[column], use_na_sentinel=False)[0]
        blocks = [np.flatnonzero(codes == c) for c in range(int(codes.max()) + 1)]
        draw_name = column
    declared = Dependence(unit=draw_name, column=column, n_draws=len(blocks), n_units=len(frame))

    fixed_baseline = baseline
    if baseline_spec is not None:
        if baseline is not None:
            raise NormError("baseline= and baseline_spec= are two comparators for one backlog; pass one")
        if baseline_spec.moves_with_the_population:
            fixed_baseline = None  # recomputed inside every replicate: it is a statistic of these units
        elif baseline_spec.reference_score is None:
            raise NormError(
                f"comparator {baseline_spec.baseline_id!r} is fixed by an earlier decision but states no "
                "reference score, so there is nothing to hold the replicates against"
            )
        else:
            fixed_baseline = float(baseline_spec.reference_score)
    ranking = {"view": None, "gamma": gamma, "volume": volume, "min_cases": min_cases, "baseline": fixed_baseline}
    point = _rank_backlog(frame, by, **ranking)

    rng = np.random.default_rng(int(seed))
    variants: list[dict[tuple[Any, ...], tuple[int, float]]] = []
    for _ in range(int(B)):
        picked = rng.integers(0, len(blocks), size=len(blocks))
        rows = np.concatenate([blocks[i] for i in picked]) if len(blocks) else np.array([], dtype=int)
        replicate = frame.iloc[rows]
        try:
            variants.append(_ranks_of(_rank_backlog(replicate, by, **ranking), "stable_PI"))
        except NormError:  # pragma: no cover - a replicate with no scored unit
            variants.append({})

    extra: list[Qualification] = []
    if declared.exchangeable:
        extra.append(
            Qualification(
                QualificationCode.EXCHANGEABLE_UNITS_ASSUMED,
                "this interval resampled the assessment units one at a time, so it assumes they are independent "
                "draws. Units that share a document, an order or a vendor are not, and the interval is then too "
                f"narrow; pass dependence=<attribute> to resample blocks instead. {declared.statement()}",
                scope="run",
            )
        )
    return _summarise(
        source=UncertaintySource.SAMPLING,
        point=point,
        variants=variants,
        view=view,
        by=by,
        metric="stable_PI",
        k=k,
        ci=ci,
        min_support=min_support,
        dependence=declared,
        settings=({"B": int(B), "seed": int(seed), **ranking},),
        covers=(
            f"the variability of the ranking when the {declared.n_draws} {draw_name} draw(s) behind it are "
            "resampled with replacement",
            "the comparator, recomputed per replicate when it is a current-population comparator and held when it is not",
        ),
        does_not_cover=(
            "the dependence itself: blocks are drawn whole, so a wrong choice of "
            f"{draw_name!r} is not detected here, only obeyed",
            "the replicates a slice was not drawn into at all. Every rank interval is conditional on the slice "
            "being present, and 'present_share' is the other half of the reading — resampling blocks makes a whole "
            "block absent far more often than resampling units does",
            "any setting of the method — gamma, minimum support, volume and the comparator kind are held fixed "
            "(see parameter_sensitivity)",
            "any alternative construction of the assessment — mode, view, unit type (see construction_sensitivity)",
            *_NEVER_COVERED,
        ),
        extra_qualifications=extra,
    )


# ------------------------------------------------------------------- parameters
def parameter_sensitivity(
    result: ScoreResult | OCScoreResult,
    by: str | Sequence[str],
    view: str | None = None,
    *,
    settings: Sequence[Mapping[str, Any]],
    k: int = 5,
    ci: float = 0.90,
    min_support: int = 1,
    reference: Mapping[str, Any] | None = None,
) -> SensitivityReport:
    """Re-rank the same units under a grid of method settings.

    Each entry of ``settings`` is a mapping of :func:`wise.prioritize`
    arguments — ``gamma``, ``volume``, ``min_cases``, ``baseline`` — and each
    is one variant. ``reference`` is the setting the point estimate uses, and
    defaults to the first entry, so the report contrasts a chosen configuration
    with the alternatives rather than with a hidden default.

    This is the source no resampling can reach: the data do not move at all,
    and the ranking still does.
    """
    by = _keys(by)
    grid = [dict(s) for s in settings]
    if not grid:
        raise NormError("parameter_sensitivity needs at least one setting to vary over")
    unknown = sorted({key for s in grid for key in s} - {"gamma", "volume", "min_cases", "baseline", "z"})
    if unknown:
        raise NormError(f"unknown prioritize setting(s) {unknown}; this varies gamma, volume, min_cases and baseline")
    frame, view = _scored_frame(result, view, by, None)
    base = dict(reference if reference is not None else grid[0])
    point = _rank_backlog(frame, by, view=None, **base)
    variants = [_ranks_of(_rank_backlog(frame, by, view=None, **s), "stable_PI") for s in grid]
    return _summarise(
        source=UncertaintySource.PARAMETER,
        point=point,
        variants=variants,
        view=view,
        by=by,
        metric="stable_PI",
        k=k,
        ci=ci,
        min_support=min_support,
        dependence=None,
        settings=tuple(grid),
        covers=(
            f"the {len(grid)} declared setting(s) of gamma, volume, minimum support and the scalar comparator",
            "nothing else: this is a set of configurations somebody chose, not a distribution over them",
        ),
        does_not_cover=(
            "any setting outside the grid, and the grid is not a prior",
            "sampling variability: the same units are ranked every time (see sampling_sensitivity)",
            "any alternative construction of the assessment (see construction_sensitivity)",
            *_NEVER_COVERED,
        ),
    )


# ----------------------------------------------------------------- construction
def construction_sensitivity(
    constructions: Mapping[str, ScoreResult | OCScoreResult],
    by: str | Sequence[str],
    view: str | None = None,
    *,
    reference: str | None = None,
    gamma: float = 0.0,
    volume: str = "cases",
    baseline: float | None = None,
    min_cases: int = 1,
    k: int = 5,
    ci: float = 0.90,
    min_support: int = 1,
) -> SensitivityReport:
    """Re-rank across runs that constructed the assessment differently.

    ``constructions`` maps a name to a scored result: the same units under
    another scoring mode, another view, another unit type, a native evaluation
    against a projection. The point estimate is ``reference`` (default: the
    first entry).

    The slices must be comparable — that is the caller's declaration, and the
    report says so — but the *numbers* need not be: what is compared here is
    the order, which is exactly the claim a backlog makes.
    """
    by = _keys(by)
    names = list(constructions)
    if not names:
        raise NormError("construction_sensitivity needs at least one construction")
    chosen = reference if reference is not None else names[0]
    if chosen not in constructions:
        raise NormError(f"unknown reference construction {chosen!r}; this report holds {names}")
    ranking = {"view": None, "gamma": gamma, "volume": volume, "min_cases": min_cases, "baseline": baseline}
    frames = {}
    for name, run in constructions.items():
        wanted = view if view is None or view in list(run.views) else None
        frames[name], _ = _scored_frame(run, wanted, by, None)
    point = _rank_backlog(frames[chosen], by, **ranking)
    variants = [_ranks_of(_rank_backlog(frames[name], by, **ranking), "stable_PI") for name in names]
    return _summarise(
        source=UncertaintySource.CONSTRUCTION,
        point=point,
        variants=variants,
        view=view,
        by=by,
        metric="stable_PI",
        k=k,
        ci=ci,
        min_support=min_support,
        dependence=None,
        settings=tuple({"construction": name, "reference": name == chosen} for name in names),
        covers=(
            f"the {len(names)} construction(s) the caller supplied: {', '.join(names)}",
            "the order they produce, which is what a backlog claims — not their scores, which need not be comparable",
        ),
        does_not_cover=(
            "any construction nobody built: an assessment that was never run cannot move a rank here",
            "whether the slices mean the same thing in every construction, which is the caller's declaration",
            "sampling variability and the method's settings (see the other two functions)",
            *_NEVER_COVERED,
        ),
    )


# --------------------------------------------------------------------- combined
@dataclass(frozen=True)
class CombinedStability:
    """The three sources beside each other, and the weakest claim per slice."""

    grouping: tuple[str, ...]
    sources: tuple[str, ...]
    rows: pd.DataFrame = field(repr=False)
    qualifications: tuple[Qualification, ...] = ()
    schema_version: str = SENSITIVITY_SCHEMA_VERSION

    @property
    def complete(self) -> bool:
        """Whether all three sources were varied."""
        return set(self.sources) == {s.value for s in UncertaintySource}

    def table(self) -> pd.DataFrame:
        return self.rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "grouping": list(self.grouping),
            "sources": list(self.sources),
            "complete": self.complete,
            "verdicts": {str(key): str(value) for key, value in self.rows["verdict"].items()},
            "qualifications": [q.to_dict() for q in self.qualifications],
        }


#: Weakest first: a slice is only as stable as the weakest source says it is.
_VERDICT_ORDER = (
    RankVerdict.INSUFFICIENT_SUPPORT.value,
    RankVerdict.MOVES_ACROSS_THE_BACKLOG.value,
    RankVerdict.MOVES_WITHIN_BAND.value,
    RankVerdict.STABLE.value,
)


def combined_stability(reports: Sequence[SensitivityReport]) -> CombinedStability:
    """The weakest verdict per slice across the reports, and what was not varied.

    A slice is called stable only when every source that was varied calls it
    stable. When a source was not varied at all, the result says so with an
    ``uncertainty_source_not_varied`` qualification, because a badge that has
    only seen resampling is a badge about resampling.
    """
    if not reports:
        raise NormError("combined_stability needs at least one report")
    grouping = reports[0].grouping
    mismatched = [r.source for r in reports if r.grouping != grouping]
    if mismatched:
        raise NormError(f"these reports rank different groupings; {grouping} versus the ones from {mismatched}")
    order = {name: i for i, name in enumerate(_VERDICT_ORDER)}
    per_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for report in reports:
        for row in report.slices:
            entry = per_key.setdefault(row.key, {"label": row.label, "n_units": row.n_units})
            entry[f"verdict__{report.source}"] = row.verdict
            entry[f"rank_low__{report.source}"] = row.rank_low
            entry[f"rank_high__{report.source}"] = row.rank_high
    for entry in per_key.values():
        seen = [v for key, v in entry.items() if key.startswith("verdict__")]
        entry["verdict"] = min(seen, key=lambda v: order[v]) if seen else RankVerdict.INSUFFICIENT_SUPPORT.value
    frame = pd.DataFrame(list(per_key.values()))
    keys = list(per_key)
    frame.index = (
        pd.Index([k[0] for k in keys], name=grouping[0])
        if len(grouping) == 1
        else pd.MultiIndex.from_tuples(keys, names=list(grouping))
    )
    sources = tuple(dict.fromkeys(r.source for r in reports))
    missing = [s.value for s in UncertaintySource if s.value not in sources]
    qualifications: list[Qualification] = []
    if missing:
        qualifications.append(
            Qualification(
                QualificationCode.UNCERTAINTY_SOURCE_NOT_VARIED,
                f"this stability verdict varied {list(sources)} and never varied {missing}; "
                "nothing here bounds the source(s) that were not varied, and a slice called stable is stable "
                "against what was tried",
                scope="run",
            )
        )
    for report in reports:
        qualifications.extend(report.qualifications)
    return CombinedStability(
        grouping=grouping,
        sources=sources,
        rows=frame,
        qualifications=tuple(qualifications),
    )


__all__ = [
    "APPLICATION_BOOTSTRAP",
    "INDEPENDENT_UNITS",
    "SENSITIVITY_SCHEMA_VERSION",
    "CombinedStability",
    "Dependence",
    "RankVerdict",
    "SensitivityReport",
    "SliceStability",
    "UncertaintySource",
    "combined_stability",
    "construction_sensitivity",
    "parameter_sensitivity",
    "sampling_sensitivity",
]
