"""The shared comparator specification.

One backlog must rest on **one** comparison. Historically the scalar
comparator of :func:`wise.prioritize` and the current-population layer profile
of :func:`wise.layer_drivers` could differ, so a ranking against last year's
target could be explained with this year's layer means. :class:`BaselineSpec`
is the single object both functions resolve, and it carries everything needed
to say whether an additive layer explanation is meaningful at all:

* **who** the comparator is — :attr:`BaselineSpec.baseline_id` and
  :class:`BaselineKind` (current population, a frozen historical population,
  or an explicit target);
* **what** it asserts — :attr:`BaselineSpec.reference_score` and, optionally,
  a complete :attr:`BaselineSpec.reference_layer_penalties` profile;
* **under which semantics** — the view, the normalisation (the scoring mode),
  the score convention, the assessment-unit type, the layer meanings, the norm
  identity, the calibrations in force and the aggregation convention.

Compatibility is about assessment semantics, never about population identity.
A historical comparator is expected to come from a *different* population;
that is the point of it. What may not differ silently is what the numbers
mean: a different view, a different normalisation, a different layer partition
or a changed norm makes an additive contrast invalid, and this module refuses
it instead of aligning an unknown layer with zero.

A scalar-only target — "we want 0.95" — is a perfectly good priority
comparator and has no uniquely determined layer profile. It is represented by
a spec with :attr:`BaselineSpec.reference_layer_penalties` ``None``, and every
consumer reports ``reference_profile_unavailable`` rather than inventing one.

>>> spec = BaselineSpec.target(0.9, baseline_id="board-target-2026")
>>> spec.scalar_only
True
>>> spec.kind.value
'target'
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

from ..errors import NormError

if TYPE_CHECKING:  # pragma: no cover
    from ..norm import Norm
    from ..scoring import ScoreResult

#: Version of the comparator contract exported by :meth:`BaselineSpec.to_dict`.
BASELINE_SCHEMA_VERSION = "wise-baseline/1"

#: The score convention of this library: ``S = 1 − Σ_λ Δ_λ`` with effective
#: weights renormalised over the applicable constraints, so a score and its
#: layer penalties live in ``[0, 1]`` and add up exactly.
SCORE_CONVENTION = "score_is_one_minus_sum_of_effective_layer_penalties"

#: How a reference mean is formed: the unweighted mean over the *scored* units
#: of the population. Exposure scales a priority index; it never turns the mean
#: into an exposure-weighted mean.
AGGREGATION = "unweighted_mean_over_scored_units"

#: Numerical tolerance for ``reference_score == 1 − Σ reference_layer_penalties``
#: and for the "penalties sum to at most one" bound. Both are exact identities
#: in real arithmetic; the tolerance absorbs floating-point summation order
#: only. It is deliberately far below any difference a user could act on.
DEFAULT_TOLERANCE = 1e-9


class BaselineError(NormError):
    """The comparator specification is malformed, or incompatible with this run.

    Derives from :class:`~wise.errors.NormError` (and therefore from
    :class:`~wise.errors.WiseError`), so existing handlers keep working.
    """


class BaselineKind(str, Enum):
    """Which comparison a specification describes.

    The three values are the ones the roadmap's interchange contract uses.
    ``CURRENT_POPULATION`` answers *where does today's assessed
    underperformance concentrate*; ``HISTORICAL`` and ``TARGET`` answer *how
    does this group differ from a fixed comparator*. They are different
    questions and must not be mixed in one statement.
    """

    CURRENT_POPULATION = "current_population"
    HISTORICAL = "historical"
    TARGET = "target"


def _finite(value: Any, what: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise BaselineError(f"{what} must be a number, got {value!r}") from exc
    if not math.isfinite(out):
        raise BaselineError(f"{what} must be finite, got {value!r}")
    return out


def _text(value: Any, what: str) -> str:
    out = str(value)
    if not out:
        raise BaselineError(f"{what} must be a non-empty string")
    return out


@dataclass(frozen=True)
class AssessmentContext:
    """What one concrete assessment means, for comparison against a spec.

    Every field is optional in the sense that a specification which leaves the
    corresponding field ``None`` simply does not constrain it — an undeclared
    field is never silently assumed to match. What the specification *does*
    declare is compared exactly.
    """

    view: str | None = None
    scoring_mode: str | None = None
    unit_type: str | None = None
    layer_ids: tuple[str, ...] = ()
    norm_fingerprint: str | None = None
    calibrations: tuple[str, ...] = ()
    score_convention: str = SCORE_CONVENTION
    aggregation: str = AGGREGATION

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_ids", tuple(str(layer) for layer in self.layer_ids))
        object.__setattr__(self, "calibrations", tuple(sorted(str(c) for c in self.calibrations)))

    @classmethod
    def from_result(cls, result: ScoreResult, view: str | None = None, *, unit_type: str | None = None) -> AssessmentContext:
        """The context of a scored result under one view.

        ``unit_type`` is the unit the *caller* declares the scored rows to be.
        Left ``None`` — as :func:`wise.prioritize` leaves it, because a backlog
        aggregates a column of scores and has no opinion about what they count
        — a comparator's declared unit is reported as unverifiable rather than
        contradicted.
        """
        calibrations: tuple[str, ...] = ()
        if result.manifest is not None:
            calibrations = tuple(str(c.get("calibration_id", "")) for c in result.manifest.calibrations)
        return cls(
            view=view,
            scoring_mode=result.mode,
            unit_type=unit_type,
            layer_ids=tuple(result.norm.layer_ids),
            norm_fingerprint=result.norm_fingerprint or result.norm.fingerprint(),
            calibrations=calibrations,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "view": self.view,
            "scoring_mode": self.scoring_mode,
            "unit_type": self.unit_type,
            "layer_ids": list(self.layer_ids),
            "norm_fingerprint": self.norm_fingerprint,
            "calibrations": list(self.calibrations),
            "score_convention": self.score_convention,
            "aggregation": self.aggregation,
        }


@dataclass(frozen=True)
class BaselineCompatibility:
    """The verdict of comparing a specification with an assessment context.

    :attr:`compatible` concerns the *semantics* of the comparison.
    :attr:`additive_attribution` is stricter: it also requires a complete
    reference layer profile, because a scalar comparator is a valid priority
    comparator with no uniquely determined decomposition.
    """

    compatible: bool
    additive_attribution: bool
    reasons: tuple[str, ...] = ()
    undeclared: tuple[str, ...] = ()
    unverifiable: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(str(r) for r in self.reasons))
        object.__setattr__(self, "undeclared", tuple(str(u) for u in self.undeclared))
        object.__setattr__(self, "unverifiable", tuple(str(u) for u in self.unverifiable))
        object.__setattr__(self, "notes", tuple(str(n) for n in self.notes))

    def raise_if_incompatible(self, baseline_id: str) -> None:
        """Refuse the comparison, naming every mismatch."""
        if self.compatible:
            return
        raise BaselineError(
            f"baseline {baseline_id!r} is not comparable with this assessment: "
            + "; ".join(self.reasons)
            + ". Supply a reviewed mapping, or use a comparator produced under the same semantics — "
            "an unknown layer is never aligned with zero."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "additive_attribution": self.additive_attribution,
            "reasons": list(self.reasons),
            "undeclared": list(self.undeclared),
            "unverifiable": list(self.unverifiable),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class BaselineSpec:
    """A comparator: who it is, what it asserts, and under which semantics.

    Parameters mirror the fields. ``reference_score`` and
    ``reference_layer_penalties`` may both be ``None`` for
    :attr:`BaselineKind.CURRENT_POPULATION`, which is resolved from the run
    itself; for a historical or target comparator the score is required.

    When a complete profile is supplied, the identity

    ``reference_score == 1 − Σ reference_layer_penalties``

    is verified within :attr:`tolerance` (default ``1e-9``), and the score is
    derived from the profile when only the profile is given.

    >>> spec = BaselineSpec.historical(0.75, {"a": 0.15, "b": 0.10}, baseline_id="2025-Q4")
    >>> spec.reference_score
    0.75
    >>> spec.profile_for(["a", "b"])
    {'a': 0.15, 'b': 0.1}
    >>> spec.profile_for(["a", "b", "c"])
    Traceback (most recent call last):
        ...
    wise.explain.baseline.BaselineError: baseline '2025-Q4' has no penalty for layer 'c'; an unknown layer is not zero
    """

    baseline_id: str
    kind: BaselineKind = BaselineKind.CURRENT_POPULATION
    reference_score: float | None = None
    reference_layer_penalties: dict[str, float] | None = None
    view: str | None = None
    scoring_mode: str | None = None
    unit_type: str = "case"
    norm_fingerprint: str | None = None
    score_convention: str = SCORE_CONVENTION
    aggregation: str = AGGREGATION
    calibrations: tuple[str, ...] = ()
    reviewed_mapping: dict[str, Any] | None = None
    population_id: str | None = None
    population_size: int | None = None
    source_run_id: str | None = None
    observed_at: str | None = None
    description: str = ""
    tolerance: float = DEFAULT_TOLERANCE
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = BASELINE_SCHEMA_VERSION

    # ------------------------------------------------------------ validation
    def __post_init__(self) -> None:
        object.__setattr__(self, "baseline_id", _text(self.baseline_id, "baseline_id"))
        object.__setattr__(self, "kind", BaselineKind(self.kind))
        object.__setattr__(self, "unit_type", _text(self.unit_type, "unit_type"))
        object.__setattr__(self, "calibrations", tuple(sorted(str(c) for c in self.calibrations)))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "reviewed_mapping", None if self.reviewed_mapping is None else dict(self.reviewed_mapping))
        tolerance = _finite(self.tolerance, "tolerance")
        if tolerance < 0:
            raise BaselineError(f"tolerance must be non-negative, got {tolerance!r}")
        object.__setattr__(self, "tolerance", tolerance)

        profile = self.reference_layer_penalties
        if profile is not None:
            clean: dict[str, float] = {}
            for layer, value in dict(profile).items():
                name = _text(layer, "layer id")
                penalty = _finite(value, f"reference penalty of layer {name!r}")
                if penalty < 0:
                    raise BaselineError(
                        f"reference penalty of layer {name!r} is {penalty!r}; a layer penalty is non-negative by construction"
                    )
                clean[name] = penalty
            if not clean:
                raise BaselineError(
                    f"baseline {self.baseline_id!r}: an empty layer profile is not a profile; "
                    "pass None for a scalar-only comparator"
                )
            total = math.fsum(clean.values())
            if total > 1.0 + tolerance:
                raise BaselineError(
                    f"baseline {self.baseline_id!r}: the reference layer penalties sum to {total!r}, which exceeds one; "
                    "penalties are shares of a score in [0, 1]"
                )
            object.__setattr__(self, "reference_layer_penalties", clean)
            implied = 1.0 - total
            if self.reference_score is None:
                object.__setattr__(self, "reference_score", implied)
            else:
                score = _finite(self.reference_score, "reference_score")
                if abs(score - implied) > tolerance:
                    raise BaselineError(
                        f"baseline {self.baseline_id!r}: reference_score {score!r} and the layer profile disagree — "
                        f"1 − Σ penalties is {implied!r}, a difference of {abs(score - implied):.3e} "
                        f"above the tolerance {tolerance:g}"
                    )
                object.__setattr__(self, "reference_score", score)

        if self.reference_score is not None:
            score = _finite(self.reference_score, "reference_score")
            if not (0.0 - tolerance <= score <= 1.0 + tolerance):
                raise BaselineError(f"baseline {self.baseline_id!r}: reference_score {score!r} is outside [0, 1]")
            object.__setattr__(self, "reference_score", score)
        elif self.kind is not BaselineKind.CURRENT_POPULATION:
            raise BaselineError(
                f"baseline {self.baseline_id!r}: a {self.kind.value} comparator must state its reference score; "
                "only a current-population comparator is resolved from the run itself"
            )

    # --------------------------------------------------------- constructors
    @classmethod
    def current_population(
        cls,
        *,
        baseline_id: str = "current-population",
        view: str | None = None,
        scoring_mode: str | None = None,
        unit_type: str = "case",
        norm_fingerprint: str | None = None,
        description: str = "",
    ) -> BaselineSpec:
        """The legacy comparator, named: the mean over the run's own scored units."""
        return cls(
            baseline_id=baseline_id,
            kind=BaselineKind.CURRENT_POPULATION,
            view=view,
            scoring_mode=scoring_mode,
            unit_type=unit_type,
            norm_fingerprint=norm_fingerprint,
            description=description or "the unweighted mean over the scored units of this run",
        )

    @classmethod
    def historical(
        cls,
        reference_score: float | None,
        reference_layer_penalties: Mapping[str, float] | None = None,
        **kwargs: Any,
    ) -> BaselineSpec:
        """A frozen comparator observed on an earlier population."""
        kwargs.setdefault("baseline_id", "historical")
        return cls(
            kind=BaselineKind.HISTORICAL,
            reference_score=reference_score,
            reference_layer_penalties=None if reference_layer_penalties is None else dict(reference_layer_penalties),
            **kwargs,
        )

    @classmethod
    def target(
        cls,
        reference_score: float | None,
        reference_layer_penalties: Mapping[str, float] | None = None,
        **kwargs: Any,
    ) -> BaselineSpec:
        """An agreed target. Usually scalar-only, and then explicitly so."""
        kwargs.setdefault("baseline_id", "target")
        return cls(
            kind=BaselineKind.TARGET,
            reference_score=reference_score,
            reference_layer_penalties=None if reference_layer_penalties is None else dict(reference_layer_penalties),
            **kwargs,
        )

    @classmethod
    def from_result(
        cls,
        result: ScoreResult,
        view: str | None = None,
        *,
        baseline_id: str | None = None,
        kind: BaselineKind | str = BaselineKind.HISTORICAL,
        population_id: str | None = None,
        unit_type: str = "case",
        description: str = "",
    ) -> BaselineSpec:
        """Freeze the comparator a scored result actually exhibits.

        The reference score is the unweighted mean over the scored units and
        the profile is their mean layer penalties, so the identity
        ``reference_score == 1 − Σ penalties`` holds by construction. Use it to
        turn last period's run into this period's comparator; the two
        populations are expected to differ.
        """
        view = _one_view(result, view)
        scored = result.scores[view].notna()
        if not bool(scored.any()):
            raise BaselineError(f"no scored {unit_type} under view {view!r}, so no comparator can be frozen from this result")
        contributions = result.contributions[view][scored]
        profile = {str(layer): float(contributions[layer].mean()) for layer in result.norm.layer_ids}
        run_id = None if result.manifest is None else result.manifest.run_id
        return cls(
            baseline_id=baseline_id or f"{BaselineKind(kind).value}:{view}:{run_id or 'unidentified-run'}",
            kind=BaselineKind(kind),
            reference_score=float(result.scores[view][scored].mean()),
            reference_layer_penalties=profile,
            view=view,
            scoring_mode=result.mode,
            unit_type=unit_type,
            norm_fingerprint=result.norm_fingerprint or result.norm.fingerprint(),
            calibrations=tuple(
                str(c.get("calibration_id", "")) for c in (result.manifest.calibrations if result.manifest else ())
            ),
            population_id=population_id,
            population_size=int(scored.sum()),
            source_run_id=run_id,
            description=description,
        )

    # -------------------------------------------------------------- queries
    @property
    def scalar_only(self) -> bool:
        """True when no layer profile is available, so no additive attribution is."""
        return self.reference_layer_penalties is None

    @property
    def layer_ids(self) -> tuple[str, ...]:
        """The layers this comparator declares, in their stated order."""
        return () if self.reference_layer_penalties is None else tuple(self.reference_layer_penalties)

    @property
    def normalisation(self) -> str | None:
        """The normalisation identity — in this library, the scoring mode.

        ``"flat"`` renormalises the raw constraint weights over the applicable
        set; ``"layer_balanced"`` first averages within each applicable layer.
        They give different penalties for the same observations, so a profile
        produced under one is not a reference for the other.
        """
        return self.scoring_mode

    @property
    def supports_additive_attribution(self) -> bool:
        """Whether a layer profile is stated, or resolvable from the run itself.

        A current-population comparator that states no score of its own is
        resolved from the assessment, profile included, so it does support an
        additive contrast even though :attr:`scalar_only` is true of the
        *specification*. Any other comparator must state its profile.
        """
        return not self.scalar_only or (self.kind is BaselineKind.CURRENT_POPULATION and self.reference_score is None)

    @property
    def explanation_kind(self) -> str:
        """The roadmap's interchange label for what this comparator supports."""
        return "absolute_and_reference_contrast" if self.supports_additive_attribution else "absolute_only"

    def profile_for(self, layer_ids: Sequence[str]) -> dict[str, float]:
        """The reference penalties for exactly ``layer_ids``, or an error.

        A layer the comparator does not mention is **not** zero: it is unknown,
        and an unknown reference cannot be subtracted. A layer the comparator
        mentions but the norm does not define is equally a mismatch.
        """
        if self.reference_layer_penalties is None:
            raise BaselineError(
                f"baseline {self.baseline_id!r} is scalar-only: it has no layer profile, so no additive "
                "layer attribution against it is determined (reference_profile_unavailable)"
            )
        wanted = [str(layer) for layer in layer_ids]
        for layer in wanted:
            if layer not in self.reference_layer_penalties:
                raise BaselineError(
                    f"baseline {self.baseline_id!r} has no penalty for layer {layer!r}; an unknown layer is not zero"
                )
        extra = sorted(set(self.reference_layer_penalties) - set(wanted))
        if extra:
            raise BaselineError(
                f"baseline {self.baseline_id!r} declares layers this assessment does not define: {extra}; "
                "the layer partitions differ, so the profiles are not comparable"
            )
        return {layer: float(self.reference_layer_penalties[layer]) for layer in wanted}

    def check_compatible(self, context: AssessmentContext) -> BaselineCompatibility:
        """Compare assessment semantics — never population identity.

        A field the specification leaves ``None`` is reported as *undeclared*
        rather than assumed to match; a field the *assessment* cannot state —
        a bare score frame knows no scoring mode — is reported as
        *unverifiable*, which is a limitation, not a mismatch. A field both
        sides declare must be equal. A changed norm or a changed calibration
        set is a mismatch unless :attr:`reviewed_mapping` authorises it.
        """
        reasons: list[str] = []
        undeclared: list[str] = []
        unverifiable: list[str] = []
        notes: list[str] = []

        def compare(name: str, declared: Any, actual: Any, what: str) -> None:
            if declared is None:
                undeclared.append(name)
                return
            if actual is None:
                unverifiable.append(name)
                return
            if declared != actual:
                reasons.append(f"{what}: baseline {declared!r} vs. assessment {actual!r}")

        compare("view", self.view, context.view, "view")
        compare("scoring_mode", self.scoring_mode, context.scoring_mode, "normalisation (scoring mode)")
        compare("unit_type", self.unit_type, context.unit_type, "assessment-unit type")
        if self.score_convention != context.score_convention:
            reasons.append(f"score convention: baseline {self.score_convention!r} vs. assessment {context.score_convention!r}")
        if self.aggregation != context.aggregation:
            reasons.append(f"aggregation convention: baseline {self.aggregation!r} vs. assessment {context.aggregation!r}")
        if self.reference_layer_penalties is not None and context.layer_ids:
            declared, actual = set(self.layer_ids), set(context.layer_ids)
            if declared != actual:
                missing = sorted(actual - declared)
                extra = sorted(declared - actual)
                detail = []
                if missing:
                    detail.append(f"no reference for {missing}")
                if extra:
                    detail.append(f"unknown here: {extra}")
                reasons.append("layer semantics: " + ", ".join(detail))
        if self.norm_fingerprint is None:
            undeclared.append("norm_fingerprint")
        elif context.norm_fingerprint is None:
            unverifiable.append("norm_fingerprint")
        elif self.norm_fingerprint != context.norm_fingerprint:
            if self.reviewed_mapping:
                notes.append(
                    "the norm changed since this comparator was observed; an explicit reviewed mapping was supplied "
                    f"({sorted(self.reviewed_mapping)})"
                )
            else:
                reasons.append(
                    f"norm identity: baseline {self.norm_fingerprint[:12]}… vs. assessment "
                    f"{(context.norm_fingerprint or '')[:12]}…; a changed norm needs an explicit reviewed mapping"
                )
        if self.calibrations != context.calibrations:
            if self.reviewed_mapping:
                notes.append("the calibrations differ; an explicit reviewed mapping was supplied")
            else:
                reasons.append(
                    f"calibration mapping: baseline {list(self.calibrations)} vs. assessment {list(context.calibrations)}"
                )
        compatible = not reasons
        return BaselineCompatibility(
            compatible=compatible,
            additive_attribution=compatible and self.supports_additive_attribution,
            reasons=tuple(reasons),
            undeclared=tuple(undeclared),
            unverifiable=tuple(unverifiable),
            notes=tuple(notes),
        )

    # --------------------------------------------------------------- export
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "baseline_id": self.baseline_id,
            "kind": self.kind.value,
            "reference_score": self.reference_score,
            "reference_layer_penalties": None if self.reference_layer_penalties is None else dict(self.reference_layer_penalties),
            "explanation_kind": self.explanation_kind,
            "view": self.view,
            "scoring_mode": self.scoring_mode,
            "normalisation": self.normalisation,
            "unit_type": self.unit_type,
            "norm_fingerprint": self.norm_fingerprint,
            "score_convention": self.score_convention,
            "aggregation": self.aggregation,
            "calibrations": list(self.calibrations),
            "reviewed_mapping": None if self.reviewed_mapping is None else dict(self.reviewed_mapping),
            "population_id": self.population_id,
            "population_size": self.population_size,
            "source_run_id": self.source_run_id,
            "observed_at": self.observed_at,
            "description": self.description,
            "tolerance": self.tolerance,
            "metadata": dict(self.metadata),
        }

    def to_interchange(self) -> dict[str, Any]:
        """The comparator in the roadmap's illustrative interchange shape.

        Four fields — ``baseline_id``, ``kind``, ``score``, ``layer_profile`` —
        as :func:`wise.evidence.to_interchange` expects them. A scalar-only
        comparator exports ``layer_profile: null``, which is what forces
        ``explanation_kind: "absolute_only"`` in that contract.
        """
        if self.reference_score is None:
            raise BaselineError(
                f"baseline {self.baseline_id!r} has no reference score yet; resolve it against a run before exporting it"
            )
        return {
            "baseline_id": self.baseline_id,
            "kind": self.kind.value,
            "score": float(self.reference_score),
            "layer_profile": None if self.reference_layer_penalties is None else dict(self.reference_layer_penalties),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BaselineSpec:
        """Rebuild a specification from :meth:`to_dict` or a reviewed JSON file.

        Unknown keys are rejected: a comparator read from a file is
        configuration, and a silently ignored field is a silently different
        comparison.
        """
        known = {
            "schema_version",
            "baseline_id",
            "kind",
            "reference_score",
            "reference_layer_penalties",
            "view",
            "scoring_mode",
            "unit_type",
            "norm_fingerprint",
            "score_convention",
            "aggregation",
            "calibrations",
            "reviewed_mapping",
            "population_id",
            "population_size",
            "source_run_id",
            "observed_at",
            "description",
            "tolerance",
            "metadata",
        }
        derived = {"explanation_kind", "normalisation"}
        unknown = sorted(set(data) - known - derived)
        if unknown:
            raise BaselineError(f"unknown keys in a baseline specification: {unknown}")
        version = str(data.get("schema_version", BASELINE_SCHEMA_VERSION))
        if version != BASELINE_SCHEMA_VERSION:
            raise BaselineError(
                f"unsupported baseline schema version {version!r}; this library reads {BASELINE_SCHEMA_VERSION!r}"
            )
        payload = {k: v for k, v in data.items() if k in known and k != "schema_version"}
        if "calibrations" in payload and payload["calibrations"] is not None:
            payload["calibrations"] = tuple(payload["calibrations"])
        if "tolerance" in payload and payload["tolerance"] is None:
            payload.pop("tolerance")
        return cls(**payload)  # type: ignore[arg-type]

    def with_view(self, view: str) -> BaselineSpec:
        """A copy pinned to ``view``. Used when a call names the view instead."""
        return replace(self, view=view)


@dataclass(frozen=True)
class ResolvedBaseline:
    """One comparator, resolved once, for both the priority and its explanation.

    :attr:`reference_score` is the number the gap is measured against;
    :attr:`reference_layer_penalties` is the profile the layer deltas are
    measured against, or ``None`` when the comparator is scalar-only.
    :attr:`population_mean` is always the run's own mean, so a report can say
    what the current population looks like even when the comparator is not it.
    """

    spec: BaselineSpec
    reference_score: float
    reference_layer_penalties: dict[str, float] | None
    compatibility: BaselineCompatibility
    resolved_from: str
    population_mean: float
    population_size: int
    context: AssessmentContext

    @property
    def additive_attribution(self) -> bool:
        return self.reference_layer_penalties is not None

    @property
    def explanation_kind(self) -> str:
        return "absolute_and_reference_contrast" if self.additive_attribution else "absolute_only"

    @property
    def is_global(self) -> bool:
        """Whether the reference really is the current population's own mean."""
        return self.spec.kind is BaselineKind.CURRENT_POPULATION and self.resolved_from == "current_population"

    def to_interchange(self) -> dict[str, Any]:
        """The **resolved** comparator in the interchange shape.

        Unlike :meth:`BaselineSpec.to_interchange` this always has a score,
        and a current-population comparator carries the profile that was
        actually resolved from the run.
        """
        return {
            "baseline_id": self.spec.baseline_id,
            "kind": self.spec.kind.value,
            "score": float(self.reference_score),
            "layer_profile": None if self.reference_layer_penalties is None else dict(self.reference_layer_penalties),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.spec.to_dict(),
            "reference_score": self.reference_score,
            "reference_layer_penalties": (
                None if self.reference_layer_penalties is None else dict(self.reference_layer_penalties)
            ),
            "explanation_kind": self.explanation_kind,
            "resolved_from": self.resolved_from,
            "population_mean": self.population_mean,
            "population_size": self.population_size,
            "is_current_population": self.is_global,
            "compatibility": self.compatibility.to_dict(),
            "context": self.context.to_dict(),
        }


def resolve_baseline(
    spec: BaselineSpec,
    context: AssessmentContext,
    *,
    population_mean: float,
    population_size: int,
    population_profile: Mapping[str, float] | None = None,
    require_profile: bool = False,
) -> ResolvedBaseline:
    """Resolve a specification against one assessment, once.

    ``population_mean`` and ``population_profile`` describe the run's own
    scored population, computed before any minimum-support filtering, and are
    used only when the specification is a current-population comparator that
    states no score of its own.

    Raises :class:`BaselineError` when the semantics do not match, and — with
    ``require_profile`` — when an additive contrast was demanded of a
    scalar-only comparator.
    """
    compatibility = spec.check_compatible(context)
    compatibility.raise_if_incompatible(spec.baseline_id)
    resolved_from = "specification"
    reference_score = spec.reference_score
    profile: dict[str, float] | None
    if not spec.scalar_only:
        profile = spec.profile_for(context.layer_ids) if context.layer_ids else dict(spec.reference_layer_penalties or {})
    elif reference_score is None and spec.kind is BaselineKind.CURRENT_POPULATION:
        # the legacy comparator, made explicit: the run's own scored population.
        # Score and profile are taken from the same population, so the identity
        # ``score == 1 − Σ penalties`` still holds.
        reference_score = float(population_mean)
        resolved_from = "current_population"
        profile = None if population_profile is None else {str(k): float(v) for k, v in population_profile.items()}
    else:
        # a scalar comparator someone stated: its layer profile is unknown, and
        # the current population's profile is not a substitute for it
        profile = None
    if reference_score is None:  # pragma: no cover - forbidden by BaselineSpec validation
        raise BaselineError(f"baseline {spec.baseline_id!r} has no reference score and none could be resolved")
    if require_profile and profile is None:
        raise BaselineError(
            f"baseline {spec.baseline_id!r} is scalar-only, so an additive layer contrast against it is not determined "
            "(reference_profile_unavailable); report absolute layer penalties instead"
        )
    return ResolvedBaseline(
        spec=spec,
        reference_score=float(reference_score),
        reference_layer_penalties=profile,
        compatibility=compatibility,
        resolved_from=resolved_from,
        population_mean=float(population_mean),
        population_size=int(population_size),
        context=context,
    )


def _one_view(result: ScoreResult, view: str | None) -> str:
    if view is not None:
        if view not in result.views:
            raise BaselineError(f"unknown view {view!r}; available: {result.views}")
        return view
    if len(result.views) != 1:
        raise BaselineError(f"specify view=...; available: {result.views}")
    return result.views[0]


def load_baseline(path: Any) -> BaselineSpec:
    """Read a reviewed comparator from a JSON file.

    The file is configuration, not data: it is parsed strictly, unknown keys
    are refused, and nothing in it is executed.

    >>> import json, tempfile, pathlib
    >>> d = pathlib.Path(tempfile.mkdtemp())
    >>> _ = (d / "b.json").write_text(json.dumps({"baseline_id": "t", "kind": "target", "reference_score": 0.95}))
    >>> load_baseline(d / "b.json").reference_score
    0.95
    """
    import json
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BaselineError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise BaselineError(f"{path}: a baseline file must contain a JSON object")
    return BaselineSpec.from_dict(data)


def norm_context(norm: Norm, view: str | None, *, mode: str | None = None, unit_type: str | None = None) -> AssessmentContext:
    """The assessment context implied by a norm, a view and a scoring mode."""
    return AssessmentContext(
        view=view,
        scoring_mode=mode or norm.scoring_mode,
        unit_type=unit_type,
        layer_ids=tuple(norm.layer_ids),
        norm_fingerprint=norm.fingerprint(),
    )
