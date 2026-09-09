"""Exact, comparator-consistent explanation of one slice's priority.

:func:`explain_priority` answers the five questions a review actually asks —
what was observed, which expectation was used, why this slice ranks where it
does, how another approved view changes the reading, and what still needs
verification — and keeps them apart. It computes nothing of its own: the
backlog row comes from :func:`wise.prioritize` and the layer means from
:func:`wise.layer_drivers`, both called with the **same** comparator, view,
keys, scored mask and aggregation convention, so the explanation cannot drift
from the ranking it explains.

The arithmetic, for a group ``s`` under view ``p`` and comparator ``B``:

.. code-block:: text

    delta_layer  = mean_group_layer_penalty − reference_layer_penalty
    signed_gap   = reference_score − mean_group_score
    rho          = n_scored / (n_scored + gamma)
    PI           = volume · max(signed_gap, 0)
    stable_PI    = volume · rho · max(signed_gap, 0)
    component    = volume · rho · delta_layer

Under a compatible complete profile the layer deltas sum to ``signed_gap``
exactly — the identity is asserted, not assumed — so the signed components sum
to the stabilised index. Negative offsets are kept: a slice can be *better*
than the comparator in one layer and still carry priority. A non-positive gap
clips the index to zero; the unclipped contrasts stay visible beside the
clipping, and no percentage is ever formed by dividing by a zero net gap.

Under a scalar-only comparator there is no profile to subtract. The
explanation then reports the absolute layer penalties, marks the deltas
``null`` and carries the ``reference_profile_unavailable`` limitation. It
never falls back to the current-population profile: that would explain a
comparison nobody made.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..constraints import as_list
from ..errors import EvidenceError
from ..evidence.manifest import _canonical
from ..evidence.models import EvaluationRecord, Qualification, QualificationCode, WitnessRef
from ..prioritization import (
    _is_scored_result,
    _keys,
    _reject_two_comparators,
    constraint_drivers,
    layer_drivers,
    prioritize,
)
from ..scoring import ScoreResult
from .baseline import (
    AssessmentContext,
    BaselineCompatibility,
    BaselineError,
    BaselineKind,
    BaselineSpec,
    ResolvedBaseline,
    _one_view,
    resolve_baseline,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..evidence.manifest import RunManifest
    from ..evidence.models import EvidencePacket
    from ..oc.evaluation import OCScoreResult

#: Version of the explanation contract. Bumped when a field changes meaning.
EXPLANATION_SCHEMA_VERSION = "wise-explanation/1"

#: Tolerance for the reconciliation identities asserted while building a
#: packet (layer deltas against the signed gap, components against the
#: stabilised index, the explanation's comparator against the ranking's). They
#: are exact in real arithmetic; only summation order can move them.
DEFAULT_ATOL = 1e-9

_SAFE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"


def safe_identifier(text: Any, *, max_length: int = 48) -> str:
    """A filesystem- and id-safe token derived from arbitrary text.

    Business labels are untrusted: a vendor may be called ``../../etc`` or
    carry a newline. Everything outside ``[A-Za-z0-9._-]`` becomes ``_``,
    leading dots are dropped so no hidden or relative path can be formed, and
    a truncated or altered label keeps a short digest so two different labels
    cannot collapse into one identifier.

    >>> safe_identifier("Vendor A/B")
    'Vendor_A_B-11db9d71'
    >>> safe_identifier("../../etc/passwd")
    'etc_passwd-3754d6cb'
    >>> safe_identifier("plain")
    'plain'
    """
    raw = str(text)
    cleaned = "".join(ch if ch in _SAFE else "_" for ch in raw).lstrip("._-")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    trimmed = cleaned[:max_length].rstrip("._-")
    if trimmed == raw and trimmed:
        return trimmed
    return f"{trimmed}-{digest}" if trimmed else f"label-{digest}"


class FactKind(str, Enum):
    """Which of the four different statements a fact belongs to.

    Keeping them apart is the point: an observed penalty, a relative priority,
    the same slice under another approved view and a limitation of the
    evidence are four kinds of claim, and merging them is how a report starts
    saying more than it knows.
    """

    OBSERVED_ASSESSMENT = "observed_assessment"
    RELATIVE_PRIORITY = "relative_priority"
    ALTERNATIVE_VIEW = "alternative_view"
    EVIDENCE_QUALIFICATION = "evidence_qualification"


_FACT_PREFIX = {
    FactKind.OBSERVED_ASSESSMENT: "OBS",
    FactKind.RELATIVE_PRIORITY: "PRI",
    FactKind.ALTERNATIVE_VIEW: "ALT",
    FactKind.EVIDENCE_QUALIFICATION: "QUAL",
}


def _number(value: Any) -> float | None:
    """A finite float, or ``None`` — NaN is a status, never a value."""
    if value is None:
        return None
    out = float(value)
    return out if math.isfinite(out) else None


# --------------------------------------------------------------------- parts
@dataclass(frozen=True)
class Denominator:
    """What a number is *of*. A share without one is not a fact."""

    name: str
    description: str
    count: float | None
    unit_of_counting: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "count": _number(self.count),
            "unit_of_counting": self.unit_of_counting,
        }


@dataclass(frozen=True)
class Fact:
    """One authoritative number with its identity, unit and denominator.

    The renderer prints facts; it never computes one. Anything that is not a
    fact of some packet cannot appear as a number in a report.
    """

    fact_id: str
    kind: FactKind
    name: str
    value: float | str | bool | None
    unit: str
    description: str
    evidence_refs: tuple[str, ...]
    denominator: str | None = None
    method: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", FactKind(self.kind))
        object.__setattr__(self, "evidence_refs", tuple(str(r) for r in self.evidence_refs))
        if not self.evidence_refs:
            raise EvidenceError(f"fact {self.fact_id!r} has no evidence reference; every presented number resolves to a run")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise EvidenceError(f"fact {self.fact_id!r}: a non-finite value is a status, not a number; use null and a reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "kind": self.kind.value,
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "description": self.description,
            "denominator": self.denominator,
            "method": self.method,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class LayerComponent:
    """One layer's contribution to one slice's priority.

    ``delta``, ``component`` and ``clipped_component`` are ``None`` under a
    scalar-only comparator — the absolute :attr:`group_penalty` is what is
    known, and nothing else is invented.
    """

    layer: str
    group_penalty: float
    reference_penalty: float | None
    delta: float | None
    component: float | None
    clipped_component: float | None
    share_of_stable_pi: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "group_penalty": _number(self.group_penalty),
            "reference_penalty": _number(self.reference_penalty),
            "delta": _number(self.delta),
            "component": _number(self.component),
            "clipped_component": _number(self.clipped_component),
            "share_of_stable_pi": _number(self.share_of_stable_pi),
        }


@dataclass(frozen=True)
class ConstraintComponent:
    """Which expectations carry the slice's penalty, with their denominators.

    ``share_evaluated`` says how much of the slice this check could be answered
    on. It says nothing about *how* it was answered, and a censored evaluation
    is an answer that is only a bound: ``share_lower_bound`` is the share of the
    evaluated checks that carry a ``lower_bound`` qualification in the run's
    evidence. It is ``None`` when no evidence packet was supplied — unknown,
    which is not the same as zero.
    """

    constraint_id: str
    layer: str
    constraint_type: str
    mean_penalty: float | None
    mean_violation: float | None
    share_violated: float | None
    share_in_scope: float | None
    share_evaluated: float | None
    description: str = ""
    share_lower_bound: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "layer": self.layer,
            "constraint_type": self.constraint_type,
            "mean_penalty": _number(self.mean_penalty),
            "mean_violation": _number(self.mean_violation),
            "share_violated": _number(self.share_violated),
            "share_in_scope": _number(self.share_in_scope),
            "share_evaluated": _number(self.share_evaluated),
            "share_lower_bound": _number(self.share_lower_bound),
            "description": self.description,
        }


@dataclass(frozen=True)
class PriorityComponents:
    """Every number of the priority arithmetic, none of them recomputed."""

    view: str
    n_scored: int
    n_units: int
    volume: float
    volume_definition: str
    mean_score: float
    reference_score: float
    signed_gap: float
    gap: float
    gamma: float
    rho: float
    stable_mean: float
    stable_gap: float
    priority_index: float
    stable_priority_index: float
    clipped: bool
    exposure: float | None = None
    rank: int | None = None
    n_groups: int | None = None
    se: float | None = None
    gap_lower: float | None = None
    priority_index_lower: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "view": self.view,
            "n_scored": int(self.n_scored),
            "n_units": int(self.n_units),
            "volume": _number(self.volume),
            "volume_definition": self.volume_definition,
            "exposure": _number(self.exposure),
            "mean_score": _number(self.mean_score),
            "reference_score": _number(self.reference_score),
            "signed_gap": _number(self.signed_gap),
            "gap": _number(self.gap),
            "gamma": _number(self.gamma),
            "rho": _number(self.rho),
            "stable_mean": _number(self.stable_mean),
            "stable_gap": _number(self.stable_gap),
            "PI": _number(self.priority_index),
            "stable_PI": _number(self.stable_priority_index),
            "clipped": bool(self.clipped),
            "rank": self.rank,
            "n_groups": self.n_groups,
            "se": _number(self.se),
            "gap_lower": _number(self.gap_lower),
            "PI_lower": _number(self.priority_index_lower),
        }


@dataclass(frozen=True)
class ViewContrast:
    """The same slice under another approved view.

    Each view is ranked against **its own** current population, because a
    comparator declared for one view is not a comparator for another. When the
    scored populations differ between views, that is disclosed rather than
    quietly compared.
    """

    view: str
    comparator: str
    n_scored: int
    mean_score: float
    reference_score: float
    signed_gap: float
    stable_priority_index: float
    rank: int | None
    scored_population_differs: bool
    n_scored_population: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "view": self.view,
            "comparator": self.comparator,
            "n_scored": int(self.n_scored),
            "n_scored_population": int(self.n_scored_population),
            "mean_score": _number(self.mean_score),
            "reference_score": _number(self.reference_score),
            "signed_gap": _number(self.signed_gap),
            "stable_PI": _number(self.stable_priority_index),
            "rank": self.rank,
            "scored_population_differs": bool(self.scored_population_differs),
        }


# -------------------------------------------------------------------- packet
@dataclass(frozen=True)
class ExplanationPacket:
    """A deterministic, versioned explanation of one slice's priority.

    It carries the authoritative facts and their identities, the comparator it
    used, the profile and priority components, the denominators every number
    is *of*, the witnesses that support the observations, and the limitations
    that survive whoever writes the prose. Rendering it — as text, Markdown or
    JSON — introduces no number that is not already a :class:`Fact`.
    """

    explanation_id: str
    run_id: str
    unit_type: str
    grouping: tuple[str, ...]
    group_key: tuple[Any, ...]
    group_label: str
    baseline: BaselineSpec
    reference_score: float
    reference_layer_penalties: dict[str, float] | None
    compatibility: BaselineCompatibility
    explanation_kind: str
    priority: PriorityComponents
    layers: tuple[LayerComponent, ...]
    facts: tuple[Fact, ...]
    denominators: tuple[Denominator, ...]
    limitations: tuple[Qualification, ...]
    constraints: tuple[ConstraintComponent, ...] = ()
    view_contrasts: tuple[ViewContrast, ...] = ()
    witnesses: tuple[WitnessRef, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    run: dict[str, Any] = field(default_factory=dict)
    schema_version: str = EXPLANATION_SCHEMA_VERSION
    manifest: RunManifest | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("grouping", "layers", "facts", "denominators", "limitations", "constraints", "view_contrasts", "witnesses"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "group_key", tuple(self.group_key))
        object.__setattr__(self, "evidence_refs", tuple(str(r) for r in self.evidence_refs))
        object.__setattr__(self, "run", dict(self.run))
        ids = [f.fact_id for f in self.facts]
        if len(set(ids)) != len(ids):
            raise EvidenceError("fact ids must be unique inside an explanation packet")

    def __repr__(self) -> str:
        return (
            f"ExplanationPacket({self.group_label!r} under {self.priority.view!r}, "
            f"comparator={self.baseline.baseline_id!r} ({self.baseline.kind.value}), "
            f"stable_PI={self.priority.stable_priority_index:.6g}, {len(self.facts)} facts)"
        )

    # ------------------------------------------------------------- lookups
    @property
    def additive_attribution(self) -> bool:
        """Whether the layer deltas mean anything against this comparator."""
        return self.reference_layer_penalties is not None

    def fact(self, fact_id: str) -> Fact:
        """One fact by id."""
        for f in self.facts:
            if f.fact_id == fact_id:
                return f
        raise EvidenceError(f"unknown fact id {fact_id!r}")

    def facts_of(self, kind: FactKind | str) -> tuple[Fact, ...]:
        """The facts of one kind, in packet order."""
        wanted = FactKind(kind)
        return tuple(f for f in self.facts if f.kind is wanted)

    def denominator(self, name: str) -> Denominator | None:
        for d in self.denominators:
            if d.name == name:
                return d
        return None

    def layer_frame(self) -> pd.DataFrame:
        """The layer components as a table, for a notebook or a join."""
        return pd.DataFrame([layer.to_dict() for layer in self.layers]).set_index("layer")

    def fact_frame(self) -> pd.DataFrame:
        """Every fact as a row, with its kind, unit, denominator and references."""
        rows = [f.to_dict() for f in self.facts]
        frame = pd.DataFrame(rows)
        return frame.set_index("fact_id") if len(frame) else frame

    # -------------------------------------------------------------- export
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "explanation_id": self.explanation_id,
            "run_id": self.run_id,
            "unit_type": self.unit_type,
            "grouping": list(self.grouping),
            "group_key": [None if _isna(k) else _plain(k) for k in self.group_key],
            "group_label": self.group_label,
            "baseline": self.baseline.to_dict(),
            "reference_score": _number(self.reference_score),
            "reference_layer_penalties": (
                None if self.reference_layer_penalties is None else dict(self.reference_layer_penalties)
            ),
            "explanation_kind": self.explanation_kind,
            "compatibility": self.compatibility.to_dict(),
            "priority": self.priority.to_dict(),
            "layers": [layer.to_dict() for layer in self.layers],
            "constraints": [c.to_dict() for c in self.constraints],
            "view_contrasts": [v.to_dict() for v in self.view_contrasts],
            "facts": [f.to_dict() for f in self.facts],
            "denominators": [d.to_dict() for d in self.denominators],
            "limitations": [q.to_dict() for q in self.limitations],
            "witnesses": [w.to_dict() for w in self.witnesses],
            "evidence_refs": list(self.evidence_refs),
            "run": dict(self.run),
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Deterministic UTF-8 JSON with explicit nulls and ``allow_nan=False``.

        Key order is the packet's own, not alphabetical, so two renderings of
        one packet are byte-identical and a diff of two runs reads top to
        bottom. A non-finite number is refused rather than exported.
        """
        return json.dumps(_canonical(self.to_dict()), indent=indent, ensure_ascii=False, allow_nan=False, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExplanationPacket:
        """Rebuild a packet from :meth:`to_dict`, for rendering it elsewhere.

        The live run manifest is not reconstructed — its serialised form stays
        available under :attr:`run` — because a manifest is a record of an
        execution, not something a reader may re-derive.
        """
        version = str(data.get("schema_version", ""))
        if version != EXPLANATION_SCHEMA_VERSION:
            raise EvidenceError(
                f"unsupported explanation schema version {version!r}; this library reads {EXPLANATION_SCHEMA_VERSION!r}"
            )
        return cls(
            explanation_id=str(data["explanation_id"]),
            run_id=str(data["run_id"]),
            unit_type=str(data["unit_type"]),
            grouping=tuple(data["grouping"]),
            group_key=tuple(data["group_key"]),
            group_label=str(data["group_label"]),
            baseline=BaselineSpec.from_dict(data["baseline"]),
            reference_score=float(data["reference_score"]),
            reference_layer_penalties=data.get("reference_layer_penalties"),
            compatibility=BaselineCompatibility(**data["compatibility"]),
            explanation_kind=str(data["explanation_kind"]),
            priority=_priority_from_dict(data["priority"]),
            layers=tuple(LayerComponent(**row) for row in data["layers"]),
            constraints=tuple(ConstraintComponent(**row) for row in data.get("constraints", [])),
            view_contrasts=tuple(_view_contrast_from_dict(row) for row in data.get("view_contrasts", [])),
            facts=tuple(_fact_from_dict(row) for row in data["facts"]),
            denominators=tuple(Denominator(**row) for row in data["denominators"]),
            limitations=tuple(Qualification(**row) for row in data["limitations"]),
            witnesses=tuple(_witness_from_dict(row) for row in data.get("witnesses", [])),
            evidence_refs=tuple(data.get("evidence_refs", ())),
            run=dict(data.get("run", {})),
        )


def _priority_from_dict(row: Mapping[str, Any]) -> PriorityComponents:
    payload = dict(row)
    payload["priority_index"] = payload.pop("PI")
    payload["stable_priority_index"] = payload.pop("stable_PI")
    payload["priority_index_lower"] = payload.pop("PI_lower")
    return PriorityComponents(**payload)


def _view_contrast_from_dict(row: Mapping[str, Any]) -> ViewContrast:
    payload = dict(row)
    payload["stable_priority_index"] = payload.pop("stable_PI")
    return ViewContrast(**payload)


def _fact_from_dict(row: Mapping[str, Any]) -> Fact:
    payload = dict(row)
    payload["evidence_refs"] = tuple(payload.get("evidence_refs", ()))
    return Fact(**payload)


def _witness_from_dict(row: Mapping[str, Any]) -> WitnessRef:
    from ..evidence.models import AbsenceSearch

    payload = {k: v for k, v in row.items() if k != "witness_id"}
    search = payload.get("search")
    payload["search"] = None if search is None else AbsenceSearch(**search)
    return WitnessRef(**payload)


def _plain(value: Any) -> Any:
    """A JSON-native copy of a group-key part (NumPy scalars included)."""
    return value.item() if hasattr(value, "item") else value


def _isna(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False


# ------------------------------------------------------------- key handling
def _as_key(value: Any, n_keys: int) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        key = value
    elif isinstance(value, list):
        key = tuple(value)
    else:
        key = (value,)
    if len(key) != n_keys:
        raise BaselineError(f"group key {value!r} has {len(key)} part(s), but the grouping has {n_keys}")
    return key


def _index_keys(frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    return [k if isinstance(k, tuple) else (k,) for k in frame.index]


def _same(a: Any, b: Any) -> bool:
    if _isna(a) and _isna(b):
        return True
    if _isna(a) or _isna(b):
        return False
    return bool(a == b) or str(a) == str(b)


def _locate(frame: pd.DataFrame, key: Sequence[Any]) -> int:
    for position, candidate in enumerate(_index_keys(frame)):
        if len(candidate) == len(key) and all(_same(a, b) for a, b in zip(candidate, key)):
            return position
    raise LookupError(key)


def _label(key: Sequence[Any]) -> str:
    return " | ".join("<null>" if _isna(part) else str(part) for part in key)


# ------------------------------------------------------------------ helpers
def _implicit_spec(result: ScoreResult, view: str, baseline: float | None, unit_type: str) -> BaselineSpec:
    """Name the comparator a legacy call used, without changing it."""
    context = AssessmentContext.from_result(result, view, unit_type=unit_type)
    if baseline is None:
        return BaselineSpec.current_population(
            baseline_id=f"current-population:{view}",
            view=view,
            scoring_mode=context.scoring_mode,
            unit_type=unit_type,
            norm_fingerprint=context.norm_fingerprint,
        )
    return BaselineSpec(
        baseline_id=f"explicit-scalar:{float(baseline):.12g}",
        kind=BaselineKind.TARGET,
        reference_score=float(baseline),
        view=view,
        scoring_mode=context.scoring_mode,
        unit_type=unit_type,
        norm_fingerprint=context.norm_fingerprint,
        calibrations=context.calibrations,
        description="a scalar comparator supplied through the historical baseline= argument; it has no layer profile",
    )


def _declared_unit_type(result: ScoreResult | OCScoreResult, declared: str | None) -> str:
    """What the scored rows count, resolved once.

    An explicit ``unit_type=`` always wins — it is the caller's declaration and
    :class:`~wise.explain.baseline.BaselineSpec` compares it exactly. Left
    unset, a case run stays ``"case"`` (the historical default, unchanged) and
    an object run reports its own assessment unit, which is the whole reason
    that run exists. A run holding several unit types has no single answer and
    is refused here rather than labelled with one of them.
    """
    if declared is not None:
        return str(declared)
    held = tuple(getattr(result, "unit_types", ()) or ())
    if not held:
        return "case"
    if len(held) > 1:
        raise BaselineError(
            f"this run scored the unit types {list(held)}; an explanation declares what its rows count, "
            "so name it with unit_type=... (and rank inside one type: their counts are not one volume)"
        )
    return str(held[0])


def _group_mask(result: ScoreResult | OCScoreResult, keys: Sequence[str], key: Sequence[Any]) -> pd.Series:
    """A null-safe membership mask over the cases of one group."""
    table = result.unit_table
    mask = pd.Series(True, index=table.index)
    for name, value in zip(keys, key):
        if name in table.columns:
            column = table[name]
        elif name == table.index.name:
            column = pd.Series(table.index, index=table.index)
        else:  # pragma: no cover - prioritize already refused this grouping
            raise BaselineError(f"slice column {name!r} is not a case attribute")
        mask &= column.isna() if _isna(value) else (column == value)
    return mask


def _row_value(row: pd.Series, name: str) -> float | None:
    return _number(row[name]) if name in row.index else None


def _check(actual: float, expected: float, atol: float, what: str) -> None:
    if abs(actual - expected) > atol:
        raise BaselineError(
            f"{what}: the explanation and the ranking disagree by {abs(actual - expected):.3e} "
            f"({actual!r} vs. {expected!r}); they must use the same comparator, mask and aggregation"
        )


def explain_priority(
    result: ScoreResult,
    by: str | Sequence[str],
    group: Any = None,
    *,
    view: str | None = None,
    gamma: float = 0.0,
    volume: str = "cases",
    min_cases: int = 1,
    z: float | None = None,
    baseline: float | None = None,
    baseline_spec: BaselineSpec | None = None,
    layers: Sequence[str] | None = None,
    views: Sequence[str] | None = None,
    evidence: EvidencePacket | None = None,
    witness_limit: int = 4,
    max_constraints: int = 10,
    unit_type: str | None = None,
    atol: float = DEFAULT_ATOL,
) -> ExplanationPacket:
    """Explain exactly why one slice carries the priority it does.

    Parameters
    ----------
    result
        The scored result. An explanation carries a run identity, so a bare
        score frame is not enough.
    by, view, gamma, volume, min_cases, z, baseline, baseline_spec
        Exactly the arguments of :func:`wise.prioritize`, and used to call it:
        the packet explains the backlog those arguments produce, never a
        differently configured one. ``baseline`` and ``baseline_spec`` remain
        mutually exclusive.
    group
        The slice key — a scalar for a single key, a tuple for several,
        ``None`` for the top-ranked slice. Null keys are matched as nulls.
    layers
        Restrict the layer decomposition, as in :func:`wise.layer_drivers`.
    views
        Views to contrast against, each ranked against its own current
        population. Default: every view of the result.
    evidence
        An :class:`~wise.evidence.EvidencePacket` from the same run. When
        given, bounded witnesses and coverage facts are attached.
    unit_type
        What the scored rows count. ``None`` (the default) resolves to
        ``"case"`` for a :class:`~wise.scoring.ScoreResult` and to the run's
        own assessment unit for an
        :class:`~wise.oc.evaluation.OCScoreResult`, so an object run is
        explained as what it actually assessed rather than as a case run.

    Returns
    -------
    :class:`ExplanationPacket`

    Raises
    ------
    ~wise.explain.BaselineError
        When the comparator is incompatible with this assessment, when an
        additive contrast is demanded of a scalar-only comparator, or when the
        reconciliation identities do not hold within ``atol``.

    Examples
    --------
    >>> import wise
    >>> result = wise.score(wise.datasets.running_p2p_log(), wise.datasets.running_p2p_norm())
    >>> packet = explain_priority(result, "company", "B", view="Finance", gamma=1.0)
    >>> round(packet.priority.stable_priority_index, 6)
    0.157125
    >>> packet.explanation_kind
    'absolute_and_reference_contrast'
    >>> round(sum(layer.delta for layer in packet.layers), 12) == round(packet.priority.signed_gap, 12)
    True
    """
    if not _is_scored_result(result):
        raise BaselineError(
            "explain_priority needs a scored result (a ScoreResult or an OCScoreResult): an explanation "
            "resolves every number to a run, and a bare score frame carries no run identity"
        )
    unit_type = _declared_unit_type(result, unit_type)
    _reject_two_comparators(baseline, baseline_spec)
    view = _one_view(result, view)
    keys = _keys(by)
    spec = baseline_spec if baseline_spec is not None else _implicit_spec(result, view, baseline, unit_type)
    layer_names = [str(layer) for layer in (as_list(layers) or result.norm.layer_ids)]

    # 1. the ranking itself, and the ranking without the support filter, so the
    #    slice can be explained even when the filter removed it from the backlog
    full = prioritize(result, keys, view=view, gamma=gamma, volume=volume, min_cases=1, z=z, baseline_spec=spec)
    ranked = (
        full
        if min_cases <= 1
        else prioritize(result, keys, view=view, gamma=gamma, volume=volume, min_cases=min_cases, z=z, baseline_spec=spec)
    )
    if group is None:
        source = ranked if len(ranked) else full
        if not len(source):  # pragma: no cover - prioritize raises before this
            raise BaselineError("no slice to explain")
        key = _index_keys(source)[0]
    else:
        key = _as_key(group, len(keys))
    try:
        row = full.iloc[_locate(full, key)]
    except LookupError:
        raise BaselineError(f"no slice {_label(key)!r} in the backlog grouped by {keys}") from None
    try:
        rank: int | None = _locate(ranked, key) + 1
    except LookupError:
        rank = None

    # 2. the same comparator, resolved once, with the run's own population
    scored = result.scores[view].notna()
    population_mean = float(result.scores[view][scored].mean())
    # the population profile always covers the norm's whole layer partition: a
    # layers= restriction narrows what is *shown*, never what is compared
    population_profile = {str(name): float(result.contributions[view][name][scored].mean()) for name in result.norm.layer_ids}
    resolved = resolve_baseline(
        spec,
        AssessmentContext.from_result(result, view, unit_type=unit_type),
        population_mean=population_mean,
        population_size=int(scored.sum()),
        population_profile=population_profile,
    )
    reference_score = float(full.attrs["baseline"])
    _check(resolved.reference_score, reference_score, atol, "the resolved comparator and the ranking's reference")

    # 3. the layer means, from the same grouping, mask and comparator
    drivers = layer_drivers(result, keys, view=view, layers=layers, baseline_spec=spec)
    driver_row = drivers.iloc[_locate(drivers, key)]
    n_scored = int(row["n_cases"])
    if int(driver_row["n_cases"]) != n_scored:  # pragma: no cover - defensive
        raise BaselineError("the priority and its layer decomposition disagree about the scored population of this slice")

    priority = _components(row, view=view, gamma=gamma, volume=volume, reference_score=reference_score, atol=atol)
    priority = _with_group_counts(priority, result, keys, key, rank=rank, n_groups=len(ranked))
    complete_partition = set(layer_names) == set(str(name) for name in result.norm.layer_ids)
    layer_components = _layer_components(
        driver_row, layer_names, resolved=resolved, priority=priority, atol=atol, complete=complete_partition
    )
    # the evidence rows of this slice, read once: the constraint table needs
    # their censored share and the limitations need their qualifications
    slice_records = _records_in_slice(evidence, _slice_unit_ids(result, view, keys, key))
    constraints = _constraint_components(
        result, view, keys, key, max_constraints=max_constraints, records=slice_records, evidence=evidence
    )
    contrasts, absent_views = _view_contrasts(
        result, keys, key, view=view, views=views, gamma=gamma, volume=volume, min_cases=min_cases
    )
    differing_views = _differing_populations(result, view, views)
    witnesses, evidence_refs, missing_witnesses = _witnesses_for_group(result, view, keys, key, evidence, limit=witness_limit)

    limitations = _limitations(
        resolved=resolved,
        priority=priority,
        key=key,
        rank=rank,
        min_cases=min_cases,
        volume=volume,
        differing_views=differing_views,
        absent_views=absent_views,
        evidence=evidence,
        complete_partition=complete_partition,
        record_codes=_record_qualification_counts(slice_records),
        missing_witnesses=missing_witnesses,
        unit_type=unit_type,
    )
    group_label = _label(key)
    aggregate_ref = f"aggregate:{result.manifest.run_id if result.manifest else 'unidentified-run'}:{view}:" + safe_identifier(
        group_label
    )
    facts = _facts(
        priority=priority,
        layers=layer_components,
        resolved=resolved,
        contrasts=contrasts,
        evidence=evidence,
        refs=(aggregate_ref, *evidence_refs),
        view=view,
        volume=volume,
        unit_type=unit_type,
    )
    denominators = _denominators(priority, resolved, volume=volume, evidence=evidence)
    run_id = result.manifest.run_id if result.manifest is not None else "unidentified-run"
    manifest = None
    if result.manifest is not None:
        manifest = result.manifest.finalize(
            grouping=keys,
            view=view,
            volume=volume,
            comparator=f"{resolved.spec.kind.value}:{resolved.spec.baseline_id}",
            comparator_value=reference_score,
            min_cases=min_cases,
            gamma=gamma,
            z=z,
            diagnostics=("layer_drivers", "constraint_drivers"),
        )
    return ExplanationPacket(
        explanation_id=_explanation_id(run_id, view, keys, key, resolved, gamma, volume, min_cases),
        run_id=run_id,
        unit_type=unit_type,
        grouping=tuple(keys),
        group_key=key,
        group_label=group_label,
        baseline=resolved.spec,
        reference_score=reference_score,
        reference_layer_penalties=resolved.reference_layer_penalties,
        compatibility=resolved.compatibility,
        explanation_kind=resolved.explanation_kind,
        priority=priority,
        layers=layer_components,
        facts=facts,
        denominators=denominators,
        limitations=limitations,
        constraints=constraints,
        view_contrasts=contrasts,
        witnesses=witnesses,
        evidence_refs=(aggregate_ref, *evidence_refs),
        run={} if manifest is None else manifest.to_dict(),
        manifest=manifest,
    )


def _explanation_id(
    run_id: str,
    view: str,
    keys: Sequence[str],
    key: Sequence[Any],
    resolved: ResolvedBaseline,
    gamma: float,
    volume: str,
    min_cases: int,
) -> str:
    """A deterministic identity: the same explanation of the same run repeats it."""
    payload = json.dumps(
        {
            "run": run_id,
            "view": view,
            "by": list(keys),
            "key": [None if _isna(part) else str(part) for part in key],
            "baseline": resolved.spec.baseline_id,
            "reference": resolved.reference_score,
            "gamma": float(gamma),
            "volume": volume,
            "min_cases": int(min_cases),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "exp-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _components(
    row: pd.Series, *, view: str, gamma: float, volume: str, reference_score: float, atol: float
) -> PriorityComponents:
    n_scored = int(row["n_cases"])
    volume_value = float(row["volume"])
    mean_score = float(row["mean_score"])
    signed_gap = reference_score - mean_score
    gap = max(signed_gap, 0.0)
    rho = n_scored / (n_scored + float(gamma)) if gamma > 0 else 1.0
    priority_index = volume_value * gap
    stable_priority_index = volume_value * rho * gap
    # the ranking's own numbers must be reproduced exactly by these definitions
    _check(gap, float(row["gap"]), atol, "gap")
    _check(priority_index, float(row["PI"]), atol, "PI")
    _check(rho * gap, float(row["stable_gap"]), atol, "stable_gap")
    _check(stable_priority_index, float(row["stable_PI"]), atol, "stable_PI")
    return PriorityComponents(
        view=view,
        n_scored=n_scored,
        n_units=n_scored,
        volume=volume_value,
        volume_definition=(
            "the number of scored cases in the slice"
            if volume == "cases"
            else f"the sum of {volume!r} over the scored cases of the slice"
        ),
        exposure=_row_value(row, "exposure"),
        mean_score=mean_score,
        reference_score=reference_score,
        signed_gap=signed_gap,
        gap=gap,
        gamma=float(gamma),
        rho=rho,
        stable_mean=float(row["stable_mean"]),
        stable_gap=float(row["stable_gap"]),
        priority_index=priority_index,
        stable_priority_index=stable_priority_index,
        clipped=signed_gap <= 0,
        se=_row_value(row, "se"),
        gap_lower=_row_value(row, "gap_lower"),
        priority_index_lower=_row_value(row, "PI_lower"),
    )


def _with_group_counts(
    priority: PriorityComponents, result: ScoreResult, keys: Sequence[str], key: Sequence[Any], *, rank: int | None, n_groups: int
) -> PriorityComponents:
    """Attach the slice's unit count — scored and unscored — and its rank."""
    n_units = int(_group_mask(result, keys, key).sum())
    return dataclasses.replace(priority, n_units=n_units, rank=rank, n_groups=n_groups)


def _layer_components(
    driver_row: pd.Series,
    layer_names: Sequence[str],
    *,
    resolved: ResolvedBaseline,
    priority: PriorityComponents,
    atol: float,
    complete: bool = True,
) -> tuple[LayerComponent, ...]:
    profile = resolved.reference_layer_penalties
    scale = priority.volume * priority.rho
    out: list[LayerComponent] = []
    for name in layer_names:
        group_penalty = float(driver_row[name])
        if profile is None:
            out.append(LayerComponent(name, group_penalty, None, None, None, None, None))
            continue
        delta = _number(driver_row[f"{name}__delta"])
        if delta is None:  # pragma: no cover - a compatible profile always yields a delta
            raise BaselineError(f"layer {name!r} has no reference delta although the comparator declares a profile")
        component = scale * delta
        clipped = 0.0 if priority.clipped else component
        share = None if priority.stable_priority_index <= 0 else component / priority.stable_priority_index
        out.append(LayerComponent(name, group_penalty, float(profile[name]), delta, component, clipped, share))
    if profile is not None and complete:
        _check(
            math.fsum(layer.delta or 0.0 for layer in out),
            priority.signed_gap,
            atol,
            "the layer deltas must sum to the signed score gap",
        )
        _check(
            math.fsum(layer.component or 0.0 for layer in out),
            scale * priority.signed_gap,
            atol,
            "the signed layer components must sum to volume x rho x the signed gap",
        )
        if not priority.clipped:
            _check(
                math.fsum(layer.clipped_component or 0.0 for layer in out),
                priority.stable_priority_index,
                atol,
                "the clipped layer components must sum to the stabilised priority index",
            )
    return tuple(out)


#: Record-scope qualifications that change what an *aggregate* over those
#: records means. Each is a statement about how a check was answered, so a mean
#: built on them is qualified in the same way — which is why they are lifted to
#: the slice rather than left in the evidence packet.
_AGGREGATE_RELEVANT_RECORD_CODES = (
    QualificationCode.LOWER_BOUND,
    QualificationCode.VACUOUS_SATISFACTION,
    QualificationCode.BOTH_TOTALS_ZERO,
)

_RECORD_CODE_CONSEQUENCE = {
    QualificationCode.LOWER_BOUND: (
        "their violations are bounds measured at the censoring horizon, so this slice's mean penalty, its gap "
        "and every index built on them are bounds too, not completed measurements"
    ),
    QualificationCode.VACUOUS_SATISFACTION: (
        "they are satisfied because the regulated activity never occurred, so the satisfaction they contribute "
        "to this slice's mean is vacuous rather than observed compliance"
    ),
    QualificationCode.BOTH_TOTALS_ZERO: (
        "the balance holds because nothing was recorded on either side, so the zero violation they contribute "
        "rests on an absence rather than on two matching totals"
    ),
}


def _slice_unit_ids(result: ScoreResult, view: str, keys: Sequence[str], key: Sequence[Any]) -> set[str]:
    """The scored units of the slice, as the evidence packet names them.

    The same population :func:`wise.constraint_drivers` aggregates over, so a
    share taken here has the denominator the constraint table shows.
    """
    mask = _group_mask(result, keys, key) & result.scores[view].notna()
    return {str(unit) for unit in result.unit_table.index[mask]}


def _records_in_slice(evidence: EvidencePacket | None, unit_ids: set[str]) -> tuple[EvaluationRecord, ...]:
    if evidence is None:
        return ()
    return tuple(record for record in evidence.records if record.unit_id in unit_ids)


def _record_qualification_counts(
    records: Sequence[EvaluationRecord],
) -> dict[QualificationCode, dict[str, int]]:
    """Per code, how many records of each constraint carry it inside the slice."""
    out: dict[QualificationCode, dict[str, int]] = {}
    for record in records:
        for qualification in record.qualifications:
            if qualification.code not in _AGGREGATE_RELEVANT_RECORD_CODES:
                continue
            per_constraint = out.setdefault(qualification.code, {})
            per_constraint[record.constraint_id] = per_constraint.get(record.constraint_id, 0) + 1
    return out


def _lower_bound_shares(records: Sequence[EvaluationRecord]) -> dict[str, float]:
    """Share of each constraint's *evaluated* records in the slice that are bounds."""
    evaluated: dict[str, int] = {}
    bounded: dict[str, int] = {}
    for record in records:
        if not record.evaluable:
            continue
        evaluated[record.constraint_id] = evaluated.get(record.constraint_id, 0) + 1
        if any(q.code is QualificationCode.LOWER_BOUND for q in record.qualifications):
            bounded[record.constraint_id] = bounded.get(record.constraint_id, 0) + 1
    return {cid: bounded.get(cid, 0) / n for cid, n in evaluated.items() if n}


def _constraint_components(
    result: ScoreResult,
    view: str,
    keys: Sequence[str],
    key: Sequence[Any],
    *,
    max_constraints: int,
    records: Sequence[EvaluationRecord] = (),
    evidence: EvidencePacket | None = None,
) -> tuple[ConstraintComponent, ...]:
    """Reuse :func:`wise.constraint_drivers` inside the slice, with denominators."""
    mask = _group_mask(result, keys, key)
    frame = constraint_drivers(result, view, mask)
    layer_of = result.norm.layer_of
    shares = _lower_bound_shares(records) if evidence is not None else {}
    rows: list[ConstraintComponent] = []
    for cid, row in frame.head(max_constraints).iterrows():
        rows.append(
            ConstraintComponent(
                constraint_id=str(cid),
                layer=str(layer_of[str(cid)]),
                constraint_type=str(row["type"]),
                mean_penalty=_number(row["mean_penalty"]),
                mean_violation=_number(row["mean_violation"]),
                share_violated=_number(row["share_violated"]),
                share_in_scope=_number(row["share_in_scope"]),
                share_evaluated=_number(row["share_evaluated"]),
                # None, not 0.0, when nothing was captured: unknown is not "none of them"
                share_lower_bound=None if evidence is None else _number(shares.get(str(cid), 0.0)),
                description=str(row["description"]),
            )
        )
    return tuple(rows)


def _view_contrasts(
    result: ScoreResult,
    keys: Sequence[str],
    key: Sequence[Any],
    *,
    view: str,
    views: Sequence[str] | None,
    gamma: float,
    volume: str,
    min_cases: int,
) -> tuple[tuple[ViewContrast, ...], tuple[str, ...]]:
    wanted = _wanted_views(result, views)
    here = result.scores[view].notna()
    out: list[ViewContrast] = []
    absent: list[str] = []
    for other in wanted:
        backlog = prioritize(result, keys, view=other, gamma=gamma, volume=volume, min_cases=min_cases)
        try:
            position = _locate(backlog, key)
        except LookupError:
            # the slice has no scored unit under this view: no row, and a
            # disclosure rather than a silently missing comparison
            absent.append(other)
            continue
        row = backlog.iloc[position]
        there = result.scores[other].notna()
        reference = float(backlog.attrs["baseline"])
        out.append(
            ViewContrast(
                view=other,
                comparator=f"current-population:{other}",
                n_scored=int(row["n_cases"]),
                n_scored_population=int(there.sum()),
                mean_score=float(row["mean_score"]),
                reference_score=reference,
                signed_gap=reference - float(row["mean_score"]),
                stable_priority_index=float(row["stable_PI"]),
                rank=position + 1,
                scored_population_differs=bool(not here.equals(there)),
            )
        )
    return tuple(out), tuple(absent)


def _wanted_views(result: ScoreResult, views: Sequence[str] | None) -> list[str]:
    wanted = list(views) if views is not None else list(result.views)
    unknown = [v for v in wanted if v not in result.views]
    if unknown:
        raise BaselineError(f"unknown view(s) {unknown}; available: {result.views}")
    return wanted


def _differing_populations(result: ScoreResult, view: str, views: Sequence[str] | None) -> tuple[str, ...]:
    """Views whose scored population is not the one this explanation used."""
    here = result.scores[view].notna()
    return tuple(
        other for other in _wanted_views(result, views) if other != view and not here.equals(result.scores[other].notna())
    )


def _witnesses_for_group(
    result: ScoreResult,
    view: str,
    keys: Sequence[str],
    key: Sequence[Any],
    evidence: EvidencePacket | None,
    *,
    limit: int,
) -> tuple[tuple[WitnessRef, ...], tuple[str, ...], tuple[int, int]]:
    """Bounded witnesses: the worst scored units of the slice, their worst check.

    Returns the witnesses, their evidence references, and ``(n_missing,
    n_requested)`` — how many of the units this explanation asked for had no
    record in the evidence packet. A bounded capture is the usual reason, and
    dropping those units silently would leave a packet that merely has fewer
    references with nothing saying why.
    """
    if evidence is None or limit <= 0:
        return (), (), (0, 0)
    mask = _group_mask(result, keys, key) & result.scores[view].notna()
    if not bool(mask.any()):  # pragma: no cover - a ranked slice has scored units
        return (), (), (0, 0)
    worst = result.scores[view][mask].sort_values(kind="mergesort").head(limit)
    penalties = result.penalties(view)
    witnesses: dict[str, WitnessRef] = {}
    refs: list[str] = []
    requested = 0
    missing = 0
    for unit in worst.index:
        row = penalties.loc[unit]
        constraint_id = str(row.idxmax())
        if not float(row.max()) > 0:
            continue
        requested += 1
        # addressed by what it is about, not by a formatted id: the case capture
        # and the object evaluation compose their evaluation ids in different
        # orders, and a packet holds at most one record per (unit, constraint)
        found = evidence.records_for(unit_id=unit, constraint_id=constraint_id)
        if not found:
            missing += 1  # a bounded capture did not keep this unit; say so, do not skip it
            continue
        record = found[0]
        refs.append(record.evaluation_id)
        for witness in record.witnesses[:2]:
            witnesses.setdefault(witness.witness_id, witness)
    return tuple(witnesses.values()), tuple(refs), (missing, requested)


# --------------------------------------------------------------- narrative
def _limitations(
    *,
    resolved: ResolvedBaseline,
    priority: PriorityComponents,
    key: Sequence[Any],
    rank: int | None,
    min_cases: int,
    volume: str,
    differing_views: Sequence[str],
    absent_views: Sequence[str],
    evidence: EvidencePacket | None,
    complete_partition: bool,
    record_codes: Mapping[QualificationCode, Mapping[str, int]] | None = None,
    missing_witnesses: tuple[int, int] = (0, 0),
    unit_type: str = "case",
) -> tuple[Qualification, ...]:
    out: list[Qualification] = []
    spec = resolved.spec
    if resolved.reference_layer_penalties is None:
        out.append(
            Qualification(
                QualificationCode.REFERENCE_PROFILE_UNAVAILABLE,
                f"comparator {spec.baseline_id!r} states a score but no layer profile, so no additive layer "
                "attribution against it is determined. The absolute layer penalties below are what is known; "
                "the current population's profile is not a substitute for the target's.",
                scope="explanation",
            )
        )
    if not complete_partition:
        out.append(
            Qualification(
                QualificationCode.REFERENCE_PROFILE_UNAVAILABLE,
                "only part of the layer partition is shown, so the layer contrasts below do not add up to the score "
                "gap. Drop the layers= restriction for a decomposition that reconciles.",
                scope="explanation",
            )
        )
    if priority.clipped:
        out.append(
            Qualification(
                QualificationCode.NON_POSITIVE_GAP_CLIPPED,
                f"the signed gap is {priority.signed_gap:+.6g}, so the priority index is clipped to zero. "
                "The unclipped layer contrasts are reported unchanged; no contribution share is formed, "
                "because dividing by a zero net gap would invent one.",
                scope="group",
            )
        )
    if not resolved.is_global:
        out.append(
            Qualification(
                QualificationCode.COMPARATOR_IS_NOT_GLOBAL,
                f"the reference {priority.reference_score:.6g} comes from comparator {spec.baseline_id!r} "
                f"({spec.kind.value}); the column named 'global_mean' in the backlog holds it, and it is not "
                f"the current population's mean, which is {resolved.population_mean:.6g}.",
                scope="explanation",
            )
        )
    if spec.reviewed_mapping:
        out.append(
            Qualification(
                QualificationCode.REVIEWED_MAPPING_APPLIED,
                "an explicit reviewed mapping authorised this comparison across a changed norm or calibration: "
                + "; ".join(resolved.compatibility.notes),
                scope="explanation",
            )
        )
    undeclared = [*resolved.compatibility.undeclared, *resolved.compatibility.unverifiable]
    if undeclared:
        out.append(
            Qualification(
                QualificationCode.COMPARATOR_SEMANTICS_UNDECLARED,
                "the comparison could not be verified in every respect: "
                + ", ".join(sorted(set(undeclared)))
                + " was not declared on both sides, so it was not compared.",
                scope="explanation",
            )
        )
    if rank is None and min_cases > 1:
        out.append(
            Qualification(
                QualificationCode.BELOW_MINIMUM_SUPPORT,
                f"this slice has {priority.n_scored} scored {'case' if priority.n_scored == 1 else 'cases'} and the "
                f"minimum support is {min_cases}, so it is explained but not ranked. The reference was computed over "
                "the whole scored population before that filter, exactly as in the backlog.",
                scope="group",
            )
        )
    if any(_isna(part) for part in key):
        out.append(
            Qualification(
                QualificationCode.NULL_GROUP_KEY,
                "part of this slice key is null. Null keys form their own slice; they are not dropped, and they are "
                "not merged with any other value.",
                scope="group",
            )
        )
    if priority.n_scored == 1:
        out.append(
            Qualification(
                QualificationCode.SINGLE_UNIT_GROUP,
                "the slice holds a single scored unit: its mean is that unit's score and its dispersion is undefined, "
                "so no standard error and no lower bound are available.",
                scope="group",
            )
        )
    if priority.n_units > priority.n_scored:
        out.append(
            Qualification(
                QualificationCode.UNSCORED_UNIT,
                f"{priority.n_units - priority.n_scored} of {priority.n_units} units in this slice have no score under "
                f"view {priority.view!r} and are excluded from every number above. No score is not a score of zero.",
                scope="group",
            )
        )
    if priority.gamma > 0:
        out.append(
            Qualification(
                QualificationCode.SHRINKAGE_STABILISED_NOT_STABLE,
                f"the index is shrinkage-stabilised with gamma = {priority.gamma:g}, in units of scored cases: "
                f"rho = n/(n+gamma) = {priority.rho:.6g}. Shrinkage borrows strength from the comparator; it is not a "
                "confidence statement and does not make the estimate stable.",
                scope="group",
            )
        )
    if volume != "cases":
        out.append(
            Qualification(
                QualificationCode.EXPOSURE_SCALES_PRIORITY,
                f"volume is {volume!r}: it scales the gap into the index and nothing else. The mean score above is the "
                "unweighted mean over scored units; it is not exposure-weighted.",
                scope="group",
            )
        )
    if differing_views or absent_views:
        detail = []
        if differing_views:
            detail.append(
                "the scored populations differ between views ("
                + ", ".join(sorted(differing_views))
                + f" versus {priority.view!r}), because a view can give zero weight to the only constraint that "
                "applies to a unit"
            )
        if absent_views:
            detail.append(
                "this slice has no scored unit at all under " + ", ".join(sorted(absent_views)) + ", so those views "
                "have no row below rather than a row of zeros"
            )
        out.append(
            Qualification(
                QualificationCode.DIFFERENT_SCORED_POPULATIONS,
                "; ".join(detail) + ". The per-view rows are therefore not a common-population comparison.",
                scope="explanation",
            )
        )
    # a record-scope qualification is a statement about how a check was
    # answered; an aggregate over those checks inherits it, so it is named here
    # with the constraints and the counts rather than left inside the packet
    for code in _AGGREGATE_RELEVANT_RECORD_CODES:
        per_constraint = (record_codes or {}).get(code)
        if not per_constraint:
            continue
        total = sum(per_constraint.values())
        named = ", ".join(f"{cid} ({n})" for cid, n in sorted(per_constraint.items()))
        out.append(
            Qualification(
                code,
                f"{total} evaluation{'' if total == 1 else 's'} of this slice carry {code.value!r} — {named}: "
                + _RECORD_CODE_CONSEQUENCE[code]
                + ".",
                scope="group",
            )
        )
    n_missing, n_requested = missing_witnesses
    if n_missing:
        out.append(
            Qualification(
                QualificationCode.RECORDS_TRUNCATED,
                f"{n_missing} of the {n_requested} {unit_type}s selected for witnesses have no record in the "
                "evidence packet, because the capture that produced it was bounded. Their witnesses are absent, "
                "not absent evidence: the packet below simply shows fewer references than the slice would support.",
                scope="group",
            )
        )
    out.append(
        Qualification(
            QualificationCode.PRIORITY_IS_NOT_A_CAUSE,
            "a large layer component is where the assessed gap sits, not a demonstrated cause of it and not an "
            "avoidable amount of money.",
            scope="explanation",
        )
    )
    if evidence is not None:
        out.extend(q for q in evidence.qualifications if q.scope == "run")
    return tuple(out)


def _denominators(
    priority: PriorityComponents, resolved: ResolvedBaseline, *, volume: str, evidence: EvidencePacket | None
) -> tuple[Denominator, ...]:
    unit = resolved.spec.unit_type
    out = [
        Denominator(
            "scored_units",
            f"{unit}s of this slice with a score under view {priority.view!r}; the denominator of the slice mean "
            "and the support n in the shrinkage factor",
            priority.n_scored,
            unit,
        ),
        Denominator(
            "slice_units",
            f"{unit}s of this slice, scored or not; the difference is excluded from every mean above",
            priority.n_units,
            unit,
        ),
        Denominator(
            "reference_population",
            f"scored {unit}s the reference was computed over, before the minimum-support filter",
            resolved.population_size,
            unit,
        ),
        Denominator(
            "volume",
            "the scale factor of the priority index: " + priority.volume_definition,
            priority.volume,
            unit if volume == "cases" else volume,
        ),
    ]
    if priority.n_groups is not None:
        out.append(Denominator("ranked_slices", "slices that passed the minimum-support filter", priority.n_groups, "slices"))
    if evidence is not None:
        coverage = evidence.coverage
        out.append(
            Denominator(
                "in_scope_checks",
                "constraint x unit checks in scope in the captured evidence; the denominator of the evaluated share",
                coverage.n_in_scope,
                "checks",
            )
        )
    return tuple(out)


def _fact(
    kind: FactKind,
    name: str,
    value: Any,
    unit: str,
    description: str,
    refs: Sequence[str],
    *,
    denominator: str | None = None,
    method: str = "",
) -> Fact:
    return Fact(
        fact_id=f"{_FACT_PREFIX[kind]}-{safe_identifier(name)}",
        kind=kind,
        name=name,
        value=value if not isinstance(value, float) else _number(value),
        unit=unit,
        description=description,
        evidence_refs=tuple(refs),
        denominator=denominator,
        method=method,
    )


def _facts(
    *,
    priority: PriorityComponents,
    layers: Sequence[LayerComponent],
    resolved: ResolvedBaseline,
    contrasts: Sequence[ViewContrast],
    evidence: EvidencePacket | None,
    refs: Sequence[str],
    view: str,
    volume: str,
    unit_type: str,
) -> tuple[Fact, ...]:
    obs, pri, alt, qual = (
        FactKind.OBSERVED_ASSESSMENT,
        FactKind.RELATIVE_PRIORITY,
        FactKind.ALTERNATIVE_VIEW,
        FactKind.EVIDENCE_QUALIFICATION,
    )
    out = [
        _fact(
            obs,
            "mean_score",
            priority.mean_score,
            "score",
            f"assessed mean score of the slice under view {view!r}",
            refs,
            denominator="scored_units",
            method="unweighted mean over the scored units",
        ),
        _fact(
            obs,
            "assessed_penalty",
            1.0 - priority.mean_score,
            "score",
            "1 - mean score: the assessed penalty of the slice",
            refs,
            denominator="scored_units",
            method="1 - mean score",
        ),
        _fact(obs, "n_scored", priority.n_scored, unit_type, "scored units of the slice", refs, denominator="scored_units"),
        _fact(obs, "n_units", priority.n_units, unit_type, "units of the slice, scored or not", refs, denominator="slice_units"),
        _fact(
            obs,
            "volume",
            priority.volume,
            unit_type if volume == "cases" else volume,
            priority.volume_definition,
            refs,
            denominator="volume",
        ),
    ]
    if priority.exposure is not None:
        out.append(
            _fact(
                obs,
                "exposure",
                priority.exposure,
                "exposure",
                "summed exposure of the slice; it scales the index, it does not weight the mean",
                refs,
                denominator="scored_units",
            )
        )
    for layer in layers:
        out.append(
            _fact(
                obs,
                f"penalty.{layer.layer}",
                layer.group_penalty,
                "score",
                f"mean penalty the slice carries in layer {layer.layer!r}",
                refs,
                denominator="scored_units",
                method="mean of the per-unit effective-weighted layer penalties",
            )
        )
    out.extend(
        [
            _fact(
                pri,
                "reference_score",
                priority.reference_score,
                "score",
                f"the comparator: {resolved.spec.baseline_id!r} ({resolved.spec.kind.value})",
                refs,
                denominator="reference_population",
                method=resolved.resolved_from,
            ),
            _fact(
                pri,
                "population_mean",
                resolved.population_mean,
                "score",
                "the current scored population's own mean, shown whether or not it is the comparator",
                refs,
                denominator="reference_population",
            ),
            _fact(
                pri,
                "signed_gap",
                priority.signed_gap,
                "score",
                "reference score - slice mean score; negative means the slice is above the comparator",
                refs,
                method="reference_score - mean_score",
            ),
            _fact(
                pri,
                "gap",
                priority.gap,
                "score",
                "the signed gap clipped at zero, as used by the index",
                refs,
                method="max(signed_gap, 0)",
            ),
            _fact(
                pri,
                "gamma",
                priority.gamma,
                unit_type,
                f"shrinkage constant, in the same unit the support is counted in (scored {unit_type}s)",
                refs,
            ),
            _fact(
                pri,
                "rho",
                priority.rho,
                "ratio",
                "shrinkage factor n/(n+gamma)",
                refs,
                denominator="scored_units",
                method="n_scored / (n_scored + gamma)",
            ),
            _fact(
                pri,
                "PI",
                priority.priority_index,
                f"score x {unit_type}s",
                "priority index: volume x clipped gap",
                refs,
                denominator="volume",
                method="volume * max(signed_gap, 0)",
            ),
            _fact(
                pri,
                "stable_PI",
                priority.stable_priority_index,
                f"score x {unit_type}s",
                "shrinkage-stabilised priority index: volume x rho x clipped gap",
                refs,
                denominator="volume",
                method="volume * rho * max(signed_gap, 0)",
            ),
        ]
    )
    if priority.rank is not None:
        out.append(
            _fact(
                pri,
                "rank",
                priority.rank,
                "position",
                "position in the backlog ranked by the stabilised index",
                refs,
                denominator="ranked_slices",
            )
        )
    for layer in layers:
        out.append(
            _fact(
                pri,
                f"delta.{layer.layer}",
                layer.delta,
                "score",
                f"slice penalty minus comparator penalty in layer {layer.layer!r}; null when the comparator has no profile",
                refs,
                denominator="scored_units",
                method="mean_group_layer_penalty - reference_layer_penalty",
            )
        )
        out.append(
            _fact(
                pri,
                f"component.{layer.layer}",
                layer.component,
                f"score x {unit_type}s",
                f"signed priority component of layer {layer.layer!r}",
                refs,
                denominator="volume",
                method="volume * rho * delta_layer",
            )
        )
        out.append(
            _fact(
                pri,
                f"clipped_component.{layer.layer}",
                layer.clipped_component,
                f"score x {unit_type}s",
                "the same component after clipping; zero for a non-positive net gap",
                refs,
                denominator="volume",
            )
        )
    for contrast in contrasts:
        out.append(
            _fact(
                alt,
                f"{contrast.view}.mean_score",
                contrast.mean_score,
                "score",
                f"the same slice's mean score under view {contrast.view!r}",
                refs,
                denominator="scored_units",
            )
        )
        out.append(
            _fact(
                alt,
                f"{contrast.view}.stable_PI",
                contrast.stable_priority_index,
                f"score x {unit_type}s",
                f"its stabilised index under view {contrast.view!r}, against that view's own current population",
                refs,
                denominator="volume",
                method=contrast.comparator,
            )
        )
        out.append(
            _fact(
                alt,
                f"{contrast.view}.n_scored",
                contrast.n_scored,
                unit_type,
                f"scored units of the slice under view {contrast.view!r}",
                refs,
                denominator="scored_units",
            )
        )
    if evidence is not None:
        coverage = evidence.coverage
        out.append(
            _fact(
                qual,
                "evaluated_checks",
                coverage.n_evaluated,
                "checks",
                "checks actually evaluated in the captured evidence of this run",
                refs,
                denominator="in_scope_checks",
            )
        )
        out.append(
            _fact(
                qual,
                "unevaluable_checks",
                coverage.n_unevaluable,
                "checks",
                "in-scope checks that could not be evaluated; each carries a reason code",
                refs,
                denominator="in_scope_checks",
            )
        )
        out.append(
            _fact(
                qual,
                "evaluated_share",
                coverage.evaluated_share,
                "ratio",
                "evaluated over in-scope checks; coverage, never a confidence",
                refs,
                denominator="in_scope_checks",
            )
        )
    return tuple(out)
