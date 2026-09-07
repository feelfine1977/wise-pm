"""Typed observations, witnesses, qualifications and packets.

The unit of evidence is **one constraint evaluated on one declared assessment
unit**. :class:`EvaluationRecord` answers three different questions that the
violation matrix answers with one ``NaN``:

* **in scope?** — is the expectation relevant to this unit at all;
* **evaluable?** — were the observations the check needs available under the
  declared missingness policy;
* **satisfied or violated?** — what did the check return.

Only view-independent facts belong in a record. Effective weights and
penalties depend on the chosen view and are separate :class:`ViewAnnotation`
rows, so adding a view never produces a second, contradictory copy of the
observed evidence.

Measurements stay structured. A lag keeps its activation timestamp, its
response timestamp and the duration between them, with units; a balance keeps
both totals, the relative mismatch and the unit each total is measured in.
There is deliberately no single unexplained ``raw_value`` in this contract —
the interchange export in :mod:`wise.evidence.capture` has one, and says so.

Absence has no event identity. A rule violated *because nothing happened* is
supported by an :class:`AbsenceSearch` witness that declares the unit, the
window, the filters in force and the completeness assumption behind the
search, never by a fabricated event id.

This module imports no scoring, no log and no optional adapter, so it can be
used to read exported evidence on its own.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from ..errors import EvidenceError
from .manifest import RunManifest, _canonical, only_fields

if TYPE_CHECKING:  # pragma: no cover
    from ..log import LogSnapshot
    from ..norm import Norm

PACKET_SCHEMA_VERSION = "wise-evidence/1"


class ReasonCode(str, Enum):
    """Why a check has — or has not — a violation value.

    The five *evaluable* codes carry a violation; the others do not. A missing
    endpoint under a ``violate`` policy is an evaluated, policy-based violation
    (``POLICY_VIOLATION_MISSING_ACTIVATION`` / ``..._RESPONSE``), not an
    unevaluable check; a ``censor`` policy yields ``OPEN_OBSERVATION_WINDOW``,
    which is evaluated but only as a lower bound.

    >>> ReasonCode.OUT_OF_SCOPE.evaluable
    False
    >>> ReasonCode.OPEN_OBSERVATION_WINDOW.evaluable
    True
    """

    OBSERVED = "observed"
    SATISFIED_VACUOUSLY = "satisfied_vacuously"
    POLICY_VIOLATION_MISSING_ACTIVATION = "policy_violation_missing_activation"
    POLICY_VIOLATION_MISSING_RESPONSE = "policy_violation_missing_response"
    OPEN_OBSERVATION_WINDOW = "open_observation_window"

    OUT_OF_SCOPE = "out_of_scope"
    SKIPPED_MISSING_ACTIVATION = "skipped_missing_activation"
    SKIPPED_MISSING_RESPONSE = "skipped_missing_response"
    SKIPPED_MISSING_ANCHOR = "skipped_missing_anchor"
    MISSING_ATTRIBUTE = "missing_attribute"
    AMBIGUOUS_MATCH = "ambiguous_match"
    MISSING_SOURCE_IDENTITY = "missing_source_identity"
    BUDGET_TRUNCATED = "budget_truncated"
    #: A shortfall was measured but the scope was never declared complete, so
    #: "we did not see it" cannot be reported as "it is not there"
    #: (:mod:`wise.oc.constraints`).
    UNVERIFIED_ABSENCE = "unverified_absence"
    #: Amounts in different currencies or units with no declared conversion
    #: rule; adding them would invent an exchange rate.
    INCOMPATIBLE_UNITS = "incompatible_units"

    @property
    def evaluable(self) -> bool:
        """Whether a record with this reason carries a violation value."""
        return self in EVALUABLE_REASONS


#: Reason codes that come with a violation in ``[0, 1]``.
EVALUABLE_REASONS = frozenset(
    {
        ReasonCode.OBSERVED,
        ReasonCode.SATISFIED_VACUOUSLY,
        ReasonCode.POLICY_VIOLATION_MISSING_ACTIVATION,
        ReasonCode.POLICY_VIOLATION_MISSING_RESPONSE,
        ReasonCode.OPEN_OBSERVATION_WINDOW,
    }
)


class WitnessKind(str, Enum):
    """What a witness points at."""

    EVENT = "event"
    ABSENCE = "absence"
    ATTRIBUTE = "attribute"
    AGGREGATE = "aggregate"


class SourceIdentity(str, Enum):
    """How far the identity of a witness is actually established."""

    #: A source system event id, supplied through ``EventLog(event_id_col=...)``.
    SOURCE = "source_event_id"
    #: A row position in one snapshot. Not a source identity, and labelled as such.
    SNAPSHOT_LOCAL = "snapshot_local_row"
    #: No identity at all — an absence, an attribute or an aggregate.
    NONE = "none"


class Completeness(str, Enum):
    """The assumption a search over a window rests on."""

    #: Every relevant event of this unit is in the snapshot for this window.
    ASSUMED_COMPLETE = "assumed_complete"
    #: The window is still open, so later events may still arrive.
    OPEN = "open"
    #: A filter could have removed relevant events; absence proves nothing.
    UNKNOWN = "unknown"


class MeasurementKind(str, Enum):
    """The dimension of a measurement, so a unit can be checked against it."""

    COUNT = "count"
    DURATION = "duration"
    AMOUNT = "amount"
    RATIO = "ratio"
    TIMESTAMP = "timestamp"
    THRESHOLD = "threshold"
    INDEX = "index"


class QualificationCode(str, Enum):
    """Named limitations attached to a record, a unit or a whole run."""

    LOWER_BOUND = "lower_bound"
    CENSORED_HORIZON = "censored_horizon"
    VACUOUS_SATISFACTION = "vacuous_satisfaction"
    AVERAGED_OVER_ACTIVATIONS = "averaged_over_activations"
    BOTH_TOTALS_ZERO = "both_totals_zero"
    WITNESSES_TRUNCATED = "witnesses_truncated"
    RECORDS_TRUNCATED = "records_truncated"
    EVALUATION_TRUNCATED = "evaluation_truncated"
    NO_SOURCE_EVENT_IDENTITY = "no_source_event_identity"
    ABSENCE_COMPLETENESS_UNKNOWN = "absence_completeness_unknown"
    UNSCORED_UNIT = "unscored_unit"
    ZERO_DENOMINATOR = "zero_denominator"
    COVERAGE_IS_NOT_CONFIDENCE = "coverage_is_not_confidence"
    INPUT_MUTATED_BY_DERIVE = "input_mutated_by_derive"
    SNAPSHOT_NOT_CONTENT_HASHED = "snapshot_not_content_hashed"

    # comparator and explanation limitations (:mod:`wise.explain`)
    REFERENCE_PROFILE_UNAVAILABLE = "reference_profile_unavailable"
    NON_POSITIVE_GAP_CLIPPED = "non_positive_gap_clipped"
    COMPARATOR_IS_NOT_GLOBAL = "comparator_is_not_global"
    COMPARATOR_SEMANTICS_UNDECLARED = "comparator_semantics_undeclared"
    DIFFERENT_SCORED_POPULATIONS = "different_scored_populations"
    BELOW_MINIMUM_SUPPORT = "below_minimum_support"
    NULL_GROUP_KEY = "null_group_key"
    SINGLE_UNIT_GROUP = "single_unit_group"
    SHRINKAGE_STABILISED_NOT_STABLE = "shrinkage_stabilised_not_stable"
    EXPOSURE_SCALES_PRIORITY = "exposure_scales_priority"
    PRIORITY_IS_NOT_A_CAUSE = "priority_is_not_a_cause"
    REVIEWED_MAPPING_APPLIED = "reviewed_mapping_applied"

    # object-centric limitations (:mod:`wise.oc`)
    AMBIGUOUS_MATCH_RESOLVED = "ambiguous_match_resolved"
    COUNT_IS_A_LOWER_BOUND = "count_is_a_lower_bound"
    UNIT_CONVERSION_APPLIED = "unit_conversion_applied"
    ALLOCATION_INCOMPLETE = "allocation_incomplete"
    OVERLAPPING_GROUPS_NOT_SUMMED = "overlapping_groups_not_summed"
    HETEROGENEOUS_UNIT_TYPES = "heterogeneous_unit_types"
    CONTEXT_TRUNCATED = "context_truncated"
    DEPTH_LIMIT_REACHED = "depth_limit_reached"
    FAN_OUT_LIMIT_REACHED = "fan_out_limit_reached"
    BINDING_LIMIT_REACHED = "binding_limit_reached"
    ATTRIBUTE_NOT_SET_AT_TIME = "attribute_not_set_at_time"
    ATTRIBUTE_READ_OUTSIDE_EVALUATION_TIME = "attribute_read_outside_evaluation_time"
    RELATION_VALIDITY_UNKNOWN = "relation_validity_unknown"


def _finite(value: Any, what: str) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{what} must be a number or null, got {value!r}") from exc
    if not math.isfinite(out):
        raise EvidenceError(f"{what} must be finite or null, got {value!r}; NaN is a status, not a value")
    return out


def _text(value: Any, what: str) -> str:
    out = str(value)
    if not out:
        raise EvidenceError(f"{what} must be a non-empty string")
    return out


# ------------------------------------------------------------------ measurements
@dataclass(frozen=True)
class Measurement:
    """One named, dimensioned observation behind a check.

    >>> Measurement("lag", 25.0, "D", MeasurementKind.DURATION).to_dict()["unit"]
    'D'
    """

    name: str
    value: float | str | None
    unit: str
    kind: MeasurementKind
    lower_bound: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "measurement name"))
        object.__setattr__(self, "unit", _text(self.unit, "measurement unit"))
        object.__setattr__(self, "kind", MeasurementKind(self.kind))
        if self.kind is MeasurementKind.TIMESTAMP:
            if self.value is not None and not isinstance(self.value, str):
                raise EvidenceError(f"measurement {self.name!r}: a timestamp value must be an ISO-8601 string or null")
        else:
            object.__setattr__(self, "value", _finite(self.value, f"measurement {self.name!r}"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "kind": self.kind.value,
            "lower_bound": bool(self.lower_bound),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Measurement:
        """Rebuild from :meth:`to_dict`, keeping ``lower_bound`` and the unit."""
        return cls(**only_fields(cls, data))


@dataclass(frozen=True)
class AbsenceSearch:
    """The declared search behind an absence: what was looked for, where, under what assumption."""

    unit_id: str
    activities: tuple[str, ...]
    window_start: str | None
    window_end: str | None
    completeness: Completeness
    n_events_searched: int = 0
    filters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit_id", _text(self.unit_id, "absence unit_id"))
        object.__setattr__(self, "activities", tuple(str(a) for a in self.activities))
        if not self.activities:
            raise EvidenceError("an absence search must declare what it searched for")
        object.__setattr__(self, "completeness", Completeness(self.completeness))
        object.__setattr__(self, "filters", dict(self.filters))
        object.__setattr__(self, "n_events_searched", int(self.n_events_searched))

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "activities": list(self.activities),
            "window_start": self.window_start,
            "window_end": self.window_end,
            "completeness": self.completeness.value,
            "n_events_searched": self.n_events_searched,
            "filters": dict(self.filters),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AbsenceSearch:
        """Rebuild from :meth:`to_dict`."""
        payload = only_fields(cls, data)
        payload["activities"] = tuple(str(a) for a in payload.get("activities", ()))
        return cls(**payload)


@dataclass(frozen=True)
class WitnessRef:
    """A reference to what supports a record: an event, an absence, an attribute.

    An event witness carries a source event id when the log declared one, and
    otherwise a snapshot-local row reference, with :attr:`identity` saying
    which of the two it is. An absence witness carries no event reference at
    all — only the search that established it.
    """

    kind: WitnessKind
    role: str
    unit_id: str
    identity: SourceIdentity = SourceIdentity.NONE
    event_id: str | None = None
    reference: str | None = None
    activity: str | None = None
    timestamp: str | None = None
    attribute: str | None = None
    search: AbsenceSearch | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", WitnessKind(self.kind))
        object.__setattr__(self, "identity", SourceIdentity(self.identity))
        object.__setattr__(self, "role", _text(self.role, "witness role"))
        object.__setattr__(self, "unit_id", _text(self.unit_id, "witness unit_id"))
        if self.kind is WitnessKind.EVENT:
            if (self.event_id is None) == (self.reference is None):
                raise EvidenceError("an event witness needs exactly one of event_id or reference")
            if self.event_id is not None and self.identity is not SourceIdentity.SOURCE:
                raise EvidenceError("a witness with a source event id must declare identity=SOURCE")
            if self.reference is not None and self.identity is not SourceIdentity.SNAPSHOT_LOCAL:
                raise EvidenceError("a snapshot-local row reference must declare identity=SNAPSHOT_LOCAL")
            if self.search is not None:
                raise EvidenceError("an event witness does not carry an absence search")
        elif self.kind is WitnessKind.ABSENCE:
            if self.search is None:
                raise EvidenceError("an absence witness must declare the search it rests on")
            if self.event_id is not None or self.reference is not None:
                raise EvidenceError("absence has no event identity; an absence witness must not reference an event")
            if self.identity is not SourceIdentity.NONE:
                raise EvidenceError("an absence witness has no source identity")
        elif self.kind is WitnessKind.ATTRIBUTE and not self.attribute:
            raise EvidenceError("an attribute witness must name the attribute")

    @property
    def witness_id(self) -> str:
        """A stable id for this witness inside its packet."""
        if self.kind is WitnessKind.EVENT:
            return str(self.event_id if self.event_id is not None else self.reference)
        if self.kind is WitnessKind.ABSENCE:
            return f"absence:{self.unit_id}:{'|'.join(self.search.activities) if self.search else ''}"
        if self.kind is WitnessKind.ATTRIBUTE:
            return f"attribute:{self.unit_id}:{self.attribute}"
        return f"aggregate:{self.unit_id}:{self.role}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "witness_id": self.witness_id,
            "kind": self.kind.value,
            "role": self.role,
            "unit_id": self.unit_id,
            "identity": self.identity.value,
            "event_id": self.event_id,
            "reference": self.reference,
            "activity": self.activity,
            "timestamp": self.timestamp,
            "attribute": self.attribute,
            "search": None if self.search is None else self.search.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WitnessRef:
        """Rebuild from :meth:`to_dict`; ``witness_id`` is derived, not read."""
        payload = only_fields(cls, data)
        search = data.get("search")
        payload["search"] = None if search is None else AbsenceSearch.from_dict(search)
        return cls(**payload)


#: What a qualification is about. ``"group"`` and ``"explanation"`` are the
#: aggregate scopes used by :mod:`wise.explain`: a limitation of one slice's
#: comparison, and a limitation of the whole explanation.
QUALIFICATION_SCOPES = ("evaluation", "unit", "run", "group", "explanation")


@dataclass(frozen=True)
class Qualification:
    """A named limitation. Never a probability, never a score multiplier."""

    code: QualificationCode
    message: str
    scope: str = "evaluation"

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", QualificationCode(self.code))
        object.__setattr__(self, "message", _text(self.message, "qualification message"))
        if self.scope not in QUALIFICATION_SCOPES:
            raise EvidenceError(f"qualification scope must be one of {QUALIFICATION_SCOPES}, got {self.scope!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code.value, "message": self.message, "scope": self.scope}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Qualification:
        """Rebuild from :meth:`to_dict`; an unknown code is refused, not dropped."""
        return cls(**only_fields(cls, data))


# ------------------------------------------------------------------ evaluations
@dataclass(frozen=True)
class EvaluationRecord:
    """One constraint on one declared assessment unit.

    Invariants, checked on construction:

    * out of scope implies not evaluable and no violation;
    * not evaluable implies no violation — a missing observation is never a
      zero penalty;
    * evaluable implies a violation in ``[0, 1]``;
    * the reason code agrees with :attr:`evaluable`.
    """

    run_id: str
    evaluation_id: str
    unit_id: str
    unit_type: str
    constraint_id: str
    constraint_type: str
    in_scope: bool
    evaluable: bool
    reason_code: ReasonCode
    violation: float | None
    constraint_version: str = ""
    measurements: tuple[Measurement, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    policies: dict[str, Any] = field(default_factory=dict)
    witnesses: tuple[WitnessRef, ...] = ()
    qualifications: tuple[Qualification, ...] = ()
    witnesses_materialised: bool = False
    n_witnesses_total: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_code", ReasonCode(self.reason_code))
        object.__setattr__(self, "in_scope", bool(self.in_scope))
        object.__setattr__(self, "evaluable", bool(self.evaluable))
        object.__setattr__(self, "measurements", tuple(self.measurements))
        object.__setattr__(self, "witnesses", tuple(self.witnesses))
        object.__setattr__(self, "qualifications", tuple(self.qualifications))
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "policies", dict(self.policies))
        for name in ("run_id", "evaluation_id", "unit_id", "unit_type", "constraint_id", "constraint_type"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if not self.in_scope and self.evaluable:
            raise EvidenceError(f"{self.evaluation_id}: an out-of-scope check cannot be evaluable")
        if self.reason_code.evaluable != self.evaluable:
            raise EvidenceError(
                f"{self.evaluation_id}: reason {self.reason_code.value!r} and evaluable={self.evaluable} disagree"
            )
        if not self.in_scope and self.reason_code is not ReasonCode.OUT_OF_SCOPE:
            raise EvidenceError(f"{self.evaluation_id}: an out-of-scope check must carry the out_of_scope reason")
        if self.evaluable:
            value = _finite(self.violation, f"{self.evaluation_id}: violation")
            if value is None or not (0.0 <= value <= 1.0):
                raise EvidenceError(f"{self.evaluation_id}: an evaluated check needs a violation in [0, 1], got {value!r}")
            object.__setattr__(self, "violation", value)
        elif self.violation is not None:
            raise EvidenceError(
                f"{self.evaluation_id}: a check that was not evaluated must have violation=None, "
                f"got {self.violation!r} — a missing observation is not a zero penalty"
            )
        names = [m.name for m in self.measurements]
        if len(set(names)) != len(names):
            raise EvidenceError(f"{self.evaluation_id}: duplicate measurement names {sorted(names)}")
        if self.n_witnesses_total is not None and self.n_witnesses_total < len(self.witnesses):
            raise EvidenceError(f"{self.evaluation_id}: n_witnesses_total is smaller than the witnesses kept")

    @property
    def measurement(self) -> dict[str, Measurement]:
        """The measurements by name."""
        return {m.name: m for m in self.measurements}

    @property
    def primary_measurement(self) -> Measurement | None:
        """The first measurement, used where a single value is unavoidable."""
        return self.measurements[0] if self.measurements else None

    @property
    def witnesses_truncated(self) -> bool:
        return self.n_witnesses_total is not None and self.n_witnesses_total > len(self.witnesses)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "evaluation_id": self.evaluation_id,
            "unit_id": self.unit_id,
            "unit_type": self.unit_type,
            "constraint_id": self.constraint_id,
            "constraint_version": self.constraint_version,
            "constraint_type": self.constraint_type,
            "in_scope": self.in_scope,
            "evaluable": self.evaluable,
            "reason_code": self.reason_code.value,
            "violation": self.violation,
            "measurements": [m.to_dict() for m in self.measurements],
            "parameters": dict(self.parameters),
            "policies": dict(self.policies),
            "witnesses": [w.to_dict() for w in self.witnesses],
            "witnesses_materialised": bool(self.witnesses_materialised),
            "n_witnesses_total": self.n_witnesses_total,
            "witnesses_truncated": self.witnesses_truncated,
            "qualifications": [q.to_dict() for q in self.qualifications],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvaluationRecord:
        """Rebuild one record from :meth:`to_dict`.

        Every invariant is re-checked on construction, so a file claiming an
        unevaluated check with a violation is refused on read exactly as it
        would be on capture. ``witnesses_truncated`` is derived and dropped.
        """
        payload = only_fields(cls, data)
        payload["measurements"] = tuple(Measurement.from_dict(m) for m in data.get("measurements", ()))
        payload["witnesses"] = tuple(WitnessRef.from_dict(w) for w in data.get("witnesses", ()))
        payload["qualifications"] = tuple(Qualification.from_dict(q) for q in data.get("qualifications", ()))
        return cls(**payload)


@dataclass(frozen=True)
class ViewAnnotation:
    """What one view makes of one record. Carries no observation of its own."""

    run_id: str
    evaluation_id: str
    unit_id: str
    constraint_id: str
    view: str
    effective_weight: float | None
    penalty: float | None
    scored: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "effective_weight", _finite(self.effective_weight, "effective_weight"))
        object.__setattr__(self, "penalty", _finite(self.penalty, "penalty"))
        object.__setattr__(self, "scored", bool(self.scored))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "evaluation_id": self.evaluation_id,
            "unit_id": self.unit_id,
            "constraint_id": self.constraint_id,
            "view": self.view,
            "effective_weight": self.effective_weight,
            "penalty": self.penalty,
            "scored": self.scored,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ViewAnnotation:
        """Rebuild from :meth:`to_dict`."""
        return cls(**only_fields(cls, data))


# ------------------------------------------------------------------ diagnostics
@dataclass(frozen=True)
class DiagnosticResult:
    """A diagnostic with its denominator stated, not just its number.

    ``value`` is ``None`` when the denominator is zero: a share of nothing is
    undefined, not zero.
    """

    name: str
    policy: str
    numerator: float
    denominator: float
    unit_of_counting: str
    scope: str
    interpretation: str
    threshold: float | None = None
    source_identity: SourceIdentity = SourceIdentity.NONE
    qualifications: tuple[Qualification, ...] = ()

    def __post_init__(self) -> None:
        for name in ("name", "policy", "unit_of_counting", "scope", "interpretation"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("numerator", "denominator"):
            number = _finite(getattr(self, name), name)
            if number is None:
                raise EvidenceError(f"a diagnostic needs a {name}; that is the point of the type")
            object.__setattr__(self, name, number)
        if self.denominator < 0:
            raise EvidenceError("a diagnostic denominator cannot be negative")
        object.__setattr__(self, "threshold", _finite(self.threshold, "threshold"))
        object.__setattr__(self, "source_identity", SourceIdentity(self.source_identity))
        quals = list(self.qualifications)
        if self.denominator == 0 and not any(q.code is QualificationCode.ZERO_DENOMINATOR for q in quals):
            quals.append(
                Qualification(
                    QualificationCode.ZERO_DENOMINATOR,
                    f"{self.name}: no {self.unit_of_counting} in scope, so the share is undefined rather than zero",
                    scope="run",
                )
            )
        object.__setattr__(self, "qualifications", tuple(quals))

    @property
    def value(self) -> float | None:
        """Numerator over denominator, or ``None`` when the denominator is zero."""
        if self.denominator == 0:
            return None
        return float(self.numerator) / float(self.denominator)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "policy": self.policy,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
            "unit_of_counting": self.unit_of_counting,
            "scope": self.scope,
            "threshold": self.threshold,
            "interpretation": self.interpretation,
            "source_identity": self.source_identity.value,
            "qualifications": [q.to_dict() for q in self.qualifications],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DiagnosticResult:
        """Rebuild from :meth:`to_dict`; ``value`` is derived from the two counts."""
        payload = only_fields(cls, data)
        payload["qualifications"] = tuple(Qualification.from_dict(q) for q in data.get("qualifications", ()))
        return cls(**payload)


COVERAGE_IS_NOT_CONFIDENCE = (
    "Coverage says how much was observed, not how likely the result is correct. "
    "It is not a probability and must never multiply a score or a priority index."
)


@dataclass(frozen=True)
class CoverageReport:
    """In-scope, evaluated, scored and excluded counts, with their denominators."""

    unit_type: str
    n_units: int
    n_checks: int
    n_in_scope: int
    n_evaluated: int
    n_out_of_scope: int
    n_unevaluable: int
    n_scored: dict[str, int] = field(default_factory=dict)
    n_unscored: dict[str, int] = field(default_factory=dict)
    reasons: dict[str, int] = field(default_factory=dict)
    interpretation: str = COVERAGE_IS_NOT_CONFIDENCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_scored", {str(k): int(v) for k, v in self.n_scored.items()})
        object.__setattr__(self, "n_unscored", {str(k): int(v) for k, v in self.n_unscored.items()})
        object.__setattr__(self, "reasons", {str(k): int(v) for k, v in self.reasons.items()})
        if self.n_in_scope + self.n_out_of_scope != self.n_checks:
            raise EvidenceError("coverage: in-scope and out-of-scope checks must add up to the checks performed")
        if self.n_evaluated + self.n_unevaluable != self.n_in_scope:
            raise EvidenceError("coverage: evaluated and unevaluable checks must add up to the in-scope checks")

    @property
    def evaluated_share(self) -> float | None:
        """Evaluated checks over in-scope checks; ``None`` when nothing is in scope."""
        return None if self.n_in_scope == 0 else self.n_evaluated / self.n_in_scope

    @property
    def scope_share(self) -> float | None:
        return None if self.n_checks == 0 else self.n_in_scope / self.n_checks

    @property
    def n_excluded(self) -> dict[str, int]:
        """Units excluded from each view's aggregates.

        A unit is excluded from an aggregate exactly when it has no score under
        that view, so this is :attr:`n_unscored` under its other name. It is
        exposed separately because "excluded from the backlog" and "unscored"
        are the same fact seen from the two ends of the pipeline, and because
        neither is a score of zero.
        """
        return dict(self.n_unscored)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_type": self.unit_type,
            "n_units": int(self.n_units),
            "n_checks": int(self.n_checks),
            "n_in_scope": int(self.n_in_scope),
            "n_evaluated": int(self.n_evaluated),
            "n_out_of_scope": int(self.n_out_of_scope),
            "n_unevaluable": int(self.n_unevaluable),
            "evaluated_share": self.evaluated_share,
            "scope_share": self.scope_share,
            "n_scored": dict(self.n_scored),
            "n_unscored": dict(self.n_unscored),
            "n_excluded": self.n_excluded,
            "reasons": dict(self.reasons),
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CoverageReport:
        """Rebuild from :meth:`to_dict`; the shares and ``n_excluded`` are derived."""
        return cls(**only_fields(cls, data))


@dataclass(frozen=True)
class Truncation:
    """What a bounded capture left out — and whether the evaluation itself was cut.

    ``records_captured < records_total`` means the *display* was bounded; the
    numeric result is still complete. ``evaluation_truncated`` means a
    traversal or evaluation stopped early, so its own number is incomplete and
    must not be described as final.
    """

    records_captured: int
    records_total: int
    witnesses_captured: int = 0
    witnesses_total: int = 0
    evaluation_truncated: bool = False

    @property
    def records_truncated(self) -> bool:
        return self.records_captured < self.records_total

    @property
    def witnesses_truncated(self) -> bool:
        return self.witnesses_captured < self.witnesses_total

    @property
    def complete(self) -> bool:
        return not (self.records_truncated or self.witnesses_truncated or self.evaluation_truncated)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records_captured": int(self.records_captured),
            "records_total": int(self.records_total),
            "records_truncated": self.records_truncated,
            "witnesses_captured": int(self.witnesses_captured),
            "witnesses_total": int(self.witnesses_total),
            "witnesses_truncated": self.witnesses_truncated,
            "evaluation_truncated": bool(self.evaluation_truncated),
            "complete": self.complete,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Truncation:
        """Rebuild from :meth:`to_dict`; the three flags are derived from the counts."""
        return cls(**only_fields(cls, data))


# ---------------------------------------------------------------------- packet
@dataclass(frozen=True)
class EvidencePacket:
    """A run's evidence: records, per-view annotations, coverage and limitations.

    The packet keeps the snapshot it was captured from so that witnesses can be
    materialised later — through :meth:`witnesses`, which refuses stale inputs.

    A packet rebuilt by :meth:`from_dict` carries neither the snapshot nor the
    norm — an exported file has no live log to re-read — and says so:
    :attr:`restored` is ``True``, the repr and the export declare it, and
    :meth:`witnesses` refuses rather than returning an empty tuple.
    """

    manifest: RunManifest
    unit_type: str
    records: tuple[EvaluationRecord, ...]
    coverage: CoverageReport
    annotations: tuple[ViewAnnotation, ...] = ()
    diagnostics: tuple[DiagnosticResult, ...] = ()
    qualifications: tuple[Qualification, ...] = ()
    truncation: Truncation | None = None
    capture_mode: str = "summary"
    schema_version: str = PACKET_SCHEMA_VERSION
    snapshot: LogSnapshot | None = field(default=None, repr=False, compare=False)
    norm: Norm | None = field(default=None, repr=False, compare=False)
    #: ``True`` for a packet read back from an export rather than captured from
    #: a run. It compares equal to the packet it came from — the evidence is the
    #: same evidence — but it can no longer reach the log behind it.
    restored: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "annotations", tuple(self.annotations))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "qualifications", tuple(self.qualifications))
        ids = [r.evaluation_id for r in self.records]
        if len(set(ids)) != len(ids):
            raise EvidenceError("evaluation ids must be unique inside a packet")
        known = set(ids)
        for annotation in self.annotations:
            if annotation.evaluation_id not in known:
                raise EvidenceError(f"annotation refers to unknown evaluation {annotation.evaluation_id!r}")

    @property
    def run_id(self) -> str:
        return self.manifest.run_id

    @property
    def views(self) -> list[str]:
        return list(dict.fromkeys(a.view for a in self.annotations))

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        state = "complete" if self.truncation is None or self.truncation.complete else "truncated"
        origin = ", restored from an export (no snapshot, no norm)" if self.restored else ""
        return (
            f"EvidencePacket({len(self.records):,} records, mode={self.capture_mode!r}, "
            f"views={self.views}, {state}, run={self.run_id!r}{origin})"
        )

    def record(self, evaluation_id: str) -> EvaluationRecord:
        """One record by id."""
        for r in self.records:
            if r.evaluation_id == evaluation_id:
                return r
        raise EvidenceError(f"unknown evaluation id {evaluation_id!r}")

    def records_for(self, *, unit_id: Any = None, constraint_id: str | None = None) -> tuple[EvaluationRecord, ...]:
        """The records of one unit, one constraint, or their intersection."""
        out = self.records
        if unit_id is not None:
            out = tuple(r for r in out if r.unit_id == str(unit_id))
        if constraint_id is not None:
            out = tuple(r for r in out if r.constraint_id == constraint_id)
        return out

    def witnesses(self, evaluation_id: str, *, limit: int | None = None) -> tuple[WitnessRef, ...]:
        """Witnesses of one record, materialised on demand.

        Already-materialised witnesses are returned as captured. Otherwise the
        packet asks its snapshot, which raises
        :class:`~wise.errors.StaleEvidenceError` when the log has changed since
        capture — a later trace query is not automatically the evidence of an
        earlier score.
        """
        record = self.record(evaluation_id)
        if record.witnesses_materialised:
            return record.witnesses if limit is None else record.witnesses[:limit]
        if self.snapshot is None:
            from ..errors import EvidenceUnavailableError

            if self.restored:
                raise EvidenceUnavailableError(
                    f"{evaluation_id}: this packet was restored from an export, so it carries neither the "
                    "snapshot nor the norm and cannot reach the log the score was taken from. Only witnesses "
                    "captured with evidence='full' survive an export; re-score to materialise the rest."
                )
            raise EvidenceUnavailableError(
                f"{evaluation_id}: no snapshot was kept, so witnesses cannot be materialised; "
                "capture with evidence='full', or keep the snapshot"
            )
        from .capture import materialise_witnesses

        return materialise_witnesses(self, record, limit=limit)

    def to_dict(self, *, include_witnesses: bool = True) -> dict[str, Any]:
        records = []
        for r in self.records:
            row = r.to_dict()
            if not include_witnesses:
                # dropping the witnesses drops the claim that they were
                # materialised with them: a reader of this export must be told
                # that no witness is available, not handed an empty list
                row["witnesses"] = []
                row["witnesses_materialised"] = False
            records.append(row)
        return {
            "schema_version": self.schema_version,
            "capture_mode": self.capture_mode,
            "unit_type": self.unit_type,
            "run": self.manifest.to_dict(),
            "records": records,
            "annotations": [a.to_dict() for a in self.annotations],
            "coverage": self.coverage.to_dict(),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "qualifications": [q.to_dict() for q in self.qualifications],
            "truncation": None if self.truncation is None else self.truncation.to_dict(),
            # what the reader gets back: an export carries no snapshot and no
            # norm, and says so rather than leaving two silent Nones
            "witness_access": self._witness_access(),
        }

    def _witness_access(self) -> dict[str, Any]:
        return {
            "restored_from_export": bool(self.restored),
            "snapshot_available": self.snapshot is not None,
            "norm_available": self.norm is not None,
            "materialised_records": sum(1 for r in self.records if r.witnesses_materialised),
            "note": (
                "witnesses already materialised travel with the records; "
                "any other witness needs the run's own snapshot and norm, which an export does not carry"
            ),
        }

    def to_json(self, indent: int | None = 2, *, include_witnesses: bool = True) -> str:
        """UTF-8 JSON with explicit nulls and ``allow_nan=False``.

        A value that is missing stays ``null`` next to the reason code that
        explains it; there is no NaN in the output and no silent ``str()`` of
        an unsupported type.
        """
        return json.dumps(
            _canonical(self.to_dict(include_witnesses=include_witnesses)),
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidencePacket:
        """Read a packet back from :meth:`to_dict` or :meth:`to_json`.

        Every record is rebuilt through :meth:`EvaluationRecord.from_dict`, so
        the three questions stay three answers after a reload: an unevaluated
        check still carries its reason code and a null violation, a censored one
        still carries ``open_observation_window``, its ``lower_bound``
        qualification and its ``lower_bound: true`` measurement, and a file
        claiming a violation for an unevaluated check is refused on read.

        The snapshot and the norm are **not** restored, and the packet says so
        (:attr:`restored`): an export is a record of a run, not a licence to
        re-read the log behind it. Witnesses captured with ``evidence="full"``
        survive; anything else raises
        :class:`~wise.errors.EvidenceUnavailableError` instead of an empty tuple.

        >>> import json, wise
        >>> result = wise.score(wise.datasets.running_p2p_log(), wise.datasets.running_p2p_norm(), evidence="full")
        >>> again = EvidencePacket.from_dict(json.loads(result.evidence.to_json()))
        >>> again.records == result.evidence.records
        True
        """
        version = str(data.get("schema_version", ""))
        if version != PACKET_SCHEMA_VERSION:
            raise EvidenceError(f"unsupported evidence schema version {version!r}; this library reads {PACKET_SCHEMA_VERSION!r}")
        if "run" not in data:
            raise EvidenceError("an evidence packet needs its run record under 'run'; this payload has none")
        return cls(
            manifest=RunManifest.from_dict(data["run"]),
            unit_type=str(data.get("unit_type", "case")),
            records=tuple(EvaluationRecord.from_dict(row) for row in data.get("records", ())),
            coverage=CoverageReport.from_dict(data["coverage"]),
            annotations=tuple(ViewAnnotation.from_dict(row) for row in data.get("annotations", ())),
            diagnostics=tuple(DiagnosticResult.from_dict(row) for row in data.get("diagnostics", ())),
            qualifications=tuple(Qualification.from_dict(row) for row in data.get("qualifications", ())),
            truncation=None if data.get("truncation") is None else Truncation.from_dict(data["truncation"]),
            capture_mode=str(data.get("capture_mode", "summary")),
            schema_version=version,
            restored=True,
        )


def qualification_index(items: Iterable[Qualification]) -> dict[str, list[str]]:
    """Group qualification messages by code — handy for a report header."""
    out: dict[str, list[str]] = {}
    for q in items:
        out.setdefault(q.code.value, []).append(q.message)
    return out


def to_records_frame(records: Sequence[EvaluationRecord], annotations: Mapping[str, ViewAnnotation] | None = None) -> Any:
    """Long-format table of records (one row per evaluation).

    Imported lazily by :mod:`wise.evidence.capture`; kept here so that a
    consumer that only has records can build the same frame.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for r in records:
        row: dict[str, Any] = {
            "run_id": r.run_id,
            "evaluation_id": r.evaluation_id,
            "unit_id": r.unit_id,
            "unit_type": r.unit_type,
            "constraint_id": r.constraint_id,
            "constraint_type": r.constraint_type,
            "constraint_version": r.constraint_version,
            "in_scope": r.in_scope,
            "evaluable": r.evaluable,
            "reason_code": r.reason_code.value,
            "violation": r.violation,
        }
        for m in r.measurements:
            row[f"m__{m.name}"] = m.value
            row[f"m__{m.name}__unit"] = m.unit
        row["n_witnesses"] = len(r.witnesses)
        row["n_witnesses_total"] = r.n_witnesses_total
        row["witnesses_materialised"] = r.witnesses_materialised
        row["qualifications"] = ";".join(q.code.value for q in r.qualifications)
        if annotations is not None:
            annotation = annotations.get(r.evaluation_id)
            row["effective_weight"] = None if annotation is None else annotation.effective_weight
            row["penalty"] = None if annotation is None else annotation.penalty
            row["scored"] = None if annotation is None else annotation.scored
        rows.append(row)
    return pd.DataFrame(rows)
