"""Three native relational check families, evaluated over preserved relations.

These are the checks a case log cannot express. Each one reads an
:class:`~wise.oc.units.AssessmentUnit` — an anchor, its typed role bindings and
the events in scope — and returns a :class:`CheckOutcome`: a violation in
``[0, 1]`` *or* an honest refusal to produce one, with the reason, the raw
measurements, the witnesses and the limitations it carries.

:class:`RelatedObjectCardinality`
    How many distinct related objects a role binds, against a declared minimum
    and maximum. Missing relationship information is not proof of absence: a
    shortfall is only reported as a violation when the configuration declares
    the scope complete, and a shortfall measured inside a *truncated* context
    is not reported at all.

:class:`CrossObjectLag`
    The time from an activation event to a matching response event, where the
    two are found through different objects of the unit. Activation, matching
    rule, clock, equal-time interpretation and missingness are all declared
    fields. When two candidate responses tie, the result is a typed
    ``ambiguous_match`` — never a silent nearest-timestamp choice.

:class:`RelationalBalance`
    Two sets of matched amounts against each other, with a declared currency or
    unit policy and a tolerance. The totals are formed through
    :mod:`wise.oc.accounting`, so one source amount cannot be consumed twice in
    one reconciliation, and currencies are never converted without a declared
    rate.

These live in their own catalogue. They are **not** members of
:data:`wise.constraints` and they are not loadable through
:meth:`wise.norm.Norm.from_dict`: a check whose parameters name roles and
qualifiers is not a schema-2 case constraint, and pretending otherwise would
make an old norm file mean something new.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd

from ..constraints import _UNITS, sat
from ..errors import OCConstraintError
from ..evidence.models import (
    AbsenceSearch,
    Completeness,
    Measurement,
    MeasurementKind,
    Qualification,
    QualificationCode,
    ReasonCode,
    SourceIdentity,
    WitnessKind,
    WitnessRef,
)
from .accounting import Allocation, AllocationReport, QuantityRecord, allocate
from .model import AttributePolicy, OCEvent, OCEventLog
from .units import AssessmentUnit, UnitSpec

#: What a role name may be. ``None`` and ``"anchor"`` both mean the anchor.
ANCHOR = "anchor"


def _role_ids(unit: AssessmentUnit, role: str | None) -> tuple[str, ...]:
    """The object ids one selector points at: the anchor, or a bound role."""
    if role is None or role == ANCHOR:
        return (unit.anchor_id,)
    return unit.role(role)


@dataclass(frozen=True)
class CheckOutcome:
    """What one native check made of one unit.

    ``violation is None`` means *not evaluable*, and the ``reason`` says why.
    It never means zero: this library does not turn a missing observation into
    a satisfied check.
    """

    violation: float | None
    reason: ReasonCode
    measurements: tuple[Measurement, ...] = ()
    witnesses: tuple[WitnessRef, ...] = ()
    qualifications: tuple[Qualification, ...] = ()
    policies: dict[str, Any] = field(default_factory=dict)
    n_witnesses_total: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", ReasonCode(self.reason))
        object.__setattr__(self, "measurements", tuple(self.measurements))
        object.__setattr__(self, "witnesses", tuple(self.witnesses))
        object.__setattr__(self, "qualifications", tuple(self.qualifications))
        object.__setattr__(self, "policies", dict(self.policies))
        if self.reason.evaluable:
            if self.violation is None or not np.isfinite(self.violation):
                raise OCConstraintError(f"reason {self.reason.value!r} promises a violation, got {self.violation!r}")
            object.__setattr__(self, "violation", float(np.clip(float(self.violation), 0.0, 1.0)))
        elif self.violation is not None:
            raise OCConstraintError(
                f"reason {self.reason.value!r} carries no violation, got {self.violation!r} — "
                "a check that was not evaluated is not a satisfied check"
            )

    @property
    def evaluable(self) -> bool:
        return self.reason.evaluable


@dataclass(frozen=True)
class ObjectCheck:
    """Base class of the native catalogue. Parameters only, no state."""

    type: ClassVar[str] = ""

    def roles(self) -> tuple[str, ...]:
        """Role names this check reads, excluding the anchor."""
        raise NotImplementedError  # pragma: no cover - abstract

    def check_spec(self, spec: UnitSpec) -> tuple[str, ...]:
        """Problems with evaluating this check against one unit type."""
        declared = set(spec.role_names)
        return tuple(
            f"role {role!r} is not declared by unit type {spec.unit_type!r}" for role in self.roles() if role not in declared
        )

    def evaluate(self, log: OCEventLog, unit: AssessmentUnit) -> CheckOutcome:
        raise NotImplementedError  # pragma: no cover - abstract

    def params(self) -> dict[str, Any]:
        """The declared parameters, for the evidence record."""
        return {f.name: _plain(getattr(self, f.name)) for f in fields(self)}

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "params": self.params()}


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _dedupe(qualifications: Sequence[Qualification]) -> list[Qualification]:
    """The same qualifications, each stated once, in the order first seen."""
    seen: set[tuple[Any, str, str]] = set()
    out: list[Qualification] = []
    for q in qualifications:
        key = (q.code, q.message, q.scope)
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out


def _truncated_role(unit: AssessmentUnit, role: str | None) -> bool:
    """Whether the context of this particular role was cut."""
    if role is None or role == ANCHOR:
        return unit.truncation.visit_limited
    return role in unit.truncation.limited_roles or unit.truncation.visit_limited


# ------------------------------------------------------- related-object cardinality
@dataclass(frozen=True)
class RelatedObjectCardinality(ObjectCheck):
    """How many distinct objects a role binds, against a declared range.

    ``minimum`` states an obligation ("every invoice belongs to exactly one
    order"), ``maximum`` a limit ("no more than three partial deliveries").
    A shortfall against ``minimum`` is an *absence* claim, and this library
    will not make one from silence: unless ``completeness`` is
    :data:`~wise.evidence.models.Completeness.ASSUMED_COMPLETE`, a shortfall
    produces :data:`~wise.evidence.models.ReasonCode.UNVERIFIED_ABSENCE` and no
    violation. An excess over ``maximum`` needs no such assumption — the
    objects that prove it are right there.

    ``width`` grades the violation over that many missing or extra objects;
    ``width=0`` makes it a step.

    >>> RelatedObjectCardinality(role="order", minimum=1).type
    'related_object_cardinality'
    """

    type: ClassVar[str] = "related_object_cardinality"

    role: str
    minimum: int | None = None
    maximum: int | None = None
    target_type: str | None = None
    width: float = 0.0
    completeness: Completeness = Completeness.UNKNOWN

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", str(self.role))
        object.__setattr__(self, "completeness", Completeness(self.completeness))
        for name in ("minimum", "maximum"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, int(value))
                if int(value) < 0:
                    raise OCConstraintError(f"cardinality: {name} must be at least 0, got {value}")
        if self.minimum is None and self.maximum is None:
            raise OCConstraintError("cardinality: declare a minimum, a maximum or both; a range with neither checks nothing")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise OCConstraintError(f"cardinality: minimum {self.minimum} exceeds maximum {self.maximum}")
        object.__setattr__(self, "width", float(self.width))
        if self.width < 0:
            raise OCConstraintError(f"cardinality: width must be at least 0, got {self.width}")

    def roles(self) -> tuple[str, ...]:
        return () if self.role in (ANCHOR, "") else (self.role,)

    def evaluate(self, log: OCEventLog, unit: AssessmentUnit) -> CheckOutcome:
        bound = _role_ids(unit, self.role)
        if self.target_type is not None:
            bound = tuple(oid for oid in bound if log.obj(oid).object_type == self.target_type)
        count = len(set(bound))
        shortfall = max(0, self.minimum - count) if self.minimum is not None else 0
        excess = max(0, count - self.maximum) if self.maximum is not None else 0
        truncated = _truncated_role(unit, self.role)

        measurements = (
            Measurement("related_objects", float(count), "objects", MeasurementKind.COUNT, lower_bound=truncated),
            Measurement("minimum", None if self.minimum is None else float(self.minimum), "objects", MeasurementKind.THRESHOLD),
            Measurement("maximum", None if self.maximum is None else float(self.maximum), "objects", MeasurementKind.THRESHOLD),
        )
        # One aggregate witness for the role, with the bound identities recorded
        # beside it. The per-object justification is the unit's own
        # ``witnesses_for(role)`` path witness; duplicating it here as event
        # witnesses would claim event identity that a relation does not have.
        witnesses = (
            WitnessRef(
                kind=WitnessKind.AGGREGATE,
                role=self.role,
                unit_id=unit.unit_id,
                identity=SourceIdentity.NONE,
            ),
        )
        policies = {
            "role": self.role,
            "target_type": self.target_type,
            "completeness": self.completeness.value,
            "context_complete": unit.complete,
            "bound_objects": sorted(set(bound)),
            "bound_types": sorted({log.obj(oid).object_type for oid in set(bound)}),
        }
        quals: list[Qualification] = []

        if truncated:
            quals.append(
                Qualification(
                    QualificationCode.COUNT_IS_A_LOWER_BOUND,
                    f"{unit.unit_id}: role {self.role!r} was cut by a traversal limit "
                    f"({'; '.join(unit.truncation.reasons)}), so {count} related object(s) is a lower bound",
                    scope="evaluation",
                )
            )
        if shortfall > 0 and truncated:
            return CheckOutcome(
                None,
                ReasonCode.BUDGET_TRUNCATED,
                measurements,
                witnesses,
                (*quals, _context_truncated(unit)),
                policies,
            )
        if truncated and self.maximum is not None and excess == 0:
            # The mirror of the branch above. A cut count is a lower bound, so
            # a limit it stays under is not established: the objects that would
            # have broken the limit are exactly the ones the traversal did not
            # reach. Reporting this as ``observed``, ν = 0.0 would let the cut
            # itself produce the pass.
            return CheckOutcome(
                None,
                ReasonCode.BUDGET_TRUNCATED,
                measurements,
                witnesses,
                (*quals, _limit_not_established(unit, self.role, count, self.maximum)),
                policies,
            )
        if shortfall > 0 and self.completeness is not Completeness.ASSUMED_COMPLETE:
            quals.append(
                Qualification(
                    QualificationCode.ABSENCE_COMPLETENESS_UNKNOWN,
                    f"{unit.unit_id}: {count} object(s) are bound to role {self.role!r} where {self.minimum} are required, "
                    f"but the scope is declared {self.completeness.value!r}. Missing relationship information is not "
                    "proof of absence; declare completeness='assumed_complete' to report this shortfall as a violation",
                    scope="evaluation",
                )
            )
            return CheckOutcome(None, ReasonCode.UNVERIFIED_ABSENCE, measurements, witnesses, tuple(quals), policies)

        violation = max(float(sat(shortfall, 0.0, self.width)), float(sat(excess, 0.0, self.width)))
        if shortfall > 0:
            quals.append(
                Qualification(
                    QualificationCode.ABSENCE_COMPLETENESS_UNKNOWN,
                    f"{unit.unit_id}: the shortfall of {shortfall} object(s) is reported as a violation because the "
                    f"configuration declares the scope of role {self.role!r} complete; that assumption is the "
                    "caller's, not the data's",
                    scope="evaluation",
                )
            )
        return CheckOutcome(violation, ReasonCode.OBSERVED, measurements, witnesses, tuple(quals), policies)


def _context_truncated(unit: AssessmentUnit) -> Qualification:
    return Qualification(
        QualificationCode.CONTEXT_TRUNCATED,
        f"{unit.unit_id}: the context was cut ({'; '.join(unit.truncation.reasons)}), so an absence cannot be "
        "established from it at all",
        scope="evaluation",
    )


def _limit_not_established(unit: AssessmentUnit, role: str, count: int, maximum: int) -> Qualification:
    return Qualification(
        QualificationCode.CONTEXT_TRUNCATED,
        f"{unit.unit_id}: the context was cut ({'; '.join(unit.truncation.reasons)}), so the {count} object(s) "
        f"bound to role {role!r} are a lower bound; staying under the maximum of {maximum} is a property of the "
        "cut, not a finding about the unit",
        scope="evaluation",
    )


# ------------------------------------------------------------------- cross-object lag
@dataclass(frozen=True)
class EventSelector:
    """Which events of a unit count, and which of them is *the* one.

    ``role=None`` selects the anchor's events; a role name selects the events
    of the objects bound to that role — which is what makes this cross-object:
    the activation may sit on the purchase order and the response on the
    invoice, and the relation between them is what brings the two together.
    """

    role: str | None = None
    activities: tuple[str, ...] = ()
    qualifier: str | None = None
    occurrence: Literal["first", "last"] = "first"

    def __post_init__(self) -> None:
        object.__setattr__(self, "activities", tuple(str(a) for a in self.activities))
        if not self.activities:
            raise OCConstraintError("an event selector must name at least one activity")
        if self.occurrence not in ("first", "last"):
            raise OCConstraintError(f"occurrence must be 'first' or 'last', got {self.occurrence!r}")

    def collect(self, log: OCEventLog, unit: AssessmentUnit) -> tuple[OCEvent, ...]:
        """Every event this selector matches, in ``(timestamp, event_id)`` order."""
        picked: dict[str, OCEvent] = {}
        for object_id in _role_ids(unit, self.role):
            for activity in self.activities:
                for event in log.events_of(object_id, qualifier=self.qualifier, activity=activity):
                    picked[event.event_id] = event
        return tuple(sorted(picked.values(), key=lambda e: (e.timestamp, e.event_id)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "activities": list(self.activities),
            "qualifier": self.qualifier,
            "occurrence": self.occurrence,
        }


@dataclass(frozen=True)
class CrossObjectLag(ObjectCheck):
    """Time from an activation event to its matching response, across objects.

    Every semantic choice is a declared field, because each of them silently
    changes the answer:

    ``activation`` / ``response``
        :class:`EventSelector`\\ s naming the role, the activities and — for the
        activation — whether the first or the last occurrence is meant.
    ``match``
        ``"first_after"`` takes the earliest response at or after the
        activation; ``"first_overall"`` takes the earliest response in the unit
        and counts it as missing when it precedes the activation.
    ``clock``
        The clock the timestamps are on. Recorded, and checked against the
        log's own declared time zone.
    ``equal_time``
        Whether a response bearing the *same* timestamp as the activation
        counts as a response (``"counts"``) or not (``"excluded"``, the
        default). Equal timestamps on distinct events are common in extracted
        logs and this is the only honest place to decide what they mean.
    ``missing_activation`` / ``missing_response``
        ``"violate"``, ``"skip"``, and for the response also ``"censor"``,
        which scores the time elapsed so far against ``horizon`` and marks the
        result a lower bound.
    ``on_ambiguous``
        What to do when two candidate responses tie on the matching timestamp.
        ``"qualify"`` (the default) refuses to choose and returns
        ``ambiguous_match``; ``"tie_break"`` takes the smaller event id and says
        so; ``"error"`` raises. There is no silent nearest-timestamp rule.

    >>> CrossObjectLag(EventSelector(activities=("A",)), EventSelector(activities=("B",))).type
    'cross_object_lag'
    """

    type: ClassVar[str] = "cross_object_lag"

    activation: EventSelector
    response: EventSelector
    delta: float | None = 0.0
    width: float = 0.0
    unit: str = "D"
    clock: str = "event_timestamp"
    match: Literal["first_after", "first_overall"] = "first_after"
    equal_time: Literal["counts", "excluded"] = "excluded"
    missing_activation: Literal["violate", "skip"] = "violate"
    missing_response: Literal["violate", "skip", "censor"] = "violate"
    horizon: pd.Timestamp | None = None
    on_ambiguous: Literal["qualify", "tie_break", "error"] = "qualify"

    def __post_init__(self) -> None:
        for name in ("activation", "response"):
            if not isinstance(getattr(self, name), EventSelector):
                raise OCConstraintError(f"cross-object lag: {name} must be an EventSelector")
        if self.delta is not None:
            object.__setattr__(self, "delta", float(self.delta))
            if self.delta < 0:
                raise OCConstraintError(f"cross-object lag: delta must be at least 0, got {self.delta}")
        object.__setattr__(self, "width", float(self.width))
        if self.width < 0:
            raise OCConstraintError(f"cross-object lag: width must be at least 0, got {self.width}")
        if self.unit not in _UNITS:
            raise OCConstraintError(f"cross-object lag: unit must be one of {sorted(set(_UNITS.values()))}, got {self.unit!r}")
        object.__setattr__(self, "unit", _UNITS[self.unit])
        for name, allowed in (
            ("match", ("first_after", "first_overall")),
            ("equal_time", ("counts", "excluded")),
            ("missing_activation", ("violate", "skip")),
            ("missing_response", ("violate", "skip", "censor")),
            ("on_ambiguous", ("qualify", "tie_break", "error")),
        ):
            if getattr(self, name) not in allowed:
                raise OCConstraintError(f"cross-object lag: {name} must be one of {allowed}, got {getattr(self, name)!r}")
        if self.horizon is not None:
            object.__setattr__(self, "horizon", pd.Timestamp(self.horizon))
        if self.missing_response == "censor" and self.delta is None:
            raise OCConstraintError("cross-object lag: censoring an open item needs a delta to censor against")

    def roles(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(r for r in (self.activation.role, self.response.role) if r not in (None, ANCHOR)))

    def params(self) -> dict[str, Any]:
        out = super().params()
        out["activation"] = self.activation.to_dict()
        out["response"] = self.response.to_dict()
        return out

    # ------------------------------------------------------------------ evaluate
    def evaluate(self, log: OCEventLog, unit: AssessmentUnit) -> CheckOutcome:
        policies = {
            "clock": self.clock,
            "match": self.match,
            "equal_time": self.equal_time,
            "missing_activation": self.missing_activation,
            "missing_response": self.missing_response,
            "on_ambiguous": self.on_ambiguous,
            "log_timezone": log.source.timezone,
            "observed_precision": log.source.precision.get("events"),
            "context_complete": unit.complete,
        }
        activations = self.activation.collect(log, unit)
        responses = self.response.collect(log, unit)
        searched = len(unit.event_ids)

        if not activations:
            return self._no_activation(unit, responses, searched, policies)

        chosen = activations[0] if self.activation.occurrence == "first" else activations[-1]
        tied_activation = [e for e in activations if e.timestamp == chosen.timestamp]
        quals: list[Qualification] = []
        if len(tied_activation) > 1:
            outcome = self._ambiguous(unit, "activation", tied_activation, policies)
            if outcome is not None:
                return outcome
            quals.append(_resolved(unit, "activation", tied_activation))
            chosen = min(tied_activation, key=lambda e: e.event_id)

        candidates = self._candidates(responses, chosen.timestamp)
        if not candidates:
            return self._no_response(log, unit, chosen, responses, searched, policies, quals)

        earliest = candidates[0].timestamp
        tied_response = [e for e in candidates if e.timestamp == earliest]
        if len(tied_response) > 1:
            outcome = self._ambiguous(unit, "response", tied_response, policies)
            if outcome is not None:
                return outcome
            quals.append(_resolved(unit, "response", tied_response))
        matched = min(tied_response, key=lambda e: e.event_id)

        lag = _in_units(matched.timestamp - chosen.timestamp, self.unit)
        violation = float(sat(lag, self.delta, self.width)) if self.delta is not None else 0.0
        measurements = (
            Measurement("lag", float(lag), self.unit, MeasurementKind.DURATION),
            Measurement("activation_timestamp", chosen.timestamp.isoformat(), "timestamp", MeasurementKind.TIMESTAMP),
            Measurement("response_timestamp", matched.timestamp.isoformat(), "timestamp", MeasurementKind.TIMESTAMP),
            Measurement("delta", None if self.delta is None else float(self.delta), self.unit, MeasurementKind.THRESHOLD),
            Measurement("candidate_responses", float(len(candidates)), "events", MeasurementKind.COUNT),
        )
        witnesses = (_event_witness(chosen, "activation", unit), _event_witness(matched, "response", unit))
        if not unit.complete:
            quals.append(_context_truncated(unit))
        return CheckOutcome(violation, ReasonCode.OBSERVED, measurements, witnesses, tuple(quals), policies, len(candidates) + 1)

    # ---------------------------------------------------------------- internals
    def _candidates(self, responses: Sequence[OCEvent], t_a: pd.Timestamp) -> tuple[OCEvent, ...]:
        if self.match == "first_overall":
            first = responses[0] if responses else None
            if first is None:
                return ()
            same = [e for e in responses if e.timestamp == first.timestamp]
            if first.timestamp < t_a or (first.timestamp == t_a and self.equal_time == "excluded"):
                return ()  # the earliest response precedes the activation: it is not a response to it
            return tuple(same)
        if self.equal_time == "counts":
            return tuple(e for e in responses if e.timestamp >= t_a)
        return tuple(e for e in responses if e.timestamp > t_a)

    def _ambiguous(
        self, unit: AssessmentUnit, side: str, tied: Sequence[OCEvent], policies: dict[str, Any]
    ) -> CheckOutcome | None:
        names = ", ".join(sorted(e.event_id for e in tied))
        message = (
            f"{unit.unit_id}: {len(tied)} {side} events share the timestamp {tied[0].timestamp.isoformat()} ({names}); "
            "they are distinct events, and choosing one of them by timestamp alone would be a guess"
        )
        if self.on_ambiguous == "error":
            raise OCConstraintError(message + ". Declare on_ambiguous='tie_break' or a narrower selector.")
        if self.on_ambiguous == "tie_break":
            return None
        return CheckOutcome(
            None,
            ReasonCode.AMBIGUOUS_MATCH,
            (Measurement(f"tied_{side}_events", float(len(tied)), "events", MeasurementKind.COUNT),),
            tuple(_event_witness(e, side, unit) for e in tied),
            (
                Qualification(
                    # not AMBIGUOUS_MATCH_RESOLVED: nothing was resolved. The
                    # prose said so already; the code has to as well, or a
                    # consumer grouping by code cannot separate a tie that was
                    # broken from a tie that stood.
                    QualificationCode.AMBIGUOUS_MATCH_UNRESOLVED,
                    message + "; no match was made",
                    scope="evaluation",
                ),
            ),
            policies,
        )

    def _no_activation(
        self, unit: AssessmentUnit, responses: Sequence[OCEvent], searched: int, policies: dict[str, Any]
    ) -> CheckOutcome:
        search = AbsenceSearch(
            unit_id=unit.unit_id,
            activities=self.activation.activities,
            window_start=None if unit.scope.window_start is None else unit.scope.window_start.isoformat(),
            window_end=None if unit.scope.window_end is None else unit.scope.window_end.isoformat(),
            # never ASSUMED_COMPLETE: an untruncated traversal says the *context* was
            # gathered whole, not that the log observed every event of it
            completeness=Completeness.UNKNOWN,
            n_events_searched=searched,
            filters={"role": self.activation.role, "qualifier": self.activation.qualifier, "context_complete": unit.complete},
        )
        witness = WitnessRef(kind=WitnessKind.ABSENCE, role="activation", unit_id=unit.unit_id, search=search)
        measurements = (Measurement("responses_seen", float(len(responses)), "events", MeasurementKind.COUNT),)
        if self.missing_activation == "violate":
            return CheckOutcome(1.0, ReasonCode.POLICY_VIOLATION_MISSING_ACTIVATION, measurements, (witness,), (), policies)
        return CheckOutcome(None, ReasonCode.SKIPPED_MISSING_ACTIVATION, measurements, (witness,), (), policies)

    def _no_response(
        self,
        log: OCEventLog,
        unit: AssessmentUnit,
        activation: OCEvent,
        responses: Sequence[OCEvent],
        searched: int,
        policies: dict[str, Any],
        quals: list[Qualification],
    ) -> CheckOutcome:
        search = AbsenceSearch(
            unit_id=unit.unit_id,
            activities=self.response.activities,
            window_start=activation.timestamp.isoformat(),
            window_end=None if unit.scope.window_end is None else unit.scope.window_end.isoformat(),
            completeness=Completeness.OPEN if self.missing_response == "censor" else Completeness.UNKNOWN,
            n_events_searched=searched,
            filters={"role": self.response.role, "equal_time": self.equal_time, "match": self.match},
        )
        witnesses = (
            _event_witness(activation, "activation", unit),
            WitnessRef(kind=WitnessKind.ABSENCE, role="response", unit_id=unit.unit_id, search=search),
        )
        measurements: tuple[Measurement, ...] = (
            Measurement("activation_timestamp", activation.timestamp.isoformat(), "timestamp", MeasurementKind.TIMESTAMP),
            Measurement("responses_before_activation", float(len(responses)), "events", MeasurementKind.COUNT),
        )
        if self.missing_response == "violate":
            return CheckOutcome(
                1.0, ReasonCode.POLICY_VIOLATION_MISSING_RESPONSE, measurements, witnesses, tuple(quals), policies
            )
        if self.missing_response == "skip":
            return CheckOutcome(None, ReasonCode.SKIPPED_MISSING_RESPONSE, measurements, witnesses, tuple(quals), policies)
        horizon = self.horizon if self.horizon is not None else _log_horizon(log)
        elapsed = _in_units(horizon - activation.timestamp, self.unit)
        quals.append(
            Qualification(
                QualificationCode.CENSORED_HORIZON,
                f"{unit.unit_id}: no response had arrived by {pd.Timestamp(horizon).isoformat()}; the item is scored on "
                f"the {elapsed:.4g} {self.unit} elapsed so far, which is a lower bound on its final lag",
                scope="evaluation",
            )
        )
        quals.append(
            Qualification(
                QualificationCode.LOWER_BOUND,
                f"{unit.unit_id}: the observation window is still open, so this violation can only grow",
                scope="evaluation",
            )
        )
        measurements = (
            *measurements,
            Measurement("elapsed_at_horizon", float(elapsed), self.unit, MeasurementKind.DURATION, lower_bound=True),
            Measurement("horizon", pd.Timestamp(horizon).isoformat(), "timestamp", MeasurementKind.TIMESTAMP),
        )
        # delta is not None here: construction refuses censoring without one
        violation = float(sat(elapsed, float(self.delta or 0.0), self.width))
        return CheckOutcome(violation, ReasonCode.OPEN_OBSERVATION_WINDOW, measurements, witnesses, tuple(quals), policies)


def _resolved(unit: AssessmentUnit, side: str, tied: Sequence[OCEvent]) -> Qualification:
    return Qualification(
        QualificationCode.AMBIGUOUS_MATCH_RESOLVED,
        f"{unit.unit_id}: {len(tied)} {side} events share {tied[0].timestamp.isoformat()}; the declared tie-break "
        f"(smallest event id) chose {min(e.event_id for e in tied)!r}. The choice is the configuration's, not the data's",
        scope="evaluation",
    )


def _event_witness(event: OCEvent, role: str, unit: AssessmentUnit) -> WitnessRef:
    return WitnessRef(
        kind=WitnessKind.EVENT,
        role=role,
        unit_id=unit.unit_id,
        identity=SourceIdentity.SOURCE,
        event_id=event.event_id,
        activity=event.activity,
        timestamp=event.timestamp.isoformat(),
    )


def _in_units(delta: pd.Timedelta, unit: str) -> float:
    return float(pd.Timedelta(delta).total_seconds() / pd.Timedelta(f"1{_UNITS[unit]}").total_seconds())


def _log_horizon(log: OCEventLog) -> pd.Timestamp:
    """The last moment the log observed anything. Not a business deadline."""
    if not log.events:
        raise OCConstraintError("censoring needs a horizon and the log has no events to take one from")
    return max(e.timestamp for e in log.events)


# ------------------------------------------------------------------ relational balance
@dataclass(frozen=True)
class AmountSelector:
    """One side of a balance: whose amount, read how, in what unit."""

    role: str | None = None
    attribute: str = "amount"
    unit: str = ""
    unit_attribute: str | None = None
    policy: AttributePolicy | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "attribute", str(self.attribute))
        if not self.attribute:
            raise OCConstraintError("a balance selector must name the attribute it reads")
        if not self.unit and self.unit_attribute is None:
            raise OCConstraintError(
                "a balance selector must declare its unit or currency, or the attribute that carries it; "
                "an amount without a unit cannot be compared with another one"
            )
        if self.policy is not None:
            object.__setattr__(self, "policy", AttributePolicy(self.policy))

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "attribute": self.attribute,
            "unit": self.unit,
            "unit_attribute": self.unit_attribute,
            "policy": None if self.policy is None else self.policy.value,
        }


@dataclass(frozen=True)
class RelationalBalance(ObjectCheck):
    """Two matched sets of amounts against each other, with nothing double counted.

    Both sides are read at the unit's evaluation instant and turned into
    :class:`~wise.oc.accounting.QuantityRecord`\\ s with canonical identities,
    then allocated through :func:`~wise.oc.accounting.allocate`. That is what
    guarantees the rule "no amount consumed twice": a configuration that binds
    one object into both sides is refused rather than quietly counted on both.

    ``unit_policy="require_same"`` (the default) refuses to add amounts in
    different currencies. ``"declared_rates"`` converts with the ``rates`` table
    the caller supplies and marks the result as converted. Nothing is netted or
    converted implicitly.

    >>> RelationalBalance(AmountSelector(unit="EUR"), AmountSelector(role="items", unit="EUR")).type
    'relational_balance'
    """

    type: ClassVar[str] = "relational_balance"

    left: AmountSelector
    right: AmountSelector
    tolerance: float = 0.0
    width: float = 0.0
    eps: float = 1e-9
    unit_policy: Literal["require_same", "declared_rates"] = "require_same"
    rates: dict[str, float] = field(default_factory=dict)
    target_unit: str | None = None
    on_missing: Literal["skip", "violate"] = "skip"

    def __post_init__(self) -> None:
        for name in ("left", "right"):
            if not isinstance(getattr(self, name), AmountSelector):
                raise OCConstraintError(f"relational balance: {name} must be an AmountSelector")
        for name in ("tolerance", "width", "eps"):
            object.__setattr__(self, name, float(getattr(self, name)))
            if getattr(self, name) < 0:
                raise OCConstraintError(f"relational balance: {name} must be at least 0")
        if self.unit_policy not in ("require_same", "declared_rates"):
            raise OCConstraintError("relational balance: unit_policy must be 'require_same' or 'declared_rates'")
        object.__setattr__(self, "rates", {str(k): float(v) for k, v in dict(self.rates).items()})
        if self.unit_policy == "declared_rates" and self.target_unit is None:
            raise OCConstraintError("relational balance: declared_rates needs the target_unit it converts to")
        if self.on_missing not in ("skip", "violate"):
            raise OCConstraintError("relational balance: on_missing must be 'skip' or 'violate'")

    def roles(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(r for r in (self.left.role, self.right.role) if r not in (None, ANCHOR)))

    def params(self) -> dict[str, Any]:
        out = super().params()
        out["left"] = self.left.to_dict()
        out["right"] = self.right.to_dict()
        return out

    def evaluate(self, log: OCEventLog, unit: AssessmentUnit) -> CheckOutcome:
        policies = {
            "unit_policy": self.unit_policy,
            "rates": dict(self.rates),
            "target_unit": self.target_unit,
            "on_missing": self.on_missing,
            "evaluation_time": None if unit.evaluation_time is None else unit.evaluation_time.isoformat(),
            "context_complete": unit.complete,
        }
        left, missing_left, read_quals_left, bound_left = self._read(log, unit, self.left)
        right, missing_right, read_quals_right, bound_right = self._read(log, unit, self.right)
        policies["bound_objects"] = {"left": bound_left, "right": bound_right}
        shared = {r.record_id for r in left} & {r.record_id for r in right}
        if shared:
            raise OCConstraintError(
                f"{unit.unit_id}: record(s) {sorted(shared)} appear on both sides of the balance; one source amount "
                "must not be consumed twice in the same reconciliation. Narrow one of the selectors."
            )
        witnesses = tuple(
            WitnessRef(
                kind=WitnessKind.ATTRIBUTE,
                role=side,
                unit_id=unit.unit_id,
                identity=SourceIdentity.NONE,
                attribute=record.attribute,
                reference=None,
                timestamp=None if record.effective_from is None else record.effective_from.isoformat(),
            )
            for side, records in (("left", left), ("right", right))
            for record in records
        )
        quals: list[Qualification] = []
        if not unit.complete:
            quals.append(_context_truncated(unit))
        # a reading that says it looked at the wrong moment, or found nothing in
        # force at the right one, is part of what this outcome rests on
        quals.extend(_dedupe(read_quals_left + read_quals_right))
        counts: tuple[Measurement, ...] = (
            Measurement("left_records", float(len(left)), "records", MeasurementKind.COUNT),
            Measurement("right_records", float(len(right)), "records", MeasurementKind.COUNT),
            Measurement("missing_left", float(missing_left), "records", MeasurementKind.COUNT),
            Measurement("missing_right", float(missing_right), "records", MeasurementKind.COUNT),
        )
        if (not left or not right) and self.on_missing == "skip":
            empty = counts
            quals.append(
                Qualification(
                    QualificationCode.ABSENCE_COMPLETENESS_UNKNOWN,
                    f"{unit.unit_id}: {missing_left + missing_right} object(s) carry no {self.left.attribute!r} value at "
                    f"the evaluation time, so one side of the balance is empty; an unread amount is not a zero amount",
                    scope="evaluation",
                )
            )
            return CheckOutcome(None, ReasonCode.MISSING_ATTRIBUTE, empty, witnesses, tuple(quals), policies)

        try:
            report = self._reconcile(left, right)
        except _MixedUnits as exc:
            quals.append(
                Qualification(
                    QualificationCode.UNIT_CONVERSION_APPLIED,
                    f"{unit.unit_id}: {exc}. No rate was declared, so the two sides were not made comparable",
                    scope="evaluation",
                )
            )
            return CheckOutcome(
                None,
                ReasonCode.INCOMPATIBLE_UNITS,
                (Measurement("distinct_units", float(exc.n_units), "units", MeasurementKind.COUNT),),
                witnesses,
                tuple(quals),
                policies,
            )
        if self.unit_policy == "declared_rates" and self.rates:
            quals.append(
                Qualification(
                    QualificationCode.UNIT_CONVERSION_APPLIED,
                    f"{unit.unit_id}: amounts were converted to {report.unit} with the declared rates {self.rates}; "
                    "the rate is the caller's declaration and dates from no particular day",
                    scope="evaluation",
                )
            )
        total_left = report.allocated.get("left", 0.0)
        total_right = report.allocated.get("right", 0.0)
        denominator = max(total_left, total_right, self.eps)
        mismatch = abs(total_left - total_right) / denominator
        if max(total_left, total_right) <= 0:
            quals.append(
                Qualification(
                    QualificationCode.BOTH_TOTALS_ZERO,
                    f"{unit.unit_id}: both sides total zero, so the relative mismatch rests on eps={self.eps} rather "
                    "than on any observed amount",
                    scope="evaluation",
                )
            )
        for side, missing, bound, records, total, selector in (
            ("left", missing_left, bound_left, left, total_left, self.left),
            ("right", missing_right, bound_right, right, total_right, self.right),
        ):
            if not missing:
                continue
            # the side was summed over fewer objects than it binds. Dropping an
            # unread amount from a sum is arithmetically a zero, and a total
            # that is short by an unknown quantity is not a complete total.
            quals.append(
                Qualification(
                    QualificationCode.ABSENCE_COMPLETENESS_UNKNOWN,
                    f"{unit.unit_id}: {missing} of the {bound} object(s) bound to the {side} side carry no "
                    f"{selector.attribute!r} value at the evaluation time, so the {side} total of {float(total)} is a "
                    f"sum over {len(records)} of them; an unread amount is not a zero amount and this total is not complete",
                    scope="evaluation",
                )
            )
        measurements: tuple[Measurement, ...] = (
            Measurement("total_left", float(total_left), report.unit, MeasurementKind.AMOUNT),
            Measurement("total_right", float(total_right), report.unit, MeasurementKind.AMOUNT),
            Measurement("relative_mismatch", float(mismatch), "ratio", MeasurementKind.RATIO),
            Measurement("tolerance", float(self.tolerance), "ratio", MeasurementKind.THRESHOLD),
            *counts,
        )
        policies["allocation"] = {"complete": report.complete, "residual_total": report.residual_total}
        return CheckOutcome(
            float(sat(mismatch, self.tolerance, self.width)),
            ReasonCode.OBSERVED,
            measurements,
            witnesses,
            tuple(quals),
            policies,
        )

    def _read(
        self, log: OCEventLog, unit: AssessmentUnit, selector: AmountSelector
    ) -> tuple[list[QuantityRecord], int, list[Qualification], int]:
        """One side's amounts, what could not be read, and what the readings said.

        A reading carries its own qualifications —
        :data:`~wise.evidence.models.QualificationCode.ATTRIBUTE_READ_OUTSIDE_EVALUATION_TIME`
        when ``latest_known`` reached past the instant, and
        :data:`~wise.evidence.models.QualificationCode.ATTRIBUTE_NOT_SET_AT_TIME`
        when nothing was in force at it. Taking ``reading.value`` and discarding
        those is how a balance computed from the wrong day comes back
        unqualified, so they are returned and reach the outcome.
        """
        records: list[QuantityRecord] = []
        missing = 0
        quals: list[Qualification] = []
        policy = selector.policy or unit.attribute_policy
        if unit.evaluation_time is None and policy is AttributePolicy.AS_OF:
            raise OCConstraintError(
                f"{unit.unit_id}: a balance reads amounts as of an instant and this unit was built without one; "
                "rebuild the units with build_units(..., at=...), or declare policy='latest_known' on the selector "
                "and accept that the amount may postdate the assessment"
            )
        bound = _role_ids(unit, selector.role)
        if bound and unit.evaluation_time is None:
            # ``value_at`` compares the chosen change with the instant, and with
            # no instant there is nothing to compare: the reading is silent
            # precisely where the amount is least anchored. Say it here.
            quals.append(
                Qualification(
                    QualificationCode.ATTRIBUTE_READ_OUTSIDE_EVALUATION_TIME,
                    f"{unit.unit_id}: {selector.attribute!r} was read as the latest known value because this unit "
                    f"was built with no evaluation instant, so the amount is not established as of any moment and "
                    "may postdate the assessment",
                    scope="unit",
                )
            )
        for object_id in bound:
            reading = log.value_at(object_id, selector.attribute, unit.evaluation_time, policy=policy)
            quals.extend(reading.qualifications)
            if not reading.found:
                missing += 1
                continue
            unit_name = selector.unit
            if selector.unit_attribute is not None:
                currency = log.value_at(object_id, selector.unit_attribute, unit.evaluation_time, policy=policy)
                quals.extend(currency.qualifications)
                if not currency.found:
                    missing += 1
                    continue
                unit_name = str(currency.value)
            record = QuantityRecord.from_reading(reading, unit_name)
            if record is not None:
                records.append(record)
        return records, missing, quals, len(bound)

    def _reconcile(self, left: Sequence[QuantityRecord], right: Sequence[QuantityRecord]) -> AllocationReport:
        records = [*left, *right]
        units = {r.unit for r in records}
        rates = self.rates if self.unit_policy == "declared_rates" else None
        target = self.target_unit if self.unit_policy == "declared_rates" else None
        if rates is None and len(units) > 1:
            raise _MixedUnits(sorted(units))
        allocations = [
            *(Allocation(r.record_id, "left", 1.0) for r in left),
            *(Allocation(r.record_id, "right", 1.0) for r in right),
        ]
        return allocate(records, allocations, rates=rates, unit=target)


class _MixedUnits(Exception):
    def __init__(self, units: Sequence[str]) -> None:
        self.units = tuple(units)
        self.n_units = len(self.units)
        super().__init__(f"the amounts are in {list(self.units)} and unit_policy is 'require_same'")


#: The native catalogue, by declared type name. Deliberately separate from
#: :data:`wise.norm.CONSTRAINT_TYPES`: these checks read roles and qualifiers,
#: which a schema-2 case norm has no way to express.
OBJECT_CHECKS: dict[str, type[ObjectCheck]] = {
    RelatedObjectCardinality.type: RelatedObjectCardinality,
    CrossObjectLag.type: CrossObjectLag,
    RelationalBalance.type: RelationalBalance,
}


def object_check_from_dict(payload: Mapping[str, Any]) -> ObjectCheck:
    """Build one native check from ``{"type": ..., "params": {...}}``.

    The separate validation path the handoff asks for: an unknown type is
    refused by name, and nothing here can reach
    :meth:`wise.norm.Norm.from_dict`.

    >>> object_check_from_dict({"type": "related_object_cardinality", "params": {"role": "order", "minimum": 1}}).role
    'order'
    """
    if "type" not in payload:
        raise OCConstraintError("an object check needs a 'type'")
    name = str(payload["type"])
    try:
        cls = OBJECT_CHECKS[name]
    except KeyError:
        raise OCConstraintError(f"unknown object check type {name!r}; the native catalogue is {sorted(OBJECT_CHECKS)}") from None
    params = dict(payload.get("params", {}))
    if cls is CrossObjectLag:
        for side in ("activation", "response"):
            if isinstance(params.get(side), Mapping):
                params[side] = EventSelector(**dict(params[side]))
    if cls is RelationalBalance:
        for side in ("left", "right"):
            if isinstance(params.get(side), Mapping):
                params[side] = AmountSelector(**dict(params[side]))
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(params) - known)
    if unknown:
        raise OCConstraintError(f"object check {name!r} does not accept {unknown}; it accepts {sorted(known)}")
    return cls(**params)  # type: ignore[arg-type]


__all__ = [
    "ANCHOR",
    "OBJECT_CHECKS",
    "AmountSelector",
    "CheckOutcome",
    "CrossObjectLag",
    "EventSelector",
    "ObjectCheck",
    "RelatedObjectCardinality",
    "RelationalBalance",
    "object_check_from_dict",
]
