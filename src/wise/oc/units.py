"""Typed, bounded assessment units over an object log.

A case-based check knows what it is about: the case. An object-centric check
does not, and that is the whole difficulty. "Is this invoice matched?" needs an
*anchor* (the invoice), *related objects* in named roles (its purchase order,
its goods receipts), the *events* that count as observations of that
neighbourhood, and — because business graphs are not trees — a hard boundary
that stops the question from quietly becoming "is this vendor matched?".

:class:`UnitSpec` declares all four. :func:`build_units` evaluates it into
:class:`AssessmentUnit` objects, each carrying the subgraph witnesses that
justify its bindings and an explicit statement of whether its context is
complete.

What this deliberately is **not**
---------------------------------
There is no process-query language here, no ``MATCH`` clause, no wildcard hop.
A path is an ordered tuple of typed, directed, qualified steps
(:class:`PathStep`), each naming the object type it reaches. A step may be
declared ``transitive`` — follow this same relation while it leads to the same
type — and even then it is bounded by :attr:`TraversalLimits.max_depth`.

Three limits, and what happens at each
--------------------------------------
``max_fan_out``
    How many neighbours one object may contribute at one step. A purchase
    order with 400 items does not silently pull 400 objects into an invoice's
    context.
``max_bindings``
    How many objects one role may hold in one unit.
``max_depth`` / ``max_visited``
    How far a transitive step may run, and how many objects one unit may
    touch in total.

When any of them bites, the unit is still returned — a bounded partial context
is often exactly what a reviewer wants — but :attr:`AssessmentUnit.complete` is
``False``, :attr:`AssessmentUnit.truncation` says which limit bit, and a
:data:`~wise.evidence.models.QualificationCode.CONTEXT_TRUNCATED` qualification
travels with it. :meth:`AssessmentUnit.require_complete` exists so that a later
evaluator can refuse to present a truncated context as a finished assessment
rather than remembering to check a flag.

Time
----
:func:`build_units` takes the evaluation instant ``at`` and every attribute
read through :meth:`AssessmentUnit.attribute` uses it. Relations are a
different matter: OCEL 2.0 states *that* two objects are related and not *when*,
so a unit built at an instant from an atemporal source carries
:data:`~wise.evidence.models.QualificationCode.RELATION_VALIDITY_UNKNOWN`. The
bindings are not filtered by a validity interval that nobody recorded. A source
that *does* declare ``relation_time_semantics="interval"`` is a different
matter: its intervals are used, on the half-open convention
``[valid_from, valid_to)``, because using a recorded fact is not inference.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..errors import OCUnitError
from ..evidence.models import Qualification, QualificationCode
from .model import O2O, AttributePolicy, AttributeReading, OCEvent, OCEventLog, _timestamp

DIRECTIONS = ("forward", "reverse")


def _name(value: Any, what: str) -> str:
    out = str(value)
    if not out:
        raise OCUnitError(f"{what} must be a non-empty string")
    return out


# ---------------------------------------------------------------------- paths
@dataclass(frozen=True)
class PathStep:
    """One typed, directed, qualified hop from an object to related objects.

    ``qualifier=None`` means *any* qualifier, which is allowed but rarely what
    a check wants: the qualifier is the role the relation plays, and dropping
    it is how "the purchase order of this invoice" becomes "anything attached
    to this invoice".

    >>> PathStep("Invoice Receipt of Purchase Order", target_type="purchase_order").describe()
    '-[Invoice Receipt of Purchase Order]->purchase_order'
    """

    qualifier: str | None
    target_type: str
    direction: str = "forward"
    transitive: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_type", _name(self.target_type, "a path step's target_type"))
        if self.qualifier is not None:
            object.__setattr__(self, "qualifier", _name(self.qualifier, "a path step's qualifier"))
        if self.direction not in DIRECTIONS:
            raise OCUnitError(f"a path step's direction must be one of {DIRECTIONS}, got {self.direction!r}")
        object.__setattr__(self, "transitive", bool(self.transitive))

    def describe(self) -> str:
        arrow = "-[{}]->" if self.direction == "forward" else "<-[{}]-"
        label = "*" if self.qualifier is None else self.qualifier
        return arrow.format(label + ("+" if self.transitive else "")) + self.target_type

    def to_dict(self) -> dict[str, Any]:
        return {
            "qualifier": self.qualifier,
            "target_type": self.target_type,
            "direction": self.direction,
            "transitive": self.transitive,
        }


@dataclass(frozen=True)
class RolePath:
    """A named role and the ordered steps that bind objects to it."""

    role: str
    steps: tuple[PathStep, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _name(self.role, "a role name"))
        object.__setattr__(self, "steps", tuple(self.steps))
        if not self.steps:
            raise OCUnitError(f"role {self.role!r} declares no steps; a role with no path binds nothing")
        for step in self.steps:
            if not isinstance(step, PathStep):
                raise OCUnitError(f"role {self.role!r}: every step must be a PathStep, got {type(step).__name__}")

    @property
    def target_type(self) -> str:
        """The object type this role binds."""
        return self.steps[-1].target_type

    @property
    def min_hops(self) -> int:
        """The shortest number of hops the path can take."""
        return len(self.steps)

    def describe(self) -> str:
        return f"{self.role}: anchor" + "".join(step.describe() for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "steps": [s.to_dict() for s in self.steps], "describes": self.describe()}


@dataclass(frozen=True)
class TraversalLimits:
    """Hard boundaries on one unit's context. Every one of them is strict.

    >>> TraversalLimits().max_fan_out
    25
    """

    max_depth: int = 3
    max_fan_out: int = 25
    max_bindings: int = 100
    max_visited: int = 1000

    def __post_init__(self) -> None:
        for name in ("max_depth", "max_fan_out", "max_bindings", "max_visited"):
            value = int(getattr(self, name))
            if value < 1:
                raise OCUnitError(f"{name} must be at least 1, got {value}")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "max_fan_out": self.max_fan_out,
            "max_bindings": self.max_bindings,
            "max_visited": self.max_visited,
        }


@dataclass(frozen=True)
class UnitScope:
    """Which events count as observations of a unit.

    Distinct from :class:`wise.evidence.manifest.ObservationScope`, which is the
    *run's* window over a case log. This one answers "whose events am I looking
    at" for one object-centric unit: the anchor's, and those of the roles named
    here, optionally narrowed to some relation qualifiers, some activities and
    a time window.
    """

    roles: tuple[str, ...] = ()
    include_anchor: bool = True
    qualifiers: tuple[str, ...] | None = None
    activities: tuple[str, ...] | None = None
    window_start: pd.Timestamp | None = None
    window_end: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", tuple(_name(r, "a scope role") for r in self.roles))
        object.__setattr__(self, "include_anchor", bool(self.include_anchor))
        for name in ("qualifiers", "activities"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(str(v) for v in value))
        for name in ("window_start", "window_end"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _timestamp(value, f"scope {name}"))
        if not self.include_anchor and not self.roles:
            raise OCUnitError("an observation scope that excludes the anchor must name at least one role")

    def to_dict(self) -> dict[str, Any]:
        return {
            "roles": list(self.roles),
            "include_anchor": self.include_anchor,
            "qualifiers": None if self.qualifiers is None else list(self.qualifiers),
            "activities": None if self.activities is None else list(self.activities),
            "window_start": None if self.window_start is None else self.window_start.isoformat(),
            "window_end": None if self.window_end is None else self.window_end.isoformat(),
        }


@dataclass(frozen=True)
class UnitSpec:
    """The declaration of a unit type: anchor, roles, scope and limits.

    >>> spec = UnitSpec(
    ...     unit_type="invoice_review",
    ...     anchor_type="invoice",
    ...     roles=(RolePath("order", (PathStep("belongs to", target_type="purchase_order"),)),),
    ... )
    >>> spec.role_names
    ('order',)
    """

    unit_type: str
    anchor_type: str
    roles: tuple[RolePath, ...] = ()
    scope: UnitScope = field(default_factory=UnitScope)
    limits: TraversalLimits = field(default_factory=TraversalLimits)
    attribute_policy: AttributePolicy = AttributePolicy.AS_OF

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit_type", _name(self.unit_type, "unit_type"))
        object.__setattr__(self, "anchor_type", _name(self.anchor_type, "anchor_type"))
        object.__setattr__(self, "roles", tuple(self.roles))
        object.__setattr__(self, "attribute_policy", AttributePolicy(self.attribute_policy))
        names = [r.role for r in self.roles]
        if len(set(names)) != len(names):
            raise OCUnitError(f"role names must be unique inside a unit type, got {sorted(names)}")
        for role in self.roles:
            if role.min_hops > self.limits.max_depth:
                raise OCUnitError(
                    f"role {role.role!r} declares {role.min_hops} hops but max_depth is {self.limits.max_depth}; "
                    "a path that cannot be walked is a configuration error, not a truncated result"
                )
        unknown = [r for r in self.scope.roles if r not in set(names)]
        if unknown:
            raise OCUnitError(f"the observation scope names undeclared role(s) {unknown}; declared roles are {names}")

    @property
    def role_names(self) -> tuple[str, ...]:
        return tuple(r.role for r in self.roles)

    def role(self, name: str) -> RolePath:
        for candidate in self.roles:
            if candidate.role == name:
                return candidate
        raise OCUnitError(f"undeclared role {name!r}; this unit type declares {list(self.role_names)}")

    def check(self, log: OCEventLog) -> tuple[str, ...]:
        """Names in this spec that do not occur in ``log``.

        A path whose qualifier is absent from the log selects nothing, and
        "selected nothing" is indistinguishable from "there is nothing" once
        the result is a number. :func:`build_units` refuses such a spec by
        default rather than returning confidently empty roles.
        """
        object_types = set(log.object_types)
        qualifiers = set(log.qualifiers)
        problems: list[str] = []
        if self.anchor_type not in object_types:
            problems.append(f"anchor type {self.anchor_type!r} does not occur in the log")
        for role in self.roles:
            for position, step in enumerate(role.steps, start=1):
                if step.target_type not in object_types:
                    problems.append(f"role {role.role!r} step {position}: object type {step.target_type!r} does not occur")
                if step.qualifier is not None and step.qualifier not in qualifiers:
                    problems.append(f"role {role.role!r} step {position}: qualifier {step.qualifier!r} does not occur")
        for activity in self.scope.activities or ():
            if activity not in set(log.activities):
                problems.append(f"observation scope: activity {activity!r} does not occur")
        return tuple(problems)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_type": self.unit_type,
            "anchor_type": self.anchor_type,
            "roles": [r.to_dict() for r in self.roles],
            "scope": self.scope.to_dict(),
            "limits": self.limits.to_dict(),
            "attribute_policy": self.attribute_policy.value,
        }


# ------------------------------------------------------------------ witnesses
@dataclass(frozen=True)
class Hop:
    """One traversed relation, kept so a binding can be justified."""

    source_id: str
    direction: str
    qualifier: str
    target_id: str

    def describe(self) -> str:
        arrow = f"-[{self.qualifier}]->" if self.direction == "forward" else f"<-[{self.qualifier}]-"
        return f"{self.source_id} {arrow} {self.target_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "direction": self.direction,
            "qualifier": self.qualifier,
            "target_id": self.target_id,
        }


@dataclass(frozen=True)
class PathWitness:
    """How one object came to be bound to one role: the hops, in order."""

    role: str
    object_id: str
    object_type: str
    hops: tuple[Hop, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "hops", tuple(self.hops))

    @property
    def depth(self) -> int:
        return len(self.hops)

    def describe(self) -> str:
        return " ".join(hop.describe() for hop in self.hops)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "object_id": self.object_id,
            "object_type": self.object_type,
            "depth": self.depth,
            "hops": [h.to_dict() for h in self.hops],
            "describes": self.describe(),
        }


@dataclass(frozen=True)
class UnitTruncation:
    """Which limit bit, for which role, and how much was left out."""

    depth_limited: tuple[str, ...] = ()
    fan_out_limited: tuple[str, ...] = ()
    binding_limited: tuple[str, ...] = ()
    visit_limited: bool = False
    seen: dict[str, int] = field(default_factory=dict)
    kept: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("depth_limited", "fan_out_limited", "binding_limited"):
            object.__setattr__(self, name, tuple(str(r) for r in getattr(self, name)))
        object.__setattr__(self, "visit_limited", bool(self.visit_limited))
        object.__setattr__(self, "seen", {str(k): int(v) for k, v in self.seen.items()})
        object.__setattr__(self, "kept", {str(k): int(v) for k, v in self.kept.items()})

    @property
    def complete(self) -> bool:
        return not (self.depth_limited or self.fan_out_limited or self.binding_limited or self.visit_limited)

    @property
    def limited_roles(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.depth_limited) | set(self.fan_out_limited) | set(self.binding_limited)))

    @property
    def reasons(self) -> tuple[str, ...]:
        out = []
        if self.depth_limited:
            out.append(f"max_depth reached for role(s) {list(self.depth_limited)}")
        if self.fan_out_limited:
            out.append(f"max_fan_out reached for role(s) {list(self.fan_out_limited)}")
        if self.binding_limited:
            out.append(f"max_bindings reached for role(s) {list(self.binding_limited)}")
        if self.visit_limited:
            out.append("max_visited reached for the unit as a whole")
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "depth_limited": list(self.depth_limited),
            "fan_out_limited": list(self.fan_out_limited),
            "binding_limited": list(self.binding_limited),
            "visit_limited": self.visit_limited,
            "seen": dict(self.seen),
            "kept": dict(self.kept),
            "reasons": list(self.reasons),
        }


# ----------------------------------------------------------------------- unit
@dataclass(frozen=True)
class AssessmentUnit:
    """One bounded, typed context: an anchor, its roles, and the events in scope.

    The unit is a *statement about what was looked at*. It carries the witness
    for every binding, the evaluation instant its attribute reads use, and
    :attr:`complete` — which is ``False`` whenever a limit bit, so that no
    downstream number can be reported as final without noticing.
    """

    unit_type: str
    unit_id: str
    anchor_id: str
    anchor_type: str
    bindings: dict[str, tuple[str, ...]] = field(default_factory=dict)
    witnesses: tuple[PathWitness, ...] = ()
    event_ids: tuple[str, ...] = ()
    evaluation_time: pd.Timestamp | None = None
    scope: UnitScope = field(default_factory=UnitScope)
    limits: TraversalLimits = field(default_factory=TraversalLimits)
    truncation: UnitTruncation = field(default_factory=UnitTruncation)
    qualifications: tuple[Qualification, ...] = ()
    attribute_policy: AttributePolicy = AttributePolicy.AS_OF

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", {str(k): tuple(str(v) for v in value) for k, value in self.bindings.items()})
        object.__setattr__(self, "witnesses", tuple(self.witnesses))
        object.__setattr__(self, "event_ids", tuple(str(e) for e in self.event_ids))
        object.__setattr__(self, "qualifications", tuple(self.qualifications))
        object.__setattr__(self, "attribute_policy", AttributePolicy(self.attribute_policy))

    def __repr__(self) -> str:
        state = "complete" if self.complete else f"truncated ({'; '.join(self.truncation.reasons)})"
        bound = ", ".join(f"{role}={len(ids)}" for role, ids in self.bindings.items())
        return f"AssessmentUnit({self.unit_id!r}, {bound or 'no roles'}, {len(self.event_ids)} events, {state})"

    @property
    def complete(self) -> bool:
        """Whether the declared context was gathered without hitting a limit."""
        return self.truncation.complete

    @property
    def object_ids(self) -> tuple[str, ...]:
        """The anchor and every bound object, deduplicated and sorted."""
        ids = {self.anchor_id}
        for bound in self.bindings.values():
            ids.update(bound)
        return tuple(sorted(ids))

    @property
    def unbound_roles(self) -> tuple[str, ...]:
        """Roles that bound no object at all. Not an absence — a fact about scope."""
        return tuple(role for role, ids in self.bindings.items() if not ids)

    def role(self, name: str) -> tuple[str, ...]:
        """The objects bound to one role."""
        try:
            return self.bindings[name]
        except KeyError:
            raise OCUnitError(f"undeclared role {name!r}; this unit binds {sorted(self.bindings)}") from None

    def witnesses_for(self, role: str) -> tuple[PathWitness, ...]:
        """The subgraph witnesses that justify one role's bindings."""
        self.role(role)
        return tuple(w for w in self.witnesses if w.role == role)

    def require_complete(self) -> AssessmentUnit:
        """Return the unit, or refuse if its context was truncated.

        For the callers that must not present a bounded partial context as a
        finished assessment. It raises rather than returning a flag, because a
        flag can be forgotten and an exception cannot.
        """
        if not self.complete:
            raise OCUnitError(
                f"{self.unit_id}: the context is truncated ({'; '.join(self.truncation.reasons)}), so it cannot be "
                "presented as a complete assessment; raise the limits deliberately or report the result as partial"
            )
        return self

    def events(self, log: OCEventLog) -> tuple[OCEvent, ...]:
        """The events in scope, in ``(timestamp, event_id)`` order.

        Two events with the same activity and the same timestamp are two
        events here, and both are returned.
        """
        events = [log.event(event_id) for event_id in self.event_ids]
        return tuple(sorted(events, key=lambda e: (e.timestamp, e.event_id)))

    def attribute(
        self,
        log: OCEventLog,
        attribute: str,
        *,
        object_id: str | None = None,
        policy: AttributePolicy | str | None = None,
    ) -> AttributeReading:
        """Read an object attribute **at this unit's evaluation time**.

        Defaults to the anchor. A unit built without an evaluation time cannot
        answer this under the ``as_of`` policy and says so instead of
        substituting the first or the latest value.
        """
        chosen = self.attribute_policy if policy is None else AttributePolicy(policy)
        if self.evaluation_time is None and chosen is AttributePolicy.AS_OF:
            raise OCUnitError(
                f"{self.unit_id}: this unit was built without an evaluation time, so an attribute cannot be read as of "
                "one; rebuild the units with build_units(..., at=...), or ask for policy='latest_known' explicitly"
            )
        target = self.anchor_id if object_id is None else str(object_id)
        if target not in self.object_ids:
            raise OCUnitError(f"{self.unit_id}: object {target!r} is not part of this unit; it binds {list(self.object_ids)}")
        return log.value_at(target, attribute, self.evaluation_time, policy=chosen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_type": self.unit_type,
            "unit_id": self.unit_id,
            "anchor_id": self.anchor_id,
            "anchor_type": self.anchor_type,
            "bindings": {role: list(ids) for role, ids in self.bindings.items()},
            "unbound_roles": list(self.unbound_roles),
            "witnesses": [w.to_dict() for w in self.witnesses],
            "event_ids": list(self.event_ids),
            "n_events": len(self.event_ids),
            "evaluation_time": None if self.evaluation_time is None else self.evaluation_time.isoformat(),
            "attribute_policy": self.attribute_policy.value,
            "scope": self.scope.to_dict(),
            "limits": self.limits.to_dict(),
            "truncation": self.truncation.to_dict(),
            "complete": self.complete,
            "qualifications": [q.to_dict() for q in self.qualifications],
        }


# -------------------------------------------------------------------- builder
def build_units(
    log: OCEventLog,
    spec: UnitSpec,
    *,
    at: pd.Timestamp | str | None = None,
    anchors: Iterable[str] | None = None,
    strict: bool = True,
) -> tuple[AssessmentUnit, ...]:
    """Evaluate a :class:`UnitSpec` against a log into bounded units.

    One unit per anchor object of the declared type, in canonical id order.
    ``at`` is the evaluation instant every attribute read will use; ``anchors``
    restricts the build to named objects; ``strict`` (the default) refuses a
    spec whose types, qualifiers or activities do not occur in the log, because
    a path that selects nothing looks exactly like an absence once it becomes a
    number.

    >>> import pandas as pd
    >>> from wise.oc.model import E2O, O2O, OCEvent, OCEventLog, OCObject
    >>> log = OCEventLog.build(
    ...     events=[OCEvent("e1", "Create Invoice", pd.Timestamp("2024-01-02T09:00:00Z"))],
    ...     objects=[OCObject("i1", "invoice"), OCObject("p1", "purchase_order")],
    ...     e2o=[E2O("e1", "i1", "invoice")],
    ...     o2o=[O2O("i1", "p1", "belongs to")],
    ... )
    >>> spec = UnitSpec(
    ...     unit_type="invoice_review",
    ...     anchor_type="invoice",
    ...     roles=(RolePath("order", (PathStep("belongs to", target_type="purchase_order"),)),),
    ... )
    >>> units = build_units(log, spec, at="2024-06-30T00:00:00Z")
    >>> units[0].role("order"), units[0].complete
    (('p1',), True)
    >>> units[0].witnesses_for("order")[0].describe()
    'i1 -[belongs to]-> p1'
    """
    if strict:
        problems = spec.check(log)
        if problems:
            raise OCUnitError(
                f"unit type {spec.unit_type!r} cannot be evaluated against this log: "
                + "; ".join(problems)
                + ". A path that selects nothing is not evidence of absence; fix the spec, or pass strict=False "
                "if the empty selection is genuinely intended."
            )
    moment = None if at is None else _timestamp(at, "evaluation time")
    if anchors is None:
        candidates = sorted(o.object_id for o in log.objects if o.object_type == spec.anchor_type)
    else:
        candidates = []
        for anchor in anchors:
            obj = log.obj(anchor)
            if obj.object_type != spec.anchor_type:
                raise OCUnitError(
                    f"anchor {obj.object_id!r} is of type {obj.object_type!r}, not the declared {spec.anchor_type!r}"
                )
            candidates.append(obj.object_id)
    return tuple(_build_one(log, spec, anchor, moment) for anchor in candidates)


def _build_one(log: OCEventLog, spec: UnitSpec, anchor: str, moment: pd.Timestamp | None) -> AssessmentUnit:
    bindings: dict[str, tuple[str, ...]] = {}
    witnesses: list[PathWitness] = []
    depth_limited: list[str] = []
    fan_out_limited: list[str] = []
    binding_limited: list[str] = []
    seen: dict[str, int] = {}
    kept: dict[str, int] = {}
    touched: set[str] = {anchor}
    visit_limited = False

    at_filter = moment if log.source.relation_time_semantics == "interval" else None
    for role in spec.roles:
        found, role_flags = _walk(log, spec.limits, anchor, role, touched, at_filter)
        seen[role.role] = role_flags["seen"]
        if role_flags["depth"]:
            depth_limited.append(role.role)
        if role_flags["fan_out"]:
            fan_out_limited.append(role.role)
        if role_flags["visited"]:
            visit_limited = True
        ordered = sorted(found)
        if len(ordered) > spec.limits.max_bindings:
            binding_limited.append(role.role)
            ordered = ordered[: spec.limits.max_bindings]
        bindings[role.role] = tuple(ordered)
        kept[role.role] = len(ordered)
        witnesses.extend(
            PathWitness(role=role.role, object_id=oid, object_type=log.obj(oid).object_type, hops=found[oid]) for oid in ordered
        )
        touched.update(ordered)

    truncation = UnitTruncation(
        depth_limited=tuple(depth_limited),
        fan_out_limited=tuple(fan_out_limited),
        binding_limited=tuple(binding_limited),
        visit_limited=visit_limited,
        seen=seen,
        kept=kept,
    )
    event_ids = _events_in_scope(log, spec, anchor, bindings)
    return AssessmentUnit(
        unit_type=spec.unit_type,
        unit_id=f"{spec.unit_type}:{anchor}",
        anchor_id=anchor,
        anchor_type=spec.anchor_type,
        bindings=bindings,
        witnesses=tuple(witnesses),
        event_ids=event_ids,
        evaluation_time=moment,
        scope=spec.scope,
        limits=spec.limits,
        truncation=truncation,
        qualifications=_qualify(log, spec, f"{spec.unit_type}:{anchor}", truncation, moment),
        attribute_policy=spec.attribute_policy,
    )


def _walk(
    log: OCEventLog,
    limits: TraversalLimits,
    anchor: str,
    role: RolePath,
    touched: set[str],
    at_filter: pd.Timestamp | None,
) -> tuple[dict[str, tuple[Hop, ...]], dict[str, Any]]:
    """Walk one role's path from the anchor, bounded at every step.

    Returns the objects the last step reached, with the hops that reached them,
    and the flags saying which limits bit. The anchor itself is never bound to
    a role: a cycle that leads back to it is a cycle, not a related object.
    """
    frontier: dict[str, tuple[Hop, ...]] = {anchor: ()}
    visited: set[str] = {anchor}
    flags: dict[str, Any] = {"depth": False, "fan_out": False, "visited": False, "seen": 0}
    hops_used = 0
    reached: dict[str, tuple[Hop, ...]] = {}

    for step in role.steps:
        remaining = limits.max_depth - hops_used
        rounds = remaining if step.transitive else min(1, remaining)
        if rounds < 1:
            # the depth budget ran out before this step could be walked at all;
            # an empty role here is a limit, not an absence, and says so
            flags["depth"] = True
            return {}, flags
        collected: dict[str, tuple[Hop, ...]] = {}
        for _ in range(rounds):
            nxt = _expand(log, step, frontier, visited, limits, flags, touched, at_filter)
            hops_used += 1
            visited.update(nxt)
            collected.update(nxt)
            frontier = nxt
            if not step.transitive or not nxt or flags["visited"]:
                break
        if step.transitive and frontier and hops_used >= limits.max_depth:
            flags["depth"] = True  # the frontier was still growing when the budget ran out
        frontier = collected
        reached = collected
        if not frontier:
            break
    return reached, flags


def _expand(
    log: OCEventLog,
    step: PathStep,
    frontier: Mapping[str, tuple[Hop, ...]],
    visited: set[str],
    limits: TraversalLimits,
    flags: dict[str, Any],
    touched: set[str],
    at_filter: pd.Timestamp | None,
) -> dict[str, tuple[Hop, ...]]:
    """One hop from every object in the frontier, fan-out and visits bounded.

    ``at_filter`` is set only when the source actually supplies relation
    validity intervals; relations outside the half-open interval
    ``[valid_from, valid_to)`` are then not followed. For an atemporal source
    it is ``None`` and nothing is filtered, because there is no interval to
    filter by and inventing one is exactly what this package refuses to do.
    """
    far = "target_id" if step.direction == "forward" else "source_id"
    nxt: dict[str, tuple[Hop, ...]] = {}
    for source in sorted(frontier):
        by_target: dict[str, Any] = {}
        for relation in log.related(source, direction=step.direction, qualifier=step.qualifier, target_type=step.target_type):
            if not _valid_at(relation, at_filter):
                continue
            by_target.setdefault(str(getattr(relation, far)), relation)
        targets = sorted(by_target.items())
        flags["seen"] = int(flags["seen"]) + len(targets)
        if len(targets) > limits.max_fan_out:
            flags["fan_out"] = True
            targets = targets[: limits.max_fan_out]
        for target_id, relation in targets:
            if target_id in visited or target_id in nxt:
                continue
            if len(touched) >= limits.max_visited:
                flags["visited"] = True
                return nxt
            touched.add(target_id)
            nxt[target_id] = (*frontier[source], Hop(source, step.direction, relation.qualifier, target_id))
    return nxt


def _valid_at(relation: O2O, at_filter: pd.Timestamp | None) -> bool:
    """Whether a relation is in force at an instant, on a half-open interval."""
    if at_filter is None:
        return True
    if relation.valid_from is not None and relation.valid_from > at_filter:
        return False
    return not (relation.valid_to is not None and relation.valid_to <= at_filter)


def _events_in_scope(log: OCEventLog, spec: UnitSpec, anchor: str, bindings: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    scope = spec.scope
    sources: list[str] = [anchor] if scope.include_anchor else []
    for role in scope.roles or ():
        sources.extend(bindings.get(role, ()))
    picked: dict[str, OCEvent] = {}
    for object_id in dict.fromkeys(sources):
        for qualifier in scope.qualifiers or (None,):
            for event in log.events_of(object_id, qualifier=qualifier, since=scope.window_start, until=scope.window_end):
                if scope.activities is not None and event.activity not in scope.activities:
                    continue
                picked[event.event_id] = event
    return tuple(e.event_id for e in sorted(picked.values(), key=lambda e: (e.timestamp, e.event_id)))


def _qualify(
    log: OCEventLog, spec: UnitSpec, unit_id: str, truncation: UnitTruncation, moment: pd.Timestamp | None
) -> tuple[Qualification, ...]:
    quals: list[Qualification] = []
    if not truncation.complete:
        quals.append(
            Qualification(
                QualificationCode.CONTEXT_TRUNCATED,
                f"{unit_id}: the declared context was cut ({'; '.join(truncation.reasons)}), so this unit is a bounded "
                "partial view and any count taken from it is a lower bound, not a complete assessment",
                scope="unit",
            )
        )
    for roles, code in (
        (truncation.depth_limited, QualificationCode.DEPTH_LIMIT_REACHED),
        (truncation.fan_out_limited, QualificationCode.FAN_OUT_LIMIT_REACHED),
        (truncation.binding_limited, QualificationCode.BINDING_LIMIT_REACHED),
    ):
        for role_name in roles:
            quals.append(
                Qualification(
                    code,
                    f"{unit_id}: role {role_name!r} reached the {code.value.removesuffix('_reached')} limit; "
                    f"{truncation.seen.get(role_name, 0)} related object(s) were seen and "
                    f"{truncation.kept.get(role_name, 0)} kept",
                    scope="unit",
                )
            )
    if spec.roles and log.source.relation_time_semantics == "atemporal" and moment is not None:
        quals.append(
            Qualification(
                QualificationCode.RELATION_VALIDITY_UNKNOWN,
                f"{unit_id}: the source records that these objects are related, not when; the bindings are therefore "
                f"not filtered to {moment.isoformat()}, and no validity interval has been inferred",
                scope="unit",
            )
        )
    elif spec.roles and log.source.relation_time_semantics == "interval" and moment is None:
        quals.append(
            Qualification(
                QualificationCode.RELATION_VALIDITY_UNKNOWN,
                f"{unit_id}: the source supplies relation validity intervals, but this unit was built without an "
                "evaluation time, so no interval filter was applied and the bindings span every interval on record",
                scope="unit",
            )
        )
    return tuple(quals)


def context_report(units: Sequence[AssessmentUnit]) -> dict[str, Any]:
    """How complete a set of units is, and where the limits bit.

    The number a reviewer needs before reading any aggregate over these units:
    a backlog computed over 900 complete and 100 truncated contexts is not a
    backlog over 1000 contexts.
    """
    reasons: dict[str, int] = {}
    bound: dict[str, int] = {}
    for unit in units:
        for reason in unit.truncation.reasons:
            key = reason.split(" reached")[0]
            reasons[key] = reasons.get(key, 0) + 1
        for role, ids in unit.bindings.items():
            bound[role] = bound.get(role, 0) + len(ids)
    complete = sum(1 for u in units if u.complete)
    return {
        "n_units": len(units),
        "n_complete": complete,
        "n_truncated": len(units) - complete,
        "complete_share": None if not units else complete / len(units),
        "truncation_reasons": reasons,
        "bindings_per_role": bound,
        "unit_types": sorted({u.unit_type for u in units}),
    }


__all__ = [
    "DIRECTIONS",
    "AssessmentUnit",
    "Hop",
    "PathStep",
    "PathWitness",
    "RolePath",
    "TraversalLimits",
    "UnitScope",
    "UnitSpec",
    "UnitTruncation",
    "build_units",
    "context_report",
]
