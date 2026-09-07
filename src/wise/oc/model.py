"""Native object-centric data: identities, qualified relations, histories.

A case-based :class:`~wise.log.EventLog` answers *what happened to this case*.
An :class:`OCEventLog` answers three questions a flattened log cannot:

* **which object** an event touched, and **in which role** — the qualifier;
* **how objects relate to each other**, directed and qualified;
* **what an object's attribute was at a given moment**, not merely what its
  first or last recorded value happens to be.

The contract is deliberately narrow and versioned (:data:`OC_SCHEMA_VERSION`).
It holds immutable, indexed tables and no graph database, and every accessor
is a bounded lookup on one of those indices.

Four rules are enforced rather than documented:

1. **Identity is never manufactured.** Two rows claiming the same event id
   with different content are a conflict and raise; they are not merged, and
   the later one does not win. Conversely, two *different* events that share
   an activity and a timestamp are two events — nothing here collapses them.
2. **Identical rows may be deduplicated, with a report.** A relation table
   that repeats one triple loses the repeat and gains a
   :class:`ValidationIssue` saying so.
3. **A link that points nowhere is not silently dropped.** Dangling relations
   raise under the default policy and are reported when the caller explicitly
   asks for them to be dropped.
4. **An attribute is read at a declared time.** :meth:`OCEventLog.value_at`
   takes the instant as an argument and returns the value in force then. It
   never falls back to "the first non-null value", and when there is no value
   yet it says so with :attr:`AttributeReading.found` — a missing value is
   not an empty string and not the next value that happens to arrive later.

Relation time is the one place where the source's own semantics decide. OCEL
2.0 object-to-object relations are *atemporal*: the standard records that an
invoice belongs to a purchase order, not the interval during which it did.
:attr:`SourceMetadata.relation_time_semantics` therefore stays ``"atemporal"``
for such a source, an interval on a relation from such a source is refused,
and any unit built at an evaluation time carries
:data:`~wise.evidence.models.QualificationCode.RELATION_VALIDITY_UNKNOWN`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from .._version import __version__
from ..errors import OCValidationError
from ..evidence.models import Qualification, QualificationCode

#: Version of the native object-log contract. Independent of the norm schema
#: version and of the evidence packet version: object data is a separate
#: contract, not a new field on an old one.
OC_SCHEMA_VERSION = "wise-oc/1"

#: What a source says about the validity in time of its object-to-object
#: relations. ``"atemporal"`` means the source states *that* two objects are
#: related and nothing about *when*; ``"interval"`` means the source supplies
#: effective intervals and they are stored as given.
RELATION_TIME_SEMANTICS = ("atemporal", "interval")


# --------------------------------------------------------------------- helpers
def _text(value: Any, what: str) -> str:
    out = str(value)
    if not out:
        raise OCValidationError(f"{what} must be a non-empty string")
    return out


def _timestamp(value: Any, what: str) -> pd.Timestamp:
    try:
        ts = value if isinstance(value, pd.Timestamp) else pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise OCValidationError(f"{what} must be a timestamp, got {value!r}") from exc
    if ts is pd.NaT or pd.isna(ts):
        raise OCValidationError(f"{what} must be a real timestamp, not a missing one")
    return ts


def _scalar(value: Any, what: str) -> Any:
    """One attribute value: a scalar, a timestamp, or nothing.

    NumPy and pandas scalars — what a table hands over — are converted to their
    Python equivalents, and every flavour of missing becomes ``None``. A nested
    structure is refused rather than stringified: an attribute whose value is a
    list of invoices is a relation in disguise, and this contract keeps
    relations in their own table.
    """
    if isinstance(value, np.datetime64 | datetime):
        return pd.Timestamp(value)
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, str | bool | int | float | pd.Timestamp):
        return value
    raise OCValidationError(f"{what}: attribute values must be scalars or timestamps, got {type(value).__name__}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


def _digest(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_PRECISION_STEPS: tuple[tuple[str, int], ...] = (
    ("day", 86_400_000_000_000),
    ("hour", 3_600_000_000_000),
    ("minute", 60_000_000_000),
    ("second", 1_000_000_000),
    ("millisecond", 1_000_000),
    ("microsecond", 1_000),
    ("nanosecond", 1),
)


def observed_precision(timestamps: Iterable[pd.Timestamp]) -> str:
    """The coarsest grid every timestamp sits on, measured against the UTC epoch.

    This is what the data *shows*, not what the source system claims. A log
    whose timestamps are all whole minutes cannot support a check that argues
    about seconds, and this is the fact that says so.

    >>> observed_precision([pd.Timestamp("2024-01-01T10:30:00Z"), pd.Timestamp("2024-01-02T11:00:00Z")])
    'minute'
    >>> observed_precision([])
    'none'
    """
    values = [int(ts.value) for ts in timestamps]
    if not values:
        return "none"
    for name, step in _PRECISION_STEPS:
        if all(value % step == 0 for value in values):
            return name
    return "nanosecond"  # pragma: no cover - the last step divides every integer


# ------------------------------------------------------------------ validation
class IssueSeverity(str, Enum):
    """How a validation issue was resolved."""

    #: The log could not be built without inventing a fact. Always raised.
    ERROR = "error"
    #: An identical duplicate was removed, or a dangling row dropped on request.
    REPAIRED = "repaired"
    #: Information an export cannot carry, listed before it is lost.
    DROPPED = "dropped"
    #: Worth knowing, changes nothing: an empty qualifier, a self-relation.
    INFO = "info"


class OCIssueCode(str, Enum):
    """Named findings of :func:`validate`, the readers and the writers."""

    DUPLICATE_EVENT_ROW = "duplicate_event_row"
    DUPLICATE_OBJECT_ROW = "duplicate_object_row"
    DUPLICATE_E2O_ROW = "duplicate_e2o_row"
    DUPLICATE_O2O_ROW = "duplicate_o2o_row"
    DUPLICATE_ATTRIBUTE_ROW = "duplicate_attribute_row"
    CONFLICTING_EVENT_ID = "conflicting_event_id"
    CONFLICTING_OBJECT_ID = "conflicting_object_id"
    CONFLICTING_ATTRIBUTE_VALUE = "conflicting_attribute_value"
    DANGLING_E2O_EVENT = "dangling_e2o_event"
    DANGLING_E2O_OBJECT = "dangling_e2o_object"
    DANGLING_O2O_SOURCE = "dangling_o2o_source"
    DANGLING_O2O_TARGET = "dangling_o2o_target"
    DANGLING_ATTRIBUTE_OBJECT = "dangling_attribute_object"
    MISSING_QUALIFIER = "missing_qualifier"
    SELF_RELATION = "self_relation"
    MIXED_TIMEZONES = "mixed_timezones"
    UNDECLARED_RELATION_INTERVAL = "undeclared_relation_interval"
    UNSUPPORTED_FIELD = "unsupported_field"
    UNDECLARED_TYPE = "undeclared_type"
    NOT_REPRESENTABLE = "not_representable"


@dataclass(frozen=True)
class ValidationIssue:
    """One finding, with how many rows it covers and a few of them by name."""

    code: OCIssueCode
    severity: IssueSeverity
    message: str
    count: int = 1
    examples: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", OCIssueCode(self.code))
        object.__setattr__(self, "severity", IssueSeverity(self.severity))
        object.__setattr__(self, "message", _text(self.message, "issue message"))
        object.__setattr__(self, "count", int(self.count))
        object.__setattr__(self, "examples", tuple(str(e) for e in self.examples))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "count": self.count,
            "examples": list(self.examples),
        }


@dataclass(frozen=True)
class ValidationReport:
    """Everything :func:`validate` found, kept with the log it describes.

    A clean log has an empty report. A log built from a source that repeated
    two relation rows has a report with one ``duplicate_e2o_row`` issue and is
    still a usable log — the point of :attr:`repaired` is that the repair is
    visible rather than assumed.
    """

    issues: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(self.issues))

    def __len__(self) -> int:
        return len(self.issues)

    def __bool__(self) -> bool:  # an empty report is falsey, a clean log is not "no report"
        return bool(self.issues)

    @property
    def clean(self) -> bool:
        """No repair, no drop, no error — only informational findings, if any."""
        return not any(i.severity in (IssueSeverity.ERROR, IssueSeverity.REPAIRED, IssueSeverity.DROPPED) for i in self.issues)

    @property
    def repaired(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is IssueSeverity.REPAIRED)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is IssueSeverity.ERROR)

    def of(self, code: OCIssueCode | str) -> tuple[ValidationIssue, ...]:
        """Every issue with one code."""
        wanted = OCIssueCode(code)
        return tuple(i for i in self.issues if i.code is wanted)

    def count(self, code: OCIssueCode | str) -> int:
        """How many rows one code covers, summed over its issues."""
        return sum(i.count for i in self.of(code))

    def to_dict(self) -> dict[str, Any]:
        return {"issues": [i.to_dict() for i in self.issues], "clean": self.clean}


# ----------------------------------------------------------------- provenance
@dataclass(frozen=True)
class SourceMetadata:
    """Where a log came from, how it was read, and what its times can support.

    Nothing in this record is discovered by magic: :attr:`source` is what the
    caller passed, :attr:`digest` is filled only when the reader actually
    hashed the bytes it read, and there is no wall-clock field, so two reads of
    one file produce one identity.
    """

    interchange: str
    spec: str = ""
    source: str | None = None
    reader: str = ""
    library_version: str = __version__
    schema_version: str = OC_SCHEMA_VERSION
    options: dict[str, Any] = field(default_factory=dict)
    relation_time_semantics: str = "atemporal"
    timezone: str | None = None
    precision: dict[str, str] = field(default_factory=dict)
    digest: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "interchange", _text(self.interchange, "interchange"))
        object.__setattr__(self, "options", dict(self.options))
        object.__setattr__(self, "precision", {str(k): str(v) for k, v in self.precision.items()})
        object.__setattr__(self, "notes", tuple(str(n) for n in self.notes))
        if self.relation_time_semantics not in RELATION_TIME_SEMANTICS:
            raise OCValidationError(
                f"relation_time_semantics must be one of {RELATION_TIME_SEMANTICS}, got {self.relation_time_semantics!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "interchange": self.interchange,
            "spec": self.spec,
            "source": self.source,
            "reader": self.reader,
            "library_version": self.library_version,
            "schema_version": self.schema_version,
            "options": dict(self.options),
            "relation_time_semantics": self.relation_time_semantics,
            "timezone": self.timezone,
            "precision": dict(self.precision),
            "digest": self.digest,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------- tables
@dataclass(frozen=True)
class OCEvent:
    """One event, with its own identity. Never derived from its labels."""

    event_id: str
    activity: str
    timestamp: pd.Timestamp
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _text(self.event_id, "event_id"))
        object.__setattr__(self, "activity", _text(self.activity, "activity"))
        object.__setattr__(self, "timestamp", _timestamp(self.timestamp, f"event {self.event_id} timestamp"))
        object.__setattr__(
            self,
            "attributes",
            {str(k): _scalar(v, f"event {self.event_id}") for k, v in dict(self.attributes).items()},
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "activity": self.activity,
            "timestamp": self.timestamp,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class OCObject:
    """One object: a canonical id and a type. Attributes live in the history."""

    object_id: str
    object_type: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_id", _text(self.object_id, "object_id"))
        object.__setattr__(self, "object_type", _text(self.object_type, "object_type"))

    def to_row(self) -> dict[str, Any]:
        return {"object_id": self.object_id, "object_type": self.object_type}


@dataclass(frozen=True)
class E2O:
    """An event touching an object **in a role**: the qualifier is the role."""

    event_id: str
    object_id: str
    qualifier: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _text(self.event_id, "e2o event_id"))
        object.__setattr__(self, "object_id", _text(self.object_id, "e2o object_id"))
        object.__setattr__(self, "qualifier", str(self.qualifier))

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.event_id, self.object_id, self.qualifier)

    def to_row(self) -> dict[str, Any]:
        return {"event_id": self.event_id, "object_id": self.object_id, "qualifier": self.qualifier}


@dataclass(frozen=True)
class O2O:
    """A directed, qualified relation between two objects.

    ``valid_from`` and ``valid_to`` exist for sources that actually supply
    effective intervals. They must stay ``None`` under
    ``relation_time_semantics="atemporal"``: an interval nobody recorded is
    not an interval this library will invent.
    """

    source_id: str
    target_id: str
    qualifier: str = ""
    valid_from: pd.Timestamp | None = None
    valid_to: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _text(self.source_id, "o2o source_id"))
        object.__setattr__(self, "target_id", _text(self.target_id, "o2o target_id"))
        object.__setattr__(self, "qualifier", str(self.qualifier))
        for name in ("valid_from", "valid_to"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _timestamp(value, f"o2o {name}"))
        if self.valid_from is not None and self.valid_to is not None and self.valid_to < self.valid_from:
            raise OCValidationError(f"o2o {self.source_id}->{self.target_id}: valid_to precedes valid_from")

    @property
    def timed(self) -> bool:
        return self.valid_from is not None or self.valid_to is not None

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (
            self.source_id,
            self.target_id,
            self.qualifier,
            "" if self.valid_from is None else self.valid_from.isoformat(),
            "" if self.valid_to is None else self.valid_to.isoformat(),
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "qualifier": self.qualifier,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
        }


@dataclass(frozen=True)
class AttributeChange:
    """One value of one object attribute, effective from one instant."""

    object_id: str
    attribute: str
    timestamp: pd.Timestamp
    value: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_id", _text(self.object_id, "attribute change object_id"))
        object.__setattr__(self, "attribute", _text(self.attribute, "attribute name"))
        object.__setattr__(self, "timestamp", _timestamp(self.timestamp, f"attribute {self.attribute} timestamp"))
        object.__setattr__(self, "value", _scalar(self.value, f"object {self.object_id} attribute {self.attribute}"))

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.object_id, self.attribute, self.timestamp.isoformat())

    def to_row(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "attribute": self.attribute,
            "timestamp": self.timestamp,
            "value": self.value,
        }


# ------------------------------------------------------------------- readings
class AttributePolicy(str, Enum):
    """How :meth:`OCEventLog.value_at` resolves a changing attribute.

    There is no "first non-null" policy, and there never will be: a value that
    first appears after the moment being assessed is not evidence about that
    moment.
    """

    #: The last change at or before the declared instant. The default.
    AS_OF = "as_of"
    #: The last change in the whole history, whenever it happened. Only ever
    #: applied on request, and always carries a qualification when the value
    #: comes from after the declared instant.
    LATEST_KNOWN = "latest_known"


@dataclass(frozen=True)
class AttributeReading:
    """The answer to "what was this attribute at that moment", with its evidence.

    :attr:`found` is the honest part. ``found=False`` means the history has no
    value in force at :attr:`at` — not that the attribute is empty, and not
    that some later value applies.
    """

    object_id: str
    attribute: str
    at: pd.Timestamp | None
    policy: AttributePolicy
    value: Any = None
    found: bool = False
    effective_from: pd.Timestamp | None = None
    n_changes: int = 0
    n_at_or_before: int = 0
    qualifications: tuple[Qualification, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", AttributePolicy(self.policy))
        object.__setattr__(self, "qualifications", tuple(self.qualifications))
        if not self.found and self.value is not None:
            raise OCValidationError(f"{self.object_id}.{self.attribute}: a reading that found nothing cannot carry a value")

    @property
    def later_changes(self) -> int:
        """Changes recorded after the declared instant — a hint, never a value."""
        return max(0, self.n_changes - self.n_at_or_before)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "attribute": self.attribute,
            "at": None if self.at is None else self.at.isoformat(),
            "policy": self.policy.value,
            "value": _jsonable(self.value),
            "found": self.found,
            "effective_from": None if self.effective_from is None else self.effective_from.isoformat(),
            "n_changes": self.n_changes,
            "n_at_or_before": self.n_at_or_before,
            "later_changes": self.later_changes,
            "qualifications": [q.to_dict() for q in self.qualifications],
        }


# ------------------------------------------------------------------------ log
@dataclass(frozen=True)
class OCEventLog:
    """A versioned, immutable object-centric log with bounded indexed access.

    Build one with :meth:`build`, which validates and repairs, or with a reader
    from :mod:`wise.oc.io`. Constructing the dataclass directly is allowed but
    unforgiving: it re-checks every structural invariant and raises instead of
    repairing, so a log that exists is a log that holds together.

    >>> log = OCEventLog.build(
    ...     events=[OCEvent("e1", "Create Invoice", pd.Timestamp("2024-01-02T09:00:00Z"))],
    ...     objects=[OCObject("i1", "invoice"), OCObject("p1", "purchase_order")],
    ...     e2o=[E2O("e1", "i1", "invoice")],
    ...     o2o=[O2O("i1", "p1", "belongs to")],
    ... )
    >>> log.summary()["n_events"], log.summary()["object_types"]["invoice"]
    (1, 1)
    >>> [r.qualifier for r in log.related("i1")]
    ['belongs to']
    """

    events: tuple[OCEvent, ...]
    objects: tuple[OCObject, ...]
    e2o: tuple[E2O, ...] = ()
    o2o: tuple[O2O, ...] = ()
    attribute_history: tuple[AttributeChange, ...] = ()
    source: SourceMetadata = field(default_factory=lambda: SourceMetadata(interchange="in-memory"))
    validation: ValidationReport = field(default_factory=ValidationReport)
    #: Declared attribute name -> declared value type, per object type. Kept as
    #: the source declared it so an export can be exact and a consumer can
    #: coerce deliberately; the reader never coerces a value on its own.
    object_type_attributes: dict[str, dict[str, str]] = field(default_factory=dict)
    #: Declared attribute name -> declared value type, per event type.
    event_type_attributes: dict[str, dict[str, str]] = field(default_factory=dict)
    schema_version: str = OC_SCHEMA_VERSION

    _by_event: dict[str, OCEvent] = field(default_factory=dict, init=False, repr=False, compare=False)
    _by_object: dict[str, OCObject] = field(default_factory=dict, init=False, repr=False, compare=False)
    _e2o_by_event: dict[str, tuple[E2O, ...]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _e2o_by_object: dict[str, tuple[E2O, ...]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _o2o_out: dict[str, tuple[O2O, ...]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _o2o_in: dict[str, tuple[O2O, ...]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _history: dict[tuple[str, str], tuple[AttributeChange, ...]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    # ------------------------------------------------------------ construction
    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "e2o", tuple(self.e2o))
        object.__setattr__(self, "o2o", tuple(self.o2o))
        object.__setattr__(self, "attribute_history", tuple(self.attribute_history))
        for name in ("object_type_attributes", "event_type_attributes"):
            declared = {str(k): {str(a): str(t) for a, t in dict(v).items()} for k, v in getattr(self, name).items()}
            object.__setattr__(self, name, declared)

        by_event: dict[str, OCEvent] = {}
        for event in self.events:
            if event.event_id in by_event:
                raise OCValidationError(
                    f"event id {event.event_id!r} appears twice; build the log with OCEventLog.build, "
                    "which deduplicates identical rows and refuses conflicting ones"
                )
            by_event[event.event_id] = event
        by_object: dict[str, OCObject] = {}
        for obj in self.objects:
            if obj.object_id in by_object:
                raise OCValidationError(
                    f"object id {obj.object_id!r} appears twice; build the log with OCEventLog.build, "
                    "which deduplicates identical rows and refuses conflicting ones"
                )
            by_object[obj.object_id] = obj
        object.__setattr__(self, "_by_event", by_event)
        object.__setattr__(self, "_by_object", by_object)

        e2o_by_event: dict[str, list[E2O]] = {}
        e2o_by_object: dict[str, list[E2O]] = {}
        for link in self.e2o:
            if link.event_id not in by_event:
                raise OCValidationError(f"event-to-object relation refers to unknown event {link.event_id!r}")
            if link.object_id not in by_object:
                raise OCValidationError(f"event-to-object relation refers to unknown object {link.object_id!r}")
            e2o_by_event.setdefault(link.event_id, []).append(link)
            e2o_by_object.setdefault(link.object_id, []).append(link)
        object.__setattr__(self, "_e2o_by_event", {k: tuple(v) for k, v in e2o_by_event.items()})
        object.__setattr__(self, "_e2o_by_object", {k: tuple(v) for k, v in e2o_by_object.items()})

        timed_allowed = self.source.relation_time_semantics == "interval"
        o2o_out: dict[str, list[O2O]] = {}
        o2o_in: dict[str, list[O2O]] = {}
        for rel in self.o2o:
            if rel.source_id not in by_object:
                raise OCValidationError(f"object-to-object relation refers to unknown source object {rel.source_id!r}")
            if rel.target_id not in by_object:
                raise OCValidationError(f"object-to-object relation refers to unknown target object {rel.target_id!r}")
            if rel.timed and not timed_allowed:
                raise OCValidationError(
                    f"relation {rel.source_id!r}->{rel.target_id!r} carries a validity interval while the source declares "
                    "relation_time_semantics='atemporal'; declare 'interval' with its provenance, or drop the interval"
                )
            o2o_out.setdefault(rel.source_id, []).append(rel)
            o2o_in.setdefault(rel.target_id, []).append(rel)
        object.__setattr__(self, "_o2o_out", {k: tuple(v) for k, v in o2o_out.items()})
        object.__setattr__(self, "_o2o_in", {k: tuple(v) for k, v in o2o_in.items()})

        history: dict[tuple[str, str], list[AttributeChange]] = {}
        for change in self.attribute_history:
            if change.object_id not in by_object:
                raise OCValidationError(f"attribute history refers to unknown object {change.object_id!r}")
            history.setdefault((change.object_id, change.attribute), []).append(change)
        indexed: dict[tuple[str, str], tuple[AttributeChange, ...]] = {}
        for key, changes in history.items():
            ordered = sorted(changes, key=lambda c: c.timestamp)
            stamps = [c.timestamp for c in ordered]
            if len(set(stamps)) != len(stamps):
                raise OCValidationError(
                    f"object {key[0]!r} attribute {key[1]!r} has two values for one instant; "
                    "build the log with OCEventLog.build, which reports identical repeats and refuses conflicting ones"
                )
            indexed[key] = tuple(ordered)
        object.__setattr__(self, "_history", indexed)

    @classmethod
    def build(
        cls,
        *,
        events: Sequence[OCEvent],
        objects: Sequence[OCObject],
        e2o: Sequence[E2O] = (),
        o2o: Sequence[O2O] = (),
        attribute_history: Sequence[AttributeChange] = (),
        source: SourceMetadata | None = None,
        object_type_attributes: Mapping[str, Mapping[str, str]] | None = None,
        event_type_attributes: Mapping[str, Mapping[str, str]] | None = None,
        on_dangling: str = "error",
        deduplicate: bool = True,
        issues: Sequence[ValidationIssue] = (),
    ) -> OCEventLog:
        """Validate, repair what can be repaired, and build the log.

        ``on_dangling="error"`` (the default) refuses a relation that points at
        an event or object the log does not contain. ``"drop"`` removes it and
        records a :class:`ValidationIssue` — a caller who knows the extract is
        partial can say so, and the log then carries the admission.

        ``deduplicate=False`` turns identical repeated rows into errors as
        well, for a caller who needs the source to be exact.

        ``issues`` seeds the report with findings a reader already made — an
        unsupported field in a file, for instance — so that one log carries one
        report.
        """
        if on_dangling not in ("error", "drop"):
            raise OCValidationError(f"on_dangling must be 'error' or 'drop', got {on_dangling!r}")
        found: list[ValidationIssue] = list(issues)
        metadata = source or SourceMetadata(interchange="in-memory")

        kept_events, kept_objects = _unique_entities(events, objects, found, deduplicate=deduplicate)
        known_events = {e.event_id for e in kept_events}
        known_objects = {o.object_id for o in kept_objects}

        kept_e2o = _unique_relations(
            e2o,
            key=lambda link: link.key,
            code=OCIssueCode.DUPLICATE_E2O_ROW,
            what="event-to-object relation",
            found=found,
            deduplicate=deduplicate,
        )
        kept_e2o = _drop_dangling(
            kept_e2o,
            checks=(
                (lambda link: link.event_id in known_events, OCIssueCode.DANGLING_E2O_EVENT, "unknown event"),
                (lambda link: link.object_id in known_objects, OCIssueCode.DANGLING_E2O_OBJECT, "unknown object"),
            ),
            label=lambda link: f"{link.event_id}->{link.object_id}",
            found=found,
            on_dangling=on_dangling,
        )
        kept_o2o = _unique_relations(
            o2o,
            key=lambda rel: rel.key,
            code=OCIssueCode.DUPLICATE_O2O_ROW,
            what="object-to-object relation",
            found=found,
            deduplicate=deduplicate,
        )
        kept_o2o = _drop_dangling(
            kept_o2o,
            checks=(
                (lambda rel: rel.source_id in known_objects, OCIssueCode.DANGLING_O2O_SOURCE, "unknown source object"),
                (lambda rel: rel.target_id in known_objects, OCIssueCode.DANGLING_O2O_TARGET, "unknown target object"),
            ),
            label=lambda rel: f"{rel.source_id}->{rel.target_id}",
            found=found,
            on_dangling=on_dangling,
        )
        kept_history = _unique_history(attribute_history, found, deduplicate=deduplicate)
        kept_history = _drop_dangling(
            kept_history,
            checks=((lambda c: c.object_id in known_objects, OCIssueCode.DANGLING_ATTRIBUTE_OBJECT, "unknown object"),),
            label=lambda c: f"{c.object_id}.{c.attribute}",
            found=found,
            on_dangling=on_dangling,
        )

        _report_qualifiers(kept_e2o, kept_o2o, found)
        metadata = replace(
            metadata,
            timezone=_one_timezone(kept_events, kept_history, found),
            precision={
                "events": observed_precision(e.timestamp for e in kept_events),
                "attribute_history": observed_precision(c.timestamp for c in kept_history),
            },
            options={**metadata.options, "on_dangling": on_dangling, "deduplicate": bool(deduplicate)},
        )
        errors = [i for i in found if i.severity is IssueSeverity.ERROR]
        if errors:
            raise OCValidationError(
                "the log cannot be built without inventing a fact: "
                + "; ".join(f"{i.code.value} ({i.count}): {i.message}" for i in errors)
            )
        return cls(
            events=tuple(kept_events),
            objects=tuple(kept_objects),
            e2o=tuple(kept_e2o),
            o2o=tuple(kept_o2o),
            attribute_history=tuple(kept_history),
            source=metadata,
            validation=ValidationReport(tuple(found)),
            object_type_attributes={}
            if object_type_attributes is None
            else {k: dict(v) for k, v in object_type_attributes.items()},
            event_type_attributes={} if event_type_attributes is None else {k: dict(v) for k, v in event_type_attributes.items()},
        )

    def with_source(self, **changes: Any) -> OCEventLog:
        """A copy whose source metadata differs in the named fields."""
        return replace(self, source=replace(self.source, **changes))

    # ----------------------------------------------------------------- access
    def __len__(self) -> int:
        return len(self.events)

    def __repr__(self) -> str:
        state = "clean" if self.validation.clean else f"{len(self.validation)} validation issue(s)"
        return (
            f"OCEventLog({len(self.events):,} events, {len(self.objects):,} objects, "
            f"{len(self.e2o):,} E2O, {len(self.o2o):,} O2O, {len(self.attribute_history):,} attribute changes, "
            f"{self.source.interchange}, {state})"
        )

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(e.event_id for e in self.events)

    @property
    def object_ids(self) -> tuple[str, ...]:
        return tuple(o.object_id for o in self.objects)

    @property
    def object_types(self) -> tuple[str, ...]:
        """Object types present in the data, in first-seen order."""
        return tuple(dict.fromkeys(o.object_type for o in self.objects))

    @property
    def activities(self) -> tuple[str, ...]:
        """Event types present in the data, in first-seen order."""
        return tuple(dict.fromkeys(e.activity for e in self.events))

    @property
    def qualifiers(self) -> tuple[str, ...]:
        """Every qualifier used by either relation table, sorted."""
        return tuple(sorted({link.qualifier for link in self.e2o} | {rel.qualifier for rel in self.o2o}))

    def event(self, event_id: str) -> OCEvent:
        """One event by canonical id."""
        try:
            return self._by_event[str(event_id)]
        except KeyError:
            raise OCValidationError(f"unknown event id {event_id!r}") from None

    def obj(self, object_id: str) -> OCObject:
        """One object by canonical id."""
        try:
            return self._by_object[str(object_id)]
        except KeyError:
            raise OCValidationError(f"unknown object id {object_id!r}") from None

    def objects_of(self, event_id: str, *, qualifier: str | None = None) -> tuple[E2O, ...]:
        """The objects one event touched, optionally in one role."""
        links = self._e2o_by_event.get(str(event_id), ())
        return links if qualifier is None else tuple(link for link in links if link.qualifier == qualifier)

    def events_of(
        self,
        object_id: str,
        *,
        qualifier: str | None = None,
        activity: str | None = None,
        since: pd.Timestamp | None = None,
        until: pd.Timestamp | None = None,
    ) -> tuple[OCEvent, ...]:
        """The events of one object, ordered by ``(timestamp, event_id)``.

        The ordering is total and stable, so two events with equal timestamps
        keep a deterministic order **and both appear**: nothing here treats a
        shared timestamp as a shared identity.
        """
        links = self._e2o_by_object.get(str(object_id), ())
        picked = []
        for link in links:
            if qualifier is not None and link.qualifier != qualifier:
                continue
            event = self._by_event[link.event_id]
            if activity is not None and event.activity != activity:
                continue
            if since is not None and event.timestamp < since:
                continue
            if until is not None and event.timestamp > until:
                continue
            picked.append(event)
        unique = {e.event_id: e for e in picked}
        return tuple(sorted(unique.values(), key=lambda e: (e.timestamp, e.event_id)))

    def related(
        self,
        object_id: str,
        *,
        direction: str = "forward",
        qualifier: str | None = None,
        target_type: str | None = None,
    ) -> tuple[O2O, ...]:
        """Object-to-object relations of one object, in one direction.

        ``direction="forward"`` follows relations whose *source* is
        ``object_id``; ``"reverse"`` follows those whose *target* it is. There
        is no undirected mode: an invoice belonging to an order and an order
        owning an invoice are different statements, and a traversal that mixes
        them is how a bounded context becomes the whole business.
        """
        oid = str(object_id)
        if direction == "forward":
            relations = self._o2o_out.get(oid, ())
            other = "target_id"
        elif direction == "reverse":
            relations = self._o2o_in.get(oid, ())
            other = "source_id"
        else:
            raise OCValidationError(f"direction must be 'forward' or 'reverse', got {direction!r}")
        out = []
        for rel in relations:
            if qualifier is not None and rel.qualifier != qualifier:
                continue
            if target_type is not None and self._by_object[getattr(rel, other)].object_type != target_type:
                continue
            out.append(rel)
        return tuple(out)

    def neighbours(self, object_id: str, *, direction: str = "forward", **kwargs: Any) -> tuple[str, ...]:
        """The ids on the far side of :meth:`related`, deduplicated and sorted."""
        far = "target_id" if direction == "forward" else "source_id"
        relations = self.related(object_id, direction=direction, **kwargs)
        return tuple(sorted({getattr(rel, far) for rel in relations}))

    def attributes_of(self, object_id: str) -> tuple[str, ...]:
        """Attribute names this object has a history for, sorted."""
        oid = str(object_id)
        return tuple(sorted(attribute for (obj_id, attribute) in self._history if obj_id == oid))

    def history(self, object_id: str, attribute: str) -> tuple[AttributeChange, ...]:
        """One attribute's changes, in time order. Empty when there are none."""
        return self._history.get((str(object_id), str(attribute)), ())

    def value_at(
        self,
        object_id: str,
        attribute: str,
        at: pd.Timestamp | str | None,
        *,
        policy: AttributePolicy | str = AttributePolicy.AS_OF,
        role: str = "attribute",
    ) -> AttributeReading:
        """The value of a changing attribute **at a declared instant**.

        Under the default :data:`AttributePolicy.AS_OF` the answer is the last
        change at or before ``at``. When there is none the reading reports
        ``found=False`` with an
        :data:`~wise.evidence.models.QualificationCode.ATTRIBUTE_NOT_SET_AT_TIME`
        qualification; it does not reach forward to the next change, and it
        does not fall back to the first value the object ever had.

        :data:`AttributePolicy.LATEST_KNOWN` is available for the cases where a
        reviewer genuinely wants today's owner rather than the owner at the
        time. When that value comes from after ``at``, the reading says so with
        :data:`~wise.evidence.models.QualificationCode.ATTRIBUTE_READ_OUTSIDE_EVALUATION_TIME`.
        """
        chosen = AttributePolicy(policy)
        self.obj(object_id)  # unknown object is an error, not an empty reading
        changes = self.history(object_id, attribute)
        moment = None if at is None else _timestamp(at, "evaluation time")
        if moment is None and chosen is AttributePolicy.AS_OF:
            raise OCValidationError(
                f"{object_id}.{attribute}: reading an attribute as of a time needs that time; "
                "pass the evaluation instant, or ask for policy='latest_known' explicitly"
            )
        before = [c for c in changes if moment is None or c.timestamp <= moment]
        quals: list[Qualification] = []
        picked: AttributeChange | None
        if chosen is AttributePolicy.AS_OF:
            picked = before[-1] if before else None
        else:
            picked = changes[-1] if changes else None
            if picked is not None and moment is not None and picked.timestamp > moment:
                quals.append(
                    Qualification(
                        QualificationCode.ATTRIBUTE_READ_OUTSIDE_EVALUATION_TIME,
                        f"{object_id}.{attribute} was read as the latest known value, effective "
                        f"{picked.timestamp.isoformat()}, which is after the evaluation time {moment.isoformat()}",
                        scope="unit",
                    )
                )
        if picked is None:
            quals.append(
                Qualification(
                    QualificationCode.ATTRIBUTE_NOT_SET_AT_TIME,
                    f"{object_id}.{attribute} has no value in force"
                    + (f" at {moment.isoformat()}" if moment is not None else "")
                    + f"; {len(changes)} change(s) are recorded in total, "
                    + f"{len(changes) - len(before)} of them later. Missing is not empty and not the next value.",
                    scope="unit",
                )
            )
        return AttributeReading(
            object_id=str(object_id),
            attribute=str(attribute),
            at=moment,
            policy=chosen,
            value=None if picked is None else picked.value,
            found=picked is not None,
            effective_from=None if picked is None else picked.timestamp,
            n_changes=len(changes),
            n_at_or_before=len(before),
            qualifications=tuple(quals),
        )

    # ------------------------------------------------------------- identities
    def summary(self) -> dict[str, Any]:
        """Counts by table and by type — the shape of the log in one dict."""
        object_types: dict[str, int] = {}
        for obj in self.objects:
            object_types[obj.object_type] = object_types.get(obj.object_type, 0) + 1
        activities: dict[str, int] = {}
        for event in self.events:
            activities[event.activity] = activities.get(event.activity, 0) + 1
        return {
            "schema_version": self.schema_version,
            "n_events": len(self.events),
            "n_objects": len(self.objects),
            "n_e2o": len(self.e2o),
            "n_o2o": len(self.o2o),
            "n_attribute_changes": len(self.attribute_history),
            "object_types": object_types,
            "activities": activities,
            "qualifiers": list(self.qualifiers),
            "interchange": self.source.interchange,
            "precision": dict(self.source.precision),
            "relation_time_semantics": self.source.relation_time_semantics,
            "validation": self.validation.to_dict(),
        }

    def content_fingerprint(self) -> str:
        """SHA-256 over the log's content, excluding where it was read from.

        Two reads of one file through two adapters produce the same
        fingerprint; a round-trip that loses a qualifier does not.
        """
        return _digest(
            {
                "schema_version": self.schema_version,
                "events": [
                    {
                        "id": e.event_id,
                        "activity": e.activity,
                        "time": e.timestamp.isoformat(),
                        "attributes": {k: _jsonable(v) for k, v in sorted(e.attributes.items())},
                    }
                    for e in sorted(self.events, key=lambda e: e.event_id)
                ],
                "objects": [{"id": o.object_id, "type": o.object_type} for o in sorted(self.objects, key=lambda o: o.object_id)],
                "e2o": sorted([list(link.key) for link in self.e2o]),
                "o2o": sorted([list(rel.key) for rel in self.o2o]),
                "history": sorted(
                    [[c.object_id, c.attribute, c.timestamp.isoformat(), _jsonable(c.value)] for c in self.attribute_history]
                ),
            }
        )


# ------------------------------------------------------------ build internals
def _unique_entities(
    events: Sequence[OCEvent],
    objects: Sequence[OCObject],
    found: list[ValidationIssue],
    *,
    deduplicate: bool,
) -> tuple[list[OCEvent], list[OCObject]]:
    kept_events: dict[str, OCEvent] = {}
    repeats: list[str] = []
    conflicts: list[str] = []
    for event in events:
        seen = kept_events.get(event.event_id)
        if seen is None:
            kept_events[event.event_id] = event
        elif seen == event:
            repeats.append(event.event_id)
        else:
            conflicts.append(event.event_id)
    _record_repeats(found, OCIssueCode.DUPLICATE_EVENT_ROW, "identical event row", repeats, deduplicate)
    if conflicts:
        found.append(
            ValidationIssue(
                OCIssueCode.CONFLICTING_EVENT_ID,
                IssueSeverity.ERROR,
                "one event id carries different activities, timestamps or attributes; an identity cannot be repaired by "
                "choosing a row",
                len(conflicts),
                tuple(sorted(set(conflicts))[:5]),
            )
        )
    kept_objects: dict[str, OCObject] = {}
    object_repeats: list[str] = []
    object_conflicts: list[str] = []
    for obj in objects:
        seen_object = kept_objects.get(obj.object_id)
        if seen_object is None:
            kept_objects[obj.object_id] = obj
        elif seen_object == obj:
            object_repeats.append(obj.object_id)
        else:
            object_conflicts.append(obj.object_id)
    _record_repeats(found, OCIssueCode.DUPLICATE_OBJECT_ROW, "identical object row", object_repeats, deduplicate)
    if object_conflicts:
        found.append(
            ValidationIssue(
                OCIssueCode.CONFLICTING_OBJECT_ID,
                IssueSeverity.ERROR,
                "one object id is declared with two different types",
                len(object_conflicts),
                tuple(sorted(set(object_conflicts))[:5]),
            )
        )
    return list(kept_events.values()), list(kept_objects.values())


def _record_repeats(
    found: list[ValidationIssue], code: OCIssueCode, what: str, repeats: Sequence[str], deduplicate: bool
) -> None:
    if not repeats:
        return
    found.append(
        ValidationIssue(
            code,
            IssueSeverity.REPAIRED if deduplicate else IssueSeverity.ERROR,
            f"{len(repeats)} repeated {what}(s) were {'removed' if deduplicate else 'refused'}",
            len(repeats),
            tuple(sorted({str(r) for r in repeats})[:5]),
        )
    )


def _unique_relations(
    rows: Sequence[Any],
    *,
    key: Any,
    code: OCIssueCode,
    what: str,
    found: list[ValidationIssue],
    deduplicate: bool,
) -> list[Any]:
    kept: dict[Any, Any] = {}
    repeats: list[str] = []
    for row in rows:
        row_key = key(row)
        if row_key in kept:
            repeats.append("|".join(str(part) for part in row_key))
        else:
            kept[row_key] = row
    _record_repeats(found, code, f"identical {what}", repeats, deduplicate)
    return list(kept.values())


def _unique_history(rows: Sequence[AttributeChange], found: list[ValidationIssue], *, deduplicate: bool) -> list[AttributeChange]:
    kept: dict[tuple[str, str, str], AttributeChange] = {}
    repeats: list[str] = []
    conflicts: list[str] = []
    for change in rows:
        seen = kept.get(change.key)
        if seen is None:
            kept[change.key] = change
        elif seen.value == change.value or (seen.value != seen.value and change.value != change.value):
            repeats.append("|".join(change.key))
        else:
            conflicts.append("|".join(change.key))
    _record_repeats(found, OCIssueCode.DUPLICATE_ATTRIBUTE_ROW, "identical attribute history row", repeats, deduplicate)
    if conflicts:
        found.append(
            ValidationIssue(
                OCIssueCode.CONFLICTING_ATTRIBUTE_VALUE,
                IssueSeverity.ERROR,
                "one object attribute has two different values effective at one instant; the history cannot be ordered "
                "without choosing one of them",
                len(conflicts),
                tuple(sorted(set(conflicts))[:5]),
            )
        )
    return sorted(kept.values(), key=lambda c: (c.object_id, c.attribute, c.timestamp))


def _drop_dangling(
    rows: Sequence[Any],
    *,
    checks: Sequence[tuple[Any, OCIssueCode, str]],
    label: Any,
    found: list[ValidationIssue],
    on_dangling: str,
) -> list[Any]:
    kept: list[Any] = []
    broken: dict[OCIssueCode, list[str]] = {}
    for row in rows:
        failed = next((code for predicate, code, _ in checks if not predicate(row)), None)
        if failed is None:
            kept.append(row)
        else:
            broken.setdefault(failed, []).append(str(label(row)))
    for code, rows_broken in broken.items():
        reason = next(text for _, issue_code, text in checks if issue_code is code)
        found.append(
            ValidationIssue(
                code,
                IssueSeverity.REPAIRED if on_dangling == "drop" else IssueSeverity.ERROR,
                f"{len(rows_broken)} relation row(s) point at an {reason}; "
                + ("dropped on request (on_dangling='drop')" if on_dangling == "drop" else "the log refuses to hide them"),
                len(rows_broken),
                tuple(sorted(set(rows_broken))[:5]),
            )
        )
    return kept


def _report_qualifiers(e2o: Sequence[E2O], o2o: Sequence[O2O], found: list[ValidationIssue]) -> None:
    unqualified = [link.event_id for link in e2o if not link.qualifier]
    unqualified += [f"{rel.source_id}->{rel.target_id}" for rel in o2o if not rel.qualifier]
    if unqualified:
        found.append(
            ValidationIssue(
                OCIssueCode.MISSING_QUALIFIER,
                IssueSeverity.INFO,
                "relation rows carry no qualifier, so the role an object played in them is unknown; "
                "a path that requires a qualifier will not select them",
                len(unqualified),
                tuple(sorted(set(unqualified))[:5]),
            )
        )
    loops = [rel.source_id for rel in o2o if rel.source_id == rel.target_id]
    if loops:
        found.append(
            ValidationIssue(
                OCIssueCode.SELF_RELATION,
                IssueSeverity.INFO,
                "objects are related to themselves; traversal visits each object once, so these add no context",
                len(loops),
                tuple(sorted(set(loops))[:5]),
            )
        )


def _one_timezone(events: Sequence[OCEvent], history: Sequence[AttributeChange], found: list[ValidationIssue]) -> str | None:
    zones = {str(e.timestamp.tz) for e in events} | {str(c.timestamp.tz) for c in history}
    if len(zones) > 1:
        found.append(
            ValidationIssue(
                OCIssueCode.MIXED_TIMEZONES,
                IssueSeverity.ERROR,
                f"timestamps mix time zones ({sorted(zones)}); comparing them would compare different clocks",
                len(zones),
                tuple(sorted(zones)),
            )
        )
        return None
    if not zones:
        return None
    only = next(iter(zones))
    return None if only == "None" else only


__all__ = [
    "E2O",
    "O2O",
    "OC_SCHEMA_VERSION",
    "RELATION_TIME_SEMANTICS",
    "AttributeChange",
    "AttributePolicy",
    "AttributeReading",
    "IssueSeverity",
    "OCEvent",
    "OCEventLog",
    "OCIssueCode",
    "OCObject",
    "SourceMetadata",
    "ValidationIssue",
    "ValidationReport",
    "observed_precision",
]
