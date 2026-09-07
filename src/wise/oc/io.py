"""Adapters for the interchanges this library actually supports.

Three, and only three, and each says what it can and cannot carry
(:func:`interchange_support`):

``ocel2-json`` — **read and write**
    The OCEL 2.0 JSON serialisation: ``objectTypes``, ``eventTypes``,
    ``objects`` and ``events``, with qualified relationships and timestamped
    object attribute values. This is the pinned interchange. A document read
    and written back is compared as a *document*, not as bytes, and the tests
    pin a small fixture whose exact structure this module must reproduce.

``ocel2-sqlite`` — **read only**, with a declared loss
    The OCEL 2.0 SQLite serialisation. Its per-type tables name attributes by
    *sanitised column identifier* (``Invoice Receipt (MSEG-WEAHR)`` becomes
    ``InvoiceReceiptMSEGWEAHR``) and it carries no mapping back. The original
    attribute names are therefore **not recoverable** from this format; the
    reader says so in the log's validation report instead of pretending
    otherwise, which is why there is no writer.

``wise-oc-tables`` — **read and write**
    Six flat pandas tables. This is the native round-trip, and it is a
    *different* acceptance test from the OCEL export: one shows that nothing is
    lost inside the library, the other that what leaves it conforms to the
    standard.

Nothing is claimed for OCEL 1.0, for XML, for a later OCEL revision, or for a
CSV extract whose files happen to be readable. Reading generic JSON is not
support for a standard.

Values are preserved exactly as the source wrote them. OCEL declares an
attribute's type (``string``, ``float``, ``time``); this module keeps the
declaration in :attr:`~wise.oc.model.OCEventLog.object_type_attributes` and
does **not** coerce the value to it, because a silent coercion is an
unrecorded edit of the data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..errors import OCInterchangeError, OCValidationError
from .model import (
    E2O,
    O2O,
    AttributeChange,
    IssueSeverity,
    OCEvent,
    OCEventLog,
    OCIssueCode,
    OCObject,
    SourceMetadata,
    ValidationIssue,
)

#: The OCEL 2.0 JSON serialisation. Read and write.
OCEL2_JSON = "ocel2-json"
#: The OCEL 2.0 SQLite serialisation. Read only; attribute names arrive sanitised.
OCEL2_SQLITE = "ocel2-sqlite"
#: Six flat pandas tables — the library's own lossless view.
NATIVE_TABLES = "wise-oc-tables"

#: The OCEL 2.0 revision these adapters were written against.
OCEL_SPEC = "OCEL 2.0"

_TOP_LEVEL_KEYS = ("objectTypes", "eventTypes", "objects", "events")
_EVENT_KEYS = ("id", "type", "time", "attributes", "relationships")
_OBJECT_KEYS = ("id", "type", "attributes", "relationships")
_NATIVE_TABLE_NAMES = ("events", "event_attributes", "objects", "object_attributes", "e2o", "o2o")


def interchange_support() -> dict[str, dict[str, Any]]:
    """What each adapter supports, in one machine-readable declaration.

    >>> interchange_support()["ocel2-json"]["write"]
    True
    >>> interchange_support()["ocel2-sqlite"]["write"]
    False
    """
    return {
        OCEL2_JSON: {
            "spec": OCEL_SPEC,
            "read": True,
            "write": True,
            "reader": "wise.oc.io.read_ocel2_json",
            "writer": "wise.oc.io.write_ocel2_json",
            "preserves": [
                "event id, type, timestamp and attributes",
                "object id and type",
                "qualified event-to-object relationships",
                "qualified object-to-object relationships",
                "timestamped object attribute values",
                "declared attribute names and types per object and event type",
            ],
            "cannot_carry": [
                "object-to-object validity intervals (the standard has no place for them)",
                "source and extraction metadata (provenance of the read, not content of the log)",
            ],
        },
        OCEL2_SQLITE: {
            "spec": OCEL_SPEC,
            "read": True,
            "write": False,
            "reader": "wise.oc.io.read_ocel2_sqlite",
            "writer": None,
            "preserves": [
                "event id, type, timestamp and attributes",
                "object id and type",
                "qualified event-to-object relationships",
                "qualified object-to-object relationships",
                "timestamped object attribute values",
            ],
            "cannot_carry": [
                "original attribute names: the serialisation stores sanitised column identifiers and no mapping back",
                "declared OCEL attribute types: only SQLite column affinities are available, and they are reported as such",
            ],
        },
        NATIVE_TABLES: {
            "spec": "wise-oc/1",
            "read": True,
            "write": True,
            "reader": "wise.oc.io.from_tables",
            "writer": "wise.oc.io.to_tables",
            "preserves": ["every field of the object log, including relation validity intervals"],
            "cannot_carry": ["the validation report and the source metadata, which describe the read rather than the log"],
        },
    }


@dataclass(frozen=True)
class LossReport:
    """What an export could not carry. Empty means nothing was lost.

    >>> LossReport(OCEL2_JSON, "write").lossless
    True
    """

    interchange: str
    direction: str
    issues: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(self.issues))
        if self.direction not in ("read", "write"):
            raise OCInterchangeError(f"direction must be 'read' or 'write', got {self.direction!r}")

    @property
    def lossless(self) -> bool:
        return not any(i.severity is IssueSeverity.DROPPED for i in self.issues)

    def __len__(self) -> int:
        return len(self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interchange": self.interchange,
            "direction": self.direction,
            "lossless": self.lossless,
            "issues": [i.to_dict() for i in self.issues],
        }


# ---------------------------------------------------------------- OCEL 2 JSON
def _mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OCInterchangeError(f"{what} must be a JSON object, got {type(value).__name__}")
    return value


def _sequence(value: Any, what: str) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise OCInterchangeError(f"{what} must be a JSON array, got {type(value).__name__}")
    return value


def _required(entry: Mapping[str, Any], key: str, what: str) -> Any:
    if key not in entry:
        raise OCInterchangeError(f"{what} has no {key!r}; this is not the OCEL 2.0 JSON structure")
    return entry[key]


def _unknown_keys(entry: Mapping[str, Any], known: Sequence[str], seen: dict[str, int]) -> None:
    for key in entry:
        if key not in known:
            seen[key] = seen.get(key, 0) + 1


def _declared_types(entries: Sequence[Any], what: str) -> dict[str, dict[str, str]]:
    declared: dict[str, dict[str, str]] = {}
    for entry in entries:
        row = _mapping(entry, what)
        name = str(_required(row, "name", what))
        attributes: dict[str, str] = {}
        for candidate in _sequence(row.get("attributes"), f"{what} attributes"):
            pair = _mapping(candidate, f"{what} attribute")
            attributes[str(_required(pair, "name", f"{what} attribute"))] = str(pair.get("type", ""))
        declared[name] = attributes
    return declared


def read_ocel2_json(
    source: Any,
    *,
    name: str | None = None,
    on_dangling: str = "error",
    deduplicate: bool = True,
) -> OCEventLog:
    """Read an OCEL 2.0 JSON document into an :class:`~wise.oc.model.OCEventLog`.

    ``source`` is a path, or an already-parsed mapping for a caller who has the
    document in hand. A path is hashed as it is read, and the digest travels
    with the log in :attr:`~wise.oc.model.SourceMetadata.digest`.

    Structural failures raise :class:`~wise.errors.OCInterchangeError` — a
    document without ``events`` is not an OCEL 2.0 document, however valid its
    JSON. Content failures follow :meth:`~wise.oc.model.OCEventLog.build`:
    identical repeated rows are removed and reported, conflicting identities
    raise, and dangling relations raise unless ``on_dangling="drop"``.

    Keys the OCEL 2.0 structure does not define are **not** carried into the
    log; each is reported once, with a count, in the validation report.
    """
    digest: str | None = None
    label = name
    if isinstance(source, Mapping):
        document: Mapping[str, Any] = source
        label = label or "<mapping>"
    else:
        path = Path(source)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        try:
            document = _mapping(json.loads(raw.decode("utf-8")), "an OCEL 2.0 JSON document")
        except json.JSONDecodeError as exc:
            raise OCInterchangeError(f"{path}: not valid JSON ({exc})") from exc
        label = label or str(path)

    issues: list[ValidationIssue] = []
    unknown: dict[str, int] = {}
    _unknown_keys(document, _TOP_LEVEL_KEYS, unknown)
    for key in ("events", "objects"):
        if key not in document:
            raise OCInterchangeError(f"an OCEL 2.0 JSON document needs {key!r}; this payload has {sorted(document)}")

    object_types = _declared_types(_sequence(document.get("objectTypes"), "objectTypes"), "an object type")
    event_types = _declared_types(_sequence(document.get("eventTypes"), "eventTypes"), "an event type")

    events: list[OCEvent] = []
    e2o: list[E2O] = []
    for entry in _sequence(document["events"], "events"):
        row = _mapping(entry, "an event")
        _unknown_keys(row, _EVENT_KEYS, unknown)
        event_id = str(_required(row, "id", "an event"))
        attributes: dict[str, Any] = {}
        for attribute in _sequence(row.get("attributes"), f"event {event_id} attributes"):
            pair = _mapping(attribute, f"event {event_id} attribute")
            _unknown_keys(pair, ("name", "value"), unknown)
            attributes[str(_required(pair, "name", f"event {event_id} attribute"))] = pair.get("value")
        events.append(
            OCEvent(
                event_id=event_id,
                activity=str(_required(row, "type", f"event {event_id}")),
                timestamp=_required(row, "time", f"event {event_id}"),
                attributes=attributes,
            )
        )
        for relationship in _sequence(row.get("relationships"), f"event {event_id} relationships"):
            link = _mapping(relationship, f"event {event_id} relationship")
            _unknown_keys(link, ("objectId", "qualifier"), unknown)
            e2o.append(
                E2O(
                    event_id=event_id,
                    object_id=str(_required(link, "objectId", f"event {event_id} relationship")),
                    qualifier=str(link.get("qualifier", "")),
                )
            )

    objects: list[OCObject] = []
    o2o: list[O2O] = []
    history: list[AttributeChange] = []
    for entry in _sequence(document["objects"], "objects"):
        row = _mapping(entry, "an object")
        _unknown_keys(row, _OBJECT_KEYS, unknown)
        object_id = str(_required(row, "id", "an object"))
        objects.append(OCObject(object_id=object_id, object_type=str(_required(row, "type", f"object {object_id}"))))
        for attribute in _sequence(row.get("attributes"), f"object {object_id} attributes"):
            pair = _mapping(attribute, f"object {object_id} attribute")
            _unknown_keys(pair, ("name", "value", "time"), unknown)
            history.append(
                AttributeChange(
                    object_id=object_id,
                    attribute=str(_required(pair, "name", f"object {object_id} attribute")),
                    timestamp=_required(pair, "time", f"object {object_id} attribute"),
                    value=pair.get("value"),
                )
            )
        for relationship in _sequence(row.get("relationships"), f"object {object_id} relationships"):
            link = _mapping(relationship, f"object {object_id} relationship")
            _unknown_keys(link, ("objectId", "qualifier"), unknown)
            o2o.append(
                O2O(
                    source_id=object_id,
                    target_id=str(_required(link, "objectId", f"object {object_id} relationship")),
                    qualifier=str(link.get("qualifier", "")),
                )
            )

    if unknown:
        issues.append(
            ValidationIssue(
                OCIssueCode.UNSUPPORTED_FIELD,
                IssueSeverity.DROPPED,
                "the document carries keys the OCEL 2.0 structure does not define; they were read but not kept: "
                + ", ".join(f"{key} ({count})" for key, count in sorted(unknown.items())),
                sum(unknown.values()),
                tuple(sorted(unknown)[:5]),
            )
        )
    undeclared = sorted({o.object_type for o in objects} - set(object_types)) + sorted(
        {e.activity for e in events} - set(event_types)
    )
    if undeclared:
        issues.append(
            ValidationIssue(
                OCIssueCode.UNDECLARED_TYPE,
                IssueSeverity.INFO,
                "types are used by the data but not declared in objectTypes/eventTypes; an export will declare them "
                "from the observed values",
                len(undeclared),
                tuple(undeclared[:5]),
            )
        )

    return OCEventLog.build(
        events=events,
        objects=objects,
        e2o=e2o,
        o2o=o2o,
        attribute_history=history,
        source=SourceMetadata(
            interchange=OCEL2_JSON,
            spec=OCEL_SPEC,
            source=label,
            reader="wise.oc.io.read_ocel2_json",
            relation_time_semantics="atemporal",
            digest=digest,
            notes=(
                "OCEL 2.0 object-to-object relations carry no validity interval; this log states that two objects are "
                "related, not when they were.",
                "attribute values are kept exactly as the document wrote them; the declared OCEL types are recorded, "
                "not applied.",
            ),
        ),
        object_type_attributes=object_types,
        event_type_attributes=event_types,
        on_dangling=on_dangling,
        deduplicate=deduplicate,
        issues=issues,
    )


def _ocel_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and value != value:  # NaN has no JSON spelling
        return None
    return value


def _ocel_type_of(values: Sequence[Any]) -> str:
    kinds = {type(v).__name__ for v in values if v is not None}
    if kinds == {"str"}:
        return "string"
    if kinds == {"bool"}:
        return "boolean"
    if kinds == {"int"}:
        return "integer"
    if kinds <= {"int", "float"} and kinds:
        return "float"
    if kinds == {"Timestamp"}:
        return "time"
    return "string"


def ocel2_json_document(log: OCEventLog, *, allow_loss: bool = False) -> tuple[dict[str, Any], LossReport]:
    """Build the OCEL 2.0 JSON document for a log, with what it could not carry.

    The document is ordered deterministically: types by declared then observed
    order, objects and events in the log's own order, each object's attribute
    values by ``(name, time)``, and relationships as the log holds them. Two
    exports of one log are therefore identical documents.
    """
    losses: list[ValidationIssue] = []
    timed = [rel for rel in log.o2o if rel.timed]
    if timed:
        losses.append(
            ValidationIssue(
                OCIssueCode.NOT_REPRESENTABLE,
                IssueSeverity.DROPPED,
                "object-to-object validity intervals have no place in OCEL 2.0 JSON; exporting drops them, so the "
                "exported file says two objects are related and no longer says when",
                len(timed),
                tuple(sorted(f"{rel.source_id}->{rel.target_id}" for rel in timed)[:5]),
            )
        )
    dropped_values = [
        f"{c.object_id}.{c.attribute}" for c in log.attribute_history if isinstance(c.value, float) and c.value != c.value
    ]
    if dropped_values:
        losses.append(
            ValidationIssue(
                OCIssueCode.NOT_REPRESENTABLE,
                IssueSeverity.DROPPED,
                "attribute values that are NaN are written as null; JSON has no NaN, and a null is a missing value, "
                "not the same fact",
                len(dropped_values),
                tuple(sorted(set(dropped_values))[:5]),
            )
        )
    report = LossReport(OCEL2_JSON, "write", tuple(losses))
    if not report.lossless and not allow_loss:
        raise OCInterchangeError(
            "exporting to OCEL 2.0 JSON would lose information: "
            + "; ".join(i.message for i in report.issues)
            + ". Pass allow_loss=True to export anyway and keep the report."
        )

    observed_object_attributes: dict[str, dict[str, list[Any]]] = {}
    for change in log.attribute_history:
        by_type = observed_object_attributes.setdefault(log.obj(change.object_id).object_type, {})
        by_type.setdefault(change.attribute, []).append(change.value)
    observed_event_attributes: dict[str, dict[str, list[Any]]] = {}
    for event in log.events:
        by_activity = observed_event_attributes.setdefault(event.activity, {})
        for key, value in event.attributes.items():
            by_activity.setdefault(key, []).append(value)

    def declarations(
        declared: Mapping[str, Mapping[str, str]], observed: Mapping[str, Mapping[str, list[Any]]], names: Sequence[str]
    ) -> list[dict[str, Any]]:
        out = []
        for name in list(declared) + [n for n in names if n not in declared]:
            attributes = dict(declared.get(name, {}))
            for attribute, values in observed.get(name, {}).items():
                attributes.setdefault(attribute, _ocel_type_of(values))
            out.append({"name": name, "attributes": [{"name": a, "type": t} for a, t in attributes.items()]})
        return out

    events = [
        {
            "id": event.event_id,
            "type": event.activity,
            "time": event.timestamp.isoformat(),
            "attributes": [{"name": key, "value": _ocel_value(value)} for key, value in event.attributes.items()],
            "relationships": [
                {"objectId": link.object_id, "qualifier": link.qualifier} for link in log.objects_of(event.event_id)
            ],
        }
        for event in log.events
    ]
    objects = []
    for obj in log.objects:
        attributes = [
            {"name": change.attribute, "value": _ocel_value(change.value), "time": change.timestamp.isoformat()}
            for attribute in log.attributes_of(obj.object_id)
            for change in log.history(obj.object_id, attribute)
        ]
        objects.append(
            {
                "id": obj.object_id,
                "type": obj.object_type,
                "attributes": attributes,
                "relationships": [{"objectId": rel.target_id, "qualifier": rel.qualifier} for rel in log.related(obj.object_id)],
            }
        )
    document = {
        "objectTypes": declarations(log.object_type_attributes, observed_object_attributes, log.object_types),
        "eventTypes": declarations(log.event_type_attributes, observed_event_attributes, log.activities),
        "objects": objects,
        "events": events,
    }
    return document, report


def write_ocel2_json(log: OCEventLog, path: Any, *, allow_loss: bool = False, indent: int | None = None) -> LossReport:
    """Write a log as OCEL 2.0 JSON and return what the format could not carry.

    Refuses rather than loses: when the log holds something the standard has no
    place for, this raises :class:`~wise.errors.OCInterchangeError` unless the
    caller passes ``allow_loss=True`` and takes the report with them.
    """
    document, report = ocel2_json_document(log, allow_loss=allow_loss)
    Path(path).write_text(json.dumps(document, ensure_ascii=False, indent=indent, allow_nan=False), encoding="utf-8")
    return report


# -------------------------------------------------------------- OCEL 2 SQLite
_SQLITE_AFFINITY = {"TEXT": "string", "REAL": "float", "INTEGER": "integer", "TIMESTAMP": "time", "NUMERIC": "float"}


def read_ocel2_sqlite(path: Any, *, on_dangling: str = "error", deduplicate: bool = True) -> OCEventLog:
    """Read the OCEL 2.0 SQLite serialisation. Read-only, and it says why.

    The database is opened read-only (``mode=ro``) and only the tables the
    serialisation defines are queried; no statement is built from user input.

    Two limitations are recorded in the log's validation report rather than
    hidden. The per-type tables name attributes by sanitised column identifier
    with no mapping back to the original name, so a log read from SQLite does
    **not** have the same attribute names as the same log read from JSON; and
    the declared OCEL attribute types are gone, leaving only SQLite column
    affinities, which are reported as the approximation they are.
    """
    import sqlite3

    uri = f"{Path(path).resolve().as_uri()}?mode=ro"
    issues: list[ValidationIssue] = [
        ValidationIssue(
            OCIssueCode.UNSUPPORTED_FIELD,
            IssueSeverity.DROPPED,
            "the OCEL 2.0 SQLite serialisation stores object and event attributes as sanitised column identifiers and "
            "carries no mapping back, so the original attribute names are not recoverable from this file",
            1,
            ("attribute names",),
        )
    ]
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.row_factory = sqlite3.Row
        tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
        for required in ("event", "object", "event_object", "object_object", "event_map_type", "object_map_type"):
            if required not in tables:
                raise OCInterchangeError(f"{path}: table {required!r} is missing; this is not the OCEL 2.0 SQLite serialisation")

        event_tables = {row["ocel_type"]: row["ocel_type_map"] for row in connection.execute("select * from event_map_type")}
        object_tables = {row["ocel_type"]: row["ocel_type_map"] for row in connection.execute("select * from object_map_type")}
        event_type_of = {row["ocel_id"]: row["ocel_type"] for row in connection.execute("select * from event")}
        object_type_of = {row["ocel_id"]: row["ocel_type"] for row in connection.execute("select * from object")}

        events: list[OCEvent] = []
        event_type_attributes: dict[str, dict[str, str]] = {}
        for activity, suffix in event_tables.items():
            table = f"event_{suffix}"
            if table not in tables:
                issues.append(_missing_table(table, activity))
                continue
            columns = _columns(connection, table)
            attribute_columns = {k: v for k, v in columns.items() if k not in ("ocel_id", "ocel_time")}
            event_type_attributes[activity] = {k: _SQLITE_AFFINITY.get(v.upper(), "string") for k, v in attribute_columns.items()}
            # the table name comes from the file's own type map, never from a caller
            for row in connection.execute(f'select * from "{table}"'):
                events.append(
                    OCEvent(
                        event_id=row["ocel_id"],
                        activity=activity,
                        timestamp=row["ocel_time"],
                        attributes={key: row[key] for key in attribute_columns if row[key] is not None},
                    )
                )

        objects: list[OCObject] = [OCObject(oid, otype) for oid, otype in object_type_of.items()]
        history: list[AttributeChange] = []
        object_type_attributes: dict[str, dict[str, str]] = {}
        for object_type, suffix in object_tables.items():
            table = f"object_{suffix}"
            if table not in tables:
                issues.append(_missing_table(table, object_type))
                continue
            columns = _columns(connection, table)
            attribute_columns = {k: v for k, v in columns.items() if k not in ("ocel_id", "ocel_time", "ocel_changed_field")}
            object_type_attributes[object_type] = {
                k: _SQLITE_AFFINITY.get(v.upper(), "string") for k, v in attribute_columns.items()
            }
            # the table name comes from the file's own type map, never from a caller
            for row in connection.execute(f'select * from "{table}"'):
                changed = row["ocel_changed_field"] if "ocel_changed_field" in columns else None
                names = list(attribute_columns) if not changed else [changed]
                for attribute in names:
                    if attribute not in attribute_columns:
                        continue
                    if changed is None and row[attribute] is None:
                        continue  # the initial row carries every attribute the object had at once
                    history.append(
                        AttributeChange(
                            object_id=row["ocel_id"], attribute=attribute, timestamp=row["ocel_time"], value=row[attribute]
                        )
                    )

        e2o = [
            E2O(row["ocel_event_id"], row["ocel_object_id"], row["ocel_qualifier"] or "")
            for row in connection.execute("select * from event_object")
        ]
        o2o = [
            O2O(row["ocel_source_id"], row["ocel_target_id"], row["ocel_qualifier"] or "")
            for row in connection.execute("select * from object_object")
        ]
    finally:
        connection.close()

    missing = sorted(set(event_type_of) - {e.event_id for e in events})
    if missing:
        issues.append(
            ValidationIssue(
                OCIssueCode.UNDECLARED_TYPE,
                IssueSeverity.ERROR,
                "events are listed in the 'event' table but absent from their own type table, so they have no timestamp",
                len(missing),
                tuple(missing[:5]),
            )
        )
    return OCEventLog.build(
        events=events,
        objects=objects,
        e2o=e2o,
        o2o=o2o,
        attribute_history=history,
        source=SourceMetadata(
            interchange=OCEL2_SQLITE,
            spec=OCEL_SPEC,
            source=str(path),
            reader="wise.oc.io.read_ocel2_sqlite",
            relation_time_semantics="atemporal",
            notes=(
                "attribute names are the serialisation's sanitised column identifiers, not the source system's names",
                "attribute types are SQLite column affinities, not the OCEL declarations",
            ),
        ),
        object_type_attributes=object_type_attributes,
        event_type_attributes=event_type_attributes,
        on_dangling=on_dangling,
        deduplicate=deduplicate,
        issues=issues,
    )


def _missing_table(table: str, type_name: str) -> ValidationIssue:
    return ValidationIssue(
        OCIssueCode.UNDECLARED_TYPE,
        IssueSeverity.ERROR,
        f"the type map names table {table!r} for type {type_name!r}, but the database has no such table",
        1,
        (table,),
    )


def _columns(connection: Any, table: str) -> dict[str, str]:
    return {row["name"]: str(row["type"]) for row in connection.execute(f'pragma table_info("{table}")')}


# -------------------------------------------------------------- native tables
def to_tables(log: OCEventLog) -> dict[str, pd.DataFrame]:
    """The log as six flat, immutable-in-spirit tables.

    ``events``, ``event_attributes``, ``objects``, ``object_attributes``,
    ``e2o`` and ``o2o``. This is the shape the bounded joins of
    :mod:`wise.oc.units` work on, and the shape a caller can hand to pandas
    without learning a graph query language.
    """
    return {
        "events": pd.DataFrame(
            [{"event_id": e.event_id, "activity": e.activity, "timestamp": e.timestamp} for e in log.events],
            columns=["event_id", "activity", "timestamp"],
        ),
        "event_attributes": pd.DataFrame(
            [
                {"event_id": e.event_id, "attribute": key, "value": value}
                for e in log.events
                for key, value in e.attributes.items()
            ],
            columns=["event_id", "attribute", "value"],
        ),
        "objects": pd.DataFrame([o.to_row() for o in log.objects], columns=["object_id", "object_type"]),
        "object_attributes": pd.DataFrame(
            [c.to_row() for c in log.attribute_history], columns=["object_id", "attribute", "timestamp", "value"]
        ),
        "e2o": pd.DataFrame([link.to_row() for link in log.e2o], columns=["event_id", "object_id", "qualifier"]),
        "o2o": pd.DataFrame(
            [rel.to_row() for rel in log.o2o], columns=["source_id", "target_id", "qualifier", "valid_from", "valid_to"]
        ),
    }


def from_tables(
    tables: Mapping[str, pd.DataFrame],
    *,
    source: SourceMetadata | None = None,
    object_type_attributes: Mapping[str, Mapping[str, str]] | None = None,
    event_type_attributes: Mapping[str, Mapping[str, str]] | None = None,
    on_dangling: str = "error",
    deduplicate: bool = True,
) -> OCEventLog:
    """Rebuild a log from the tables of :func:`to_tables`.

    ``events`` and ``objects`` are required; the other four default to empty.
    NumPy and pandas scalars are converted back to Python values, and every
    flavour of missing becomes ``None``.
    """
    for required in ("events", "objects"):
        if required not in tables:
            raise OCValidationError(f"a native table set needs {required!r}; it has {sorted(tables)}")
    unknown = sorted(set(tables) - set(_NATIVE_TABLE_NAMES))
    if unknown:
        raise OCValidationError(f"unknown native table(s) {unknown}; the contract has {list(_NATIVE_TABLE_NAMES)}")

    def rows(name: str) -> list[dict[str, Any]]:
        frame = tables.get(name)
        if frame is None or frame.empty:
            return []
        return frame.to_dict("records")  # type: ignore[return-value]

    attributes: dict[str, dict[str, Any]] = {}
    for row in rows("event_attributes"):
        attributes.setdefault(str(row["event_id"]), {})[str(row["attribute"])] = row["value"]
    events = [
        OCEvent(
            event_id=str(row["event_id"]),
            activity=str(row["activity"]),
            timestamp=row["timestamp"],
            attributes=attributes.get(str(row["event_id"]), {}),
        )
        for row in rows("events")
    ]
    objects = [OCObject(str(row["object_id"]), str(row["object_type"])) for row in rows("objects")]
    history = [
        AttributeChange(
            object_id=str(row["object_id"]),
            attribute=str(row["attribute"]),
            timestamp=row["timestamp"],
            value=row["value"],
        )
        for row in rows("object_attributes")
    ]
    e2o = [E2O(str(row["event_id"]), str(row["object_id"]), str(row["qualifier"])) for row in rows("e2o")]
    o2o = [
        O2O(
            source_id=str(row["source_id"]),
            target_id=str(row["target_id"]),
            qualifier=str(row["qualifier"]),
            valid_from=None if pd.isna(row.get("valid_from")) else row["valid_from"],
            valid_to=None if pd.isna(row.get("valid_to")) else row["valid_to"],
        )
        for row in rows("o2o")
    ]
    return OCEventLog.build(
        events=events,
        objects=objects,
        e2o=e2o,
        o2o=o2o,
        attribute_history=history,
        source=source or SourceMetadata(interchange=NATIVE_TABLES, spec="wise-oc/1", reader="wise.oc.io.from_tables"),
        object_type_attributes=object_type_attributes,
        event_type_attributes=event_type_attributes,
        on_dangling=on_dangling,
        deduplicate=deduplicate,
    )


__all__ = [
    "NATIVE_TABLES",
    "OCEL2_JSON",
    "OCEL2_SQLITE",
    "OCEL_SPEC",
    "LossReport",
    "from_tables",
    "interchange_support",
    "ocel2_json_document",
    "read_ocel2_json",
    "read_ocel2_sqlite",
    "to_tables",
    "write_ocel2_json",
]
