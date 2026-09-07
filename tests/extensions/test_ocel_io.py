"""The object log contract and its adapters (O01).

The interchange these tests pin is **OCEL 2.0 JSON**: ``objectTypes``,
``eventTypes``, ``objects`` and ``events``, with qualified relationships and
timestamped object attribute values. ``tests/fixtures/ocel/p2p_mini.json`` is
that pin — a complete, hand-written document that the writer must reproduce
key for key and list order for list order.

Two further adapters are exercised here and are deliberately *different*
acceptance tests:

* the **OCEL 2.0 SQLite** serialisation, read-only, built inside the test from
  the same fixture so that the schema is pinned without needing an external
  file (a real-world file is checked as well when ``WISE_OCEL2_SQLITE`` points
  at one);
* the **native pandas tables**, which are the library's own round-trip and say
  nothing about conformance to the standard.

The large published procure-to-pay log is checked when ``WISE_OCEL2_JSON``
(and optionally ``WISE_OCEL2_SQLITE``) point at it. Those tests skip rather
than fabricate: the data is not distributed with the package.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from wise import oc
from wise.errors import OCInterchangeError, OCUnitError, OCValidationError
from wise.oc.model import (
    E2O,
    O2O,
    AttributeChange,
    AttributePolicy,
    AttributeReading,
    IssueSeverity,
    OCEvent,
    OCEventLog,
    OCIssueCode,
    OCObject,
    SourceMetadata,
    observed_precision,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ocel"
MINI = FIXTURES / "p2p_mini.json"
DEFECTS = FIXTURES / "p2p_mini_defects.json"

BIG_JSON = os.environ.get("WISE_OCEL2_JSON")
BIG_SQLITE = os.environ.get("WISE_OCEL2_SQLITE")
needs_big_json = pytest.mark.skipif(
    not BIG_JSON or not Path(BIG_JSON).exists(), reason="set WISE_OCEL2_JSON to an OCEL 2.0 JSON log"
)
needs_big_sqlite = pytest.mark.skipif(
    not BIG_SQLITE or not Path(BIG_SQLITE).exists(), reason="set WISE_OCEL2_SQLITE to an OCEL 2.0 SQLite log"
)


@pytest.fixture
def mini() -> OCEventLog:
    return oc.read_ocel2_json(MINI)


def ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text)


def tiny(**changes):
    """A two-object log builder whose parts individual tests can break."""
    payload = {
        "events": [OCEvent("e1", "A", ts("2024-01-01T00:00:00Z"))],
        "objects": [OCObject("o1", "invoice"), OCObject("o2", "purchase_order")],
        "e2o": [E2O("e1", "o1", "invoice")],
        "o2o": [O2O("o1", "o2", "belongs to")],
        "attribute_history": [AttributeChange("o1", "amount", ts("2024-01-01T00:00:00Z"), 10.0)],
    }
    payload.update(changes)
    return OCEventLog.build(**payload)


# ----------------------------------------------------------- declared support
def test_the_supported_interchanges_are_declared_and_narrow():
    support = oc.interchange_support()
    assert sorted(support) == ["ocel2-json", "ocel2-sqlite", "wise-oc-tables"]
    assert support["ocel2-json"] == {**support["ocel2-json"], "spec": "OCEL 2.0", "read": True, "write": True}
    assert support["ocel2-sqlite"]["read"] is True
    assert support["ocel2-sqlite"]["write"] is False, "there is no OCEL 2.0 SQLite writer, and none is claimed"
    for name, declared in support.items():
        assert declared["cannot_carry"], f"{name} must say what it cannot carry, not only what it can"


def test_no_support_is_claimed_for_formats_that_are_merely_readable():
    support = oc.interchange_support()
    assert not {"ocel1", "ocel-xml", "ocel2-xml", "csv"} & set(support)


# ------------------------------------------------------ the pinned OCEL 2 JSON
def test_the_pinned_document_round_trips_key_for_key(mini):
    written, report = oc.ocel2_json_document(mini)
    assert report.lossless
    assert written == json.loads(MINI.read_text(encoding="utf-8")), "the export is not the document that was read"


def test_writing_and_reading_back_preserves_the_content(mini, tmp_path):
    target = tmp_path / "out.json"
    report = oc.write_ocel2_json(mini, target, indent=2)
    assert report.lossless and len(report) == 0
    again = oc.read_ocel2_json(target)
    assert again.content_fingerprint() == mini.content_fingerprint()
    assert again.object_type_attributes == mini.object_type_attributes
    assert again.event_type_attributes == mini.event_type_attributes


def test_identities_types_qualifiers_and_histories_all_survive(mini):
    assert mini.event_ids == ("e1", "e2", "e3", "e4", "e5", "e6")
    assert mini.obj("inv1").object_type == "invoice"
    assert [link.qualifier for link in mini.objects_of("e1")] == ["purchase_order", "item", "item", "item"]
    assert mini.neighbours("inv1", qualifier="belongs to") == ("po1",)
    assert mini.neighbours("po1", direction="reverse", qualifier="belongs to") == ("inv1", "inv2")
    assert [(c.timestamp.isoformat(), c.value) for c in mini.history("inv1", "amount")] == [
        ("2024-01-05T10:00:00+00:00", 100.0),
        ("2024-01-20T10:00:00+00:00", 120.0),
    ]
    assert mini.attributes_of("inv1") == ("amount", "owner")


def test_the_declared_attribute_types_are_kept_and_not_applied(mini):
    assert mini.object_type_attributes["invoice"] == {"amount": "float", "owner": "string"}
    # the declaration says "float"; the value is what the document wrote, untouched
    assert mini.history("inv1", "amount")[0].value == 100.0
    assert mini.event_type_attributes["Execute Payment"] == {"resource": "string"}


def test_the_timestamp_spelling_is_not_the_identity(mini, tmp_path):
    document = json.loads(MINI.read_text(encoding="utf-8"))
    for event in document["events"]:
        event["time"] = event["time"].replace("+00:00", "Z")
    for obj in document["objects"]:
        for attribute in obj["attributes"]:
            attribute["time"] = attribute["time"].replace("+00:00", "Z")
    assert oc.read_ocel2_json(document).content_fingerprint() == mini.content_fingerprint()


def test_two_events_with_one_activity_and_one_timestamp_both_survive(mini):
    both = [e for e in mini.events if e.activity == "Record Invoice Receipt"]
    assert [e.event_id for e in both] == ["e4", "e5"]
    assert both[0].timestamp == both[1].timestamp
    assert mini.events_of("po1", activity="Record Invoice Receipt") == tuple(both), "shared labels are not shared identity"


def test_a_document_without_events_is_not_an_ocel_document():
    with pytest.raises(OCInterchangeError, match="needs 'events'"):
        oc.read_ocel2_json({"objects": []})


def test_invalid_json_is_refused_as_such(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(OCInterchangeError, match="not valid JSON"):
        oc.read_ocel2_json(broken)


def test_an_event_without_an_id_is_refused_rather_than_numbered():
    with pytest.raises(OCInterchangeError, match="no 'id'"):
        oc.read_ocel2_json({"objects": [], "events": [{"type": "A", "time": "2024-01-01T00:00:00Z"}]})


def test_the_read_records_the_source_the_reader_and_the_digest(mini):
    assert mini.source.interchange == "ocel2-json"
    assert mini.source.spec == "OCEL 2.0"
    assert mini.source.reader == "wise.oc.io.read_ocel2_json"
    assert mini.source.digest == hashlib.sha256(MINI.read_bytes()).hexdigest()
    assert mini.source.relation_time_semantics == "atemporal"
    assert mini.source.precision == {"events": "hour", "attribute_history": "hour"}
    assert mini.source.timezone == "UTC"


def test_reading_the_same_file_twice_gives_one_identity():
    assert oc.read_ocel2_json(MINI).source.to_dict() == oc.read_ocel2_json(MINI).source.to_dict()


# --------------------------------------------------- repairs, reports, refusals
def test_identical_repeats_are_removed_with_a_report_and_defects_are_named():
    log = oc.read_ocel2_json(DEFECTS, on_dangling="drop")
    report = log.validation
    assert report.count(OCIssueCode.DUPLICATE_E2O_ROW) == 1
    assert report.count(OCIssueCode.DUPLICATE_O2O_ROW) == 1
    assert report.count(OCIssueCode.DUPLICATE_ATTRIBUTE_ROW) == 1
    assert report.count(OCIssueCode.DANGLING_O2O_TARGET) == 1
    assert report.count(OCIssueCode.MISSING_QUALIFIER) == 1
    assert report.count(OCIssueCode.SELF_RELATION) == 1
    assert not report.clean
    assert len(log.e2o) == 2, "two distinct events referring to one invoice; only the repeated row went"
    assert len(log.attribute_history) == 2, "inv1.amount deduplicated to one row, po1.vendor untouched"
    assert [i.severity for i in report.of(OCIssueCode.DUPLICATE_E2O_ROW)] == [IssueSeverity.REPAIRED]


def test_keys_outside_the_ocel_structure_are_reported_and_not_carried():
    log = oc.read_ocel2_json(DEFECTS, on_dangling="drop")
    unsupported = log.validation.of(OCIssueCode.UNSUPPORTED_FIELD)
    assert len(unsupported) == 1
    assert set(unsupported[0].examples) == {"provenance", "colour"}
    assert unsupported[0].severity is IssueSeverity.DROPPED
    assert "colour" not in json.dumps(oc.ocel2_json_document(log)[0])


def test_two_events_of_the_defect_fixture_survive_their_equal_timestamps():
    log = oc.read_ocel2_json(DEFECTS, on_dangling="drop")
    assert log.event_ids == ("e1", "e2")


def test_a_dangling_relation_is_refused_by_default():
    with pytest.raises(OCValidationError, match="dangling_o2o_target"):
        oc.read_ocel2_json(DEFECTS)


def test_dropping_a_dangling_relation_is_an_explicit_request_that_is_recorded():
    log = oc.read_ocel2_json(DEFECTS, on_dangling="drop")
    issue = log.validation.of(OCIssueCode.DANGLING_O2O_TARGET)[0]
    assert issue.severity is IssueSeverity.REPAIRED
    assert "dropped on request" in issue.message
    assert log.source.options["on_dangling"] == "drop"


def test_deduplication_can_be_switched_off_and_then_repeats_are_errors():
    with pytest.raises(OCValidationError, match="duplicate_e2o_row"):
        oc.read_ocel2_json(DEFECTS, on_dangling="drop", deduplicate=False)


def test_one_event_id_with_two_different_rows_is_a_conflict_not_a_merge():
    with pytest.raises(OCValidationError, match="conflicting_event_id"):
        tiny(
            events=[
                OCEvent("e1", "A", ts("2024-01-01T00:00:00Z")),
                OCEvent("e1", "B", ts("2024-01-01T00:00:00Z")),
            ]
        )


def test_one_object_id_with_two_types_is_a_conflict():
    with pytest.raises(OCValidationError, match="conflicting_object_id"):
        tiny(objects=[OCObject("o1", "invoice"), OCObject("o1", "payment"), OCObject("o2", "purchase_order")])


def test_an_identical_repeated_event_row_is_only_repaired():
    log = tiny(events=[OCEvent("e1", "A", ts("2024-01-01T00:00:00Z"))] * 2)
    assert len(log.events) == 1
    assert log.validation.count(OCIssueCode.DUPLICATE_EVENT_ROW) == 1


def test_two_values_for_one_attribute_at_one_instant_are_a_conflict():
    with pytest.raises(OCValidationError, match="conflicting_attribute_value"):
        tiny(
            attribute_history=[
                AttributeChange("o1", "amount", ts("2024-01-01T00:00:00Z"), 10.0),
                AttributeChange("o1", "amount", ts("2024-01-01T00:00:00Z"), 11.0),
            ]
        )


def test_the_history_of_one_attribute_comes_back_in_time_order():
    log = tiny(
        attribute_history=[
            AttributeChange("o1", "amount", ts("2024-03-01T00:00:00Z"), 30.0),
            AttributeChange("o1", "amount", ts("2024-01-01T00:00:00Z"), 10.0),
            AttributeChange("o1", "amount", ts("2024-02-01T00:00:00Z"), 20.0),
        ]
    )
    assert [c.value for c in log.history("o1", "amount")] == [10.0, 20.0, 30.0]


def test_mixed_time_zones_are_refused_because_they_are_different_clocks():
    with pytest.raises(OCValidationError, match="mixed_timezones"):
        tiny(
            events=[
                OCEvent("e1", "A", ts("2024-01-01T00:00:00Z")),
                OCEvent("e2", "A", pd.Timestamp("2024-01-01T00:00:00")),
            ]
        )


def test_a_missing_timestamp_is_refused():
    with pytest.raises(OCValidationError, match="real timestamp"):
        OCEvent("e1", "A", pd.NaT)


def test_a_nested_attribute_value_is_refused_not_stringified():
    with pytest.raises(OCValidationError, match="must be scalars"):
        OCEvent("e1", "A", ts("2024-01-01T00:00:00Z"), attributes={"lines": [1, 2]})


def test_constructing_the_dataclass_directly_still_checks_the_invariants():
    with pytest.raises(OCValidationError, match="appears twice"):
        OCEventLog(
            events=(OCEvent("e1", "A", ts("2024-01-01T00:00:00Z")), OCEvent("e1", "A", ts("2024-01-01T00:00:00Z"))),
            objects=(),
        )
    with pytest.raises(OCValidationError, match="unknown event"):
        OCEventLog(events=(), objects=(OCObject("o1", "invoice"),), e2o=(E2O("e1", "o1", "invoice"),))


def test_an_unknown_id_is_an_error_not_an_empty_answer(mini):
    with pytest.raises(OCValidationError, match="unknown event id"):
        mini.event("nope")
    with pytest.raises(OCValidationError, match="unknown object id"):
        mini.obj("nope")


# --------------------------------------------------------------- relation time
def test_an_interval_on_an_atemporal_source_is_refused():
    with pytest.raises(OCValidationError, match="atemporal"):
        tiny(o2o=[O2O("o1", "o2", "belongs to", valid_from=ts("2024-01-01T00:00:00Z"))])


def test_an_interval_is_kept_when_the_source_declares_that_it_supplies_them():
    log = tiny(
        o2o=[O2O("o1", "o2", "belongs to", valid_from=ts("2024-01-01T00:00:00Z"), valid_to=ts("2024-06-01T00:00:00Z"))],
        source=SourceMetadata(interchange="in-memory", relation_time_semantics="interval"),
    )
    assert log.o2o[0].valid_to == ts("2024-06-01T00:00:00Z")
    assert log.source.relation_time_semantics == "interval"


def test_an_interval_that_ends_before_it_starts_is_refused():
    with pytest.raises(OCValidationError, match="valid_to precedes valid_from"):
        O2O("o1", "o2", "q", valid_from=ts("2024-06-01T00:00:00Z"), valid_to=ts("2024-01-01T00:00:00Z"))


def test_exporting_intervals_to_ocel_json_refuses_rather_than_loses():
    log = tiny(
        o2o=[O2O("o1", "o2", "belongs to", valid_from=ts("2024-01-01T00:00:00Z"))],
        source=SourceMetadata(interchange="in-memory", relation_time_semantics="interval"),
    )
    with pytest.raises(OCInterchangeError, match="would lose information"):
        oc.ocel2_json_document(log)
    document, report = oc.ocel2_json_document(log, allow_loss=True)
    assert not report.lossless
    assert report.issues[0].code is OCIssueCode.NOT_REPRESENTABLE
    assert "valid_from" not in json.dumps(document)


# ------------------------------------------------------- reading an attribute
def test_an_attribute_is_read_at_the_declared_time_not_at_the_first_value(mini):
    early = mini.value_at("inv1", "amount", ts("2024-01-10T00:00:00Z"))
    late = mini.value_at("inv1", "amount", ts("2024-02-10T00:00:00Z"))
    assert (early.value, early.found, early.later_changes) == (100.0, True, 1)
    assert (late.value, late.effective_from.isoformat()) == (120.0, "2024-01-20T10:00:00+00:00")


def test_before_the_first_change_the_answer_is_not_the_first_value(mini):
    reading = mini.value_at("inv1", "owner", ts("2024-01-01T00:00:00Z"))
    assert reading.found is False and reading.value is None
    assert [q.code.value for q in reading.qualifications] == ["attribute_not_set_at_time"]
    assert "not the next value" in reading.qualifications[0].message


def test_a_changing_owner_is_read_at_the_time_and_not_as_the_current_owner(mini):
    assert mini.value_at("inv1", "owner", ts("2024-01-10T00:00:00Z")).value == "team-a"
    assert mini.value_at("inv1", "owner", ts("2024-02-10T00:00:00Z")).value == "team-b"
    latest = mini.value_at("inv1", "owner", ts("2024-01-10T00:00:00Z"), policy=AttributePolicy.LATEST_KNOWN)
    assert latest.value == "team-b"
    assert [q.code.value for q in latest.qualifications] == ["attribute_read_outside_evaluation_time"]


def test_reading_as_of_requires_a_time(mini):
    with pytest.raises(OCValidationError, match="needs that time"):
        mini.value_at("inv1", "owner", None)


def test_reading_an_attribute_of_an_unknown_object_is_an_error(mini):
    with pytest.raises(OCValidationError, match="unknown object id"):
        mini.value_at("nope", "owner", ts("2024-01-10T00:00:00Z"))


def test_a_reading_that_found_nothing_may_not_carry_a_value():
    with pytest.raises(OCValidationError, match="cannot carry a value"):
        AttributeReading(
            object_id="inv1",
            attribute="owner",
            at=ts("2024-01-01T00:00:00Z"),
            policy=AttributePolicy.AS_OF,
            value="invented",
            found=False,
        )


# ------------------------------------------------------------------ precision
def test_observed_precision_is_measured_from_the_data():
    assert observed_precision([ts("2024-01-01T00:00:00Z")]) == "day"
    assert observed_precision([ts("2024-01-01T10:30:00Z")]) == "minute"
    assert observed_precision([ts("2024-01-01T10:30:05Z")]) == "second"
    assert observed_precision([ts("2024-01-01T10:30:05.500Z")]) == "millisecond"
    assert observed_precision([]) == "none"


def test_the_precision_of_the_pinned_fixture_is_carried_with_it(mini):
    assert mini.source.precision["events"] == "hour"


# ---------------------------------------------------------------- fingerprint
def test_the_content_fingerprint_ignores_where_the_log_was_read_from(mini):
    assert mini.with_source(source="elsewhere").content_fingerprint() == mini.content_fingerprint()


def test_the_content_fingerprint_notices_a_lost_qualifier(mini):
    stripped = OCEventLog.build(
        events=list(mini.events),
        objects=list(mini.objects),
        e2o=[E2O(link.event_id, link.object_id, "") for link in mini.e2o],
        o2o=list(mini.o2o),
        attribute_history=list(mini.attribute_history),
    )
    assert stripped.content_fingerprint() != mini.content_fingerprint()


# --------------------------------------------------------- native table round
def test_the_native_tables_round_trip_every_field(mini):
    tables = oc.to_tables(mini)
    assert sorted(tables) == ["e2o", "event_attributes", "events", "o2o", "object_attributes", "objects"]
    assert list(tables["events"].columns) == ["event_id", "activity", "timestamp"]
    assert len(tables["object_attributes"]) == len(mini.attribute_history)
    again = oc.from_tables(
        tables,
        object_type_attributes=mini.object_type_attributes,
        event_type_attributes=mini.event_type_attributes,
    )
    assert again.content_fingerprint() == mini.content_fingerprint()
    assert again.object_type_attributes == mini.object_type_attributes


def test_the_native_tables_carry_relation_intervals_that_ocel_json_cannot():
    log = tiny(
        o2o=[O2O("o1", "o2", "belongs to", valid_from=ts("2024-01-01T00:00:00Z"))],
        source=SourceMetadata(interchange="in-memory", relation_time_semantics="interval"),
    )
    again = oc.from_tables(
        oc.to_tables(log),
        source=SourceMetadata(interchange="wise-oc-tables", relation_time_semantics="interval"),
    )
    assert again.o2o[0].valid_from == ts("2024-01-01T00:00:00Z")


def test_the_native_tables_refuse_a_table_they_do_not_define(mini):
    tables = oc.to_tables(mini)
    tables["invented"] = pd.DataFrame()
    with pytest.raises(OCValidationError, match="unknown native table"):
        oc.from_tables(tables)


def test_the_native_tables_need_events_and_objects(mini):
    with pytest.raises(OCValidationError, match="needs 'objects'"):
        oc.from_tables({"events": oc.to_tables(mini)["events"]})


# --------------------------------------------------------------- OCEL 2 SQLite
def _sanitise(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum() or ch == "_")


def write_ocel2_sqlite(log: OCEventLog, path: Path) -> None:
    """Write a log in the OCEL 2.0 SQLite serialisation.

    Only in the tests: the library reads this format and does not claim to
    write it. The point of the helper is to pin the schema the reader supports
    without depending on a file that is not distributed with the package.
    """
    connection = sqlite3.connect(path)
    with connection:
        connection.execute('create table "event" (ocel_id TEXT primary key, ocel_type TEXT)')
        connection.execute('create table "event_map_type" (ocel_type TEXT primary key, ocel_type_map TEXT)')
        connection.execute('create table "object" (ocel_id TEXT primary key, ocel_type TEXT)')
        connection.execute('create table "object_map_type" (ocel_type TEXT primary key, ocel_type_map TEXT)')
        connection.execute('create table "event_object" (ocel_event_id TEXT, ocel_object_id TEXT, ocel_qualifier TEXT)')
        connection.execute('create table "object_object" (ocel_source_id TEXT, ocel_target_id TEXT, ocel_qualifier TEXT)')

        for activity in log.activities:
            suffix = _sanitise(activity)
            columns = log.event_type_attributes.get(activity, {})
            declared = ", ".join(f'"{_sanitise(a)}" {"REAL" if t == "float" else "TEXT"}' for a, t in columns.items())
            connection.execute(
                f'create table "event_{suffix}" (ocel_id TEXT primary key, ocel_time TIMESTAMP'
                + (f", {declared}" if declared else "")
                + ")"
            )
            connection.execute("insert into event_map_type values (?, ?)", (activity, suffix))
        for event in log.events:
            suffix = _sanitise(event.activity)
            names = list(log.event_type_attributes.get(event.activity, {}))
            placeholders = ", ".join("?" for _ in range(2 + len(names)))
            connection.execute(
                f'insert into "event_{suffix}" values ({placeholders})',
                [event.event_id, event.timestamp.isoformat(), *[event.attributes.get(n) for n in names]],
            )
            connection.execute("insert into event values (?, ?)", (event.event_id, event.activity))

        for object_type in log.object_types:
            suffix = _sanitise(object_type)
            columns = log.object_type_attributes.get(object_type, {})
            declared = ", ".join(f'"{_sanitise(a)}" {"REAL" if t == "float" else "TEXT"}' for a, t in columns.items())
            connection.execute(
                f'create table "object_{suffix}" (ocel_id TEXT, ocel_time TIMESTAMP'
                + (f", {declared}" if declared else "")
                + ", ocel_changed_field TEXT)"
            )
            connection.execute("insert into object_map_type values (?, ?)", (object_type, suffix))
        for obj in log.objects:
            connection.execute("insert into object values (?, ?)", (obj.object_id, obj.object_type))
            suffix = _sanitise(obj.object_type)
            names = list(log.object_type_attributes.get(obj.object_type, {}))
            changes = [c for c in log.attribute_history if c.object_id == obj.object_id]
            if not changes:
                continue
            first = min(c.timestamp for c in changes)
            initial = {c.attribute: c.value for c in changes if c.timestamp == first}
            placeholders = ", ".join("?" for _ in range(3 + len(names)))
            connection.execute(
                f'insert into "object_{suffix}" values ({placeholders})',
                [obj.object_id, first.isoformat(), *[initial.get(n) for n in names], None],
            )
            for change in changes:
                if change.timestamp == first:
                    continue
                values = [change.value if n == change.attribute else None for n in names]
                connection.execute(
                    f'insert into "object_{suffix}" values ({placeholders})',
                    [obj.object_id, change.timestamp.isoformat(), *values, change.attribute],
                )

        connection.executemany(
            "insert into event_object values (?, ?, ?)", [(link.event_id, link.object_id, link.qualifier) for link in log.e2o]
        )
        connection.executemany(
            "insert into object_object values (?, ?, ?)", [(r.source_id, r.target_id, r.qualifier) for r in log.o2o]
        )
    connection.close()


def test_the_sqlite_serialisation_reads_back_to_the_same_log(mini, tmp_path):
    database = tmp_path / "mini.sqlite"
    write_ocel2_sqlite(mini, database)
    again = oc.read_ocel2_sqlite(database)
    assert again.content_fingerprint() == mini.content_fingerprint()
    assert again.object_type_attributes == mini.object_type_attributes
    assert again.source.interchange == "ocel2-sqlite"


def test_the_sqlite_reader_declares_that_attribute_names_are_sanitised(mini, tmp_path):
    database = tmp_path / "mini.sqlite"
    write_ocel2_sqlite(mini, database)
    again = oc.read_ocel2_sqlite(database)
    dropped = again.validation.of(OCIssueCode.UNSUPPORTED_FIELD)
    assert len(dropped) == 1 and dropped[0].severity is IssueSeverity.DROPPED
    assert "not recoverable" in dropped[0].message
    assert "sanitised" in " ".join(again.source.notes)


def test_a_database_that_is_not_the_ocel_serialisation_is_refused(tmp_path):
    database = tmp_path / "other.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("create table something (a TEXT)")
    connection.close()
    with pytest.raises(OCInterchangeError, match="is missing"):
        oc.read_ocel2_sqlite(database)


# ------------------------------------------------- the published p2p log (opt in)
@needs_big_json
def test_the_published_p2p_log_reads_with_its_defects_named():
    log = oc.read_ocel2_json(BIG_JSON, on_dangling="drop")
    summary = log.summary()
    assert (summary["n_events"], summary["n_objects"], summary["n_e2o"]) == (14671, 9543, 35927)
    assert summary["n_o2o"] == 18374, "20402 relation rows, 2028 of which point at an undeclared invoice receipt"
    assert summary["n_attribute_changes"] == 78213, "78508 history rows, 295 of them identical repeats"
    assert log.validation.count(OCIssueCode.DANGLING_O2O_TARGET) == 2028
    assert log.validation.count(OCIssueCode.DUPLICATE_ATTRIBUTE_ROW) == 295
    assert summary["precision"] == {"events": "minute", "attribute_history": "minute"}


@needs_big_json
def test_the_published_p2p_log_is_refused_by_default_because_of_those_defects():
    with pytest.raises(OCValidationError, match="dangling_o2o_target"):
        oc.read_ocel2_json(BIG_JSON)


@needs_big_json
def test_the_published_p2p_log_round_trips_through_both_writers():
    log = oc.read_ocel2_json(BIG_JSON, on_dangling="drop")
    document, report = oc.ocel2_json_document(log)
    assert report.lossless
    assert oc.read_ocel2_json(document, on_dangling="drop").content_fingerprint() == log.content_fingerprint()
    assert oc.from_tables(oc.to_tables(log)).content_fingerprint() == log.content_fingerprint()


@needs_big_json
@needs_big_sqlite
def test_the_two_adapters_agree_on_everything_except_the_names_sqlite_cannot_keep():
    from_json = oc.read_ocel2_json(BIG_JSON, on_dangling="drop")
    from_sqlite = oc.read_ocel2_sqlite(BIG_SQLITE, on_dangling="drop")
    assert {(e.event_id, e.activity, e.timestamp) for e in from_json.events} == {
        (e.event_id, e.activity, e.timestamp) for e in from_sqlite.events
    }
    assert {(o.object_id, o.object_type) for o in from_json.objects} == {
        (o.object_id, o.object_type) for o in from_sqlite.objects
    }
    assert {link.key for link in from_json.e2o} == {link.key for link in from_sqlite.e2o}
    assert {rel.key for rel in from_json.o2o} == {rel.key for rel in from_sqlite.o2o}
    assert {(c.object_id, c.timestamp) for c in from_json.attribute_history} == {
        (c.object_id, c.timestamp) for c in from_sqlite.attribute_history
    }
    json_names = {c.attribute for c in from_json.attribute_history}
    sqlite_names = {c.attribute for c in from_sqlite.attribute_history}
    assert json_names != sqlite_names, "the documented sanitisation loss, asserted rather than assumed"
    assert {n.replace(" ", "").replace("(", "").replace(")", "").replace("-", "") for n in json_names} == sqlite_names


# ------------------------------------------------------------ the case log is untouched
def test_importing_wise_does_not_import_the_object_package():
    proc = subprocess.run(
        [sys.executable, "-c", "import sys, wise; assert 'wise.oc' not in sys.modules; print(wise.__version__)"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_the_object_package_is_not_added_to_the_top_level_exports():
    import wise

    assert not {name for name in wise.__all__ if name.startswith("OC") or name in {"oc", "AssessmentUnit"}}


def test_importing_the_object_package_needs_no_optional_package_and_no_database_driver():
    code = (
        "import sys, socket\n"
        "socket.socket = None\n"
        "import wise.oc\n"
        "assert 'sqlite3' not in sys.modules, 'the SQLite adapter must import its driver inside the reader'\n"
        "assert 'pm4py' not in sys.modules and 'scipy' not in sys.modules\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_the_object_errors_live_in_the_one_hierarchy():
    import wise

    for error in (OCValidationError, OCInterchangeError, OCUnitError):
        assert issubclass(error, wise.WiseError)
        assert not issubclass(error, wise.EvidenceError)


def test_an_empty_log_is_a_log_and_says_its_precision_is_nothing():
    empty = OCEventLog.build(events=[], objects=[])
    assert empty.summary()["n_events"] == 0
    assert empty.source.precision == {"events": "none", "attribute_history": "none"}
    assert empty.source.timezone is None
    assert empty.validation.clean and not empty.validation


def test_a_nan_attribute_value_cannot_be_written_as_itself():
    log = tiny(attribute_history=[AttributeChange("o1", "amount", ts("2024-01-01T00:00:00Z"), float("nan"))])
    with pytest.raises(OCInterchangeError, match="would lose information"):
        oc.ocel2_json_document(log)
    document, report = oc.ocel2_json_document(log, allow_loss=True)
    assert not report.lossless
    assert document["objects"][0]["attributes"][0]["value"] is None
