"""The evidence contract itself: invariants, units, witnesses, JSON (E02, E04).

These tests exercise the types without scoring anything, so they pin what an
evidence row is *allowed to say* independently of how it was produced.
"""

from __future__ import annotations

import json

import pytest

from wise.errors import EvidenceError
from wise.evidence.models import (
    EVALUABLE_REASONS,
    AbsenceSearch,
    Completeness,
    CoverageReport,
    DiagnosticResult,
    EvaluationRecord,
    Measurement,
    MeasurementKind,
    Qualification,
    QualificationCode,
    ReasonCode,
    SourceIdentity,
    Truncation,
    ViewAnnotation,
    WitnessKind,
    WitnessRef,
)


def record(**changes):
    base = {
        "run_id": "run-1",
        "evaluation_id": "run-1:c1:A",
        "unit_id": "A",
        "unit_type": "case",
        "constraint_id": "c1",
        "constraint_type": "presence",
        "in_scope": True,
        "evaluable": True,
        "reason_code": ReasonCode.OBSERVED,
        "violation": 0.25,
    }
    return EvaluationRecord(**{**base, **changes})


# ------------------------------------------------------------------- reasons
def test_every_reason_code_declares_whether_it_carries_a_violation():
    for code in ReasonCode:
        assert isinstance(code.evaluable, bool)
        assert code.evaluable == (code in EVALUABLE_REASONS)


def test_the_reason_taxonomy_covers_the_required_distinctions():
    values = {c.value for c in ReasonCode}
    assert {
        "out_of_scope",
        "skipped_missing_activation",
        "skipped_missing_response",
        "missing_attribute",
        "ambiguous_match",
        "missing_source_identity",
        "open_observation_window",
        "budget_truncated",
    } <= values


def test_a_missing_endpoint_under_violate_is_an_evaluated_violation():
    """Policy-based violation, not an unevaluable check (F08)."""
    assert ReasonCode.POLICY_VIOLATION_MISSING_RESPONSE.evaluable
    r = record(reason_code=ReasonCode.POLICY_VIOLATION_MISSING_RESPONSE, violation=1.0)
    assert r.evaluable and r.violation == 1.0


def test_a_censored_check_is_evaluated_but_only_as_a_lower_bound():
    r = record(
        reason_code=ReasonCode.OPEN_OBSERVATION_WINDOW,
        violation=0.2,
        measurements=(Measurement("elapsed_at_horizon", 14.0, "D", MeasurementKind.DURATION, lower_bound=True),),
        qualifications=(Qualification(QualificationCode.LOWER_BOUND, "not a completed duration"),),
    )
    assert r.evaluable
    assert r.measurement["elapsed_at_horizon"].lower_bound
    assert QualificationCode.LOWER_BOUND in {q.code for q in r.qualifications}


# ---------------------------------------------------------------- invariants
def test_out_of_scope_is_never_evaluable():
    with pytest.raises(EvidenceError, match="out-of-scope check cannot be evaluable"):
        record(in_scope=False, evaluable=True)


def test_out_of_scope_must_carry_the_out_of_scope_reason():
    with pytest.raises(EvidenceError, match="out_of_scope reason"):
        record(in_scope=False, evaluable=False, reason_code=ReasonCode.MISSING_ATTRIBUTE, violation=None)


def test_an_unevaluated_check_has_no_violation_not_a_zero():
    with pytest.raises(EvidenceError, match="not a zero penalty"):
        record(evaluable=False, reason_code=ReasonCode.MISSING_ATTRIBUTE, violation=0.0)


def test_an_evaluated_check_needs_a_violation_in_the_unit_interval():
    with pytest.raises(EvidenceError, match=r"violation in \[0, 1\]"):
        record(violation=1.5)
    with pytest.raises(EvidenceError, match="violation in"):
        record(violation=None)


def test_reason_and_evaluable_must_agree():
    with pytest.raises(EvidenceError, match="disagree"):
        record(reason_code=ReasonCode.MISSING_ATTRIBUTE, violation=0.5)


def test_nan_is_a_status_not_a_value():
    with pytest.raises(EvidenceError, match="NaN is a status"):
        record(violation=float("nan"))


def test_duplicate_measurement_names_are_rejected():
    with pytest.raises(EvidenceError, match="duplicate measurement"):
        record(
            measurements=(
                Measurement("lag", 1.0, "D", MeasurementKind.DURATION),
                Measurement("lag", 2.0, "D", MeasurementKind.DURATION),
            )
        )


# -------------------------------------------------------------- measurements
def test_a_measurement_must_declare_a_unit():
    with pytest.raises(EvidenceError, match="unit"):
        Measurement("lag", 1.0, "", MeasurementKind.DURATION)


def test_a_lag_keeps_both_endpoints_and_the_duration():
    """Structured measurement, not one unexplained raw value."""
    r = record(
        constraint_type="lag",
        measurements=(
            Measurement("activation_timestamp", "2024-01-01T00:00:00", "timestamp", MeasurementKind.TIMESTAMP),
            Measurement("response_timestamp", "2024-01-26T00:00:00", "timestamp", MeasurementKind.TIMESTAMP),
            Measurement("lag", 25.0, "D", MeasurementKind.DURATION),
        ),
    )
    assert set(r.measurement) == {"activation_timestamp", "response_timestamp", "lag"}
    assert r.measurement["lag"].unit == "D"


def test_a_balance_keeps_both_totals_their_units_and_the_mismatch():
    r = record(
        constraint_type="balance",
        measurements=(
            Measurement("total_x", 82.0, "amount", MeasurementKind.AMOUNT),
            Measurement("total_y", 100.0, "amount", MeasurementKind.AMOUNT),
            Measurement("relative_mismatch", 0.18, "ratio", MeasurementKind.RATIO),
        ),
    )
    assert [m.unit for m in r.measurements] == ["amount", "amount", "ratio"]


def test_a_timestamp_measurement_is_an_iso_string_or_null():
    with pytest.raises(EvidenceError, match="ISO-8601"):
        Measurement("t", 1700000000.0, "timestamp", MeasurementKind.TIMESTAMP)
    assert Measurement("t", None, "timestamp", MeasurementKind.TIMESTAMP).value is None


# ----------------------------------------------------------------- witnesses
def test_an_event_witness_declares_which_identity_it_has():
    source = WitnessRef(WitnessKind.EVENT, "activation", "A", SourceIdentity.SOURCE, event_id="EV-9")
    local = WitnessRef(WitnessKind.EVENT, "activation", "A", SourceIdentity.SNAPSHOT_LOCAL, reference="snapshot-local:row=3")
    assert source.witness_id == "EV-9"
    assert local.witness_id == "snapshot-local:row=3"


def test_a_snapshot_local_reference_may_not_pose_as_a_source_id():
    with pytest.raises(EvidenceError, match="source event id must declare identity=SOURCE"):
        WitnessRef(WitnessKind.EVENT, "r", "A", SourceIdentity.SNAPSHOT_LOCAL, event_id="EV-9")
    with pytest.raises(EvidenceError, match="snapshot-local row reference"):
        WitnessRef(WitnessKind.EVENT, "r", "A", SourceIdentity.SOURCE, reference="snapshot-local:row=3")
    with pytest.raises(EvidenceError, match="exactly one"):
        WitnessRef(WitnessKind.EVENT, "r", "A", SourceIdentity.SOURCE)


def test_absence_has_no_event_identity():
    search = AbsenceSearch(
        unit_id="E",
        activities=("Record Invoice Receipt",),
        window_start="2024-01-01T00:00:00",
        window_end="2024-01-31T00:00:00",
        completeness=Completeness.ASSUMED_COMPLETE,
        n_events_searched=2,
        filters={"dedupe": False},
    )
    witness = WitnessRef(WitnessKind.ABSENCE, "searched", "E", search=search)
    assert witness.event_id is None and witness.reference is None
    assert witness.identity is SourceIdentity.NONE
    assert witness.to_dict()["search"]["completeness"] == "assumed_complete"
    with pytest.raises(EvidenceError, match="absence witness must not reference an event"):
        WitnessRef(WitnessKind.ABSENCE, "searched", "E", SourceIdentity.NONE, event_id="made-up", search=search)
    with pytest.raises(EvidenceError, match="declare the search"):
        WitnessRef(WitnessKind.ABSENCE, "searched", "E")


def test_an_absence_search_must_say_what_it_searched_for():
    with pytest.raises(EvidenceError, match="declare what it searched for"):
        AbsenceSearch("E", (), None, None, Completeness.UNKNOWN)


def test_witness_totals_may_not_be_smaller_than_the_witnesses_kept():
    witness = WitnessRef(WitnessKind.ATTRIBUTE, "attribute", "A", attribute="touches")
    with pytest.raises(EvidenceError, match="smaller than the witnesses kept"):
        record(witnesses=(witness,), n_witnesses_total=0)


# ----------------------------------------------------------- view separation
def test_view_annotations_carry_no_observation_of_their_own():
    fields = {f for f in ViewAnnotation.__dataclass_fields__}
    assert "violation" not in fields and "measurements" not in fields
    assert {"effective_weight", "penalty", "scored", "view"} <= fields


# --------------------------------------------------------------- diagnostics
def test_a_diagnostic_states_its_denominator_and_its_policy():
    d = DiagnosticResult(
        name="event_replication",
        policy="timestamp_multiplicity_within_case",
        numerator=3,
        denominator=12,
        unit_of_counting="cases",
        scope="all cases",
        interpretation="timestamp ties, not identified duplicates",
        threshold=2.0,
    )
    assert d.value == pytest.approx(0.25)
    assert d.to_dict()["policy"] == "timestamp_multiplicity_within_case"


def test_a_zero_denominator_is_undefined_not_zero():
    d = DiagnosticResult("x", "p", 0, 0, "cases", "empty", "nothing in scope")
    assert d.value is None
    assert QualificationCode.ZERO_DENOMINATOR in {q.code for q in d.qualifications}


def test_a_diagnostic_needs_both_numbers():
    with pytest.raises(EvidenceError, match="needs a numerator"):
        DiagnosticResult("x", "p", None, 1, "cases", "s", "i")


def test_coverage_counts_must_add_up():
    with pytest.raises(EvidenceError, match="add up"):
        CoverageReport("case", 5, 30, 25, 25, 4, 0)
    with pytest.raises(EvidenceError, match="add up"):
        CoverageReport("case", 5, 30, 25, 20, 5, 0)


def test_coverage_is_not_confidence():
    report = CoverageReport("case", 5, 30, 25, 20, 5, 5, n_scored={"F": 4}, n_unscored={"F": 1})
    assert report.evaluated_share == pytest.approx(0.8)
    assert "not a probability" in report.interpretation
    assert not any("confidence" in f for f in CoverageReport.__dataclass_fields__)


def test_an_empty_scope_leaves_the_share_undefined():
    empty = CoverageReport("case", 0, 0, 0, 0, 0, 0)
    assert empty.evaluated_share is None and empty.scope_share is None


# --------------------------------------------------------------- truncation
def test_display_truncation_and_evaluation_truncation_are_different():
    display = Truncation(records_captured=10, records_total=30, witnesses_captured=2, witnesses_total=9)
    assert display.records_truncated and display.witnesses_truncated
    assert not display.evaluation_truncated and not display.complete
    cut = Truncation(records_captured=30, records_total=30, evaluation_truncated=True)
    assert not cut.records_truncated and not cut.complete


# --------------------------------------------------------------------- JSON
def test_json_export_uses_explicit_nulls_and_no_nan():
    r = record(
        evaluable=False,
        reason_code=ReasonCode.MISSING_ATTRIBUTE,
        violation=None,
        measurements=(Measurement("touches", None, "touches", MeasurementKind.INDEX),),
    )
    text = json.dumps(r.to_dict(), allow_nan=False)
    assert '"violation": null' in text
    assert '"reason_code": "missing_attribute"' in text
    assert "NaN" not in text


# ------------------------------------------------------------- export/reload
def _censored_capture():
    """A ``full`` capture holding one censored record, one absence and one plain check."""
    import pandas as pd

    import wise

    rows = [("open", "GR", 0, 1, "F"), ("done", "GR", 0, 1, "F"), ("done", "INV", 2, 1, "F")]
    frame = pd.DataFrame(rows, columns=["case", "activity", "day", "amount", "flow_type"])
    frame["time"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(frame["day"], unit="D")
    log = wise.EventLog(
        frame,
        case_col="case",
        activity_col="activity",
        timestamp_col="time",
        case_attributes=["flow_type"],
        window=("2024-01-01", "2024-01-26"),
    )
    norm = wise.Norm(
        constraints=(
            wise.NormConstraint("c_lag", "L", wise.Lag("GR", "INV", delta=10, width=20, missing_b="censor")),
            wise.NormConstraint("c_excl", "L", wise.Exclusion("BLOCK")),
        ),
        layers=(wise.Layer("L"),),
        views=(wise.View("V", constraint_weights={"c_lag": 1.0, "c_excl": 1.0}),),
    )
    return wise.score(log, norm, evidence="full").evidence


def test_a_packet_round_trips_through_its_own_json():
    """ "Distinguishable after export **and reload**" needs a way back."""
    from wise.evidence.models import EvidencePacket

    packet = _censored_capture()
    again = EvidencePacket.from_dict(json.loads(packet.to_json()))

    assert again.records == packet.records, "the records are the evidence; they must survive byte for byte"
    assert again.annotations == packet.annotations
    assert again.coverage == packet.coverage
    assert again.qualifications == packet.qualifications
    assert again.truncation == packet.truncation
    assert again.capture_mode == packet.capture_mode
    assert again.manifest.to_dict() == packet.manifest.to_dict()
    assert again.manifest.config_fingerprint() == packet.manifest.config_fingerprint()

    # the censored record is still a censored record, not a satisfied one
    censored = again.record(f"{again.run_id}:c_lag:open")
    assert censored.reason_code is ReasonCode.OPEN_OBSERVATION_WINDOW
    assert censored.evaluable and censored.violation is not None
    assert QualificationCode.LOWER_BOUND in {q.code for q in censored.qualifications}
    assert censored.measurement["elapsed_at_horizon"].lower_bound is True
    assert censored.measurement["lag"].value is None

    # an unevaluated check keeps its reason and its null, and the three
    # questions are still three answers after the reload
    absence = again.record(f"{again.run_id}:c_excl:open")
    assert absence.in_scope and absence.evaluable
    assert [w.kind for w in absence.witnesses] == [WitnessKind.ABSENCE]
    assert absence.witnesses[0].search.completeness is Completeness.ASSUMED_COMPLETE

    # the export is stable: writing the restored packet gives the same bytes
    assert EvidencePacket.from_dict(json.loads(again.to_json())).records == packet.records


def test_a_restored_packet_declares_that_it_has_no_snapshot_and_no_norm():
    """The absent snapshot and norm are declared, not two silent ``None``\\ s."""
    from wise.errors import EvidenceUnavailableError
    from wise.evidence.models import EvidencePacket

    captured = _censored_capture()
    again = EvidencePacket.from_dict(json.loads(captured.to_json(include_witnesses=False)))
    assert again.restored is True
    assert again.snapshot is None and again.norm is None
    assert "restored from an export" in repr(again)
    access = again.to_dict()["witness_access"]
    assert access["restored_from_export"] is True
    assert access["snapshot_available"] is False and access["norm_available"] is False

    # dropping the witnesses on export drops the materialised flag with them, so
    # asking for one is refused rather than answered with an empty tuple
    with pytest.raises(EvidenceUnavailableError, match="restored from an export"):
        again.witnesses(f"{again.run_id}:c_lag:open")


def test_a_reloaded_packet_is_checked_against_the_invariants_it_was_written_under():
    """A file is data, not a licence: the read path re-checks every rule."""
    from wise.evidence.models import EvidencePacket

    payload = json.loads(_censored_capture().to_json())
    broken = json.loads(json.dumps(payload))
    row = next(r for r in broken["records"] if r["reason_code"] == "open_observation_window")
    row["reason_code"] = "skipped_missing_response"
    row["evaluable"] = False  # ... while the file keeps the violation
    with pytest.raises(EvidenceError, match="not a zero penalty"):
        EvidencePacket.from_dict(broken)

    wrong_version = json.loads(json.dumps(payload))
    wrong_version["schema_version"] = "wise-evidence/99"
    with pytest.raises(EvidenceError, match="unsupported evidence schema version"):
        EvidencePacket.from_dict(wrong_version)


def test_load_evidence_reads_a_written_packet(tmp_path):
    from wise.evidence import load_evidence

    packet = _censored_capture()
    path = tmp_path / "evidence.json"
    path.write_text(packet.to_json(), encoding="utf-8")
    assert load_evidence(path).records == packet.records

    not_json = tmp_path / "broken.json"
    not_json.write_text("{not json", encoding="utf-8")
    with pytest.raises(EvidenceError, match="not a JSON evidence packet"):
        load_evidence(not_json)

    a_list = tmp_path / "list.json"
    a_list.write_text("[]", encoding="utf-8")
    with pytest.raises(EvidenceError, match="a JSON object"):
        load_evidence(a_list)
