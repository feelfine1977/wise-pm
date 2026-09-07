"""Typed diagnostics, denominators and coverage (E04).

The point of the wrappers is that the numbers do not change and their meaning
becomes explicit: what is counted, over what denominator, under which policy,
and what the number does *not* prove.
"""

from __future__ import annotations

import pandas as pd
import pytest

import wise
from wise.evidence import DiagnosticResult, coverage_report
from wise.evidence.models import QualificationCode, SourceIdentity


def make_log(rows, attrs=("flow_type",), **kwargs):
    df = pd.DataFrame(rows, columns=["case", "activity", "day", "amount", "flow_type"])
    df["time"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(df["day"], unit="D")
    return wise.EventLog(
        df, case_col="case", activity_col="activity", timestamp_col="time", case_attributes=list(attrs), **kwargs
    )


@pytest.fixture
def replicated():
    """One case whose events all share a timestamp, one that does not."""
    rows = [
        ("tied", "GR", 0, 1, "F"),
        ("tied", "INV", 0, 1, "F"),
        ("tied", "CLR", 0, 1, "F"),
        ("clean", "GR", 0, 1, "F"),
        ("clean", "INV", 1, 1, "F"),
    ]
    return make_log(rows)


# ------------------------------------------------------- wrapping, not redefining
def test_the_typed_wrapper_reports_the_same_flag_as_the_existing_diagnostic(replicated):
    frame = wise.event_replication(replicated)
    typed = wise.typed_event_replication(replicated, ratio_flag=2.0)
    assert typed.numerator == float((frame["replication_ratio"] > 2.0).sum())
    assert typed.denominator == float(len(replicated))
    assert typed.value == pytest.approx(0.5)
    assert typed.unit_of_counting == "cases"
    assert typed.threshold == 2.0


def test_timestamp_multiplicity_is_not_claimed_to_be_event_replication(replicated):
    typed = wise.typed_event_replication(replicated)
    assert typed.policy == "timestamp_multiplicity_within_case"
    assert "not an identified duplicate event" in typed.interpretation
    assert typed.source_identity is SourceIdentity.NONE


def test_cross_case_matching_declares_which_key_established_identity():
    rows = [("a", "POST", 0, 1, "F"), ("b", "POST", 0, 1, "F"), ("a", "X", 1, 1, "F"), ("b", "Y", 1, 1, "F")]
    df = pd.DataFrame(rows, columns=["case", "activity", "day", "amount", "flow_type"])
    df["time"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(df["day"], unit="D")
    df["doc"] = "D1"
    df["eid"] = ["E1", "E2", "E3", "E4"]

    common = {"case_col": "case", "activity_col": "activity", "timestamp_col": "time", "case_attributes": ["flow_type"]}
    without_ids = wise.EventLog(df, **common)
    by_key = wise.typed_cross_case_replication(without_ids, "doc")
    assert by_key.policy == "cross_case_activity_timestamp_key_match"
    assert by_key.source_identity is SourceIdentity.SNAPSHOT_LOCAL
    assert by_key.numerator == 2.0 and by_key.denominator == 2.0
    assert QualificationCode.NO_SOURCE_EVENT_IDENTITY in {q.code for q in by_key.qualifications}
    assert "upper bound" in by_key.qualifications[0].message

    with_ids = wise.EventLog(df, event_id_col="eid", **common)
    by_identity = wise.typed_cross_case_replication(with_ids, "doc")
    assert by_identity.policy == "cross_case_source_event_id_match"
    assert by_identity.source_identity is SourceIdentity.SOURCE
    assert by_identity.numerator == 0.0, "distinct source ids are distinct events"
    assert by_identity.qualifications == ()


def test_a_censoring_diagnostic_records_its_horizon_and_calls_the_duration_a_bound():
    rows = [("open", "GR", 0, 1, "F"), ("closed", "GR", 0, 1, "F"), ("closed", "CLR", 1, 1, "F")]
    log = make_log(rows, window=("2024-01-01", "2024-01-10"))
    typed = wise.typed_right_censored(log, "CLR", window="20D")
    assert typed.numerator == 1.0 and typed.denominator == 2.0
    assert typed.value == pytest.approx(0.5)
    assert "2024-01-10" in typed.scope
    assert "lower bound" in typed.interpretation
    assert typed.name == "right_censored"


def test_a_diagnostic_over_an_empty_log_has_an_undefined_share():
    empty = pd.DataFrame(columns=["case", "activity", "time", "flow_type"])
    empty["time"] = pd.to_datetime(empty["time"])
    log = wise.EventLog(empty, case_col="case", activity_col="activity", timestamp_col="time", case_attributes=["flow_type"])
    typed = wise.typed_event_replication(log)
    assert typed.denominator == 0.0
    assert typed.value is None
    assert QualificationCode.ZERO_DENOMINATOR in {q.code for q in typed.qualifications}


# ---------------------------------------------------------------- coverage
def test_coverage_counts_scope_evaluation_and_scoring_separately():
    log = wise.datasets.running_p2p_log()
    result = wise.score(log, wise.datasets.running_p2p_norm())
    report = coverage_report(result)
    assert report.n_units == 5
    assert report.n_checks == 30
    assert report.n_in_scope == 25 and report.n_out_of_scope == 5
    assert report.n_evaluated == 25 and report.n_unevaluable == 0
    assert report.n_scored == {"Finance": 5, "Logistics": 5}
    assert report.evaluated_share == 1.0
    assert report.scope_share == pytest.approx(25 / 30)


def test_coverage_agrees_with_the_captured_packet():
    log = wise.datasets.running_p2p_log()
    result = wise.score(log, wise.datasets.running_p2p_norm(), evidence="summary")
    standalone = coverage_report(result)
    assert standalone.to_dict() | {"reasons": {}} == result.evidence.coverage.to_dict() | {"reasons": {}}
    assert result.evidence.coverage.reasons["out_of_scope"] == 5


def test_coverage_counts_unevaluable_checks_apart_from_out_of_scope():
    rows = [("a", "GR", 0, 1, "DF1"), ("b", "X", 0, 1, "DF2")]
    log = make_log(rows)
    norm = wise.Norm(
        constraints=(
            wise.NormConstraint("skipped", "L", wise.Lag("GR", "INV", missing_a="skip", missing_b="skip")),
            wise.NormConstraint("scoped", "L", wise.Presence("GR"), applicability={"flow_type": ["DF1"]}),
        ),
        layers=(wise.Layer("L"),),
        views=(wise.View("V", constraint_weights={"skipped": 1.0, "scoped": 1.0}),),
    )
    report = coverage_report(wise.score(log, norm))
    assert report.n_checks == 4
    assert report.n_out_of_scope == 1
    assert report.n_unevaluable == 2
    assert report.n_evaluated == 1


def test_no_api_multiplies_a_score_by_a_confidence():
    """Coverage is not confidence, and nothing here offers to combine them."""
    names = dir(wise) + dir(wise.evidence)
    assert not [n for n in names if "confidence" in n.lower()]
    assert not [f for f in DiagnosticResult.__dataclass_fields__ if "confidence" in f]
    log = wise.datasets.running_p2p_log()
    packet = wise.score(log, wise.datasets.running_p2p_norm(), evidence="summary").evidence
    assert QualificationCode.COVERAGE_IS_NOT_CONFIDENCE in {q.code for q in packet.qualifications}


def test_excluded_units_are_the_unscored_ones_and_never_a_zero():
    log = wise.datasets.running_p2p_log()
    norm = wise.Norm(
        constraints=(wise.NormConstraint("c1", "L", wise.Presence("X"), applicability={"flow_type": ["DF1"]}),),
        layers=(wise.Layer("L"),),
        views=(wise.View("V", constraint_weights={"c1": 1.0}),),
    )
    result = wise.score(log, norm)
    report = coverage_report(result)
    assert report.n_excluded == report.n_unscored == {"V": 5}
    assert report.to_dict()["n_excluded"] == {"V": 5}
    assert result.scores["V"].isna().all(), "excluded means no score, not a score of zero"


# ------------------------------------------------------- bounded captures (F14)
def test_a_bounded_capture_emits_a_records_truncated_qualification():
    """A declared code that no path emits is a field, not a behaviour."""
    from wise.evidence import capture_evidence

    log = wise.datasets.running_p2p_log()
    norm = wise.datasets.running_p2p_norm()
    result = wise.score(log, norm, evidence="summary")
    details = wise.evaluate_detailed(log, norm)[2]

    whole = capture_evidence(result, details=details, mode="summary")
    assert whole.truncation.complete
    assert QualificationCode.RECORDS_TRUNCATED not in {q.code for q in whole.qualifications}
    assert QualificationCode.EVALUATION_TRUNCATED not in {q.code for q in whole.qualifications}

    # units= bounds the display; every constraint is still evaluated on what was kept
    by_unit = capture_evidence(result, details=details, mode="summary", units=["A", "E"])
    unit_note = next(q for q in by_unit.qualifications if q.code is QualificationCode.RECORDS_TRUNCATED)
    assert unit_note.scope == "run"
    assert "12 of 30" in unit_note.message, "the qualification carries the captured and total counts"
    assert not by_unit.truncation.evaluation_truncated
    assert QualificationCode.EVALUATION_TRUNCATED not in {q.code for q in by_unit.qualifications}

    # max_records stops the traversal: the constraints below the cut have no row
    # at all, so the coverage counts are not the run's and say so separately
    budgeted = capture_evidence(result, details=details, mode="summary", max_records=7)
    codes = {q.code for q in budgeted.qualifications}
    assert QualificationCode.RECORDS_TRUNCATED in codes
    assert QualificationCode.EVALUATION_TRUNCATED in codes
    assert budgeted.truncation.evaluation_truncated
    cut = next(q for q in budgeted.qualifications if q.code is QualificationCode.EVALUATION_TRUNCATED)
    assert "c3" in cut.message and "c6" in cut.message, "the constraints never reached are named"
    assert "7 of 30" in next(q for q in budgeted.qualifications if q.code is QualificationCode.RECORDS_TRUNCATED).message


def test_every_declared_truncation_code_has_a_path_that_emits_it():
    """F14's three truncation codes are behaviours now, not declarations."""
    from wise.evidence import capture_evidence

    log = wise.datasets.running_p2p_log()
    norm = wise.datasets.running_p2p_norm()
    result = wise.score(log, norm, evidence="summary")
    details = wise.evaluate_detailed(log, norm)[2]
    emitted = set()
    for kwargs in ({"mode": "full", "witness_limit": 1}, {"max_records": 7}, {"units": ["A"]}):
        packet = capture_evidence(result, details=details, **kwargs)
        emitted |= {q.code for q in packet.qualifications}
        emitted |= {q.code for record in packet.records for q in record.qualifications}
    assert {
        QualificationCode.WITNESSES_TRUNCATED,
        QualificationCode.RECORDS_TRUNCATED,
        QualificationCode.EVALUATION_TRUNCATED,
    } <= emitted
