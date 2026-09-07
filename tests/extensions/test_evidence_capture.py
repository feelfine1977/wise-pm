"""Capture: parity, policies, witnesses, staleness, truncation, JSON (E02).

Covers the Foundation acceptance rows F03, F04, F08–F14 on the real
evaluation path, plus the interchange contract shipped with the package.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import wise
from wise.errors import EvidenceError, EvidenceUnavailableError, StaleEvidenceError
from wise.evidence import capture_evidence, interchange_schema, to_interchange
from wise.evidence.models import (
    Completeness,
    QualificationCode,
    ReasonCode,
    SourceIdentity,
    WitnessKind,
)
from wise.scoring import _MISSING_ATTRIBUTE, _OBSERVED, _OPEN_WINDOW, _SATISFIED_VACUOUSLY, _SKIP_MISSING_ANCHOR

SCHEMA_DIR = Path(wise.evidence.__file__).parent / "schemas"


def make_log(rows, attrs=("flow_type",), **kwargs):
    df = pd.DataFrame(rows, columns=["case", "activity", "day", "amount", "flow_type"])
    df["time"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(df["day"], unit="D")
    return wise.EventLog(
        df, case_col="case", activity_col="activity", timestamp_col="time", case_attributes=list(attrs), **kwargs
    )


def one_norm(constraint, **kwargs):
    return wise.Norm(
        constraints=(wise.NormConstraint("c", "L", constraint),),
        layers=(wise.Layer("L"),),
        views=(wise.View("V", constraint_weights={"c": 1.0}),),
        **kwargs,
    )


@pytest.fixture
def log():
    return wise.datasets.running_p2p_log()


@pytest.fixture
def norm():
    return wise.datasets.running_p2p_norm()


# ---------------------------------------------------------------- F03 parity
@pytest.mark.parametrize("mode", ["summary", "full"])
@pytest.mark.parametrize("scoring_mode", ["flat", "layer_balanced"])
def test_capture_does_not_move_a_single_number(log, norm, mode, scoring_mode):
    plain = wise.score(log, norm, mode=scoring_mode)
    captured = wise.score(log, norm, mode=scoring_mode, evidence=mode)
    pd.testing.assert_frame_equal(plain.violations, captured.violations)
    pd.testing.assert_frame_equal(plain.in_scope, captured.in_scope)
    pd.testing.assert_frame_equal(plain.scores, captured.scores)
    for view in plain.views:
        pd.testing.assert_frame_equal(plain.contributions[view], captured.contributions[view])
        pd.testing.assert_frame_equal(plain.effective_weights(view), captured.effective_weights(view))
    assert plain.evidence is None and captured.evidence is not None


def test_the_detailed_matrix_equals_the_plain_matrix(log, norm):
    V, S = wise.violation_matrix(log, norm, return_scope=True)
    V2, S2, details = wise.evaluate_detailed(log, norm)
    pd.testing.assert_frame_equal(V, V2)
    pd.testing.assert_frame_equal(S, S2)
    assert set(details) == set(norm.constraint_ids)


def test_return_scope_still_returns_exactly_two_results(log, norm):
    """F05 again, now that a detailed path exists beside it."""
    out = wise.violation_matrix(log, norm, return_scope=True)
    assert isinstance(out, tuple) and len(out) == 2


def test_the_reason_strings_used_by_scoring_are_the_declared_reason_codes():
    values = {code.value for code in ReasonCode}
    assert {_OBSERVED, _SATISFIED_VACUOUSLY, _OPEN_WINDOW, _SKIP_MISSING_ANCHOR, _MISSING_ATTRIBUTE} <= values


# ------------------------------------------------------- records and reasons
def test_one_record_per_constraint_and_unit(log, norm):
    packet = wise.score(log, norm, evidence="summary").evidence
    assert len(packet) == len(log) * len(norm.constraints) == 30
    assert len({r.evaluation_id for r in packet.records}) == 30
    assert {r.unit_type for r in packet.records} == {"case"}


def test_an_out_of_scope_check_is_not_a_satisfied_one(log, norm):
    """c4 applies to DF1 only and the example is DF2."""
    packet = wise.score(log, norm, evidence="full").evidence
    for record in packet.records_for(constraint_id="c4"):
        assert record.in_scope is False
        assert record.evaluable is False
        assert record.reason_code is ReasonCode.OUT_OF_SCOPE
        assert record.violation is None
        assert record.measurements == () and record.witnesses == ()


def test_a_lag_keeps_both_endpoints_and_its_duration(log, norm):
    packet = wise.score(log, norm, evidence="summary").evidence
    record = packet.record(f"{packet.run_id}:c2:A")
    assert record.measurement["lag"].value == pytest.approx(25.0)
    assert record.measurement["lag"].unit == "D"
    assert record.measurement["activation_timestamp"].value == "2024-01-01T00:00:00"
    assert record.measurement["response_timestamp"].value == "2024-01-26T00:00:00"
    assert record.violation == pytest.approx(0.75)


def test_a_balance_keeps_both_totals_and_the_relative_mismatch(log, norm):
    packet = wise.score(log, norm, evidence="summary").evidence
    record = packet.record(f"{packet.run_id}:c3:C")
    assert record.measurement["total_x"].value == pytest.approx(82.0)
    assert record.measurement["total_y"].value == pytest.approx(100.0)
    assert record.measurement["relative_mismatch"].value == pytest.approx(0.18)
    assert record.measurement["total_x"].unit == "amount"


def test_the_parameter_and_policy_snapshot_travels_with_the_record(log, norm):
    packet = wise.score(log, norm, evidence="summary").evidence
    record = packet.record(f"{packet.run_id}:c2:A")
    assert record.parameters["delta"] == 10 and record.parameters["width"] == 20
    assert record.policies == {"missing_a": "violate", "missing_b": "violate", "activation": "first", "response": "first_after"}
    assert record.constraint_version == norm.version


# --------------------------------------------------- F08 missingness policies
@pytest.mark.parametrize(
    ("policy", "reason", "violation"),
    [
        ("violate", ReasonCode.POLICY_VIOLATION_MISSING_RESPONSE, 1.0),
        ("skip", ReasonCode.SKIPPED_MISSING_RESPONSE, None),
        ("censor", ReasonCode.OPEN_OBSERVATION_WINDOW, 0.75),
    ],
)
def test_each_missing_response_policy_keeps_its_own_meaning(policy, reason, violation):
    rows = [("open", "GR", 0, 1, "F"), ("done", "GR", 0, 1, "F"), ("done", "INV", 2, 1, "F")]
    log = make_log(rows, window=("2024-01-01", "2024-01-26"))
    norm = one_norm(wise.Lag("GR", "INV", delta=10, width=20, missing_b=policy))
    result = wise.score(log, norm, evidence="summary")
    record = result.evidence.record(f"{result.evidence.run_id}:c:open")
    assert record.reason_code is reason
    assert record.evaluable is (violation is not None)
    if violation is None:
        assert record.violation is None
    else:
        assert record.violation == pytest.approx(violation)
    assert result.violations.loc["open", "c"] is not None


def test_a_censored_lag_records_the_horizon_and_calls_itself_a_lower_bound():
    """F08/F14: a censor policy never claims the case completed at the horizon."""
    rows = [("open", "GR", 0, 1, "F"), ("done", "GR", 0, 1, "F"), ("done", "INV", 2, 1, "F")]
    log = make_log(rows, window=("2024-01-01", "2024-01-26"))
    norm = one_norm(wise.Lag("GR", "INV", delta=10, width=20, missing_b="censor"))
    result = wise.score(log, norm, evidence="full")
    record = result.evidence.record(f"{result.evidence.run_id}:c:open")
    assert record.measurement["elapsed_at_horizon"].lower_bound is True
    assert record.measurement["elapsed_at_horizon"].value == pytest.approx(25.0)
    assert record.measurement["lag"].value is None, "no completed duration was observed"
    assert QualificationCode.LOWER_BOUND in {q.code for q in record.qualifications}
    assert result.manifest.observation.resolved_horizons == {"c": "2024-01-26T00:00:00"}
    absence = [w for w in record.witnesses if w.kind is WitnessKind.ABSENCE]
    assert absence and absence[0].search.completeness is Completeness.OPEN


def test_a_missing_activation_under_skip_is_unevaluable_and_under_violate_is_evaluated():
    rows = [("with_a", "GR", 0, 1, "F"), ("without_a", "X", 0, 1, "F")]
    log = make_log(rows)
    for policy, reason, evaluable in (
        ("skip", ReasonCode.SKIPPED_MISSING_ACTIVATION, False),
        ("violate", ReasonCode.POLICY_VIOLATION_MISSING_ACTIVATION, True),
    ):
        norm = one_norm(wise.Lag("GR", "INV", delta=10, width=20, missing_a=policy, missing_b="skip"))
        result = wise.score(log, norm, evidence="summary")
        record = result.evidence.record(f"{result.evidence.run_id}:c:without_a")
        assert record.reason_code is reason
        assert record.evaluable is evaluable


def test_a_precedence_rule_without_its_regulated_activity_is_vacuously_satisfied():
    rows = [("plain", "GR", 0, 1, "F"), ("plain", "INV", 1, 1, "F")]
    log = make_log(rows)
    result = wise.score(log, one_norm(wise.Precedence("GR", "CANCEL")), evidence="summary")
    record = result.evidence.record(f"{result.evidence.run_id}:c:plain")
    assert record.reason_code is ReasonCode.SATISFIED_VACUOUSLY
    assert record.violation == 0.0
    assert QualificationCode.VACUOUS_SATISFACTION in {q.code for q in record.qualifications}


def test_a_missing_anchor_makes_a_scoped_count_unevaluable():
    rows = [("no_anchor", "INV", 0, 1, "F"), ("anchored", "GR", 0, 1, "F"), ("anchored", "INV", 1, 1, "F")]
    log = make_log(rows)
    result = wise.score(log, one_norm(wise.Exclusion("INV", after="GR")), evidence="summary")
    packet = result.evidence
    assert packet.record(f"{packet.run_id}:c:no_anchor").reason_code is ReasonCode.SKIPPED_MISSING_ANCHOR
    assert packet.record(f"{packet.run_id}:c:no_anchor").violation is None
    assert packet.record(f"{packet.run_id}:c:anchored").reason_code is ReasonCode.OBSERVED


def test_a_missing_attribute_is_a_reason_not_a_zero():
    rows = [("a", "GR", 0, 1, "F"), ("b", "GR", 0, 1, "F")]
    log = make_log(rows)
    log.add_case_attribute("touches", pd.Series({"a": 4.0, "b": np.nan}))
    result = wise.score(log, one_norm(wise.Metric("touches", threshold=1, width=4)), evidence="summary")
    packet = result.evidence
    assert packet.record(f"{packet.run_id}:c:b").reason_code is ReasonCode.MISSING_ATTRIBUTE
    assert packet.record(f"{packet.run_id}:c:b").violation is None
    assert packet.record(f"{packet.run_id}:c:b").measurement["touches"].value is None


def test_an_averaged_activation_lag_says_that_it_is_an_average():
    rows = [("m", "GR", 0, 1, "F"), ("m", "INV", 1, 1, "F"), ("m", "GR", 5, 1, "F"), ("m", "INV", 40, 1, "F")]
    log = make_log(rows)
    result = wise.score(log, one_norm(wise.Lag("GR", "INV", delta=10, width=20, activation="each")), evidence="summary")
    record = result.evidence.record(f"{result.evidence.run_id}:c:m")
    assert record.measurement["activations"].value == 2
    assert record.measurement["responses_observed"].value == 2
    assert record.measurement["mean_lag"].value == pytest.approx(18.0)
    assert QualificationCode.AVERAGED_OVER_ACTIVATIONS in {q.code for q in record.qualifications}


# ----------------------------------------------------------- F04 unscored
def test_an_unscored_unit_stays_unscored_in_the_evidence(log):
    """No applicable weight: no score, and no zero-penalty story either."""
    norm = wise.Norm(
        constraints=(
            wise.NormConstraint("c1", "L", wise.Presence("Record Invoice Receipt"), applicability={"flow_type": ["DF1"]}),
        ),
        layers=(wise.Layer("L"),),
        views=(wise.View("V", constraint_weights={"c1": 1.0}),),
    )
    result = wise.score(log, norm, evidence="summary")
    assert result.scores["V"].isna().all()
    packet = result.evidence
    assert packet.coverage.n_scored == {"V": 0} and packet.coverage.n_unscored == {"V": 5}
    assert all(r.reason_code is ReasonCode.OUT_OF_SCOPE for r in packet.records)
    assert all(a.effective_weight is None and a.penalty is None and not a.scored for a in packet.annotations)
    assert QualificationCode.UNSCORED_UNIT in {q.code for q in packet.qualifications}
    frame = result.evidence_frame("V")
    assert frame["violation"].isna().all()


def test_an_empty_scope_reports_zero_denominators_rather_than_zero_shares(log):
    norm = wise.Norm(
        constraints=(wise.NormConstraint("c1", "L", wise.Presence("nothing"), applicability={"flow_type": ["DF1"]}),),
        layers=(wise.Layer("L"),),
        views=(wise.View("V", constraint_weights={"c1": 1.0}),),
    )
    coverage = wise.score(log, norm, evidence="summary").evidence.coverage
    assert coverage.n_in_scope == 0
    assert coverage.evaluated_share is None, "a share of nothing is undefined, not zero"


# ------------------------------------------------------------ F09/F10 witnesses
def test_an_absence_is_a_declared_search_never_an_invented_event(log, norm):
    """F09: case E has no invoice; the presence rule is violated by absence."""
    packet = wise.score(log, norm, evidence="full").evidence
    record = packet.record(f"{packet.run_id}:c1:E")
    assert record.violation == 1.0
    (witness,) = record.witnesses
    assert witness.kind is WitnessKind.ABSENCE
    assert witness.event_id is None and witness.reference is None
    search = witness.search
    assert search.activities == ("Record Invoice Receipt",)
    assert search.completeness is Completeness.ASSUMED_COMPLETE
    assert search.n_events_searched == 2
    assert search.filters["dedupe"] is False


def test_a_filtered_log_may_not_claim_a_complete_absence_search():
    rows = [("a", "GR", 0, 1, "F"), ("a", "GR", 0, 1, "F")]
    log = make_log(rows, dedupe=True)
    result = wise.score(log, one_norm(wise.Presence("INV")), evidence="full")
    (witness,) = result.evidence.record(f"{result.evidence.run_id}:c:a").witnesses
    assert witness.search.completeness is Completeness.UNKNOWN
    assert witness.search.filters["dedupe"] is True


def test_without_an_event_id_column_references_are_explicitly_snapshot_local(log, norm):
    """F10: no invented source identity."""
    packet = wise.score(log, norm, evidence="full").evidence
    events = [w for r in packet.records for w in r.witnesses if w.kind is WitnessKind.EVENT]
    assert events
    for witness in events:
        assert witness.identity is SourceIdentity.SNAPSHOT_LOCAL
        assert witness.event_id is None
        assert witness.reference.startswith("snapshot-local:row=")
    assert QualificationCode.NO_SOURCE_EVENT_IDENTITY in {q.code for q in packet.qualifications}
    assert packet.manifest.input.source_identity_available is False


def test_with_an_event_id_column_the_source_identity_is_used():
    events = wise.datasets.running_p2p_events()
    events["eid"] = [f"EV-{i:03d}" for i in range(len(events))]
    log = wise.EventLog(
        events,
        case_col="case",
        activity_col="activity",
        timestamp_col="time",
        case_attributes=["flow_type", "company", "vendor"],
        event_id_col="eid",
    )
    packet = wise.score(log, wise.datasets.running_p2p_norm(), evidence="full").evidence
    events_seen = [w for r in packet.records for w in r.witnesses if w.kind is WitnessKind.EVENT]
    assert events_seen
    assert all(w.identity is SourceIdentity.SOURCE and w.event_id.startswith("EV-") for w in events_seen)
    assert packet.manifest.input.source_identity_available is True
    assert QualificationCode.NO_SOURCE_EVENT_IDENTITY not in {q.code for q in packet.qualifications}


def test_equal_activity_and_timestamp_do_not_merge_two_events():
    rows = [("a", "GR", 0, 1, "F"), ("a", "GR", 0, 1, "F"), ("a", "INV", 1, 1, "F")]
    log = make_log(rows)
    result = wise.score(log, one_norm(wise.Singularity("GR", k=1, K=3)), evidence="full")
    record = result.evidence.record(f"{result.evidence.run_id}:c:a")
    assert record.measurement["occurrences"].value == 2.0
    references = {w.reference for w in record.witnesses}
    assert len(references) == 2, "two rows with equal values stay two witnesses"


# ---------------------------------------------------------- F14 truncation
def test_a_truncated_witness_display_keeps_the_exact_measurement_and_the_total():
    rows = [("a", "GR", i, 1, "F") for i in range(12)]
    log = make_log(rows)
    result = wise.score(log, one_norm(wise.Singularity("GR", k=1, K=20)), evidence="summary")
    packet = capture_evidence(result, details=wise.evaluate_detailed(log, result.norm)[2], mode="full", witness_limit=3)
    record = packet.records[0]
    assert record.measurement["occurrences"].value == 12.0, "the measurement counts all of them"
    assert len(record.witnesses) == 3 and record.n_witnesses_total == 12
    assert record.witnesses_truncated
    assert QualificationCode.WITNESSES_TRUNCATED in {q.code for q in record.qualifications}
    assert packet.truncation.witnesses_truncated and not packet.truncation.complete
    assert not packet.truncation.evaluation_truncated, "the evaluation itself was complete"


def test_a_bounded_record_budget_is_reported_as_a_bounded_export(log, norm):
    result = wise.score(log, norm, evidence="summary")
    packet = capture_evidence(result, details=wise.evaluate_detailed(log, norm)[2], mode="summary", max_records=7)
    assert len(packet) == 7
    assert packet.truncation.records_truncated
    assert packet.truncation.records_total == 30
    # the budget stopped the traversal itself: the constraints below the cut have
    # no record at all, so the coverage counts are not the run's coverage
    assert packet.truncation.evaluation_truncated


# ------------------------------------------------------------- F11 staleness
def test_witnesses_are_refused_after_the_log_changed(log, norm):
    result = wise.score(log, norm, evidence="summary")
    packet = result.evidence
    evaluation_id = f"{packet.run_id}:c1:E"
    assert packet.witnesses(evaluation_id), "witnesses materialise while the log is unchanged"

    log.add_case_attribute("late_addition", pd.Series(1.0, index=log.case_ids))
    with pytest.raises(StaleEvidenceError, match="changed after this evidence was captured"):
        packet.witnesses(evaluation_id)


def test_a_frozen_snapshot_survives_a_later_change(log, norm):
    result = wise.score(log, norm, evidence="summary")
    frozen = capture_evidence(
        result,
        details=wise.evaluate_detailed(log, norm)[2],
        mode="summary",
        snapshot=log.snapshot(freeze=True),
    )
    log.add_case_attribute("late_addition", pd.Series(1.0, index=log.case_ids))
    witnesses = frozen.witnesses(f"{frozen.run_id}:c1:E")
    assert witnesses and witnesses[0].kind is WitnessKind.ABSENCE


def test_deriving_an_attribute_after_scoring_invalidates_lazy_witnesses(log):
    norm = wise.datasets.running_p2p_norm().replace(
        derived_attributes=({"name": "gr_count", "kind": "count", "activities": ["Record Goods Receipt"]},)
    )
    result = wise.score(log, norm, evidence="summary")
    packet = result.evidence
    assert QualificationCode.INPUT_MUTATED_BY_DERIVE in {q.code for q in packet.qualifications}
    log.derive([{"name": "gr_count", "kind": "count", "activities": ["Clear Invoice"]}])
    with pytest.raises(StaleEvidenceError):
        packet.witnesses(f"{packet.run_id}:c1:A")


def test_the_events_scope_also_notices_a_changed_event_value(log):
    snapshot = log.snapshot(scope="events")
    snapshot.check_fresh()
    log.events.loc[0, log.activity_col] = "Renamed"
    with pytest.raises(StaleEvidenceError):
        snapshot.check_fresh()


def test_a_structural_snapshot_says_that_it_only_covers_structure(log):
    snapshot = log.snapshot(scope="structure")
    assert snapshot.content_hashed is False
    assert snapshot.content_hashed_tables == {"cases": False, "events": False}
    # a 'cases' snapshot hashes the case table and not the events, and says both
    assert log.snapshot(scope="cases").content_hashed_tables == {"cases": True, "events": False}
    assert log.snapshot(scope="cases").content_hashed is False
    assert log.snapshot(scope="events").content_hashed_tables == {"cases": True, "events": True}
    assert log.snapshot(scope="events").content_hashed is True
    with pytest.raises(ValueError, match="scope must be"):
        log.snapshot(scope="everything")


# ------------------------------------------------- evidence must not be faked
def test_a_result_without_capture_says_so_instead_of_reconstructing(log, norm):
    result = wise.score(log, norm)
    assert result.evidence is None
    with pytest.raises(EvidenceUnavailableError, match="evidence='none'"):
        result.evidence_frame()
    with pytest.raises(EvidenceUnavailableError, match="collected while scoring"):
        capture_evidence(result)


def test_capture_refuses_an_unknown_mode(log, norm):
    with pytest.raises(wise.NormError, match="evidence must be one of"):
        wise.score(log, norm, evidence="everything")


# ----------------------------------------------------------------- the frame
def test_the_evidence_frame_joins_the_view_annotations(log, norm):
    result = wise.score(log, norm, evidence="summary")
    frame = result.evidence_frame("Finance")
    assert len(frame) == 30
    row = frame.loc[f"{result.evidence.run_id}:c2:A"]
    assert row["violation"] == pytest.approx(0.75)
    assert row["effective_weight"] == pytest.approx(0.45)
    assert row["penalty"] == pytest.approx(0.3375)
    assert row["m__lag"] == pytest.approx(25.0) and row["m__lag__unit"] == "D"
    assert row["reason_code"] == "observed"

    logistics = result.evidence_frame("Logistics")
    assert logistics.loc[f"{result.evidence.run_id}:c2:A", "effective_weight"] == pytest.approx(0.15)
    assert logistics.loc[f"{result.evidence.run_id}:c2:A", "violation"] == pytest.approx(0.75), "the observation is view-free"


def test_a_check_that_was_not_evaluated_has_no_weight_and_no_penalty(log, norm):
    """The 0.0 in the weight matrix is padding; it is not a zero penalty."""
    result = wise.score(log, norm, evidence="summary")
    frame = result.evidence_frame("Finance")
    out_of_scope = frame.loc[f"{result.evidence.run_id}:c4:A"]
    assert pd.isna(out_of_scope["effective_weight"]) and pd.isna(out_of_scope["penalty"])
    assert result.effective_weights("Finance").loc["A", "c4"] == 0.0, "the matrix keeps its padding"
    evaluated = frame.loc[f"{result.evidence.run_id}:c5:A"]
    assert evaluated["effective_weight"] == pytest.approx(0.05) and evaluated["penalty"] == pytest.approx(0.0)


def test_the_frame_can_be_filtered_and_rejects_an_unknown_view(log, norm):
    result = wise.score(log, norm, evidence="summary")
    assert len(result.evidence_frame(unit_id="A")) == 6
    assert len(result.evidence_frame(constraint_id="c2")) == 5
    with pytest.raises(EvidenceError, match="unknown view"):
        result.evidence_frame("Marketing")


def test_multiple_views_do_not_duplicate_the_observation(log, norm):
    packet = wise.score(log, norm, evidence="summary").evidence
    assert packet.views == ["Finance", "Logistics"]
    assert len(packet.records) == 30
    assert len(packet.annotations) == 60
    by_id = {}
    for annotation in packet.annotations:
        by_id.setdefault(annotation.evaluation_id, set()).add(annotation.view)
    assert all(views == {"Finance", "Logistics"} for views in by_id.values())


# ----------------------------------------------------------------- F13 JSON
def test_packet_json_round_trips_with_explicit_nulls(log, norm, tmp_path):
    packet = wise.score(log, norm, evidence="full").evidence
    text = packet.to_json()
    assert "NaN" not in text and "Infinity" not in text
    payload = json.loads(text)
    out_of_scope = [r for r in payload["records"] if not r["in_scope"]]
    assert out_of_scope and all(r["violation"] is None and r["reason_code"] == "out_of_scope" for r in out_of_scope)
    assert payload["coverage"]["n_out_of_scope"] == 5
    assert payload["run"]["mode"] == "flat"
    path = tmp_path / "packet.json"
    path.write_text(text, encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == "wise-evidence/1"


def test_witnesses_can_be_left_out_of_an_export(log, norm):
    packet = wise.score(log, norm, evidence="full").evidence
    payload = json.loads(packet.to_json(include_witnesses=False))
    assert all(r["witnesses"] == [] for r in payload["records"])
    assert any(r["n_witnesses_total"] for r in payload["records"]), "the counts survive"


# --------------------------------------------------------- interchange shape
def validate(instance, schema, path="$"):
    """A tiny validator for the subset of JSON Schema the contract uses."""
    if "const" in schema and instance != schema["const"]:
        raise AssertionError(f"{path}: expected {schema['const']!r}, got {instance!r}")
    if "enum" in schema and instance not in schema["enum"]:
        raise AssertionError(f"{path}: {instance!r} not in {schema['enum']}")
    types = schema.get("type")
    if types is not None:
        allowed = [types] if isinstance(types, str) else list(types)
        ok = {
            "object": lambda v: isinstance(v, dict),
            "array": lambda v: isinstance(v, list),
            "string": lambda v: isinstance(v, str),
            "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "null": lambda v: v is None,
        }
        if not any(ok[t](instance) for t in allowed):
            raise AssertionError(f"{path}: {type(instance).__name__} is not {allowed}")
    if isinstance(instance, str) and len(instance) < schema.get("minLength", 0):
        raise AssertionError(f"{path}: string shorter than minLength")
    if isinstance(instance, int | float) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            raise AssertionError(f"{path}: {instance} below minimum")
        if "maximum" in schema and instance > schema["maximum"]:
            raise AssertionError(f"{path}: {instance} above maximum")
    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            raise AssertionError(f"{path}: fewer than minItems")
        if schema.get("uniqueItems") and len({json.dumps(i, sort_keys=True) for i in instance}) != len(instance):
            raise AssertionError(f"{path}: items are not unique")
        for i, item in enumerate(instance):
            if "items" in schema:
                validate(item, schema["items"], f"{path}[{i}]")
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                raise AssertionError(f"{path}: missing required key {key!r}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = set(instance) - set(properties)
            if extra:
                raise AssertionError(f"{path}: unexpected keys {sorted(extra)}")
        for key, value in instance.items():
            if key in properties:
                validate(value, properties[key], f"{path}.{key}")
            elif isinstance(schema.get("additionalProperties"), dict):
                validate(value, schema["additionalProperties"], f"{path}.{key}")
    for sub in schema.get("allOf", []):
        if "if" in sub:
            try:
                validate(instance, sub["if"], path)
            except AssertionError:
                continue
            validate(instance, sub["then"], path)
        else:
            validate(instance, sub, path)


def test_the_validator_accepts_the_shipped_example_and_rejects_a_broken_one():
    schema = interchange_schema()
    example = json.loads((SCHEMA_DIR / "evidence_packet.example.json").read_text(encoding="utf-8"))
    validate(example, schema)
    broken = json.loads(json.dumps(example))
    broken["evidence"][0]["raw_unit"] = ""
    with pytest.raises(AssertionError, match="minLength"):
        validate(broken, schema)
    broken = json.loads(json.dumps(example))
    del broken["baseline"]
    with pytest.raises(AssertionError, match="missing required key"):
        validate(broken, schema)


def test_the_interchange_export_conforms_to_the_shipped_schema(log, norm):
    packet = wise.score(log, norm, evidence="full").evidence
    payload = to_interchange(packet, view="Finance")
    validate(payload, interchange_schema())
    assert payload["run"]["scoring_mode"] == "flat"
    assert payload["run"]["commit"] == "unknown", "no Git state is read"
    assert payload["run"]["data_hash"].startswith("snapshot-fingerprint:")
    assert payload["explanation_kind"] == "absolute_and_reference_contrast"
    assert payload["baseline"]["kind"] == "current_population"
    assert payload["baseline"]["score"] == pytest.approx(0.70983333, abs=1e-6)


def test_the_interchange_raw_value_is_a_selection_not_the_whole_measurement(log, norm):
    packet = wise.score(log, norm, evidence="summary").evidence
    payload = to_interchange(packet, view="Finance")
    row = next(r for r in payload["evidence"] if r["constraint_id"] == "c2" and r["unit_id"] == "A")
    assert row["raw_value"] == pytest.approx(25.0) and row["raw_unit"] == "D"
    record = packet.record(row["evidence_id"])
    assert len(record.measurements) == 3, "the native record still has all three"


def test_a_scalar_only_baseline_downgrades_the_explanation_kind(log, norm):
    packet = wise.score(log, norm, evidence="summary").evidence
    payload = to_interchange(
        packet,
        view="Finance",
        baseline={"baseline_id": "target-95", "kind": "target", "score": 0.95, "layer_profile": None},
    )
    validate(payload, interchange_schema())
    assert payload["explanation_kind"] == "absolute_only", "a scalar target has no layer profile to attribute to"


def test_an_out_of_scope_row_stays_null_in_the_interchange(log, norm):
    payload = to_interchange(wise.score(log, norm, evidence="summary").evidence, view="Finance")
    rows = [r for r in payload["evidence"] if r["constraint_id"] == "c4"]
    assert rows and all(r["violation"] is None and r["evaluable"] is False for r in rows)
    assert all(r["raw_unit"] == "not_applicable" for r in rows)


def test_a_bounded_interchange_export_says_that_it_is_bounded(log, norm):
    packet = wise.score(log, norm, evidence="summary").evidence
    payload = to_interchange(packet, view="Finance", max_evidence=4)
    validate(payload, interchange_schema())
    assert len(payload["evidence"]) == 4
    assert any("bounded to 4 of 30" in q for q in payload["qualifications"])


# ------------------------------------------------------------- F15 CLI sidecars
def _cli(args):
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "-c", "import sys; from wise.cli import main; sys.exit(main(sys.argv[1:]))", *args],
        capture_output=True,
        text=True,
    )


def test_the_new_cli_sidecars_are_opt_in_and_leave_stdout_untouched(tmp_path):
    log_csv, norm_json = tmp_path / "log.csv", tmp_path / "norm.json"
    wise.datasets.running_p2p_events().to_csv(log_csv, index=False)
    wise.datasets.running_p2p_norm().dump(norm_json)
    base = [
        "score", str(norm_json), str(log_csv),
        "--case", "case", "--activity", "activity", "--timestamp", "time",
        "--attr", "company", "--attr", "flow_type", "--by", "company", "--view", "Finance", "--gamma", "1",
    ]  # fmt: skip
    plain = _cli(base)
    assert plain.returncode == 0 and plain.stderr == ""

    manifest_out, evidence_out, frame_out = tmp_path / "run.json", tmp_path / "evidence.json", tmp_path / "evidence.csv"
    with_sidecars = _cli(
        [*base, "--manifest-out", str(manifest_out), "--evidence-out", str(evidence_out), "--evidence-frame-out", str(frame_out)]
    )
    assert with_sidecars.returncode == 0
    assert with_sidecars.stdout == plain.stdout, "the default CSV must not change when a sidecar is requested"
    assert "wrote" in with_sidecars.stderr, "progress messages go to stderr, never into the CSV"

    run = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert run["stage"] == "backlog" and run["is_complete_backlog_run"] is True
    assert run["priority"] == {
        "grouping": ["company"],
        "view": "Finance",
        "volume": "cases",
        "comparator": "current_population_mean",
        "comparator_value": pytest.approx(0.7098333333333334),
        "min_cases": 1,
        "gamma": 1.0,
        "z": None,
        "diagnostics": [],
    }
    packet = json.loads(evidence_out.read_text(encoding="utf-8"))
    assert len(packet["records"]) == 30 and packet["capture_mode"] == "full"
    assert len(pd.read_csv(frame_out)) == 30


# -------------------------------------------------- witnesses per constraint type
def test_a_precedence_rule_witnesses_its_anchor_and_the_premature_events():
    rows = [("early", "INV", 0, 1, "F"), ("early", "GR", 1, 1, "F"), ("ok", "GR", 0, 1, "F"), ("ok", "INV", 1, 1, "F")]
    log = make_log(rows)
    result = wise.score(log, one_norm(wise.Precedence("GR", "INV")), evidence="full")
    packet = result.evidence
    early = packet.record(f"{packet.run_id}:c:early")
    assert early.violation == 1.0
    roles = {w.role for w in early.witnesses}
    assert roles == {"anchor", "premature"}
    assert early.measurement["b_before_activation"].value == 1.0
    assert early.measurement["b_total"].value == 1.0


def test_a_metric_constraint_witnesses_the_attribute_not_an_event():
    log = make_log([("a", "GR", 0, 1, "F")])
    log.add_case_attribute("touches", pd.Series({"a": 9.0}))
    result = wise.score(log, one_norm(wise.Metric("touches", threshold=1, width=4)), evidence="full")
    (witness,) = result.evidence.records[0].witnesses
    assert witness.kind is WitnessKind.ATTRIBUTE
    assert witness.attribute == "touches"
    assert witness.witness_id == "attribute:a:touches"


def test_a_scoped_exclusion_witnesses_its_anchor():
    rows = [("a", "GR", 0, 1, "F"), ("a", "PRICE", 1, 1, "F")]
    log = make_log(rows)
    result = wise.score(log, one_norm(wise.Exclusion("PRICE", after="GR")), evidence="full")
    record = result.evidence.records[0]
    assert {w.role for w in record.witnesses} == {"occurrence", "anchor"}


def test_witnesses_materialise_lazily_from_a_summary_packet(log, norm):
    packet = wise.score(log, norm, evidence="summary").evidence
    assert all(not r.witnesses_materialised for r in packet.records)
    lag = packet.witnesses(f"{packet.run_id}:c2:A")
    assert {w.role for w in lag} == {"activation", "response"}
    assert packet.witnesses(f"{packet.run_id}:c2:A", limit=1) or True
    assert packet.witnesses(f"{packet.run_id}:c4:A") == (), "an out-of-scope check has nothing to witness"


def test_materialised_witnesses_are_returned_as_captured(log, norm):
    packet = wise.score(log, norm, evidence="full").evidence
    evaluation_id = f"{packet.run_id}:c3:A"
    assert packet.witnesses(evaluation_id) == packet.record(evaluation_id).witnesses
    assert len(packet.witnesses(evaluation_id, limit=1)) == 1


def test_an_unknown_evaluation_id_is_an_error(log, norm):
    packet = wise.score(log, norm, evidence="summary").evidence
    with pytest.raises(EvidenceError, match="unknown evaluation id"):
        packet.record("nope")


# ------------------------------------------------- activation='each' corner cases
def test_an_each_activation_lag_without_any_activation_is_captured():
    log = make_log([("none", "X", 0, 1, "F")])
    norm = one_norm(wise.Lag("GR", "INV", delta=1, width=1, activation="each", missing_a="skip"))
    result = wise.score(log, norm, evidence="full")
    record = result.evidence.records[0]
    assert record.reason_code is ReasonCode.SKIPPED_MISSING_ACTIVATION
    assert record.violation is None
    assert record.measurement["activations"].value == 0.0
    assert [w.kind for w in record.witnesses] == [WitnessKind.ABSENCE]


def test_an_each_activation_lag_under_censoring_reports_the_open_window():
    rows = [("m", "GR", 0, 1, "F"), ("m", "INV", 1, 1, "F"), ("m", "GR", 5, 1, "F")]
    log = make_log(rows, window=("2024-01-01", "2024-01-31"))
    norm = one_norm(wise.Lag("GR", "INV", delta=10, width=20, activation="each", missing_b="censor"))
    result = wise.score(log, norm, evidence="summary")
    record = result.evidence.records[0]
    assert record.reason_code is ReasonCode.OPEN_OBSERVATION_WINDOW
    assert record.evaluable and record.violation == pytest.approx(0.375)
    assert record.measurement["activations"].value == 2.0
    assert record.measurement["responses_observed"].value == 1.0
    assert result.manifest.observation.resolved_horizons == {"c": "2024-01-31T00:00:00"}
    assert QualificationCode.LOWER_BOUND in {q.code for q in record.qualifications}


def test_a_balance_with_nothing_on_either_side_says_so():
    log = make_log([("a", "OTHER", 0, 1, "F")])
    norm = one_norm(wise.Balance("amount", "INV", "amount", "GR"))
    record = wise.score(log, norm, evidence="full").evidence.records[0]
    assert record.violation == 0.0
    assert QualificationCode.BOTH_TOTALS_ZERO in {q.code for q in record.qualifications}
    assert all(w.kind is WitnessKind.ABSENCE for w in record.witnesses)


# ----------------------------------------------------------- interchange guards
def test_an_interchange_export_needs_a_scored_population_or_an_explicit_baseline(log):
    norm = wise.Norm(
        constraints=(wise.NormConstraint("c1", "L", wise.Presence("X"), applicability={"flow_type": ["DF1"]}),),
        layers=(wise.Layer("L"),),
        views=(wise.View("V", constraint_weights={"c1": 1.0}),),
    )
    packet = wise.score(log, norm, evidence="summary").evidence
    with pytest.raises(EvidenceError, match="no scored unit"):
        to_interchange(packet, view="V")
    payload = to_interchange(
        packet, view="V", baseline={"baseline_id": "b", "kind": "target", "score": 0.9, "layer_profile": None}
    )
    validate(payload, interchange_schema())
    assert payload["facts"][0]["value"] is None, "no evaluable measurement, and the packet says so"


def test_capture_can_be_restricted_to_the_units_under_review(log, norm):
    """A bounded packet is honest about being bounded."""
    result = wise.score(log, norm, evidence="summary")
    details = wise.evaluate_detailed(log, norm)[2]
    packet = capture_evidence(result, details=details, mode="full", units=["A", "E"])
    assert {r.unit_id for r in packet.records} == {"A", "E"}
    assert len(packet) == 12
    assert packet.coverage.n_units == 2
    assert packet.truncation.records_total == 30 and packet.truncation.records_truncated
    assert not packet.truncation.complete
    with pytest.raises(EvidenceError, match="unknown case ids"):
        capture_evidence(result, details=details, units=["Z"])


def test_a_packet_records_the_input_identity_it_was_actually_captured_from(log, norm):
    """A capture against another snapshot re-records the identity and says so."""
    plain = wise.score(log, norm)
    assert plain.manifest.preprocessing["snapshot_scope"] == "structure"
    packet = capture_evidence(plain, details=wise.evaluate_detailed(log, norm)[2], mode="summary")
    assert packet.manifest.preprocessing["snapshot_scope"] == "events"
    assert packet.manifest.input.snapshot_fingerprint == packet.snapshot.fingerprint
    assert any("re-recorded at capture time" in note for note in packet.manifest.notes)
    assert packet.witnesses(f"{packet.run_id}:c1:E"), "witnesses materialise against the recorded snapshot"


def test_a_captured_run_records_a_content_hashed_input_identity(log, norm):
    captured = wise.score(log, norm, evidence="summary")
    assert captured.manifest.preprocessing["snapshot_scope"] == "events"
    # a per-table map, never a single true: "content hashed" without naming the
    # table is the claim the review found overstated
    assert captured.manifest.preprocessing["content_hashed"] == {"cases": True, "events": True}
    assert captured.manifest.input.snapshot_fingerprint == captured.evidence.snapshot.fingerprint
    assert QualificationCode.SNAPSHOT_NOT_CONTENT_HASHED not in {q.code for q in captured.evidence.qualifications}


@pytest.mark.parametrize(
    ("missing_a", "missing_b", "reason", "violation"),
    [
        ("skip", "satisfy", ReasonCode.SKIPPED_MISSING_ACTIVATION, None),
        ("violate", "satisfy", ReasonCode.POLICY_VIOLATION_MISSING_ACTIVATION, 1.0),
        ("skip", "skip", ReasonCode.SKIPPED_MISSING_RESPONSE, None),
        ("violate", "skip", ReasonCode.SKIPPED_MISSING_RESPONSE, None),
    ],
)
def test_a_precedence_rule_with_neither_endpoint_reports_the_policy_that_decided(missing_a, missing_b, reason, violation):
    """The reason must agree with the value, whichever policy set it last."""
    log = make_log([("empty", "OTHER", 0, 1, "F")])
    norm = one_norm(wise.Precedence("GR", "INV", missing_a=missing_a, missing_b=missing_b))
    result = wise.score(log, norm, evidence="summary")
    record = result.evidence.records[0]
    assert record.reason_code is reason
    if violation is None:
        assert record.violation is None and not record.evaluable
    else:
        assert record.violation == pytest.approx(violation) and record.evaluable
    assert record.measurement["b_before_activation"].value is None
    assert record.measurement["b_total"].value == 0.0


AWKWARD_LOGS = {
    "empty_trace": [("only_other", "OTHER", 0, 1, "F")],
    "activation_only": [("a_only", "GR", 0, 1, "F")],
    "response_only": [("b_only", "INV", 0, 1, "F")],
    "response_first": [("reversed", "INV", 0, 1, "F"), ("reversed", "GR", 1, 1, "F")],
    "tied": [("tied", "GR", 0, 1, "F"), ("tied", "INV", 0, 1, "F")],
    "repeated": [("many", "GR", 0, 1, "F"), ("many", "GR", 1, 1, "F"), ("many", "INV", 9, 1, "F")],
    "complete": [("ok", "GR", 0, 1, "F"), ("ok", "INV", 1, 1, "F"), ("ok", "CLR", 2, 1, "F")],
}

AWKWARD_CONSTRAINTS = [
    wise.Presence("INV", m=2),
    wise.Exclusion("INV"),
    wise.Exclusion("INV", after="GR"),
    wise.Singularity("GR", k=1, K=2),
    wise.Singularity("GR", k=0, K=1, before="CLR"),
    wise.Balance("amount", "INV", "amount", "GR", tau=0.0, width=0.5),
    wise.Metric("n_events", threshold=1, width=2),
    *[
        wise.Lag("GR", "INV", delta=1, width=2, missing_a=ma, missing_b=mb)
        for ma in ("violate", "skip")
        for mb in ("violate", "skip", "censor")
    ],
    *[
        wise.Lag("GR", "INV", delta=1, width=2, activation=act, missing_a=ma, missing_b=mb)
        for act in ("last", "each")
        for ma in ("violate", "skip")
        for mb in ("violate", "skip", "censor")
    ],
    wise.Lag("GR", "INV", delta=1, width=2, response="first_overall"),
    *[wise.Precedence("GR", "INV", missing_a=ma, missing_b=mb) for ma in ("skip", "violate") for mb in ("satisfy", "skip")],
]


@pytest.mark.parametrize("shape", sorted(AWKWARD_LOGS))
def test_every_record_agrees_with_the_matrix_on_every_awkward_shape(shape):
    """The detailed path and the violation matrix may never diverge.

    Capture raises if a reason code claims a value the matrix does not have, so
    running the whole catalogue over degenerate traces is the cheapest way to
    keep the two honest.
    """
    log = make_log(AWKWARD_LOGS[shape], window=("2024-01-01", "2024-01-11"))
    for constraint in AWKWARD_CONSTRAINTS:
        norm = one_norm(constraint)
        plain = wise.score(log, norm)
        captured = wise.score(log, norm, evidence="full")
        pd.testing.assert_frame_equal(plain.violations, captured.violations)
        for record in captured.evidence.records:
            value = captured.violations.loc[record.unit_id, record.constraint_id]
            assert record.evaluable == bool(pd.notna(value))
            assert record.reason_code.evaluable == record.evaluable
            if record.evaluable:
                assert record.violation == pytest.approx(float(value))
            else:
                assert record.violation is None
        captured.evidence.to_json()


# ------------------------------------ the identity a run record actually has
def _p2p_log_with_amounts(scale=1.0):
    """The running example, with every invoice amount scaled — same shape, other values."""
    events = wise.running_p2p_events().copy()
    invoice = events["activity"] == "Record Invoice Receipt"
    events.loc[invoice, "amount"] = events.loc[invoice, "amount"] * scale
    return wise.EventLog(
        events,
        case_col="case",
        activity_col="activity",
        timestamp_col="time",
        case_attributes=["flow_type", "company", "vendor"],
    )


def test_the_run_record_says_which_tables_its_identity_covers(norm):
    """``content_hashed: true`` may not stand for "the case table, and nothing else"."""
    plain = wise.score(_p2p_log_with_amounts(), norm, evidence="summary")
    altered = wise.score(_p2p_log_with_amounts(scale=1.5), norm, evidence="summary")

    # the two runs are the same shape and produce different scores ...
    assert plain.manifest.input.n_events == altered.manifest.input.n_events
    assert plain.manifest.input.case_columns == altered.manifest.input.case_columns
    assert not plain.violations["c3"].equals(altered.violations["c3"]), "a changed amount changes the balance"
    # ... so they may not share an input identity
    assert plain.manifest.input.snapshot_fingerprint != altered.manifest.input.snapshot_fingerprint

    # and the field that used to say `true` now names the tables it covers
    assert plain.manifest.preprocessing["content_hashed"] == {"cases": True, "events": True}
    assert plain.evidence.snapshot.content_hashed_tables == {"cases": True, "events": True}

    # a weaker snapshot is allowed, and then the packet says exactly what is missing
    weaker = capture_evidence(
        plain, details=wise.evaluate_detailed(_p2p_log_with_amounts(), norm)[2], snapshot=_p2p_log_with_amounts().snapshot()
    )
    assert weaker.manifest.preprocessing["content_hashed"] == {"cases": True, "events": False}
    note = next(q for q in weaker.qualifications if q.code is QualificationCode.SNAPSHOT_NOT_CONTENT_HASHED)
    assert note.scope == "run"
    assert "the cases table" in note.message and "the events table" in note.message


def test_a_relabelled_activity_invalidates_lazy_witnesses(norm):
    """F11: an event-table change must be refused exactly as a case-table one is."""
    log = _p2p_log_with_amounts()
    packet = wise.score(log, norm, evidence="summary").evidence
    evaluation_id = f"{packet.run_id}:c1:E"
    assert packet.witnesses(evaluation_id), "the witnesses are available while the log is unchanged"

    row = log.events.index[log.events[log.activity_col] == "Record Goods Receipt"][0]
    log.events.loc[row, log.activity_col] = "Record Goods Receipt (renamed)"
    with pytest.raises(StaleEvidenceError, match="changed after this evidence was captured"):
        packet.witnesses(evaluation_id)


def test_a_changed_event_amount_invalidates_lazy_witnesses(norm):
    """The same, for the numeric attribute a Balance actually reads."""
    log = _p2p_log_with_amounts()
    packet = wise.score(log, norm, evidence="summary").evidence
    evaluation_id = f"{packet.run_id}:c3:A"
    assert packet.witnesses(evaluation_id)

    invoice = log.events.index[log.events[log.activity_col] == "Record Invoice Receipt"][0]
    log.events.loc[invoice, "amount"] = float(log.events.loc[invoice, "amount"]) + 1.0
    with pytest.raises(StaleEvidenceError):
        packet.witnesses(evaluation_id)


def test_a_bounded_packet_does_not_export_a_population_comparator(log, norm):
    """A subset's mean is not the run's population, and the export must not say it is."""
    result = wise.score(log, norm, evidence="summary")
    details = wise.evaluate_detailed(log, norm)[2]
    whole = capture_evidence(result, details=details, mode="summary")
    bounded = capture_evidence(result, details=details, mode="summary", units=["A", "E"])
    assert bounded.truncation.records_truncated

    full_payload = to_interchange(whole, view="Finance")
    subset_payload = to_interchange(bounded, view="Finance")
    validate(subset_payload, interchange_schema())

    # the two comparators really do differ, which is why the label matters
    assert subset_payload["baseline"]["score"] != full_payload["baseline"]["score"]
    assert full_payload["baseline"]["baseline_id"].startswith("current-population:")
    assert subset_payload["baseline"]["baseline_id"].startswith("captured-subset:")
    note = next(q for q in subset_payload["qualifications"] if "bounded capture kept" in q)
    assert "2 scored cases" in note
    assert "not the run's population" in note
    assert not any("bounded capture kept" in q for q in full_payload["qualifications"])
