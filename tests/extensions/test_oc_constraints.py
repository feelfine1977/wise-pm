"""The three native relational check families (O03).

What is under test is not "does it compute a number" but the four places where
an object-centric checker is normally wrong:

1. **Absence.** A role that binds nothing is not an obligation unmet; it is a
   question the extract did not answer. A shortfall becomes a violation only
   under a declared completeness assumption, and never inside a truncated
   context.
2. **Ambiguity.** Two response events sharing a timestamp are two events. The
   default is to refuse the match and say so, not to take the smaller id and
   hope.
3. **Money.** One amount is consumed once. Currencies are not converted, and
   credit notes are not netted, without a declared rule.
4. **Time.** Attributes are read at the declared instant, and every temporal
   policy — activation, matching, clock, equal-time, missingness — is a field
   somebody had to fill in.
"""

from __future__ import annotations

import dataclasses

import pytest
from _oc_fixtures import (
    GR,
    INV,
    PO,
    changing_amount,
    day,
    equal_timestamps,
    invoice_spec,
    many_items,
    partial_deliveries,
    shared_payment,
    unknown_matching,
    wide_vendor,
)

import wise
from wise import oc
from wise.errors import OCConstraintError, OCUnitError
from wise.evidence.models import Completeness, MeasurementKind, QualificationCode, ReasonCode, WitnessKind
from wise.norm import Layer, Norm, NormConstraint, View
from wise.oc.constraints import object_check_from_dict


def units_of(log, spec=None, **kwargs):
    """Build units, without the spec's strict name check.

    These fixtures are deliberately partial — a log about payments has no
    goods receipts — and refusing a spec whose names do not all occur is O02's
    subject, tested there. Every assertion below pins a concrete count, so a
    role that silently bound nothing would fail rather than pass.
    """
    kwargs.setdefault("strict", False)
    return oc.build_units(log, spec or invoice_spec(), **kwargs)


def only(log, spec, check, **kwargs):
    """Evaluate one check on one unit and hand back the outcome."""
    units = units_of(log, spec, **kwargs)
    return check.evaluate(log, units[0]), units[0]


# ============================================================ related-object cardinality
def test_a_role_is_counted_by_distinct_identity_not_by_relation_row():
    log = many_items(100)
    spec = invoice_spec()
    outcome, unit = only(log, spec, oc.RelatedObjectCardinality(role="items", maximum=1000))
    assert outcome.measurements[0].name == "related_objects"
    assert outcome.measurements[0].value == 100.0
    assert outcome.violation == 0.0
    assert len(unit.role("items")) == 100


def test_one_invoice_with_a_hundred_items_keeps_one_invoice_exposure():
    """O03: distinct obligations retained; the additive amount is not × 100."""
    log = many_items(100)
    spec = invoice_spec()
    units = units_of(log, spec, at="2024-06-01T00:00:00Z")
    balance = oc.RelationalBalance(
        left=oc.AmountSelector(attribute="amount", unit="EUR"),
        right=oc.AmountSelector(role="items", attribute="quantity", unit="EUR"),
        tolerance=0.0,
    )
    outcome = balance.evaluate(log, units[0])
    totals = {m.name: m.value for m in outcome.measurements}
    assert totals["total_left"] == 1000.0, "the invoice's amount appears once, not once per item"
    assert totals["right_records"] == 100.0
    assert totals["total_right"] == 1000.0, "100 items of 10 sum to 1000 — added, not multiplied"
    assert outcome.violation == 0.0


def test_a_shortfall_without_a_completeness_assumption_is_not_a_violation():
    """O05: unknown relationship information is not automatically absence."""
    log = unknown_matching()
    spec = invoice_spec()
    units = units_of(log, spec)
    unmatched = next(u for u in units if u.anchor_id == "inv2")
    check = oc.RelatedObjectCardinality(role="order", minimum=1)
    outcome = check.evaluate(log, unmatched)
    assert outcome.violation is None
    assert outcome.reason is ReasonCode.UNVERIFIED_ABSENCE
    assert not outcome.evaluable
    codes = {q.code for q in outcome.qualifications}
    assert QualificationCode.ABSENCE_COMPLETENESS_UNKNOWN in codes


def test_the_same_shortfall_is_a_violation_once_completeness_is_declared():
    log = unknown_matching()
    units = units_of(log, invoice_spec())
    unmatched = next(u for u in units if u.anchor_id == "inv2")
    check = oc.RelatedObjectCardinality(role="order", minimum=1, completeness=Completeness.ASSUMED_COMPLETE)
    outcome = check.evaluate(log, unmatched)
    assert outcome.violation == 1.0
    assert outcome.reason is ReasonCode.OBSERVED
    message = " ".join(q.message for q in outcome.qualifications)
    assert "the configuration declares" in message, "the assumption travels with the number"
    assert outcome.policies["completeness"] == "assumed_complete"


def test_a_matched_invoice_satisfies_the_same_check_either_way():
    log = unknown_matching()
    units = units_of(log, invoice_spec())
    matched = next(u for u in units if u.anchor_id == "inv1")
    for completeness in (Completeness.UNKNOWN, Completeness.ASSUMED_COMPLETE):
        outcome = oc.RelatedObjectCardinality(role="order", minimum=1, completeness=completeness).evaluate(log, matched)
        assert outcome.violation == 0.0, "what is present is present regardless of what the scope assumes"


def test_an_excess_needs_no_assumption_because_the_objects_are_right_there():
    log = partial_deliveries(3)
    units = units_of(log, invoice_spec())
    outcome = oc.RelatedObjectCardinality(role="receipts", maximum=1).evaluate(log, units[0])
    assert outcome.violation == 1.0
    assert outcome.reason is ReasonCode.OBSERVED
    assert QualificationCode.ABSENCE_COMPLETENESS_UNKNOWN not in {q.code for q in outcome.qualifications}


def test_width_grades_the_violation_over_missing_or_extra_objects():
    log = partial_deliveries(3)
    units = units_of(log, invoice_spec())
    graded = oc.RelatedObjectCardinality(role="receipts", maximum=1, width=4).evaluate(log, units[0])
    assert graded.violation == pytest.approx(0.5), "two extra receipts over a width of four"
    step = oc.RelatedObjectCardinality(role="receipts", maximum=1, width=0).evaluate(log, units[0])
    assert step.violation == 1.0


def test_a_truncated_role_cannot_prove_an_absence_and_says_so():
    """O07 meets O05: a bounded partial context is not evidence of nothing."""
    log = wide_vendor(40)
    spec = oc.UnitSpec(
        unit_type="vendor_review",
        anchor_type="vendor",
        roles=(oc.RolePath("invoices", (oc.PathStep("billed by", target_type="invoice", direction="reverse"),)),),
        limits=oc.TraversalLimits(max_fan_out=5),
    )
    unit = oc.build_units(log, spec)[0]
    assert not unit.complete
    outcome = oc.RelatedObjectCardinality(role="invoices", minimum=100, completeness=Completeness.ASSUMED_COMPLETE).evaluate(
        log, unit
    )
    assert outcome.violation is None
    assert outcome.reason is ReasonCode.BUDGET_TRUNCATED
    codes = {q.code for q in outcome.qualifications}
    assert QualificationCode.COUNT_IS_A_LOWER_BOUND in codes
    assert QualificationCode.CONTEXT_TRUNCATED in codes
    assert outcome.measurements[0].lower_bound is True


def test_a_truncated_role_can_still_prove_an_excess():
    log = wide_vendor(40)
    spec = oc.UnitSpec(
        unit_type="vendor_review",
        anchor_type="vendor",
        roles=(oc.RolePath("invoices", (oc.PathStep("billed by", target_type="invoice", direction="reverse"),)),),
        limits=oc.TraversalLimits(max_fan_out=5),
    )
    unit = oc.build_units(log, spec)[0]
    outcome = oc.RelatedObjectCardinality(role="invoices", maximum=2).evaluate(log, unit)
    assert outcome.violation == 1.0, "five is already more than two; the ones that were cut cannot undo that"
    assert QualificationCode.COUNT_IS_A_LOWER_BOUND in {q.code for q in outcome.qualifications}


def test_a_maximum_satisfied_only_because_the_context_was_cut_is_not_reported_as_satisfied():
    """The cut is what produced the pass, so the pass is not a finding.

    Five of forty invoices against ``maximum=10``: the count is a lower bound,
    and a lower bound below a limit establishes nothing about the limit. The
    shortfall case is already ``BUDGET_TRUNCATED``; this is its mirror.
    """
    log = wide_vendor(40)
    spec = oc.UnitSpec(
        unit_type="vendor_review",
        anchor_type="vendor",
        roles=(oc.RolePath("invoices", (oc.PathStep("billed by", target_type="invoice", direction="reverse"),)),),
        limits=oc.TraversalLimits(max_fan_out=5),
    )
    unit = oc.build_units(log, spec)[0]
    assert not unit.complete

    outcome = oc.RelatedObjectCardinality(role="invoices", maximum=10).evaluate(log, unit)
    assert outcome.violation is None, "a satisfied maximum over a cut context is not a clean pass"
    assert outcome.reason is ReasonCode.BUDGET_TRUNCATED
    assert not outcome.evaluable
    codes = {q.code for q in outcome.qualifications}
    assert QualificationCode.COUNT_IS_A_LOWER_BOUND in codes
    assert QualificationCode.CONTEXT_TRUNCATED in codes
    assert outcome.measurements[0].lower_bound is True

    whole = oc.build_units(log, dataclasses.replace(spec, limits=oc.TraversalLimits()))[0]
    complete = oc.RelatedObjectCardinality(role="invoices", maximum=10).evaluate(log, whole)
    assert complete.violation == 1.0, "uncut, the same vendor exceeds the same maximum"
    assert complete.reason is ReasonCode.OBSERVED


def test_a_minimum_a_cut_context_already_meets_is_still_observed():
    """Truncation only ever hides *more* objects, so a met minimum stays met."""
    log = wide_vendor(40)
    spec = oc.UnitSpec(
        unit_type="vendor_review",
        anchor_type="vendor",
        roles=(oc.RolePath("invoices", (oc.PathStep("billed by", target_type="invoice", direction="reverse"),)),),
        limits=oc.TraversalLimits(max_fan_out=5),
    )
    unit = oc.build_units(log, spec)[0]
    outcome = oc.RelatedObjectCardinality(role="invoices", minimum=3).evaluate(log, unit)
    assert outcome.violation == 0.0 and outcome.reason is ReasonCode.OBSERVED


def test_cardinality_records_what_it_bound_and_why():
    log = partial_deliveries(3)
    units = units_of(log, invoice_spec())
    outcome = oc.RelatedObjectCardinality(role="receipts", maximum=1).evaluate(log, units[0])
    assert outcome.policies["bound_objects"] == ["gr1", "gr2", "gr3"]
    assert outcome.policies["bound_types"] == ["goods_receipt"]
    assert [w.kind for w in outcome.witnesses] == [WitnessKind.AGGREGATE]
    kinds = {m.name: m.kind for m in outcome.measurements}
    assert kinds["related_objects"] is MeasurementKind.COUNT
    assert kinds["maximum"] is MeasurementKind.THRESHOLD


def test_a_cardinality_check_is_refused_before_it_can_be_wrong():
    with pytest.raises(OCConstraintError, match="declare a minimum, a maximum or both"):
        oc.RelatedObjectCardinality(role="order")
    with pytest.raises(OCConstraintError, match="minimum 3 exceeds maximum 1"):
        oc.RelatedObjectCardinality(role="order", minimum=3, maximum=1)
    with pytest.raises(OCConstraintError, match="width must be at least 0"):
        oc.RelatedObjectCardinality(role="order", minimum=1, width=-1)
    check = oc.RelatedObjectCardinality(role="nonexistent", minimum=1)
    assert check.check_spec(invoice_spec()) == ("role 'nonexistent' is not declared by unit type 'invoice_review'",)
    with pytest.raises(OCUnitError, match="undeclared role"):
        check.evaluate(many_items(2), units_of(many_items(2), invoice_spec())[0])


def test_target_type_narrows_the_count_without_changing_the_role():
    log = partial_deliveries(2)
    units = units_of(log, invoice_spec())
    same = oc.RelatedObjectCardinality(role="receipts", maximum=99, target_type="goods_receipt").evaluate(log, units[0])
    none = oc.RelatedObjectCardinality(role="receipts", maximum=99, target_type="payment").evaluate(log, units[0])
    assert same.measurements[0].value == 2.0
    assert none.measurements[0].value == 0.0


# ==================================================================== cross-object lag
def lag_spec():
    return invoice_spec()


def po_to_invoice(**changes):
    payload = {
        "activation": oc.EventSelector(role="order", activities=(PO,)),
        "response": oc.EventSelector(activities=(INV,)),
        "delta": 5.0,
        "width": 10.0,
        "unit": "D",
    }
    payload.update(changes)
    return oc.CrossObjectLag(**payload)


def test_the_lag_is_measured_across_two_objects_through_their_relation():
    """The activation sits on the order, the response on the invoice."""
    log = many_items(3)
    units = units_of(log, lag_spec())
    outcome = po_to_invoice().evaluate(log, units[0])
    values = {m.name: m.value for m in outcome.measurements}
    assert values["lag"] == 9.0
    assert values["activation_timestamp"] == day(0).isoformat()
    assert values["response_timestamp"] == day(9).isoformat()
    assert outcome.violation == pytest.approx(0.4)
    events = {w.event_id for w in outcome.witnesses if w.kind is WitnessKind.EVENT}
    assert events == {"e_po", "e_inv"}, "the witnesses are source event ids, not row positions"


def test_every_temporal_policy_is_declared_and_recorded():
    log = many_items(2)
    units = units_of(log, lag_spec())
    outcome = po_to_invoice().evaluate(log, units[0])
    assert outcome.policies["clock"] == "event_timestamp"
    assert outcome.policies["match"] == "first_after"
    assert outcome.policies["equal_time"] == "excluded"
    assert outcome.policies["missing_activation"] == "violate"
    assert outcome.policies["missing_response"] == "violate"
    assert outcome.policies["on_ambiguous"] == "qualify"


def test_two_responses_sharing_an_instant_are_an_ambiguous_match_by_default():
    """O08: no hidden nearest-event rule."""
    log = equal_timestamps()
    spec = invoice_spec()
    units = units_of(log, spec)
    check = oc.CrossObjectLag(
        activation=oc.EventSelector(role="order", activities=(PO,)),
        response=oc.EventSelector(activities=(INV,)),
        delta=1.0,
        width=10.0,
    )
    outcome = check.evaluate(log, units[0])
    assert outcome.violation is None
    assert outcome.reason is ReasonCode.AMBIGUOUS_MATCH
    assert {w.event_id for w in outcome.witnesses} == {"e_inv1", "e_inv2"}
    message = outcome.qualifications[0].message
    assert "distinct events" in message and "guess" in message


def test_a_declared_tie_break_resolves_it_and_admits_that_it_did():
    log = equal_timestamps()
    units = units_of(log, invoice_spec())
    check = oc.CrossObjectLag(
        activation=oc.EventSelector(role="order", activities=(PO,)),
        response=oc.EventSelector(activities=(INV,)),
        delta=1.0,
        width=10.0,
        on_ambiguous="tie_break",
    )
    outcome = check.evaluate(log, units[0])
    assert outcome.violation == pytest.approx(0.5)
    quals = {q.code: q.message for q in outcome.qualifications}
    assert QualificationCode.AMBIGUOUS_MATCH_RESOLVED in quals
    assert "'e_inv1'" in quals[QualificationCode.AMBIGUOUS_MATCH_RESOLVED]
    assert "the configuration's, not the data's" in quals[QualificationCode.AMBIGUOUS_MATCH_RESOLVED]


def test_an_unresolved_tie_and_a_tie_break_carry_different_codes():
    """A consumer filtering on codes must be able to tell the two apart.

    "The configuration chose for you" and "nothing was chosen" are different
    findings. The prose already says which is which; the code did not, and a
    reader who groups by code is the reader this library is written for.
    """
    log = equal_timestamps()
    units = units_of(log, invoice_spec())
    common = {
        "activation": oc.EventSelector(role="order", activities=(PO,)),
        "response": oc.EventSelector(activities=(INV,)),
        "delta": 1.0,
        "width": 10.0,
    }
    unresolved = oc.CrossObjectLag(**common, on_ambiguous="qualify").evaluate(log, units[0])
    broken = oc.CrossObjectLag(**common, on_ambiguous="tie_break").evaluate(log, units[0])

    unresolved_codes = {q.code for q in unresolved.qualifications}
    broken_codes = {q.code for q in broken.qualifications}
    assert QualificationCode.AMBIGUOUS_MATCH_UNRESOLVED in unresolved_codes
    assert QualificationCode.AMBIGUOUS_MATCH_RESOLVED not in unresolved_codes
    assert QualificationCode.AMBIGUOUS_MATCH_RESOLVED in broken_codes
    assert QualificationCode.AMBIGUOUS_MATCH_UNRESOLVED not in broken_codes
    assert unresolved_codes.isdisjoint(broken_codes)
    assert unresolved.violation is None and broken.violation == pytest.approx(0.5)


def test_a_caller_can_insist_that_an_ambiguous_match_is_an_error():
    log = equal_timestamps()
    units = units_of(log, invoice_spec())
    check = oc.CrossObjectLag(
        activation=oc.EventSelector(role="order", activities=(PO,)),
        response=oc.EventSelector(activities=(INV,)),
        delta=1.0,
        on_ambiguous="error",
    )
    with pytest.raises(OCConstraintError, match="share the timestamp"):
        check.evaluate(log, units[0])


def test_both_events_of_an_equal_timestamp_pair_survive_into_the_candidates():
    """O02 again, from the check's side: the tie exists because both are there."""
    log = equal_timestamps()
    units = units_of(log, invoice_spec())
    check = oc.CrossObjectLag(
        activation=oc.EventSelector(role="order", activities=(PO,)),
        response=oc.EventSelector(activities=(INV,)),
        delta=1.0,
        on_ambiguous="tie_break",
    )
    outcome = check.evaluate(log, units[0])
    assert {m.name: m.value for m in outcome.measurements}["candidate_responses"] == 2.0


def test_the_equal_time_interpretation_changes_the_answer_and_is_a_field():
    log = oc.OCEventLog.build(
        events=[oc.OCEvent("a1", PO, day(3)), oc.OCEvent("b1", INV, day(3))],
        objects=[oc.OCObject("po1", "purchase_order"), oc.OCObject("inv1", "invoice")],
        e2o=[oc.E2O("a1", "po1", "purchase_order"), oc.E2O("b1", "inv1", "invoice")],
        o2o=[oc.O2O("inv1", "po1", "belongs to")],
    )
    units = oc.build_units(log, invoice_spec(), strict=False)
    excluded = po_to_invoice(equal_time="excluded", missing_response="skip").evaluate(log, units[0])
    counts = po_to_invoice(equal_time="counts").evaluate(log, units[0])
    assert excluded.violation is None and excluded.reason is ReasonCode.SKIPPED_MISSING_RESPONSE
    assert counts.violation == 0.0
    assert {m.name: m.value for m in counts.measurements}["lag"] == 0.0


def test_a_missing_activation_follows_its_declared_policy():
    log = unknown_matching()
    units = units_of(log, invoice_spec())
    unmatched = next(u for u in units if u.anchor_id == "inv2")  # no order, so no PO event in scope
    violating = po_to_invoice().evaluate(log, unmatched)
    skipping = po_to_invoice(missing_activation="skip").evaluate(log, unmatched)
    assert violating.violation == 1.0 and violating.reason is ReasonCode.POLICY_VIOLATION_MISSING_ACTIVATION
    assert skipping.violation is None and skipping.reason is ReasonCode.SKIPPED_MISSING_ACTIVATION
    absence = next(w for w in violating.witnesses if w.kind is WitnessKind.ABSENCE)
    assert absence.search is not None
    assert absence.search.activities == (PO,)
    assert absence.search.completeness is Completeness.UNKNOWN, "an untruncated traversal is not a complete observation"


def test_a_missing_response_can_violate_skip_or_be_censored():
    log = oc.OCEventLog.build(
        events=[oc.OCEvent("a1", PO, day(0)), oc.OCEvent("x1", "Other", day(1))],
        objects=[oc.OCObject("po1", "purchase_order"), oc.OCObject("inv1", "invoice")],
        e2o=[oc.E2O("a1", "po1", "purchase_order"), oc.E2O("x1", "inv1", "invoice")],
        o2o=[oc.O2O("inv1", "po1", "belongs to")],
    )
    unit = oc.build_units(log, invoice_spec(), strict=False)[0]
    assert po_to_invoice().evaluate(log, unit).violation == 1.0
    assert po_to_invoice(missing_response="skip").evaluate(log, unit).violation is None
    censored = po_to_invoice(missing_response="censor", horizon=day(8)).evaluate(log, unit)
    assert censored.reason is ReasonCode.OPEN_OBSERVATION_WINDOW
    assert censored.violation == pytest.approx(0.3), "eight days elapsed against delta 5 over a width of 10"
    codes = {q.code for q in censored.qualifications}
    assert {QualificationCode.CENSORED_HORIZON, QualificationCode.LOWER_BOUND} <= codes
    assert next(m for m in censored.measurements if m.name == "elapsed_at_horizon").lower_bound is True


def test_censoring_without_a_horizon_takes_the_last_moment_the_log_observed():
    log = many_items(2)
    unit = units_of(log, invoice_spec())[0]
    check = po_to_invoice(
        response=oc.EventSelector(activities=("Never Happens",)), missing_response="censor", delta=1.0, width=100.0
    )
    outcome = check.evaluate(log, unit)
    horizon = next(m for m in outcome.measurements if m.name == "horizon").value
    assert horizon == day(9).isoformat(), "the horizon is the last observed event, and it is reported"


def test_first_overall_counts_a_response_that_precedes_the_activation_as_missing():
    log = oc.OCEventLog.build(
        events=[oc.OCEvent("b0", INV, day(1)), oc.OCEvent("a1", PO, day(4)), oc.OCEvent("b1", INV, day(6))],
        objects=[oc.OCObject("po1", "purchase_order"), oc.OCObject("inv1", "invoice")],
        e2o=[
            oc.E2O("b0", "inv1", "invoice"),
            oc.E2O("a1", "po1", "purchase_order"),
            oc.E2O("b1", "inv1", "invoice"),
        ],
        o2o=[oc.O2O("inv1", "po1", "belongs to")],
    )
    unit = oc.build_units(log, invoice_spec(), strict=False)[0]
    after = po_to_invoice(match="first_after").evaluate(log, unit)
    overall = po_to_invoice(match="first_overall", missing_response="skip").evaluate(log, unit)
    assert {m.name: m.value for m in after.measurements}["lag"] == 2.0
    assert overall.violation is None, "the first invoice in the unit is not a response to a later order"


def test_the_activation_occurrence_is_a_choice_and_both_are_available():
    log = partial_deliveries(3)
    unit = units_of(log, invoice_spec())[0]
    first = oc.CrossObjectLag(
        activation=oc.EventSelector(role="receipts", activities=(GR,), occurrence="first"),
        response=oc.EventSelector(activities=(INV,)),
        delta=0.0,
        width=100.0,
    ).evaluate(log, unit)
    last = oc.CrossObjectLag(
        activation=oc.EventSelector(role="receipts", activities=(GR,), occurrence="last"),
        response=oc.EventSelector(activities=(INV,)),
        delta=0.0,
        width=100.0,
    ).evaluate(log, unit)
    assert {m.name: m.value for m in first.measurements}["lag"] == 11.0
    assert {m.name: m.value for m in last.measurements}["lag"] == 9.0


def test_a_lag_check_is_refused_before_it_can_be_wrong():
    good = oc.EventSelector(activities=(INV,))
    with pytest.raises(OCConstraintError, match="at least one activity"):
        oc.EventSelector(activities=())
    with pytest.raises(OCConstraintError, match="occurrence must be"):
        oc.EventSelector(activities=(INV,), occurrence="middle")
    with pytest.raises(OCConstraintError, match="must be an EventSelector"):
        oc.CrossObjectLag(activation="PO", response=good)  # type: ignore[arg-type]
    with pytest.raises(OCConstraintError, match="unit must be one of"):
        oc.CrossObjectLag(activation=good, response=good, unit="fortnight")
    with pytest.raises(OCConstraintError, match="equal_time must be one of"):
        oc.CrossObjectLag(activation=good, response=good, equal_time="maybe")
    with pytest.raises(OCConstraintError, match="needs a delta to censor against"):
        oc.CrossObjectLag(activation=good, response=good, delta=None, missing_response="censor")


# =================================================================== relational balance
def test_a_balance_adds_a_role_up_through_the_relation_that_defines_it():
    """Ordered 90 against invoiced 100 — a mismatch only the relations reveal."""
    log = partial_deliveries(3)
    unit = units_of(log, invoice_spec(), at="2024-06-01T00:00:00Z")[0]
    check = oc.RelationalBalance(
        left=oc.AmountSelector(attribute="amount", unit="EUR"),
        right=oc.AmountSelector(role="receipts", attribute="amount", unit="EUR"),
        tolerance=0.05,
        width=0.20,
    )
    outcome = check.evaluate(log, unit)
    values = {m.name: m.value for m in outcome.measurements}
    assert values["total_left"] == 100.0
    assert values["total_right"] == 90.0
    assert values["relative_mismatch"] == pytest.approx(0.10)
    assert outcome.violation == pytest.approx(0.25)
    assert outcome.policies["allocation"] == {"complete": True, "residual_total": 0.0}


def test_the_same_object_on_both_sides_is_refused_not_counted_twice():
    log = partial_deliveries(2)
    unit = units_of(log, invoice_spec(), at="2024-06-01T00:00:00Z")[0]
    check = oc.RelationalBalance(
        left=oc.AmountSelector(attribute="amount", unit="EUR"),
        right=oc.AmountSelector(attribute="amount", unit="EUR"),
    )
    with pytest.raises(OCConstraintError, match="both sides of the balance"):
        check.evaluate(log, unit)


def test_the_amount_read_is_the_one_in_force_at_the_evaluation_instant():
    """O06: a changed amount is a different record, not a correction of the old."""
    log = changing_amount()
    check = oc.RelationalBalance(
        left=oc.AmountSelector(attribute="amount", unit="EUR"),
        right=oc.AmountSelector(role="items", attribute="amount", unit="EUR"),
        tolerance=0.0,
    )
    early = check.evaluate(log, units_of(log, invoice_spec(), at="2024-01-10T00:00:00Z")[0])
    late = check.evaluate(log, units_of(log, invoice_spec(), at="2024-03-01T00:00:00Z")[0])
    assert {m.name: m.value for m in early.measurements}["total_left"] == 100.0
    assert {m.name: m.value for m in late.measurements}["total_left"] == 120.0
    assert early.violation == 0.0
    assert late.violation == 1.0


def test_a_balance_needs_the_instant_it_reads_at():
    log = unknown_matching()
    unit = units_of(log, invoice_spec())[0]
    check = oc.RelationalBalance(
        left=oc.AmountSelector(attribute="amount", unit="EUR"),
        right=oc.AmountSelector(role="order", attribute="amount", unit="EUR"),
    )
    with pytest.raises(OCConstraintError, match="built without one"):
        check.evaluate(log, unit)


def test_an_unread_amount_is_missing_and_not_zero():
    log = unknown_matching()
    unit = next(u for u in units_of(log, invoice_spec(), at="2024-06-01T00:00:00Z") if u.anchor_id == "inv2")
    check = oc.RelationalBalance(
        left=oc.AmountSelector(attribute="amount", unit="EUR"),
        right=oc.AmountSelector(role="order", attribute="amount", unit="EUR"),
    )
    outcome = check.evaluate(log, unit)
    assert outcome.violation is None
    assert outcome.reason is ReasonCode.MISSING_ATTRIBUTE
    assert "not a zero amount" in " ".join(q.message for q in outcome.qualifications)


def _receipts_missing_one_amount() -> oc.OCEventLog:
    """``partial_deliveries(3)`` with ``gr3``'s amount never recorded.

    The role still binds three receipts. Two of them carry 30; the third
    carries nothing at all, which is not the same as carrying zero.
    """
    receipts = [oc.OCObject(f"gr{i}", "goods_receipt") for i in (1, 2, 3)]
    events = [oc.OCEvent("e_po", PO, day(0)), oc.OCEvent("e_inv", INV, day(12))]
    e2o = [
        oc.E2O("e_po", "po1", "purchase_order"),
        oc.E2O("e_inv", "inv1", "invoice"),
        oc.E2O("e_inv", "po1", "purchase_order"),
    ]
    o2o = [oc.O2O("inv1", "po1", "belongs to")]
    history = [oc.AttributeChange("inv1", "amount", day(12), 100.0)]
    for i, receipt in enumerate(receipts, start=1):
        events.append(oc.OCEvent(f"e_gr{i}", GR, day(i)))
        e2o += [oc.E2O(f"e_gr{i}", receipt.object_id, "goods_receipt"), oc.E2O(f"e_gr{i}", "po1", "purchase_order")]
        o2o.append(oc.O2O(receipt.object_id, "po1", "receipt for"))
        if i < 3:
            history.append(oc.AttributeChange(receipt.object_id, "amount", day(i), 30.0))
    return oc.OCEventLog.build(
        events=events,
        objects=[oc.OCObject("po1", "purchase_order"), oc.OCObject("inv1", "invoice"), *receipts],
        e2o=e2o,
        o2o=o2o,
        attribute_history=history,
    )


def test_a_side_with_an_unread_amount_is_not_a_complete_total():
    """O03/O05: dropping an amount from a sum is arithmetically a zero.

    Three receipts bind; two carry 30. Summing 60 against an invoiced 100
    quadruples the mismatch, and until the count of unread amounts sits beside
    the count of records the number says nothing about how it was reached.
    """
    balance = oc.RelationalBalance(
        left=oc.AmountSelector(attribute="amount", unit="EUR"),
        right=oc.AmountSelector(role="receipts", attribute="amount", unit="EUR"),
        tolerance=0.05,
        width=0.20,
    )
    at = "2024-06-01T00:00:00Z"
    whole = balance.evaluate(partial_deliveries(3), units_of(partial_deliveries(3), invoice_spec(), at=at)[0])
    assert whole.violation == pytest.approx(0.25)
    assert {m.name: m.value for m in whole.measurements}["missing_right"] == 0.0
    assert QualificationCode.ABSENCE_COMPLETENESS_UNKNOWN not in {q.code for q in whole.qualifications}

    log = _receipts_missing_one_amount()
    unit = units_of(log, invoice_spec(), at=at)[0]
    assert len(unit.role("receipts")) == 3, "the relation still binds three receipts"
    outcome = balance.evaluate(log, unit)

    values = {m.name: m.value for m in outcome.measurements}
    assert values["total_right"] == 60.0
    assert values["right_records"] == 2.0
    assert values["missing_right"] == 1.0, "the unread amount is counted beside the records that were read"
    assert values["missing_left"] == 0.0
    assert QualificationCode.ABSENCE_COMPLETENESS_UNKNOWN in {q.code for q in outcome.qualifications}
    message = " ".join(q.message for q in outcome.qualifications)
    assert "1 of the 3" in message, "the qualification compares the unread count with the bound objects"
    assert outcome.policies["bound_objects"] == {"left": 1, "right": 3}


def test_a_balance_read_outside_the_evaluation_time_says_so():
    """O06: the reading knows it looked at the wrong moment; the outcome must too."""
    log = changing_amount()
    balance = oc.RelationalBalance(
        left=oc.AmountSelector(attribute="amount", unit="EUR", policy="latest_known"),
        right=oc.AmountSelector(role="items", attribute="amount", unit="EUR", policy="latest_known"),
        tolerance=0.0,
        width=0.20,
    )
    at_the_time = balance.evaluate(log, units_of(log, invoice_spec(), at="2024-01-10T00:00:00Z")[0])
    values = {m.name: m.value for m in at_the_time.measurements}
    assert values["total_left"] == 120.0, "latest_known read the March value for a January unit"
    assert QualificationCode.ATTRIBUTE_READ_OUTSIDE_EVALUATION_TIME in {q.code for q in at_the_time.qualifications}
    assert "after the evaluation time" in " ".join(q.message for q in at_the_time.qualifications)

    no_instant = balance.evaluate(log, units_of(log, invoice_spec())[0])
    assert {m.name: m.value for m in no_instant.measurements}["total_left"] == 120.0
    assert no_instant.qualifications, "a balance read at no particular instant is not an unqualified balance"
    assert QualificationCode.ATTRIBUTE_READ_OUTSIDE_EVALUATION_TIME in {q.code for q in no_instant.qualifications}
    assert "no evaluation instant" in " ".join(q.message for q in no_instant.qualifications)


def test_an_attribute_not_set_at_the_evaluation_time_reaches_the_outcome():
    """The other half of the same line: ``attribute_not_set_at_time`` travels too."""
    log = changing_amount()
    balance = oc.RelationalBalance(
        left=oc.AmountSelector(attribute="amount", unit="EUR"),
        right=oc.AmountSelector(role="items", attribute="amount", unit="EUR"),
    )
    # day 2: the order's item already carries 100, the invoice's amount does not exist yet
    outcome = balance.evaluate(log, units_of(log, invoice_spec(), at="2024-01-03T00:00:00Z")[0])
    assert outcome.reason is ReasonCode.MISSING_ATTRIBUTE
    codes = {q.code for q in outcome.qualifications}
    assert QualificationCode.ATTRIBUTE_NOT_SET_AT_TIME in codes


def test_two_currencies_are_not_added_without_a_declared_rate():
    log = oc.OCEventLog.build(
        events=[oc.OCEvent("e1", INV, day(1))],
        objects=[oc.OCObject("inv1", "invoice"), oc.OCObject("po1", "purchase_order")],
        e2o=[oc.E2O("e1", "inv1", "invoice")],
        o2o=[oc.O2O("inv1", "po1", "belongs to")],
        attribute_history=[
            oc.AttributeChange("inv1", "amount", day(1), 100.0),
            oc.AttributeChange("inv1", "currency", day(1), "EUR"),
            oc.AttributeChange("po1", "amount", day(0), 110.0),
            oc.AttributeChange("po1", "currency", day(0), "USD"),
        ],
    )
    unit = oc.build_units(log, invoice_spec(), at="2024-06-01T00:00:00Z", strict=False)[0]
    strict = oc.RelationalBalance(
        left=oc.AmountSelector(attribute="amount", unit_attribute="currency"),
        right=oc.AmountSelector(role="order", attribute="amount", unit_attribute="currency"),
    )
    outcome = strict.evaluate(log, unit)
    assert outcome.violation is None
    assert outcome.reason is ReasonCode.INCOMPATIBLE_UNITS
    assert "No rate was declared" in " ".join(q.message for q in outcome.qualifications)

    converting = oc.RelationalBalance(
        left=oc.AmountSelector(attribute="amount", unit_attribute="currency"),
        right=oc.AmountSelector(role="order", attribute="amount", unit_attribute="currency"),
        unit_policy="declared_rates",
        rates={"USD": 0.9},
        target_unit="EUR",
        tolerance=0.05,
    )
    converted = converting.evaluate(log, unit)
    values = {m.name: m.value for m in converted.measurements}
    assert values["total_right"] == pytest.approx(99.0)
    assert QualificationCode.UNIT_CONVERSION_APPLIED in {q.code for q in converted.qualifications}
    assert "dates from no particular day" in " ".join(q.message for q in converted.qualifications)


def test_two_zero_totals_say_what_the_ratio_rests_on():
    log = oc.OCEventLog.build(
        events=[oc.OCEvent("e1", INV, day(1))],
        objects=[oc.OCObject("inv1", "invoice"), oc.OCObject("po1", "purchase_order")],
        e2o=[oc.E2O("e1", "inv1", "invoice")],
        o2o=[oc.O2O("inv1", "po1", "belongs to")],
        attribute_history=[
            oc.AttributeChange("inv1", "amount", day(1), 0.0),
            oc.AttributeChange("po1", "amount", day(0), 0.0),
        ],
    )
    unit = oc.build_units(log, invoice_spec(), at="2024-06-01T00:00:00Z", strict=False)[0]
    outcome = oc.RelationalBalance(
        left=oc.AmountSelector(attribute="amount", unit="EUR"),
        right=oc.AmountSelector(role="order", attribute="amount", unit="EUR"),
    ).evaluate(log, unit)
    assert outcome.violation == 0.0
    assert QualificationCode.BOTH_TOTALS_ZERO in {q.code for q in outcome.qualifications}


def test_a_balance_selector_is_refused_before_it_can_be_wrong():
    with pytest.raises(OCConstraintError, match="must declare its unit or currency"):
        oc.AmountSelector(attribute="amount")
    with pytest.raises(OCConstraintError, match="must name the attribute"):
        oc.AmountSelector(attribute="", unit="EUR")
    with pytest.raises(OCConstraintError, match="declared_rates needs the target_unit"):
        oc.RelationalBalance(
            left=oc.AmountSelector(unit="EUR"), right=oc.AmountSelector(unit="EUR"), unit_policy="declared_rates"
        )
    with pytest.raises(OCConstraintError, match="must be an AmountSelector"):
        oc.RelationalBalance(left="amount", right=oc.AmountSelector(unit="EUR"))  # type: ignore[arg-type]


def test_one_payment_across_several_invoices_is_one_amount_on_each_side():
    """O04, from the check's side: the payment is not multiplied by the invoices."""
    log = shared_payment((100.0, 60.0, 40.0))
    spec = invoice_spec()
    units = units_of(log, spec, at="2024-06-01T00:00:00Z")
    check = oc.RelationalBalance(
        left=oc.AmountSelector(attribute="amount", unit="EUR"),
        right=oc.AmountSelector(role="payment", attribute="amount", unit="EUR"),
        tolerance=0.0,
    )
    totals = [{m.name: m.value for m in check.evaluate(log, u).measurements} for u in units]
    assert [t["total_left"] for t in totals] == [100.0, 60.0, 40.0]
    assert [t["total_right"] for t in totals] == [200.0, 200.0, 200.0]
    assert all(t["right_records"] == 1.0 for t in totals), "one payment record, read three times, never doubled"


# ================================================================ the separate catalogue
def test_the_native_catalogue_is_its_own_and_loads_through_its_own_path():
    check = object_check_from_dict(
        {"type": "related_object_cardinality", "params": {"role": "order", "minimum": 1, "completeness": "assumed_complete"}}
    )
    assert isinstance(check, oc.RelatedObjectCardinality)
    assert check.completeness is Completeness.ASSUMED_COMPLETE
    lag = object_check_from_dict(
        {
            "type": "cross_object_lag",
            "params": {
                "activation": {"role": "order", "activities": [PO]},
                "response": {"activities": [INV]},
                "delta": 5,
                "width": 10,
            },
        }
    )
    assert isinstance(lag.activation, oc.EventSelector)
    balance = object_check_from_dict(
        {"type": "relational_balance", "params": {"left": {"unit": "EUR"}, "right": {"role": "items", "unit": "EUR"}}}
    )
    assert isinstance(balance.right, oc.AmountSelector)
    assert set(oc.OBJECT_CHECKS) == {"related_object_cardinality", "cross_object_lag", "relational_balance"}


def test_an_unknown_or_malformed_object_check_is_refused_by_name():
    with pytest.raises(OCConstraintError, match="needs a 'type'"):
        object_check_from_dict({"params": {}})
    with pytest.raises(OCConstraintError, match="unknown object check type 'presence'"):
        object_check_from_dict({"type": "presence", "params": {}})
    with pytest.raises(OCConstraintError, match="does not accept"):
        object_check_from_dict({"type": "related_object_cardinality", "params": {"role": "order", "minimum": 1, "m": 3}})


def test_a_native_check_cannot_be_smuggled_into_the_case_norm_schema():
    """The catalogues are separate, and the old loader has never heard of these."""
    payload = {
        "schema_version": 2,
        "layers": [{"id": "matching"}],
        "views": [{"name": "Finance", "constraint_weights": {"m1": 1.0}}],
        "constraints": [{"id": "m1", "layer": "matching", "type": "related_object_cardinality", "params": {"role": "order"}}],
    }
    with pytest.raises(wise.NormError):
        wise.Norm.from_dict(payload)
    assert "related_object_cardinality" not in {
        c.type for c in (wise.Presence("a"), wise.Lag("a", "b"), wise.Balance("x", "a", "y", "b"))
    }


def test_the_object_check_round_trips_through_its_own_dictionary():
    for check in (
        oc.RelatedObjectCardinality(role="order", minimum=1, completeness=Completeness.ASSUMED_COMPLETE),
        po_to_invoice(),
        oc.RelationalBalance(left=oc.AmountSelector(unit="EUR"), right=oc.AmountSelector(role="items", unit="EUR")),
    ):
        payload = check.to_dict()
        assert object_check_from_dict(payload) == check
        assert isinstance(payload, dict)


def test_an_outcome_cannot_claim_a_violation_it_did_not_evaluate():
    from wise.oc.constraints import CheckOutcome

    with pytest.raises(OCConstraintError, match="is not a satisfied check"):
        CheckOutcome(0.0, ReasonCode.AMBIGUOUS_MATCH)
    with pytest.raises(OCConstraintError, match="promises a violation"):
        CheckOutcome(None, ReasonCode.OBSERVED)


def test_the_catalogue_reuses_the_norms_layers_and_views_unchanged():
    """Their meaning has not changed, so they are not re-invented."""
    norm = oc.ObjectNorm(
        constraints=(oc.ObjectConstraint("m1", "matching", oc.RelatedObjectCardinality(role="order", minimum=1)),),
        layers=(Layer("matching", "Match consistency"),),
        views=(View("Finance", constraint_weights={"m1": 1.0}),),
    )
    assert isinstance(norm.layers[0], Layer)
    assert isinstance(norm.views[0], View)
    case_norm = Norm(
        constraints=(NormConstraint("c1", "matching", wise.Presence(INV)),),
        layers=(Layer("matching", "Match consistency"),),
        views=(View("Finance", constraint_weights={"c1": 1.0}),),
    )
    assert case_norm.layers[0] == norm.layers[0], "one Layer class, one meaning"


# ================================================== the per-check evaluation budget
def budgeted_catalogue() -> oc.ObjectNorm:
    """Two checks in two layers, so a budget can stop between them."""
    return oc.ObjectNorm(
        constraints=(
            oc.ObjectConstraint("m1", "matching", oc.RelatedObjectCardinality(role="order", minimum=1)),
            oc.ObjectConstraint("h1", "handling", oc.RelatedObjectCardinality(role="receipts", maximum=1, width=3)),
        ),
        layers=(Layer("matching"), Layer("handling")),
        views=(View("Finance", constraint_weights={"m1": 0.5, "h1": 0.5}),),
        name="budgeted invoice review",
    )


def test_a_bounded_object_evaluation_is_qualified():
    """O03's per-check budget: a cut evaluation says so, with both counts.

    ``max_evaluations`` stops the run part-way through the (unit x check)
    grid. The pairs below the cut have no outcome, so the scores that remain
    are computed over fewer checks than the catalogue declares — which is a
    complete-looking number over an incomplete assessment unless the run says
    otherwise.
    """
    log = partial_deliveries(3)
    spec = invoice_spec()
    units = units_of(log, spec)
    norm = budgeted_catalogue()

    whole = oc.score_units(log, norm, units)
    assert whole.budget.max_evaluations is None
    assert whole.budget.evaluated == whole.budget.in_scope == len(units) * 2
    assert not whole.budget.truncated
    assert QualificationCode.EVALUATION_TRUNCATED not in {q.code for q in whole.qualifications()}

    cut = oc.score_units(log, norm, units, max_evaluations=1)
    assert cut.budget.truncated
    assert (cut.budget.evaluated, cut.budget.in_scope) == (1, len(units) * 2)
    note = next(q for q in cut.qualifications() if q.code is QualificationCode.EVALUATION_TRUNCATED)
    assert note.scope == "run"
    assert f"{cut.budget.evaluated} of {cut.budget.in_scope}" in note.message

    # the pairs below the cut are not out of scope and are not satisfied
    unreached = cut.record(units[0].unit_id, "h1")
    assert unreached.in_scope and not unreached.evaluable
    assert unreached.reason_code is ReasonCode.NOT_EVALUATED_BUDGET
    assert unreached.violation is None

    with pytest.raises(OCConstraintError, match="max_evaluations"):
        oc.score_units(log, norm, units, max_evaluations=0)
