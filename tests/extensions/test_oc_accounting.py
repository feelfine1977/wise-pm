"""Additive quantities: identity, allocation, residual and overlap (O04).

The failure this module exists to prevent is arithmetic, not conceptual. One
payment settles three invoices; three reviews each look at it; three backlogs
each report its value; someone adds them and reports three times the money.
Every test below is a way of making that impossible or, where it is a
legitimate multiple *obligation*, of making the difference explicit.

The distinction the handoff insists on runs through the whole file: the
conservation contract is about **additive quantities**. Three checks resting on
one payment are three checks — that is not double counting. Three backlogs
claiming the payment's value are one amount counted three times, and that is.
"""

from __future__ import annotations

import pandas as pd
import pytest
from _oc_fixtures import invoice_spec, shared_payment

from wise import oc
from wise.errors import AccountingError
from wise.evidence.models import Completeness, QualificationCode
from wise.oc.accounting import (
    DEFAULT_TOLERANCE,
    AccountingIssueCode,
    Allocation,
    QuantityRecord,
    allocate,
    canonical_record_id,
    equal_split,
)


def record(object_id: str, amount: float, unit: str = "EUR", when: str = "2024-01-05T10:00:00Z") -> QuantityRecord:
    return QuantityRecord.build(object_id, "amount", amount, unit, pd.Timestamp(when))


# ------------------------------------------------------------------- identity
def test_a_record_is_identified_by_object_attribute_and_effective_instant():
    assert canonical_record_id("inv1", "amount", pd.Timestamp("2024-01-05T10:00:00Z")) == "inv1:amount@2024-01-05T10:00:00+00:00"
    assert canonical_record_id("inv1", "amount", None) == "inv1:amount@unknown"


def test_a_changed_amount_is_a_different_record_not_a_correction():
    first = record("inv1", 100.0, when="2024-01-05T10:00:00Z")
    second = record("inv1", 120.0, when="2024-01-20T10:00:00Z")
    assert first.record_id != second.record_id
    report = allocate([first, second], [Allocation(first.record_id, "review", 1.0), Allocation(second.record_id, "review", 1.0)])
    assert report.total == 220.0, "two recorded values are two records; netting them needs a declared rule"


def test_two_different_values_claiming_one_identity_are_refused():
    a = QuantityRecord("inv1:amount@x", "inv1", "amount", 100.0, "EUR")
    b = QuantityRecord("inv1:amount@x", "inv1", "amount", 120.0, "EUR")
    with pytest.raises(AccountingError, match="claimed by two different values"):
        allocate([a, b])


def test_a_record_needs_a_finite_amount_and_a_declared_unit():
    with pytest.raises(AccountingError, match="needs a declared unit"):
        QuantityRecord.build("inv1", "amount", 10.0, "")
    with pytest.raises(AccountingError, match="must be finite"):
        QuantityRecord.build("inv1", "amount", float("nan"), "EUR")
    with pytest.raises(AccountingError, match="is not a number"):
        QuantityRecord.build("inv1", "amount", "one hundred", "EUR")


def test_a_reading_that_found_nothing_produces_no_record():
    log = shared_payment()
    reading = log.value_at("pay1", "amount", "2024-01-02T00:00:00Z")
    assert not reading.found
    assert QuantityRecord.from_reading(reading, "EUR") is None, "a missing amount is missing, not zero"
    later = log.value_at("pay1", "amount", "2024-06-01T00:00:00Z")
    built = QuantityRecord.from_reading(later, "EUR")
    assert built is not None and built.amount == 200.0


# ----------------------------------------------------------------- allocation
def test_a_complete_allocation_sums_to_one_and_leaves_no_residual():
    payment = record("pay1", 200.0)
    report = allocate(
        [payment],
        [Allocation(payment.record_id, "invoice-1", 0.5), Allocation(payment.record_id, "invoice-2", 0.5)],
    )
    assert report.complete
    assert report.residual == {}
    assert report.allocated == {"invoice-1": 100.0, "invoice-2": 100.0}
    assert report.total == 200.0
    assert report.qualifications() == ()


def test_an_incomplete_allocation_reports_a_residual_rather_than_failing():
    payment = record("pay1", 200.0)
    report = allocate([payment], [Allocation(payment.record_id, "invoice-1", 0.25)])
    assert not report.complete
    assert report.residual == {payment.record_id: 150.0}
    assert report.residual_total == 150.0
    assert [i.code for i in report.issues] == [AccountingIssueCode.INCOMPLETE_ALLOCATION]
    codes = {q.code for q in report.qualifications()}
    assert QualificationCode.ALLOCATION_INCOMPLETE in codes
    assert "belongs to no group" in report.qualifications()[0].message


def test_require_complete_turns_a_residual_into_a_refusal():
    payment = record("pay1", 200.0)
    with pytest.raises(AccountingError, match="a complete allocation was required"):
        allocate([payment], [Allocation(payment.record_id, "invoice-1", 0.25)], require_complete=True)


def test_a_negative_share_is_refused_because_netting_needs_a_rule():
    payment = record("pay1", 200.0)
    with pytest.raises(AccountingError, match="a negative share is a credit note"):
        allocate([payment], [Allocation(payment.record_id, "invoice-1", -0.5)])


def test_one_group_cannot_consume_one_record_twice():
    payment = record("pay1", 200.0)
    with pytest.raises(AccountingError, match=r"consumes record .* twice"):
        allocate(
            [payment],
            [Allocation(payment.record_id, "invoice-1", 0.5), Allocation(payment.record_id, "invoice-1", 0.5)],
        )


def test_over_allocation_is_refused_across_groups_too():
    """O04: matching capacity is declared; the payment cannot settle 300 of 200."""
    payment = record("pay1", 200.0)
    with pytest.raises(AccountingError, match="counted more than once"):
        allocate(
            [payment],
            [
                Allocation(payment.record_id, "invoice-1", 0.6),
                Allocation(payment.record_id, "invoice-2", 0.6),
            ],
        )


def test_an_allocation_cannot_create_the_amount_it_consumes():
    with pytest.raises(AccountingError, match="cannot create the amount it consumes"):
        allocate([record("pay1", 200.0)], [Allocation("does-not-exist", "invoice-1", 1.0)])


def test_the_tolerance_is_a_tolerance_and_not_a_licence():
    payment = record("pay1", 200.0)
    just_over = allocate([payment], [Allocation(payment.record_id, "g", 1.0 + DEFAULT_TOLERANCE / 2)])
    assert just_over.complete
    with pytest.raises(AccountingError):
        allocate([payment], [Allocation(payment.record_id, "g", 1.01)])


def test_equal_split_is_a_declared_rule_and_completes_the_allocation():
    payment = record("pay1", 90.0)
    report = allocate([payment], equal_split([payment], ["a", "b", "c"]))
    assert report.complete
    assert report.allocated == pytest.approx({"a": 30.0, "b": 30.0, "c": 30.0})
    assert "equal split over 3 group(s)" in report.allocations[0].note
    with pytest.raises(AccountingError, match="at least one group"):
        equal_split([payment], [])


# --------------------------------------------------------------------- units
def test_two_currencies_are_not_added_without_a_declared_rate():
    euros, dollars = record("inv1", 100.0, "EUR"), record("inv2", 100.0, "USD")
    with pytest.raises(AccountingError, match="would invent an exchange rate"):
        allocate([euros, dollars])
    converted = allocate([euros, dollars], rates={"USD": 0.9}, unit="EUR")
    assert converted.unit == "EUR"
    assert converted.total == pytest.approx(190.0)


def test_a_declared_rate_must_cover_every_unit_and_be_positive():
    euros, dollars = record("inv1", 100.0, "EUR"), record("inv2", 100.0, "USD")
    with pytest.raises(AccountingError, match="needs the unit it converts to"):
        allocate([euros, dollars], rates={"USD": 0.9})
    with pytest.raises(AccountingError, match="no declared rate from 'USD'"):
        allocate([euros, dollars], rates={"GBP": 1.2}, unit="EUR")
    with pytest.raises(AccountingError, match="must be finite and positive"):
        allocate([euros, dollars], rates={"USD": 0.0}, unit="EUR")


def test_a_target_unit_that_contradicts_the_records_is_refused():
    with pytest.raises(AccountingError, match="no rate was declared"):
        allocate([record("inv1", 100.0, "EUR")], unit="USD")


# ------------------------------------------------------- coverage and overlap
def three_reviews():
    """One payment and two invoices, seen by three overlapping reviews."""
    payment = record("pay1", 200.0)
    inv1, inv2 = record("inv1", 120.0), record("inv2", 80.0)
    allocations = [
        Allocation(payment.record_id, "vendor_review", 0.5),
        Allocation(payment.record_id, "late_payment_review", 0.5),
        Allocation(inv1.record_id, "vendor_review", 1.0),
        Allocation(inv2.record_id, "late_payment_review", 1.0),
    ]
    return allocate([payment, inv1, inv2], allocations)


def test_coverage_reports_how_much_distinct_evidence_a_group_rests_on():
    report = three_reviews()
    coverage = report.coverage()
    assert coverage["vendor_review"]["amount"] == 220.0
    assert coverage["vendor_review"]["distinct_records"] == 2.0
    assert coverage["vendor_review"]["distinct_objects"] == 2.0
    assert coverage["late_payment_review"]["amount"] == 180.0
    assert coverage["vendor_review"]["share_of_total"] == pytest.approx(220.0 / 400.0)


def test_overlap_names_the_records_two_groups_share():
    report = three_reviews()
    overlap = report.overlap()
    assert list(overlap) == [("late_payment_review", "vendor_review")]
    shared = overlap[("late_payment_review", "vendor_review")]
    assert shared["n_shared_records"] == 1.0
    assert shared["shared_amount"] == 200.0


def test_overlapping_groups_are_never_summed_and_the_difference_is_a_number():
    payment = record("pay1", 200.0)
    report = allocate(
        [payment],
        [Allocation(payment.record_id, "vendor_review", 0.5), Allocation(payment.record_id, "late_payment_review", 0.5)],
    )
    combined = report.combined(["vendor_review", "late_payment_review"])
    assert combined["allocated"] == 200.0, "the declared shares are conserved and safe to add"
    assert combined["evidence_value"] == 200.0, "one payment, worth its own amount once"
    assert combined["sum_if_each_group_claimed_all"] == 400.0, "what reporting the full value per group would claim"
    assert combined["double_counted"] == 200.0
    codes = {q.code for q in combined["qualifications"]}
    assert QualificationCode.OVERLAPPING_GROUPS_NOT_SUMMED in codes
    message = combined["qualifications"][0].message
    assert "counted twice, not a second amount" in message
    assert "the conserved figure is the allocated 200.00" in message


def test_a_shared_credit_note_is_qualified_like_any_other_shared_record():
    """A reversal is the case where a double count is most embarrassing.

    Two groups share a −200 credit note. The naive sum is −400 against evidence
    worth −200, so exactly one amount is counted twice — and a guard written as
    ``naive > evidence`` skips the whole family of returns, reversals and
    credit notes.
    """
    credit = record("cn1", -200.0)
    report = allocate(
        [credit],
        [Allocation(credit.record_id, "vendor_review", 0.5), Allocation(credit.record_id, "late_payment_review", 0.5)],
    )
    combined = report.combined(["vendor_review", "late_payment_review"])
    assert combined["sum_if_each_group_claimed_all"] == -400.0
    assert combined["evidence_value"] == -200.0
    assert combined["double_counted"] == -200.0
    assert list(report.overlap()) == [("late_payment_review", "vendor_review")]

    codes = {q.code for q in combined["qualifications"]}
    assert QualificationCode.OVERLAPPING_GROUPS_NOT_SUMMED in codes, "a shared negative is shared evidence too"
    message = combined["qualifications"][0].message
    assert "-200.00" in message, "the message states the sign of the difference"
    assert "counted twice, not a second amount" in message


def test_disjoint_groups_add_up_without_a_warning():
    inv1, inv2 = record("inv1", 120.0), record("inv2", 80.0)
    report = allocate([inv1, inv2], [Allocation(inv1.record_id, "a", 1.0), Allocation(inv2.record_id, "b", 1.0)])
    combined = report.combined()
    assert combined["evidence_value"] == combined["sum_if_each_group_claimed_all"] == 200.0
    assert combined["allocated"] == 200.0
    assert combined["double_counted"] == 0.0
    assert combined["qualifications"] == ()


def test_combining_an_unknown_group_is_refused():
    with pytest.raises(AccountingError, match="unknown group"):
        three_reviews().combined(["nobody"])


def test_several_obligations_may_rest_on_one_payment_counted_once():
    """The distinction that keeps this contract from being wrong.

    Three checks look at one payment. That is three obligations — a real,
    non-duplicated multiple. What must not triple is the *money*.
    """
    payment = record("pay1", 200.0)
    obligations = ["three_way_match", "payment_terms", "duplicate_payment"]
    report = allocate([payment], equal_split([payment], obligations))
    assert len(report.records_of("three_way_match")) == 1
    assert len(report.groups) == 3, "three obligations remain three obligations"
    assert sum(report.allocated.values()) == pytest.approx(200.0), "and one amount remains one amount"
    combined = report.combined(obligations)
    assert combined["allocated"] == pytest.approx(200.0)
    assert combined["evidence_value"] == 200.0
    assert combined["sum_if_each_group_claimed_all"] == 600.0, "three checks reporting the payment's full value"
    assert combined["double_counted"] == 400.0


# -------------------------------------------------------- from a real object log
def test_one_payment_across_several_invoices_allocates_without_repetition():
    """O04 end to end: the payment of a three-invoice settlement, once."""
    log = shared_payment((100.0, 60.0, 40.0))
    at = "2024-06-01T00:00:00Z"
    units = oc.build_units(log, invoice_spec(), at=at, strict=False)
    payment = QuantityRecord.from_reading(log.value_at("pay1", "amount", at), "EUR")
    assert payment is not None and payment.amount == 200.0

    invoices = [QuantityRecord.from_reading(log.value_at(u.anchor_id, "amount", at), "EUR") for u in units]
    shares = {u.unit_id: inv.amount / payment.amount for u, inv in zip(units, invoices)}
    report = allocate([payment], [Allocation(payment.record_id, unit_id, share) for unit_id, share in shares.items()])
    assert report.complete, "the three invoices settle exactly the payment"
    assert report.allocated == pytest.approx(
        {"invoice_review:inv1": 100.0, "invoice_review:inv2": 60.0, "invoice_review:inv3": 40.0}
    )
    assert sum(report.allocated.values()) == pytest.approx(200.0)
    overlap = report.overlap()
    assert len(overlap) == 3, "all three invoices draw on the one payment record, and the report says so"
    assert all(v["n_shared_records"] == 1.0 for v in overlap.values())
    assert report.combined()["allocated"] == pytest.approx(200.0), "their declared shares still add to one payment"


def test_a_partial_settlement_leaves_a_residual_that_belongs_to_nobody():
    log = shared_payment((100.0, 60.0, 40.0))
    at = "2024-06-01T00:00:00Z"
    payment = QuantityRecord.from_reading(log.value_at("pay1", "amount", at), "EUR")
    assert payment is not None
    report = allocate([payment], [Allocation(payment.record_id, "invoice_review:inv1", 0.5)])
    assert report.residual_total == 100.0
    assert not report.complete


# ------------------------------------------------------------------ reporting
def test_the_report_renders_a_frame_and_a_dictionary_that_agree():
    report = three_reviews()
    frame = report.frame()
    assert list(frame.columns) == [
        "object_id",
        "attribute",
        "amount",
        "unit",
        "consumed_share",
        "residual",
        "n_groups",
        "groups",
    ]
    assert frame.loc["pay1:amount@2024-01-05T10:00:00+00:00", "n_groups"] == 2
    assert frame["amount"].sum() == report.total
    payload = report.to_dict()
    assert payload["total"] == 400.0
    assert payload["complete"] is True
    assert "late_payment_review|vendor_review" in payload["overlap"]
    assert payload["coverage"]["vendor_review"]["distinct_records"] == 2.0


def test_an_empty_allocation_is_a_report_and_not_an_error():
    report = allocate([])
    assert report.total == 0.0
    assert report.groups == ()
    assert report.frame().empty
    assert repr(report).startswith("AllocationReport(0 record(s)")


def test_an_unknown_record_lookup_is_refused():
    with pytest.raises(AccountingError, match="unknown record"):
        three_reviews().record("nope")


def test_the_accounting_vocabulary_is_reachable_from_the_package():
    assert oc.QuantityRecord is QuantityRecord
    assert oc.Allocation is Allocation
    assert oc.allocate is allocate
    assert oc.canonical_record_id is canonical_record_id
    assert oc.equal_split is equal_split
    assert Completeness.ASSUMED_COMPLETE.value == "assumed_complete"
