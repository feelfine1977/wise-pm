"""Native object-centric review: relational checks, accounting, and a projection.

Run it from anywhere::

    python examples/oc_invoice_review.py

It builds a small purchase-to-pay object log in code, so it needs no data file
and no optional dependency. Three invoices, one of them matched to an order
with three partial goods receipts, one with no order recorded at all, and one
settled by two payments.

It shows, in order:

1. three native relational checks over preserved relations (O03);
2. a score and a backlog through the *same* numerical kernel the case-based
   :func:`wise.score` uses (O03, the shared kernel);
3. the payment allocated across the invoices it settles, once (O04);
4. the same log flattened onto invoices, and what that costs (O05).

Nothing here is imported by ``import wise``; ``wise.oc`` is an explicit import.
"""

from __future__ import annotations

import pandas as pd

import wise
from wise import oc
from wise.evaluation.ocel import identity_preserving_projection, measure, naive_projection
from wise.norm import Layer, View

T0 = pd.Timestamp("2024-01-01T00:00:00Z")
PO, GR, INV, PAY = "Create Purchase Order", "Record Goods Receipt", "Record Invoice Receipt", "Execute Payment"


def day(offset: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(days=offset)


# ---------------------------------------------------------------- 0. the log
log = oc.OCEventLog.build(
    events=[
        oc.OCEvent("e_po", PO, day(0), {"amount": 90.0}),
        oc.OCEvent("e_gr1", GR, day(1), {"amount": 30.0}),
        oc.OCEvent("e_gr2", GR, day(2), {"amount": 30.0}),
        oc.OCEvent("e_gr3", GR, day(3), {"amount": 30.0}),
        oc.OCEvent("e_inv1", INV, day(12), {"amount": 100.0}),
        oc.OCEvent("e_inv2", INV, day(13), {"amount": 100.0}),
        oc.OCEvent("e_inv3", INV, day(14), {"amount": 100.0}),
        # one payment settling two invoices: the event a case log has to duplicate
        oc.OCEvent("e_pay", PAY, day(30), {"amount": 200.0}),
        oc.OCEvent("e_pay2", PAY, day(35), {"amount": 50.0}),
        oc.OCEvent("e_pay3", PAY, day(36), {"amount": 50.0}),
    ],
    objects=[
        oc.OCObject("po1", "purchase_order"),
        *[oc.OCObject(f"gr{i}", "goods_receipt") for i in (1, 2, 3)],
        *[oc.OCObject(f"inv{i}", "invoice") for i in (1, 2, 3)],
        *[oc.OCObject(f"pay{i}", "payment") for i in (1, 2, 3)],
    ],
    e2o=[
        oc.E2O("e_po", "po1", "purchase_order"),
        *[oc.E2O(f"e_gr{i}", f"gr{i}", "goods_receipt") for i in (1, 2, 3)],
        *[oc.E2O(f"e_gr{i}", "po1", "purchase_order") for i in (1, 2, 3)],
        oc.E2O("e_inv1", "inv1", "invoice"),
        oc.E2O("e_inv1", "po1", "purchase_order"),
        oc.E2O("e_inv2", "inv2", "invoice"),
        oc.E2O("e_inv3", "inv3", "invoice"),
        oc.E2O("e_pay", "pay1", "payment"),
        oc.E2O("e_pay", "inv1", "invoice"),
        oc.E2O("e_pay", "inv2", "invoice"),
        oc.E2O("e_pay2", "pay2", "payment"),
        oc.E2O("e_pay2", "inv3", "invoice"),
        oc.E2O("e_pay3", "pay3", "payment"),
        oc.E2O("e_pay3", "inv3", "invoice"),
    ],
    o2o=[
        oc.O2O("inv1", "po1", "belongs to"),  # inv2 and inv3 record no order
        *[oc.O2O(f"gr{i}", "po1", "receipt for") for i in (1, 2, 3)],
        oc.O2O("pay1", "inv1", "settles"),
        oc.O2O("pay1", "inv2", "settles"),
        oc.O2O("pay2", "inv3", "settles"),
        oc.O2O("pay3", "inv3", "settles"),
    ],
    attribute_history=[
        oc.AttributeChange("po1", "amount", day(0), 90.0),
        *[oc.AttributeChange(f"gr{i}", "amount", day(i), 30.0) for i in (1, 2, 3)],
        *[oc.AttributeChange(f"inv{i}", "amount", day(11 + i), 100.0) for i in (1, 2, 3)],
        oc.AttributeChange("pay1", "amount", day(30), 200.0),
        oc.AttributeChange("pay2", "amount", day(35), 50.0),
        oc.AttributeChange("pay3", "amount", day(36), 50.0),
    ],
)
print(log)

AT = "2024-06-01T00:00:00Z"
spec = oc.UnitSpec(
    unit_type="invoice_review",
    anchor_type="invoice",
    roles=(
        oc.RolePath("order", (oc.PathStep("belongs to", target_type="purchase_order"),)),
        oc.RolePath(
            "receipts",
            (
                oc.PathStep("belongs to", target_type="purchase_order"),
                oc.PathStep("receipt for", target_type="goods_receipt", direction="reverse"),
            ),
        ),
        oc.RolePath("payment", (oc.PathStep("settles", target_type="payment", direction="reverse"),)),
    ),
    scope=oc.UnitScope(roles=("order", "receipts", "payment")),
)
units = oc.build_units(log, spec, at=AT)
print("\n1. Bounded typed units")
for unit in units:
    print(f"   {unit}")

# ------------------------------------------------- 1. three relational checks
norm = oc.ObjectNorm(
    constraints=(
        oc.ObjectConstraint(
            "match",
            "matching",
            # completeness is UNKNOWN, so a missing order is reported as unknown
            oc.RelatedObjectCardinality(role="order", minimum=1),
            description="every invoice belongs to a purchase order",
        ),
        oc.ObjectConstraint(
            "three_way",
            "matching",
            oc.RelationalBalance(
                left=oc.AmountSelector(attribute="amount", unit="EUR"),
                right=oc.AmountSelector(role="receipts", attribute="amount", unit="EUR"),
                tolerance=0.05,
                width=0.20,
            ),
            description="invoiced amount against the goods actually received",
        ),
        oc.ObjectConstraint(
            "settled",
            "handling",
            oc.RelatedObjectCardinality(role="payment", maximum=1),
            description="one invoice, at most one payment",
        ),
        oc.ObjectConstraint(
            "paid_late",
            "lead_times",
            oc.CrossObjectLag(
                activation=oc.EventSelector(activities=(INV,)),
                response=oc.EventSelector(role="payment", activities=(PAY,)),
                delta=10,
                width=20,
                equal_time="counts",
            ),
            description="invoice to payment within ten days",
        ),
    ),
    layers=(Layer("matching"), Layer("lead_times"), Layer("handling")),
    views=(View("Finance", constraint_weights={"match": 0.3, "three_way": 0.3, "settled": 0.1, "paid_late": 0.3}),),
    name="invoice review",
    scoring_mode="flat",
)
result = oc.score_units(log, norm, units, spec=spec)
print("\n2. Violations (NaN = not evaluable, which is not zero)")
print(result.violations.round(4).to_string())
print("\n   Scores and layer contributions")
print(result.frame("Finance").drop(columns=["evaluation_time"]).round(4).to_string())
print(f"   layer decomposition error: {result.check_decomposition():.1e}")

print("\n3. What the run declined to answer, and why")
for unit_id in ("invoice_review:inv2",):
    record = result.record(unit_id, "match")
    print(f"   {unit_id}/match -> {record.reason_code.value}")
    for qualification in record.qualifications:
        print(f"      {qualification.code.value}: {qualification.message[:120]}")

# ------------------------------------------------------------- 2. a backlog
backlog = oc.object_backlog(result, by="anchor_type", view="Finance")
print("\n4. Backlog (one unit type; ranking across types is refused by default)")
print(backlog.round(4).to_string())

# ---------------------------------------------------------- 3. the accounting
payment = oc.QuantityRecord.from_reading(log.value_at("pay1", "amount", AT), "EUR")
assert payment is not None
sharing = [u for u in units if "pay1" in u.role("payment")]
report = oc.allocate(
    [payment],
    [oc.Allocation(payment.record_id, u.unit_id, 1.0 / len(sharing)) for u in sharing],
)
print("\n5. The shared payment, allocated once")
print(report.frame().to_string())
print(f"   complete={report.complete}  total={report.total:.2f} {report.unit}  residual={report.residual_total:.2f}")
combined = report.combined()
print(
    f"   if each review claimed the whole payment: {combined['sum_if_each_group_claimed_all']:.2f}"
    f" for {combined['evidence_value']:.2f} of evidence"
)
for qualification in combined["qualifications"]:
    print(f"      {qualification.code.value}: {qualification.message[:140]}")

# ------------------------------------------------- 4. what a projection costs
print("\n6. The same log flattened onto invoices")
case_norm = wise.Norm(
    constraints=(
        wise.NormConstraint("match", "matching", wise.Presence(PO, m=1)),
        wise.NormConstraint("three_way", "matching", wise.Balance("amount", INV, "amount", GR, tau=0.05, width=0.20)),
        wise.NormConstraint("settled", "handling", wise.Presence(PAY, m=1)),
        wise.NormConstraint("paid_late", "lead_times", wise.Lag(INV, PAY, delta=10, width=20)),
    ),
    layers=(Layer("matching"), Layer("lead_times"), Layer("handling")),
    views=(View("Finance", constraint_weights={"match": 0.3, "three_way": 0.3, "settled": 0.1, "paid_late": 0.3}),),
    name="invoice review, projected",
    scoring_mode="flat",
)
for name, projection in (("naive", naive_projection), ("identity-preserving", identity_preserving_projection)):
    run = measure(lambda projection=projection: projection(log, case_type="invoice", amount_attribute="amount"))
    events, projection_report = run.value
    projected = wise.score(events, case_norm)
    print(f"\n   {name}: {projection_report.n_rows} rows from {projection_report.n_source_events} events")
    print(
        f"      duplicated_amount={projection_report.duplicated_amount:.2f}, "
        f"unassigned_events={projection_report.unassigned_events}, lost_o2o={projection_report.lost_o2o}, "
        f"keeps_event_identity={projection_report.keeps_event_identity}"
    )
    print("      " + projected.violations.round(4).to_string().replace("\n", "\n      "))
    print(f"      {run.seconds * 1000:.2f} ms, peak {run.peak_bytes / 1024:.1f} KiB")

print(
    "\n   The projection reports 'no purchase order' and 'nothing received' for every invoice, "
    "\n   because after flattening there is no purchase order and no goods receipt to find. "
    "\n   It cannot see the second payment on inv3 either. The native evaluation says "
    "\n   'unknown' where the data is silent and 1.0 where the objects prove the excess."
)
