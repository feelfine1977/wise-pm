"""Small object-centric logs with answers a human can write down (O03–O05).

Every builder here produces a log whose right answer is obvious by inspection,
which is the only kind of fixture worth comparing a projection against. They
are deliberately tiny; the published OCEL 2.0 logs are a technical benchmark
and are exercised elsewhere, gated on the data being present.

The named awkward cases the handoff asks for:

``many_items``           one invoice, one order, 100 items, one amount
``shared_payment``       one payment settling several invoices
``partial_deliveries``   one order, several goods receipts, one invoice
``equal_timestamps``     distinct events sharing an instant, on purpose
``changing_amount``      an attribute whose value depends on when you ask
``unknown_matching``     an invoice with no recorded order at all
``repeated_rows``        identical relation rows, and dangling ones
``wide_vendor``          one vendor shared by many invoices (fan-out)
``case_shaped``          the running P2P example as an object log, so that the
                         legacy calculation can be reproduced exactly
"""

from __future__ import annotations

import pandas as pd

import wise
from wise.norm import Layer, Norm, NormConstraint, View
from wise.oc import (
    E2O,
    O2O,
    AttributeChange,
    CrossObjectLag,
    EventSelector,
    ObjectConstraint,
    ObjectNorm,
    OCEvent,
    OCEventLog,
    OCObject,
    PathStep,
    RelatedObjectCardinality,
    RolePath,
    SourceMetadata,
    TraversalLimits,
    UnitScope,
    UnitSpec,
)

T0 = pd.Timestamp("2024-01-01T00:00:00Z")

PO = "Create Purchase Order"
GR = "Record Goods Receipt"
INV = "Record Invoice Receipt"
PAY = "Execute Payment"


def day(offset: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(days=offset)


def in_memory(**changes: object) -> SourceMetadata:
    return SourceMetadata(interchange="in-memory", **changes)  # type: ignore[arg-type]


# --------------------------------------------------------------------- fixtures
def many_items(n: int = 100) -> OCEventLog:
    """One order with ``n`` items, one invoice for 1000, one purchase event.

    The purchase event touches the order *and* every item, which is what makes
    a projection onto items multiply the amount by ``n``.
    """
    items = [OCObject(f"it{i}", "item") for i in range(1, n + 1)]
    return OCEventLog.build(
        events=[
            OCEvent("e_po", PO, day(0), {"amount": 1000.0}),
            OCEvent("e_inv", INV, day(9), {"amount": 1000.0}),
        ],
        objects=[OCObject("po1", "purchase_order"), OCObject("inv1", "invoice"), *items],
        e2o=[
            E2O("e_po", "po1", "purchase_order"),
            *[E2O("e_po", item.object_id, "item") for item in items],
            E2O("e_inv", "inv1", "invoice"),
            E2O("e_inv", "po1", "purchase_order"),
        ],
        o2o=[
            O2O("inv1", "po1", "belongs to"),
            *[O2O("po1", item.object_id, "has item") for item in items],
        ],
        attribute_history=[
            AttributeChange("inv1", "amount", day(9), 1000.0),
            *[AttributeChange(item.object_id, "quantity", day(0), 10.0) for item in items],
        ],
        source=in_memory(),
    )


def shared_payment(amounts: tuple[float, ...] = (100.0, 60.0, 40.0)) -> OCEventLog:
    """One payment settling several invoices; the payment event carries the total."""
    total = float(sum(amounts))
    invoices = [OCObject(f"inv{i}", "invoice") for i in range(1, len(amounts) + 1)]
    events = [OCEvent("e_po", PO, day(0)), OCEvent("e_pay", PAY, day(30), {"amount": total})]
    e2o = [E2O("e_po", "po1", "purchase_order"), E2O("e_pay", "pay1", "payment")]
    o2o = []
    history = [AttributeChange("pay1", "amount", day(30), total)]
    for i, (invoice, amount) in enumerate(zip(invoices, amounts), start=1):
        events.append(OCEvent(f"e_inv{i}", INV, day(5 + i), {"amount": amount}))
        e2o += [
            E2O(f"e_inv{i}", invoice.object_id, "invoice"),
            E2O(f"e_inv{i}", "po1", "purchase_order"),
            E2O("e_pay", invoice.object_id, "invoice"),
        ]
        o2o += [O2O(invoice.object_id, "po1", "belongs to"), O2O("pay1", invoice.object_id, "settles")]
        history.append(AttributeChange(invoice.object_id, "amount", day(5 + i), amount))
    return OCEventLog.build(
        events=events,
        objects=[OCObject("po1", "purchase_order"), OCObject("pay1", "payment"), *invoices],
        e2o=e2o,
        o2o=o2o,
        attribute_history=history,
        source=in_memory(),
    )


def partial_deliveries(n: int = 3) -> OCEventLog:
    """One order, ``n`` goods receipts of 30 each, one invoice for 100.

    Ordered 90, invoiced 100: an 11.1 % mismatch that only exists once the
    receipts are added up *through the order*, which a per-invoice case cannot
    do.
    """
    receipts = [OCObject(f"gr{i}", "goods_receipt") for i in range(1, n + 1)]
    events = [OCEvent("e_po", PO, day(0)), OCEvent("e_inv", INV, day(12))]
    e2o = [E2O("e_po", "po1", "purchase_order"), E2O("e_inv", "inv1", "invoice"), E2O("e_inv", "po1", "purchase_order")]
    history = [AttributeChange("inv1", "amount", day(12), 100.0)]
    o2o = [O2O("inv1", "po1", "belongs to")]
    for i, receipt in enumerate(receipts, start=1):
        events.append(OCEvent(f"e_gr{i}", GR, day(i)))
        e2o += [E2O(f"e_gr{i}", receipt.object_id, "goods_receipt"), E2O(f"e_gr{i}", "po1", "purchase_order")]
        o2o.append(O2O(receipt.object_id, "po1", "receipt for"))
        history.append(AttributeChange(receipt.object_id, "amount", day(i), 30.0))
    return OCEventLog.build(
        events=events,
        objects=[OCObject("po1", "purchase_order"), OCObject("inv1", "invoice"), *receipts],
        e2o=e2o,
        o2o=o2o,
        attribute_history=history,
        source=in_memory(),
    )


def equal_timestamps() -> OCEventLog:
    """Two goods receipts and two invoice receipts, each pair sharing an instant.

    Four distinct events, two timestamps. Nothing here is a duplicate: the
    warehouse booked two deliveries in the same minute, and the extract records
    the minute.
    """
    return OCEventLog.build(
        events=[
            OCEvent("e_po", PO, day(0)),
            OCEvent("e_gr1", GR, day(2)),
            OCEvent("e_gr2", GR, day(2)),
            OCEvent("e_inv1", INV, day(6)),
            OCEvent("e_inv2", INV, day(6)),
        ],
        objects=[
            OCObject("po1", "purchase_order"),
            OCObject("inv1", "invoice"),
            OCObject("gr1", "goods_receipt"),
            OCObject("gr2", "goods_receipt"),
        ],
        e2o=[
            E2O("e_po", "po1", "purchase_order"),
            E2O("e_gr1", "gr1", "goods_receipt"),
            E2O("e_gr1", "po1", "purchase_order"),
            E2O("e_gr2", "gr2", "goods_receipt"),
            E2O("e_gr2", "po1", "purchase_order"),
            E2O("e_inv1", "inv1", "invoice"),
            E2O("e_inv2", "inv1", "invoice"),
        ],
        o2o=[
            O2O("inv1", "po1", "belongs to"),
            O2O("gr1", "po1", "receipt for"),
            O2O("gr2", "po1", "receipt for"),
        ],
        source=in_memory(),
    )


def changing_amount() -> OCEventLog:
    """An invoice whose amount and owner change; the order's total does not."""
    return OCEventLog.build(
        events=[OCEvent("e_po", PO, day(0)), OCEvent("e_inv", INV, day(4)), OCEvent("e_fix", "Correct Invoice", day(20))],
        objects=[OCObject("po1", "purchase_order"), OCObject("inv1", "invoice"), OCObject("it1", "item")],
        e2o=[
            E2O("e_po", "po1", "purchase_order"),
            E2O("e_po", "it1", "item"),
            E2O("e_inv", "inv1", "invoice"),
            E2O("e_fix", "inv1", "invoice"),
        ],
        o2o=[O2O("inv1", "po1", "belongs to"), O2O("po1", "it1", "has item")],
        attribute_history=[
            AttributeChange("inv1", "amount", day(4), 100.0),
            AttributeChange("inv1", "amount", day(20), 120.0),
            AttributeChange("inv1", "owner", day(4), "team-a"),
            AttributeChange("inv1", "owner", day(20), "team-b"),
            AttributeChange("it1", "amount", day(0), 100.0),
        ],
        source=in_memory(),
    )


def unknown_matching() -> OCEventLog:
    """Two invoices; one belongs to an order, the other records no order at all.

    ``inv2`` is not evidence that no order exists. It is evidence that this
    extract does not say.
    """
    return OCEventLog.build(
        events=[OCEvent("e_po", PO, day(0)), OCEvent("e_inv1", INV, day(3)), OCEvent("e_inv2", INV, day(4))],
        objects=[OCObject("po1", "purchase_order"), OCObject("inv1", "invoice"), OCObject("inv2", "invoice")],
        e2o=[
            E2O("e_po", "po1", "purchase_order"),
            E2O("e_inv1", "inv1", "invoice"),
            E2O("e_inv1", "po1", "purchase_order"),
            E2O("e_inv2", "inv2", "invoice"),
        ],
        o2o=[O2O("inv1", "po1", "belongs to")],
        attribute_history=[
            AttributeChange("inv1", "amount", day(3), 50.0),
            AttributeChange("inv2", "amount", day(4), 50.0),
        ],
        source=in_memory(),
    )


def repeated_rows() -> tuple[list[OCEvent], list[OCObject], list[E2O], list[O2O]]:
    """Raw tables with identical repeats and one dangling relation.

    Returned unbuilt so that a test can choose the policy: the default refuses
    the dangling row, ``on_dangling='drop'`` records it — the policy the
    published procure-to-pay log needs for its 2 028 relations to an object it
    never declares.
    """
    events = [OCEvent("e1", INV, day(1)), OCEvent("e1", INV, day(1))]
    objects = [OCObject("inv1", "invoice"), OCObject("po1", "purchase_order"), OCObject("po1", "purchase_order")]
    e2o = [E2O("e1", "inv1", "invoice"), E2O("e1", "inv1", "invoice")]
    o2o = [O2O("inv1", "po1", "belongs to"), O2O("inv1", "po1", "belongs to"), O2O("inv1", "ghost", "belongs to")]
    return events, objects, e2o, o2o


def wide_vendor(n: int = 40) -> OCEventLog:
    """One vendor, ``n`` invoices. Traversing through it reaches the whole ledger."""
    invoices = [OCObject(f"inv{i}", "invoice") for i in range(1, n + 1)]
    return OCEventLog.build(
        events=[OCEvent("e1", INV, day(1))],
        objects=[OCObject("v1", "vendor"), *invoices],
        e2o=[E2O("e1", "inv1", "invoice")],
        o2o=[O2O(inv.object_id, "v1", "billed by") for inv in invoices],
        source=in_memory(),
    )


# ------------------------------------------------- the case-shaped special case
def case_shaped() -> OCEventLog:
    """The running P2P example as an object log, one ``case`` object per case.

    Every goods receipt also becomes a ``goods_receipt`` object related to its
    case, so that counting the objects of a role and counting the events of an
    activity are the same number — which is what makes the legacy calculation
    reproducible through the native path rather than approximated by it.
    """
    events_frame = wise.running_p2p_events().reset_index(drop=True)
    cases = sorted(events_frame["case"].unique())
    objects = [OCObject(c, "case") for c in cases]
    events: list[OCEvent] = []
    e2o: list[E2O] = []
    o2o: list[O2O] = []
    receipts = 0
    for position, row in events_frame.iterrows():
        event_id = f"e{position:03d}"
        events.append(OCEvent(event_id, str(row["activity"]), pd.Timestamp(row["time"], tz="UTC")))
        e2o.append(E2O(event_id, str(row["case"]), "case"))
        if row["activity"] == GR:
            receipts += 1
            receipt_id = f"gr{receipts:03d}"
            objects.append(OCObject(receipt_id, "goods_receipt"))
            e2o.append(E2O(event_id, receipt_id, "goods_receipt"))
            o2o.append(O2O(receipt_id, str(row["case"]), "receipt for"))
    return OCEventLog.build(events=events, objects=objects, e2o=e2o, o2o=o2o, source=in_memory())


def case_shaped_spec() -> UnitSpec:
    """One unit per case, binding the goods receipts of that case."""
    return UnitSpec(
        unit_type="case_review",
        anchor_type="case",
        roles=(RolePath("receipts", (PathStep("receipt for", target_type="goods_receipt", direction="reverse"),)),),
        scope=UnitScope(roles=("receipts",)),
        limits=TraversalLimits(max_fan_out=100, max_bindings=100),
    )


#: The two running-example constraints the native catalogue can express exactly.
LEGACY_WEIGHTS = {
    "Finance": {"c2": 0.45, "c5": 0.05},
    "Logistics": {"c2": 0.15, "c5": 0.45},
}


def legacy_case_norm(mode: str = "flat") -> Norm:
    """``c2`` (lag) and ``c5`` (delivery fragmentation) of the paper's Table V."""
    return Norm(
        constraints=(
            NormConstraint("c2", "lead_times", wise.Lag(GR, INV, delta=10, width=20, unit="D")),
            NormConstraint("c5", "handling", wise.Singularity(GR, k=2, K=3)),
        ),
        layers=(Layer("lead_times"), Layer("handling")),
        views=tuple(View(name, constraint_weights=w) for name, w in LEGACY_WEIGHTS.items()),
        name="legacy subset",
        version="1",
        scoring_mode=mode,
    )


def legacy_object_norm(mode: str = "flat") -> ObjectNorm:
    """The same two obligations, stated natively over objects and relations.

    ``c2`` reaches its activation events *through* the ``receipts`` relation
    rather than through the case column, and ``c5`` counts goods-receipt
    objects rather than goods-receipt events. Both reduce to the legacy
    arithmetic on this data, which is the point of the special case.
    """
    return ObjectNorm(
        constraints=(
            ObjectConstraint(
                "c2",
                "lead_times",
                CrossObjectLag(
                    activation=EventSelector(role="receipts", activities=(GR,), occurrence="first"),
                    response=EventSelector(activities=(INV,)),
                    delta=10,
                    width=20,
                    unit="D",
                    match="first_after",
                    equal_time="counts",
                    missing_activation="violate",
                    missing_response="violate",
                    on_ambiguous="tie_break",
                ),
            ),
            ObjectConstraint("c5", "handling", RelatedObjectCardinality(role="receipts", maximum=2, width=3)),
        ),
        layers=(Layer("lead_times"), Layer("handling")),
        views=tuple(View(name, constraint_weights=w) for name, w in LEGACY_WEIGHTS.items()),
        name="legacy subset, natively",
        version="1",
        scoring_mode=mode,
    )


# ------------------------------------------------------------------ unit specs
def invoice_spec(**changes: object) -> UnitSpec:
    """An invoice, its order, that order's items and receipts, and its payment."""
    payload: dict[str, object] = {
        "unit_type": "invoice_review",
        "anchor_type": "invoice",
        "roles": (
            RolePath("order", (PathStep("belongs to", target_type="purchase_order"),)),
            RolePath(
                "items",
                (
                    PathStep("belongs to", target_type="purchase_order"),
                    PathStep("has item", target_type="item"),
                ),
            ),
            RolePath(
                "receipts",
                (
                    PathStep("belongs to", target_type="purchase_order"),
                    PathStep("receipt for", target_type="goods_receipt", direction="reverse"),
                ),
            ),
            RolePath("payment", (PathStep("settles", target_type="payment", direction="reverse"),)),
        ),
        "scope": UnitScope(roles=("order", "receipts", "payment")),
        "limits": TraversalLimits(max_fan_out=200, max_bindings=200, max_visited=5000),
    }
    payload.update(changes)
    return UnitSpec(**payload)  # type: ignore[arg-type]


# --------------------------------------------------- the projection comparison log
def comparison_log() -> OCEventLog:
    """Three invoices whose right answers a reader can check by eye (O05).

    * ``inv1`` belongs to an order with three goods receipts of 30 against an
      invoiced 100, and is settled by one payment shared with ``inv2``.
    * ``inv2`` records **no** order at all — unknown, not absent.
    * ``inv3`` records no order either and is settled by *two* payments.

    Only the payment events touch more than one invoice, which is exactly what
    a projection onto invoices has to duplicate.
    """
    return OCEventLog.build(
        events=[
            OCEvent("e_po", PO, day(0), {"amount": 90.0}),
            OCEvent("e_gr1", GR, day(1), {"amount": 30.0}),
            OCEvent("e_gr2", GR, day(2), {"amount": 30.0}),
            OCEvent("e_gr3", GR, day(3), {"amount": 30.0}),
            OCEvent("e_inv1", INV, day(12), {"amount": 100.0}),
            OCEvent("e_inv2", INV, day(13), {"amount": 100.0}),
            OCEvent("e_inv3", INV, day(14), {"amount": 100.0}),
            OCEvent("e_pay", PAY, day(30), {"amount": 200.0}),
            OCEvent("e_pay2", PAY, day(35), {"amount": 50.0}),
            OCEvent("e_pay3", PAY, day(36), {"amount": 50.0}),
        ],
        objects=[
            OCObject("po1", "purchase_order"),
            OCObject("gr1", "goods_receipt"),
            OCObject("gr2", "goods_receipt"),
            OCObject("gr3", "goods_receipt"),
            OCObject("inv1", "invoice"),
            OCObject("inv2", "invoice"),
            OCObject("inv3", "invoice"),
            OCObject("pay1", "payment"),
            OCObject("pay2", "payment"),
            OCObject("pay3", "payment"),
        ],
        e2o=[
            E2O("e_po", "po1", "purchase_order"),
            E2O("e_gr1", "gr1", "goods_receipt"),
            E2O("e_gr1", "po1", "purchase_order"),
            E2O("e_gr2", "gr2", "goods_receipt"),
            E2O("e_gr2", "po1", "purchase_order"),
            E2O("e_gr3", "gr3", "goods_receipt"),
            E2O("e_gr3", "po1", "purchase_order"),
            E2O("e_inv1", "inv1", "invoice"),
            E2O("e_inv1", "po1", "purchase_order"),
            E2O("e_inv2", "inv2", "invoice"),
            E2O("e_inv3", "inv3", "invoice"),
            E2O("e_pay", "pay1", "payment"),
            E2O("e_pay", "inv1", "invoice"),
            E2O("e_pay", "inv2", "invoice"),
            E2O("e_pay2", "pay2", "payment"),
            E2O("e_pay2", "inv3", "invoice"),
            E2O("e_pay3", "pay3", "payment"),
            E2O("e_pay3", "inv3", "invoice"),
        ],
        o2o=[
            O2O("inv1", "po1", "belongs to"),
            O2O("gr1", "po1", "receipt for"),
            O2O("gr2", "po1", "receipt for"),
            O2O("gr3", "po1", "receipt for"),
            O2O("pay1", "inv1", "settles"),
            O2O("pay1", "inv2", "settles"),
            O2O("pay2", "inv3", "settles"),
            O2O("pay3", "inv3", "settles"),
        ],
        attribute_history=[
            AttributeChange("po1", "amount", day(0), 90.0),
            AttributeChange("gr1", "amount", day(1), 30.0),
            AttributeChange("gr2", "amount", day(2), 30.0),
            AttributeChange("gr3", "amount", day(3), 30.0),
            AttributeChange("inv1", "amount", day(12), 100.0),
            AttributeChange("inv2", "amount", day(13), 100.0),
            AttributeChange("inv3", "amount", day(14), 100.0),
            AttributeChange("pay1", "amount", day(30), 200.0),
            AttributeChange("pay2", "amount", day(35), 50.0),
            AttributeChange("pay3", "amount", day(36), 50.0),
        ],
        source=in_memory(),
    )


def comparison_object_norm() -> ObjectNorm:
    """The five obligations, stated natively over objects and relations."""
    from wise.oc import AmountSelector, RelationalBalance

    return ObjectNorm(
        constraints=(
            ObjectConstraint("match", "matching", RelatedObjectCardinality(role="order", minimum=1)),
            ObjectConstraint(
                "three_way",
                "matching",
                RelationalBalance(
                    left=AmountSelector(attribute="amount", unit="EUR"),
                    right=AmountSelector(role="receipts", attribute="amount", unit="EUR"),
                    tolerance=0.05,
                    width=0.20,
                ),
            ),
            ObjectConstraint(
                "gr_to_invoice",
                "lead_times",
                CrossObjectLag(
                    activation=EventSelector(role="receipts", activities=(GR,)),
                    response=EventSelector(activities=(INV,)),
                    delta=5,
                    width=10,
                    equal_time="counts",
                    missing_activation="skip",
                ),
            ),
            ObjectConstraint("settled", "handling", RelatedObjectCardinality(role="payment", maximum=1)),
            ObjectConstraint(
                "paid_late",
                "lead_times",
                CrossObjectLag(
                    activation=EventSelector(activities=(INV,)),
                    response=EventSelector(role="payment", activities=(PAY,)),
                    delta=10,
                    width=20,
                    equal_time="counts",
                ),
            ),
        ),
        layers=(Layer("matching"), Layer("lead_times"), Layer("handling")),
        views=(
            View(
                "Finance",
                constraint_weights={"match": 0.2, "three_way": 0.3, "gr_to_invoice": 0.2, "settled": 0.1, "paid_late": 0.2},
            ),
        ),
        name="invoice review, natively",
        scoring_mode="flat",
    )


def comparison_case_norm() -> Norm:
    """The nearest the case model can come to the same five obligations.

    Each one is the constraint an analyst would reach for after importing the
    log and flattening it onto invoices. Nothing here is a straw man: they are
    the paper's own constraint types, correctly parameterised.
    """
    return Norm(
        constraints=(
            NormConstraint("match", "matching", wise.Presence(PO, m=1)),
            NormConstraint("three_way", "matching", wise.Balance("amount", INV, "amount", GR, tau=0.05, width=0.20)),
            NormConstraint("gr_to_invoice", "lead_times", wise.Lag(GR, INV, delta=5, width=10, missing_a="skip")),
            NormConstraint("settled", "handling", wise.Presence(PAY, m=1)),
            NormConstraint("paid_late", "lead_times", wise.Lag(INV, PAY, delta=10, width=20)),
        ),
        layers=(Layer("matching"), Layer("lead_times"), Layer("handling")),
        views=(
            View(
                "Finance",
                constraint_weights={"match": 0.2, "three_way": 0.3, "gr_to_invoice": 0.2, "settled": 0.1, "paid_late": 0.2},
            ),
        ),
        name="invoice review, projected",
        scoring_mode="flat",
    )
