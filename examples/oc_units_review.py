"""Object-centric review: read an object log, bound a context, read time correctly.

Run it from anywhere::

    python examples/oc_units_review.py

It builds a small purchase-to-pay object log in code, so it needs no data file
and no optional dependency. Pass a directory to also write the OCEL 2.0 export::

    python examples/oc_units_review.py /tmp/wise-oc

Covers stages O01 (the log and its adapters) and O02 (typed bounded units).
The relational checks, the quantity accounting and the projection comparison
are later stages and are deliberately absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from wise import oc
from wise.errors import OCUnitError, OCValidationError


def t(text: str) -> pd.Timestamp:
    return pd.Timestamp(text)


# 1. A small object log. One order, two invoices, two partial goods receipts,
#    three items and one payment that settles both invoices — the shape a case
#    log has to pick one identity out of.
events = [
    oc.OCEvent("e1", "Create Purchase Order", t("2024-01-01T09:00:00Z"), {"resource": "buyer-1"}),
    oc.OCEvent("e2", "Record Goods Receipt", t("2024-01-03T08:00:00Z"), {"resource": "warehouse"}),
    oc.OCEvent("e3", "Record Goods Receipt", t("2024-01-04T08:00:00Z"), {"resource": "warehouse"}),
    # two invoices recorded in the same minute: same activity, same timestamp,
    # two different events. Nothing here collapses them.
    oc.OCEvent("e4", "Record Invoice Receipt", t("2024-01-05T10:00:00Z"), {"resource": "finance"}),
    oc.OCEvent("e5", "Record Invoice Receipt", t("2024-01-05T10:00:00Z"), {"resource": "finance"}),
    oc.OCEvent("e6", "Execute Payment", t("2024-02-01T12:00:00Z"), {"resource": "treasury"}),
]
objects = [
    oc.OCObject("po1", "purchase_order"),
    oc.OCObject("inv1", "invoice"),
    oc.OCObject("inv2", "invoice"),
    oc.OCObject("gr1", "goods_receipt"),
    oc.OCObject("gr2", "goods_receipt"),
    oc.OCObject("it1", "item"),
    oc.OCObject("it2", "item"),
    oc.OCObject("it3", "item"),
    oc.OCObject("pay1", "payment"),
]
e2o = [
    oc.E2O("e1", "po1", "purchase_order"),
    oc.E2O("e1", "it1", "item"),
    oc.E2O("e1", "it2", "item"),
    oc.E2O("e1", "it3", "item"),
    oc.E2O("e2", "gr1", "goods_receipt"),
    oc.E2O("e2", "po1", "purchase_order"),
    oc.E2O("e3", "gr2", "goods_receipt"),
    oc.E2O("e3", "po1", "purchase_order"),
    oc.E2O("e4", "inv1", "invoice"),
    oc.E2O("e5", "inv2", "invoice"),
    oc.E2O("e6", "pay1", "payment"),
    oc.E2O("e6", "inv1", "invoice"),
    oc.E2O("e6", "inv2", "invoice"),
    # the same row twice, as an extract that joined one table too many
    oc.E2O("e6", "inv2", "invoice"),
]
o2o = [
    oc.O2O("inv1", "po1", "belongs to"),
    oc.O2O("inv2", "po1", "belongs to"),
    oc.O2O("po1", "it1", "has item"),
    oc.O2O("po1", "it2", "has item"),
    oc.O2O("po1", "it3", "has item"),
    oc.O2O("gr1", "po1", "receipt for"),
    oc.O2O("gr2", "po1", "receipt for"),
    oc.O2O("pay1", "inv1", "settles"),
    oc.O2O("pay1", "inv2", "settles"),
]
history = [
    oc.AttributeChange("po1", "vendor", t("2024-01-01T09:00:00Z"), "V1"),
    oc.AttributeChange("inv1", "amount", t("2024-01-05T10:00:00Z"), 100.0),
    oc.AttributeChange("inv1", "amount", t("2024-01-20T10:00:00Z"), 120.0),
    oc.AttributeChange("inv1", "owner", t("2024-01-05T10:00:00Z"), "team-a"),
    oc.AttributeChange("inv1", "owner", t("2024-01-25T10:00:00Z"), "team-b"),
    oc.AttributeChange("inv2", "amount", t("2024-01-05T10:00:00Z"), 60.0),
    oc.AttributeChange("pay1", "amount", t("2024-02-01T12:00:00Z"), 160.0),
]

log = oc.OCEventLog.build(events=events, objects=objects, e2o=e2o, o2o=o2o, attribute_history=history)

print("1. The log, and what building it repaired")
print(f"   {log}")
for issue in log.validation.issues:
    print(f"   {issue.severity.value:9} {issue.code.value:24} x{issue.count}  {issue.message}")
print(f"   observed precision  {log.source.precision}")
print(f"   relation time       {log.source.relation_time_semantics}")
print()

print("2. Two events, one activity, one timestamp — and two identities")
for event in log.events_of("po1", activity="Record Goods Receipt"):
    print(f"   {event.event_id}  {event.activity}  {event.timestamp.isoformat()}")
for event in (log.event("e4"), log.event("e5")):
    print(f"   {event.event_id}  {event.activity}  {event.timestamp.isoformat()}")
print()

print("3. A changing attribute, read at a declared time")
for moment in ("2024-01-01T00:00:00Z", "2024-01-10T00:00:00Z", "2024-02-10T00:00:00Z"):
    reading = log.value_at("inv1", "owner", moment)
    found = f"{reading.value!r} (effective {reading.effective_from.isoformat()})" if reading.found else "no value in force"
    print(f"   at {moment}: {found}")
print(f"   ...and never the first non-null value: {log.value_at('inv1', 'owner', '2024-01-01T00:00:00Z').found=}")
print()

print("4. A bounded, typed unit")
spec = oc.UnitSpec(
    unit_type="invoice_review",
    anchor_type="invoice",
    roles=(
        oc.RolePath("order", (oc.PathStep("belongs to", target_type="purchase_order"),)),
        oc.RolePath(
            "items",
            (
                oc.PathStep("belongs to", target_type="purchase_order"),
                oc.PathStep("has item", target_type="item"),
            ),
        ),
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
units = oc.build_units(log, spec, at="2024-03-01T00:00:00Z")
unit = units[0]
print(f"   {unit}")
for role, bound in unit.bindings.items():
    print(f"   {role:10} {list(bound)}")
print(f"   events in scope    {list(unit.event_ids)}")
print(f"   amount at 2024-03  {unit.attribute(log, 'amount').value}")
print("   why it binds what it binds:")
for witness in unit.witnesses_for("items"):
    print(f"     {witness.describe()}")
for qualification in unit.qualifications:
    print(f"   {qualification.code.value}: {qualification.message}")
print()

print("5. One payment, two invoices — and the payment is not shared away")
print(f"   {[u.unit_id for u in units]}")
print(f"   payment role of each: {[u.role('payment') for u in units]}")
print()

print("6. A limit is a limit, and it is visible")
crowded = oc.OCEventLog.build(
    events=[oc.OCEvent("e1", "Touch", t("2024-01-01T00:00:00Z"))],
    objects=[oc.OCObject("v1", "vendor")] + [oc.OCObject(f"i{i}", "invoice") for i in range(40)],
    e2o=[oc.E2O("e1", "i0", "invoice")],
    o2o=[oc.O2O(f"i{i}", "v1", "supplied by") for i in range(40)],
)
shared = oc.UnitSpec(
    unit_type="invoice_review",
    anchor_type="invoice",
    roles=(
        oc.RolePath(
            "siblings",
            (
                oc.PathStep("supplied by", target_type="vendor"),
                oc.PathStep("supplied by", target_type="invoice", direction="reverse"),
            ),
        ),
    ),
    limits=oc.TraversalLimits(max_fan_out=10),
)
bounded = oc.build_units(crowded, shared, anchors=["i0"])[0]
print(f"   {bounded}")
print(f"   seen {bounded.truncation.seen} kept {bounded.truncation.kept}")
try:
    bounded.require_complete()
except OCUnitError as error:
    print(f"   require_complete() refuses: {error}")
print(f"   context report: {oc.context_report(oc.build_units(crowded, shared))}")
print()

print("7. Interchange: what each format can carry")
for name, declared in oc.interchange_support().items():
    print(f"   {name:16} read={declared['read']} write={declared['write']}")
    for limitation in declared["cannot_carry"]:
        print(f"       cannot carry: {limitation}")
document, report = oc.ocel2_json_document(log)
back = oc.read_ocel2_json(document, name="round-trip")
print(f"   OCEL 2.0 JSON round-trip identical: {back.content_fingerprint() == log.content_fingerprint()}")
tables = oc.to_tables(log)
print(f"   native tables: {[f'{k}{v.shape}' for k, v in tables.items()]}")
print(f"   native round-trip identical:       {oc.from_tables(tables).content_fingerprint() == log.content_fingerprint()}")
print()

print("8. What the log refuses to guess")
try:
    oc.OCEventLog.build(
        events=[oc.OCEvent("e1", "A", t("2024-01-01T00:00:00Z")), oc.OCEvent("e1", "B", t("2024-01-01T00:00:00Z"))],
        objects=[],
    )
except OCValidationError as error:
    print(f"   two rows, one event id: {error}")
try:
    oc.build_units(
        log,
        oc.UnitSpec(
            unit_type="u",
            anchor_type="invoice",
            roles=(oc.RolePath("o", (oc.PathStep("belongs-to", target_type="purchase_order"),)),),
        ),
    )
except OCUnitError as error:
    print(f"   a qualifier that does not occur: {error}")

if len(sys.argv) > 1:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    loss = oc.write_ocel2_json(log, out / "p2p_example.ocel.json", indent=2)
    print(f"\nWrote {out / 'p2p_example.ocel.json'} (lossless={loss.lossless})")
