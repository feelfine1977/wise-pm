"""A bounded, read-only assisted review — with a fake model, and then without one.

Run it from the repository root::

    python examples/local_review.py

It uses the bundled purchase-to-pay example and a scripted fake provider, so
it needs no data, no optional dependency, no server and no model. Nothing in
this file opens a socket.

What it shows, in order:

1. the deterministic report, which exists before any model is asked anything;
2. a restricted principal getting a *different, honest* comparator instead of
   a company-wide one they are not entitled to;
3. a retrieved policy library where two approved sources disagree, so the
   review gains a question rather than a chosen answer;
4. an assisted review: the model orders facts, asks questions and offers an
   unverified hypothesis, and the numbers stay the packet's;
5. the same review with the server absent — the report is unchanged.
"""

from __future__ import annotations

import json

import wise
from wise.llm import (
    AccessPolicy,
    ApprovedDocument,
    EvidenceHost,
    FakeProvider,
    LexicalIndex,
    LocalAssistant,
    Principal,
    Scope,
    ToolGateway,
    authorised_baseline,
)
from wise.llm.policy import FULL_POPULATION_AGGREGATE

log = wise.datasets.running_p2p_log()
norm = wise.datasets.running_p2p_norm()

# 1. The deterministic path, complete on its own -----------------------------
result = wise.score(log, norm, evidence="full")
packet = wise.explain_priority(result, "company", "B", view="Finance", gamma=1.0, evidence=result.evidence)
print("=" * 100)
print("1. The deterministic report. No model has been asked anything yet.")
print("=" * 100)
print(wise.render_explanation(packet)[:900], "...\n")

# 2. Two principals, and a comparator that cannot leak -----------------------
run_id = result.manifest.run_id
broad = AccessPolicy(
    Principal("head-of-finance", roles=("reviewer",), authenticated_by="example"),
    scopes=frozenset(s.value for s in Scope),
    runs=frozenset({run_id}),
    views=frozenset(result.views),
    columns=frozenset(result.cases.columns),
    document_tags=frozenset({"public", "finance"}),
    unrestricted_rows=True,
    label="head of finance",
)
restricted = AccessPolicy(
    Principal("company-a-analyst", roles=("analyst",), authenticated_by="example"),
    scopes=frozenset(s.value for s in Scope),
    runs=frozenset({run_id}),
    views=frozenset({"Finance"}),
    columns=frozenset({"company", "flow_type"}),
    row_filters={"company": frozenset({"A"})},
    document_tags=frozenset({"public"}),
    label="analyst restricted to company A",
)

company_wide = wise.explain.BaselineSpec.from_result(result, "Finance", baseline_id=FULL_POPULATION_AGGREGATE)
spec, change = authorised_baseline(result, restricted, view="Finance", requested=company_wide)
print("=" * 100)
print("2. The restricted analyst asked for the company-wide comparator.")
print("=" * 100)
print(f"   requested  {company_wide.baseline_id}  reference {company_wide.reference_score:.6f}")
print(f"   effective  {spec.baseline_id}  reference {spec.reference_score:.6f}")
print(f"   {change.message()}\n")

# 3. Approved sources that disagree ------------------------------------------
library = LexicalIndex(
    [
        ApprovedDocument(
            "pol-clearing-a",
            "Invoice clearing policy (Finance)",
            "Invoices from approved vendors are cleared within 30 days of receipt.",
            access_tags=("public",),
            topic="invoice_clearing",
            version="3",
        ),
        ApprovedDocument(
            "pol-clearing-b",
            "Invoice clearing addendum (Treasury)",
            "Invoices from approved vendors are cleared within 14 days of receipt.",
            access_tags=("public",),
            topic="invoice_clearing",
            version="1",
        ),
        ApprovedDocument(
            "pol-restricted",
            "Vendor remuneration schedule",
            "Vendor V0 is paid a confidential retainer.",
            access_tags=("finance", "restricted"),
            topic="vendor_terms",
        ),
    ]
)
found = library.search("invoices cleared within days of receipt", policy=restricted, limit=4)
print("=" * 100)
print("3. Retrieval, filtered before it is ranked.")
print("=" * 100)
for span in found.spans:
    print(f"   {span.citation()}  {span.text}")
print(f"   excluded before ranking: {found.excluded}")
for question in found.questions:
    print(f"   [{question.kind}] {question.question}")
print()

# 4. An assisted review, against a fake provider -----------------------------
selection = json.dumps(
    {
        "schema_version": "0.1-proposal",
        "calls": [
            {"tool": "get_run_summary", "arguments": {}},
            {"tool": "compare_groups", "arguments": {"by": ["company"], "view": "Finance"}},
            {"tool": "delete_everything", "arguments": {"path": "/etc/passwd"}},
        ],
    }
)
draft = json.dumps(
    {
        "schema_version": "0.1-proposal",
        "packet_id": packet.explanation_id,
        "fact_order": [f.fact_id for f in packet.facts[:4]],
        "hypotheses": [
            {
                "text": "The gap may follow from goods receipts recorded after the invoice.",
                "evidence_refs": [packet.evidence_refs[0]],
                "status": "unverified_hypothesis",
            }
        ],
        "questions": ["Which clearing limit applies to these vendors, 30 days or 14?"],
        "limitations": [],
    }
)
gateway = ToolGateway(EvidenceHost(result, packet=packet, evidence=result.evidence, index=library), broad)
review = LocalAssistant(FakeProvider([selection, draft]), gateway).review(packet, retrieval=found)
print("=" * 100)
print("4. An assisted review. The model asked for three tools; one does not exist.")
print("=" * 100)
for outcome in review.tool_results:
    print(f"   {outcome.tool:<24} {'ok' if outcome.ok else 'refused — ' + outcome.reason}")
print(f"   status: {review.status}; references check: {review.reference_check.ok}")
print()
print(review.report[review.report.index("Suggested reading order") :])

# 5. The same review with nothing answering ----------------------------------
silent = LocalAssistant(FakeProvider([]), ToolGateway(EvidenceHost(result, packet=packet), broad)).review(packet)
print("=" * 100)
print("5. The same review with no server. Note what changed, and what did not.")
print("=" * 100)
print(f"   status:  {silent.status}")
print(f"   reasons: {list(silent.reasons)}")
print(f"   the report is byte-identical to the deterministic one: {silent.report == wise.render_explanation(packet)}")
