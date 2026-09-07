"""Review a proposed norm change: validate it, preview it, and apply nothing.

Run it from the repository root::

    python examples/review_norm_draft.py

It uses the bundled purchase-to-pay example. No model, no server and no
optional dependency; the "proposals" below are written out so the checks are
the point rather than the source of the text.

What it shows, in order:

1. what the catalogue actually supports, read off the implementation;
2. a hostile draft — an ``eval`` recipe — refused before anything is computed;
3. a stale parent fingerprint reported as a conflict;
4. a well-formed draft previewed on an isolated copy: the diff, the trade-off,
   and the proof that the approved norm and the caller's log are untouched;
5. a threshold no source states, kept as an unresolved proposal.
"""

from __future__ import annotations

import wise
from wise.errors import DraftConflict, UnsafeDraft
from wise.llm.drafts import (
    NormDraft,
    ProposedExample,
    check_threshold_support,
    clarification_questions,
    preview_norm_draft,
    stakeholder_tradeoffs,
)
from wise.llm.retrieval import RetrievedSpan
from wise.schema import catalogue_digest, constraint_catalogue, recipe_catalogue

log = wise.datasets.running_p2p_log()
approved = wise.datasets.running_p2p_norm()

# 1. What exists ------------------------------------------------------------
print("=" * 100)
print("1. The supported vocabulary, derived from the code rather than a second list.")
print("=" * 100)
for name, spec in sorted(constraint_catalogue().items()):
    bounded = [f"{p.name} {p.bound.describe()}" for p in spec.parameters if p.bound is not None]
    print(f"   {name:<12} {', '.join(spec.parameter_names)}")
    if bounded:
        print(f"   {'':<12} ranges: {'; '.join(bounded)}")
unsafe = [k for k, spec in recipe_catalogue().items() if spec.evaluates_expression]
print(f"\n   recipe kinds that evaluate an expression, and so are refused in the untrusted path: {unsafe}")
print(f"   catalogue digest: {catalogue_digest()[:16]}…\n")


def candidate_with_recipe(recipe):
    payload = approved.to_dict()
    payload["derived_attributes"] = [recipe]
    return payload


def candidate_with_lag(delta):
    payload = approved.to_dict()
    for entry in payload["constraints"]:
        if entry["type"] == "lag":
            entry["params"]["delta"] = delta
            break
    return payload


# 2. A hostile draft --------------------------------------------------------
print("=" * 100)
print("2. A draft that would evaluate an expression.")
print("=" * 100)
hostile = NormDraft(
    parent_norm_hash=approved.fingerprint(),
    candidate_norm=candidate_with_recipe({"name": "sneaky", "kind": "eval", "expr": "n_events * 2"}),
)
try:
    preview_norm_draft(hostile, approved, log)
except UnsafeDraft as exc:
    print(f"   refused: {exc}")
print("   Nothing was derived, scored or loaded: the refusal precedes all three.\n")

# 3. A stale parent ---------------------------------------------------------
print("=" * 100)
print("3. A draft written against a different norm.")
print("=" * 100)
stale = NormDraft(parent_norm_hash="0" * 64, candidate_norm=candidate_with_lag(6.0))
try:
    preview_norm_draft(stale, approved, log)
except DraftConflict as exc:
    print(f"   conflict: {exc}\n")

# 4. A well-formed draft ----------------------------------------------------
print("=" * 100)
print("4. A well-formed draft, previewed on an isolated copy.")
print("=" * 100)
before_fingerprint = approved.fingerprint()
before_columns = list(log.cases.columns)

draft = NormDraft(
    parent_norm_hash=before_fingerprint,
    candidate_norm=candidate_with_lag(20.0),
    source_refs=("pol-clearing-a#s1",),
    assumptions=("the clearing clock starts at the recorded invoice receipt",),
    proposed_examples=(ProposedExample("EX-001", "The first case of the running example.", "violated"),),
    draft_id="D-2026-001",
)
preview = preview_norm_draft(draft, approved, log, example_units={"EX-001": str(log.case_ids[0])})

print(f"   findings: {[f.code for f in preview.findings] or 'none'}")
print(f"   candidate fingerprint: {preview.candidate_fingerprint[:16]}…")
print(f"   changed constraints: {sorted(preview.assumption_diff['constraints_changed'])}")
for view, row in sorted(preview.result_diff["per_view"].items()):
    print(
        f"   {view:<10} mean {row['mean_score_approved']:.6f} → {row['mean_score_candidate']:.6f}"
        f"   units changed: {row['n_units_changed']}/{row['n_scored_candidate']}"
    )
print("\n   Stakeholder trade-off:")
for line in stakeholder_tradeoffs(preview):
    print(f"     - {line}")
print("\n   Proposed examples, checked against the case each names:")
for outcome in preview.example_outcomes:
    print(
        f"     - {outcome['example_id']}: expected {outcome['expected_result']}, observed {outcome['observed']} ({outcome['status']})"
    )
print(f"\n   applied={preview.applied}  activated={preview.activated}  review_status={draft.review_status!r}")
print(f"   the approved norm's fingerprint is unchanged: {approved.fingerprint() == before_fingerprint}")
print(f"   the caller's log kept its columns:            {list(log.cases.columns) == before_columns}\n")

# 5. A number nobody stated -------------------------------------------------
print("=" * 100)
print("5. A threshold no cited source states, and language that states none.")
print("=" * 100)
cited = RetrievedSpan(
    "pol-clearing-a",
    "pol-clearing-a#s1",
    0,
    69,
    "Invoices from approved vendors are cleared within 30 days of receipt.",
    "digest",
    topic="invoice_clearing",
)
vague = RetrievedSpan(
    "pol-vague",
    "pol-vague#s1",
    0,
    57,
    "Goods receipts are to be recorded promptly after delivery.",
    "digest",
    topic="goods_receipt",
)
findings, questions = check_threshold_support(candidate_with_lag(21.0), [cited, vague], parent=approved)
for finding in findings:
    if finding.code == "unsupported_threshold":
        print(f"   [{finding.provenance.value}] {finding.message}")
for question in (*questions, *clarification_questions([vague])):
    print(f"   ? {question}")
print("\n   No universal limit is proposed anywhere above. That is the point.")

# The same candidate with the number a source does state:
supported, _ = check_threshold_support(candidate_with_lag(30.0), [cited], parent=approved)
for finding in supported:
    if finding.code == "threshold_cited":
        print(f"   [{finding.provenance.value}] {finding.message}")

print("\n   Draft envelope, exactly as the roadmap contract defines it:")
print(f"   keys = {sorted(draft.to_contract_dict())}")
