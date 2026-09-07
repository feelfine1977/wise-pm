"""Approved-source retrieval, filtered before it is scored (item L02).

Acceptance row L03 for documents: permission, approval state and effective
dates decide what exists for a query *before* anything is ranked. Plus the
properties the roadmap asks of a retrieved span — document identity, locator,
digest, version, effective dates and access tags all survive — and the rule
that a contradiction between approved sources becomes a question rather than a
silent choice.
"""

from __future__ import annotations

from datetime import date

import pytest
from _llm_fixtures import document_library

from wise.errors import AccessDenied, BudgetExceeded, LLMError
from wise.llm.policy import AccessPolicy, Principal, Scope
from wise.llm.provider import BudgetLedger, CallBudget
from wise.llm.retrieval import (
    ApprovedDocument,
    LexicalIndex,
    RetrievedSpan,
    review_questions,
)


def reader(tags=("public",), scopes=(Scope.POLICY_DOCUMENTS,)) -> AccessPolicy:
    return AccessPolicy(Principal("reader"), scopes=frozenset(s.value for s in scopes), document_tags=frozenset(tags))


# ------------------------------------------------------- permission is first
def test_retrieval_needs_its_own_scope():
    with pytest.raises(AccessDenied, match="policy_documents"):
        document_library().search("clearing", policy=AccessPolicy(Principal("p")))


def test_a_document_whose_tags_are_not_granted_is_never_ranked():
    found = document_library().search("vendor retainer confidential", policy=reader())
    assert "pol-restricted" not in found.document_ids
    assert found.excluded["access_tags"] >= 1
    assert all("90 000" not in span.text for span in found.spans)


def test_the_restricted_document_is_reachable_only_with_its_tag():
    policy = reader(tags=("public", "finance", "restricted"))
    found = document_library().search("vendor retainer", policy=policy)
    assert "pol-restricted" in found.document_ids


def test_exact_lookup_is_not_a_way_round_the_tags():
    index = document_library()
    with pytest.raises(AccessDenied):
        index.get("pol-restricted", policy=reader())
    assert index.get("pol-clearing-a", policy=reader()).document_id == "pol-clearing-a"


def test_a_draft_document_is_not_retrievable():
    found = document_library().search("cleared within 7 days", policy=reader())
    assert "pol-draft" not in found.document_ids
    assert found.excluded["approval_state"] >= 1


def test_a_document_outside_its_effective_dates_is_not_retrievable():
    found = document_library().search("cleared within 60 days", policy=reader(), at=date(2025, 6, 1))
    assert "pol-expired" not in found.document_ids
    assert found.excluded["effective_dates"] >= 1


def test_the_same_document_is_retrievable_while_it_was_in_force():
    found = document_library().search("cleared 60 days", policy=reader(), at=date(2022, 6, 1))
    assert "pol-expired" in found.document_ids
    assert "pol-clearing-a" not in found.document_ids, "a 2024 policy was not in force in 2022"


# ------------------------------------------------------------ identity kept
def test_every_span_carries_its_locator_digest_version_and_tags():
    found = document_library().search("cleared within days", policy=reader())
    span = next(s for s in found.spans if s.document_id == "pol-clearing-a")
    assert span.version == "3"
    assert span.access_tags == ("public",)
    assert span.effective_from == "2024-01-01"
    assert len(span.digest) == 64
    assert span.start >= 0 and span.end > span.start
    assert span.citation().startswith("pol-clearing-a v3 [")


def test_a_changed_document_gets_a_different_digest():
    a = ApprovedDocument("d", "t", "one", access_tags=("public",))
    b = ApprovedDocument("d", "t", "two", access_tags=("public",))
    assert a.digest != b.digest


def test_the_result_serialises_with_its_exclusions_and_policy_fingerprint():
    found = document_library().search("invoices cleared", policy=reader())
    payload = found.to_dict()
    assert payload["excluded"] and payload["policy_fingerprint"]
    assert payload["spans"][0]["document_id"]


# ----------------------------------------------- contradictions and vagueness
def test_two_approved_sources_that_disagree_produce_a_question_not_a_choice():
    found = document_library().search("invoices cleared within days of receipt", policy=reader())
    assert {"pol-clearing-a", "pol-clearing-b"} <= set(found.document_ids)
    contradiction = next(q for q in found.questions if q.kind == "contradiction")
    assert "30 days" in contradiction.question and "14 days" in contradiction.question
    assert set(contradiction.document_ids) == {"pol-clearing-a", "pol-clearing-b"}
    assert found.contradictory


def test_vague_language_with_no_quantity_becomes_a_question_and_no_number():
    found = document_library().search("goods receipts recorded promptly", policy=reader())
    question = next(q for q in found.questions if q.kind == "vague_requirement")
    assert "promptly" in question.question
    assert "No default is assumed" in question.question
    assert not any(char.isdigit() for char in question.question)


def test_one_consistent_source_raises_no_contradiction():
    index = LexicalIndex([ApprovedDocument("only", "t", "Cleared within 30 days.", access_tags=("public",), topic="clearing")])
    assert index.search("cleared days", policy=reader()).questions == ()


def test_a_contradiction_is_detected_on_the_spans_themselves():
    a = RetrievedSpan("p1", "p1#s1", 0, 9, "within 30 days", "d1", topic="clearing")
    b = RetrievedSpan("p2", "p2#s1", 0, 9, "within 30 days", "d2", topic="clearing")
    assert review_questions([a, b]) == (), "agreeing sources are not a contradiction"


# ------------------------------------------------- the event log is not prose
@pytest.mark.parametrize("kind", ["event_log", "trace", "cases", "anything_else"])
def test_an_event_log_cannot_be_indexed_as_a_document(kind):
    with pytest.raises(LLMError, match="not a retrievable source"):
        ApprovedDocument("log", "The log", "case c1: A then B", source_kind=kind)


def test_a_duplicate_document_identity_is_refused():
    index = LexicalIndex([ApprovedDocument("d", "t", "x", access_tags=("public",))])
    with pytest.raises(LLMError, match="one identity"):
        index.add(ApprovedDocument("d", "t", "y", access_tags=("public",)))


# --------------------------------------------------------------- budgets
def test_the_retrieval_budget_is_bounded():
    ledger = BudgetLedger(CallBudget(max_retrieved_documents=1))
    index = document_library()
    index.search("invoices cleared within days", policy=reader(), limit=1, ledger=ledger)
    with pytest.raises(BudgetExceeded, match="retrieval budget"):
        index.search("invoices cleared within days", policy=reader(), limit=1, ledger=ledger)


def test_the_retrieval_byte_budget_is_bounded():
    ledger = BudgetLedger(CallBudget(max_retrieved_bytes=8))
    with pytest.raises(BudgetExceeded, match="byte budget"):
        document_library().search("invoices cleared", policy=reader(), ledger=ledger)


def test_the_limit_must_be_positive():
    with pytest.raises(LLMError):
        document_library().search("x", policy=reader(), limit=0)


def test_ranking_is_deterministic_and_bounded():
    index = document_library()
    first = index.search("invoices cleared within days of receipt", policy=reader(), limit=3)
    second = index.search("invoices cleared within days of receipt", policy=reader(), limit=3)
    assert [s.span_id for s in first.spans] == [s.span_id for s in second.spans]
    assert len(first.spans) <= 3


def test_a_query_with_no_matching_term_returns_nothing_rather_than_everything():
    assert document_library().search("zzzz-nothing-matches", policy=reader()).spans == ()
