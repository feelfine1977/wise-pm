"""What a draft may and may not say about numbers and examples (item N02).

Acceptance row A05: a vague policy with no threshold produces an unresolved
question, never an invented numeric requirement. Plus the two properties the
roadmap asks of N02 — a draft is checked against cases it did not choose, and
a stakeholder comparison shows the trade-off instead of resolving it.

Provenance is the thread running through the file: policy-derived,
expert-chosen, empirically measured and merely proposed are four different
claims, and the draft keeps them apart.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from _llm_fixtures import document_library

import wise
from wise.llm.drafts import (
    NormDraft,
    ProposedExample,
    Provenance,
    Severity,
    check_threshold_support,
    clarification_questions,
    preview_norm_draft,
    stakeholder_tradeoffs,
)
from wise.llm.policy import AccessPolicy, Principal, Scope
from wise.llm.retrieval import RetrievedSpan

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "norm_drafts"

#: Any stated time quantity. A clarification question may carry a locator with
#: digits in it; what it must never carry is a limit nobody agreed to.
A_TIME_LIMIT = re.compile(r"\d+\s*(day|days|hour|hours|week|weeks|month|months)\b", re.IGNORECASE)


@pytest.fixture
def approved():
    return wise.running_p2p_norm()


@pytest.fixture
def log():
    return wise.running_p2p_log()


def reader():
    return AccessPolicy(Principal("reader"), scopes=frozenset({Scope.POLICY_DOCUMENTS}), document_tags=frozenset({"public"}))


def lag_candidate(norm, delta):
    payload = norm.to_dict()
    for entry in payload["constraints"]:
        if entry["type"] == "lag":
            entry["params"]["delta"] = delta
            entry["id"] = entry["id"]
            break
    return payload


def vague_span():
    return RetrievedSpan(
        "pol-vague",
        "pol-vague#s1",
        0,
        48,
        "Goods receipts are to be recorded promptly after delivery.",
        "digest",
        topic="goods_receipt",
    )


def stating_span(text="Invoices are cleared within 30 days of receipt."):
    return RetrievedSpan("pol-clearing-a", "pol-clearing-a#s1", 0, len(text), text, "digest", topic="invoice_clearing")


# --------------------------------------------- A05 vagueness becomes a question
def test_vague_language_produces_a_question_and_no_number():
    questions = clarification_questions([vague_span()])
    assert len(questions) == 1
    assert "'promptly'" in questions[0]
    assert "No default is assumed and none is proposed" in questions[0]
    assert not A_TIME_LIMIT.search(questions[0]), "the question must not propose a limit of its own"


def test_a_precise_source_produces_no_clarification_question():
    assert clarification_questions([stating_span()]) == ()


@pytest.mark.parametrize("term", ["timely", "as soon as possible", "without undue delay", "immediately", "regularly"])
def test_every_declared_vague_term_is_recognised(term):
    span = RetrievedSpan("d", "d#s1", 0, 10, f"The step is performed {term}.", "digest")
    assert clarification_questions([span])


def test_a_threshold_no_source_states_is_an_unresolved_proposal(approved):
    candidate = lag_candidate(approved, 21.0)
    findings, questions = check_threshold_support(candidate, [vague_span()])
    unsupported = [f for f in findings if f.code == "unsupported_threshold"]
    assert unsupported, "a number nobody stated must be marked"
    assert all(f.provenance is Provenance.UNRESOLVED_PROPOSAL for f in unsupported)
    assert all(f.severity is Severity.QUESTION for f in unsupported)
    assert any("21" in q for q in questions)
    assert any("on what basis" in q for q in questions)


def test_a_threshold_a_source_states_is_policy_derived(approved):
    candidate = lag_candidate(approved, 30.0)
    findings, _ = check_threshold_support(candidate, [stating_span()])
    cited = [f for f in findings if f.code == "threshold_cited" and "delta=30" in f.message]
    assert cited and cited[0].provenance is Provenance.POLICY_DERIVED


def test_the_preview_of_a_vague_source_carries_the_question_not_a_limit(approved, log):
    draft = NormDraft(
        parent_norm_hash=approved.fingerprint(),
        candidate_norm=None,
        source_refs=("pol-vague#s1",),
    )
    preview = preview_norm_draft(draft, approved, log, spans=[vague_span()])
    assert preview.candidate_fingerprint is None
    assert any("promptly" in question for question in preview.questions)
    assert preview.result_diff == {}


def test_the_library_proposes_no_universal_time_limit(approved, log):
    """There is no default anywhere: an empty draft yields questions, not numbers."""
    draft = NormDraft(parent_norm_hash=approved.fingerprint(), candidate_norm=None)
    preview = preview_norm_draft(draft, approved, log, spans=[vague_span()])
    assert preview.questions, "a vague source must raise something"
    assert not any(A_TIME_LIMIT.search(question) for question in preview.questions)


def test_a_contradiction_in_the_sources_reaches_the_reviewer_not_a_choice():
    found = document_library().search("invoices cleared within days of receipt", policy=reader())
    assert found.contradictory
    readings = next(q for q in found.questions if q.kind == "contradiction").question
    assert "30 days" in readings and "14 days" in readings
    assert "does not choose between them" in readings


# ------------------------------------------------- N02 unseen-case behaviour
def test_a_proposed_example_without_a_named_unit_stays_unverified(approved, log):
    draft = NormDraft(
        parent_norm_hash=approved.fingerprint(),
        candidate_norm=lag_candidate(approved, 6.0),
        proposed_examples=(ProposedExample("EX-001", "An invoice with no clearing yet.", "unknown"),),
    )
    preview = preview_norm_draft(draft, approved, log)
    outcome = preview.example_outcomes[0]
    assert outcome["status"] == "unverified"
    assert outcome["observed"] is None
    assert "no unit was named" in outcome["note"]


def test_a_proposed_example_is_checked_against_the_case_it_names(approved, log):
    unit = str(log.case_ids[0])
    draft = NormDraft(
        parent_norm_hash=approved.fingerprint(),
        candidate_norm=lag_candidate(approved, 6.0),
        proposed_examples=(
            ProposedExample("EX-A", "The first case of the running example.", "violated"),
            ProposedExample("EX-B", "The same case, expected to be satisfied.", "satisfied"),
        ),
    )
    preview = preview_norm_draft(draft, approved, log, example_units={"EX-A": unit, "EX-B": unit})
    observed = {row["example_id"]: row for row in preview.example_outcomes}
    assert observed["EX-A"]["observed"] == observed["EX-B"]["observed"]
    assert {observed["EX-A"]["status"], observed["EX-B"]["status"]} == {"matches", "differs"}


def test_an_example_naming_a_unit_that_does_not_exist_says_so(approved, log):
    draft = NormDraft(
        parent_norm_hash=approved.fingerprint(),
        candidate_norm=lag_candidate(approved, 6.0),
        proposed_examples=(ProposedExample("EX-X", "A case from another log.", "violated"),),
    )
    preview = preview_norm_draft(draft, approved, log, example_units={"EX-X": "not-a-case"})
    assert preview.example_outcomes[0]["status"] == "unknown_unit"


def test_the_example_fixture_round_trips_and_previews(approved, log):
    payload = json.loads((FIXTURES / "clearing_draft.json").read_text(encoding="utf-8"))
    payload["parent_norm_hash"] = approved.fingerprint()
    draft = NormDraft.from_dict(payload)
    preview = preview_norm_draft(draft, approved, log, spans=[stating_span()])
    assert preview.candidate_fingerprint is None, "the fixture proposes questions, not a configuration"
    assert draft.source_refs and draft.unresolved_questions


# ------------------------------------------------------- N02 trade-offs
def test_the_trade_off_is_stated_for_every_view_and_resolved_for_none(approved, log):
    draft = NormDraft(parent_norm_hash=approved.fingerprint(), candidate_norm=lag_candidate(approved, 20.0))
    preview = preview_norm_draft(draft, approved, log)
    lines = stakeholder_tradeoffs(preview)
    assert len(lines) == len(approved.view_names)
    assert all("Acceptance by a model is not approval" in line for line in lines)
    assert any("higher" in line or "lower" in line for line in lines)


def test_a_dropped_view_is_reported_as_a_stakeholder_losing_their_perspective(approved, log):
    payload = approved.to_dict()
    payload["views"] = payload["views"][:1]
    draft = NormDraft(parent_norm_hash=approved.fingerprint(), candidate_norm=payload)
    preview = preview_norm_draft(draft, approved, log)
    lines = stakeholder_tradeoffs(preview)
    assert any("loses their perspective entirely" in line for line in lines)


def test_a_parameter_carried_over_unchanged_is_not_re_opened(approved):
    """Only what a draft actually proposes is a proposal."""
    without_parent, _ = check_threshold_support(lag_candidate(approved, 21.0), [stating_span()])
    with_parent, questions = check_threshold_support(lag_candidate(approved, 21.0), [stating_span()], parent=approved)
    unsupported_all = [f for f in without_parent if f.code == "unsupported_threshold"]
    unsupported_changed = [f for f in with_parent if f.code == "unsupported_threshold"]
    assert len(unsupported_all) > len(unsupported_changed) > 0
    assert len(unsupported_changed) == 1 and "delta=21" in unsupported_changed[0].message
    assert len(questions) == 1
    carried = next(f for f in with_parent if f.code == "threshold_carried_over")
    assert carried.provenance is Provenance.EXPERT_CHOICE
    assert "their authority is whatever approved that norm" in carried.message


def test_the_preview_only_questions_what_the_draft_changed(approved, log):
    draft = NormDraft(parent_norm_hash=approved.fingerprint(), candidate_norm=lag_candidate(approved, 21.0))
    preview = preview_norm_draft(draft, approved, log, spans=[stating_span()])
    assert [f.code for f in preview.findings if f.code == "unsupported_threshold"] == ["unsupported_threshold"]


def test_the_findings_carry_distinct_provenance_kinds(approved):
    candidate = lag_candidate(approved, 30.0)
    findings, _ = check_threshold_support(candidate, [stating_span()])
    kinds = {f.provenance for f in findings}
    assert Provenance.POLICY_DERIVED in kinds
    assert set(Provenance) >= kinds
    assert len(set(Provenance)) == 4
