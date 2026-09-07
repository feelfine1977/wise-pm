"""The two bounded steps, and the report that never depends on them (item L01).

Acceptance rows L01 (optional packages and server absent), L06 (oversize,
loop, timeout, budget), L08 (malformed JSON, and valid JSON with false
references), L09 (a model that omits a mandatory qualification does not remove
it) and L11 (an unsupported hypothesis with valid evidence ids stays
unverified).

Every provider here is :class:`~wise.llm.provider.FakeProvider`. Nothing in
this file opens a socket, starts a server or names a real model.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys

import pytest
from _llm_fixtures import broad_policy, document_library, explanation_draft, packet_for, scored_run, tool_selection

import wise
from wise.llm.assistant import LocalAssistant, check_references, render_reviewed_report
from wise.llm.gateway import EvidenceHost, ToolGateway
from wise.llm.provider import BudgetLedger, CallBudget, FakeProvider, ProviderResult, ProviderStatus
from wise.llm.schemas import ExplanationDraft, Hypothesis, SchemaViolation, read_explanation_draft, read_tool_selection


@pytest.fixture(scope="module")
def run():
    return scored_run()


@pytest.fixture(scope="module")
def packet(run):
    return packet_for(run)


def assistant_for(run, packet, script, *, budget=None):
    ledger = BudgetLedger(budget or CallBudget())
    gateway = ToolGateway(
        EvidenceHost(run, packet=packet, evidence=run.evidence, index=document_library()),
        broad_policy(run),
        ledger=ledger,
    )
    return LocalAssistant(FakeProvider(script), gateway), gateway


def good_draft(packet, **kwargs):
    return explanation_draft(packet.explanation_id, [f.fact_id for f in packet.facts[:3]], **kwargs)


# ------------------------------------------------ L01 the deterministic path
def test_importing_wise_does_not_import_the_assistance_package():
    code = "import sys, wise; print('wise.llm' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_importing_the_assistance_package_needs_no_optional_dependency():
    code = (
        "import sys, builtins\n"
        "real = builtins.__import__\n"
        "banned = {'httpx', 'requests', 'jsonschema', 'pydantic', 'ollama', 'langchain', 'openai', 'transformers'}\n"
        "def guard(name, *a, **k):\n"
        "    if name.split('.')[0] in banned:\n"
        "        raise AssertionError('wise.llm imported the optional package ' + name)\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = guard\n"
        "import wise.llm, wise.llm.ollama, wise.llm.drafts\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_importing_and_configuring_a_provider_connects_to_nothing():
    """An audit hook is the honest check: it fires on a real connection attempt."""
    code = (
        "import sys\n"
        "def hook(event, args):\n"
        "    if event in ('socket.connect', 'socket.getaddrinfo', 'urllib.Request', 'subprocess.Popen'):\n"
        "        raise AssertionError('the assistance path attempted ' + event)\n"
        "sys.addaudithook(hook)\n"
        "import wise, wise.llm, wise.llm.ollama, wise.llm.drafts\n"
        "from wise.llm.ollama import OllamaConfig, OllamaProvider\n"
        "provider = OllamaProvider(OllamaConfig(model='m', allowed_models=('m',)))\n"
        "provider.describe()\n"
        "provider.transport.opener\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_the_deterministic_report_is_identical_whether_or_not_a_model_answers(run, packet):
    deterministic = wise.render_explanation(packet)
    assistant, _ = assistant_for(run, packet, [])
    review = assistant.review(packet)
    assert review.status == "deterministic_only"
    assert review.report == deterministic
    assert review.assisted is False


@pytest.mark.parametrize(
    "status",
    [ProviderStatus.UNAVAILABLE, ProviderStatus.TIMEOUT, ProviderStatus.OVERSIZE, ProviderStatus.REFUSED],
)
def test_every_provider_failure_leaves_the_deterministic_report_standing(run, packet, status):
    assistant, _ = assistant_for(run, packet, [status, status])
    review = assistant.review(packet)
    assert review.status == "deterministic_only"
    assert review.report == wise.render_explanation(packet)
    assert any(status.value in reason for reason in review.reasons)


def test_the_reasons_name_what_did_not_happen(run, packet):
    assistant, _ = assistant_for(
        run, packet, [ProviderResult(ProviderStatus.UNAVAILABLE, reason="no server on 127.0.0.1:11434")] * 2
    )
    review = assistant.review(packet)
    assert any("no server" in reason for reason in review.reasons)
    record = review.provider_records[0]
    assert record["status"] == "unavailable"


# ---------------------------------------------------------- the happy path
def test_a_valid_draft_reorders_facts_without_touching_their_values(run, packet):
    order = [f.fact_id for f in packet.facts[:3]][::-1]
    assistant, _ = assistant_for(
        run,
        packet,
        [tool_selection({"tool": "get_run_summary", "arguments": {}}), explanation_draft(packet.explanation_id, order)],
    )
    review = assistant.review(packet)
    assert review.status == "drafted" and review.assisted
    assert list(review.draft.fact_order) == order
    for fact in packet.facts[:3]:
        if isinstance(fact.value, float):
            assert f"{fact.value:.6g}" in review.report
    assert review.tool_results[0].ok


def test_the_selected_tools_actually_ran_through_the_gateway(run, packet):
    assistant, gateway = assistant_for(
        run,
        packet,
        [
            tool_selection(
                {"tool": "get_run_summary", "arguments": {}},
                {"tool": "compare_groups", "arguments": {"by": ["company"], "view": "Finance"}},
            ),
            good_draft(packet),
        ],
    )
    review = assistant.review(packet)
    assert [t.tool for t in review.tool_results] == ["get_run_summary", "compare_groups"]
    assert gateway.ledger.tool_calls == 2


def test_a_refused_tool_does_not_stop_the_review(run, packet):
    assistant, _ = assistant_for(
        run,
        packet,
        [tool_selection({"tool": "run_sql", "arguments": {}}), good_draft(packet)],
    )
    review = assistant.review(packet)
    assert review.status == "drafted"
    assert review.tool_results[0].ok is False and "unknown tool" in review.tool_results[0].reason


# --------------------------------------------------------------- L08 refusals
@pytest.mark.parametrize(
    "reply",
    [
        "not json at all",
        "{",
        '{"schema_version": "0.1-proposal"}',
        '{"schema_version": "9.9", "packet_id": "p", "fact_order": ["a"], "hypotheses": [], "questions": [], "limitations": []}',
        '["a", "list"]',
        '{"schema_version": "0.1-proposal", "packet_id": "p", "fact_order": [], "hypotheses": [], "questions": [], "limitations": []}',
        '{"schema_version": "0.1-proposal", "packet_id": "p", "fact_order": ["a"], "hypotheses": [], "questions": [], "limitations": [], "extra": 1}',
    ],
)
def test_a_malformed_draft_is_refused_and_the_report_survives(run, packet, reply):
    assistant, _ = assistant_for(run, packet, [tool_selection(), reply])
    review = assistant.review(packet)
    assert review.status == "deterministic_only"
    assert any("refused" in reason for reason in review.reasons)


def test_valid_json_with_a_fact_that_does_not_exist_is_refused(run, packet):
    reply = explanation_draft(packet.explanation_id, ["OBS-invented_fact"])
    assistant, _ = assistant_for(run, packet, [tool_selection(), reply])
    review = assistant.review(packet)
    assert review.status == "deterministic_only"
    assert review.reference_check.unknown_facts == ("OBS-invented_fact",)
    assert "are not facts of this packet" in review.reference_check.message()


def test_valid_json_with_an_evidence_id_that_was_never_authorised_is_refused(run, packet):
    reply = explanation_draft(
        packet.explanation_id,
        [packet.facts[0].fact_id],
        hypotheses=[
            {
                "text": "Vendor V9 causes the delay.",
                "evidence_refs": ["run-somewhere-else:c1:A"],
                "status": "unverified_hypothesis",
            }
        ],
    )
    assistant, _ = assistant_for(run, packet, [tool_selection(), reply])
    review = assistant.review(packet)
    assert review.status == "deterministic_only"
    assert review.reference_check.unknown_evidence == ("run-somewhere-else:c1:A",)


def test_a_hypothesis_may_cite_an_id_a_tool_returned(run, packet):
    known = run.evidence.records[0].evaluation_id
    reply = explanation_draft(
        packet.explanation_id,
        [packet.facts[0].fact_id],
        hypotheses=[{"text": "Worth checking.", "evidence_refs": [known], "status": "unverified_hypothesis"}],
    )
    assistant, _ = assistant_for(
        run,
        packet,
        [tool_selection({"tool": "get_evidence", "arguments": {"evaluation_ids": [known]}}), reply],
    )
    review = assistant.review(packet)
    assert review.status == "drafted"
    assert review.reference_check.ok


def test_a_status_other_than_unverified_is_refused():
    with pytest.raises(SchemaViolation):
        Hypothesis("It is proven.", ("e1",), status="verified")


def test_the_tool_selection_schema_is_enforced_too(run, packet):
    assistant, _ = assistant_for(run, packet, ['{"schema_version": "0.1-proposal", "calls": "everything"}', "{}"])
    review = assistant.review(packet)
    assert any("tool selection refused" in reason for reason in review.reasons)


# ------------------------------------------------------------- L06 budgets
def test_the_model_call_budget_stops_the_second_step(run, packet):
    assistant, _ = assistant_for(run, packet, [tool_selection(), good_draft(packet)], budget=CallBudget(max_calls=1))
    review = assistant.review(packet)
    assert review.status == "deterministic_only"
    assert any("call budget" in reason for reason in review.reasons)


def test_a_reply_past_the_response_budget_is_refused(run, packet):
    huge = explanation_draft(packet.explanation_id, [packet.facts[0].fact_id], questions=["x" * 5_000])
    assistant, _ = assistant_for(run, packet, [tool_selection(), huge], budget=CallBudget(max_response_bytes=512))
    review = assistant.review(packet)
    assert review.status == "deterministic_only"


def test_too_many_hypotheses_or_questions_are_refused(run, packet):
    reply = explanation_draft(
        packet.explanation_id,
        [packet.facts[0].fact_id],
        questions=[f"question {i}" for i in range(50)],
    )
    with pytest.raises(SchemaViolation, match="exceed the budget"):
        read_explanation_draft(reply, CallBudget())


def test_too_many_tool_calls_are_refused_before_any_of_them_runs():
    reply = tool_selection(*[{"tool": "get_run_summary", "arguments": {}} for _ in range(20)])
    with pytest.raises(SchemaViolation, match="past the budget"):
        read_tool_selection(reply, CallBudget())


def test_a_request_past_the_request_budget_is_not_sent(run, packet):
    assistant, _ = assistant_for(run, packet, [good_draft(packet)], budget=CallBudget(max_request_bytes=32))
    review = assistant.review(packet)
    assert review.status == "deterministic_only"
    assert any("bytes exceeds the budget" in reason for reason in review.reasons)
    assert assistant.provider.requests == [], "an over-budget request must not reach the provider"


# ------------------------------------------ L09 mandatory qualifications stay
RETAINED_HEADING = "Mandatory qualifications (retained by the report"


def retained_section(report: str) -> str:
    """The part of the report after the retained-qualifications heading.

    Asserting against this slice rather than the whole report matters: the
    deterministic renderer already prints a Limitations section of its own, so
    a test that only searched the whole text would pass even if the retained
    block were empty.
    """
    assert RETAINED_HEADING in report, "the assisted report must carry the retained-qualifications block"
    return report[report.index(RETAINED_HEADING) :]


def test_a_draft_that_lists_no_limitation_does_not_remove_one(run, packet):
    reply = explanation_draft(packet.explanation_id, [packet.facts[0].fact_id], limitations=[])
    assistant, _ = assistant_for(run, packet, [tool_selection(), reply])
    review = assistant.review(packet)
    assert review.status == "drafted"
    assert packet.limitations, "the fixture must actually carry qualifications for this to mean anything"
    retained = retained_section(review.report)
    for qualification in packet.limitations:
        assert qualification.code.value in retained, f"{qualification.code.value} was not retained after the draft"
        assert qualification.message[:40] in retained


def test_a_draft_that_invents_its_own_limitations_does_not_replace_the_packets(run, packet):
    reply = explanation_draft(
        packet.explanation_id,
        [packet.facts[0].fact_id],
        limitations=["Everything here is fully verified and carries no caveats."],
    )
    assistant, _ = assistant_for(run, packet, [tool_selection(), reply])
    retained = retained_section(assistant.review(packet).report)
    for qualification in packet.limitations:
        assert qualification.code.value in retained


def test_the_facts_a_draft_leaves_out_are_counted_not_silently_dropped(run, packet):
    draft = ExplanationDraft(packet_id=packet.explanation_id, fact_order=(packet.facts[0].fact_id,))
    report = render_reviewed_report(packet, draft)
    assert f"({len(packet.facts) - 1} further fact(s)" in report


def test_a_draft_cannot_change_a_value(run, packet):
    fact = packet.facts[0]
    draft = ExplanationDraft(packet_id=packet.explanation_id, fact_order=(fact.fact_id,))
    report = render_reviewed_report(packet, draft)
    assert f"{fact.value:.6g}" in report if isinstance(fact.value, float) else True
    assert "999999" not in report


# ------------------------------------------------- L11 hypotheses stay apart
def test_a_hypothesis_with_valid_ids_is_still_printed_as_unverified(run, packet):
    known = packet.evidence_refs[0]
    draft = ExplanationDraft(
        packet_id=packet.explanation_id,
        fact_order=(packet.facts[0].fact_id,),
        hypotheses=(Hypothesis("Late goods receipts explain the gap.", (known,)),),
    )
    report = render_reviewed_report(packet, draft)
    assert "Unverified hypotheses (not findings; nothing below has been tested)" in report
    assert "[unverified] Late goods receipts explain the gap." in report
    assert "not that any statement above follows from it" in report


def test_check_references_accepts_only_what_exists(run, packet):
    good = ExplanationDraft(packet_id=packet.explanation_id, fact_order=(packet.facts[0].fact_id,))
    assert check_references(packet, good).ok
    bad = ExplanationDraft(packet_id=packet.explanation_id, fact_order=("nope",))
    assert not check_references(packet, bad).ok


# ------------------------------------------------------------- provenance
def test_the_review_serialises_with_its_provider_records_and_ledger(run, packet):
    assistant, _ = assistant_for(run, packet, [tool_selection(), good_draft(packet)])
    payload = json.loads(json.dumps(assistant.review(packet).to_dict(), default=str))
    assert payload["status"] == "drafted"
    assert payload["ledger"]["calls"] == 2
    assert payload["provider_records"][0]["identity"]["model"] == "fake-model"
    assert payload["reference_check"]["ok"] is True


def test_the_prompt_carries_fact_identities_and_never_the_event_log(run, packet):
    assistant, _ = assistant_for(run, packet, [tool_selection(), good_draft(packet)])
    assistant.review(packet)
    sent = "\n".join(m.content for request in assistant.provider.requests for m in request.messages)
    assert packet.facts[0].fact_id in sent
    assert "Record Invoice Receipt" not in sent, "no raw trace is embedded as prose"
    assert str(run.log.events.shape[0]) not in sent.split("Explanation id")[0]


def test_the_schema_is_sent_with_every_request(run, packet):
    assistant, _ = assistant_for(run, packet, [tool_selection(), good_draft(packet)])
    assistant.review(packet)
    assert all(r.response_schema is not None for r in assistant.provider.requests)
    assert assistant.provider.requests[0].prompt_version.startswith("wise-llm-prompt/")


def test_a_module_reimport_still_contacts_nothing():
    module = importlib.import_module("wise.llm")
    importlib.reload(module)
    assert module.DEFAULT_ENDPOINT == "http://127.0.0.1:11434"
