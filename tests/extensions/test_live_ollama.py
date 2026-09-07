"""The opt-in live harness. It does not run, and its result is not a result.

The whole of the local-assistance work is built and tested against
:class:`~wise.llm.provider.FakeProvider`. This file is the one place a real
server could be contacted, and it is skipped unless three environment
variables are set deliberately:

``WISE_OLLAMA_LIVE``
    must be ``1`` — the explicit opt-in;
``WISE_OLLAMA_MODEL``
    the exact model identifier, which must already be installed. Nothing here
    pulls one;
``WISE_OLLAMA_ENDPOINT``
    optional; defaults to the loopback literal.

What it checks is the *shape* of the exchange — that a bounded request to an
already-running local server comes back as a typed result, and that whatever
happens the deterministic report is still the report. It measures no quality
and no latency, and a passing run of this file establishes neither. Reporting
it as evidence of anything about a model would be a fabricated result.
"""

from __future__ import annotations

import os

import pytest

import wise
from wise.llm.assistant import LocalAssistant
from wise.llm.gateway import EvidenceHost, ToolGateway
from wise.llm.ollama import OllamaConfig, OllamaProvider
from wise.llm.policy import AccessPolicy, Principal, Scope

LIVE = os.environ.get("WISE_OLLAMA_LIVE") == "1"
MODEL = os.environ.get("WISE_OLLAMA_MODEL", "")
ENDPOINT = os.environ.get("WISE_OLLAMA_ENDPOINT", "http://127.0.0.1:11434")

pytestmark = pytest.mark.live_llm


@pytest.mark.skipif(
    not (LIVE and MODEL),
    reason="live local-model run is opt-in: set WISE_OLLAMA_LIVE=1 and WISE_OLLAMA_MODEL to an installed model",
)
def test_a_live_local_model_produces_a_typed_result_and_never_replaces_the_report():
    result = wise.score(wise.running_p2p_log(), wise.running_p2p_norm(), evidence="summary")
    packet = wise.explain_priority(result, "company", "B", view="Finance", gamma=1.0)
    policy = AccessPolicy(
        Principal("live-harness"),
        scopes=frozenset({Scope.EXPLANATION, Scope.RUN_SUMMARY}),
        runs=frozenset({result.manifest.run_id}),
        views=frozenset(result.views),
        columns=frozenset(result.cases.columns),
        unrestricted_rows=True,
    )
    gateway = ToolGateway(EvidenceHost(result, packet=packet), policy)
    provider = OllamaProvider(OllamaConfig(model=MODEL, allowed_models=(MODEL,), endpoint=ENDPOINT))
    review = LocalAssistant(provider, gateway).review(packet)

    assert review.status in ("drafted", "deterministic_only")
    assert review.report.startswith("Priority explanation")
    for qualification in packet.limitations:
        assert qualification.code.value in review.report
    if review.draft is not None:
        assert review.reference_check.ok
        assert set(review.draft.fact_order) <= {f.fact_id for f in packet.facts}
