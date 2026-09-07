"""The strict local transport and the read-only Ollama provider (item L01).

Acceptance rows L06 (oversize, loop, timeout), L07 (redirect, unexpected host,
proxy, model) and L10 (no cloud fallback, no automatic pull, no automatic
start) live here, plus the "no digest was invented" property.

**No socket is opened anywhere in this file.** Every request path is driven
through an injected opener, and the handler composition is asserted on the
opener object itself. A local Ollama service may exist on the machine running
these tests; nothing here contacts, configures or starts it.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest
from _llm_fixtures import FakeHTTPResponse, FakeOpener, chat_reply

from wise.errors import LLMError, TransportError
from wise.llm.ollama import CHAT_ROUTE, OllamaConfig, OllamaProvider, embedding_payload
from wise.llm.provider import ChatMessage, ChatRequest, ProviderStatus
from wise.llm.transport import (
    DEFAULT_ENDPOINT,
    Outcome,
    StrictLocalTransport,
    TransportConfig,
    build_opener,
    check_endpoint,
    is_loopback,
)

REQUEST = ChatRequest((ChatMessage("user", "summarise"),), response_schema={"type": "object"})

#: One attempt, so a script of one entry is exactly one call. Retries have
#: their own tests; everywhere else they would only obscure the assertion.
NO_RETRY = TransportConfig(max_retries=0)


def provider_with(script, *, config=None, allowed=("test-model",)):
    settings = config or OllamaConfig(model="test-model", allowed_models=allowed, transport=NO_RETRY)
    opener = FakeOpener(script)
    transport = StrictLocalTransport(settings.transport_config, opener=opener)
    return OllamaProvider(settings, transport=transport), opener


# ------------------------------------------------------- L07 endpoint policy
def test_the_default_endpoint_is_a_loopback_literal():
    assert DEFAULT_ENDPOINT == "http://127.0.0.1:11434"
    assert TransportConfig().loopback_only is True


@pytest.mark.parametrize("host", ["127.0.0.1", "127.5.5.5", "::1", "localhost"])
def test_loopback_hosts_are_accepted(host):
    assert is_loopback(host)
    netloc = f"[{host}]" if ":" in host else host
    assert check_endpoint(f"http://{netloc}:11434")


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://10.0.0.5:11434",
        "https://ollama.example.invalid",
        "http://ollama-gateway",
        "http://169.254.169.254",
    ],
)
def test_a_non_loopback_endpoint_is_refused_without_an_explicit_review(endpoint):
    with pytest.raises(TransportError, match="not loopback"):
        check_endpoint(endpoint)


def test_a_non_loopback_endpoint_needs_both_the_flag_and_the_named_host():
    with pytest.raises(TransportError, match="not loopback"):
        check_endpoint("http://ollama.corp:11434", allow_non_loopback=True)
    with pytest.raises(TransportError, match="not loopback"):
        check_endpoint("http://ollama.corp:11434", reviewed_hosts=("ollama.corp",))
    assert (
        check_endpoint("http://ollama.corp:11434", allow_non_loopback=True, reviewed_hosts=("ollama.corp",))
        == "http://ollama.corp:11434"
    )


@pytest.mark.parametrize(
    "endpoint,match",
    [
        ("ftp://127.0.0.1", "scheme"),
        ("file:///etc/passwd", "scheme"),
        ("http://user:secret@127.0.0.1:11434", "credentials"),
        ("http://127.0.0.1:11434/api/chat", "bare origin"),
        ("http://127.0.0.1:11434/?x=1", "bare origin"),
    ],
)
def test_a_malformed_or_credentialed_endpoint_is_refused(endpoint, match):
    with pytest.raises(TransportError, match=match):
        check_endpoint(endpoint)


def test_only_declared_routes_can_be_posted_to():
    config = TransportConfig()
    assert config.url_for("/api/chat").endswith("/api/chat")
    for route in ("/api/pull", "/api/create", "/../etc/passwd", "/api/chat/../../x"):
        with pytest.raises(TransportError, match="not one of"):
            config.url_for(route)


def test_a_route_with_traversal_cannot_even_be_configured():
    with pytest.raises(TransportError, match="traversal"):
        TransportConfig(routes=("/api/../etc",))


# ---------------------------------------------------------------- L07 proxy
def test_an_inherited_proxy_is_dropped_unless_explicitly_trusted(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://attacker.invalid:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://attacker.invalid:8080")
    opener = build_opener(TransportConfig())
    proxied = [h for h in opener.handlers if isinstance(h, urllib.request.ProxyHandler) and h.proxies]
    assert proxied == [], "an inherited proxy would carry a 'local' request off the machine"
    assert not any(hasattr(h, "http_open") and isinstance(h, urllib.request.ProxyHandler) for h in opener.handlers)

    trusting = build_opener(TransportConfig(trust_environment_proxy=True))
    assert any(isinstance(h, urllib.request.ProxyHandler) and h.proxies for h in trusting.handlers)


# ------------------------------------------------------------- L07 redirects
def test_a_redirect_is_refused_rather_than_followed():
    opener = build_opener(TransportConfig())
    handler = next(h for h in opener.handlers if isinstance(h, urllib.request.HTTPRedirectHandler))
    request = urllib.request.Request("http://127.0.0.1:11434/api/chat")
    with pytest.raises(TransportError, match="does not follow redirects"):
        handler.redirect_request(request, None, 302, "Found", {}, "https://api.openai.example/v1/chat")


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_every_redirect_status_is_refused(code):
    opener = build_opener(TransportConfig())
    handler = next(h for h in opener.handlers if isinstance(h, urllib.request.HTTPRedirectHandler))
    request = urllib.request.Request("http://127.0.0.1:11434/api/chat")
    method = getattr(handler, f"http_error_{code}")
    with pytest.raises(TransportError, match="redirect"):
        method(request, None, code, "Moved", {"location": "https://elsewhere.invalid/"})


def test_a_redirect_raised_during_a_call_stays_loud():
    provider, _ = provider_with([TransportError("http://127.0.0.1:11434/api/chat: the server answered 302 redirecting")])
    with pytest.raises(TransportError):
        provider.chat(REQUEST)


def test_a_reply_from_an_unexpected_host_is_refused():
    reply = FakeHTTPResponse(chat_reply("{}"), url="https://api.elsewhere.invalid/v1/chat")
    provider, _ = provider_with([reply])
    with pytest.raises(TransportError, match="not the configured endpoint"):
        provider.chat(REQUEST)


# --------------------------------------------------------------- L06 bounds
def test_an_oversize_reply_is_bounded_and_typed_not_consumed():
    huge = json.dumps(chat_reply("x" * 100_000)).encode("utf-8")
    config = OllamaConfig(
        model="test-model",
        allowed_models=("test-model",),
        transport=TransportConfig(max_response_bytes=2_048),
    )
    provider, _ = provider_with([FakeHTTPResponse(huge)], config=config)
    result = provider.chat(REQUEST)
    assert result.status is ProviderStatus.OVERSIZE
    assert result.response_bytes <= 2_048, "more than the budget was read into memory"


def test_a_model_that_never_stops_is_reported_as_oversize_not_as_an_answer():
    provider, _ = provider_with([FakeHTTPResponse(chat_reply("partial", done_reason="length"))])
    result = provider.chat(REQUEST)
    assert result.status is ProviderStatus.OVERSIZE
    assert "token limit" in result.reason


def test_an_oversize_request_is_refused_before_anything_is_sent():
    opener = FakeOpener([])
    transport = StrictLocalTransport(TransportConfig(max_request_bytes=64), opener=opener)
    with pytest.raises(TransportError, match="exceeds the configured limit"):
        transport.post_json(CHAT_ROUTE, {"messages": ["x" * 500]})
    assert opener.requests == [], "the request was sent despite being over budget"


def test_a_timeout_is_a_typed_status_and_is_retried_at_most_the_declared_number_of_times():
    config = OllamaConfig(
        model="test-model",
        allowed_models=("test-model",),
        transport=TransportConfig(max_retries=2, timeout_s=0.01),
    )
    provider, opener = provider_with([TimeoutError("timed out")] * 3, config=config)
    result = provider.chat(REQUEST)
    assert result.status is ProviderStatus.TIMEOUT
    assert len(opener.requests) == 3, "retries are bounded by max_retries + 1"
    assert result.attempts == 3


def test_retries_stop_as_soon_as_an_answer_arrives():
    config = OllamaConfig(model="test-model", allowed_models=("test-model",), transport=TransportConfig(max_retries=3))
    script = [urllib.error.URLError(ConnectionRefusedError(61, "refused")), FakeHTTPResponse(chat_reply('{"ok": true}'))]
    provider, opener = provider_with(script, config=config)
    result = provider.chat(REQUEST)
    assert result.status is ProviderStatus.OK
    assert len(opener.requests) == 2


# ------------------------------------------------- L10 absence, never a pull
def test_a_missing_server_is_a_specific_unavailable_result_not_an_exception():
    provider, _ = provider_with([urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))])
    result = provider.chat(REQUEST)
    assert result.status is ProviderStatus.UNAVAILABLE
    assert "never pulls one" in result.reason and "no remote fallback" in result.reason


def test_a_missing_model_is_unavailable_and_nothing_is_downloaded():
    error = urllib.error.HTTPError(
        "http://127.0.0.1:11434/api/chat",
        404,
        "Not Found",
        {},
        None,  # type: ignore[arg-type]
    )
    provider, opener = provider_with([error])
    result = provider.chat(REQUEST)
    assert result.status is ProviderStatus.UNAVAILABLE
    assert len(opener.requests) == 1
    assert all(r.full_url.endswith("/api/chat") for r in opener.requests), "no /api/pull was attempted"


def test_the_transport_can_only_ever_reach_the_two_declared_routes():
    assert set(TransportConfig().routes) == {"/api/chat", "/api/embed"}
    for forbidden in ("/api/pull", "/api/create", "/api/delete", "/api/push"):
        assert forbidden not in TransportConfig().routes


def test_there_is_no_second_endpoint_to_fall_back_to():
    provider, opener = provider_with([urllib.error.URLError("unreachable")])
    provider.chat(REQUEST)
    hosts = {urllib.request.Request(r.full_url).host for r in opener.requests}
    assert hosts == {"127.0.0.1:11434"}


# --------------------------------------------------------- L07 model identity
def test_the_model_is_an_exact_allowlist_not_a_family():
    with pytest.raises(LLMError, match="not in the allowlist"):
        OllamaConfig(model="llama3.1:70b", allowed_models=("llama3.1:8b",))
    with pytest.raises(LLMError, match="allowed_models must list"):
        OllamaConfig(model="anything", allowed_models=())


def test_a_reply_cannot_choose_the_model():
    provider, _ = provider_with([FakeHTTPResponse(chat_reply("{}", model="some-cloud-model"))])
    result = provider.chat(REQUEST)
    assert result.status is ProviderStatus.INVALID
    assert "does not choose the model" in result.reason


def test_a_digest_is_reported_only_when_the_server_offers_one():
    provider, _ = provider_with([FakeHTTPResponse(chat_reply("{}"))])
    assert provider.chat(REQUEST).identity.digest is None
    provider, _ = provider_with([FakeHTTPResponse(chat_reply("{}", digest="sha256:abc"))])
    assert provider.chat(REQUEST).identity.digest == "sha256:abc"


def test_describe_contacts_nothing_and_claims_no_digest():
    provider, opener = provider_with([])
    identity = provider.describe()
    assert identity.model == "test-model" and identity.digest is None
    assert opener.requests == []


# --------------------------------------------------------------- the payload
def test_the_payload_never_streams_and_carries_the_schema():
    provider, opener = provider_with([FakeHTTPResponse(chat_reply("{}"))])
    provider.chat(REQUEST)
    body = json.loads(opener.requests[0].data.decode("utf-8"))
    assert body["stream"] is False
    assert body["format"] == {"type": "object"}
    assert body["model"] == "test-model"
    assert body["options"]["temperature"] == 0.0


def test_a_reply_that_is_not_json_or_has_no_content_is_invalid_not_ok():
    provider, _ = provider_with([FakeHTTPResponse(b"<html>gateway</html>")])
    assert provider.chat(REQUEST).status is ProviderStatus.INVALID
    provider, _ = provider_with([FakeHTTPResponse({"model": "test-model", "done": True})])
    result = provider.chat(REQUEST)
    assert result.status is ProviderStatus.INVALID and "no message content" in result.reason


def test_an_http_error_other_than_404_is_invalid_and_bounded():
    error = urllib.error.HTTPError("http://127.0.0.1:11434/api/chat", 500, "boom", {}, None)  # type: ignore[arg-type]
    provider, _ = provider_with([error])
    result = provider.chat(REQUEST)
    assert result.status is ProviderStatus.INVALID and "500" in result.reason


def test_server_metadata_is_recorded_but_the_reply_text_is_not():
    provider, _ = provider_with([FakeHTTPResponse(chat_reply('{"a": "SECRET-CONTENT"}', eval_count=17))])
    result = provider.chat(REQUEST)
    assert result.text == '{"a": "SECRET-CONTENT"}'
    assert result.server_metadata["eval_count"] == 17
    assert "SECRET-CONTENT" not in json.dumps(result.to_dict()), "a run record is provenance, not a transcript"


def test_the_embedding_payload_exists_but_nothing_calls_it():
    payload = embedding_payload("nomic-embed-text", ["one", "two"])
    assert payload["input"] == ["one", "two"] and payload["truncate"] is True
    with pytest.raises(LLMError):
        embedding_payload("nomic-embed-text", [])


def test_a_transport_response_reports_the_route_it_used():
    opener = FakeOpener([FakeHTTPResponse({"ok": True})])
    transport = StrictLocalTransport(TransportConfig(), opener=opener)
    response = transport.post_json("/api/chat", {"model": "m"})
    assert response.outcome is Outcome.OK and transport.calls == ["/api/chat"]
    assert response.json() == {"ok": True}
