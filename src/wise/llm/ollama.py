"""A read-only provider for a local Ollama server that is already running.

What this module does: post one bounded request to ``/api/chat`` with
``stream=false`` and a JSON schema in ``format``, read a bounded reply, and
report what actually answered.

What it deliberately does not do:

* start a server — a missing server is :attr:`ProviderStatus.UNAVAILABLE`;
* pull a model — a missing model is :attr:`ProviderStatus.UNAVAILABLE`;
* fall back to a hosted service — there is no second endpoint in this code;
* trust the reply's model identity — the model is compared against an exact
  allowlist, and a reply from a model that was not asked for is
  :attr:`ProviderStatus.INVALID`;
* hard-code a model family as semantically required — the model name is
  configuration, and every check in the library is deterministic regardless
  of which model answered, or whether one did.

The endpoint is a loopback literal by default, validated by
:mod:`wise.llm.transport` when it is configured rather than when it is used.

Nothing here runs at import: constructing an :class:`OllamaProvider` builds no
socket and contacts nothing, so ``import wise.llm.ollama`` is as offline as
``import json``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..errors import LLMError, TransportError
from .provider import ChatRequest, ModelIdentity, ProviderResult, ProviderStatus
from .transport import DEFAULT_ENDPOINT, Outcome, StrictLocalTransport, TransportConfig

#: The chat route this provider posts to. A constant, never an argument.
CHAT_ROUTE = "/api/chat"

#: The embedding route, used only when an embedding component is explicitly
#: configured. This library's retrieval baseline is lexical and needs none.
EMBED_ROUTE = "/api/embed"

#: Inference options sent unless the caller overrides them. Deterministic on
#: purpose: a review that reruns should get the same draft where the server
#: honours a seed, and nothing in the library depends on it doing so.
DEFAULT_OPTIONS: dict[str, Any] = {"temperature": 0.0, "seed": 0, "num_predict": 1024}


@dataclass(frozen=True)
class OllamaConfig:
    """Which local model may be asked, and under which limits.

    ``model`` must appear in ``allowed_models``: an exact allowlist, not a
    prefix or a family. The default endpoint is loopback; a non-loopback
    deployment needs the transport's explicit reviewed configuration and its
    own authentication and network controls.

    >>> OllamaConfig(model="llama3.1:8b", allowed_models=("llama3.1:8b",)).endpoint
    'http://127.0.0.1:11434'
    >>> OllamaConfig(model="gpt-4o", allowed_models=("llama3.1:8b",))
    Traceback (most recent call last):
        ...
    wise.errors.LLMError: model 'gpt-4o' is not in the allowlist ('llama3.1:8b',)
    """

    model: str
    allowed_models: tuple[str, ...] = ()
    endpoint: str = DEFAULT_ENDPOINT
    options: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_OPTIONS))
    keep_alive: str | int | None = 0
    transport: TransportConfig | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", str(self.model))
        object.__setattr__(self, "allowed_models", tuple(str(m) for m in self.allowed_models))
        object.__setattr__(self, "options", dict(self.options))
        if not self.model:
            raise LLMError("a model name is required; this library never picks one for you")
        if not self.allowed_models:
            raise LLMError(
                "allowed_models must list the model identifiers this deployment has reviewed; "
                "an empty allowlist fails closed rather than allowing anything"
            )
        if self.model not in self.allowed_models:
            raise LLMError(f"model {self.model!r} is not in the allowlist {self.allowed_models}")
        config = self.transport if self.transport is not None else TransportConfig(endpoint=self.endpoint)
        if config.endpoint != TransportConfig(endpoint=self.endpoint).endpoint and self.transport is None:
            raise LLMError("endpoint and transport disagree")  # pragma: no cover - defensive
        object.__setattr__(self, "transport", config)
        object.__setattr__(self, "endpoint", config.endpoint)

    @property
    def transport_config(self) -> TransportConfig:
        assert self.transport is not None  # set in __post_init__
        return self.transport

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "allowed_models": list(self.allowed_models),
            "endpoint": self.endpoint,
            "options": dict(self.options),
            "keep_alive": self.keep_alive,
            "transport": self.transport_config.to_dict(),
        }


class OllamaProvider:
    """A :class:`~wise.llm.provider.LLMProvider` over an already-running local server.

    >>> provider = OllamaProvider(OllamaConfig(model="m", allowed_models=("m",)))
    >>> provider.describe().model
    'm'
    >>> provider.describe().digest is None       # nothing was contacted to find out
    True
    """

    def __init__(self, config: OllamaConfig, *, transport: StrictLocalTransport | None = None) -> None:
        self.config = config
        self._transport = transport if transport is not None else StrictLocalTransport(config.transport_config)

    @property
    def transport(self) -> StrictLocalTransport:
        return self._transport

    def describe(self) -> ModelIdentity:
        """The *configured* identity. No digest, because none was asked for."""
        return ModelIdentity(
            provider="ollama",
            server=self.config.endpoint,
            model=self.config.model,
            digest=None,
            options=dict(self.config.options),
        )

    def chat(self, request: ChatRequest) -> ProviderResult:
        """One bounded exchange with the local server."""
        payload = self.build_payload(request)
        try:
            response = self._transport.post_json(
                CHAT_ROUTE,
                payload,
                timeout_s=min(float(request.timeout_s), float(self.config.transport_config.timeout_s)),
            )
        except TransportError:
            # a redirect, an unexpected host or an oversize request is a
            # boundary violation, not an absent server: it stays loud
            raise
        return self._interpret(request, response)

    # ------------------------------------------------------------- internals
    def build_payload(self, request: ChatRequest) -> dict[str, Any]:
        """The exact JSON posted to ``/api/chat``.

        ``stream`` is always ``false`` — a streamed reply has no single
        bounded size — and the response schema, when there is one, goes in
        ``format``.

        >>> from wise.llm.provider import ChatMessage, ChatRequest
        >>> provider = OllamaProvider(OllamaConfig(model="m", allowed_models=("m",)))
        >>> payload = provider.build_payload(ChatRequest((ChatMessage("user", "hi"),)))
        >>> payload["stream"], payload["model"]
        (False, 'm')
        """
        options = dict(self.config.options)
        options.update(request.options)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.to_dict() for m in request.messages],
            "stream": False,
            "options": options,
        }
        if request.response_schema is not None:
            payload["format"] = request.response_schema
        if self.config.keep_alive is not None:
            payload["keep_alive"] = self.config.keep_alive
        return payload

    def _interpret(self, request: ChatRequest, response: Any) -> ProviderResult:
        configured = self.describe()
        base = {
            "request_bytes": request.text_bytes,
            "response_bytes": len(response.body),
            "attempts": response.attempts,
            "elapsed_s": response.elapsed_s,
        }
        if response.outcome is Outcome.UNAVAILABLE:
            return ProviderResult(
                ProviderStatus.UNAVAILABLE,
                identity=configured,
                reason=self._absence_reason(response),
                **base,
            )
        if response.outcome is Outcome.TIMEOUT:
            return ProviderResult(ProviderStatus.TIMEOUT, identity=configured, reason=response.reason, **base)
        if response.outcome is Outcome.OVERSIZE:
            return ProviderResult(ProviderStatus.OVERSIZE, identity=configured, reason=response.reason, **base)
        if response.outcome is Outcome.HTTP_ERROR:
            return ProviderResult(
                ProviderStatus.INVALID,
                identity=configured,
                reason=f"the server answered {response.reason}",
                **base,
            )
        try:
            body = response.json()
        except TransportError as exc:
            return ProviderResult(ProviderStatus.INVALID, identity=configured, reason=str(exc), **base)
        if not isinstance(body, Mapping):
            return ProviderResult(ProviderStatus.INVALID, identity=configured, reason="the reply is not a JSON object", **base)
        answered = str(body.get("model", "") or "")
        if answered and answered not in self.config.allowed_models:
            return ProviderResult(
                ProviderStatus.INVALID,
                identity=configured,
                reason=(
                    f"the reply names model {answered!r}, which is not in the allowlist "
                    f"{self.config.allowed_models}; a response does not choose the model"
                ),
                **base,
            )
        message = body.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            return ProviderResult(
                ProviderStatus.INVALID,
                identity=configured,
                reason="the reply carries no message content",
                **base,
            )
        identity = ModelIdentity(
            provider="ollama",
            server=self.config.endpoint,
            model=answered or self.config.model,
            digest=_digest_of(body),
            server_version=_text_or_none(body.get("server_version")),
            options=dict(self.config.options) | dict(request.options),
            prompt_version=request.prompt_version,
            schema_version=request.schema_version,
        )
        status = ProviderStatus.OK
        reason = ""
        done_reason = str(body.get("done_reason", "") or "")
        if done_reason == "length":
            status = ProviderStatus.OVERSIZE
            reason = "the model stopped at its token limit, so the reply is incomplete"
        return ProviderResult(
            status,
            text=content,
            identity=identity,
            reason=reason,
            server_metadata=_bounded_metadata(body),
            **base,
        )

    @staticmethod
    def _absence_reason(response: Any) -> str:
        detail = ""
        if response.body:
            try:
                parsed = json.loads(response.body.decode("utf-8"))
                if isinstance(parsed, Mapping):
                    detail = str(parsed.get("error", ""))
            except (UnicodeDecodeError, json.JSONDecodeError):
                detail = ""
        base = response.reason or "the local server did not answer"
        note = (
            "the model must already be installed on the local server; this library never pulls one, "
            "never starts a server and has no remote fallback"
        )
        return f"{base}{': ' + detail if detail else ''} — {note}"


def _digest_of(body: Mapping[str, Any]) -> str | None:
    """The reply's content digest where the server offers one, else ``None``."""
    for key in ("digest", "model_digest"):
        value = body.get(key)
        if isinstance(value, str) and value:
            return value
    details = body.get("details")
    if isinstance(details, Mapping):
        value = details.get("digest")
        if isinstance(value, str) and value:
            return value
    return None


def _text_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _bounded_metadata(body: Mapping[str, Any]) -> dict[str, Any]:
    """The small, numeric part of a reply worth recording in a run record."""
    keys = (
        "created_at",
        "done",
        "done_reason",
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "eval_count",
    )
    out: dict[str, Any] = {}
    for key in keys:
        if key in body and isinstance(body[key], str | int | float | bool | type(None)):
            out[key] = body[key]
    return out


def embedding_payload(model: str, inputs: Sequence[str]) -> dict[str, Any]:
    """The body of an ``/api/embed`` request, for a caller that configures one.

    Provided so an application that has explicitly enabled a local embedding
    component has a checked payload shape. This library's retrieval baseline
    is lexical and calls it from nowhere.

    >>> embedding_payload("nomic-embed-text", ["a", "b"])["input"]
    ['a', 'b']
    """
    if not inputs:
        raise LLMError("an embedding request needs at least one input")
    return {"model": str(model), "input": [str(text) for text in inputs], "truncate": True}


__all__ = [
    "CHAT_ROUTE",
    "DEFAULT_OPTIONS",
    "EMBED_ROUTE",
    "OllamaConfig",
    "OllamaProvider",
    "embedding_payload",
]
