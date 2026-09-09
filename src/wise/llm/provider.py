"""The provider protocol, its budgets, and the fake every test runs against.

A provider is the *only* thing in this package that talks to a model, and it
is deliberately small: one method, one typed request, one typed result. It
returns text and an identity; it never returns a number the library will
believe, and it never decides who the caller is.

Failure is a status, not an accident. A server that is not running, a model
that is not installed, a reply larger than the budget, a reply that never
stops, a timeout, an exhausted call budget — each is a
:class:`ProviderStatus`, so the caller can render the deterministic report and
say precisely what did not happen. Only a *boundary* violation raises: a
redirect, an unexpected host, a request past its byte budget. Those are
configuration errors and must not be swallowed.

:class:`FakeProvider` is the provider every test in this repository uses. It
replays a script, records what it was asked, and can be told to be
unavailable, slow, oversized or wrong. No test in this library starts a
server, downloads a model or opens a socket.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from ..errors import BudgetExceeded, LLMError

#: Version of the provider contract. Bumped when a field changes meaning.
PROVIDER_SCHEMA_VERSION = "wise-llm-provider/1"

#: Roles a message may carry. ``assistant`` exists for a reviewed few-shot
#: example; nothing in this library puts model output back into a prompt.
ROLES = ("system", "user", "assistant")


class ProviderStatus(str, Enum):
    """What became of one bounded request."""

    #: A reply arrived, within budget, from the model that was asked for.
    OK = "ok"
    #: No server, or the model is not installed on it. Not an error: the
    #: deterministic report is still the report.
    UNAVAILABLE = "unavailable"
    #: The reply did not arrive inside the declared timeout.
    TIMEOUT = "timeout"
    #: The reply exceeded the declared byte budget, or did not stop.
    OVERSIZE = "oversize"
    #: A reply arrived but is not what was asked for — unparseable, or from a
    #: model the caller did not allow.
    INVALID = "invalid"
    #: A budget (calls, bytes, retries) was already spent.
    BUDGET_EXHAUSTED = "budget_exhausted"
    #: The provider declined to answer. An abstention is a valid answer.
    REFUSED = "refused"


@dataclass(frozen=True)
class ModelIdentity:
    """Who actually answered, as reported by the server — not as configured.

    ``model`` is the identifier the server named in its reply; ``digest`` is
    the content digest where the server offers one and ``None`` where it does
    not, because an unknown digest is unknown, not empty.
    """

    provider: str
    server: str
    model: str
    digest: str | None = None
    server_version: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    prompt_version: str = ""
    schema_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "server": self.server,
            "model": self.model,
            "digest": self.digest,
            "server_version": self.server_version,
            "options": dict(self.options),
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class ChatMessage:
    """One message. Content is text; nothing here is executed anywhere."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise LLMError(f"message role must be one of {ROLES}, got {self.role!r}")
        if not isinstance(self.content, str):
            raise LLMError(f"message content must be a string, got {type(self.content).__name__}")

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ChatRequest:
    """One bounded request: messages, a required response schema, and limits.

    ``response_schema`` is not advisory. It is handed to the server as the
    structural constraint on the reply and re-checked locally afterwards,
    because a server honouring a schema is a convenience, never the guarantee.
    """

    messages: tuple[ChatMessage, ...]
    response_schema: dict[str, Any] | None = None
    prompt_version: str = ""
    schema_version: str = ""
    options: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 30.0
    max_response_bytes: int = 262_144

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "options", dict(self.options))
        if not self.messages:
            raise LLMError("a chat request needs at least one message")
        if self.timeout_s <= 0:
            raise LLMError(f"timeout_s must be positive, got {self.timeout_s!r}")
        if self.max_response_bytes <= 0:
            raise LLMError(f"max_response_bytes must be positive, got {self.max_response_bytes!r}")

    @property
    def text_bytes(self) -> int:
        """Bytes of message content, for the request budget.

        >>> ChatRequest((ChatMessage("user", "hello"),)).text_bytes
        5
        """
        return sum(len(m.content.encode("utf-8")) for m in self.messages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [m.to_dict() for m in self.messages],
            "response_schema": self.response_schema,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "options": dict(self.options),
            "timeout_s": self.timeout_s,
            "max_response_bytes": self.max_response_bytes,
        }


@dataclass(frozen=True)
class ProviderResult:
    """The typed outcome of one request. Never an exception for absence."""

    status: ProviderStatus
    text: str = ""
    identity: ModelIdentity | None = None
    reason: str = ""
    request_bytes: int = 0
    response_bytes: int = 0
    attempts: int = 1
    elapsed_s: float | None = None
    server_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ProviderStatus(self.status))
        object.__setattr__(self, "server_metadata", dict(self.server_metadata))

    @property
    def ok(self) -> bool:
        """Whether the request completed successfully.

        >>> ProviderResult(ProviderStatus.UNAVAILABLE).ok
        False
        """
        return self.status is ProviderStatus.OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "identity": None if self.identity is None else self.identity.to_dict(),
            "request_bytes": int(self.request_bytes),
            "response_bytes": int(self.response_bytes),
            "attempts": int(self.attempts),
            "elapsed_s": self.elapsed_s,
            "server_metadata": dict(self.server_metadata),
            # deliberately not the reply text: a run record is provenance, not
            # a transcript, and a transcript can carry retrieved content
        }


@runtime_checkable
class LLMProvider(Protocol):
    """What the library needs of a model, and nothing more."""

    def describe(self) -> ModelIdentity:
        """The configured identity, without contacting anything."""

    def chat(self, request: ChatRequest) -> ProviderResult:
        """One bounded exchange."""


# --------------------------------------------------------------------- budgets
@dataclass(frozen=True)
class CallBudget:
    """Every limit the assistance path spends against, declared in one place.

    Defaults are small on purpose: two model steps, a quarter of a megabyte
    each way, one retry, and counts on everything the model can ask for. A
    caller raises a limit deliberately; nothing raises one for it.
    """

    max_calls: int = 2
    max_tool_calls: int = 6
    max_request_bytes: int = 262_144
    max_response_bytes: int = 262_144
    max_retries: int = 1
    timeout_s: float = 30.0
    max_retrieved_documents: int = 8
    max_retrieved_bytes: int = 32_768
    max_facts: int = 40
    max_witnesses: int = 8
    max_hypotheses: int = 5
    max_questions: int = 8
    max_text_bytes: int = 4_096

    def __post_init__(self) -> None:
        for name in (
            "max_calls",
            "max_tool_calls",
            "max_request_bytes",
            "max_response_bytes",
            "timeout_s",
            "max_retrieved_documents",
            "max_retrieved_bytes",
            "max_facts",
            "max_witnesses",
            "max_hypotheses",
            "max_questions",
            "max_text_bytes",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise LLMError(f"{name} must be positive, got {value!r}")
        if self.max_retries < 0:
            raise LLMError(f"max_retries must be non-negative, got {self.max_retries!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_calls": self.max_calls,
            "max_tool_calls": self.max_tool_calls,
            "max_request_bytes": self.max_request_bytes,
            "max_response_bytes": self.max_response_bytes,
            "max_retries": self.max_retries,
            "timeout_s": self.timeout_s,
            "max_retrieved_documents": self.max_retrieved_documents,
            "max_retrieved_bytes": self.max_retrieved_bytes,
            "max_facts": self.max_facts,
            "max_witnesses": self.max_witnesses,
            "max_hypotheses": self.max_hypotheses,
            "max_questions": self.max_questions,
            "max_text_bytes": self.max_text_bytes,
        }


@dataclass
class BudgetLedger:
    """What one review has spent so far. Mutable, and never reset silently.

    >>> ledger = BudgetLedger(CallBudget(max_calls=1))
    >>> ledger.spend_call()
    1
    >>> ledger.spend_call()
    Traceback (most recent call last):
        ...
    wise.errors.BudgetExceeded: model call budget of 1 is exhausted
    """

    budget: CallBudget = field(default_factory=CallBudget)
    calls: int = 0
    tool_calls: int = 0
    request_bytes: int = 0
    response_bytes: int = 0
    retrieved_documents: int = 0
    retrieved_bytes: int = 0

    def spend_call(self) -> int:
        if self.calls >= self.budget.max_calls:
            raise BudgetExceeded(f"model call budget of {self.budget.max_calls} is exhausted")
        self.calls += 1
        return self.calls

    def spend_tool_call(self, tool: str) -> int:
        if self.tool_calls >= self.budget.max_tool_calls:
            raise BudgetExceeded(f"tool call budget of {self.budget.max_tool_calls} is exhausted (refusing {tool!r})")
        self.tool_calls += 1
        return self.tool_calls

    def spend_request_bytes(self, n: int) -> int:
        if n > self.budget.max_request_bytes:
            raise BudgetExceeded(f"request of {n} bytes exceeds the budget of {self.budget.max_request_bytes}")
        self.request_bytes += int(n)
        return self.request_bytes

    def record_response_bytes(self, n: int) -> int:
        self.response_bytes += int(n)
        return self.response_bytes

    def spend_retrieval(self, n_documents: int, n_bytes: int) -> None:
        if self.retrieved_documents + n_documents > self.budget.max_retrieved_documents:
            raise BudgetExceeded(
                f"retrieval budget of {self.budget.max_retrieved_documents} documents is exhausted "
                f"({self.retrieved_documents} already retrieved)"
            )
        if self.retrieved_bytes + n_bytes > self.budget.max_retrieved_bytes:
            raise BudgetExceeded(f"retrieval byte budget of {self.budget.max_retrieved_bytes} is exhausted")
        self.retrieved_documents += int(n_documents)
        self.retrieved_bytes += int(n_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget.to_dict(),
            "calls": self.calls,
            "tool_calls": self.tool_calls,
            "request_bytes": self.request_bytes,
            "response_bytes": self.response_bytes,
            "retrieved_documents": self.retrieved_documents,
            "retrieved_bytes": self.retrieved_bytes,
        }


# ----------------------------------------------------------------------- fake
class FakeProvider:
    """A scripted provider: the one every test in this library uses.

    ``script`` is replayed in order. Each entry may be

    * a string — returned as an ``ok`` reply;
    * a :class:`ProviderResult` — returned as it stands;
    * a :class:`ProviderStatus` — returned as an empty reply with that status;
    * an :class:`Exception` — raised, for testing a boundary violation.

    When the script runs out the provider answers ``exhausted_status``, which
    defaults to :attr:`ProviderStatus.UNAVAILABLE` so a test that calls one
    time too many notices.

    >>> provider = FakeProvider(['{"ok": true}'])
    >>> result = provider.chat(ChatRequest((ChatMessage("user", "hi"),)))
    >>> result.status.value, result.text
    ('ok', '{"ok": true}')
    >>> provider.requests[0].messages[0].content
    'hi'
    """

    def __init__(
        self,
        script: Sequence[Any] = (),
        *,
        model: str = "fake-model",
        server: str = "fake://local",
        digest: str | None = "sha256:fake",
        exhausted_status: ProviderStatus = ProviderStatus.UNAVAILABLE,
    ) -> None:
        self._script = list(script)
        self._identity = ModelIdentity(provider="fake", server=server, model=model, digest=digest)
        self._exhausted = ProviderStatus(exhausted_status)
        #: every request this provider was handed, in order
        self.requests: list[ChatRequest] = []

    def describe(self) -> ModelIdentity:
        return self._identity

    def chat(self, request: ChatRequest) -> ProviderResult:
        if not isinstance(request, ChatRequest):  # pragma: no cover - defensive
            raise LLMError(f"chat expects a ChatRequest, got {type(request).__name__}")
        self.requests.append(request)
        identity = replace(
            self._identity,
            options=dict(request.options),
            prompt_version=request.prompt_version,
            schema_version=request.schema_version,
        )
        if not self._script:
            return ProviderResult(
                self._exhausted,
                identity=identity,
                reason="the fake provider's script is exhausted",
                request_bytes=request.text_bytes,
            )
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, ProviderResult):
            return replace(item, identity=item.identity or identity, request_bytes=item.request_bytes or request.text_bytes)
        if isinstance(item, ProviderStatus):
            return ProviderResult(item, identity=identity, request_bytes=request.text_bytes)
        text = item if isinstance(item, str) else json.dumps(item)
        encoded = len(text.encode("utf-8"))
        if encoded > request.max_response_bytes:
            return ProviderResult(
                ProviderStatus.OVERSIZE,
                identity=identity,
                reason=f"reply of {encoded} bytes exceeds the declared limit of {request.max_response_bytes}",
                request_bytes=request.text_bytes,
                response_bytes=encoded,
            )
        return ProviderResult(
            ProviderStatus.OK,
            text=text,
            identity=identity,
            request_bytes=request.text_bytes,
            response_bytes=encoded,
        )


def unavailable_result(reason: str, identity: ModelIdentity | None = None) -> ProviderResult:
    """The specific result a missing server or model produces.

    >>> unavailable_result("no server").status.value
    'unavailable'
    """
    return ProviderResult(ProviderStatus.UNAVAILABLE, identity=identity, reason=reason)


def system_and_user(system: str, user: str) -> tuple[ChatMessage, ...]:
    """The two-message shape every prompt in this library uses.

    >>> [m.role for m in system_and_user("rules", "question")]
    ['system', 'user']
    """
    return (ChatMessage("system", system), ChatMessage("user", user))


def messages_from(pairs: Iterable[Mapping[str, str]]) -> tuple[ChatMessage, ...]:
    """Build messages from plain mappings, validating each role."""
    return tuple(ChatMessage(str(p["role"]), str(p["content"])) for p in pairs)


__all__ = [
    "PROVIDER_SCHEMA_VERSION",
    "ROLES",
    "BudgetLedger",
    "CallBudget",
    "ChatMessage",
    "ChatRequest",
    "FakeProvider",
    "LLMProvider",
    "ModelIdentity",
    "ProviderResult",
    "ProviderStatus",
    "messages_from",
    "system_and_user",
    "unavailable_result",
]
