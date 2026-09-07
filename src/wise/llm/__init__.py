"""Optional local language assistance: a thin, bounded, read-only protocol.

The library owns the *boundary*, not the assistant. What lives here is the
provider protocol and its fake, a strict local transport, an explicit tool
gateway, an access policy applied before anything is read or aggregated, a
lexical retrieval baseline over approved sources, and the norm-draft envelope
with its deep validation and its preview. The prompts, the digests, the audit
trail and the approval workflow belong to the application.

What a model can do through this package, exhaustively:

* ask for one of six named, read-only tools, whose arguments are validated and
  whose results are size-bounded;
* propose an order for facts that already exist, questions, and hypotheses
  explicitly marked unverified;
* propose a norm draft, which is deeply validated and can be previewed on an
  isolated copy.

What it cannot do — structurally, rather than by instruction: compute or
change a number, reach a row, column, run or document the policy does not
allow, name a tool that is not in the frozen approved set, cause an import, a
shell, an ``eval`` or a path to be used, follow a redirect off the loopback
interface, approve a rule, or make a report drop a mandatory qualification.

Nothing here runs at import, and there is no optional dependency to install
for the code in this package: the transport is :mod:`urllib.request` and the
schema validator is written out, so ``import wise.llm`` contacts nothing,
starts nothing and downloads nothing. The ``[llm]`` extra exists for
applications that want a richer HTTP client or a JSON-Schema library of their
own; the library never requires it.

Usage::

    import wise
    from wise.llm import AccessPolicy, EvidenceHost, LocalAssistant, Principal, Scope, ToolGateway
    from wise.llm.ollama import OllamaConfig, OllamaProvider

    policy = AccessPolicy(Principal("reviewer-7"), scopes=frozenset({Scope.EXPLANATION}),
                          views=frozenset({"Finance"}), runs=frozenset({run_id}))
    gateway = ToolGateway(EvidenceHost(result, packet=packet), policy)
    provider = OllamaProvider(OllamaConfig(model="llama3.1:8b", allowed_models=("llama3.1:8b",)))
    review = LocalAssistant(provider, gateway).review(packet)
    print(review.report)          # deterministic, whether or not a model answered
"""

from __future__ import annotations

from .assistant import (
    ASSISTANT_SCHEMA_VERSION,
    SYSTEM_PROMPT,
    AssistedReview,
    LocalAssistant,
    ReferenceCheck,
    check_references,
    render_reviewed_report,
)
from .drafts import (
    DraftLimits,
    DraftPreview,
    NormDraft,
    ProposedExample,
    Provenance,
    Severity,
    ValidationFinding,
    check_recipe,
    check_regex,
    check_threshold_support,
    clarification_questions,
    isolated_log,
    preview_norm_draft,
    stakeholder_tradeoffs,
    validate_candidate,
)
from .gateway import (
    APPROVED_TOOLS,
    DEFAULT_ENABLED_TOOLS,
    ArgumentSpec,
    EvidenceHost,
    ToolGateway,
    ToolResult,
    ToolSpec,
)
from .policy import (
    FULL_POPULATION_AGGREGATE,
    AccessPolicy,
    ComparatorChange,
    Principal,
    Scope,
    authorised_baseline,
    authorised_population,
)
from .provider import (
    BudgetLedger,
    CallBudget,
    ChatMessage,
    ChatRequest,
    FakeProvider,
    LLMProvider,
    ModelIdentity,
    ProviderResult,
    ProviderStatus,
)
from .retrieval import (
    ApprovedDocument,
    LexicalIndex,
    RetrievalResult,
    RetrievedSpan,
    ReviewQuestion,
    untrusted_block,
)
from .schemas import (
    DRAFT_SCHEMA_VERSION,
    EXPLANATION_DRAFT_SCHEMA,
    NORM_DRAFT_SCHEMA,
    TOOL_SELECTION_SCHEMA,
    ExplanationDraft,
    Hypothesis,
    SchemaViolation,
    ToolCall,
    ToolSelection,
)
from .transport import (
    DEFAULT_ENDPOINT,
    StrictLocalTransport,
    TransportConfig,
    check_endpoint,
    is_loopback,
)

__all__ = [
    "APPROVED_TOOLS",
    "ASSISTANT_SCHEMA_VERSION",
    "DEFAULT_ENABLED_TOOLS",
    "DEFAULT_ENDPOINT",
    "DRAFT_SCHEMA_VERSION",
    "EXPLANATION_DRAFT_SCHEMA",
    "FULL_POPULATION_AGGREGATE",
    "NORM_DRAFT_SCHEMA",
    "SYSTEM_PROMPT",
    "TOOL_SELECTION_SCHEMA",
    "AccessPolicy",
    "ApprovedDocument",
    "ArgumentSpec",
    "AssistedReview",
    "BudgetLedger",
    "CallBudget",
    "ChatMessage",
    "ChatRequest",
    "ComparatorChange",
    "DraftLimits",
    "DraftPreview",
    "EvidenceHost",
    "ExplanationDraft",
    "FakeProvider",
    "Hypothesis",
    "LLMProvider",
    "LexicalIndex",
    "LocalAssistant",
    "ModelIdentity",
    "NormDraft",
    "Principal",
    "ProposedExample",
    "Provenance",
    "ProviderResult",
    "ProviderStatus",
    "ReferenceCheck",
    "RetrievalResult",
    "RetrievedSpan",
    "ReviewQuestion",
    "SchemaViolation",
    "Scope",
    "Severity",
    "StrictLocalTransport",
    "ToolCall",
    "ToolGateway",
    "ToolResult",
    "ToolSelection",
    "ToolSpec",
    "TransportConfig",
    "ValidationFinding",
    "authorised_baseline",
    "authorised_population",
    "check_endpoint",
    "check_recipe",
    "check_references",
    "check_regex",
    "check_threshold_support",
    "clarification_questions",
    "is_loopback",
    "isolated_log",
    "preview_norm_draft",
    "render_reviewed_report",
    "stakeholder_tradeoffs",
    "untrusted_block",
    "validate_candidate",
]
