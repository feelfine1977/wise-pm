"""The typed shapes a model may return, and a strict reader for them.

Three payloads cross the boundary from a model into this library:
:data:`TOOL_SELECTION_SCHEMA` (which read-only tools to run),
:data:`EXPLANATION_DRAFT_SCHEMA` (which authoritative facts to present, in
what order, with which questions and unverified hypotheses) and
:data:`NORM_DRAFT_SCHEMA` (a proposed configuration change, for review). Each
is a JSON Schema, sent to the server as the structural constraint on its reply
*and* re-checked here afterwards, because a server that honours a schema is a
convenience and never the guarantee.

The reader is a small validator over the subset of JSON Schema these payloads
use. It is written out rather than pulled in so that ``import wise`` gains no
dependency and the base install can validate a draft. It refuses unknown keys
(``additionalProperties: false`` is not decoration here), it enforces
``const`` and ``enum`` exactly, and it bounds every string and array — a reply
that is structurally valid but a megabyte long is still a reply this library
will not hold.

Passing the schema is the *first* check, never the last. Structure says
nothing about reference: :mod:`wise.llm.assistant` then checks that every fact
and evidence id a draft names actually exists in the authorised packet, and
:mod:`wise.llm.drafts` deeply validates a candidate norm before anything
touches it. Valid JSON with invented references is the failure mode this
split exists to catch.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..errors import DraftError, LLMError
from .provider import CallBudget

#: The interchange version of these payloads, matching the roadmap contracts.
DRAFT_SCHEMA_VERSION = "0.1-proposal"

#: Version of the prompts these schemas accompany. Recorded in the run record
#: so a later reader knows which instructions produced a draft.
PROMPT_VERSION = "wise-llm-prompt/1"

#: The status a hypothesis must carry. There is no other admissible value:
#: a model does not get to mark its own guess verified.
UNVERIFIED = "unverified_hypothesis"

#: The status a norm draft must carry. A draft cannot approve itself.
PENDING_REVIEW = "pending_human_review"

_STRING = {"type": "string", "minLength": 1}

TOOL_SELECTION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Read-only tools to run before drafting",
    "type": "object",
    "properties": {
        "schema_version": {"const": DRAFT_SCHEMA_VERSION},
        "calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": _STRING,
                    "arguments": {"type": "object"},
                    "reason": {"type": "string"},
                },
                "required": ["tool", "arguments"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["schema_version", "calls"],
    "additionalProperties": False,
}

EXPLANATION_DRAFT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Proposed LLM explanation selection; facts rendered deterministically",
    "type": "object",
    "properties": {
        "schema_version": {"const": DRAFT_SCHEMA_VERSION},
        "packet_id": _STRING,
        "fact_order": {"type": "array", "items": _STRING, "minItems": 1, "uniqueItems": True},
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": _STRING,
                    "evidence_refs": {"type": "array", "items": _STRING, "minItems": 1, "uniqueItems": True},
                    "status": {"const": UNVERIFIED},
                },
                "required": ["text", "evidence_refs", "status"],
                "additionalProperties": False,
            },
        },
        "questions": {"type": "array", "items": _STRING, "uniqueItems": True},
        "limitations": {"type": "array", "items": _STRING, "uniqueItems": True},
    },
    "required": ["schema_version", "packet_id", "fact_order", "hypotheses", "questions", "limitations"],
    "additionalProperties": False,
}

NORM_DRAFT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Proposed draft envelope; candidate_norm requires the actual WISE loader",
    "type": "object",
    "properties": {
        "schema_version": {"const": DRAFT_SCHEMA_VERSION},
        "parent_norm_hash": _STRING,
        "review_status": {"const": PENDING_REVIEW},
        "candidate_norm": {"type": ["object", "null"]},
        "source_refs": {"type": "array", "items": _STRING, "uniqueItems": True},
        "assumptions": {"type": "array", "items": _STRING, "uniqueItems": True},
        "unresolved_questions": {"type": "array", "items": _STRING, "uniqueItems": True},
        "proposed_examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "example_id": _STRING,
                    "description": _STRING,
                    "expected_result": {"enum": ["satisfied", "violated", "not_evaluable", "out_of_scope", "unknown"]},
                },
                "required": ["example_id", "description", "expected_result"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "schema_version",
        "parent_norm_hash",
        "review_status",
        "candidate_norm",
        "source_refs",
        "assumptions",
        "unresolved_questions",
        "proposed_examples",
    ],
    "additionalProperties": False,
}

#: The results a proposed example may declare, from the roadmap contract.
EXPECTED_RESULTS = ("satisfied", "violated", "not_evaluable", "out_of_scope", "unknown")


# ------------------------------------------------------------------ validator
class SchemaViolation(DraftError):
    """A payload does not match the schema it was required to match."""


def validate_against(schema: Mapping[str, Any], data: Any, *, path: str = "$") -> None:
    """Check ``data`` against the subset of JSON Schema these payloads use.

    Supports ``type`` (including a list of types), ``const``, ``enum``,
    ``properties``, ``required``, ``additionalProperties: false``, ``items``,
    ``minItems``, ``maxItems``, ``uniqueItems``, ``minLength`` and
    ``maxLength``. Anything else in a schema is ignored rather than silently
    treated as satisfied — which is why the schemas in this module use only
    what is listed here.

    >>> validate_against({"type": "object", "properties": {"a": {"type": "string"}},
    ...                   "required": ["a"], "additionalProperties": False}, {"a": "x"})
    >>> validate_against({"type": "object", "additionalProperties": False}, {"a": 1})
    Traceback (most recent call last):
        ...
    wise.llm.schemas.SchemaViolation: $: unknown key 'a'
    """
    if "const" in schema and data != schema["const"]:
        raise SchemaViolation(f"{path}: expected {schema['const']!r}, got {data!r}")
    if "enum" in schema and data not in schema["enum"]:
        raise SchemaViolation(f"{path}: {data!r} is not one of {list(schema['enum'])}")
    declared = schema.get("type")
    if declared is not None:
        wanted = [declared] if isinstance(declared, str) else list(declared)
        if not any(_is_type(data, t) for t in wanted):
            raise SchemaViolation(f"{path}: expected {'/'.join(wanted)}, got {_type_name(data)}")
    if isinstance(data, str):
        if "minLength" in schema and len(data) < int(schema["minLength"]):
            raise SchemaViolation(f"{path}: an empty value is not admissible here")
        if "maxLength" in schema and len(data) > int(schema["maxLength"]):
            raise SchemaViolation(f"{path}: {len(data)} characters exceeds the limit of {schema['maxLength']}")
    if isinstance(data, list):
        if "minItems" in schema and len(data) < int(schema["minItems"]):
            raise SchemaViolation(f"{path}: needs at least {schema['minItems']} entries, got {len(data)}")
        if "maxItems" in schema and len(data) > int(schema["maxItems"]):
            raise SchemaViolation(f"{path}: {len(data)} entries exceeds the limit of {schema['maxItems']}")
        if schema.get("uniqueItems") and len(_unique(data)) != len(data):
            raise SchemaViolation(f"{path}: entries must be unique")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for i, item in enumerate(data):
                validate_against(item_schema, item, path=f"{path}[{i}]")
    if isinstance(data, Mapping):
        properties = schema.get("properties") or {}
        for name in schema.get("required", ()):
            if name not in data:
                raise SchemaViolation(f"{path}: missing required key {name!r}")
        if schema.get("additionalProperties") is False:
            for key in data:
                if key not in properties:
                    raise SchemaViolation(f"{path}: unknown key {key!r}")
        for key, sub in properties.items():
            if key in data and isinstance(sub, Mapping):
                validate_against(sub, data[key], path=f"{path}.{key}")


def _unique(values: Sequence[Any]) -> list[Any]:
    seen: list[Any] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _is_type(data: Any, name: str) -> bool:
    if name == "object":
        return isinstance(data, Mapping)
    if name == "array":
        return isinstance(data, list)
    if name == "string":
        return isinstance(data, str)
    if name == "boolean":
        return isinstance(data, bool)
    if name == "integer":
        return isinstance(data, int) and not isinstance(data, bool)
    if name == "number":
        return isinstance(data, int | float) and not isinstance(data, bool)
    if name == "null":
        return data is None
    raise LLMError(f"schema declares an unsupported type {name!r}")  # pragma: no cover - defensive


def _type_name(data: Any) -> str:
    if isinstance(data, Mapping):
        return "object"
    if isinstance(data, bool):
        return "boolean"
    if isinstance(data, list):
        return "array"
    if data is None:
        return "null"
    return type(data).__name__


_FENCES = ("```json", "```JSON", "```")


def parse_json_object(text: str, *, max_bytes: int) -> dict[str, Any]:
    """Read a model reply as one bounded JSON object.

    A single surrounding code fence is stripped — that is a deterministic
    normalisation of a common formatting habit, not a tolerance for prose —
    and anything else is refused. Non-object payloads, trailing content and
    over-long replies are all :class:`SchemaViolation`, because a reply this
    library cannot read exactly is a reply it will not read at all.

    >>> parse_json_object('{"a": 1}', max_bytes=100)
    {'a': 1}
    >>> parse_json_object('```json\\n{"a": 1}\\n```', max_bytes=100)
    {'a': 1}
    >>> parse_json_object("not json", max_bytes=100)
    Traceback (most recent call last):
        ...
    wise.llm.schemas.SchemaViolation: the reply is not valid JSON: Expecting value: line 1 column 1 (char 0)
    """
    encoded = len(text.encode("utf-8"))
    if encoded > max_bytes:
        raise SchemaViolation(f"the reply is {encoded} bytes, past the limit of {max_bytes}")
    stripped = text.strip()
    for fence in _FENCES:
        if stripped.startswith(fence) and stripped.endswith("```"):
            stripped = stripped[len(fence) : -3].strip()
            break
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise SchemaViolation(f"the reply is not valid JSON: {exc}") from exc
    if not isinstance(data, Mapping):
        raise SchemaViolation(f"the reply must be a JSON object, got {_type_name(data)}")
    return dict(data)


# --------------------------------------------------------------- typed reads
@dataclass(frozen=True)
class ToolCall:
    """One requested tool call, as the model asked for it — not yet permitted."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "arguments": dict(self.arguments), "reason": self.reason}


@dataclass(frozen=True)
class ToolSelection:
    """The model's requested read-only calls, still to be validated by the host."""

    calls: tuple[ToolCall, ...] = ()
    schema_version: str = DRAFT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "calls", tuple(self.calls))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "calls": [c.to_dict() for c in self.calls]}


@dataclass(frozen=True)
class Hypothesis:
    """An unverified suggestion, tied to evidence and labelled as unproven.

    Valid evidence ids do not make a sentence true. The status is a constant
    for exactly that reason.
    """

    text: str
    evidence_refs: tuple[str, ...]
    status: str = UNVERIFIED

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_refs", tuple(str(r) for r in self.evidence_refs))
        if self.status != UNVERIFIED:
            raise SchemaViolation(f"a hypothesis is always {UNVERIFIED!r}; got {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "evidence_refs": list(self.evidence_refs), "status": self.status}


@dataclass(frozen=True)
class ExplanationDraft:
    """What a model proposed about an explanation packet — order and prose only.

    It selects and orders authoritative facts and adds questions, limitations
    and unverified hypotheses. It carries no values: the numbers are the
    packet's, inserted by the deterministic renderer.
    """

    packet_id: str
    fact_order: tuple[str, ...]
    hypotheses: tuple[Hypothesis, ...] = ()
    questions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    schema_version: str = DRAFT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "fact_order", tuple(str(f) for f in self.fact_order))
        object.__setattr__(self, "hypotheses", tuple(self.hypotheses))
        object.__setattr__(self, "questions", tuple(str(q) for q in self.questions))
        object.__setattr__(self, "limitations", tuple(str(limitation) for limitation in self.limitations))

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        seen: list[str] = []
        for hypothesis in self.hypotheses:
            for ref in hypothesis.evidence_refs:
                if ref not in seen:
                    seen.append(ref)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "packet_id": self.packet_id,
            "fact_order": list(self.fact_order),
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "questions": list(self.questions),
            "limitations": list(self.limitations),
        }


def read_tool_selection(text: str, budget: CallBudget) -> ToolSelection:
    """Parse and validate a tool-selection reply. Nothing is dispatched here.

    >>> from wise.llm.provider import CallBudget
    >>> reply = '{"schema_version": "0.1-proposal", "calls": [{"tool": "get_run_summary", "arguments": {}}]}'
    >>> read_tool_selection(reply, CallBudget()).calls[0].tool
    'get_run_summary'
    """
    data = parse_json_object(text, max_bytes=budget.max_response_bytes)
    validate_against(TOOL_SELECTION_SCHEMA, data)
    calls = list(data["calls"])
    if len(calls) > budget.max_tool_calls:
        raise SchemaViolation(f"the reply requests {len(calls)} tool calls, past the budget of {budget.max_tool_calls}")
    return ToolSelection(
        calls=tuple(
            ToolCall(tool=str(c["tool"]), arguments=dict(c["arguments"]), reason=str(c.get("reason", ""))) for c in calls
        ),
        schema_version=str(data["schema_version"]),
    )


def read_explanation_draft(text: str, budget: CallBudget) -> ExplanationDraft:
    """Parse and validate an explanation draft. References are checked later.

    Structure only: whether the facts and evidence it names exist is decided
    against the authorised packet, in :mod:`wise.llm.assistant`.

    >>> from wise.llm.provider import CallBudget
    >>> reply = ('{"schema_version": "0.1-proposal", "packet_id": "p", "fact_order": ["OBS-001"], '
    ...          '"hypotheses": [], "questions": [], "limitations": []}')
    >>> read_explanation_draft(reply, CallBudget()).fact_order
    ('OBS-001',)
    """
    data = parse_json_object(text, max_bytes=budget.max_response_bytes)
    validate_against(EXPLANATION_DRAFT_SCHEMA, data)
    _bound(data["fact_order"], budget.max_facts, "fact_order")
    _bound(data["hypotheses"], budget.max_hypotheses, "hypotheses")
    _bound(data["questions"], budget.max_questions, "questions")
    _bound(data["limitations"], budget.max_questions, "limitations")
    for name in ("questions", "limitations"):
        for item in data[name]:
            _bound_text(item, budget.max_text_bytes, name)
    hypotheses = []
    for h in data["hypotheses"]:
        _bound_text(h["text"], budget.max_text_bytes, "hypothesis")
        _bound(h["evidence_refs"], budget.max_witnesses, "evidence_refs")
        hypotheses.append(Hypothesis(text=str(h["text"]), evidence_refs=tuple(h["evidence_refs"]), status=str(h["status"])))
    return ExplanationDraft(
        packet_id=str(data["packet_id"]),
        fact_order=tuple(data["fact_order"]),
        hypotheses=tuple(hypotheses),
        questions=tuple(data["questions"]),
        limitations=tuple(data["limitations"]),
        schema_version=str(data["schema_version"]),
    )


def _bound(values: Sequence[Any], limit: int, what: str) -> None:
    if len(values) > limit:
        raise SchemaViolation(f"{what}: {len(values)} entries exceed the budget of {limit}")


def _bound_text(value: Any, limit: int, what: str) -> None:
    encoded = len(str(value).encode("utf-8"))
    if encoded > limit:
        raise SchemaViolation(f"{what}: {encoded} bytes exceed the budget of {limit}")


__all__ = [
    "DRAFT_SCHEMA_VERSION",
    "EXPECTED_RESULTS",
    "EXPLANATION_DRAFT_SCHEMA",
    "NORM_DRAFT_SCHEMA",
    "PENDING_REVIEW",
    "PROMPT_VERSION",
    "TOOL_SELECTION_SCHEMA",
    "UNVERIFIED",
    "ExplanationDraft",
    "Hypothesis",
    "SchemaViolation",
    "ToolCall",
    "ToolSelection",
    "parse_json_object",
    "read_explanation_draft",
    "read_tool_selection",
    "validate_against",
]
