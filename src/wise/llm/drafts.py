"""Proposed norm changes: an envelope, a deep check, and a preview that cannot apply.

A draft is a *proposal*, and this module is built around the three ways a
proposal goes wrong.

**It executes something.** :func:`wise.derive.compute_recipe` supports an
``eval`` recipe that reaches :meth:`pandas.DataFrame.eval`. That is a fine
thing for a maintainer to write by hand and an unacceptable thing to accept
from generated or retrieved content. :func:`validate_candidate` refuses
``kind="eval"`` — and unknown kinds, unknown keys, unknown aggregations,
unbounded regular expressions and out-of-range resource shapes — *before*
anything is computed, and it does so without calling the loader first, because
strict JSON loading is not an execution sandbox. Trusted, explicitly authored
recipes are untouched: the rejection lives on this path only.

**It applies itself.** :func:`preview_norm_draft` scores a candidate on an
isolated copy of the log and returns a diff. It never mutates the approved
norm, the caller's log, its derived columns or its caches, and there is no
function in this module that activates anything. ``review_status`` is a
constant: a model marking a draft approved does not make it approved, and a
:class:`~wise.errors.DraftConflict` is raised when the parent fingerprint is
not the approved norm's, because a stale parent is a conflict rather than
permission to patch a different norm.

**It invents a number.** A policy that says work is done "promptly" does not
imply fourteen days. :func:`clarification_questions` turns vague language into
questions, and :func:`check_threshold_support` marks every numeric parameter
that no cited source states, so a threshold nobody agreed to is visible as an
unresolved proposal rather than as a requirement. Provenance is kept distinct
throughout: what a policy says, what an expert chose, what a measurement
showed and what is still only proposed are four different things.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from ..derive import KINDS, OPTIONAL_RECIPE_KEYS, REQUIRED_RECIPE_KEYS, UNSAFE_RECIPE_KINDS
from ..errors import DraftConflict, DraftError, UnsafeDraft
from ..norm import Norm
from ..schema import RECIPE_AGGREGATIONS, WHERE_OPERATORS, catalogue_digest, check_parameters, norm_top_level_keys
from .provider import CallBudget
from .retrieval import RetrievedSpan
from .schemas import DRAFT_SCHEMA_VERSION, NORM_DRAFT_SCHEMA, PENDING_REVIEW, SchemaViolation, parse_json_object
from .schemas import validate_against as _validate_schema

if TYPE_CHECKING:  # pragma: no cover
    from ..log import EventLog

#: Version of the draft envelope this module writes.
DRAFT_ENVELOPE_VERSION = "wise-norm-draft/1"

#: The keys a constraint entry of a candidate norm may carry.
CONSTRAINT_KEYS = ("id", "layer", "type", "params", "weight", "applicability", "description")

#: The keys this library adds to the roadmap envelope in :meth:`NormDraft.to_dict`.
#: Everything outside these and the contract's own keys is refused on read.
_LIBRARY_KEYS = ("envelope_version", "catalogue_digest", "draft_id", "validation_findings")


class Provenance(str, Enum):
    """Where a value in a draft came from. Four different kinds of authority."""

    #: Stated in an approved source, and cited.
    POLICY_DERIVED = "policy_derived"
    #: Chosen by a named person exercising judgement.
    EXPERT_CHOICE = "expert_choice"
    #: Taken from a measurement on data, with its run identity.
    EMPIRICAL_REFERENCE = "empirical_reference"
    #: Proposed and not yet settled by any of the above.
    UNRESOLVED_PROPOSAL = "unresolved_proposal"


class Severity(str, Enum):
    """How a finding bears on the decision."""

    #: The candidate cannot be used as it stands.
    BLOCKING = "blocking"
    #: Usable, but a reviewer must look.
    WARNING = "warning"
    #: Something a person has to answer before this can be approved.
    QUESTION = "question"
    #: Recorded for the reader.
    NOTE = "note"


@dataclass(frozen=True)
class ValidationFinding:
    """One thing the check noticed, with where and how much it matters."""

    code: str
    message: str
    severity: Severity = Severity.WARNING
    pointer: str = ""
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", Severity(self.severity))
        if self.provenance is not None:
            object.__setattr__(self, "provenance", Provenance(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "pointer": self.pointer,
            "provenance": None if self.provenance is None else self.provenance.value,
        }


@dataclass(frozen=True)
class ProposedExample:
    """A case a reviewer should check the candidate against."""

    example_id: str
    description: str
    expected_result: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "description": self.description,
            "expected_result": self.expected_result,
        }


@dataclass(frozen=True)
class DraftLimits:
    """Resource shapes a candidate may not exceed. Small, and declared."""

    max_bytes: int = 65_536
    max_depth: int = 12
    max_constraints: int = 200
    max_layers: int = 30
    max_views: int = 20
    max_recipes: int = 50
    max_activities: int = 50
    max_regex_chars: int = 200
    max_repetition: int = 1_000
    max_string_chars: int = 2_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_bytes": self.max_bytes,
            "max_depth": self.max_depth,
            "max_constraints": self.max_constraints,
            "max_layers": self.max_layers,
            "max_views": self.max_views,
            "max_recipes": self.max_recipes,
            "max_activities": self.max_activities,
            "max_regex_chars": self.max_regex_chars,
            "max_repetition": self.max_repetition,
            "max_string_chars": self.max_string_chars,
        }


@dataclass(frozen=True)
class NormDraft:
    """A proposed configuration change, kept entirely outside the norm schema.

    The interchange form (:meth:`to_contract_dict`) is exactly the roadmap's
    ``NormDraft`` envelope. :meth:`to_dict` adds this library's own
    ``validation_findings`` and catalogue digest, so a reader can see what the
    candidate was checked against — but none of it ever enters schema-2 norm
    JSON, and a draft is not a norm.

    >>> draft = NormDraft(parent_norm_hash="abc", candidate_norm=None)
    >>> draft.review_status
    'pending_human_review'
    >>> NormDraft(parent_norm_hash="abc", candidate_norm=None, review_status="approved")
    Traceback (most recent call last):
        ...
    wise.errors.DraftError: review_status is always 'pending_human_review'; a draft cannot approve itself
    """

    parent_norm_hash: str
    candidate_norm: dict[str, Any] | None
    source_refs: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    proposed_examples: tuple[ProposedExample, ...] = ()
    validation_findings: tuple[ValidationFinding, ...] = ()
    review_status: str = PENDING_REVIEW
    schema_version: str = DRAFT_SCHEMA_VERSION
    envelope_version: str = DRAFT_ENVELOPE_VERSION
    catalogue_digest: str = ""
    draft_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "parent_norm_hash", str(self.parent_norm_hash))
        for name in ("source_refs", "assumptions", "unresolved_questions"):
            object.__setattr__(self, name, tuple(str(v) for v in getattr(self, name)))
        object.__setattr__(self, "proposed_examples", tuple(self.proposed_examples))
        object.__setattr__(self, "validation_findings", tuple(self.validation_findings))
        if self.review_status != PENDING_REVIEW:
            raise DraftError(f"review_status is always {PENDING_REVIEW!r}; a draft cannot approve itself")
        if self.schema_version != DRAFT_SCHEMA_VERSION:
            raise DraftError(f"unsupported draft schema version {self.schema_version!r}")
        if not self.parent_norm_hash:
            raise DraftError("a draft states the fingerprint of the norm it was written against")
        if self.candidate_norm is not None and not isinstance(self.candidate_norm, Mapping):
            raise DraftError(f"candidate_norm must be an object or null, got {type(self.candidate_norm).__name__}")
        if self.candidate_norm is not None:
            object.__setattr__(self, "candidate_norm", copy.deepcopy(dict(self.candidate_norm)))
        if not self.catalogue_digest:
            object.__setattr__(self, "catalogue_digest", catalogue_digest())

    @property
    def blocking(self) -> tuple[ValidationFinding, ...]:
        return tuple(f for f in self.validation_findings if f.severity is Severity.BLOCKING)

    @property
    def usable(self) -> bool:
        """Whether a reviewer could act on this at all. Not whether they should."""
        return self.candidate_norm is not None and not self.blocking

    def with_findings(self, findings: Sequence[ValidationFinding], questions: Sequence[str] = ()) -> NormDraft:
        """A copy carrying the findings and questions a check produced."""
        merged = list(self.unresolved_questions) + [q for q in questions if q not in self.unresolved_questions]
        return NormDraft(
            parent_norm_hash=self.parent_norm_hash,
            candidate_norm=self.candidate_norm,
            source_refs=self.source_refs,
            assumptions=self.assumptions,
            unresolved_questions=tuple(merged),
            proposed_examples=self.proposed_examples,
            validation_findings=tuple(findings),
            catalogue_digest=self.catalogue_digest,
            draft_id=self.draft_id,
        )

    def to_contract_dict(self) -> dict[str, Any]:
        """Exactly the roadmap envelope — no library-specific keys."""
        return {
            "schema_version": self.schema_version,
            "parent_norm_hash": self.parent_norm_hash,
            "review_status": self.review_status,
            "candidate_norm": copy.deepcopy(self.candidate_norm),
            "source_refs": list(self.source_refs),
            "assumptions": list(self.assumptions),
            "unresolved_questions": list(self.unresolved_questions),
            "proposed_examples": [e.to_dict() for e in self.proposed_examples],
        }

    def to_dict(self) -> dict[str, Any]:
        """The contract envelope plus what this library checked it against."""
        return {
            **self.to_contract_dict(),
            "envelope_version": self.envelope_version,
            "catalogue_digest": self.catalogue_digest,
            "draft_id": self.draft_id,
            "validation_findings": [f.to_dict() for f in self.validation_findings],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NormDraft:
        """Read either form back. Anything outside both is refused, not dropped.

        A key the envelope does not define — ``approved``, a confirmed owner,
        a state a payload assigned itself — is a claim, and dropping a claim
        silently is how it stops being reviewed.

        >>> NormDraft.from_dict({"schema_version": "0.1-proposal", "parent_norm_hash": "h",
        ...                      "review_status": "pending_human_review", "candidate_norm": None,
        ...                      "source_refs": [], "assumptions": [], "unresolved_questions": [],
        ...                      "proposed_examples": []}).parent_norm_hash
        'h'
        """
        unknown = sorted(set(data) - set(NORM_DRAFT_SCHEMA["properties"]) - set(_LIBRARY_KEYS))
        if unknown:
            raise DraftError(
                f"unknown key(s) {unknown} in a draft envelope; the contract keys are "
                f"{sorted(NORM_DRAFT_SCHEMA['properties'])}. A key this envelope does not define is a claim "
                "nobody reviewed, so it is refused rather than dropped."
            )
        contract = {k: v for k, v in data.items() if k in NORM_DRAFT_SCHEMA["properties"]}
        _validate_schema(NORM_DRAFT_SCHEMA, contract)
        return cls(
            parent_norm_hash=str(contract["parent_norm_hash"]),
            candidate_norm=contract["candidate_norm"],
            source_refs=tuple(contract["source_refs"]),
            assumptions=tuple(contract["assumptions"]),
            unresolved_questions=tuple(contract["unresolved_questions"]),
            proposed_examples=tuple(ProposedExample(**e) for e in contract["proposed_examples"]),
            validation_findings=tuple(
                ValidationFinding(**{k: v for k, v in f.items() if k != "provenance"}, provenance=f.get("provenance"))
                for f in data.get("validation_findings", ())
            ),
            review_status=str(contract["review_status"]),
            schema_version=str(contract["schema_version"]),
            catalogue_digest=str(data.get("catalogue_digest", "") or ""),
            draft_id=str(data.get("draft_id", "") or ""),
        )

    @classmethod
    def from_model_payload(cls, text: str | Mapping[str, Any], budget: CallBudget | None = None) -> NormDraft:
        """Read a model's reply as a draft — never as an approval.

        Anything but ``pending_human_review`` is refused here rather than
        normalised, because a payload that tried to approve itself is a
        payload a reviewer should be told about.
        """
        limits = budget or CallBudget()
        data = text if isinstance(text, Mapping) else parse_json_object(str(text), max_bytes=limits.max_response_bytes)
        status = data.get("review_status")
        if status != PENDING_REVIEW:
            raise DraftError(
                f"the reply carries review_status {status!r}; a model does not set review state, "
                "and an imported approved=true is never sufficient authority"
            )
        return cls.from_dict(data)


# ------------------------------------------------------------- deep checking
def _depth(value: Any, level: int = 0) -> int:
    if isinstance(value, Mapping):
        return max((_depth(v, level + 1) for v in value.values()), default=level)
    if isinstance(value, list | tuple):
        return max((_depth(v, level + 1) for v in value), default=level)
    return level


def check_regex(pattern: Any, *, pointer: str, limits: DraftLimits) -> None:
    """Refuse a regular expression that is unbounded, huge or uncompilable.

    A quantified group whose body itself contains a quantifier or an
    alternation is the classic shape of catastrophic backtracking, and Python's
    engine has no timeout to fall back on, so the shape is refused rather than
    run. The rule is conservative: an unquantified group may contain whatever
    it likes, and a quantified group may contain a plain sequence.

    >>> check_regex("INV-\\\\d+", pointer="r", limits=DraftLimits())
    >>> check_regex("(?:INV|PO)-\\\\d+", pointer="r", limits=DraftLimits())
    >>> check_regex("(a+)+$", pointer="r", limits=DraftLimits())
    Traceback (most recent call last):
        ...
    wise.errors.UnsafeDraft: r: the pattern quantifies a group that itself contains a quantifier or an alternation, which can backtrack catastrophically
    """
    if not isinstance(pattern, str):
        raise UnsafeDraft(f"{pointer}: a regular expression must be a string, got {type(pattern).__name__}")
    if len(pattern) > limits.max_regex_chars:
        raise UnsafeDraft(f"{pointer}: the pattern is {len(pattern)} characters, past the limit of {limits.max_regex_chars}")
    if re.search(r"\([^)]*[+*}|][^)]*\)\s*[+*{]", pattern):
        raise UnsafeDraft(
            f"{pointer}: the pattern quantifies a group that itself contains a quantifier or an "
            "alternation, which can backtrack catastrophically"
        )
    for count in re.findall(r"\{\s*(\d+)\s*(?:,\s*(\d*)\s*)?\}", pattern):
        for number in count:
            if number and int(number) > limits.max_repetition:
                raise UnsafeDraft(f"{pointer}: a repetition count of {number} is past the limit of {limits.max_repetition}")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise UnsafeDraft(f"{pointer}: the pattern does not compile ({exc})") from exc


def check_recipe(recipe: Any, *, pointer: str, limits: DraftLimits) -> None:
    """Refuse an untrusted recipe that would evaluate, import or run away.

    Raised *before* any derivation: this is the security boundary, and
    :meth:`wise.Norm.from_dict` is not.

    >>> check_recipe({"name": "x", "kind": "eval", "expr": "1"}, pointer="r", limits=DraftLimits())
    Traceback (most recent call last):
        ...
    wise.errors.UnsafeDraft: r: recipe kind 'eval' evaluates an expression supplied by the proposal itself and is refused in the untrusted path; a trusted, explicitly authored recipe still works
    """
    if not isinstance(recipe, Mapping):
        raise UnsafeDraft(f"{pointer}: a recipe must be an object, got {type(recipe).__name__}")
    kind = recipe.get("kind")
    if kind in UNSAFE_RECIPE_KINDS:
        raise UnsafeDraft(
            f"{pointer}: recipe kind {kind!r} evaluates an expression supplied by the proposal itself and is "
            "refused in the untrusted path; a trusted, explicitly authored recipe still works"
        )
    if not isinstance(kind, str) or kind not in KINDS:
        raise UnsafeDraft(
            f"{pointer}: unknown recipe kind {kind!r}; supported: {[k for k in KINDS if k not in UNSAFE_RECIPE_KINDS]}"
        )
    name = recipe.get("name")
    if not isinstance(name, str) or not name or not name.replace("_", "").isalnum():
        raise UnsafeDraft(f"{pointer}: a derived attribute needs a plain identifier as its name, got {name!r}")
    allowed = {"name", "kind", *REQUIRED_RECIPE_KEYS[kind], *OPTIONAL_RECIPE_KEYS[kind]}
    unknown = sorted(set(recipe) - allowed)
    if unknown:
        raise UnsafeDraft(f"{pointer}: unrecognised key(s) {unknown} for a {kind!r} recipe; an unreviewed key is not accepted")
    missing = [k for k in REQUIRED_RECIPE_KEYS[kind] if k not in recipe]
    if missing:
        raise UnsafeDraft(f"{pointer}: a {kind!r} recipe needs {missing}")
    if kind == "agg" and recipe.get("agg") not in RECIPE_AGGREGATIONS:
        raise UnsafeDraft(
            f"{pointer}: unknown evaluator name {recipe.get('agg')!r}; supported aggregations: {list(RECIPE_AGGREGATIONS)}"
        )
    if kind == "quantile_scale":
        q = recipe.get("q", 0.95)
        if isinstance(q, bool) or not isinstance(q, int | float) or not (0.0 < float(q) <= 1.0):
            raise UnsafeDraft(f"{pointer}: q must be a number in (0, 1], got {q!r}")
    if kind == "cv":
        eps = recipe.get("eps", 1e-9)
        if isinstance(eps, bool) or not isinstance(eps, int | float) or float(eps) <= 0:
            raise UnsafeDraft(f"{pointer}: eps must be a positive number, got {eps!r}")
    for key in ("activities", "a", "b", "after", "before"):
        value = recipe.get(key)
        if value is None:
            continue
        labels = [value] if isinstance(value, str) else value
        if not isinstance(labels, list | tuple) or len(labels) > limits.max_activities:
            raise UnsafeDraft(f"{pointer}.{key}: expected at most {limits.max_activities} activity labels")
    where = recipe.get("where")
    if where is not None:
        if not isinstance(where, Mapping):
            raise UnsafeDraft(f"{pointer}.where: expected an object")
        unknown_where = sorted(set(where) - {"column", *WHERE_OPERATORS})
        if unknown_where:
            raise UnsafeDraft(f"{pointer}.where: unrecognised key(s) {unknown_where}")
        if "regex" in where:
            check_regex(where["regex"], pointer=f"{pointer}.where.regex", limits=limits)


def validate_candidate(
    candidate: Mapping[str, Any] | None, *, limits: DraftLimits | None = None
) -> tuple[ValidationFinding, ...]:
    """Deeply check a candidate norm before anything touches it.

    Raises :class:`~wise.errors.UnsafeDraft` for the security-class problems —
    an evaluating recipe, an unknown kind or evaluator, an unreviewed key, an
    unbounded pattern, a resource shape past its limit, a payload past its
    size or nesting bound. Everything else comes back as findings: unknown
    constraint types, parameters outside their real ranges, structures the
    loader rejects.

    A ``None`` candidate is a legitimate draft — questions and source spans
    with nothing proposed yet — and yields a single note.

    >>> [f.code for f in validate_candidate(None)]
    ['no_candidate']
    """
    limits = limits or DraftLimits()
    if candidate is None:
        return (
            ValidationFinding(
                code="no_candidate",
                message="the draft proposes no configuration; it carries questions and citations only",
                severity=Severity.NOTE,
                provenance=Provenance.UNRESOLVED_PROPOSAL,
            ),
        )
    if not isinstance(candidate, Mapping):
        raise UnsafeDraft(f"a candidate norm must be an object, got {type(candidate).__name__}")
    try:
        encoded = json.dumps(candidate, default=str)
    except (TypeError, ValueError) as exc:
        raise UnsafeDraft(f"the candidate is not JSON data ({exc}); a proposal carries data, never objects") from exc
    if len(encoded.encode("utf-8")) > limits.max_bytes:
        raise UnsafeDraft(f"the candidate is {len(encoded)} bytes, past the limit of {limits.max_bytes}")
    if _depth(candidate) > limits.max_depth:
        raise UnsafeDraft(f"the candidate nests deeper than {limits.max_depth} levels")

    findings: list[ValidationFinding] = []
    unknown_keys = sorted(set(candidate) - set(norm_top_level_keys()))
    if unknown_keys:
        raise UnsafeDraft(
            f"unsupported top-level key(s) {unknown_keys}; supported: {list(norm_top_level_keys())}. "
            "An unknown key in a proposal is an unreviewed one."
        )

    recipes = candidate.get("derived_attributes") or []
    if not isinstance(recipes, list | tuple):
        raise UnsafeDraft("derived_attributes must be a list")
    if len(recipes) > limits.max_recipes:
        raise UnsafeDraft(f"{len(recipes)} derived attributes exceed the limit of {limits.max_recipes}")
    for i, recipe in enumerate(recipes):
        check_recipe(recipe, pointer=f"derived_attributes[{i}]", limits=limits)

    for name, cap in (("constraints", limits.max_constraints), ("layers", limits.max_layers), ("views", limits.max_views)):
        entries = candidate.get(name) or []
        if not isinstance(entries, list | tuple):
            raise UnsafeDraft(f"{name} must be a list")
        if len(entries) > cap:
            raise UnsafeDraft(f"{len(entries)} {name} exceed the limit of {cap}")

    for i, entry in enumerate(candidate.get("constraints") or []):
        pointer = f"constraints[{i}]"
        if not isinstance(entry, Mapping):
            raise UnsafeDraft(f"{pointer}: a constraint must be an object")
        unknown = sorted(set(entry) - set(CONSTRAINT_KEYS))
        if unknown:
            raise UnsafeDraft(f"{pointer}: unrecognised key(s) {unknown}; supported: {list(CONSTRAINT_KEYS)}")
        params = entry.get("params") or {}
        if not isinstance(params, Mapping):
            raise UnsafeDraft(f"{pointer}.params: expected an object")
        for message in check_parameters(str(entry.get("type", "")), params):
            findings.append(
                ValidationFinding(
                    code="unsupported_parameter",
                    message=message,
                    severity=Severity.BLOCKING,
                    pointer=pointer,
                    provenance=Provenance.UNRESOLVED_PROPOSAL,
                )
            )
        applicability = entry.get("applicability")
        if applicability is not None and not isinstance(applicability, Mapping):
            raise UnsafeDraft(f"{pointer}.applicability: expected an object")

    # only now, with the untrusted shapes refused, does the trusted loader run
    try:
        Norm.from_dict(candidate)
    except Exception as exc:  # NormError and anything the loader wraps
        findings.append(
            ValidationFinding(
                code="loader_rejected",
                message=f"the norm loader rejects this candidate: {exc}",
                severity=Severity.BLOCKING,
                provenance=Provenance.UNRESOLVED_PROPOSAL,
            )
        )
    return tuple(findings)


# ------------------------------------------------------------------ language
def clarification_questions(spans: Sequence[RetrievedSpan]) -> tuple[str, ...]:
    """Questions raised by vague obligations in cited text. No number is guessed.

    >>> from wise.llm.retrieval import RetrievedSpan
    >>> span = RetrievedSpan("p1", "p1#s1", 0, 9, "Invoices are cleared promptly.", "d")
    >>> clarification_questions([span])[0].startswith("Source p1")
    True
    """
    questions: list[str] = []
    for span in spans:
        terms = span.vague_terms()
        if not terms:
            continue
        listed = ", ".join(repr(t) for t in terms)
        question = (
            f"Source {span.document_id} ({span.citation()}) states the expectation as {listed} without a quantity. "
            "What limit applies, in which unit, measured from which event, and on whose authority? "
            "No default is assumed and none is proposed."
        )
        if question not in questions:
            questions.append(question)
    return tuple(questions)


_QUANTITY_IN_TEXT = re.compile(r"\d+(?:[.,]\d+)?")


def check_threshold_support(
    candidate: Mapping[str, Any] | None,
    spans: Sequence[RetrievedSpan],
    *,
    parent: Norm | None = None,
) -> tuple[tuple[ValidationFinding, ...], tuple[str, ...]]:
    """Mark the numbers this draft *proposes* that no cited source states.

    Returns findings and questions. A number that appears in a cited span is
    :attr:`Provenance.POLICY_DERIVED`; one that does not is
    :attr:`Provenance.UNRESOLVED_PROPOSAL` — not wrong, but not agreed, and
    the draft says so instead of presenting it as a requirement.

    ``parent`` is what makes this usable rather than noise. A parameter the
    candidate carries over unchanged from the approved norm is not a new
    proposal: whatever authority approved that norm covers it, and re-raising
    it as an open question every time would bury the one number that actually
    changed. With a parent, only changed and newly introduced parameters are
    assessed, and the carried-over ones are summarised in a single note.
    Without one — a candidate written from scratch — every number is new, and
    every number is assessed.
    """
    if candidate is None:
        return (), ()
    stated: set[str] = set()
    for span in spans:
        for match in _QUANTITY_IN_TEXT.findall(span.text):
            stated.add(match.replace(",", "."))
            stated.add(str(float(match.replace(",", "."))))
    inherited: dict[str, dict[str, Any]] = {}
    if parent is not None:
        inherited = {c.id: dict(c.to_dict().get("params") or {}) for c in parent.constraints}
    findings: list[ValidationFinding] = []
    questions: list[str] = []
    carried = 0
    for i, entry in enumerate(candidate.get("constraints") or []):
        if not isinstance(entry, Mapping):
            continue
        params = entry.get("params") or {}
        if not isinstance(params, Mapping):
            continue
        previous = inherited.get(str(entry.get("id", "")), {})
        for name, value in params.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            if parent is not None and name in previous and previous[name] == value:
                carried += 1
                continue
            text = f"{value:g}"
            supported = text in stated or str(float(value)) in stated
            pointer = f"constraints[{i}].params.{name}"
            if supported:
                findings.append(
                    ValidationFinding(
                        code="threshold_cited",
                        message=f"{entry.get('id', i)}: {name}={text} appears in a cited source",
                        severity=Severity.NOTE,
                        pointer=pointer,
                        provenance=Provenance.POLICY_DERIVED,
                    )
                )
                continue
            findings.append(
                ValidationFinding(
                    code="unsupported_threshold",
                    message=(
                        f"{entry.get('id', i)}: {name}={text} is stated by no cited source. It is a proposal, "
                        "not a requirement, until somebody with the authority to set it does."
                    ),
                    severity=Severity.QUESTION,
                    pointer=pointer,
                    provenance=Provenance.UNRESOLVED_PROPOSAL,
                )
            )
            questions.append(f"Who sets {name}={text} for {entry.get('id', i)!r}, and on what basis? No source states it.")
    if carried:
        findings.append(
            ValidationFinding(
                code="threshold_carried_over",
                message=(
                    f"{carried} numeric parameter(s) are unchanged from the approved norm and are not "
                    "re-opened here; their authority is whatever approved that norm."
                ),
                severity=Severity.NOTE,
                provenance=Provenance.EXPERT_CHOICE,
            )
        )
    return tuple(findings), tuple(dict.fromkeys(questions))


# ------------------------------------------------------------------- preview
@dataclass(frozen=True)
class DraftPreview:
    """What a candidate would change — computed on a copy, applied to nothing."""

    draft_id: str
    parent_norm_hash: str
    candidate_fingerprint: str | None
    findings: tuple[ValidationFinding, ...] = ()
    questions: tuple[str, ...] = ()
    assumption_diff: dict[str, Any] = field(default_factory=dict)
    result_diff: dict[str, Any] = field(default_factory=dict)
    example_outcomes: tuple[dict[str, Any], ...] = ()
    #: Always ``False``. There is no code path in this module that sets either.
    applied: bool = False
    activated: bool = False

    @property
    def blocking(self) -> tuple[ValidationFinding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.BLOCKING)

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft_id": self.draft_id,
            "parent_norm_hash": self.parent_norm_hash,
            "candidate_fingerprint": self.candidate_fingerprint,
            "findings": [f.to_dict() for f in self.findings],
            "questions": list(self.questions),
            "assumption_diff": self.assumption_diff,
            "result_diff": self.result_diff,
            "example_outcomes": list(self.example_outcomes),
            "applied": False,
            "activated": False,
        }


def isolated_log(log: EventLog) -> EventLog:
    """A copy whose derived columns and caches are its own.

    The events frame is shared because nothing on this path writes to it; the
    case table, the attribute list and every cache are fresh, so a preview
    cannot leave a derived column, a recipe cache entry or a count cache entry
    behind in the caller's log.
    """
    clone = copy.copy(log)
    clone.cases = log.cases.copy(deep=True)
    clone.case_attributes = list(log.case_attributes)
    clone._recipe_cache = dict(log._recipe_cache)
    clone._count_cache = {}
    clone._first_cache = {}
    clone._last_cache = {}
    return clone


def assumption_diff(approved: Norm, candidate: Norm) -> dict[str, Any]:
    """What the candidate assumes that the approved norm does not, and vice versa."""
    before = {c.id: c for c in approved.constraints}
    after = {c.id: c for c in candidate.constraints}
    changed: dict[str, Any] = {}
    for cid in sorted(set(before) & set(after)):
        old, new = before[cid].to_dict(), after[cid].to_dict()
        differences = {k: [old[k], new[k]] for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)}
        if differences:
            changed[cid] = differences
    return {
        "constraints_added": sorted(set(after) - set(before)),
        "constraints_removed": sorted(set(before) - set(after)),
        "constraints_changed": changed,
        "layers_added": sorted(set(candidate.layer_ids) - set(approved.layer_ids)),
        "layers_removed": sorted(set(approved.layer_ids) - set(candidate.layer_ids)),
        "views_added": sorted(set(candidate.view_names) - set(approved.view_names)),
        "views_removed": sorted(set(approved.view_names) - set(candidate.view_names)),
        "scoring_mode": (
            None if approved.scoring_mode == candidate.scoring_mode else [approved.scoring_mode, candidate.scoring_mode]
        ),
        "derived_attributes_changed": (
            [r.get("name") for r in candidate.derived_attributes] != [r.get("name") for r in approved.derived_attributes]
        ),
    }


def preview_norm_draft(
    draft: NormDraft,
    approved: Norm,
    log: EventLog,
    *,
    spans: Sequence[RetrievedSpan] = (),
    example_units: Mapping[str, Any] | None = None,
    limits: DraftLimits | None = None,
) -> DraftPreview:
    """Check a draft, score its candidate on an isolated copy, and diff the two.

    Nothing is applied. The approved norm is not touched, the caller's log
    keeps its columns and caches, and the returned preview says ``applied`` and
    ``activated`` are false because there is no code here that could make them
    true.

    Raises :class:`~wise.errors.DraftConflict` when the draft's parent
    fingerprint is not this norm's, and :class:`~wise.errors.UnsafeDraft` when
    the candidate would evaluate something — both before any scoring.
    """
    from ..scoring import score

    fingerprint = approved.fingerprint()
    if draft.parent_norm_hash != fingerprint:
        raise DraftConflict(
            f"this draft was written against norm {draft.parent_norm_hash[:12]}… but the approved norm is "
            f"{fingerprint[:12]}…; a stale parent is a conflict to resolve, not permission to patch a different norm"
        )
    findings = list(validate_candidate(draft.candidate_norm, limits=limits))
    threshold_findings, threshold_questions = check_threshold_support(draft.candidate_norm, spans, parent=approved)
    findings.extend(threshold_findings)
    questions = list(draft.unresolved_questions)
    for question in (*clarification_questions(spans), *threshold_questions):
        if question not in questions:
            questions.append(question)

    if draft.candidate_norm is None or any(f.severity is Severity.BLOCKING for f in findings):
        return DraftPreview(
            draft_id=draft.draft_id,
            parent_norm_hash=draft.parent_norm_hash,
            candidate_fingerprint=None,
            findings=tuple(findings),
            questions=tuple(questions),
        )

    candidate = Norm.from_dict(draft.candidate_norm)
    snapshot = isolated_log(log)
    approved_result = score(isolated_log(log), approved)
    candidate_result = score(snapshot, candidate)

    shared_views = [v for v in candidate_result.views if v in approved_result.views]
    per_view: dict[str, Any] = {}
    for view in shared_views:
        before = approved_result.scores[view]
        after = candidate_result.scores[view]
        both = before.notna() & after.notna()
        per_view[view] = {
            "mean_score_approved": float(before[before.notna()].mean()) if before.notna().any() else None,
            "mean_score_candidate": float(after[after.notna()].mean()) if after.notna().any() else None,
            "n_scored_approved": int(before.notna().sum()),
            "n_scored_candidate": int(after.notna().sum()),
            "n_units_changed": int((before[both] != after[both]).sum()),
            "max_abs_change": float((before[both] - after[both]).abs().max()) if bool(both.any()) else None,
        }
    result_diff = {
        "views_compared": shared_views,
        "views_only_in_candidate": [v for v in candidate_result.views if v not in approved_result.views],
        "views_only_in_approved": [v for v in approved_result.views if v not in candidate_result.views],
        "per_view": per_view,
    }
    outcomes = _example_outcomes(draft, candidate_result, example_units)
    return DraftPreview(
        draft_id=draft.draft_id,
        parent_norm_hash=draft.parent_norm_hash,
        candidate_fingerprint=candidate.fingerprint(),
        findings=tuple(findings),
        questions=tuple(questions),
        assumption_diff={"declared": list(draft.assumptions), **assumption_diff(approved, candidate)},
        result_diff=result_diff,
        example_outcomes=outcomes,
    )


def _example_outcomes(
    draft: NormDraft, candidate_result: Any, example_units: Mapping[str, Any] | None
) -> tuple[dict[str, Any], ...]:
    """What the candidate actually does on the cases an example names.

    Without a named unit an example stays unverified: a described scenario is
    not a case, and guessing which one it means would be the invention this
    module exists to prevent.
    """
    mapping = dict(example_units or {})
    out: list[dict[str, Any]] = []
    for example in draft.proposed_examples:
        row: dict[str, Any] = {
            "example_id": example.example_id,
            "expected_result": example.expected_result,
            "description": example.description,
        }
        unit = mapping.get(example.example_id)
        if unit is None:
            row["observed"] = None
            row["status"] = "unverified"
            row["note"] = "no unit was named for this example, so nothing was evaluated against it"
            out.append(row)
            continue
        try:
            violations = candidate_result.violations.loc[unit]
            in_scope = candidate_result.in_scope.loc[unit]
        except KeyError:
            row["observed"] = None
            row["status"] = "unknown_unit"
            row["note"] = f"unit {unit!r} is not in the scored population"
            out.append(row)
            continue
        if not bool(in_scope.any()):
            observed = "out_of_scope"
        elif bool(violations.notna().any()) and float(violations.max(skipna=True)) > 0:
            observed = "violated"
        elif bool(violations.notna().any()):
            observed = "satisfied"
        else:
            observed = "not_evaluable"
        row["observed"] = observed
        row["status"] = "matches" if observed == example.expected_result else "differs"
        out.append(row)
    return tuple(out)


def stakeholder_tradeoffs(preview: DraftPreview) -> tuple[str, ...]:
    """Plain sentences naming who gains and who loses under a candidate.

    A trade-off is stated, never resolved: the point of showing every view is
    that a change which helps one stakeholder usually costs another, and that
    a model accepting the change is not an approval.
    """
    lines: list[str] = []
    for view, row in sorted(preview.result_diff.get("per_view", {}).items()):
        before, after = row.get("mean_score_approved"), row.get("mean_score_candidate")
        if before is None or after is None:
            lines.append(f"{view}: not comparable — one side has no scored units")
            continue
        direction = "unchanged" if before == after else ("higher" if after > before else "lower")
        lines.append(
            f"{view}: mean score {before:.6g} → {after:.6g} ({direction}), "
            f"{row['n_units_changed']} of {row['n_scored_candidate']} scored units move. "
            "Acceptance by a model is not approval; a person with the authority decides."
        )
    for view in preview.result_diff.get("views_only_in_candidate", []):
        lines.append(f"{view}: introduced by the candidate — no approved comparison exists for it")
    for view in preview.result_diff.get("views_only_in_approved", []):
        lines.append(f"{view}: dropped by the candidate — a stakeholder loses their perspective entirely")
    return tuple(lines)


__all__ = [
    "CONSTRAINT_KEYS",
    "DRAFT_ENVELOPE_VERSION",
    "DraftLimits",
    "DraftPreview",
    "NormDraft",
    "ProposedExample",
    "Provenance",
    "SchemaViolation",
    "Severity",
    "ValidationFinding",
    "assumption_diff",
    "check_recipe",
    "check_regex",
    "check_threshold_support",
    "clarification_questions",
    "isolated_log",
    "preview_norm_draft",
    "stakeholder_tradeoffs",
    "validate_candidate",
]
