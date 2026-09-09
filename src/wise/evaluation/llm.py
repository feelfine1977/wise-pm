"""An offline harness for the local-assistance boundary.

What it scores is the **boundary**, not a model. A scenario is recorded
material — a reply a model once produced, a document somebody wrote, a stale
fingerprint — plus the outcome the library declares for it. The runner puts
that material through the real gateway, the real access policy, the real draft
reader and the real evidence packet, and reports, per scenario, *what was
asked*, *what happened*, and *whether that is the declared behaviour*.

The distinction matters. A benchmark that asks "did the model refuse?" measures
the model. This asks "did the boundary refuse?", and the answer does not change
when the model does. Every guarantee under test is structural: the tool
allowlist is frozen at import, permissions are an argument the application
supplies, references are checked against the packet that exists, and a stale
fingerprint is a conflict rather than a licence.

**Offline by construction.** :class:`~wise.llm.provider.FakeProvider` replays
the recorded replies; nothing here opens a socket, starts a server or pulls a
model. A scenario that would need one declares ``requires_env``, and is skipped
with that variable named in the reason — it never silently passes and never
quietly contacts anything. Running such a scenario needs *both* the variable
and a ``live_provider=`` the caller supplies: this module constructs no
provider that can reach a network.

**The families**, which are the error taxonomy this boundary has:

============================== =================================================
family                         what the boundary must do
============================== =================================================
``unauthorised_population``    refuse an explanation of a slice the principal
                               holds no unit of, before any fact is served
``relabelled_comparator``      serve the slice, re-derive the comparator over
                               the authorised population, and withhold the
                               facts computed against the replaced one
``authorised_population``      serve the packet whole to a principal who may
                               see the whole run — the negative control
``unknown_tool``               refuse a name that is not in the frozen set,
                               before any argument is validated
``invented_reference``         refuse a draft naming a fact this packet does
                               not have, and keep the deterministic report
``poisoned_document``          admit the text as data, flag it, and change no
                               permission and no tool authority
``oversize_reply``             fall back to the deterministic report with the
                               status recorded
``stale_norm_fingerprint``     refuse to preview a draft written against
                               another norm
``stale_snapshot_fingerprint`` refuse to materialise a witness from a log that
                               changed after the evidence was captured
``live_model``                 nothing: skipped, with its variable named
============================== =================================================

Usage::

    from wise.evaluation.llm import load_tasks, run_tasks

    report = run_tasks(load_tasks("tests/fixtures/llm_tasks"))
    print(report.render())
    assert report.ok

>>> from wise.evaluation.llm import BoundaryMaterial, Outcome
>>> material = BoundaryMaterial.running_example()
>>> material.slice_label, material.view
('B', 'Finance')
>>> Outcome.REFUSED.value
'refused'
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..errors import DraftConflict, LLMError, StaleEvidenceError, WiseError

if TYPE_CHECKING:  # pragma: no cover
    from ..evidence.models import EvidencePacket
    from ..explain.priority import ExplanationPacket
    from ..llm.policy import AccessPolicy
    from ..llm.provider import LLMProvider
    from ..log import EventLog
    from ..norm import Norm
    from ..scoring import ScoreResult

#: Version of the recorded-task contract read by :func:`load_tasks`.
TASK_SCHEMA_VERSION = "wise-llm-task/1"

#: Version of the harness report.
HARNESS_SCHEMA_VERSION = "wise-llm-harness/1"


class Outcome(str, Enum):
    """What the boundary did with a scenario. Five distinguishable answers."""

    #: a typed refusal reached the caller and nothing was served
    REFUSED = "refused"
    #: the request was answered in full
    SERVED = "served"
    #: answered, but against a re-derived comparator, with the facts computed
    #: against the replaced one withheld and named
    SERVED_RELABELLED = "served_relabelled"
    #: untrusted content was admitted as data, flagged, and granted nothing
    CONTAINED = "contained"
    #: the model contributed nothing and the deterministic report stands
    DETERMINISTIC_ONLY = "deterministic_only"


class Status(str, Enum):
    """Whether what happened is what the library declares."""

    AS_DECLARED = "as_declared"
    NOT_AS_DECLARED = "not_as_declared"
    SKIPPED = "skipped"


# --------------------------------------------------------------- recorded tasks
@dataclass(frozen=True)
class RecordedTask:
    """One scenario: recorded material, and the outcome declared for it.

    Nothing in a task is executed. ``replies`` are strings a provider replays,
    ``documents`` are constructor arguments for
    :class:`~wise.llm.retrieval.ApprovedDocument`, and ``expected_markers`` /
    ``forbidden_markers`` are substrings that must, and must not, appear in the
    serialised transcript of what happened. Finite numeric markers match whole
    number tokens with absolute tolerance 1e-12 (zero relative tolerance), in
    both JSON values and rendered text. The same rule checks required and
    forbidden numbers, so a platform-specific last bit cannot hide a leaked
    population mean. Other markers remain literal substrings.
    """

    task_id: str
    family: str
    ask: str
    expected_outcome: str
    replies: tuple[str, ...] = ()
    documents: tuple[dict[str, Any], ...] = ()
    policy: str = "whole_run"
    retrieval_query: str = ""
    parent_norm_hash: str = ""
    expected_markers: tuple[str, ...] = ()
    forbidden_markers: tuple[str, ...] = ()
    requires_env: str | None = None
    notes: str = ""
    schema_version: str = TASK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "replies", tuple(str(r) for r in self.replies))
        object.__setattr__(self, "documents", tuple(dict(d) for d in self.documents))
        object.__setattr__(self, "expected_markers", tuple(str(m) for m in self.expected_markers))
        object.__setattr__(self, "forbidden_markers", tuple(str(m) for m in self.forbidden_markers))
        if self.schema_version != TASK_SCHEMA_VERSION:
            raise LLMError(f"task {self.task_id!r}: schema {self.schema_version!r} is not {TASK_SCHEMA_VERSION!r}")
        if self.family not in HANDLERS:
            raise LLMError(f"task {self.task_id!r}: unknown family {self.family!r}; this harness covers {sorted(HANDLERS)}")
        Outcome(self.expected_outcome)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RecordedTask:
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise LLMError(f"task {payload.get('task_id')!r}: unknown key(s) {unknown}")
        return cls(**{k: v for k, v in payload.items()})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "family": self.family,
            "ask": self.ask,
            "expected_outcome": self.expected_outcome,
            "policy": self.policy,
            "replies": list(self.replies),
            "documents": [dict(d) for d in self.documents],
            "retrieval_query": self.retrieval_query,
            "parent_norm_hash": self.parent_norm_hash,
            "expected_markers": list(self.expected_markers),
            "forbidden_markers": list(self.forbidden_markers),
            "requires_env": self.requires_env,
            "notes": self.notes,
        }


def load_tasks(directory: str | Path) -> tuple[RecordedTask, ...]:
    """Read every ``*.json`` scenario in a directory, in file-name order.

    The files are data and are parsed strictly: an unknown key, an unknown
    family or an unknown expected outcome is refused rather than ignored, so a
    scenario cannot quietly become a no-op.
    """
    root = Path(directory)
    if not root.is_dir():
        raise LLMError(f"{root} is not a directory of recorded tasks")
    tasks: list[RecordedTask] = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LLMError(f"{path}: not a JSON task ({exc.msg})") from exc
        if not isinstance(payload, Mapping):
            raise LLMError(f"{path}: a recorded task is a JSON object, got {type(payload).__name__}")
        tasks.append(RecordedTask.from_dict(payload))
    if not tasks:
        raise LLMError(f"{root} holds no *.json tasks")
    return tuple(tasks)


# ------------------------------------------------------------------- material
@dataclass(frozen=True)
class BoundaryMaterial:
    """The deterministic artefacts every scenario is run against.

    One scored run of the shipped running example, its evidence, one
    explanation packet, and four access policies that differ only in the rows
    they grant. Building it contacts nothing and reads no file outside the
    package.
    """

    result: ScoreResult
    packet: ExplanationPacket
    evidence: EvidencePacket
    log: EventLog
    norm: Norm
    view: str = "Finance"
    slice_label: str = "B"
    grouping: str = "company"

    @classmethod
    def running_example(cls, *, view: str = "Finance", group: str = "B") -> BoundaryMaterial:
        """The shipped P2P example, scored with evidence and explained once."""
        import wise

        log = wise.datasets.running_p2p_log()
        norm = wise.datasets.running_p2p_norm()
        result = wise.score(log, norm, evidence="summary")
        if result.evidence is None:  # pragma: no cover - score always captures here
            raise LLMError("the harness needs a run with evidence; score(..., evidence='summary')")
        packet = wise.explain_priority(result, "company", group, view=view, gamma=1.0, evidence=result.evidence)
        return cls(result=result, packet=packet, evidence=result.evidence, log=log, norm=norm, view=view, slice_label=group)

    @property
    def run_id(self) -> str:
        return "" if self.result.manifest is None else self.result.manifest.run_id

    def policy(self, name: str) -> AccessPolicy:
        """One of four principals, differing only in the rows they hold.

        ``whole_run`` sees every row; ``slice_only`` sees every unit of the
        explained slice and nothing else; ``other_slice`` sees a different
        company; ``nothing`` is the default, empty policy.
        """
        from ..llm.policy import AccessPolicy, Principal, Scope

        every = frozenset(s.value for s in Scope)
        common: dict[str, Any] = {
            "scopes": every,
            "runs": frozenset({self.run_id}),
            "views": frozenset(self.result.views),
            "columns": frozenset(str(c) for c in self.result.cases.columns),
            "document_tags": frozenset({"public"}),
        }
        if name == "whole_run":
            return AccessPolicy(Principal("harness-whole-run"), unrestricted_rows=True, label=name, **common)
        if name == "slice_only":
            return AccessPolicy(
                Principal("harness-slice-only"),
                row_filters={self.grouping: frozenset({self.slice_label})},
                label=name,
                **common,
            )
        if name == "other_slice":
            other = sorted(set(self.result.cases[self.grouping].astype(str)) - {self.slice_label})
            return AccessPolicy(
                Principal("harness-other-slice"),
                row_filters={self.grouping: frozenset(other)},
                label=name,
                **common,
            )
        if name == "nothing":
            return AccessPolicy(Principal("harness-nothing"), label=name)
        raise LLMError(f"unknown harness policy {name!r}; this material holds whole_run, slice_only, other_slice, nothing")

    def index(self, documents: Sequence[Mapping[str, Any]]) -> Any:
        from ..llm.retrieval import ApprovedDocument, LexicalIndex

        return LexicalIndex([ApprovedDocument(**dict(d)) for d in documents])


# ------------------------------------------------------------------- outcomes
@dataclass(frozen=True)
class ScenarioOutcome:
    """One scenario's line in the report: asked, happened, and the verdict."""

    task_id: str
    family: str
    ask: str
    expected: str
    observed: str
    happened: str
    status: str
    detail: str = ""
    skip_reason: str = ""

    @property
    def as_declared(self) -> bool:
        return self.status == Status.AS_DECLARED.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "asked": self.ask,
            "expected": self.expected,
            "observed": self.observed,
            "happened": self.happened,
            "status": self.status,
            "detail": self.detail,
            "skip_reason": self.skip_reason,
        }


@dataclass(frozen=True)
class HarnessReport:
    """Every scenario's outcome, and whether the whole run is as declared."""

    outcomes: tuple[ScenarioOutcome, ...] = ()
    schema_version: str = HARNESS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcomes", tuple(self.outcomes))

    def __len__(self) -> int:
        return len(self.outcomes)

    def __repr__(self) -> str:
        counts = self.counts()
        return (
            f"HarnessReport({len(self.outcomes)} scenarios: {counts['as_declared']} as declared, "
            f"{counts['not_as_declared']} not, {counts['skipped']} skipped)"
        )

    @property
    def ok(self) -> bool:
        """True when nothing behaved differently from the declared behaviour.

        A skipped scenario is not a pass and not a failure: it is a scenario
        that did not run, and it is counted separately.
        """
        return not any(o.status == Status.NOT_AS_DECLARED.value for o in self.outcomes)

    def counts(self) -> dict[str, int]:
        out = {status.value: 0 for status in Status}
        for outcome in self.outcomes:
            out[outcome.status] += 1
        return out

    def of(self, task_id: str) -> ScenarioOutcome:
        for outcome in self.outcomes:
            if outcome.task_id == task_id:
                return outcome
        raise LLMError(f"no scenario {task_id!r} in this report")

    def families(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(o.family for o in self.outcomes))

    def failures(self) -> tuple[ScenarioOutcome, ...]:
        return tuple(o for o in self.outcomes if o.status == Status.NOT_AS_DECLARED.value)

    def skipped(self) -> tuple[ScenarioOutcome, ...]:
        return tuple(o for o in self.outcomes if o.status == Status.SKIPPED.value)

    def table(self) -> pd.DataFrame:
        frame = pd.DataFrame([o.to_dict() for o in self.outcomes])
        return frame.set_index("task_id") if len(frame) else frame

    def render(self) -> str:
        """The per-scenario report: what was asked, what happened, the verdict."""
        counts = self.counts()
        lines = [
            "Local-assistance boundary harness (recorded material only; no model was contacted)",
            "=" * 96,
        ]
        for outcome in self.outcomes:
            lines.append(f"[{outcome.status}] {outcome.task_id}  ({outcome.family})")
            lines.append(f"    asked    : {outcome.ask}")
            if outcome.status == Status.SKIPPED.value:
                lines.append(f"    skipped  : {outcome.skip_reason}")
            else:
                lines.append(f"    happened : {outcome.happened}")
                lines.append(f"    declared : {outcome.expected}    observed: {outcome.observed}")
            if outcome.detail:
                lines.append(f"    detail   : {outcome.detail}")
        lines.append("=" * 96)
        lines.append(
            f"{counts['as_declared']} as declared, {counts['not_as_declared']} not as declared, {counts['skipped']} skipped"
        )
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ok": self.ok,
            "counts": self.counts(),
            "families": list(self.families()),
            "scenarios": [o.to_dict() for o in self.outcomes],
        }


# ------------------------------------------------------------------- handlers
@dataclass(frozen=True)
class _Run:
    """What a handler observed: the outcome, one sentence, and the transcript."""

    outcome: Outcome
    happened: str
    transcript: str = ""


def _provider(task: RecordedTask, live: LLMProvider | None) -> LLMProvider:
    from ..llm.provider import FakeProvider

    return live if live is not None else FakeProvider(list(task.replies))


def _review(
    task: RecordedTask,
    material: BoundaryMaterial,
    live: LLMProvider | None,
    *,
    index: Any = None,
    retrieval: Any = None,
) -> tuple[Any, Any]:
    from ..llm.assistant import LocalAssistant
    from ..llm.gateway import EvidenceHost, ToolGateway

    host = EvidenceHost(material.result, packet=material.packet, evidence=material.evidence, index=index)
    gateway = ToolGateway(host, material.policy(task.policy))
    assistant = LocalAssistant(_provider(task, live), gateway)
    return gateway, assistant.review(material.packet, retrieval=retrieval)


def _tool(review: Any, name: str) -> Any:
    for outcome in review.tool_results:
        if outcome.tool == name:
            return outcome
    return None


def _explanation_access(task: RecordedTask, material: BoundaryMaterial, live: LLMProvider | None) -> _Run:
    """R1's three answers: refused, relabelled, or served whole."""
    _, review = _review(task, material, live)
    called = _tool(review, "explain_priority")
    transcript = json.dumps(review.to_dict(), default=str)
    if called is None:
        return _Run(Outcome.DETERMINISTIC_ONLY, "the recorded reply asked for no explanation at all", transcript)
    if not called.ok:
        return _Run(Outcome.REFUSED, f"explain_priority refused: {called.reason}", transcript)
    payload = called.payload
    withheld = list(payload.get("withheld_facts", []))
    if payload["comparator"].get("changed"):
        return _Run(
            Outcome.SERVED_RELABELLED,
            f"served over the authorised population with the comparator re-derived and {len(withheld)} "
            "relative-priority fact(s) withheld and named",
            transcript,
        )
    return _Run(
        Outcome.SERVED,
        f"served whole: {len(payload['facts'])} facts against the packet's own comparator",
        transcript,
    )


def _unknown_tool(task: RecordedTask, material: BoundaryMaterial, live: LLMProvider | None) -> _Run:
    _, review = _review(task, material, live)
    transcript = json.dumps(review.to_dict(), default=str)
    refused = [t for t in review.tool_results if not t.ok and "unknown tool" in t.reason]
    if not refused:
        served = [t.tool for t in review.tool_results if t.ok]
        return _Run(Outcome.SERVED, f"the gateway dispatched {served}", transcript)
    names = ", ".join(t.tool for t in refused)
    return _Run(Outcome.REFUSED, f"the gateway refused {names} before any argument was validated", transcript)


def _invented_reference(task: RecordedTask, material: BoundaryMaterial, live: LLMProvider | None) -> _Run:
    _, review = _review(task, material, live)
    transcript = json.dumps(review.to_dict(), default=str)
    check = review.reference_check
    if check.ok and review.draft is not None:
        return _Run(Outcome.SERVED, "the draft was accepted and every identifier it named exists", transcript)
    unknown = list(check.unknown_facts) + list(check.unknown_evidence)
    return _Run(
        Outcome.REFUSED,
        f"the draft was refused for {len(unknown)} identifier(s) that name nothing ({', '.join(unknown[:4])}); "
        "the deterministic report was rendered anyway",
        transcript,
    )


def _poisoned_document(task: RecordedTask, material: BoundaryMaterial, live: LLMProvider | None) -> _Run:
    from ..llm.gateway import APPROVED_TOOLS

    index = material.index(task.documents)
    policy = material.policy(task.policy)
    before_tools = sorted(APPROVED_TOOLS)
    before_policy = policy.fingerprint()
    found = index.search(task.retrieval_query, policy=policy, limit=8)
    _, review = _review(task, material, live, index=index, retrieval=found)
    transcript = json.dumps({"review": review.to_dict(), "retrieval": found.to_dict()}, default=str)

    flagged = list(found.flagged_spans)
    granted = [t.tool for t in review.tool_results if t.ok and t.tool not in APPROVED_TOOLS]
    refused = [t.tool for t in review.tool_results if not t.ok]
    unchanged = sorted(APPROVED_TOOLS) == before_tools and policy.fingerprint() == before_policy
    if not flagged:
        return _Run(Outcome.SERVED, "the injected span was retrieved but not flagged for the reviewer", transcript)
    if granted or not unchanged:
        return _Run(
            Outcome.SERVED, f"the document changed something: dispatched {granted}, policy stable={unchanged}", transcript
        )
    return _Run(
        Outcome.CONTAINED,
        f"the text was retrieved as data and flagged ({', '.join(flagged)}); the approved tool set and the policy "
        f"fingerprint are unchanged, and the tool it named was refused ({', '.join(refused) or 'none was asked for'})",
        transcript,
    )


def _oversize_reply(task: RecordedTask, material: BoundaryMaterial, live: LLMProvider | None) -> _Run:
    _, review = _review(task, material, live)
    transcript = json.dumps(review.to_dict(), default=str)
    statuses = [str(record.get("status")) for record in review.provider_records]
    if review.draft is not None:
        return _Run(Outcome.SERVED, f"the oversize reply was accepted anyway (statuses {statuses})", transcript)
    return _Run(
        Outcome.DETERMINISTIC_ONLY,
        f"the reply was rejected as {statuses[-1] if statuses else 'no reply'} and the deterministic report stands "
        f"({len(review.report.splitlines())} lines, {len(material.packet.limitations)} mandatory qualifications)",
        transcript,
    )


def _stale_norm_fingerprint(task: RecordedTask, material: BoundaryMaterial, live: LLMProvider | None) -> _Run:
    from ..llm.drafts import NormDraft, preview_norm_draft

    draft = NormDraft(parent_norm_hash=task.parent_norm_hash or "0" * 64, candidate_norm=None, draft_id=task.task_id)
    try:
        preview = preview_norm_draft(draft, material.norm, material.log)
    except DraftConflict as exc:
        return _Run(Outcome.REFUSED, f"the preview refused the stale parent: {exc}", str(exc))
    return _Run(Outcome.SERVED, f"the preview ran against a norm the draft was not written for: {preview.draft_id}", "")


def _stale_snapshot_fingerprint(task: RecordedTask, material: BoundaryMaterial, live: LLMProvider | None) -> _Run:
    """This scenario needs a log it may tamper with, so it builds its own run.

    Mutating the shared material would leave every later scenario running
    against a log that had been edited underneath it.
    """
    own = BoundaryMaterial.running_example(view=material.view, group=material.slice_label)
    record = next(r for r in own.evidence.records if r.evaluable)
    before = len(own.evidence.witnesses(record.evaluation_id))
    own.log.events.loc[0, "activity"] = "Tampered By The Harness"
    try:
        own.evidence.witnesses(record.evaluation_id)
    except StaleEvidenceError as exc:
        return _Run(
            Outcome.REFUSED,
            f"{before} witness(es) were available before the log changed; afterwards the packet refused: {exc}",
            str(exc),
        )
    return _Run(Outcome.SERVED, "the packet answered from a log that had changed since the score", "")


def _live_model(task: RecordedTask, material: BoundaryMaterial, live: LLMProvider | None) -> _Run:  # pragma: no cover
    """Only reachable when the caller supplied both the variable and a provider."""
    _, review = _review(task, material, live)
    return _Run(
        Outcome.SERVED if review.draft is not None else Outcome.DETERMINISTIC_ONLY,
        f"a caller-supplied provider answered with status {review.status!r}",
        json.dumps(review.to_dict(), default=str),
    )


Handler = Callable[[RecordedTask, BoundaryMaterial, "LLMProvider | None"], _Run]

#: family -> the handler that exercises it. Written out, like the gateway's own
#: registry: a family is not looked up by ``getattr`` on a name from a file.
HANDLERS: dict[str, Handler] = {
    "unauthorised_population": _explanation_access,
    "relabelled_comparator": _explanation_access,
    "authorised_population": _explanation_access,
    "unknown_tool": _unknown_tool,
    "invented_reference": _invented_reference,
    "poisoned_document": _poisoned_document,
    "oversize_reply": _oversize_reply,
    "stale_norm_fingerprint": _stale_norm_fingerprint,
    "stale_snapshot_fingerprint": _stale_snapshot_fingerprint,
    "live_model": _live_model,
}


# --------------------------------------------------------------------- runner
_NUMBER_TOKEN = re.compile(r"(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])")


def _marker_present(marker: str, transcript: str) -> bool:
    """Compare numbers as numbers, including when a report renders them as text."""
    if _NUMBER_TOKEN.fullmatch(marker):
        expected = float(marker)
        if math.isfinite(expected):
            return any(
                math.isclose(float(match.group()), expected, rel_tol=0.0, abs_tol=1e-12)
                for match in _NUMBER_TOKEN.finditer(transcript)
            )
    return marker in transcript


def _markers(task: RecordedTask, transcript: str) -> str:
    """Which declared markers are missing, and which forbidden ones appeared."""
    missing = [m for m in task.expected_markers if not _marker_present(m, transcript)]
    # Retain every literal leak the original check caught; also catch numeric variants.
    leaked = [m for m in task.forbidden_markers if m in transcript or _marker_present(m, transcript)]
    parts = []
    if missing:
        parts.append(f"expected marker(s) absent from what happened: {missing}")
    if leaked:
        parts.append(f"forbidden marker(s) present in what was served: {leaked}")
    return "; ".join(parts)


def _skip_reason(task: RecordedTask, env: Mapping[str, str], live: LLMProvider | None) -> str | None:
    """Why a scenario does not run — always naming the variable that gates it."""
    if not task.requires_env:
        return None
    name = task.requires_env
    value = str(env.get(name, ""))
    enabled = value not in ("", "0", "false", "False")
    if not enabled:
        return f"needs a model: {name} is not set (it is {value!r}), and this harness contacts nothing on its own"
    if live is None:
        return (
            f"needs a model: {name} is set to {value!r}, but no live_provider= was supplied. This module "
            "constructs no provider that can reach a server, so the scenario still does not run"
        )
    return None


def run_tasks(
    tasks: Iterable[RecordedTask],
    *,
    material: BoundaryMaterial | None = None,
    env: Mapping[str, str] | None = None,
    live_provider: LLMProvider | None = None,
) -> HarnessReport:
    """Run every scenario against the boundary and report what happened.

    ``material`` defaults to :meth:`BoundaryMaterial.running_example`. ``env``
    defaults to the process environment and is injectable so a test can drive
    the skip path without touching it. ``live_provider`` is the only way a
    scenario can reach a model, and the caller has to build it.
    """
    artefacts = material if material is not None else BoundaryMaterial.running_example()
    environment = os.environ if env is None else env
    outcomes: list[ScenarioOutcome] = []
    for task in tasks:
        skip = _skip_reason(task, environment, live_provider)
        if skip is not None:
            outcomes.append(
                ScenarioOutcome(
                    task_id=task.task_id,
                    family=task.family,
                    ask=task.ask,
                    expected=task.expected_outcome,
                    observed="",
                    happened="",
                    status=Status.SKIPPED.value,
                    skip_reason=skip,
                )
            )
            continue
        handler = HANDLERS[task.family]
        try:
            run = handler(task, artefacts, live_provider)
        except WiseError as exc:
            run = _Run(Outcome.REFUSED, f"{type(exc).__name__}: {exc}", f"{type(exc).__name__}: {exc}")
        detail = _markers(task, run.transcript or run.happened)
        matched = run.outcome.value == task.expected_outcome and not detail
        outcomes.append(
            ScenarioOutcome(
                task_id=task.task_id,
                family=task.family,
                ask=task.ask,
                expected=task.expected_outcome,
                observed=run.outcome.value,
                happened=run.happened,
                status=Status.AS_DECLARED.value if matched else Status.NOT_AS_DECLARED.value,
                detail=detail,
            )
        )
    return HarnessReport(outcomes=tuple(outcomes))


__all__ = [
    "HANDLERS",
    "HARNESS_SCHEMA_VERSION",
    "TASK_SCHEMA_VERSION",
    "BoundaryMaterial",
    "HarnessReport",
    "Outcome",
    "RecordedTask",
    "ScenarioOutcome",
    "Status",
    "load_tasks",
    "run_tasks",
]
