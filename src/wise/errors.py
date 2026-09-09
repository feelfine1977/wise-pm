"""Exception hierarchy. All library errors derive from :class:`WiseError`."""

from __future__ import annotations


class WiseError(Exception):
    """Base class for all errors raised by :mod:`wise`."""


class NormError(WiseError, ValueError):
    """A norm, view, constraint, or applicability rule is malformed."""


class LogSchemaError(WiseError, KeyError):
    """The event log lacks a required column or violates a schema assumption."""

    def __str__(self) -> str:  # KeyError would wrap the message in quotes
        return str(self.args[0]) if self.args else ""


class NotScoredError(WiseError, ValueError):
    """An aggregation was requested but no case carries a score."""


class EvidenceError(WiseError, ValueError):
    """An evidence record, run manifest or calibration record is malformed."""


class EvidenceUnavailableError(EvidenceError):
    """Evidence was requested from a result that did not capture any.

    Raised instead of rebuilding evidence from the current — possibly
    mutated — log: an approximation reconstructed after the fact is not the
    evidence of the original run.
    """


class StaleEvidenceError(EvidenceError):
    """The inputs changed after capture, so witnesses can no longer be
    materialised from them.

    Raised by :meth:`wise.log.LogSnapshot.check_fresh` when the fingerprint
    taken at capture time no longer matches the live log.
    """


class LLMError(WiseError, ValueError):
    """The base of the optional local-assistance path (:mod:`wise.llm`).

    It is deliberately *not* an :class:`EvidenceError`: nothing a language
    model says is evidence, and a failure on this path must never be mistaken
    for a defect in the deterministic record it was asked to describe.
    """


class TransportError(LLMError):
    """The strict local transport refused a request or a response.

    Raised for a non-loopback host without an explicit reviewed
    configuration, a scheme other than ``http``/``https``, a URL carrying
    credentials, and — at call time — for a redirect or an unexpected host.
    These are configuration and boundary violations, so they are loud; a
    server that is merely absent is a *status*, not an exception.
    """


class BudgetExceeded(LLMError):
    """A call, byte, retrieval or output budget was exhausted.

    Raised when a budget would be consumed past its declared limit. The
    assistant turns it into a typed status so the deterministic report still
    renders.
    """


class AccessDenied(LLMError):
    """The supplied access policy does not permit this.

    The path fails closed: a missing scope denies, it does not widen. A model
    response can never supply or alter the principal or the policy.
    """


class GatewayError(LLMError):
    """A tool request was refused before anything was dispatched.

    Unknown tool name, unknown or malformed argument, an argument outside its
    declared bound, or a result larger than the tool's declared limit.
    """


class DraftError(LLMError):
    """A model draft is malformed, or refers to something it may not."""


class DraftConflict(DraftError):
    """The draft's parent fingerprint is not the approved norm's.

    A stale parent is a conflict to report, never permission to apply a patch
    to a different norm.
    """


class UnsafeDraft(DraftError):
    """A candidate configuration would evaluate untrusted content.

    Raised *before* any derivation or scoring runs: an ``eval`` recipe, an
    unbounded regular expression, a resource shape past its declared limit, or
    an unknown evaluator name in the untrusted path.
    """


class OCError(WiseError, ValueError):
    """An object-centric log, adapter or assessment unit is malformed.

    The base of the :mod:`wise.oc` hierarchy. It is deliberately *not* an
    :class:`EvidenceError`: an object log can be wrong long before any
    evidence is captured from it.
    """


class OCValidationError(OCError):
    """An object log violates an identity, relation or history invariant.

    Raised for the failures that cannot be repaired without inventing a fact:
    two different rows claiming one identity, a relation pointing at an
    unknown event or object, a history with two values for one instant.
    Identical duplicate rows are *not* an error — they are deduplicated and
    reported.
    """


class OCInterchangeError(OCError):
    """A file does not conform to the interchange it claims, or cannot carry a log.

    Raised on read when the payload is not the declared OCEL 2.0 structure, and
    on write when exporting would silently drop information the log holds.
    """


class OCUnitError(OCError):
    """An assessment unit specification is malformed, or its context is unusable.

    Raised for an undeclared role, a path step that names an unknown object
    type or qualifier, a limit that is not positive, and for reading an
    attribute without a declared evaluation time.
    """


class OCConstraintError(OCError):
    """A native object check or its catalogue entry is malformed or unusable.

    Raised when a check names a role its unit type does not declare, when a
    policy the check needs was not declared (an ambiguous match under
    ``on_ambiguous='error'``), and when a configuration would consume one
    amount twice in one reconciliation.
    """


class AccountingError(OCError):
    """An allocation of additive quantities is not admissible.

    Raised for a negative share, for consuming one record twice in the same
    group, for allocating more than the whole of a record, and for mixing
    units without a declared conversion. An *incomplete* allocation is not an
    error: it is reported as a residual.

    :attr:`code` carries the
    :class:`~wise.oc.accounting.AccountingIssueCode` naming which rule was
    broken, so a caller can branch on the code rather than on the sentence.
    It is ``None`` for the malformed-input refusals that are not one of the
    named accounting rules — a share that is not a number, say.
    """

    def __init__(self, *args: object, code: object = None) -> None:
        super().__init__(*args)
        #: the named accounting rule, or ``None``
        self.code = code
