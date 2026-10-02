"""The semantics profile a norm is evaluated under (decision 0001, sections 3 and 4).

A norm document of schema version 1 or 2 evaluates under the semantics of
0.1.0, so that the paper's numbers reproduce bit for bit; reading one emits a
:class:`LegacySemanticsWarning`. A document of schema version 3 evaluates
under the corrected semantics. The profile is a value, :class:`Semantics`,
that a norm states through ``Norm.semantics`` (from its schema version) and
that scoring hands to every evaluation as an argument; nothing reads it from
global state.

What the profile decides when a norm is evaluated:

``missing_values``
    The missing-value policy of applicability (:mod:`wise.selection`).
    ``"legacy"`` resolves a missing attribute per operator as 0.1.0 did (a
    case without the attribute is in scope under ``ne``, ``not_in`` and a
    negated rule); ``"three_valued"`` makes it unknown under every operator,
    negations included, and scoring leaves an unknown case out of the
    constraint's denominator.
``fitted_calibration``
    Whether a population-dependent recipe (``quantile_scale``) must carry a
    fitted calibration record to be computed. Without the requirement a
    recipe without a fitted record divides by the quantile of the scored
    cases, as 0.1.0 did, and the run records the realised divisor.

What schema version 3 changes in the document itself (a ``Lag`` states its
bound or ``unbounded``, no ``eval`` recipe, a calibration record on every
``quantile_scale``, the named ``missing_values`` field) is checked when the
document is read (:mod:`wise.norm_document.v3`).
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import NormError
from .selection import LEGACY, MISSING_POLICIES, THREE_VALUED, MissingPolicy


class LegacySemanticsWarning(FutureWarning):
    """A norm document of schema version 1 or 2 was read; it evaluates under the 0.1.0 semantics until 2.0."""


@dataclass(frozen=True)
class Semantics:
    """A semantics profile: the rules a norm is evaluated under."""

    #: ``"0.1.0"`` for the legacy profile, ``"corrected"`` for schema version 3.
    name: str
    #: Missing-value policy of applicability: ``"legacy"`` or ``"three_valued"``.
    missing_values: MissingPolicy
    #: A ``quantile_scale`` recipe is computed only with a fitted calibration record.
    fitted_calibration: bool

    def __post_init__(self) -> None:
        if self.missing_values not in MISSING_POLICIES:
            raise NormError(f"missing-value policy must be one of {MISSING_POLICIES}, got {self.missing_values!r}")


#: The 0.1.0 semantics of schema versions 1 and 2.
LEGACY_SEMANTICS = Semantics("0.1.0", LEGACY, fitted_calibration=False)
#: The corrected semantics of schema version 3.
CORRECTED_SEMANTICS = Semantics("corrected", THREE_VALUED, fitted_calibration=True)

_BY_VERSION = {1: LEGACY_SEMANTICS, 2: LEGACY_SEMANTICS, 3: CORRECTED_SEMANTICS}


def semantics_of(schema_version: int) -> Semantics:
    """The profile of a norm schema version."""
    try:
        return _BY_VERSION[schema_version]
    except (KeyError, TypeError):
        raise NormError(
            f"no semantics profile for norm schema_version {schema_version!r}; known: {sorted(_BY_VERSION)}"
        ) from None


__all__ = [
    "CORRECTED_SEMANTICS",
    "LEGACY_SEMANTICS",
    "LegacySemanticsWarning",
    "Semantics",
    "semantics_of",
]
