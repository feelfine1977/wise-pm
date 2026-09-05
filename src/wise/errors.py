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
