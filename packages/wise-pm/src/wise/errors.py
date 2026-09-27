"""Exception hierarchy. All library errors derive from :class:`WiseError` and from nothing else.

Since 0.2.0 no library error inherits a builtin such as ``ValueError`` or ``KeyError``: an
``except KeyError`` in a caller can no longer swallow a schema error (C11), and every public
entry point raises a :class:`WiseError` subclass for a bad argument instead of leaking the
builtin (C9). Catch :class:`WiseError` for "anything the library refused".
"""

from __future__ import annotations


class WiseError(Exception):
    """Base class for all errors raised by :mod:`wise`."""


class NormError(WiseError):
    """A norm, view, constraint or applicability rule is malformed, or a function argument is invalid."""


class LogSchemaError(WiseError):
    """The event log lacks a required column, violates a schema assumption, or was built with an invalid option."""


class NotScoredError(WiseError):
    """An aggregation was requested but no case carries a score."""
