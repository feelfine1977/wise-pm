"""The 0.1.0 names that changed, for callers that have not moved yet (decision record 0001, section 2).

Every name of the 0.1.0 interface whose location, type or behaviour changed in 0.2.0,
and every private name of 0.1.0 that a known consumer touches, is listed in
:data:`PATHS` with what replaces it:

``deprecated``
    The 0.1.0 form still works and emits one :class:`DeprecationWarning` per use that
    names the replacement. It returns what the replacement returns, so no number
    changes. Module-level names are served here (:func:`prioritize`,
    :func:`norm_from_dict`, :func:`_rule_attributes`); members are served by the
    object itself (``log._to_ts``, ``log._codes``, ``log.cases = ...``, a backlog used
    as a DataFrame), reachable here through the re-exported :class:`EventLog` and
    :class:`~wise.backlog.Backlog`.
``kept``
    A private name that works exactly as in 0.1.0, without a warning, because 0.2.0
    has no public replacement for it yet.
``changed``
    No compatibility path: the change is a correction from the defect ledger or a
    guarantee of the new design (an event table that cannot be changed through the
    log), and serving the 0.1.0 form would bring the defect back. The note says what
    to do instead.

The module is removed in 1.0, together with every ``deprecated`` path.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import pandas as pd

from .backlog import Backlog, legacy_frame
from .log import EventLog
from .norm import Norm
from .prioritization import prioritize as _prioritize
from .scoring import ScoreResult
from .selection import parse

__all__ = ["PATHS", "Backlog", "CompatPath", "EventLog", "ScoreResult", "norm_from_dict", "prioritize"]

Status = Literal["deprecated", "kept", "changed"]

#: The schema version 0.1.0 read a norm document without ``schema_version`` as.
LEGACY_DEFAULT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class CompatPath:
    """One 0.1.0 name and what replaces it.

    Attributes
    ----------
    status
        ``deprecated``, ``kept`` or ``changed`` (see the module documentation).
    replacement
        What to use instead; a deprecation warning of the path names it.
    served_by
        Where the 0.1.0 form is served (empty for ``changed``).
    note
        What else a caller needs to know.
    """

    status: Status
    replacement: str
    served_by: tuple[str, ...] = ()
    note: str = ""


def _index_array(name: str, replacement: str) -> CompatPath:
    return CompatPath(
        "deprecated",
        f"EventLog.case_index.{replacement}",
        (f"EventLog.{name}",),
        "a read-only view of the case index",
    )


#: Keyed by the dotted path of the 0.1.0 definition (``wise.prioritize`` is ``wise.prioritization.prioritize``).
PATHS: Mapping[str, CompatPath] = MappingProxyType(
    {
        "wise.prioritization.prioritize": CompatPath(
            "deprecated",
            "wise.prioritize(...).table and .spec",
            ("wise.compat.prioritize", "the returned Backlog used as a DataFrame"),
            "wise.prioritize returns a Backlog record; wise.compat.prioritize returns the 0.1.0 DataFrame with its attrs "
            "(view as passed); the record used as a DataFrame has the attrs of its spec (view as resolved)",
        ),
        "wise.norm.Norm.from_dict": CompatPath(
            "deprecated",
            "wise.Norm.from_dict with the document's schema_version stated",
            ("wise.compat.norm_from_dict",),
            "Norm.from_dict refuses a document without schema_version; 0.1.0 read it as schema version 2",
        ),
        "wise.norm._rule_attributes": CompatPath(
            "deprecated",
            "wise.selection.parse(rule).columns",
            ("wise.compat._rule_attributes", "wise.norm._rule_attributes"),
        ),
        "wise.log.EventLog._to_ts": CompatPath("deprecated", "EventLog.to_timestamp", ("EventLog._to_ts",)),
        "wise.log.EventLog._codes": _index_array("_codes", "case_codes"),
        "wise.log.EventLog._ts": _index_array("_ts", "timestamps"),
        "wise.log.EventLog._valid": _index_array("_valid", "valid"),
        "wise.log.EventLog._starts": _index_array("_starts", "starts"),
        "wise.log.EventLog.cases": CompatPath(
            "deprecated",
            "EventLog.add_case_attribute or EventLog.with_case_attributes",
            ("assigning EventLog.cases",),
            "reading gives a copy: a write into it (log.cases[col] = ...) does not reach the log; assign the table or "
            "use the replacement",
        ),
        "wise.scoring.ScoreResult._weights": CompatPath(
            "kept",
            "dataclasses.replace(result, log=None) for a result without its log",
            ("ScoreResult._weights", "ScoreResult(_weights=...)"),
            "the raw weight vector per view; no public replacement for building a result by hand yet",
        ),
        "wise.log.EventLog.events": CompatPath(
            "changed",
            "add the column to the event frame the log is built from",
            note="a copy since 0.2.0 (Fixed: C3): assigning it is refused and a write into it does not reach the log",
        ),
        "wise.log.EventLog.case_attributes": CompatPath(
            "changed",
            "EventLog.add_case_attribute or EventLog.with_case_attributes",
            note="a copy since 0.2.0 (Fixed: C3): assigning it is refused and a change to it does not reach the log",
        ),
        "wise.errors.NormError": CompatPath(
            "changed",
            "except wise.NormError (or wise.WiseError)",
            note="no longer a ValueError (Fixed: C11)",
        ),
        "wise.errors.NotScoredError": CompatPath(
            "changed",
            "except wise.NotScoredError (or wise.WiseError)",
            note="no longer a ValueError (Fixed: C11)",
        ),
        "wise.errors.LogSchemaError": CompatPath(
            "changed",
            "except wise.LogSchemaError (or wise.WiseError)",
            note="no longer a KeyError (Fixed: C11)",
        ),
        "wise.scoring.ScoreResult": CompatPath(
            "changed",
            "pass mode= when building a result by hand",
            note="a result built without mode takes its norm's scoring_mode instead of 'flat' (Fixed: C13)",
        ),
        "wise.scoring.ScoreResult.mode": CompatPath(
            "changed",
            "read the resolved mode after construction",
            note="annotated str | None; None resolves to the norm's scoring_mode when the result is built (Fixed: C13)",
        ),
        "wise.prioritization.compare_periods": CompatPath(
            "changed",
            "the presence column says in which periods a slice occurs",
            note="adds the column presence and leaves the change of a slice absent from one period missing instead of "
            "comparing it with zero (Fixed: C24)",
        ),
    }
)


def _deprecated(key: str, spelled: str) -> None:
    warnings.warn(
        f"{spelled} through wise.compat is deprecated and will be removed in 1.0; use {PATHS[key].replacement}",
        DeprecationWarning,
        stacklevel=3,
    )


def prioritize(
    data: ScoreResult | pd.DataFrame,
    by: str | Sequence[str],
    view: str | None = None,
    gamma: float = 0.0,
    volume: str = "cases",
    min_cases: int = 1,
    z: float | None = None,
    score_col: str = "score",
    baseline: float | None = None,
    as_index: bool = True,
) -> pd.DataFrame:
    """:func:`wise.prioritize` returning the 0.1.0 DataFrame: the record's table
    with the ranking parameters in ``DataFrame.attrs`` (``view`` as passed, as in
    0.1.0, ``gamma``, ``baseline`` = the realised global mean, ``volume``, ``by``).
    Deprecated."""
    _deprecated("wise.prioritization.prioritize", "wise.prioritize")
    backlog = _prioritize(data, by, view, gamma, volume, min_cases, z, score_col, baseline, as_index)
    frame = legacy_frame(backlog)
    frame.attrs["view"] = view
    return frame


def norm_from_dict(d: Mapping[str, Any]) -> Norm:
    """:meth:`wise.Norm.from_dict` reading a document without ``schema_version`` as
    schema version 2, as 0.1.0 did. A document that states its version is read as
    it is. Deprecated; a schema version 2 norm also announces its legacy semantics
    with a :class:`~wise.LegacySemanticsWarning`."""
    _deprecated("wise.norm.Norm.from_dict", "wise.Norm.from_dict")
    if "schema_version" not in d:
        d = {**d, "schema_version": LEGACY_DEFAULT_SCHEMA_VERSION}
    return Norm.from_dict(d)


def _rule_attributes(rule: Mapping[str, Any]) -> set[str]:
    """Case attributes an applicability rule reads. Deprecated."""
    _deprecated("wise.norm._rule_attributes", "wise.norm._rule_attributes")
    return set(parse(rule, what="applicability").columns)
