"""Native evaluation versus a projection into cases — measured, not asserted.

Flattening an object-centric log into a case log is not a neutral import. Every
projection has to pick one object type to be "the case", and then it has to do
something with the events that touch several of them. There are exactly two
honest choices and this module implements both, so the cost of each is a number:

:func:`naive_projection`
    Copy such an event into every case it touches. Every case looks complete;
    the events, and any amount they carry, are **duplicated**. This is what
    "import the OCEL and score it" usually means, and it is why one payment
    across three invoices can be reported three times.

:func:`identity_preserving_projection`
    Copy the rows too, but keep the source event id, mark the rows whose event
    is shared, and report how much was duplicated. Nothing is invented, and a
    later diagnostic can subtract the duplication — but the object-to-object
    relations are still gone, so a relational obligation still cannot be
    evaluated on it.

Both reports say what was lost. Neither can express "this invoice belongs to a
purchase order that has three items", because after the projection there is no
purchase order and no item — only a case id.

:func:`compare_representations` then puts the three answers — naive,
identity-preserving and native — beside a manually specified truth and counts
false violations, missed obligations, unsupported claims, duplicated quantity
and witness correctness, with runtime and peak memory from :func:`measure`.

A caution the handoff is explicit about: public simulated OCEL data are a
technical benchmark, not evidence of industrial benefit, and the object links
of a case-based challenge log must never be reconstructed and then called
ground truth. The truth this module compares against is the truth a human wrote
down for a small fixture.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..log import EventLog
from ..oc.model import OCEventLog

#: The three representations this module compares.
REPRESENTATIONS = ("naive_projection", "identity_preserving_projection", "native")


@dataclass(frozen=True)
class ProjectionReport:
    """What a projection did, and what it could not carry.

    ``duplicated_events`` counts source events that landed in more than one
    case; ``duplicated_rows`` how many extra rows that produced.
    ``duplicated_amount`` is the part of ``amount_attribute`` that the
    projection counts more than once — the number that turns into "three times
    the exposure" downstream.
    """

    kind: str
    case_type: str
    n_source_events: int
    n_rows: int
    n_cases: int
    duplicated_events: int
    duplicated_rows: int
    duplicated_amount: float
    total_amount: float
    unassigned_events: int
    lost_o2o: int
    lost_object_types: tuple[str, ...]
    keeps_event_identity: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "case_type": self.case_type,
            "n_source_events": self.n_source_events,
            "n_rows": self.n_rows,
            "n_cases": self.n_cases,
            "duplicated_events": self.duplicated_events,
            "duplicated_rows": self.duplicated_rows,
            "duplicated_amount": self.duplicated_amount,
            "total_amount": self.total_amount,
            "unassigned_events": self.unassigned_events,
            "lost_o2o": self.lost_o2o,
            "lost_object_types": list(self.lost_object_types),
            "keeps_event_identity": self.keeps_event_identity,
            "notes": list(self.notes),
        }


def _rows(
    log: OCEventLog,
    case_type: str,
    amount_attribute: str | None,
    qualifier: str | None,
) -> tuple[pd.DataFrame, dict[str, int], int]:
    """One row per (event, case object) pair, with the per-event case count."""
    anchors = {o.object_id for o in log.objects if o.object_type == case_type}
    if not anchors:
        raise ValueError(f"no object of type {case_type!r} in this log; a projection needs a case notion that exists")
    per_event: dict[str, list[str]] = {}
    for link in log.e2o:
        if link.object_id in anchors and (qualifier is None or link.qualifier == qualifier):
            per_event.setdefault(link.event_id, []).append(link.object_id)
    records: list[dict[str, Any]] = []
    for event in sorted(log.events, key=lambda e: (e.timestamp, e.event_id)):
        cases = sorted(set(per_event.get(event.event_id, ())))
        for case in cases:
            row: dict[str, Any] = {
                "case": case,
                "activity": event.activity,
                "time": event.timestamp,
                "event_id": event.event_id,
                "shared_event": len(cases) > 1,
                "n_cases_of_event": len(cases),
            }
            if amount_attribute is not None:
                value = event.attributes.get(amount_attribute)
                row["amount"] = float(value) if isinstance(value, int | float) and not isinstance(value, bool) else np.nan
            records.append(row)
    unassigned = sum(1 for e in log.events if not per_event.get(e.event_id))
    frame = pd.DataFrame.from_records(records) if records else pd.DataFrame(columns=["case", "activity", "time", "event_id"])
    counts = {event_id: len(set(cases)) for event_id, cases in per_event.items()}
    return frame, counts, unassigned


def _report(
    kind: str,
    log: OCEventLog,
    case_type: str,
    frame: pd.DataFrame,
    counts: Mapping[str, int],
    unassigned: int,
    amount_attribute: str | None,
    keeps_identity: bool,
    notes: Sequence[str],
) -> ProjectionReport:
    duplicated_events = sum(1 for n in counts.values() if n > 1)
    duplicated_rows = sum(n - 1 for n in counts.values() if n > 1)
    total_amount = 0.0
    duplicated_amount = 0.0
    if amount_attribute is not None:
        for event in log.events:
            value = event.attributes.get(amount_attribute)
            if isinstance(value, int | float) and not isinstance(value, bool):
                n = counts.get(event.event_id, 0)
                if n >= 1:
                    total_amount += float(value)
                    duplicated_amount += float(value) * (n - 1)
    return ProjectionReport(
        kind=kind,
        case_type=case_type,
        n_source_events=len(log.events),
        n_rows=len(frame),
        n_cases=int(frame["case"].nunique()) if len(frame) else 0,
        duplicated_events=duplicated_events,
        duplicated_rows=duplicated_rows,
        duplicated_amount=duplicated_amount,
        total_amount=total_amount,
        unassigned_events=unassigned,
        lost_o2o=len(log.o2o),
        lost_object_types=tuple(t for t in log.object_types if t != case_type),
        keeps_event_identity=keeps_identity,
        notes=tuple(notes),
    )


def naive_projection(
    log: OCEventLog,
    *,
    case_type: str,
    amount_attribute: str | None = None,
    qualifier: str | None = None,
    **kwargs: Any,
) -> tuple[EventLog, ProjectionReport]:
    """Flatten to cases, copying shared events and forgetting they were shared.

    The event log this returns has **no** event id column, which is the point:
    downstream, a payment recorded against three invoices is three payments,
    and nothing in the log can say otherwise.
    """
    frame, counts, unassigned = _rows(log, case_type, amount_attribute, qualifier)
    columns = ["case", "activity", "time"] + (["amount"] if amount_attribute is not None else [])
    events = EventLog(frame[columns].copy(), case_col="case", activity_col="activity", timestamp_col="time", **kwargs)
    notes = (
        "shared events are copied into every case and their source identity is dropped",
        "object-to-object relations are not represented at all",
    )
    return events, _report("naive_projection", log, case_type, frame, counts, unassigned, amount_attribute, False, notes)


def identity_preserving_projection(
    log: OCEventLog,
    *,
    case_type: str,
    amount_attribute: str | None = None,
    qualifier: str | None = None,
    **kwargs: Any,
) -> tuple[EventLog, ProjectionReport]:
    """Flatten to cases, but keep the source event id and mark shared rows.

    Duplication still happens — the case model has no other way to attach one
    event to two cases — but it is *visible*: ``event_id`` is a real source
    identity, ``shared_event`` marks the copies, and
    :func:`wise.diagnostics.typed_cross_case_replication` can measure it.
    """
    frame, counts, unassigned = _rows(log, case_type, amount_attribute, qualifier)
    columns = ["case", "activity", "time", "event_id", "shared_event", "n_cases_of_event"]
    if amount_attribute is not None:
        columns.append("amount")
    events = EventLog(
        frame[columns].copy(),
        case_col="case",
        activity_col="activity",
        timestamp_col="time",
        event_id_col="event_id",
        **kwargs,
    )
    notes = (
        "shared events are still copied, but their source identity survives and the copies are flagged",
        "object-to-object relations are not represented at all",
    )
    return events, _report(
        "identity_preserving_projection", log, case_type, frame, counts, unassigned, amount_attribute, True, notes
    )


# ------------------------------------------------------------------- measurement
@dataclass(frozen=True)
class Measured:
    """A value with what it cost to produce it."""

    value: Any
    seconds: float
    peak_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {"seconds": self.seconds, "peak_bytes": self.peak_bytes}


def measure(fn: Callable[[], Any]) -> Measured:
    """Run ``fn`` once, timing it and recording peak allocation.

    One run, wall clock, :mod:`tracemalloc` for the peak. It is a magnitude,
    not a benchmark: the number is honest about being a single observation.
    """
    started_here = not tracemalloc.is_tracing()
    if started_here:
        tracemalloc.start()
    before = tracemalloc.get_traced_memory()[1]
    start = time.perf_counter()
    value = fn()
    seconds = time.perf_counter() - start
    peak = tracemalloc.get_traced_memory()[1]
    if started_here:
        tracemalloc.stop()
    return Measured(value, seconds, max(0, peak - before))


# ---------------------------------------------------------------------- truth
@dataclass(frozen=True)
class Expectation:
    """What a human says is true of one unit, check by check.

    ``violations`` maps a check id to the violation that unit *should* get, or
    to ``None`` where the honest answer is "not evaluable from this data". A
    representation that produces a number where the truth says ``None`` has not
    done better; it has claimed to know something it cannot.
    """

    unit_id: str
    violations: dict[str, float | None] = field(default_factory=dict)
    amount: float | None = None
    witnesses: dict[str, tuple[str, ...]] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "violations", {str(k): (None if v is None else float(v)) for k, v in self.violations.items()})
        object.__setattr__(self, "witnesses", {str(k): tuple(str(w) for w in v) for k, v in self.witnesses.items()})


@dataclass(frozen=True)
class RepresentationScore:
    """How one representation did against the truth."""

    representation: str
    n_units: int
    n_cells: int
    false_violations: int
    missed_obligations: int
    unsupported_claims: int
    exact_matches: int
    duplicated_quantity: float
    witness_precision: float
    witness_recall: float
    seconds: float
    peak_bytes: int
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "n_units": self.n_units,
            "n_cells": self.n_cells,
            "false_violations": self.false_violations,
            "missed_obligations": self.missed_obligations,
            "unsupported_claims": self.unsupported_claims,
            "exact_matches": self.exact_matches,
            "duplicated_quantity": self.duplicated_quantity,
            "witness_precision": self.witness_precision,
            "witness_recall": self.witness_recall,
            "seconds": self.seconds,
            "peak_bytes": self.peak_bytes,
            "notes": list(self.notes),
        }


def score_representation(
    representation: str,
    observed: pd.DataFrame,
    truth: Sequence[Expectation],
    *,
    amounts: Mapping[str, float] | None = None,
    witnesses: Mapping[tuple[str, str], Sequence[str]] | None = None,
    seconds: float = float("nan"),
    peak_bytes: int = 0,
    atol: float = 1e-9,
    notes: Sequence[str] = (),
) -> RepresentationScore:
    """Compare one representation's violations with the manual truth.

    ``observed`` is units × checks with ``NaN`` for "not evaluated". Rows the
    representation could not produce at all count as missed obligations where
    the truth expects a violation, and cost nothing where it does not.

    * **false violation** — observed above the truth (including a violation
      where the truth is ``None``): the representation invented a problem.
    * **missed obligation** — the truth expects a violation and the
      representation reports none, or reports nothing at all.
    * **unsupported claim** — the truth says the answer is not knowable and the
      representation produced a number anyway, in either direction.
    """
    false_violations = missed = unsupported = exact = cells = 0
    for expectation in truth:
        for check, expected in expectation.violations.items():
            cells += 1
            seen = _cell(observed, expectation.unit_id, check)
            if expected is None:
                if seen is not None:
                    unsupported += 1
                    if seen > atol:
                        false_violations += 1
                else:
                    exact += 1
                continue
            if seen is None:
                if expected > atol:
                    missed += 1
                continue
            if abs(seen - expected) <= atol:
                exact += 1
            elif seen > expected:
                false_violations += 1
            else:
                missed += 1

    duplicated = 0.0
    if amounts is not None:
        for expectation in truth:
            if expectation.amount is not None and expectation.unit_id in amounts:
                duplicated += float(amounts[expectation.unit_id]) - expectation.amount

    tp = fp = fn = 0
    if witnesses is not None:
        for expectation in truth:
            for check, expected_ids in expectation.witnesses.items():
                seen_ids = set(witnesses.get((expectation.unit_id, check), ()))
                wanted = set(expected_ids)
                tp += len(seen_ids & wanted)
                fp += len(seen_ids - wanted)
                fn += len(wanted - seen_ids)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")

    return RepresentationScore(
        representation=representation,
        n_units=len(truth),
        n_cells=cells,
        false_violations=false_violations,
        missed_obligations=missed,
        unsupported_claims=unsupported,
        exact_matches=exact,
        duplicated_quantity=duplicated,
        witness_precision=precision,
        witness_recall=recall,
        seconds=seconds,
        peak_bytes=peak_bytes,
        notes=tuple(notes),
    )


def _cell(observed: pd.DataFrame, unit_id: str, check: str) -> float | None:
    if unit_id not in observed.index or check not in observed.columns:
        return None
    cell = observed.loc[unit_id, check]
    if isinstance(cell, pd.Series):  # a duplicated index is itself a projection artefact
        cell = cell.iloc[0]
    value = float(np.asarray(cell, dtype=float))
    return None if np.isnan(value) else value


def compare_representations(scores: Sequence[RepresentationScore]) -> pd.DataFrame:
    """The comparison table: one row per representation, sorted as given.

    The columns are the ones the handoff asks for — expected differences,
    witness correctness, duplicated amounts, missed obligations, runtime and
    peak memory — and nothing is aggregated into a single quality number,
    because the three failures are not interchangeable.
    """
    frame = pd.DataFrame([s.to_dict() for s in scores])
    return frame.set_index("representation") if len(frame) else frame


__all__ = [
    "REPRESENTATIONS",
    "Expectation",
    "Measured",
    "ProjectionReport",
    "RepresentationScore",
    "compare_representations",
    "identity_preserving_projection",
    "measure",
    "naive_projection",
    "score_representation",
]
