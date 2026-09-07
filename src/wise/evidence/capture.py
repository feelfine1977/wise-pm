"""Result-to-evidence export and bounded witness access.

Capture turns the numeric matrices of a :class:`~wise.scoring.ScoreResult`
into :class:`~wise.evidence.models.EvaluationRecord` rows without recomputing
anything: the reasons and raw measurements come from the same evaluation that
produced ``ν``. Summary capture reuses those matrices; witnesses are optional,
bounded, and materialised per record rather than by copying a trace once per
constraint.

Three rules shape this module.

**Evidence belongs to a run.** A packet keeps the snapshot it was captured
from and re-checks it before materialising a witness, so a query answered
later is either the original observation or an explicit
:class:`~wise.errors.StaleEvidenceError`.

**Bounds are visible.** When only the displayed witnesses are truncated, the
exact measurement and the total witness count stay. Truncation of a *display*
and truncation of an *evaluation* are recorded as different things.

**Absence is a search, not an event.** A record violated because nothing
happened carries an :class:`~wise.evidence.models.AbsenceSearch` witness
naming the unit, the window, the filters in force and the completeness
assumption — never an invented event id.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..constraints import Balance, Exclusion, Lag, Metric, Precedence, Presence, Singularity
from ..errors import EvidenceError
from ..log import EventLog, LogSnapshot
from ..norm import Norm, NormConstraint
from .calibration import CalibrationRecord, as_calibration_map
from .manifest import EnvironmentInfo, InputIdentity, ObservationScope, RunManifest, new_run_id
from .models import (
    AbsenceSearch,
    Completeness,
    CoverageReport,
    EvaluationRecord,
    EvidencePacket,
    Measurement,
    MeasurementKind,
    Qualification,
    QualificationCode,
    ReasonCode,
    SourceIdentity,
    Truncation,
    ViewAnnotation,
    WitnessKind,
    WitnessRef,
    to_records_frame,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..scoring import ScoreResult, _Detail

#: Default number of witnesses materialised per record.
DEFAULT_WITNESS_LIMIT = 8


# ------------------------------------------------------------------ manifests
def _content_hashed(snapshot: LogSnapshot) -> dict[str, bool]:
    """Which tables this snapshot's fingerprint actually covers, by name.

    A single ``content_hashed: true`` cannot say *what* was hashed, and a
    ``"cases"``-scope snapshot hashes only the case table: two logs whose
    invoice amounts differ share its fingerprint while their scores differ.
    The map is the honest form of the same field.
    """
    return dict(snapshot.content_hashed_tables)


def _coverage_note(snapshot: LogSnapshot) -> str:
    """One sentence naming the tables the input identity covers, and those it does not."""
    tables = _content_hashed(snapshot)
    covered = sorted(name for name, ok in tables.items() if ok)
    uncovered = sorted(name for name, ok in tables.items() if not ok)
    return (
        f"input identity (snapshot scope {snapshot.scope!r}) content-hashes "
        + ("no table" if not covered else " and ".join(f"the {name} table" for name in covered))
        + "; "
        + " and ".join(f"the {name} table" for name in uncovered)
        + " is covered by shape, columns and preparation options only, so a changed value inside an "
        "unchanged shape would not be detected there"
    )


def _observation_scope(log: EventLog, details: Mapping[str, Any] | None, q: float = 0.001) -> ObservationScope:
    explicit = log.window is not None and None not in log.window
    if explicit and log.window is not None:
        start, end = log.window
        window_source = "explicit"
        start_iso = None if start is None else str(pd.Timestamp(start).isoformat())
        end_iso = None if end is None else str(pd.Timestamp(end).isoformat())
    else:
        window_source = "derived_quantile"
        start_iso = end_iso = None
    horizons: dict[str, str] | None = None
    if details is not None:
        horizons = {}
        for cid, detail in details.items():
            if getattr(detail, "horizon", None) is not None:
                horizons[cid] = str(pd.Timestamp(detail.horizon).isoformat())
    return ObservationScope(
        window_source=window_source,
        window_start=start_iso,
        window_end=end_iso,
        quantile=q,
        missing_timestamps=str(log.options()["missing_timestamps"]),
        timezone=log.options()["timezone"],
        resolved_horizons=horizons,
    )


def resolve_calibrations(
    norm: Norm,
    calibrations: Sequence[CalibrationRecord] | Mapping[str, CalibrationRecord] | None,
    *,
    derive: bool,
) -> dict[str, CalibrationRecord]:
    """Validate ``score(calibrations=...)`` against the norm's recipes."""
    frozen = as_calibration_map(calibrations)
    if not frozen:
        return {}
    if not derive:
        raise EvidenceError("calibrations= applies frozen references while deriving; it needs derive=True")
    known = {str(r["name"]) for r in norm.derived_attributes}
    unknown = sorted(set(frozen) - known)
    if unknown:
        raise EvidenceError(f"calibrations for attributes this norm does not derive: {unknown}")
    return frozen


def build_score_manifest(
    log: EventLog,
    norm: Norm,
    *,
    views: Sequence[str],
    mode: str,
    mode_source: str,
    details: Mapping[str, Any] | None = None,
    calibrations: Mapping[str, CalibrationRecord] | None = None,
    derived_names: Sequence[str] = (),
    snapshot: LogSnapshot | None = None,
    run_id: str | None = None,
    source: str | None = None,
) -> RunManifest:
    """The run record of one :func:`wise.score` call.

    The input identity is fingerprinted at the level the run actually
    verified: a run that captured evidence content-hashes the case table *and*
    the event table, a run that did not records the cheap structural identity
    and says so. ``preprocessing["content_hashed"]`` is a per-table map, never
    a single ``true``, because "content hashed" without naming the table is the
    claim the review found overstated. Nothing here reads Git or computes the
    explicit data digest; both are separate calls.
    """
    if snapshot is None:
        snapshot = log.snapshot(scope="structure" if details is None else "events")
    notes: list[str] = []
    if derived_names:
        notes.append(
            "score(derive=True) computed "
            + ", ".join(sorted(derived_names))
            + " on the caller's log: the log was modified in place, and its case table now carries these columns"
        )
    if not snapshot.content_hashed:
        notes.append(_coverage_note(snapshot))
    return RunManifest(
        run_id=run_id or new_run_id(),
        mode=mode,
        mode_source=mode_source,
        norm_fingerprint=norm.fingerprint(),
        input=InputIdentity(
            n_events=snapshot.n_events,
            n_cases=snapshot.n_cases,
            event_columns=tuple(str(c) for c in log.events.columns),
            case_columns=tuple(str(c) for c in log.cases.columns),
            event_id_col=log.event_id_col,
            source_identity_available=snapshot.source_identity_available,
            snapshot_fingerprint=snapshot.fingerprint,
            source=source,
        ),
        observation=_observation_scope(log, details),
        preprocessing={**log.options(), "snapshot_scope": snapshot.scope, "content_hashed": _content_hashed(snapshot)},
        views=tuple(views),
        norm_name=norm.name,
        norm_version=norm.version,
        norm_default_mode=norm.scoring_mode,
        derived_recipes=tuple(dict(r) for r in norm.derived_attributes),
        calibrations=tuple(c.to_dict() for c in (calibrations or {}).values()),
        environment=EnvironmentInfo(),
        notes=tuple(notes),
    )


# ------------------------------------------------------------------ witnesses
def _filters(snapshot: LogSnapshot) -> dict[str, Any]:
    options = snapshot.options
    return {
        "lifecycle_col": options["lifecycle_col"],
        "keep_transitions": options["keep_transitions"],
        "dedupe": options["dedupe"],
        "missing_timestamps": options["missing_timestamps"],
        "keep_columns": options["keep_columns"],
        "observation_window": options["window"],
    }


def _completeness(snapshot: LogSnapshot, *, open_window: bool = False) -> Completeness:
    """What an absence in this snapshot is allowed to claim."""
    if open_window:
        return Completeness.OPEN
    options = snapshot.options
    filtered = options["lifecycle_col"] is not None or options["dedupe"] or options["missing_timestamps"] == "drop"
    return Completeness.UNKNOWN if filtered else Completeness.ASSUMED_COMPLETE


def _absence_witness(
    snapshot: LogSnapshot,
    unit_id: str,
    role: str,
    activities: Sequence[str],
    *,
    open_window: bool = False,
    since: Any = None,
    until: Any = None,
) -> WitnessRef:
    start, end = snapshot.bounds_of(unit_id)
    return WitnessRef(
        kind=WitnessKind.ABSENCE,
        role=role,
        unit_id=unit_id,
        search=AbsenceSearch(
            unit_id=unit_id,
            activities=tuple(str(a) for a in activities),
            window_start=start if since is None else str(pd.Timestamp(since).isoformat()),
            window_end=end if until is None else str(pd.Timestamp(until).isoformat()),
            completeness=_completeness(snapshot, open_window=open_window),
            n_events_searched=snapshot.n_events_of(unit_id),
            filters=_filters(snapshot),
        ),
    )


def _event_witnesses(
    snapshot: LogSnapshot,
    unit_id: str,
    role: str,
    activities: Sequence[str] | None,
    *,
    since: Any = None,
    until: Any = None,
    limit: int | None = None,
) -> tuple[list[WitnessRef], int]:
    rows, total = snapshot.event_refs(unit_id, activities, since=since, until=until, limit=limit)
    out = [
        WitnessRef(
            kind=WitnessKind.EVENT,
            role=role,
            unit_id=unit_id,
            identity=SourceIdentity.SOURCE if row["event_id"] is not None else SourceIdentity.SNAPSHOT_LOCAL,
            event_id=row["event_id"],
            reference=None if row["event_id"] is not None else row["reference"],
            activity=row["activity"],
            timestamp=row["timestamp"],
        )
        for row in rows
    ]
    return out, total


def _witnesses_for(
    snapshot: LogSnapshot,
    nc: NormConstraint,
    unit_id: str,
    record: EvaluationRecord,
    *,
    limit: int = DEFAULT_WITNESS_LIMIT,
) -> tuple[list[WitnessRef], int]:
    """Bounded witnesses for one record, and how many exist in total.

    An out-of-scope check has nothing to witness: the expectation did not
    apply, so no observation is offered in its support.
    """
    if not record.in_scope:
        return [], 0
    c = nc.constraint
    values = record.measurement
    witnesses: list[WitnessRef] = []
    total = 0

    def add_events(role: str, activities: Sequence[str], **kwargs: Any) -> None:
        nonlocal total
        found, n = _event_witnesses(snapshot, unit_id, role, activities, limit=max(limit - len(witnesses), 0), **kwargs)
        witnesses.extend(found)
        total += n

    if isinstance(c, Presence | Exclusion | Singularity):
        activities = list(c.activities())
        occurrences = values.get("occurrences")
        if occurrences is not None and occurrences.value == 0:
            witnesses.append(_absence_witness(snapshot, unit_id, "searched", list(c.activity)))
            total += 1
        else:
            add_events("occurrence", list(c.activity))
        if isinstance(c, Exclusion | Singularity) and (c.after or c.before):
            add_events("anchor", [a for a in activities if a not in set(c.activity)])
    elif isinstance(c, Lag):
        t_a = values.get("activation_timestamp")
        t_b = values.get("response_timestamp")
        if t_a is not None and t_a.value is not None:
            add_events("activation", list(c.a), since=t_a.value, until=t_a.value)
        else:
            witnesses.append(_absence_witness(snapshot, unit_id, "activation", list(c.a)))
            total += 1
        if t_b is not None and t_b.value is not None:
            add_events("response", list(c.b), since=t_b.value, until=t_b.value)
        elif t_a is not None or record.reason_code in (ReasonCode.OPEN_OBSERVATION_WINDOW, ReasonCode.SKIPPED_MISSING_RESPONSE):
            open_window = record.reason_code is ReasonCode.OPEN_OBSERVATION_WINDOW
            since = None if t_a is None or t_a.value is None else t_a.value
            witnesses.append(_absence_witness(snapshot, unit_id, "response", list(c.b), open_window=open_window, since=since))
            total += 1
    elif isinstance(c, Precedence):
        anchor = values.get("activation_timestamp")
        if anchor is not None and anchor.value is not None:
            add_events("anchor", list(c.a), since=anchor.value, until=anchor.value)
            add_events("premature", list(c.b), until=anchor.value)
        else:
            witnesses.append(_absence_witness(snapshot, unit_id, "activation", list(c.a)))
            total += 1
    elif isinstance(c, Balance):
        for role, activities, recorded in (
            ("amount_x", list(c.activities_x), values.get("total_x")),
            ("amount_y", list(c.activities_y), values.get("total_y")),
        ):
            if recorded is not None and recorded.value == 0:
                # nothing on this side: the balance rests on a declared absence,
                # not on an event that was never recorded
                witnesses.append(_absence_witness(snapshot, unit_id, role, activities))
                total += 1
            else:
                add_events(role, activities)
    elif isinstance(c, Metric):
        witnesses.append(_attribute_witness(unit_id, c.attribute))
        total += 1
    return witnesses[:limit], total


def _attribute_witness(unit_id: str, attribute: str) -> WitnessRef:
    return WitnessRef(
        kind=WitnessKind.ATTRIBUTE,
        role="attribute",
        unit_id=unit_id,
        attribute=attribute,
    )


def materialise_witnesses(
    packet: EvidencePacket,
    record: EvaluationRecord,
    *,
    limit: int | None = None,
) -> tuple[WitnessRef, ...]:
    """Materialise the witnesses of one record from the packet's snapshot.

    Raises :class:`~wise.errors.StaleEvidenceError` when the log has changed
    since capture: witnesses are evidence of *this* run or nothing.
    """
    snapshot = packet.snapshot
    if snapshot is None:  # pragma: no cover - guarded by EvidencePacket.witnesses
        raise EvidenceError("this packet kept no snapshot; witnesses cannot be materialised")
    snapshot.check_fresh()
    if packet.manifest.input.snapshot_fingerprint != snapshot.fingerprint:  # pragma: no cover - defensive
        raise EvidenceError("the packet's snapshot does not belong to its run manifest")
    norm_constraint = _constraint_of(packet, record)
    found, _total = _witnesses_for(snapshot, norm_constraint, record.unit_id, record, limit=limit or DEFAULT_WITNESS_LIMIT)
    return tuple(found)


def _constraint_of(packet: EvidencePacket, record: EvaluationRecord) -> NormConstraint:
    if packet.norm is None:
        raise EvidenceError("this packet no longer knows its norm; witnesses cannot be materialised")
    return packet.norm.get_constraint(record.constraint_id)


# ------------------------------------------------------------------- capture
def _measurement(name: str, value: Any, unit: str, kind: str, *, lower_bound: bool = False) -> Measurement:
    if kind == "timestamp":
        ts = pd.Timestamp(value) if value is not None else None
        iso = None if ts is None or pd.isna(ts) else str(ts.isoformat())
        return Measurement(name, iso, unit, MeasurementKind.TIMESTAMP)
    if value is None:
        return Measurement(name, None, unit, MeasurementKind(kind), lower_bound=lower_bound)
    number = float(value)
    # NaN is a status, not a value: it becomes an explicit null beside the reason code
    return Measurement(name, number if np.isfinite(number) else None, unit, MeasurementKind(kind), lower_bound=lower_bound)


def _qualifications_for(
    reason: ReasonCode, record_measurements: Mapping[str, Measurement], nc: NormConstraint
) -> list[Qualification]:
    out: list[Qualification] = []
    if reason is ReasonCode.OPEN_OBSERVATION_WINDOW:
        out.append(
            Qualification(
                QualificationCode.LOWER_BOUND,
                "the response has not been observed yet; the measured duration is a lower bound at the censoring horizon, "
                "not a completed duration",
            )
        )
    if reason is ReasonCode.SATISFIED_VACUOUSLY:
        out.append(
            Qualification(
                QualificationCode.VACUOUS_SATISFACTION,
                "nothing could violate this rule in this unit: it is satisfied because the regulated activity never occurred",
            )
        )
    if isinstance(nc.constraint, Lag) and nc.constraint.activation == "each":
        out.append(
            Qualification(
                QualificationCode.AVERAGED_OVER_ACTIVATIONS,
                "the violation is the mean over the unit's activations, not a single observed lag",
            )
        )
    x, y = record_measurements.get("total_x"), record_measurements.get("total_y")
    if x is not None and y is not None and x.value == 0 and y.value == 0:
        out.append(
            Qualification(
                QualificationCode.BOTH_TOTALS_ZERO,
                "both totals are zero: the balance holds because nothing was recorded on either side",
            )
        )
    return out


def capture_evidence(
    result: ScoreResult,
    *,
    details: Mapping[str, _Detail] | None = None,
    mode: str = "summary",
    snapshot: LogSnapshot | None = None,
    views: Sequence[str] | None = None,
    witness_limit: int = DEFAULT_WITNESS_LIMIT,
    max_records: int | None = None,
    units: Sequence[Any] | None = None,
    unit_type: str = "case",
) -> EvidencePacket:
    """Build an :class:`~wise.evidence.models.EvidencePacket` from a scored result.

    ``details`` are the primitives collected during scoring; :func:`wise.score`
    passes them. Called on a result scored without capture, this raises
    :class:`~wise.errors.EvidenceUnavailableError` rather than re-deriving the
    measurements from the current log.

    One record is built per unit and constraint, which costs roughly 40 µs in
    ``"summary"`` mode and 120 µs in ``"full"`` mode. On a large log capture the
    units under review rather than all of them: ``units=[...]`` restricts the
    packet to named units and ``max_records`` caps it outright. Both are
    reported in :attr:`EvidencePacket.truncation`, so a bounded packet never
    passes for a complete one.
    """
    from ..errors import EvidenceUnavailableError

    if mode not in ("summary", "full"):
        raise EvidenceError(f"capture mode must be 'summary' or 'full', got {mode!r}")
    if details is None:
        raise EvidenceUnavailableError(
            "capture_evidence needs the measurements collected while scoring; "
            "score(log, norm, evidence='summary') records them. They are not reconstructed afterwards: "
            "the log may have changed since, and a later reading is not the evidence of an earlier score."
        )
    log = result.log
    if log is None:
        raise EvidenceUnavailableError("this result carries no log, so no evidence can be captured from it")
    if snapshot is None:
        snapshot = log.snapshot(scope="events")
    if mode == "full":
        # verify once for the whole batch rather than once per witness
        snapshot.check_fresh()
    manifest = result.manifest
    if manifest is None:  # pragma: no cover - score always builds one
        manifest = build_score_manifest(
            log, result.norm, views=result.views, mode=result.mode, mode_source="norm_default", details=details, snapshot=snapshot
        )
    if manifest.input.snapshot_fingerprint != snapshot.fingerprint:
        # capturing against a snapshot the run record did not use (a manifest
        # from a run scored without capture, say): re-record the identity that
        # this evidence was actually taken from, and say that it was re-recorded
        manifest = replace(
            manifest,
            input=replace(manifest.input, snapshot_fingerprint=snapshot.fingerprint),
            preprocessing={
                **manifest.preprocessing,
                "snapshot_scope": snapshot.scope,
                "content_hashed": _content_hashed(snapshot),
            },
            notes=(*manifest.notes, f"input identity re-recorded at capture time from a {snapshot.scope!r}-scope snapshot"),
        )
    run_id = manifest.run_id
    norm = result.norm
    selected_views = list(result.views if views is None else views)

    records: list[EvaluationRecord] = []
    annotations: list[ViewAnnotation] = []
    reasons_seen: dict[str, int] = {}
    n_in_scope = n_evaluated = n_out_of_scope = 0
    witnesses_captured = witnesses_total = 0
    truncated_records = False

    index = list(result.violations.index)
    n_records_total = len(index) * len(norm.constraints)
    unit_ids = [str(u) for u in index]
    if units is None:
        rows = list(range(len(index)))
    else:
        position = {str(u): i for i, u in enumerate(index)}
        wanted = [str(u) for u in units]
        unknown = [u for u in wanted if u not in position]
        if unknown:
            raise EvidenceError(f"unknown {unit_type} ids for capture: {unknown[:5]}")
        rows = [position[u] for u in dict.fromkeys(wanted)]
    # numeric matrices, converted once: summary capture reuses them rather than
    # recomputing anything per row
    weights = {view: result.effective_weights(view).to_numpy(dtype=float) for view in selected_views}
    penalties = {view: result.penalties(view).to_numpy(dtype=float) for view in selected_views}
    scored_units = {view: result.scores[view].notna().to_numpy(dtype=bool) for view in selected_views}

    for nc in norm.constraints:
        cid = nc.id
        column = list(result.violations.columns).index(cid)
        detail = details[cid]
        scope_col = result.in_scope[cid].to_numpy(dtype=bool)
        value_col = result.violations[cid].to_numpy(dtype=float)
        reason_col = (
            detail.reasons.to_numpy()
            if detail.reasons is not None
            else np.full(len(unit_ids), ReasonCode.OBSERVED.value, dtype=object)
        )
        timestamps: dict[str, pd.Series] = {}
        numbers: dict[str, np.ndarray] = {}
        measurement_cols: list[tuple[str, str, str]] = []
        for name, series, unit, kind in detail.measurements:
            measurement_cols.append((name, unit, kind))
            if kind == "timestamp":
                timestamps[name] = series
            else:
                numbers[name] = series.to_numpy(dtype=float)
        params = dict(nc.constraint.params())
        policies = {k: v for k, v in params.items() if str(k).startswith("missing") or k in ("activation", "response", "agg")}
        for row in rows:
            unit_id = unit_ids[row]
            if max_records is not None and len(records) >= max_records:
                truncated_records = True
                break
            in_scope = bool(scope_col[row])
            value = value_col[row]
            evaluable = bool(in_scope and not np.isnan(value))
            if in_scope:
                n_in_scope += 1
                n_evaluated += int(evaluable)
            else:
                n_out_of_scope += 1
            reason = ReasonCode.OUT_OF_SCOPE if not in_scope else ReasonCode(str(reason_col[row]))
            if reason.evaluable != evaluable:  # pragma: no cover - defensive, pinned by tests
                raise EvidenceError(
                    f"{cid}/{unit_id}: reason {reason.value!r} disagrees with the violation matrix; "
                    "the detailed path and the matrix must not diverge"
                )
            measurements: list[Measurement] = []
            if in_scope:
                for name, unit, kind in measurement_cols:
                    raw = timestamps[name].iloc[row] if kind == "timestamp" else numbers[name][row]
                    measurements.append(
                        _measurement(name, raw, unit, kind, lower_bound=reason is ReasonCode.OPEN_OBSERVATION_WINDOW)
                    )
            by_name = {m.name: m for m in measurements}
            quals = _qualifications_for(reason, by_name, nc) if in_scope else []
            record = EvaluationRecord(
                run_id=run_id,
                evaluation_id=f"{run_id}:{cid}:{unit_id}",
                unit_id=unit_id,
                unit_type=unit_type,
                constraint_id=cid,
                constraint_type=nc.constraint.type,
                constraint_version=norm.version,
                in_scope=in_scope,
                evaluable=evaluable,
                reason_code=reason,
                violation=float(value) if evaluable else None,
                measurements=tuple(measurements),
                parameters=params,
                policies=policies,
                qualifications=tuple(quals),
            )
            if mode == "full":
                found, total = _witnesses_for(snapshot, nc, unit_id, record, limit=witness_limit)
                witnesses_captured += len(found)
                witnesses_total += total
                extra = list(record.qualifications)
                if total > len(found):
                    extra.append(
                        Qualification(
                            QualificationCode.WITNESSES_TRUNCATED,
                            f"{len(found)} of {total} witnesses are shown; the measurement above counts all {total}",
                        )
                    )
                record = _replace_record(record, witnesses=tuple(found), n_witnesses_total=total, qualifications=tuple(extra))
            records.append(record)
            reasons_seen[reason.value] = reasons_seen.get(reason.value, 0) + 1
            for view in selected_views:
                # an effective weight and a penalty exist only for a check that
                # was evaluated in a unit that was scored; everywhere else the
                # 0.0 in the matrix is padding, not a zero penalty
                scored = bool(scored_units[view][row])
                weighted = scored and evaluable
                weight = float(weights[view][row, column])
                penalty = float(penalties[view][row, column])
                annotations.append(
                    ViewAnnotation(
                        run_id=run_id,
                        evaluation_id=record.evaluation_id,
                        unit_id=unit_id,
                        constraint_id=cid,
                        view=view,
                        effective_weight=weight if weighted and np.isfinite(weight) else None,
                        penalty=penalty if weighted and np.isfinite(penalty) else None,
                        scored=scored,
                    )
                )
        if max_records is not None and len(records) >= max_records:
            truncated_records = True
            break

    coverage = CoverageReport(
        unit_type=unit_type,
        n_units=len(rows),
        n_checks=n_in_scope + n_out_of_scope,
        n_in_scope=n_in_scope,
        n_evaluated=n_evaluated,
        n_out_of_scope=n_out_of_scope,
        n_unevaluable=n_in_scope - n_evaluated,
        n_scored={view: int(scored_units[view][rows].sum()) for view in selected_views},
        n_unscored={view: int((~scored_units[view][rows]).sum()) for view in selected_views},
        reasons=reasons_seen,
    )
    # a record budget stops the traversal itself: the constraints below the cut
    # have no record at all, so the coverage counts are not the run's counts and
    # must not be presented as final. A units= restriction is a different thing —
    # every constraint is still evaluated on the units that were kept.
    not_reached = tuple(nc.id for nc in norm.constraints if nc.id not in {r.constraint_id for r in records})
    truncation = Truncation(
        records_captured=len(records),
        records_total=max(n_records_total, len(records)) if (truncated_records or units is not None) else len(records),
        witnesses_captured=witnesses_captured,
        witnesses_total=witnesses_total,
        evaluation_truncated=bool(truncated_records and not_reached),
    )
    qualifications = _run_qualifications(result, snapshot, coverage, truncation, not_reached=not_reached, unit_type=unit_type)
    packet = EvidencePacket(
        manifest=manifest,
        unit_type=unit_type,
        records=tuple(records),
        coverage=coverage,
        annotations=tuple(annotations),
        qualifications=tuple(qualifications),
        truncation=truncation,
        capture_mode=mode,
        snapshot=snapshot,
        norm=norm,
    )
    return packet


def _replace_record(record: EvaluationRecord, **changes: Any) -> EvaluationRecord:
    import dataclasses

    return dataclasses.replace(record, witnesses_materialised=True, **changes)


def _run_qualifications(
    result: ScoreResult,
    snapshot: LogSnapshot,
    coverage: CoverageReport,
    truncation: Truncation | None = None,
    *,
    not_reached: Sequence[str] = (),
    unit_type: str = "case",
) -> list[Qualification]:
    out = [
        Qualification(
            QualificationCode.COVERAGE_IS_NOT_CONFIDENCE,
            coverage.interpretation,
            scope="run",
        )
    ]
    if not snapshot.content_hashed:
        out.append(
            Qualification(
                QualificationCode.SNAPSHOT_NOT_CONTENT_HASHED,
                _coverage_note(snapshot),
                scope="run",
            )
        )
    if truncation is not None and truncation.records_truncated:
        out.append(
            Qualification(
                QualificationCode.RECORDS_TRUNCATED,
                f"this capture kept {truncation.records_captured} of {truncation.records_total} "
                f"constraint x {unit_type} records. The scores and the run's own numbers are complete; "
                "the evidence rows, the coverage counts and anything derived from them cover the kept records only",
                scope="run",
            )
        )
    if truncation is not None and truncation.evaluation_truncated:
        out.append(
            Qualification(
                QualificationCode.EVALUATION_TRUNCATED,
                "the record budget stopped the capture before every constraint was reached: "
                + ", ".join(str(cid) for cid in not_reached)
                + " have no evidence row at all, so the coverage counts above are not the run's coverage",
                scope="run",
            )
        )
    if not snapshot.source_identity_available:
        out.append(
            Qualification(
                QualificationCode.NO_SOURCE_EVENT_IDENTITY,
                "no event_id_col was declared: witnesses reference rows of this snapshot, "
                "and the identity of the underlying source events is unknown",
                scope="run",
            )
        )
    unscored = {view: n for view, n in coverage.n_unscored.items() if n}
    if unscored:
        out.append(
            Qualification(
                QualificationCode.UNSCORED_UNIT,
                "unscored units per view: "
                + ", ".join(f"{view}={n}" for view, n in sorted(unscored.items()))
                + "; they have no score, which is not a score of zero",
                scope="run",
            )
        )
    if result.manifest is not None:
        for note in result.manifest.notes:
            if note.startswith("score(derive=True)"):
                out.append(Qualification(QualificationCode.INPUT_MUTATED_BY_DERIVE, note, scope="run"))
    return out


# --------------------------------------------------------------------- frames
def evidence_frame(
    packet: EvidencePacket,
    *,
    view: str | None = None,
    unit_id: Any = None,
    constraint_id: str | None = None,
) -> pd.DataFrame:
    """Long-format evidence table; see :meth:`wise.ScoreResult.evidence_frame`."""
    records = packet.records_for(unit_id=unit_id, constraint_id=constraint_id)
    annotations = None
    if view is not None:
        if view not in packet.views:
            raise EvidenceError(f"unknown view {view!r}; captured: {packet.views}")
        annotations = {a.evaluation_id: a for a in packet.annotations if a.view == view}
    frame = to_records_frame(records, annotations)
    if len(frame):
        frame = frame.set_index("evaluation_id")
    return frame


# --------------------------------------------------------------- interchange
#: Measurement names preferred as the single ``raw_value`` of the roadmap's
#: interchange shape, per constraint type. The native record keeps all of them.
_PRIMARY_MEASUREMENT = ("lag", "mean_lag", "relative_mismatch", "b_before_activation", "occurrences")

INTERCHANGE_SCHEMA_VERSION = "0.1-proposal"


def _interchange_measurement(record: EvaluationRecord) -> Measurement | None:
    by_name = record.measurement
    for name in _PRIMARY_MEASUREMENT:
        if name in by_name:
            return by_name[name]
    for m in record.measurements:
        if m.kind is not MeasurementKind.TIMESTAMP:
            return m
    return None


def current_population_baseline(packet: EvidencePacket, view: str) -> dict[str, Any]:
    """The current-population comparator, computed from the packet's own annotations.

    Returns ``{"baseline_id", "kind", "score", "layer_profile", "n_units",
    "bounded"}``: the mean score of the scored units under ``view`` and their
    mean layer penalties. This is the *current population*, which is the only
    comparator stage 1 can resolve; a historical or target comparator is the
    subject of the shared baseline specification, not of an evidence packet.

    A **bounded** packet has no run population to resolve: its annotations
    cover the units the capture kept. The comparator is then the *kept
    subset's* mean, it is named ``captured-subset:…`` rather than
    ``current-population:…``, and ``n_units`` gives the denominator it was
    formed over. :func:`to_interchange` turns that into a qualification, so a
    subset's mean is never exported as the population's.
    """
    if packet.norm is None:
        raise EvidenceError("the packet no longer knows its norm, so no layer profile can be derived")
    layer_of = {c.id: c.layer for c in packet.norm.constraints}
    per_unit: dict[str, dict[str, float]] = {}
    for annotation in packet.annotations:
        if annotation.view != view or not annotation.scored:
            continue
        layers = per_unit.setdefault(annotation.unit_id, {})
        layers[layer_of[annotation.constraint_id]] = layers.get(layer_of[annotation.constraint_id], 0.0) + (
            annotation.penalty or 0.0
        )
    if not per_unit:
        raise EvidenceError(f"no scored unit under view {view!r}: supply a baseline explicitly")
    n = len(per_unit)
    profile = {layer.id: 0.0 for layer in packet.norm.layers}
    for layers in per_unit.values():
        for layer, value in layers.items():
            profile[layer] += value / n
    score = 1.0 - sum(profile.values())
    bounded = packet.truncation is not None and packet.truncation.records_truncated
    return {
        "baseline_id": f"{'captured-subset' if bounded else 'current-population'}:{view}:{packet.run_id}",
        "kind": "current_population",
        "score": min(max(score, 0.0), 1.0),
        "layer_profile": profile,
        "n_units": n,
        "bounded": bounded,
    }


def to_interchange(
    packet: EvidencePacket,
    *,
    view: str,
    baseline: Mapping[str, Any] | Any | None = None,
    commit: str | None = None,
    max_evidence: int | None = None,
    max_facts: int = 20,
) -> dict[str, Any]:
    """Export to the roadmap's illustrative ``EvidencePacket`` interchange shape.

    That contract compresses a measurement to a single ``raw_value`` and
    ``raw_unit``. This library does not: :class:`EvaluationRecord` keeps every
    measurement with its own unit, and this function selects the primary one —
    the duration of a lag, the relative mismatch of a balance — so that the
    exported shape is a *view* of the evidence rather than the evidence.

    ``commit`` is whatever the caller knows; nothing here reads Git, and the
    field is exported as ``"unknown"`` when nobody supplies it. ``data_hash``
    is the explicit digest when one was computed, otherwise the snapshot
    fingerprint, labelled as such.

    ``baseline`` is the comparator. Pass a
    :class:`~wise.explain.BaselineSpec` or a
    :class:`~wise.explain.ResolvedBaseline` — anything exposing
    ``to_interchange()`` — to export a historical or target comparator; pass
    nothing to export the current population, which is the only comparator an
    evidence packet can resolve on its own — and a **bounded** packet cannot
    resolve even that: the exported comparator is then the kept subset's mean,
    named ``captured-subset:…``, with a qualification giving its denominator.
    """
    if not packet.records:
        raise EvidenceError("an interchange packet needs at least one evidence row")
    manifest = packet.manifest
    subset_note: str | None = None
    if baseline is None:
        resolved_baseline = current_population_baseline(packet, view)
        if resolved_baseline["bounded"]:
            truncation = packet.truncation
            captured = 0 if truncation is None else truncation.records_captured
            total = 0 if truncation is None else truncation.records_total
            subset_note = (
                f"the exported comparator is the mean of the {resolved_baseline['n_units']} scored "
                f"{packet.unit_type}s this bounded capture kept ({captured} of {total} records), not the run's "
                f"population: it is labelled {resolved_baseline['baseline_id']!r} and is not the current-population "
                "reference the run itself ranked against"
            )
    elif hasattr(baseline, "to_interchange"):
        resolved_baseline = dict(baseline.to_interchange())
    else:
        resolved_baseline = dict(baseline)
    profile = resolved_baseline.get("layer_profile")
    explanation_kind = "absolute_and_reference_contrast" if profile else "absolute_only"

    records = packet.records if max_evidence is None else packet.records[:max_evidence]
    evidence: list[dict[str, Any]] = []
    for record in records:
        measurement = _interchange_measurement(record)
        evidence.append(
            {
                "evidence_id": record.evaluation_id,
                "unit_id": record.unit_id,
                "constraint_id": record.constraint_id,
                "in_scope": record.in_scope,
                "evaluable": record.evaluable,
                "violation": record.violation,
                "raw_value": None if measurement is None else measurement.value,
                "raw_unit": "not_applicable" if measurement is None else measurement.unit,
                "reason": record.reason_code.value,
                "witness_ids": list(dict.fromkeys(w.witness_id for w in record.witnesses)),
            }
        )

    facts: list[dict[str, Any]] = []
    for record in records:
        if len(facts) >= max_facts:
            break
        measurement = _interchange_measurement(record)
        if measurement is None or measurement.value is None or not record.evaluable:
            continue
        facts.append(
            {
                "fact_id": f"FACT-{len(facts) + 1:03d}",
                "value": measurement.value,
                "unit": measurement.unit,
                "description": f"{measurement.name} observed for {record.constraint_id} on {record.unit_type} {record.unit_id}",
                "evidence_refs": [record.evaluation_id],
            }
        )
    if not facts:
        first = records[0]
        facts.append(
            {
                "fact_id": "FACT-001",
                "value": None,
                "unit": "not_applicable",
                "description": f"no evaluable measurement in this packet; {first.constraint_id} carries "
                f"reason {first.reason_code.value}",
                "evidence_refs": [first.evaluation_id],
            }
        )

    qualifications = [q.message for q in packet.qualifications]
    if subset_note is not None:
        qualifications.append(subset_note)
    for record in records:
        qualifications.extend(q.message for q in record.qualifications)
    if max_evidence is not None and len(packet.records) > len(records):
        qualifications.append(
            f"the exported evidence is bounded to {len(records)} of {len(packet.records)} rows; "
            "the scores and coverage counts cover all of them"
        )
    digest = manifest.input.data_digest or f"snapshot-fingerprint:{manifest.input.snapshot_fingerprint}"
    return {
        "schema_version": INTERCHANGE_SCHEMA_VERSION,
        "run": {
            "run_id": manifest.run_id,
            "commit": commit or manifest.environment.git_commit or "unknown",
            "norm_hash": manifest.norm_fingerprint,
            "data_hash": digest,
            "scoring_mode": manifest.mode,
            "observation_policy_id": manifest.observation.policy_id,
        },
        "baseline": {
            "baseline_id": str(resolved_baseline["baseline_id"]),
            "kind": str(resolved_baseline["kind"]),
            "score": float(resolved_baseline["score"]),
            "layer_profile": None if not profile else {str(k): float(v) for k, v in profile.items()},
        },
        "explanation_kind": explanation_kind,
        "unit_type": packet.unit_type,
        "evidence": evidence,
        "facts": facts,
        "qualifications": list(dict.fromkeys(qualifications)),
    }


def coverage_report(result: ScoreResult, *, views: Sequence[str] | None = None, unit_type: str = "case") -> CoverageReport:
    """In-scope, evaluated, scored and excluded counts for a scored result.

    Computed from the result's own matrices, so it works whether or not
    evidence was captured. The counts are coverage: they say how much was
    observed, never how likely the result is to be right, and nothing here
    multiplies a score by them.
    """
    selected = list(result.views if views is None else views)
    in_scope = result.in_scope.to_numpy(dtype=bool)
    evaluated = result.violations.notna().to_numpy()
    n_in_scope = int(in_scope.sum())
    n_evaluated = int((in_scope & evaluated).sum())
    return CoverageReport(
        unit_type=unit_type,
        n_units=len(result.violations.index),
        n_checks=int(in_scope.size),
        n_in_scope=n_in_scope,
        n_evaluated=n_evaluated,
        n_out_of_scope=int(in_scope.size - n_in_scope),
        n_unevaluable=n_in_scope - n_evaluated,
        n_scored={view: int(result.scores[view].notna().sum()) for view in selected},
        n_unscored={view: int(result.scores[view].isna().sum()) for view in selected},
    )


def load_evidence(path: str | Path) -> EvidencePacket:
    """Read an evidence packet written by :meth:`EvidencePacket.to_json`.

    The counterpart of :func:`wise.explain.load_explanation` and
    :func:`wise.explain.load_baseline`, so that "distinguishable after export
    **and reload**" can be executed rather than asserted.

    The packet comes back without its snapshot and without its norm — an
    exported file has no live log — and declares that (see
    :attr:`EvidencePacket.restored`). Nothing here reads a log, and no value in
    the file is trusted over the invariants: a record whose reason code
    disagrees with its violation is refused on read.
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"{path}: not a JSON evidence packet ({exc.msg})") from exc
    if not isinstance(payload, Mapping):
        raise EvidenceError(f"{path}: an evidence packet is a JSON object, got {type(payload).__name__}")
    return EvidencePacket.from_dict(payload)


def interchange_schema() -> dict[str, Any]:
    """The shipped JSON Schema of the interchange shape produced by :func:`to_interchange`.

    Read from package data, so it is available from an installed wheel and not
    only from a source checkout. It is the roadmap's illustrative contract, not
    a new supported WISE serialisation format: the native packet is
    :meth:`~wise.evidence.models.EvidencePacket.to_json`.
    """
    from importlib import resources

    text = resources.files("wise.evidence").joinpath("schemas/evidence_packet.schema.json").read_text(encoding="utf-8")
    return dict(json.loads(text))
