"""Evidence, run records and fitted references (opt-in, standard library only).

The deterministic core of WISE answers *what* a case scores. This package
answers *why*, *from what*, and *under which run*:

* :mod:`~wise.evidence.models` — one typed record per constraint × unit, with
  structured measurements, reason codes, witnesses and qualifications;
* :mod:`~wise.evidence.manifest` — the run record: the mode actually used, the
  views, the input identity, the preparation options, the observation policy,
  the recipes and frozen references, the environment;
* :mod:`~wise.evidence.capture` — building a packet from a scored result, and
  bounded witness access that refuses stale inputs;
* :mod:`~wise.evidence.calibration` — fitting a reference once and applying it
  without silently refitting.

Usage::

    result = wise.score(log, norm, evidence="summary")
    result.manifest.mode          # the mode actually used
    result.evidence_frame("Finance")
    result.evidence.to_json()
    wise.load_evidence(path)      # and back again, under the same invariants

Importing this package pulls in nothing beyond NumPy, pandas and the standard
library, contacts no network and reads no repository state.
"""

from __future__ import annotations

from .calibration import CalibrationRecord, apply_calibration, fit_calibration, recipe_fingerprint
from .capture import capture_evidence, coverage_report, evidence_frame, interchange_schema, load_evidence, to_interchange
from .manifest import (
    EnvironmentInfo,
    InputIdentity,
    ObservationScope,
    PriorityConfig,
    RunManifest,
    fingerprint_events,
    fingerprint_result,
    new_run_id,
)
from .models import (
    AbsenceSearch,
    Completeness,
    CoverageReport,
    DiagnosticResult,
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
)

__all__ = [
    "AbsenceSearch",
    "CalibrationRecord",
    "Completeness",
    "CoverageReport",
    "DiagnosticResult",
    "EnvironmentInfo",
    "EvaluationRecord",
    "EvidencePacket",
    "InputIdentity",
    "Measurement",
    "MeasurementKind",
    "ObservationScope",
    "PriorityConfig",
    "Qualification",
    "QualificationCode",
    "ReasonCode",
    "RunManifest",
    "SourceIdentity",
    "Truncation",
    "ViewAnnotation",
    "WitnessKind",
    "WitnessRef",
    "apply_calibration",
    "capture_evidence",
    "coverage_report",
    "evidence_frame",
    "fingerprint_events",
    "fingerprint_result",
    "fit_calibration",
    "interchange_schema",
    "load_evidence",
    "new_run_id",
    "recipe_fingerprint",
    "to_interchange",
]
