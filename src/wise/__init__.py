"""WISE — Weighted Insights for Evaluating Efficiency.

Norm-based, slice-first prioritisation of process deviations from event logs
(Jessen, Fahland, Zerbato: *WISE: Actionable Norm-Based Scoring for Process
Mining*).

Typical use::

    import wise

    log = wise.EventLog(df, case_col="case", activity_col="activity",
                        timestamp_col="time", case_attributes=["company", "flow_type"])
    norm = wise.Norm.load("norm.json")
    result = wise.score(log, norm)                        # Phase 2
    backlog = wise.prioritize(result, by=["company"],     # Phase 3
                              view="Finance", gamma=20)
"""

from . import datasets, evidence, explain
from ._version import __version__
from .constraints import (
    CONSTRAINT_TYPES,
    Balance,
    Constraint,
    Exclusion,
    Lag,
    Metric,
    Precedence,
    Presence,
    Singularity,
    as_labels,
    constraint_from_dict,
    sat,
)
from .datasets import running_p2p_events, running_p2p_log, running_p2p_norm
from .diagnostics import (
    cross_case_replication,
    event_replication,
    gap_retained,
    left_truncated,
    observation_window,
    right_censored,
    timestamp_outliers,
    typed_cross_case_replication,
    typed_event_replication,
    typed_right_censored,
    validation_table,
)
from .errors import (
    EvidenceError,
    EvidenceUnavailableError,
    LogSchemaError,
    NormError,
    NotScoredError,
    StaleEvidenceError,
    WiseError,
)
from .evidence import (
    CalibrationRecord,
    EvaluationRecord,
    EvidencePacket,
    ReasonCode,
    RunManifest,
    apply_calibration,
    capture_evidence,
    coverage_report,
    fit_calibration,
    load_evidence,
)
from .explain import (
    BaselineError,
    BaselineKind,
    BaselineSpec,
    ExplanationPacket,
    explain_priority,
    load_baseline,
    load_explanation,
    render_explanation,
)
from .log import EventLog, LogSnapshot
from .norm import SCHEMA_VERSION, Layer, Norm, NormConstraint, View
from .prioritization import (
    compare_periods,
    concentration,
    constraint_drivers,
    estimate_gamma,
    hotspot_table,
    layer_drivers,
    pareto,
    penalty_mass,
    prioritize,
    top_k_overlap,
    view_agreement,
)
from .scoring import ScoreResult, evaluate_constraint, evaluate_detailed, score, violation_matrix

__all__ = [
    "CONSTRAINT_TYPES",
    "SCHEMA_VERSION",
    "Balance",
    "BaselineError",
    "BaselineKind",
    "BaselineSpec",
    "CalibrationRecord",
    "Constraint",
    "EvaluationRecord",
    "EventLog",
    "EvidenceError",
    "EvidencePacket",
    "EvidenceUnavailableError",
    "Exclusion",
    "ExplanationPacket",
    "Lag",
    "Layer",
    "LogSchemaError",
    "LogSnapshot",
    "Metric",
    "Norm",
    "NormConstraint",
    "NormError",
    "NotScoredError",
    "Precedence",
    "Presence",
    "ReasonCode",
    "RunManifest",
    "ScoreResult",
    "Singularity",
    "StaleEvidenceError",
    "View",
    "WiseError",
    "__version__",
    "apply_calibration",
    "as_labels",
    "capture_evidence",
    "compare_periods",
    "concentration",
    "constraint_drivers",
    "constraint_from_dict",
    "coverage_report",
    "cross_case_replication",
    "datasets",
    "estimate_gamma",
    "evaluate_constraint",
    "evaluate_detailed",
    "event_replication",
    "evidence",
    "explain",
    "explain_priority",
    "fit_calibration",
    "gap_retained",
    "hotspot_table",
    "layer_drivers",
    "left_truncated",
    "load_baseline",
    "load_evidence",
    "load_explanation",
    "observation_window",
    "pareto",
    "penalty_mass",
    "prioritize",
    "render_explanation",
    "right_censored",
    "running_p2p_events",
    "running_p2p_log",
    "running_p2p_norm",
    "sat",
    "score",
    "timestamp_outliers",
    "top_k_overlap",
    "typed_cross_case_replication",
    "typed_event_replication",
    "typed_right_censored",
    "validation_table",
    "view_agreement",
    "violation_matrix",
]
