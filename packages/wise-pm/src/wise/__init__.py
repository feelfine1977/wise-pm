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
    backlog.table                                         # the ranked slices
    backlog.spec                                          # what they were ranked with
"""

from . import datasets
from . import families as families  # registers the built-in constraint families
from ._version import __version__
from .backlog import Backlog, BacklogSpec
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
    validation_table,
)
from .errors import LogSchemaError, NormError, NotScoredError, WiseError
from .family import Evaluation
from .log import EventLog
from .log_schema import LogSchema, ParsedLog, parse
from .norm import SCHEMA_VERSION, Layer, Norm, NormConstraint, View
from .prioritization import (
    MISSING_KEY,
    case_slices,
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
from .scoring import ScoreResult, evaluate, evaluate_constraint, score, violation_matrix

__all__ = [
    "CONSTRAINT_TYPES",
    "MISSING_KEY",
    "SCHEMA_VERSION",
    "Backlog",
    "BacklogSpec",
    "Balance",
    "Constraint",
    "Evaluation",
    "EventLog",
    "Exclusion",
    "Lag",
    "Layer",
    "LogSchema",
    "LogSchemaError",
    "Metric",
    "Norm",
    "NormConstraint",
    "NormError",
    "NotScoredError",
    "ParsedLog",
    "Precedence",
    "Presence",
    "ScoreResult",
    "Singularity",
    "View",
    "WiseError",
    "__version__",
    "as_labels",
    "case_slices",
    "compare_periods",
    "concentration",
    "constraint_drivers",
    "constraint_from_dict",
    "cross_case_replication",
    "datasets",
    "estimate_gamma",
    "evaluate",
    "evaluate_constraint",
    "event_replication",
    "gap_retained",
    "hotspot_table",
    "layer_drivers",
    "left_truncated",
    "observation_window",
    "pareto",
    "parse",
    "penalty_mass",
    "prioritize",
    "right_censored",
    "running_p2p_events",
    "running_p2p_log",
    "running_p2p_norm",
    "sat",
    "score",
    "timestamp_outliers",
    "top_k_overlap",
    "validation_table",
    "view_agreement",
    "violation_matrix",
]
