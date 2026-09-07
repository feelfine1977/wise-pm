"""Native object-centric assessment: object logs, adapters and bounded units.

The case-based core of WISE is unchanged and untouched by this package. Nothing
here is imported by ``import wise``; ask for it explicitly::

    from wise import oc

    log = oc.read_ocel2_json("p2p.json")
    spec = oc.UnitSpec(
        unit_type="invoice_review",
        anchor_type="invoice receipt",
        roles=(oc.RolePath("order", (oc.PathStep("Invoice Receipt of Purchase Order", target_type="purchase_order"),)),),
    )
    units = oc.build_units(log, spec, at="2024-06-30T00:00:00Z")

The package holds no graph database and no query language: immutable indexed
tables, bounded typed traversal, and an explicit refusal to guess. What it
gives a later stage is a *unit* — an anchor, its qualified neighbourhood, the
events in scope and an honest statement of whether that context is complete.

Modules: :mod:`~wise.oc.model` (the log contract), :mod:`~wise.oc.io` (the
adapters, and what each can carry), :mod:`~wise.oc.units` (typed bounded
assessment units), :mod:`~wise.oc.constraints` (three native relational check
families), :mod:`~wise.oc.accounting` (canonical additive quantities,
allocation and overlap) and :mod:`~wise.oc.evaluation` (the object catalogue,
the shared numerical kernel and :class:`~wise.oc.evaluation.OCScoreResult`).
"""

from __future__ import annotations

from .accounting import (
    Allocation,
    AllocationReport,
    QuantityRecord,
    allocate,
    canonical_record_id,
    equal_split,
)
from .constraints import (
    OBJECT_CHECKS,
    AmountSelector,
    CheckOutcome,
    CrossObjectLag,
    EventSelector,
    ObjectCheck,
    RelatedObjectCardinality,
    RelationalBalance,
    object_check_from_dict,
)
from .evaluation import (
    OC_NORM_SCHEMA,
    ObjectConstraint,
    ObjectNorm,
    OCScoreResult,
    evaluate_units,
    object_backlog,
    score_units,
    unit_frame,
)
from .io import (
    NATIVE_TABLES,
    OCEL2_JSON,
    OCEL2_SQLITE,
    OCEL_SPEC,
    LossReport,
    from_tables,
    interchange_support,
    ocel2_json_document,
    read_ocel2_json,
    read_ocel2_sqlite,
    to_tables,
    write_ocel2_json,
)
from .model import (
    E2O,
    O2O,
    OC_SCHEMA_VERSION,
    AttributeChange,
    AttributePolicy,
    AttributeReading,
    IssueSeverity,
    OCEvent,
    OCEventLog,
    OCIssueCode,
    OCObject,
    SourceMetadata,
    ValidationIssue,
    ValidationReport,
    observed_precision,
)
from .units import (
    AssessmentUnit,
    Hop,
    PathStep,
    PathWitness,
    RolePath,
    TraversalLimits,
    UnitScope,
    UnitSpec,
    UnitTruncation,
    build_units,
    context_report,
)

__all__ = [
    "E2O",
    "NATIVE_TABLES",
    "O2O",
    "OBJECT_CHECKS",
    "OCEL2_JSON",
    "OCEL2_SQLITE",
    "OCEL_SPEC",
    "OC_NORM_SCHEMA",
    "OC_SCHEMA_VERSION",
    "Allocation",
    "AllocationReport",
    "AmountSelector",
    "AssessmentUnit",
    "AttributeChange",
    "AttributePolicy",
    "AttributeReading",
    "CheckOutcome",
    "CrossObjectLag",
    "EventSelector",
    "Hop",
    "IssueSeverity",
    "LossReport",
    "OCEvent",
    "OCEventLog",
    "OCIssueCode",
    "OCObject",
    "OCScoreResult",
    "ObjectCheck",
    "ObjectConstraint",
    "ObjectNorm",
    "PathStep",
    "PathWitness",
    "QuantityRecord",
    "RelatedObjectCardinality",
    "RelationalBalance",
    "RolePath",
    "SourceMetadata",
    "TraversalLimits",
    "UnitScope",
    "UnitSpec",
    "UnitTruncation",
    "ValidationIssue",
    "ValidationReport",
    "allocate",
    "build_units",
    "canonical_record_id",
    "context_report",
    "equal_split",
    "evaluate_units",
    "from_tables",
    "interchange_support",
    "object_backlog",
    "object_check_from_dict",
    "observed_precision",
    "ocel2_json_document",
    "read_ocel2_json",
    "read_ocel2_sqlite",
    "score_units",
    "to_tables",
    "unit_frame",
    "write_ocel2_json",
]
