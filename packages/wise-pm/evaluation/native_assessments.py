"""Use the public WISE API for previously evaluated, explicitly scoped criteria.

The carrier log has one synthetic assessment record per case. It is a transport
for case attributes, not a reconstruction of source process events.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

import wise


def prepare_native(
    severity: pd.DataFrame,
    in_scope: pd.DataFrame,
    layers: Mapping[str, str],
    views: Mapping[str, Mapping[str, float]],
    groups: pd.Series,
    *,
    name: str,
    mode: str = "flat",
):
    """Return a public EventLog, serializable Norm and prepared case attributes.

    Missing severity inside scope remains unevaluable. Finite severity outside
    scope is rejected. Views contain raw per-criterion weights, not layer totals.
    """
    if severity.empty or not severity.index.is_unique or not severity.columns.is_unique:
        raise ValueError("Severity requires nonempty unique case and criterion indices")
    if not severity.index.equals(in_scope.index) or not severity.columns.equals(in_scope.columns):
        raise ValueError("Scope must have exactly the severity index and column order")
    if in_scope.isna().any().any() or not all(pd.api.types.is_bool_dtype(d) for d in in_scope.dtypes):
        raise ValueError("Scope must be boolean without missing values")
    if not groups.index.equals(severity.index):
        raise ValueError("Group index must exactly match assessments")
    if set(layers) != set(severity):
        raise ValueError("Every criterion needs exactly one layer")
    if any(not isinstance(c, str) or not c for c in severity):
        raise ValueError("Criterion IDs must be nonempty strings")
    if any(not isinstance(layer, str) or not layer for layer in layers.values()):
        raise ValueError("Layer IDs must be nonempty strings")
    a = severity.to_numpy(dtype=float)
    if np.isinf(a).any() or ((a < 0) | (a > 1)).any():
        raise ValueError("Finite severity must be in [0,1]")
    if (severity.notna() & ~in_scope).any().any():
        raise ValueError("Out-of-scope severity must be missing, not zero")
    for view, w in views.items():
        if set(w) != set(severity):
            raise ValueError(f"View {view} must name every criterion")
        x = np.array([w[c] for c in severity], float)
        if not np.isfinite(x).all() or (x < 0).any() or not (x > 0).any():
            raise ValueError("Each view needs finite nonnegative weights and positive mass")
    cases = pd.DataFrame(index=severity.index.copy())
    cases.index.name = "case_id"
    cases["assessment_group"] = groups
    entries = []
    for cid in severity:
        value = f"assessment_value__{cid}"
        scope = f"assessment_scope__{cid}"
        cases[value] = severity[cid]
        cases[scope] = in_scope[cid]
        entries.append(
            wise.NormConstraint(
                cid,
                layers[cid],
                wise.Metric(value, threshold=0, width=1),
                applicability={"attr": scope, "eq": True},
                description="Identity mapping of an externally evaluated bounded severity; source evidence remains with the adapter.",
            )
        )
    norm = wise.Norm(
        tuple(entries),
        tuple(wise.Layer(layer) for layer in dict.fromkeys(layers.values())),
        tuple(wise.View(v, constraint_weights=dict(w)) for v, w in views.items()),
        name=name,
        scoring_mode=mode,
        description="Prepared assessment parity norm. Metric identity mapping preserves declared severities and scope.",
        metadata={
            "input_kind": "prepared_assessment_matrix",
            "carrier": "One non-process assessment record per case; source traces are separate.",
            "raw_operator_equivalence_claimed": False,
        },
    )
    return carrier_log(cases), norm, cases


def carrier_log(cases: pd.DataFrame):
    if not cases.index.is_unique:
        raise ValueError("Case identifiers must be unique")
    ev = cases.copy()
    ev["case_id"] = cases.index
    ev["activity"] = "prepared assessment"
    ev["timestamp"] = pd.Timestamp("1970-01-01")
    return wise.EventLog(
        ev, case_col="case_id", activity_col="activity", timestamp_col="timestamp", case_attributes=list(cases.columns)
    )


def weighted_coverage(result, view):
    """Diagnostic computed from public evaluated/scope matrices and norm weights."""
    w = result.norm.weight_vector(view)
    numerator = result.violations.notna().mul(w).sum(axis=1)
    denominator = result.in_scope.mul(w).sum(axis=1)
    return (numerator / denominator).where(denominator > 0).rename("coverage")


def write_prepared(directory, cases, norm):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cases.to_parquet(directory / "assessment_features.parquet")
    norm.dump(directory / "native_norm.json")
    # Executions use the serialized, reloaded public artifact, not only the constructor.
    return wise.Norm.load(directory / "native_norm.json")
