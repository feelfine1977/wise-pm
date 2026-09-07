"""Object catalogues, native evaluation and the typed object score result.

An :class:`ObjectNorm` is the object-centric counterpart of
:class:`wise.norm.Norm`: layers, stakeholder views and weighted checks. It
reuses :class:`wise.norm.Layer` and :class:`wise.norm.View` unchanged — a layer
still groups checks and a view still weights them — and it has its own
versioned envelope (``wise-oc-norm/1``), its own catalogue and its own
validation path, because its checks name roles and qualifiers that a schema-2
case norm cannot express. Nothing here can be loaded through
:meth:`wise.norm.Norm.from_dict`, and no object check ever appears in an old
norm file.

:func:`score_units` evaluates a catalogue over
:class:`~wise.oc.units.AssessmentUnit`\\ s and returns an
:class:`OCScoreResult`. The arithmetic is :mod:`wise._aggregation` — the same
kernel :func:`wise.scoring.score` uses, so an object score and a case score are
the same formula applied to different evidence, not two formulas that happen to
agree today. No :class:`~wise.log.EventLog` is fabricated on the way:
relational semantics are evaluated on the relations.

The result is deliberately a *different type* from
:class:`wise.scoring.ScoreResult`, whose ``log`` attribute promises a
case-based :class:`~wise.log.EventLog`. :meth:`OCScoreResult.frame` produces
the wide per-unit table the existing prioritisation path expects, and
:func:`object_backlog` checks that a ranking stays inside one unit type before
handing it to :func:`wise.prioritize`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .._aggregation import LayerAssignment, aggregate
from .._version import __version__
from ..errors import OCConstraintError, OCUnitError
from ..evidence.models import (
    EvaluationRecord,
    Qualification,
    QualificationCode,
    ReasonCode,
    to_records_frame,
)
from ..norm import DEFAULT_SCORING_MODE, SCORING_MODES, Layer, View
from .constraints import CheckOutcome, ObjectCheck, object_check_from_dict
from .model import OCEventLog
from .units import AssessmentUnit, UnitSpec

#: The object catalogue's own contract version. Independent of the norm schema
#: (2), the evidence packet (``wise-evidence/1``) and the object log
#: (``wise-oc/1``): they version separately because they change separately.
OC_NORM_SCHEMA = "wise-oc-norm/1"


@dataclass(frozen=True)
class ObjectConstraint:
    """One native check in a catalogue: an id, a layer, a weight and a scope.

    ``unit_types`` is the applicability rule and it is deliberately blunt: a
    check applies to the unit types it names, or to all of them when it names
    none. A unit type it does not name is *out of scope*, which is not the same
    as evaluated-and-satisfied.
    """

    id: str
    layer: str
    check: ObjectCheck
    weight: float = 1.0
    unit_types: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "layer", str(self.layer))
        object.__setattr__(self, "unit_types", tuple(str(u) for u in self.unit_types))
        if not self.id:
            raise OCConstraintError("an object constraint needs a non-empty id")
        if not isinstance(self.check, ObjectCheck):
            raise OCConstraintError(f"object constraint {self.id!r}: check must be an ObjectCheck")
        weight = float(self.weight)
        if not np.isfinite(weight) or weight < 0:
            raise OCConstraintError(f"object constraint {self.id!r}: weight must be finite and >= 0, got {self.weight!r}")
        object.__setattr__(self, "weight", weight)

    @property
    def type(self) -> str:
        return self.check.type

    def applies_to(self, unit: AssessmentUnit) -> bool:
        """Whether this check is in scope for one unit."""
        return not self.unit_types or unit.unit_type in self.unit_types

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "layer": self.layer,
            "weight": self.weight,
            "unit_types": list(self.unit_types),
            "description": self.description,
            **self.check.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ObjectConstraint:
        known = {"id", "layer", "weight", "unit_types", "description", "type", "params"}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise OCConstraintError(f"object constraint {payload.get('id')!r}: unknown key(s) {unknown}")
        return cls(
            id=payload["id"],
            layer=payload["layer"],
            check=object_check_from_dict(payload),
            weight=float(payload.get("weight", 1.0)),
            unit_types=tuple(payload.get("unit_types", ())),
            description=str(payload.get("description", "")),
        )


@dataclass(frozen=True)
class ObjectNorm:
    """A versioned catalogue of native checks, with layers and views.

    >>> from wise.oc.constraints import RelatedObjectCardinality
    >>> from wise.norm import Layer, View
    >>> norm = ObjectNorm(
    ...     constraints=(ObjectConstraint("m1", "matching", RelatedObjectCardinality(role="order", minimum=1)),),
    ...     layers=(Layer("matching"),),
    ...     views=(View("Finance", constraint_weights={"m1": 1.0}),),
    ... )
    >>> norm.constraint_ids, norm.schema
    (['m1'], 'wise-oc-norm/1')
    """

    constraints: tuple[ObjectConstraint, ...]
    layers: tuple[Layer, ...]
    views: tuple[View, ...]
    name: str = ""
    version: str = "1"
    scoring_mode: str = DEFAULT_SCORING_MODE
    schema: str = OC_NORM_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "views", tuple(self.views))
        object.__setattr__(self, "schema", str(self.schema))
        self.validate()

    # ------------------------------------------------------------------ lookups
    @property
    def constraint_ids(self) -> list[str]:
        return [c.id for c in self.constraints]

    @property
    def layer_ids(self) -> list[str]:
        return [layer.id for layer in self.layers]

    @property
    def view_names(self) -> list[str]:
        return [v.name for v in self.views]

    @property
    def layer_of(self) -> dict[str, str]:
        return {c.id: c.layer for c in self.constraints}

    def get_constraint(self, cid: str) -> ObjectConstraint:
        for c in self.constraints:
            if c.id == cid:
                return c
        raise OCConstraintError(f"unknown object constraint {cid!r}; this catalogue holds {self.constraint_ids}")

    def get_view(self, name: str) -> View:
        for v in self.views:
            if v.name == name:
                return v
        raise OCConstraintError(f"unknown view {name!r}; available: {self.view_names}")

    # --------------------------------------------------------------- validation
    def validate(self) -> None:
        """Structural validation. The catalogue's *own* path, not the norm's."""
        if self.schema != OC_NORM_SCHEMA:
            raise OCConstraintError(f"object catalogue schema must be {OC_NORM_SCHEMA!r}, got {self.schema!r}")
        ids = self.constraint_ids
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise OCConstraintError(f"duplicate object constraint id(s) {duplicates}")
        if not self.layers:
            raise OCConstraintError("an object catalogue needs at least one layer")
        layer_ids = self.layer_ids
        if len(set(layer_ids)) != len(layer_ids):
            raise OCConstraintError("duplicate layer id(s)")
        unknown = sorted({c.layer for c in self.constraints} - set(layer_ids))
        if unknown:
            raise OCConstraintError(f"object constraint(s) placed in undeclared layer(s) {unknown}")
        if not self.views:
            raise OCConstraintError("an object catalogue needs at least one view")
        names = self.view_names
        if len(set(names)) != len(names):
            raise OCConstraintError("duplicate view name(s)")
        for view in self.views:
            if view.constraint_weights is not None:
                stray = sorted(set(view.constraint_weights) - set(ids))
                if stray:
                    raise OCConstraintError(f"view {view.name!r} weights unknown check(s) {stray}")
            else:
                stray = sorted(set(view.layer_weights or {}) - set(layer_ids))
                if stray:
                    raise OCConstraintError(f"view {view.name!r} weights unknown layer(s) {stray}")
        if self.scoring_mode not in SCORING_MODES:
            raise OCConstraintError(f"scoring_mode must be one of {SCORING_MODES}, got {self.scoring_mode!r}")

    def check(self, spec: UnitSpec) -> tuple[str, ...]:
        """Problems with evaluating this catalogue against one unit type.

        A check that reads a role the unit type does not declare would select
        nothing and look exactly like a satisfied obligation; :func:`score_units`
        refuses such a pair rather than scoring it.
        """
        problems: list[str] = []
        for constraint in self.constraints:
            if constraint.unit_types and spec.unit_type not in constraint.unit_types:
                continue
            problems.extend(f"check {constraint.id!r}: {issue}" for issue in constraint.check.check_spec(spec))
        return tuple(problems)

    # ------------------------------------------------------------------ weights
    def raw_weights(self, view: str | View) -> dict[str, float]:
        """Raw weights ``w_c`` for one view, over every check in order."""
        v = view if isinstance(view, View) else self.get_view(view)
        if v.constraint_weights is not None:
            return {c.id: float(v.constraint_weights.get(c.id, 0.0)) for c in self.constraints}
        out: dict[str, float] = {}
        for layer in self.layer_ids:
            members = [c for c in self.constraints if c.layer == layer]
            total = sum(c.weight for c in members)
            a = float((v.layer_weights or {}).get(layer, 0.0))
            for c in members:
                out[c.id] = a * ((c.weight / total) if total > 0 else 0.0)
        return out

    def weight_vector(self, view: str | View) -> np.ndarray:
        w = self.raw_weights(view)
        return np.array([w[c] for c in self.constraint_ids], dtype=float)

    def layer_assignment(self) -> LayerAssignment:
        """The kernel's plain-data view of this catalogue."""
        return LayerAssignment.from_mapping(self.constraint_ids, self.layer_of, self.layer_ids)

    # ---------------------------------------------------------------- interchange
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "name": self.name,
            "version": self.version,
            "scoring_mode": self.scoring_mode,
            "layers": [{"id": layer.id, "name": layer.name, "description": layer.description} for layer in self.layers],
            "views": [
                {
                    "name": v.name,
                    "layer_weights": v.layer_weights,
                    "constraint_weights": v.constraint_weights,
                    "description": v.description,
                }
                for v in self.views
            ],
            "constraints": [c.to_dict() for c in self.constraints],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ObjectNorm:
        """Load a catalogue. The separate validation path, by design."""
        known = {"schema", "name", "version", "scoring_mode", "layers", "views", "constraints"}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise OCConstraintError(f"object catalogue: unknown top-level key(s) {unknown}")
        schema = str(payload.get("schema", OC_NORM_SCHEMA))
        if schema != OC_NORM_SCHEMA:
            raise OCConstraintError(
                f"object catalogue schema {schema!r} is not {OC_NORM_SCHEMA!r}; this loader reads its own envelope only"
            )
        return cls(
            constraints=tuple(ObjectConstraint.from_dict(c) for c in payload.get("constraints", ())),
            layers=tuple(Layer(**dict(layer)) for layer in payload.get("layers", ())),
            views=tuple(
                View(**{k: v for k, v in dict(view).items() if v is not None or k == "name"}) for view in payload.get("views", ())
            ),
            name=str(payload.get("name", "")),
            version=str(payload.get("version", "1")),
            scoring_mode=str(payload.get("scoring_mode", DEFAULT_SCORING_MODE)),
            schema=schema,
        )

    def dumps(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False, default=str)

    def fingerprint(self) -> str:
        """SHA-256 of the canonical catalogue. Not a norm fingerprint."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def __repr__(self) -> str:
        return (
            f"ObjectNorm({len(self.constraints)} check(s) in {len(self.layers)} layer(s), "
            f"views={self.view_names}, mode={self.scoring_mode!r})"
        )


# ------------------------------------------------------------------- evaluation
def evaluate_units(
    log: OCEventLog,
    norm: ObjectNorm,
    units: Sequence[AssessmentUnit],
    *,
    spec: UnitSpec | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], CheckOutcome]]:
    """Evaluate a catalogue over units → ``(violations, in_scope, outcomes)``.

    ``violations`` is units × checks with ``NaN`` where a check was out of
    scope **or** not evaluable; ``in_scope`` separates the two. ``outcomes``
    carries the measurements, witnesses, reasons and qualifications behind
    every cell, keyed by ``(unit_id, check_id)``.
    """
    if spec is not None:
        problems = norm.check(spec)
        if problems:
            raise OCConstraintError(
                f"catalogue {norm.name or '<unnamed>'} cannot be evaluated on unit type {spec.unit_type!r}: "
                + "; ".join(problems)
                + ". A check reading a role that does not exist selects nothing, which is indistinguishable "
                "from a satisfied obligation once it becomes a number."
            )
    unit_ids = [u.unit_id for u in units]
    if len(set(unit_ids)) != len(unit_ids):
        raise OCUnitError("two units share an id; a unit id must identify exactly one assessment unit")
    cids = norm.constraint_ids
    violations = np.full((len(units), len(cids)), np.nan)
    scope = np.zeros((len(units), len(cids)), dtype=bool)
    outcomes: dict[tuple[str, str], CheckOutcome] = {}
    for row, unit in enumerate(units):
        for column, constraint in enumerate(norm.constraints):
            if not constraint.applies_to(unit):
                continue
            scope[row, column] = True
            outcome = constraint.check.evaluate(log, unit)
            outcomes[(unit.unit_id, constraint.id)] = outcome
            if outcome.violation is not None:
                violations[row, column] = outcome.violation
    return (
        pd.DataFrame(violations, index=pd.Index(unit_ids, name="unit_id"), columns=cids),
        pd.DataFrame(scope, index=pd.Index(unit_ids, name="unit_id"), columns=cids),
        outcomes,
    )


def unit_frame(units: Sequence[AssessmentUnit], exposure: Mapping[str, float] | None = None) -> pd.DataFrame:
    """The per-unit attribute table: identity, type, completeness and volume."""
    rows = [
        {
            "unit_id": u.unit_id,
            "unit_type": u.unit_type,
            "anchor_id": u.anchor_id,
            "anchor_type": u.anchor_type,
            "complete": u.complete,
            "n_events": len(u.event_ids),
            "n_bound": sum(len(v) for v in u.bindings.values()),
            "evaluation_time": u.evaluation_time,
        }
        for u in units
    ]
    frame = pd.DataFrame(rows, columns=_UNIT_COLUMNS)
    frame = frame.set_index("unit_id") if len(frame) else frame.set_index(pd.Index([], name="unit_id"))
    if exposure is not None:
        frame["exposure"] = [float(exposure.get(uid, np.nan)) for uid in frame.index]
    return frame


_UNIT_COLUMNS = ["unit_id", "unit_type", "anchor_id", "anchor_type", "complete", "n_events", "n_bound", "evaluation_time"]


@dataclass(eq=False)
class OCScoreResult:
    """Scores over typed assessment units, with the evidence behind them.

    A separate type from :class:`wise.scoring.ScoreResult` on purpose: that
    class's :attr:`~wise.scoring.ScoreResult.log` promises a case-based
    :class:`~wise.log.EventLog`, and an object log is not one. What the two
    share is :meth:`frame`, which produces the wide per-unit table the existing
    prioritisation path consumes, and the kernel that produced the numbers.
    """

    norm: ObjectNorm
    units: pd.DataFrame
    violations: pd.DataFrame
    in_scope: pd.DataFrame
    scores: pd.DataFrame
    contributions: dict[str, pd.DataFrame]
    mode: str = DEFAULT_SCORING_MODE
    records: tuple[EvaluationRecord, ...] = ()
    run_id: str = ""
    norm_fingerprint: str = ""
    log_fingerprint: str = ""
    wise_version: str = __version__
    oc_log: OCEventLog | None = field(default=None, repr=False)
    _weights: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _eff_cache: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        return (
            f"OCScoreResult({len(self.scores):,} unit(s) of type(s) {list(self.unit_types)} × "
            f"{self.violations.shape[1]} check(s), views={self.views}, mode={self.mode!r}, "
            f"complete_context={self.complete_share():.3f})"
        )

    # ------------------------------------------------------------------ shape
    @property
    def views(self) -> list[str]:
        return list(self.scores.columns)

    @property
    def unit_types(self) -> tuple[str, ...]:
        return tuple(sorted(self.units["unit_type"].unique())) if len(self.units) else ()

    @property
    def applicable(self) -> pd.DataFrame:
        """Boolean units × checks: evaluated (in scope and evaluable)."""
        return self.violations.notna()

    def complete_share(self) -> float:
        """Share of units whose context was gathered without hitting a limit."""
        return float(self.units["complete"].mean()) if len(self.units) else float("nan")

    # ---------------------------------------------------------------- numbers
    def effective_weights(self, view: str) -> pd.DataFrame:
        if view not in self._eff_cache:
            from .._aggregation import effective_weights as kernel_weights

            M = self.violations.notna().to_numpy(dtype=float)
            w_eff = kernel_weights(M, self._weights[view], self.norm.layer_assignment(), self.mode)
            self._eff_cache[view] = pd.DataFrame(
                np.nan_to_num(w_eff), index=self.violations.index, columns=self.violations.columns
            )
        return self._eff_cache[view]

    def penalties(self, view: str) -> pd.DataFrame:
        return self.effective_weights(view) * self.violations.fillna(0.0)

    def check_decomposition(self, atol: float = 1e-9) -> float:
        worst = 0.0
        for v in self.views:
            s = self.scores[v]
            lhs = self.contributions[v].sum(axis=1)[s.notna()]
            rhs = (1.0 - s)[s.notna()]
            worst = max(worst, float((lhs - rhs).abs().max()) if len(lhs) else 0.0)
        if worst > atol:
            raise AssertionError(f"layer decomposition violated: max error {worst:.3e}")
        return worst

    def frame(self, view: str | None = None, *, unit_type: str | None = None, allow_mixed: bool = False) -> pd.DataFrame:
        """Wide per-unit table, compatible with the existing prioritisation path.

        Columns: the unit attributes, then ``score`` and ``contrib__<layer>``
        for the chosen view (or ``score__<view>`` and
        ``contrib__<view>__<layer>`` for every view).

        A result holding more than one unit type refuses to produce one table
        unless a ``unit_type`` is chosen or ``allow_mixed=True`` is passed: a
        count of invoices and a count of item obligations are not one
        interchangeable volume, and a mean over both is a mean over nothing.
        """
        rows = self._rows(unit_type, allow_mixed)
        parts: list[pd.Series | pd.DataFrame] = []
        if view is not None:
            if view not in self.contributions:
                raise OCConstraintError(f"unknown view {view!r}; available: {self.views}")
            parts.append(self.scores.loc[rows, view].rename("score"))
            parts.append(self.contributions[view].loc[rows].add_prefix("contrib__"))
        else:
            for v in self.views:
                parts.append(self.scores.loc[rows, v].rename(f"score__{v}"))
                parts.append(self.contributions[v].loc[rows].add_prefix(f"contrib__{v}__"))
        base = self.units.loc[rows]
        return pd.concat([base, *parts], axis=1) if parts else base.copy()

    def _rows(self, unit_type: str | None, allow_mixed: bool) -> pd.Index:
        if unit_type is not None:
            if unit_type not in self.unit_types:
                raise OCConstraintError(f"unknown unit type {unit_type!r}; this result holds {list(self.unit_types)}")
            return self.units.index[self.units["unit_type"] == unit_type]
        if len(self.unit_types) > 1 and not allow_mixed:
            raise OCConstraintError(
                f"this result holds {list(self.unit_types)} unit types. Ranking across them would treat a count of "
                f"{self.unit_types[0]} units and a count of {self.unit_types[1]} units as one volume. Choose "
                "frame(unit_type=...), or pass allow_mixed=True having decided that pooling them is meaningful."
            )
        return self.units.index

    def summary(self) -> pd.DataFrame:
        """Per view: scored units, mean score, mean layer contributions."""
        rows = []
        for v in self.views:
            s = self.scores[v]
            row: dict[str, Any] = {"view": v, "n_scored": int(s.notna().sum()), "mean_score": float(s.mean())}
            for layer in self.norm.layer_ids:
                row[f"contrib__{layer}"] = float(self.contributions[v][layer][s.notna()].mean())
            rows.append(row)
        return pd.DataFrame(rows).set_index("view")

    # --------------------------------------------------------------- evidence
    def records_frame(self) -> pd.DataFrame:
        """Long-format evidence: one row per check on one unit."""
        return to_records_frame(self.records)

    def record(self, unit_id: str, constraint_id: str) -> EvaluationRecord:
        for r in self.records:
            if r.unit_id == unit_id and r.constraint_id == constraint_id:
                return r
        raise OCConstraintError(f"no record for unit {unit_id!r} and check {constraint_id!r}")

    def qualifications(self) -> tuple[Qualification, ...]:
        """Every distinct limitation this result carries, by code."""
        seen: dict[tuple[str, str], Qualification] = {}
        for r in self.records:
            for q in r.qualifications:
                seen.setdefault((q.code.value, q.message), q)
        if len(self.unit_types) > 1:
            q = Qualification(
                QualificationCode.HETEROGENEOUS_UNIT_TYPES,
                f"this result mixes the unit types {list(self.unit_types)}; their scores share a scale only because "
                "they share a catalogue, and their counts are not interchangeable volumes",
                scope="run",
            )
            seen[(q.code.value, q.message)] = q
        return tuple(seen.values())

    def unscored(self, view: str) -> pd.Index:
        """Units with no positively weighted applicable check in this view."""
        return self.scores.index[self.scores[view].isna()]


def score_units(
    log: OCEventLog,
    norm: ObjectNorm,
    units: Sequence[AssessmentUnit],
    views: Sequence[str] | str | None = None,
    *,
    mode: str | None = None,
    spec: UnitSpec | None = None,
    exposure: Mapping[str, float] | None = None,
    run_id: str | None = None,
) -> OCScoreResult:
    """Score assessment units against an object catalogue.

    The numbers come from :mod:`wise._aggregation`, the same kernel
    :func:`wise.scoring.score` uses; ``mode`` defaults to the catalogue's
    ``scoring_mode`` and is recorded on the result.

    ``exposure`` attaches a per-unit additive volume — typically the output of
    :mod:`wise.oc.accounting`, so that the amount behind a backlog has been
    through the conservation contract before it is ranked.
    """
    from ..evidence.manifest import new_run_id

    chosen_views = [views] if isinstance(views, str) else list(dict.fromkeys(views or norm.view_names))
    for v in chosen_views:
        norm.get_view(v)
    chosen_mode = mode or norm.scoring_mode
    if chosen_mode not in SCORING_MODES:
        raise OCConstraintError(f"mode must be one of {SCORING_MODES}, got {chosen_mode!r}")

    V, S, outcomes = evaluate_units(log, norm, units, spec=spec)
    layers = norm.layer_assignment()
    Vm = V.to_numpy(dtype=float)
    row_ids = tuple(V.index)

    score_cols: list[np.ndarray] = []
    contributions: dict[str, pd.DataFrame] = {}
    weights: dict[str, np.ndarray] = {}
    for view in chosen_views:
        w = norm.weight_vector(view)
        weights[view] = w
        out = aggregate(Vm, w, layers, chosen_mode, row_ids=row_ids)
        score_cols.append(out.scores)
        contributions[view] = pd.DataFrame(out.contributions, index=V.index, columns=norm.layer_ids)
    scores = pd.DataFrame(
        np.column_stack(score_cols) if score_cols else np.empty((len(V), 0)), index=V.index, columns=chosen_views
    )

    identity = run_id or new_run_id("ocrun")
    records = _records(identity, norm, units, outcomes)
    return OCScoreResult(
        norm=norm,
        units=unit_frame(units, exposure),
        violations=V,
        in_scope=S,
        scores=scores,
        contributions=contributions,
        mode=chosen_mode,
        records=records,
        run_id=identity,
        norm_fingerprint=norm.fingerprint(),
        log_fingerprint=log.content_fingerprint(),
        oc_log=log,
        _weights=weights,
    )


def _records(
    run_id: str,
    norm: ObjectNorm,
    units: Sequence[AssessmentUnit],
    outcomes: Mapping[tuple[str, str], CheckOutcome],
) -> tuple[EvaluationRecord, ...]:
    """One :class:`~wise.evidence.models.EvaluationRecord` per check per unit.

    Out-of-scope pairs get a record too: "this check does not apply to this
    unit type" is a fact about the assessment, and leaving it out would make an
    unevaluated check indistinguishable from one nobody asked for.
    """
    out: list[EvaluationRecord] = []
    for unit in units:
        for constraint in norm.constraints:
            in_scope = constraint.applies_to(unit)
            outcome = outcomes.get((unit.unit_id, constraint.id))
            if not in_scope or outcome is None:
                out.append(
                    EvaluationRecord(
                        run_id=run_id,
                        evaluation_id=f"{run_id}:{unit.unit_id}:{constraint.id}",
                        unit_id=unit.unit_id,
                        unit_type=unit.unit_type,
                        constraint_id=constraint.id,
                        constraint_type=constraint.type,
                        constraint_version=norm.version,
                        in_scope=False,
                        evaluable=False,
                        reason_code=ReasonCode.OUT_OF_SCOPE,
                        violation=None,
                        parameters=constraint.check.params(),
                        qualifications=unit.qualifications,
                    )
                )
                continue
            out.append(
                EvaluationRecord(
                    run_id=run_id,
                    evaluation_id=f"{run_id}:{unit.unit_id}:{constraint.id}",
                    unit_id=unit.unit_id,
                    unit_type=unit.unit_type,
                    constraint_id=constraint.id,
                    constraint_type=constraint.type,
                    constraint_version=norm.version,
                    in_scope=True,
                    evaluable=outcome.evaluable,
                    reason_code=outcome.reason,
                    violation=outcome.violation,
                    measurements=outcome.measurements,
                    parameters=constraint.check.params(),
                    policies=outcome.policies,
                    witnesses=outcome.witnesses,
                    qualifications=(*outcome.qualifications, *unit.qualifications),
                    witnesses_materialised=bool(outcome.witnesses),
                    n_witnesses_total=outcome.n_witnesses_total,
                )
            )
    return tuple(out)


def object_backlog(
    result: OCScoreResult,
    by: str | Sequence[str],
    view: str,
    *,
    unit_type: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Rank slices of one unit type, reusing :func:`wise.prioritize`.

    The typed-grouping check the handoff asks for happens here, before the
    existing prioritisation code sees anything: a backlog is built inside one
    unit type, the grouping columns must exist on the unit table, and
    ``volume="exposure"`` needs an exposure column that :func:`score_units` was
    actually given.

    The returned frame is exactly what :func:`wise.prioritize` returns; its
    ``attrs`` gain ``unit_type`` and ``oc_run_id`` so a backlog can say which
    population it ranked, and ``context_truncated`` with
    ``context_truncated_share`` so it can say how much of that population was
    ranked on a context that was cut. :func:`wise.prioritize` does not carry
    the frame's ``complete`` column into its output, so without those two the
    aggregate a reader receives would be the one artefact in the chain that
    cannot say a unit's evidence was partial.
    """
    from ..prioritization import prioritize

    if unit_type is None and len(result.unit_types) == 1:
        unit_type = result.unit_types[0]
    frame = result.frame(view, unit_type=unit_type)
    keys = [by] if isinstance(by, str) else list(by)
    unknown = [k for k in keys if k not in frame.columns and k != frame.index.name]
    if unknown:
        raise OCConstraintError(f"unknown slice key(s) {unknown}; the unit table has {list(frame.columns)}")
    if kwargs.get("volume") == "exposure" and "exposure" not in frame.columns:
        raise OCConstraintError(
            "volume='exposure' needs an exposure column; pass exposure=... to score_units, having produced it "
            "through wise.oc.accounting so that no amount is counted twice"
        )
    out = prioritize(frame, by, **kwargs)
    out.attrs["unit_type"] = unit_type
    out.attrs["oc_run_id"] = result.run_id
    cut = int((~frame["complete"].astype(bool)).sum()) if "complete" in frame.columns else 0
    out.attrs["context_truncated"] = cut
    out.attrs["context_truncated_share"] = (cut / len(frame)) if len(frame) else float("nan")
    return out


__all__ = [
    "OC_NORM_SCHEMA",
    "OCScoreResult",
    "ObjectConstraint",
    "ObjectNorm",
    "evaluate_units",
    "object_backlog",
    "score_units",
    "unit_frame",
]
