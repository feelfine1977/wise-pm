"""Who may see what, decided by the application and applied before anything else.

The library does not authenticate anybody. The application supplies a
:class:`Principal` and an :class:`AccessPolicy`, and everything on the
assistance path is filtered through it *before* a document is retrieved and
*before* an aggregate is computed. That ordering is the whole point: a mean, a
share or a comparator computed over rows the principal may not see leaks those
rows just as surely as printing them would.

Three properties hold by construction:

**It fails closed.** A default :class:`AccessPolicy` permits nothing: no
scope, no run, no view, no row, no column, no document. Widening is always an
explicit act by the application. Nothing a model returns can supply, alter or
widen a policy — the policy is an argument to the gateway, never a field in a
response.

**Rows are filtered before aggregation.** :meth:`AccessPolicy.row_mask` is the
one place a population is narrowed, and the gateway calls it before it
computes anything. There is no "compute then hide" path.

**An unauthorised comparator is recomputed, not quietly reused.** A frozen
all-company baseline is an aggregate over rows a restricted principal may not
see. :func:`authorised_baseline` either finds an explicit aggregate
entitlement for it, or recomputes the comparator in the authorised population
and *relabels* it, so the reader is told the comparison changed rather than
being shown a number from a population they were never entitled to.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..errors import AccessDenied
from ..explain.baseline import BaselineKind, BaselineSpec, _one_view

if TYPE_CHECKING:  # pragma: no cover
    from ..evidence.models import EvaluationRecord
    from ..scoring import ScoreResult

#: Version of the policy contract.
POLICY_SCHEMA_VERSION = "wise-access-policy/1"


class Scope(str, Enum):
    """The named capabilities a policy can grant. Each is granted explicitly."""

    RUN_SUMMARY = "run_summary"
    EVIDENCE = "evidence"
    COMPARE_GROUPS = "compare_groups"
    EXPLANATION = "explanation"
    POLICY_DOCUMENTS = "policy_documents"
    NORM_PREVIEW = "norm_preview"


@dataclass(frozen=True)
class Principal:
    """The authenticated caller, as the application knows them.

    The library stores this for provenance and compares nothing against a
    directory. ``roles`` is opaque here: an application that means something
    by a role enforces it in the scopes it grants.
    """

    principal_id: str
    roles: tuple[str, ...] = ()
    tenant: str | None = None
    authenticated_by: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "principal_id", str(self.principal_id))
        object.__setattr__(self, "roles", tuple(sorted(str(r) for r in self.roles)))
        if not self.principal_id:
            raise AccessDenied("a principal needs an identifier; an anonymous caller is not authorised by default")

    def to_dict(self) -> dict[str, Any]:
        return {
            "principal_id": self.principal_id,
            "roles": list(self.roles),
            "tenant": self.tenant,
            "authenticated_by": self.authenticated_by,
        }


@dataclass(frozen=True)
class ComparatorChange:
    """A record that the comparison shown is not the comparison requested."""

    requested_baseline_id: str
    effective_baseline_id: str
    reason: str
    authorised_population_size: int
    requested_population_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_baseline_id": self.requested_baseline_id,
            "effective_baseline_id": self.effective_baseline_id,
            "reason": self.reason,
            "authorised_population_size": int(self.authorised_population_size),
            "requested_population_size": self.requested_population_size,
        }

    def message(self) -> str:
        return (
            f"comparator {self.requested_baseline_id!r} was recomputed as {self.effective_baseline_id!r} over the "
            f"{self.authorised_population_size} authorised units: {self.reason}"
        )


@dataclass(frozen=True)
class AccessPolicy:
    """What one principal may see. Empty means nothing, deliberately.

    ``row_filters`` maps a case attribute to the values that principal may
    see, combined with AND; ``unit_ids`` is an explicit allowlist where the
    application has one. Either narrows the population; ``unrestricted_rows``
    is the explicit way to say "all rows of the runs listed here".

    ``aggregate_entitlements`` names the precomputed aggregates the principal
    may be *shown* even though they were computed over a wider population —
    an all-company baseline, say. Without one, such an aggregate is recomputed
    in the authorised population and relabelled.

    >>> policy = AccessPolicy(Principal("reviewer-1"))
    >>> policy.allows(Scope.EVIDENCE)
    False
    >>> policy.require(Scope.EVIDENCE)
    Traceback (most recent call last):
        ...
    wise.errors.AccessDenied: principal 'reviewer-1' has no 'evidence' scope
    """

    principal: Principal
    scopes: frozenset[str] = frozenset()
    runs: frozenset[str] = frozenset()
    views: frozenset[str] = frozenset()
    columns: frozenset[str] = frozenset()
    row_filters: dict[str, frozenset[str]] = field(default_factory=dict)
    unit_ids: frozenset[str] | None = None
    document_tags: frozenset[str] = frozenset()
    aggregate_entitlements: frozenset[str] = frozenset()
    unrestricted_rows: bool = False
    label: str = ""
    schema_version: str = POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "scopes", frozenset(_scope_value(s) for s in self.scopes))
        for name in ("runs", "views", "columns", "document_tags", "aggregate_entitlements"):
            object.__setattr__(self, name, frozenset(str(v) for v in getattr(self, name)))
        object.__setattr__(
            self, "row_filters", {str(k): frozenset(str(v) for v in values) for k, values in dict(self.row_filters).items()}
        )
        if self.unit_ids is not None:
            object.__setattr__(self, "unit_ids", frozenset(str(u) for u in self.unit_ids))

    # ------------------------------------------------------------- capability
    def allows(self, scope: Scope | str) -> bool:
        return _scope_value(scope) in self.scopes

    def require(self, scope: Scope | str) -> None:
        """Raise :class:`~wise.errors.AccessDenied` unless the scope is granted."""
        value = _scope_value(scope)
        if value not in self.scopes:
            raise AccessDenied(f"principal {self.principal.principal_id!r} has no {value!r} scope")

    def allows_run(self, run_id: str | None) -> bool:
        return run_id is not None and str(run_id) in self.runs

    def require_run(self, run_id: str | None) -> None:
        if not self.allows_run(run_id):
            raise AccessDenied(f"principal {self.principal.principal_id!r} is not authorised for run {run_id!r}")

    def allows_view(self, view: str) -> bool:
        return str(view) in self.views

    def require_view(self, view: str) -> None:
        if not self.allows_view(view):
            raise AccessDenied(f"principal {self.principal.principal_id!r} is not authorised for view {view!r}")

    def allows_column(self, column: str) -> bool:
        return str(column) in self.columns

    def visible_columns(self, columns: Iterable[str]) -> list[str]:
        """Only the columns the principal may see, in the order given.

        >>> policy = AccessPolicy(Principal("p"), columns=frozenset({"company"}))
        >>> policy.visible_columns(["company", "vendor", "salary"])
        ['company']
        """
        return [str(c) for c in columns if self.allows_column(str(c))]

    def require_columns(self, columns: Iterable[str]) -> list[str]:
        wanted = [str(c) for c in columns]
        denied = [c for c in wanted if not self.allows_column(c)]
        if denied:
            raise AccessDenied(f"principal {self.principal.principal_id!r} is not authorised for columns {denied}")
        return wanted

    # ------------------------------------------------------------------ rows
    def allows_unit(self, unit_id: Any, attributes: Mapping[str, Any] | None = None) -> bool:
        """Whether one unit is inside the authorised population."""
        if self.unit_ids is not None and str(unit_id) not in self.unit_ids:
            return False
        if self.row_filters:
            if attributes is None:
                return False
            return all(str(attributes.get(name, "")) in allowed for name, allowed in self.row_filters.items())
        if self.unit_ids is not None:
            return True
        return bool(self.unrestricted_rows)

    def row_mask(self, cases: pd.DataFrame) -> pd.Series:
        """The authorised rows of a case table, as a boolean mask.

        This is the single place a population is narrowed, and every aggregate
        on the assistance path is computed *after* it. With nothing granted
        the mask is all ``False``: an unauthorised principal has an empty
        population, not the whole one.

        >>> import pandas as pd
        >>> cases = pd.DataFrame({"company": ["A", "B"]}, index=["c1", "c2"])
        >>> policy = AccessPolicy(Principal("p"), row_filters={"company": frozenset({"A"})})
        >>> list(policy.row_mask(cases))
        [True, False]
        """
        if self.unrestricted_rows and not self.row_filters and self.unit_ids is None:
            return pd.Series(True, index=cases.index)
        mask = pd.Series(bool(self.unrestricted_rows or self.row_filters or self.unit_ids is not None), index=cases.index)
        if self.unit_ids is not None:
            allowed = self.unit_ids
            mask &= pd.Series([str(i) in allowed for i in cases.index], index=cases.index)
        for name, values in self.row_filters.items():
            if name not in cases.columns:
                # a filter on a column this table does not carry cannot be
                # satisfied, so it denies rather than being ignored
                return pd.Series(False, index=cases.index)
            mask &= cases[name].astype("string").fillna("").isin(sorted(values)).to_numpy()
        return mask.astype(bool)

    def authorised_records(self, records: Sequence[EvaluationRecord], cases: pd.DataFrame | None = None) -> tuple[Any, ...]:
        """Only the evaluation records of authorised units."""
        attributes: dict[str, dict[str, Any]] = {}
        if cases is not None:
            attributes = {str(unit): {str(k): v for k, v in row.items()} for unit, row in cases.to_dict(orient="index").items()}
        return tuple(r for r in records if self.allows_unit(r.unit_id, attributes.get(str(r.unit_id))))

    # ------------------------------------------------------------- documents
    def allows_document_tags(self, tags: Iterable[str]) -> bool:
        """A document is visible when *every* access tag it carries is granted.

        An untagged document is not "public by omission": it is not visible
        unless the policy grants the empty requirement explicitly by listing
        no tags on it at all. Requiring all tags rather than any is the
        conservative reading, and it is the one that cannot be widened by
        adding a tag.

        >>> policy = AccessPolicy(Principal("p"), document_tags=frozenset({"finance"}))
        >>> policy.allows_document_tags(["finance"]), policy.allows_document_tags(["finance", "hr"])
        (True, False)
        """
        return all(str(t) in self.document_tags for t in tags)

    # ------------------------------------------------------------ aggregates
    def entitled_to_aggregate(self, aggregate_id: str) -> bool:
        """Whether a precomputed wider-population aggregate may be shown as is."""
        return str(aggregate_id) in self.aggregate_entitlements

    # ------------------------------------------------------------ provenance
    def fingerprint(self) -> str:
        """A stable digest of the decision this policy encodes."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "principal": self.principal.to_dict(),
            "scopes": sorted(self.scopes),
            "runs": sorted(self.runs),
            "views": sorted(self.views),
            "columns": sorted(self.columns),
            "row_filters": {k: sorted(v) for k, v in sorted(self.row_filters.items())},
            "unit_ids": None if self.unit_ids is None else sorted(self.unit_ids),
            "document_tags": sorted(self.document_tags),
            "aggregate_entitlements": sorted(self.aggregate_entitlements),
            "unrestricted_rows": bool(self.unrestricted_rows),
        }

    def describe(self) -> str:
        rows = (
            "all rows"
            if self.unrestricted_rows and not self.row_filters
            else f"rows where {dict(sorted(self.row_filters.items()))}"
        )
        return f"{self.principal.principal_id}: scopes {sorted(self.scopes)}, views {sorted(self.views)}, {rows}"


def _scope_value(scope: Scope | str) -> str:
    return scope.value if isinstance(scope, Scope) else str(scope)


# ------------------------------------------------------------------ baselines
#: The identifier an application uses for "the comparator computed over every
#: unit of the run", which is what a restricted principal is not entitled to
#: see unless the application says so.
FULL_POPULATION_AGGREGATE = "baseline:full_population"


def authorised_population(result: ScoreResult, policy: AccessPolicy, view: str) -> pd.Series:
    """The scored, authorised rows of one view — the population every aggregate uses."""
    mask = policy.row_mask(result.cases)
    return mask & result.scores[view].notna()


def authorised_baseline(
    result: ScoreResult,
    policy: AccessPolicy,
    *,
    view: str | None = None,
    requested: BaselineSpec | None = None,
    aggregate_id: str = FULL_POPULATION_AGGREGATE,
    unit_type: str = "case",
    scope: Scope = Scope.COMPARE_GROUPS,
) -> tuple[BaselineSpec, ComparatorChange | None]:
    """A comparator this principal is entitled to, and what changed to get it.

    * With no ``requested`` comparator, the reference is recomputed over the
      authorised population — never over the whole run.
    * With a ``requested`` comparator the principal holds an entitlement for,
      it is returned unchanged.
    * Otherwise it is recomputed over the authorised population and relabelled,
      and the returned :class:`ComparatorChange` says so. The point is that the
      restricted reader sees a *different, honest* comparison rather than a
      number distilled from rows they may not see.

    ``scope`` is the capability the *caller* is exercising. It defaults to
    :attr:`Scope.COMPARE_GROUPS` because that is the tool this function was
    written for; the explanation path passes :attr:`Scope.EXPLANATION`, so a
    principal entitled to an explanation and nothing else is not asked to hold
    a comparison scope in order to be told their comparator changed.
    """
    policy.require(scope)
    view = _one_view(result, view)
    policy.require_view(view)
    policy.require_run(None if result.manifest is None else result.manifest.run_id)

    if requested is not None and policy.entitled_to_aggregate(aggregate_id):
        return requested, None

    scored = authorised_population(result, policy, view)
    n = int(scored.sum())
    if n == 0:
        raise AccessDenied(
            f"principal {policy.principal.principal_id!r} has no scored units under view {view!r}; "
            "there is no population to compute a comparator over"
        )
    contributions = result.contributions[view][scored]
    profile = {str(layer): float(contributions[layer].mean()) for layer in result.norm.layer_ids}
    run_id = None if result.manifest is None else result.manifest.run_id
    baseline_id = f"authorised-population:{view}:{policy.fingerprint()[:12]}"
    spec = BaselineSpec(
        baseline_id=baseline_id,
        kind=BaselineKind.HISTORICAL,
        reference_score=float(result.scores[view][scored].mean()),
        reference_layer_penalties=profile,
        view=view,
        scoring_mode=result.mode,
        unit_type=unit_type,
        norm_fingerprint=result.norm_fingerprint or result.norm.fingerprint(),
        population_id=f"authorised:{policy.fingerprint()[:12]}",
        population_size=n,
        source_run_id=run_id,
        description=(
            "the unweighted mean over the units this principal is authorised to see; "
            "it is not the whole population's mean and must not be read as one"
        ),
        metadata={"authorised_population": True, "policy_fingerprint": policy.fingerprint()},
    )
    if requested is None:
        return spec, None
    change = ComparatorChange(
        requested_baseline_id=requested.baseline_id,
        effective_baseline_id=baseline_id,
        reason=(
            f"the principal holds no {aggregate_id!r} entitlement, so an aggregate over the wider population could not be shown"
        ),
        authorised_population_size=n,
        requested_population_size=requested.population_size,
    )
    return spec, change


__all__ = [
    "FULL_POPULATION_AGGREGATE",
    "POLICY_SCHEMA_VERSION",
    "AccessPolicy",
    "ComparatorChange",
    "Principal",
    "Scope",
    "authorised_baseline",
    "authorised_population",
]
