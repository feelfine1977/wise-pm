"""Additive quantities: canonical identities, allocation, residuals and overlap.

An object-centric backlog fails in a particular way. One payment settles three
invoices; three checks each look at that payment; three groups each report "150
000 at stake"; someone adds them up and reports 450 000 of avoided loss. The
money was counted once and the exposure three times.

This module exists to make that impossible to do by accident. Every additive
quantity gets a **canonical record identity** — the object, the attribute and
the instant the value became effective — and every use of it is an
:class:`Allocation`: a named group taking a declared, non-negative *share*.
:func:`allocate` then enforces the only rule that matters:

    the shares of one record, over all the groups of one :func:`allocate`
    call, never exceed one.

**One report is the unit of conservation**, and the qualifier above is not
decoration: two separate :func:`allocate` calls can each grant a group the
whole of the same record, because neither call can see the other's
allocations. Reconcile in one call, or read the reports as the separate claims
they are.

A complete allocation has shares summing to one within a tolerance; an
incomplete one is not an error, it is a **residual**, reported by record and in
total. Negative shares, allocating one record twice inside one group and
over-allocation are rejected outright, each carrying its
:class:`AccountingIssueCode` on the raised :class:`~wise.errors.AccountingError`.

What this contract is *not*
---------------------------
It applies to additive quantities: amounts, volumes, counts of physical things.
It does **not** apply to obligations or to priority. A payment can legitimately
support several distinct checks — three obligations are three obligations —
while its monetary value is counted once under the declared accounting rule.
:meth:`AllocationReport.coverage` reports how much distinct evidence each group
rests on, and :meth:`AllocationReport.combined` refuses to add overlapping
groups' amounts: it returns the union, the naive sum and the difference between
them, so the double count is a number on the page rather than a silent error.

>>> import pandas as pd
>>> inv = QuantityRecord.build("inv1", "amount", 120.0, "EUR", pd.Timestamp("2024-01-05T10:00:00Z"))
>>> report = allocate([inv], [Allocation(inv.record_id, "vendor_review", 0.5)])
>>> report.complete, round(report.residual_total, 2), report.allocated["vendor_review"]
(False, 60.0, 60.0)
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from ..errors import AccountingError
from ..evidence.models import Qualification, QualificationCode
from .model import AttributeReading

#: How far the shares of one record may miss 1.0 and still count as complete.
DEFAULT_TOLERANCE = 1e-9


def canonical_record_id(object_id: str, attribute: str, effective_from: pd.Timestamp | None) -> str:
    """The canonical identity of one additive value.

    Object, attribute and the instant the value became effective. A changed
    amount is a *different* record, not a correction of the old one, so a
    reconciliation cannot silently consume yesterday's figure and today's.

    >>> canonical_record_id("inv1", "amount", pd.Timestamp("2024-01-05T10:00:00Z"))
    'inv1:amount@2024-01-05T10:00:00+00:00'
    >>> canonical_record_id("inv1", "amount", None)
    'inv1:amount@unknown'
    """
    stamp = "unknown" if effective_from is None else pd.Timestamp(effective_from).isoformat()
    return f"{object_id}:{attribute}@{stamp}"


@dataclass(frozen=True)
class QuantityRecord:
    """One additive amount, with the identity that keeps it from being doubled."""

    record_id: str
    object_id: str
    attribute: str
    amount: float
    unit: str
    effective_from: pd.Timestamp | None = None
    event_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", str(self.record_id))
        object.__setattr__(self, "object_id", str(self.object_id))
        object.__setattr__(self, "attribute", str(self.attribute))
        if not str(self.unit):
            raise AccountingError(f"{self.record_id}: an additive amount needs a declared unit or currency")
        object.__setattr__(self, "unit", str(self.unit))
        try:
            amount = float(self.amount)
        except (TypeError, ValueError) as exc:
            raise AccountingError(f"{self.record_id}: amount {self.amount!r} is not a number") from exc
        if not math.isfinite(amount):
            raise AccountingError(f"{self.record_id}: amount must be finite, got {self.amount!r}")
        object.__setattr__(self, "amount", amount)

    @classmethod
    def build(
        cls,
        object_id: str,
        attribute: str,
        amount: float,
        unit: str,
        effective_from: pd.Timestamp | None = None,
        *,
        event_id: str | None = None,
    ) -> QuantityRecord:
        """A record with its canonical identity derived rather than supplied."""
        return cls(
            record_id=canonical_record_id(object_id, attribute, effective_from),
            object_id=object_id,
            attribute=attribute,
            amount=amount,
            unit=unit,
            effective_from=None if effective_from is None else pd.Timestamp(effective_from),
            event_id=event_id,
        )

    @classmethod
    def from_reading(cls, reading: AttributeReading, unit: str) -> QuantityRecord | None:
        """A record from an :class:`~wise.oc.model.AttributeReading`, or ``None``.

        ``None`` when the reading found no value in force at the declared
        instant. A missing amount is missing; it is not zero.
        """
        if not reading.found:
            return None
        return cls.build(reading.object_id, reading.attribute, reading.value, unit, reading.effective_from)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "object_id": self.object_id,
            "attribute": self.attribute,
            "amount": self.amount,
            "unit": self.unit,
            "effective_from": None if self.effective_from is None else self.effective_from.isoformat(),
            "event_id": self.event_id,
        }


@dataclass(frozen=True)
class Allocation:
    """A named group taking a declared share of one record."""

    record_id: str
    group: str
    share: float = 1.0
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", str(self.record_id))
        object.__setattr__(self, "group", str(self.group))
        try:
            share = float(self.share)
        except (TypeError, ValueError) as exc:
            raise AccountingError(f"{self.group}/{self.record_id}: share {self.share!r} is not a number") from exc
        if not math.isfinite(share):
            raise AccountingError(f"{self.group}/{self.record_id}: share must be finite, got {self.share!r}")
        object.__setattr__(self, "share", share)

    def to_dict(self) -> dict[str, Any]:
        return {"record_id": self.record_id, "group": self.group, "share": self.share, "note": self.note}


class AccountingIssueCode(str, Enum):
    """What :func:`allocate` refused, or what it is telling you about.

    Every member has a producer. The first six travel on a raised
    :class:`~wise.errors.AccountingError` as its ``code``, because those six
    conditions are refusals rather than findings; only
    :attr:`INCOMPLETE_ALLOCATION` is reported, as an :class:`AccountingIssue`
    on the returned report. A declared code that nothing builds is a promise
    the module does not keep, so a member added here needs a producer with it.
    """

    NEGATIVE_SHARE = "negative_share"
    DUPLICATE_CONSUMPTION = "duplicate_consumption"
    OVER_ALLOCATION = "over_allocation"
    UNKNOWN_RECORD = "unknown_record"
    DUPLICATE_RECORD_ID = "duplicate_record_id"
    MIXED_UNITS = "mixed_units"
    INCOMPLETE_ALLOCATION = "incomplete_allocation"


@dataclass(frozen=True)
class AccountingIssue:
    """One finding, naming the record or group it is about."""

    code: AccountingIssueCode
    message: str
    record_id: str | None = None
    group: str | None = None
    amount: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", AccountingIssueCode(self.code))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "record_id": self.record_id,
            "group": self.group,
            "amount": self.amount,
        }


@dataclass(frozen=True)
class AllocationReport:
    """The result of an allocation: what each group got, and what is left over."""

    records: tuple[QuantityRecord, ...]
    allocations: tuple[Allocation, ...]
    unit: str
    allocated: dict[str, float]
    consumed: dict[str, float]
    residual: dict[str, float]
    issues: tuple[AccountingIssue, ...] = ()
    tolerance: float = DEFAULT_TOLERANCE

    _by_record: dict[str, QuantityRecord] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "allocations", tuple(self.allocations))
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(self, "_by_record", {r.record_id: r for r in self.records})

    def __repr__(self) -> str:
        state = "complete" if self.complete else f"residual {self.residual_total:,.2f} {self.unit}"
        return f"AllocationReport({len(self.records)} record(s), {len(self.groups)} group(s), {state})"

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(sorted(self.allocated))

    @property
    def total(self) -> float:
        """The sum of the records themselves — each amount exactly once."""
        return float(sum(r.amount for r in self.records))

    @property
    def residual_total(self) -> float:
        return float(sum(self.residual.values()))

    @property
    def complete(self) -> bool:
        """Every record's shares sum to one within the tolerance."""
        return not self.residual

    def record(self, record_id: str) -> QuantityRecord:
        try:
            return self._by_record[str(record_id)]
        except KeyError:
            raise AccountingError(f"unknown record {record_id!r}", code=AccountingIssueCode.UNKNOWN_RECORD) from None

    def records_of(self, group: str) -> tuple[str, ...]:
        """The distinct records one group draws on, sorted."""
        return tuple(sorted({a.record_id for a in self.allocations if a.group == group}))

    def coverage(self) -> dict[str, dict[str, float]]:
        """Per group: how much it claims, and how much distinct evidence it rests on.

        ``distinct_records`` is the point. Two groups reporting the same amount
        are not equally supported if one of them rests on a single record that
        the other also uses.
        """
        out: dict[str, dict[str, float]] = {}
        for group in self.groups:
            record_ids = self.records_of(group)
            out[group] = {
                "amount": self.allocated[group],
                "distinct_records": float(len(record_ids)),
                "distinct_objects": float(len({self.record(r).object_id for r in record_ids})),
                "share_of_total": (self.allocated[group] / self.total) if self.total else float("nan"),
            }
        return out

    def overlap(self) -> dict[tuple[str, str], dict[str, float]]:
        """For every pair of groups, the records they share and what they are worth."""
        out: dict[tuple[str, str], dict[str, float]] = {}
        groups = self.groups
        for i, left in enumerate(groups):
            for right in groups[i + 1 :]:
                shared = set(self.records_of(left)) & set(self.records_of(right))
                if shared:
                    out[(left, right)] = {
                        "n_shared_records": float(len(shared)),
                        "shared_amount": float(sum(self.record(r).amount for r in shared)),
                    }
        return out

    def combined(self, groups: Sequence[str] | None = None) -> dict[str, Any]:
        """What several groups are worth together, three ways, so the wrong one is visible.

        ``allocated`` sums the groups' *declared shares*. Because
        :func:`allocate` refuses shares above one, this number is conserved:
        adding the group totals of an allocated report can never double count,
        and that is the number a backlog should carry.

        ``evidence_value`` is the full value of the distinct records these
        groups rest on, each amount counted once — the size of the evidence,
        not of the claim.

        ``sum_if_each_group_claimed_all`` is what happens when a group reports
        the full value of everything it touches instead of its share, which is
        the usual way one payment becomes three. ``double_counted`` is the
        difference, and it comes with a typed qualification when it is not zero.
        """
        chosen = tuple(self.groups) if groups is None else tuple(str(g) for g in groups)
        unknown = [g for g in chosen if g not in self.allocated]
        if unknown:
            raise AccountingError(f"unknown group(s) {unknown}; this report holds {list(self.groups)}")
        touched: set[str] = set()
        naive = 0.0
        for group in chosen:
            record_ids = self.records_of(group)
            touched.update(record_ids)
            naive += float(sum(self.record(rid).amount for rid in record_ids))
        evidence = float(sum(self.record(rid).amount for rid in touched))
        allocated = float(sum(self.allocated[g] for g in chosen))
        qualifications: list[Qualification] = []
        difference = naive - evidence
        # magnitude, not direction. A shared credit note makes the naive sum
        # *smaller* than the evidence, so ``naive > evidence`` skipped the whole
        # family of returns, reversals and credit notes — exactly where a double
        # count is hardest to spot by eye.
        if abs(difference) > self.tolerance:
            direction = "overstating" if difference > 0 else "understating"
            qualifications.append(
                Qualification(
                    QualificationCode.OVERLAPPING_GROUPS_NOT_SUMMED,
                    f"groups {list(chosen)} share evidence: if each of them claimed the full value of every record it "
                    f"touches, the pair would report {naive:,.2f} {self.unit} where the evidence is worth "
                    f"{evidence:,.2f} {self.unit}. The difference of {difference:,.2f} {self.unit} is one amount "
                    f"counted twice, not a second amount, {direction} the combined total; the conserved figure is "
                    f"the allocated {allocated:,.2f} {self.unit}",
                    scope="group",
                )
            )
        return {
            "groups": list(chosen),
            "allocated": allocated,
            "evidence_value": evidence,
            "sum_if_each_group_claimed_all": naive,
            "double_counted": naive - evidence,
            "unit": self.unit,
            "qualifications": tuple(qualifications),
        }

    def qualifications(self) -> tuple[Qualification, ...]:
        """The limitations this allocation carries as typed qualifications."""
        out: list[Qualification] = []
        if self.residual:
            out.append(
                Qualification(
                    QualificationCode.ALLOCATION_INCOMPLETE,
                    f"{len(self.residual)} of {len(self.records)} record(s) are not fully allocated; "
                    f"{self.residual_total:,.2f} {self.unit} of {self.total:,.2f} {self.unit} is unassigned "
                    "and belongs to no group",
                    scope="group",
                )
            )
        return tuple(out)

    def frame(self) -> pd.DataFrame:
        """One row per record: amount, consumed share, residual and its groups."""
        rows = []
        for r in self.records:
            groups = sorted({a.group for a in self.allocations if a.record_id == r.record_id})
            rows.append(
                {
                    "record_id": r.record_id,
                    "object_id": r.object_id,
                    "attribute": r.attribute,
                    "amount": r.amount,
                    "unit": r.unit,
                    "consumed_share": self.consumed.get(r.record_id, 0.0),
                    "residual": self.residual.get(r.record_id, 0.0),
                    "n_groups": len(groups),
                    "groups": "|".join(groups),
                }
            )
        return pd.DataFrame(rows).set_index("record_id") if rows else pd.DataFrame()

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "total": self.total,
            "complete": self.complete,
            "residual_total": self.residual_total,
            "records": [r.to_dict() for r in self.records],
            "allocations": [a.to_dict() for a in self.allocations],
            "allocated": dict(self.allocated),
            "consumed": dict(self.consumed),
            "residual": dict(self.residual),
            "coverage": self.coverage(),
            "overlap": {f"{a}|{b}": v for (a, b), v in self.overlap().items()},
            "issues": [i.to_dict() for i in self.issues],
            "tolerance": self.tolerance,
        }


def allocate(
    records: Iterable[QuantityRecord],
    allocations: Iterable[Allocation] = (),
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    require_complete: bool = False,
    rates: Mapping[str, float] | None = None,
    unit: str | None = None,
) -> AllocationReport:
    """Allocate additive records to groups, refusing to count anything twice.

    Rejected with :class:`~wise.errors.AccountingError`: a negative share, one
    record consumed twice inside one group, shares summing above one, an
    allocation naming a record that is not here, two different records claiming
    one identity, and mixed units without ``rates``.

    Reported, not rejected: an incomplete allocation. Its residual is the part
    of each record that belongs to no group, and ``require_complete=True``
    turns that into an error for a caller who cannot use a partial answer.

    ``rates`` converts to ``unit`` (which then must be given) by multiplication:
    ``{"USD": 0.92}`` means one USD is 0.92 of the target unit. Nothing is
    converted without such a declared rule.
    """
    kept = tuple(records)
    seen: dict[str, QuantityRecord] = {}
    for record in kept:
        previous = seen.get(record.record_id)
        if previous is not None and (previous.amount != record.amount or previous.unit != record.unit):
            raise AccountingError(
                f"record id {record.record_id!r} is claimed by two different values "
                f"({previous.amount} {previous.unit} and {record.amount} {record.unit}); "
                "a canonical identity is an identity, not a label",
                code=AccountingIssueCode.DUPLICATE_RECORD_ID,
            )
        seen[record.record_id] = record
    kept = tuple(seen.values())

    kept, target_unit = _one_unit(kept, rates=rates, unit=unit)

    consumed: dict[str, float] = {}
    per_group_records: dict[tuple[str, str], Allocation] = {}
    allocated: dict[str, float] = {}
    known = {r.record_id: r for r in kept}
    entries = tuple(allocations)
    for entry in entries:
        if entry.record_id not in known:
            raise AccountingError(
                f"allocation {entry.group!r} names record {entry.record_id!r}, which is not among the "
                f"{len(known)} record(s) supplied; an allocation cannot create the amount it consumes",
                code=AccountingIssueCode.UNKNOWN_RECORD,
            )
        if entry.share < 0:
            raise AccountingError(
                f"allocation {entry.group!r} of {entry.record_id!r} has share {entry.share}; "
                "a negative share is a credit note, and netting one needs a declared rule",
                code=AccountingIssueCode.NEGATIVE_SHARE,
            )
        key = (entry.group, entry.record_id)
        if key in per_group_records:
            raise AccountingError(
                f"group {entry.group!r} consumes record {entry.record_id!r} twice; "
                "one source amount must not be repeatedly consumed in the same reconciliation",
                code=AccountingIssueCode.DUPLICATE_CONSUMPTION,
            )
        per_group_records[key] = entry
        consumed[entry.record_id] = consumed.get(entry.record_id, 0.0) + entry.share
        allocated[entry.group] = allocated.get(entry.group, 0.0) + entry.share * known[entry.record_id].amount

    issues: list[AccountingIssue] = []
    residual: dict[str, float] = {}
    for record in kept:
        taken = consumed.get(record.record_id, 0.0)
        if taken > 1.0 + tolerance:
            raise AccountingError(
                f"record {record.record_id!r} is allocated {taken:.6f} of itself across "
                f"{sum(1 for (_, rid) in per_group_records if rid == record.record_id)} group(s); "
                "over-allocation means the same amount is counted more than once",
                code=AccountingIssueCode.OVER_ALLOCATION,
            )
        if taken < 1.0 - tolerance:
            residual[record.record_id] = (1.0 - taken) * record.amount
    if residual:
        issues.append(
            AccountingIssue(
                AccountingIssueCode.INCOMPLETE_ALLOCATION,
                f"{len(residual)} record(s) are not fully allocated; "
                f"{sum(residual.values()):,.2f} {target_unit} belongs to no group",
                amount=float(sum(residual.values())),
            )
        )
        if require_complete:
            raise AccountingError(
                f"a complete allocation was required but {len(residual)} record(s) are short by "
                f"{sum(residual.values()):,.2f} {target_unit}; allocate the remainder or accept the residual",
                code=AccountingIssueCode.INCOMPLETE_ALLOCATION,
            )
    return AllocationReport(
        records=kept,
        allocations=entries,
        unit=target_unit,
        allocated=allocated,
        consumed=consumed,
        residual=residual,
        issues=tuple(issues),
        tolerance=tolerance,
    )


def _one_unit(
    records: Sequence[QuantityRecord], *, rates: Mapping[str, float] | None, unit: str | None
) -> tuple[tuple[QuantityRecord, ...], str]:
    """All records in one unit, converting only by a declared rate."""
    units = sorted({r.unit for r in records})
    if not units:
        return tuple(records), str(unit or "")
    if rates is None:
        if len(units) > 1:
            raise AccountingError(
                f"records are in {units} and no conversion was declared; adding them would invent an exchange rate. "
                "Pass rates={'USD': 0.92, ...} with a target unit, or reconcile each unit separately",
                code=AccountingIssueCode.MIXED_UNITS,
            )
        if unit is not None and unit != units[0]:
            raise AccountingError(
                f"records are in {units[0]!r} but the target unit is {unit!r} and no rate was declared",
                code=AccountingIssueCode.MIXED_UNITS,
            )
        return tuple(records), units[0]
    if unit is None:
        raise AccountingError(
            "a conversion rate table needs the unit it converts to; pass unit=...",
            code=AccountingIssueCode.MIXED_UNITS,
        )
    converted: list[QuantityRecord] = []
    for record in records:
        if record.unit == unit:
            converted.append(record)
            continue
        if record.unit not in rates:
            raise AccountingError(
                f"no declared rate from {record.unit!r} to {unit!r} for record {record.record_id!r}",
                code=AccountingIssueCode.MIXED_UNITS,
            )
        rate = float(rates[record.unit])
        if not math.isfinite(rate) or rate <= 0:
            raise AccountingError(
                f"the declared rate from {record.unit!r} to {unit!r} must be finite and positive",
                code=AccountingIssueCode.MIXED_UNITS,
            )
        converted.append(
            QuantityRecord(
                record_id=record.record_id,
                object_id=record.object_id,
                attribute=record.attribute,
                amount=record.amount * rate,
                unit=unit,
                effective_from=record.effective_from,
                event_id=record.event_id,
            )
        )
    return tuple(converted), unit


def equal_split(records: Sequence[QuantityRecord], groups: Sequence[str]) -> tuple[Allocation, ...]:
    """Split every record equally over ``groups`` — a complete allocation.

    A convenience for the common declared rule "these n reviews share this
    amount". It is a *rule*, chosen by the caller; nothing here decides that an
    equal split is the right one.

    >>> r = QuantityRecord.build("inv1", "amount", 90.0, "EUR")
    >>> [round(a.share, 4) for a in equal_split([r], ["a", "b", "c"])]
    [0.3333, 0.3333, 0.3333]
    """
    if not groups:
        raise AccountingError("an equal split needs at least one group")
    share = 1.0 / len(groups)
    return tuple(
        Allocation(r.record_id, g, share, note=f"equal split over {len(groups)} group(s)") for r in records for g in groups
    )


__all__ = [
    "DEFAULT_TOLERANCE",
    "AccountingIssue",
    "AccountingIssueCode",
    "Allocation",
    "AllocationReport",
    "QuantityRecord",
    "allocate",
    "canonical_record_id",
    "equal_split",
]
