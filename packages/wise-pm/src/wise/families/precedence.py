"""Precedence: ``ν = sat(#{b events before the first a}; k, K)``.

A case without ``a`` is not evaluable under ``missing_a="skip"``
(:data:`~wise.family.MISSING_ANCHOR`) and a full violation under
``"violate"``. A case without ``b`` is satisfied under ``missing_b="satisfy"``
and not evaluable under ``"skip"`` (:data:`~wise.family.MISSING_ACTIVITY`).
When both are absent and both skip, the missing anchor is the reason.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..constraints import Precedence, sat
from ..family import MISSING_ACTIVITY, MISSING_ANCHOR, Evaluation, Family, Inputs

if TYPE_CHECKING:  # pragma: no cover
    from ..log import EventLog


def requires(c: Precedence) -> Inputs:
    return Inputs(activities=c.activities())


def evaluate(c: Precedence, log: EventLog) -> Evaluation:
    n_before = log.count_before(c.b, c.a)
    v = sat(n_before, c.k, c.K)
    why: list[tuple[Any, str]] = []
    if c.missing_a == "violate":
        v = v.where(n_before.notna(), 1.0)
    else:
        why.append((n_before.isna(), MISSING_ANCHOR))
    if c.missing_b == "skip":
        present_b = log.count(c.b) > 0
        v = v.where(present_b)
        why.append((~present_b, MISSING_ACTIVITY))
    return Evaluation.of(v, log.case_ids, why=why)


FAMILY = Family(
    name="precedence",
    constraint=Precedence,
    requires=requires,
    evaluate=evaluate,
    numbers=frozenset({"k", "K"}),
    aliases=("order",),
)
