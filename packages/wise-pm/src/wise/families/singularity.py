"""Singularity ``(sing, a, k, K)``: ``ν = sat(cnt(a, σ); k, K)`` (paper Sec. IV-B).

``after`` / ``before`` scope the count as for Exclusion; a case without such
an anchor is not evaluable (:data:`~wise.family.MISSING_ANCHOR`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..constraints import Singularity, sat
from ..family import MISSING_ANCHOR, Evaluation, Family, Inputs

if TYPE_CHECKING:  # pragma: no cover
    from ..log import EventLog


def requires(c: Singularity) -> Inputs:
    return Inputs(activities=c.activities())


def evaluate(c: Singularity, log: EventLog) -> Evaluation:
    cnt = log.count_scoped(c.activity, after=c.after, before=c.before) if (c.after or c.before) else log.count(c.activity)
    return Evaluation.of(sat(cnt, c.k, c.K), log.case_ids, why=[(cnt.isna(), MISSING_ANCHOR)])


FAMILY = Family(
    name="singularity",
    constraint=Singularity,
    requires=requires,
    evaluate=evaluate,
    numbers=frozenset({"k", "K"}),
    aliases=("sing",),
)
