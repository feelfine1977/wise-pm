"""Exclusion ``(excl, a)``: ``ν = 1[cnt(a, σ) > 0]`` (paper Sec. IV-B).

With ``after`` / ``before`` the count is scoped to events strictly after the
first ``after`` and/or strictly before the first ``before``; a case without
such an anchor is not evaluable (:data:`~wise.family.MISSING_ANCHOR`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..constraints import Exclusion
from ..family import MISSING_ANCHOR, Evaluation, Family, Inputs

if TYPE_CHECKING:  # pragma: no cover
    from ..log import EventLog


def requires(c: Exclusion) -> Inputs:
    return Inputs(activities=c.activities())


def evaluate(c: Exclusion, log: EventLog) -> Evaluation:
    cnt = log.count_scoped(c.activity, after=c.after, before=c.before) if (c.after or c.before) else log.count(c.activity)
    v = (cnt > 0).astype(float).where(cnt.notna())
    return Evaluation.of(v, log.case_ids, why=[(cnt.isna(), MISSING_ANCHOR)])


FAMILY = Family(
    name="exclusion",
    constraint=Exclusion,
    requires=requires,
    evaluate=evaluate,
    aliases=("excl",),
)
