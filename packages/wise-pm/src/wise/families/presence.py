"""Presence ``(pres, a, m)``: ``ν = 1 − min(cnt(a, σ) / m, 1)`` (paper Sec. IV-B).

Every case is evaluated; a case without ``a`` has ``ν = 1``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..constraints import Presence
from ..family import Evaluation, Family, Inputs

if TYPE_CHECKING:  # pragma: no cover
    from ..log import EventLog


def requires(c: Presence) -> Inputs:
    return Inputs(activities=c.activities())


def evaluate(c: Presence, log: EventLog) -> Evaluation:
    cnt = log.count(c.activity)
    return Evaluation.of(1.0 - np.minimum(cnt.to_numpy() / c.m, 1.0), log.case_ids)


FAMILY = Family(
    name="presence",
    constraint=Presence,
    requires=requires,
    evaluate=evaluate,
    numbers=frozenset({"m"}),
    aliases=("pres",),
)
