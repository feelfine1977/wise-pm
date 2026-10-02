"""Metric: the saturation rule on a numeric case attribute.

``ν = sat(att(σ); ϑ, W)`` (``direction="high"``) or ``sat(ϑ − att(σ); 0, W)``
(``"low"``). A case whose attribute is missing or not numeric is not
evaluable (:data:`~wise.family.MISSING_ATTRIBUTE`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..constraints import Metric, sat
from ..family import MISSING_ATTRIBUTE, Evaluation, Family, Inputs

if TYPE_CHECKING:  # pragma: no cover
    from ..log import EventLog


def requires(c: Metric) -> Inputs:
    return Inputs(case_attributes=(c.attribute,))


def evaluate(c: Metric, log: EventLog) -> Evaluation:
    x = log.attribute(c.attribute)
    v = sat(x, c.threshold, c.width) if c.direction == "high" else sat(c.threshold - x, 0.0, c.width)
    return Evaluation.of(v, log.case_ids, why=[(x.isna(), MISSING_ATTRIBUTE)])


FAMILY = Family(
    name="metric",
    constraint=Metric,
    requires=requires,
    evaluate=evaluate,
    numbers=frozenset({"threshold", "width"}),
)
