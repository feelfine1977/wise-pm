"""Balance ``(bal, α_x, A_x, α_y, A_y, τ, Γ)`` (paper Sec. IV-B).

``d = |tot_x − tot_y| / max(tot_x, tot_y, ε)``, ``ν = sat(d; τ, Γ)``. The two
attributes are event columns; totals must be non-negative (Sec. IV-A).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..constraints import Balance, sat
from ..errors import NormError
from ..family import Evaluation, Family, Inputs

if TYPE_CHECKING:  # pragma: no cover
    from ..log import EventLog


def requires(c: Balance) -> Inputs:
    return Inputs(activities=c.activities(), event_columns=(c.attr_x, c.attr_y))


def evaluate(c: Balance, log: EventLog) -> Evaluation:
    tot_x = log.total(c.attr_x, c.activities_x, agg=c.agg)
    tot_y = log.total(c.attr_y, c.activities_y, agg=c.agg)
    if (tot_x < 0).any() or (tot_y < 0).any():
        raise NormError("balance totals must be non-negative (paper Sec. IV-A); net credit notes first or take absolute values")
    denom = np.maximum(np.maximum(tot_x.to_numpy(), tot_y.to_numpy()), c.eps)
    d = pd.Series((tot_x - tot_y).abs().to_numpy() / denom, index=log.case_ids)
    return Evaluation.of(sat(d, c.tau, c.width), log.case_ids)


FAMILY = Family(
    name="balance",
    constraint=Balance,
    requires=requires,
    evaluate=evaluate,
    numbers=frozenset({"tau", "width", "eps"}),
    aliases=("bal",),
)
