"""The built-in constraint families, one module each, registered on import.

Each module defines the evaluation of one constraint type of
:mod:`wise.constraints` and its :class:`~wise.family.Family`; the tuple below
is the registry entry. The order is the order of
:data:`wise.constraints.CONSTRAINT_TYPES`.
"""

from __future__ import annotations

from typing import Any

from ..family import Family, register
from . import balance, exclusion, lag, metric, precedence, presence, singularity

BUILT_IN: tuple[Family[Any], ...] = (
    presence.FAMILY,
    exclusion.FAMILY,
    singularity.FAMILY,
    lag.FAMILY,
    precedence.FAMILY,
    balance.FAMILY,
    metric.FAMILY,
)

for _family in BUILT_IN:
    register(_family)

__all__ = ["BUILT_IN"]
