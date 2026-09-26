"""What the core refuses to compute, checked on the public surface.

The library scores deviations against a norm; it does not translate scores into
sigma levels, defect rates, labour, cost or savings, and no output table may carry
such a column (`docs/semantics/parity.md`, decision record 0001).
"""

from __future__ import annotations

import re

import pytest

import wise

FORBIDDEN = re.compile(r"sigma|dpmo|cpk|ppk|labou?r|cost|saving|fte|roi|benefit|effect_estimate", re.IGNORECASE)


@pytest.mark.spec("contract")
def test_no_public_symbol_promises_a_sigma_cost_or_effect_computation():
    offenders = [name for name in wise.__all__ if FORBIDDEN.search(name)]
    assert offenders == []


@pytest.mark.spec("contract")
def test_backlog_and_driver_tables_carry_no_monetary_or_effect_columns(p2p_result):
    backlog = wise.prioritize(p2p_result, "company", view="Finance", gamma=1.0)
    drivers = wise.layer_drivers(p2p_result, "company", view="Finance")
    constraints = wise.constraint_drivers(p2p_result, "Finance")
    for table in (backlog, drivers, constraints, p2p_result.frame("Finance"), p2p_result.summary()):
        offenders = [c for c in map(str, table.columns) if FORBIDDEN.search(c)]
        assert offenders == [], offenders
