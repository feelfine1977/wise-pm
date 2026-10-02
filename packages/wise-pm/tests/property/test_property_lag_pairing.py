"""The ``lag`` recipe and a Lag constraint pair activations and responses by one rule (paper Sec. IV-B).

With both endpoints skipped, a Lag constraint has a value exactly where the
recipe with the same options has a lag, and that value is ``sat`` of the lag;
a case without a value is a missing activation where the case has no ``a``
and a missing response otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

import wise
import wise.derive
from _support.strategies import ACTIVITIES, event_logs
from wise.constraints import sat
from wise.families import lag as lag_family
from wise.family import EVALUATED, MISSING_ACTIVATION, MISSING_RESPONSE

_labels = st.one_of(st.sampled_from(ACTIVITIES), st.lists(st.sampled_from(ACTIVITIES), min_size=2, max_size=2, unique=True))


@pytest.mark.property
@pytest.mark.spec("IV-B.lag")
@given(
    log=event_logs(),
    a=_labels,
    b=_labels,
    activation=st.sampled_from(["first", "last"]),
    response=st.sampled_from(["first_after", "first_overall"]),
    unit=st.sampled_from(["D", "h"]),
    delta=st.integers(0, 40).map(lambda v: v / 2),
    width=st.integers(0, 40).map(lambda v: v / 2),
)
def test_the_lag_recipe_and_a_lag_constraint_pair_alike(
    log: wise.EventLog, a: Any, b: Any, activation: str, response: str, unit: str, delta: float, width: float
) -> None:
    options = {"a": a, "b": b, "activation": activation, "response": response, "unit": unit}
    recipe = wise.derive.compute_recipe(log, {"name": "lag", "kind": "lag", **options})
    constraint = wise.Lag(delta=delta, width=width, missing="skip", **options)  # type: ignore[arg-type]
    ev = wise.evaluate(constraint, log)

    has_lag = recipe.notna()
    pd.testing.assert_series_equal(ev.evaluated, has_lag, check_names=False)
    pd.testing.assert_series_equal(ev.value[has_lag], sat(recipe[has_lag], delta, width), check_names=False, check_exact=True)
    activated = (log.first_ts(a) if activation == "first" else log.last_ts(a)).notna()
    expected = np.where(has_lag, EVALUATED, np.where(activated, MISSING_RESPONSE, MISSING_ACTIVATION))
    assert ev.reason.tolist() == expected.tolist()


@pytest.mark.property
@pytest.mark.spec("IV-B.lag")
@given(log=event_logs(), activation=st.sampled_from(["first", "last", "each"]), censor=st.booleans())
def test_the_recipe_and_the_constraint_both_pair_through_the_lag_family(
    log: wise.EventLog, activation: str, censor: bool
) -> None:
    calls: list[tuple[str, str]] = []
    pairs = lag_family.pairs

    def spy(*args: Any, **kwargs: Any) -> lag_family.Pairs:
        calls.append((kwargs["activation"], kwargs["response"]))
        return pairs(*args, **kwargs)

    recipe = {"name": "lag", "kind": "lag", "a": "PO", "b": "GR", "activation": activation}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(lag_family, "pairs", spy)
        missing_b = "censor" if censor else "skip"
        wise.evaluate(wise.Lag("PO", "GR", delta=1.0, width=2.0, activation=activation, missing_b=missing_b), log)  # type: ignore[arg-type]
        if activation == "each":  # one lag per activation is not one value per case
            with pytest.raises(wise.NormError, match="activation must be one of"):
                wise.derive.compute_recipe(log, recipe)
        else:
            wise.derive.compute_recipe(log, recipe)
    assert calls == [(activation, "first_after")] + ([] if activation == "each" else [(activation, "first_after")])
