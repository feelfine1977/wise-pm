"""Scoring is a function of its inputs: the log it is given stays as it was (paper Sec. IV-D)."""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

import wise
import wise.derive
from _support.strategies import event_logs, norms
from wise.log import EventLog
from wise.recipes import Recipe


@pytest.mark.property
@pytest.mark.spec("IV-D")
@given(log=event_logs(), norm=norms(), derive_first=st.booleans())
def test_scoring_leaves_the_log_as_it_was_and_scores_as_on_a_log_derived_first(log, norm, derive_first):
    if derive_first:
        log.derive(norm.derived_attributes)
    events, cases, attributes = log.events, log.cases, log.case_attributes
    names = [str(r["name"]) for r in norm.derived_attributes]
    computed: list[str] = []
    compute = wise.derive.compute_recipe

    def counting(on: EventLog, recipe: Recipe) -> pd.Series:
        computed.append(recipe.name)
        return compute(on, recipe)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(wise.derive, "compute_recipe", counting)
        result = wise.score(log, norm)
        assert computed == ([] if derive_first else names)
        pd.testing.assert_frame_equal(log.events, events, check_exact=True)
        pd.testing.assert_frame_equal(log.cases, cases, check_exact=True)
        assert log.case_attributes == attributes
        # recipe state: an explicit derive reuses exactly what it reused before scoring
        computed.clear()
        log.derive(norm.derived_attributes, overwrite=False)
        assert computed == ([] if derive_first else names)

    reference = wise.score(log, norm)  # the attributes now live on the log itself
    pd.testing.assert_frame_equal(result.violations, reference.violations, check_exact=True)
    pd.testing.assert_frame_equal(result.scores, reference.scores, check_exact=True)
    pd.testing.assert_frame_equal(result.cases, reference.cases, check_exact=True)
