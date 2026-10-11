"""The reason table of a score result is the reason of each constraint evaluated on its own (Hypothesis).

``wise.score`` keeps, per case and constraint, the reason of the evaluation it
scored; calling ``wise.evaluate`` on the same log for one constraint at a time
must give the same reasons, for every family, under both semantics profiles.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
import warnings
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
from hypothesis import given, note
from hypothesis import strategies as st

import wise
from _support.strategies import ACTIVITIES, CONSTRAINT_TYPES, FLOWS, REGIONS, build_log, constraints, norms

# Schema version 2 documents are read here; each read announces the 0.1.0 semantics (LegacySemanticsWarning).
pytestmark = pytest.mark.filterwarnings("ignore::wise.LegacySemanticsWarning")

PKG = Path(__file__).resolve().parents[2]

_event = st.tuples(st.sampled_from(ACTIVITIES), st.integers(0, 120).map(lambda v: v / 2), st.integers(0, 400).map(lambda v: v / 4))
# a case may lack its flow, so that applicability is unknown under schema version 3
_case = st.tuples(st.sampled_from([*FLOWS, None]), st.sampled_from(REGIONS), st.lists(_event, min_size=1, max_size=8))


@st.composite
def logs_with_missing_flows(draw: st.DrawFn) -> wise.EventLog:
    cases = draw(st.lists(_case, min_size=1, max_size=40))
    note(f"log cases: {cases}")
    return build_log(cases)


def assert_reasons_match_evaluate(result: wise.ScoreResult) -> None:
    assert result.reasons is not None
    assert result.log is not None
    assert list(result.reasons.columns) == result.norm.constraint_ids
    assert result.reasons.index.equals(result.violations.index)
    for nc in result.norm.constraints:
        ev = wise.evaluate(nc, result.log, semantics=result.norm.semantics)
        pd.testing.assert_series_equal(result.reasons[nc.id], ev.reason, check_names=False)
        assert result.reasons[nc.id].isin(wise.family.VALUED).equals(result.violations[nc.id].notna())


@pytest.mark.property
@pytest.mark.spec("IV-D")
@pytest.mark.parametrize("schema_version", [2, 3])
@pytest.mark.parametrize("kind", CONSTRAINT_TYPES)
@given(data=st.data(), log=logs_with_missing_flows())
def test_the_reason_table_equals_the_reasons_of_each_constraint_evaluated_alone(
    kind: str, schema_version: int, data: st.DataObject, log: wise.EventLog
) -> None:
    norm = data.draw(norms())
    # the first constraint is replaced by one of the family under test, so that every family is covered
    first = norm.constraints[0].replace(constraint=data.draw(constraints(kind)))
    norm = dataclasses.replace(norm, constraints=(first, *norm.constraints[1:]), schema_version=schema_version)
    assert_reasons_match_evaluate(wise.score(log, norm))


@pytest.mark.spec("IV-D")
def test_the_reason_table_matches_on_the_running_example_and_the_golden_dataset() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wise.LegacySemanticsWarning)
        running = [wise.Norm.load(PKG / "examples" / name) for name in ("running_p2p_norm.json", "running_p2p_norm_v3.json")]
        golden_norm = wise.Norm.load(PKG / "tests" / "data" / "golden" / "norm.json")
    for norm in running:
        assert_reasons_match_evaluate(wise.score(wise.running_p2p_log(), norm))
    golden_log = _generator().load_event_log(PKG / "tests" / "data" / "p2p_synthetic_2k.parquet")
    for norm in (golden_norm, wise.migrate(golden_norm.to_dict()).norm):
        assert_reasons_match_evaluate(wise.score(golden_log, norm))


def _generator() -> ModuleType:
    path = PKG / "scripts" / "make_golden.py"
    spec = importlib.util.spec_from_file_location("make_golden", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
