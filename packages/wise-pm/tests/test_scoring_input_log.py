"""Scoring leaves its input log as it was (paper Sec. IV-D: a score is a function of the log and the norm).

The norm's derived attributes are computed for the run and returned with the
result; the event frame and the case table an :class:`~wise.EventLog` hands
out are copies, so changing them changes no later answer of the log.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

import wise
import wise.derive
from wise.errors import LogSchemaError, NormError
from wise.log import EventLog

GR = "Record Goods Receipt"
N_GR = {"name": "n_gr", "kind": "count", "activities": [GR]}
GR_SHARE = {"name": "gr_share", "kind": "ratio", "numerator": "n_gr", "denominator": "n_events"}
# running example, by hand: goods receipts A 1, B 4, C 1, D 1, E 1; events A 4, B 7, C 4, D 4, E 2
N_GR_VALUES = [1.0, 4.0, 1.0, 1.0, 1.0]
GR_SHARE_VALUES = [1 / 4, 4 / 7, 1 / 4, 1 / 4, 1 / 2]


def norm_on(*recipes: Mapping[str, object], attribute: str = "n_gr") -> wise.Norm:
    """One Metric constraint (threshold 2, width 3) on ``attribute``; B (4 receipts) scores 1 − 2/3."""
    return wise.Norm(
        (wise.NormConstraint("m", "L", wise.Metric(attribute, threshold=2, width=3)),),
        (wise.Layer("L"),),
        (wise.View("v", constraint_weights={"m": 1}),),
        derived_attributes=tuple(dict(r) for r in recipes),
    )


@pytest.fixture
def computations(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []
    compute = wise.derive.compute_recipe

    def counting(log: EventLog, recipe: dict[str, object]) -> pd.Series:
        seen.append(str(recipe["name"]))
        return compute(log, recipe)

    monkeypatch.setattr(wise.derive, "compute_recipe", counting)
    return seen


# ------------------------------------------------------------------ the input log stays as it was
@pytest.mark.spec("IV-D")
def test_scoring_leaves_events_cases_attributes_and_recipe_state_as_they_were(computations: list[str]) -> None:
    log = wise.running_p2p_log()
    events, cases, attributes = log.events, log.cases, log.case_attributes

    wise.score(log, norm_on(N_GR, GR_SHARE))

    pd.testing.assert_frame_equal(log.events, events, check_exact=True)
    pd.testing.assert_frame_equal(log.cases, cases, check_exact=True)
    assert log.case_attributes == attributes == ["flow_type", "company", "vendor"]
    assert computations == ["n_gr", "gr_share"]
    # recipe state: the log records no recipe, so an explicit derive computes both, as on a fresh log
    log.derive([N_GR, GR_SHARE], overwrite=False)
    assert computations == ["n_gr", "gr_share"] * 2


@pytest.mark.spec("IV-D")
def test_scoring_a_log_with_derived_attributes_keeps_them_and_their_record(computations: list[str]) -> None:
    log = wise.running_p2p_log()
    log.derive([N_GR])
    cases = log.cases

    result = wise.score(log, norm_on(N_GR))

    assert computations == ["n_gr"]  # derived explicitly; scoring reuses it
    assert result.log is log
    pd.testing.assert_frame_equal(log.cases, cases, check_exact=True)
    log.derive([N_GR], overwrite=False)
    assert computations == ["n_gr"]


@pytest.mark.spec("IV-D")
def test_scoring_twice_computes_once_and_recomputes_after_an_input_attribute_changes(computations: list[str]) -> None:
    log = wise.running_p2p_log()
    log.add_case_attribute("budget", pd.Series(2.0, index=log.case_ids))
    norm = norm_on(
        N_GR, {"name": "per_budget", "kind": "ratio", "numerator": "n_gr", "denominator": "budget"}, attribute="per_budget"
    )

    first = wise.score(log, norm)
    again = wise.score(log, norm)
    log.add_case_attribute("budget", pd.Series(1.0, index=log.case_ids))
    changed = wise.score(log, norm)

    assert computations == ["n_gr", "per_budget", "n_gr", "per_budget"]
    pd.testing.assert_frame_equal(first.cases, again.cases)
    assert first.cases["per_budget"].tolist() == [v / 2 for v in N_GR_VALUES]
    assert changed.cases["per_budget"].tolist() == N_GR_VALUES


# ------------------------------------------------------------------ name clashes
@pytest.mark.spec("IV-D")
@pytest.mark.parametrize("name", ["company", "n_events", "exposure_like"])
def test_a_recipe_named_like_a_case_attribute_it_did_not_produce_is_refused(name: str) -> None:
    log = wise.running_p2p_log()
    log.add_case_attribute("exposure_like", pd.Series(1.0, index=log.case_ids))
    cases = log.cases

    with pytest.raises(NormError, match=f"derived attribute '{name}' would replace the case attribute '{name}'"):
        wise.score(log, norm_on({**N_GR, "name": name}, attribute=name))

    pd.testing.assert_frame_equal(log.cases, cases, check_exact=True)


@pytest.mark.spec("IV-D")
def test_an_attribute_derived_by_another_recipe_is_refused() -> None:
    log = wise.running_p2p_log()
    log.derive([{**N_GR, "activities": ["Clear Invoice"]}])

    with pytest.raises(NormError, match="'n_gr'"):
        wise.score(log, norm_on(N_GR))

    assert log.cases["n_gr"].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]


@pytest.mark.spec("IV-D")
def test_an_attribute_replaced_after_it_was_derived_is_refused() -> None:
    log = wise.running_p2p_log()
    log.derive([N_GR])
    log.add_case_attribute("n_gr", pd.Series(0.0, index=log.case_ids))

    with pytest.raises(NormError, match="'n_gr'"):
        wise.score(log, norm_on(N_GR))

    assert (log.cases["n_gr"] == 0.0).all()


# ------------------------------------------------------------------ the result carries the derived attributes
@pytest.mark.spec("IV-D")
def test_the_result_carries_the_derived_attributes_it_used() -> None:
    log = wise.running_p2p_log()

    result = wise.score(log, norm_on(N_GR, GR_SHARE))

    assert result.cases["n_gr"].tolist() == N_GR_VALUES
    assert result.cases["gr_share"].tolist() == pytest.approx(GR_SHARE_VALUES)
    assert result.scores["v"].tolist() == pytest.approx([1.0, 1 - 2 / 3, 1.0, 1.0, 1.0])
    assert result.log is not None
    assert result.log is not log
    assert result.log.cases["n_gr"].tolist() == N_GR_VALUES
    assert result.log.case_attributes == ["flow_type", "company", "vendor", "n_gr", "gr_share"]
    assert "n_gr" not in log.cases.columns


@pytest.mark.spec("IV-E")
def test_drill_down_drivers_and_prioritisation_work_by_a_derived_attribute() -> None:
    log = wise.running_p2p_log()
    result = wise.score(log, norm_on(N_GR))

    pd.testing.assert_frame_equal(result.trace("B"), log.trace("B"))
    assert result.worst_cases("v", where={"n_gr": 4.0}).index.tolist() == ["B"]
    backlog = wise.prioritize(result, by="n_gr", view="v")
    assert backlog["n_cases"].to_dict() == {1.0: 4, 4.0: 1}
    assert backlog["mean_score"].to_dict()[4.0] == pytest.approx(1 / 3)
    assert wise.layer_drivers(result, by="n_gr", view="v").index.tolist() == [1.0, 4.0]
    assert sorted(wise.constraint_drivers(result, "v", where={"n_gr": 4.0}).index) == ["m"]


# ------------------------------------------------------------------ handed-out frames are copies
@pytest.mark.spec("IV-A")
def test_changing_the_event_frame_handed_out_changes_no_answer_of_the_log() -> None:
    log = wise.running_p2p_log()
    events = log.events
    counts, totals = log.count(GR).tolist(), log.total("amount", GR).tolist()

    handed = log.events
    handed.loc[handed.index[:8], "activity"] = "tampered"
    handed["amount"] = 1e6
    handed["extra"] = 1
    handed.index.name = "renamed"

    pd.testing.assert_frame_equal(log.events, events, check_exact=True)
    assert log.count(GR).tolist() == counts
    assert log.total("amount", GR).tolist() == totals
    assert log.trace("B")["activity"].tolist() == [
        "Create Purchase Order Item",
        *[GR] * 4,
        "Record Invoice Receipt",
        "Clear Invoice",
    ]
    with pytest.raises(AttributeError):
        log.events = handed  # type: ignore[misc]


@pytest.mark.spec("IV-A")
def test_changing_the_case_table_or_attribute_list_handed_out_changes_no_answer_of_the_log() -> None:
    log = wise.running_p2p_log()
    norm = wise.running_p2p_norm().map_constraints(lambda c: c.replace(applicability={"company": ["A"]}))
    before = wise.score(log, norm).scores

    cases = log.cases
    cases["company"] = "B"
    cases.loc["A", "n_events"] = 99
    cases["extra"] = 1.0
    log.case_attributes.append("extra")

    assert log.cases["company"].tolist() == ["A", "A", "B", "B", "B"]
    assert log.attribute("n_events").tolist() == [4, 7, 4, 4, 2]
    assert "extra" not in log.cases.columns
    assert log.case_attributes == ["flow_type", "company", "vendor"]
    pd.testing.assert_frame_equal(wise.score(log, norm).scores, before, check_exact=True)


@pytest.mark.spec("IV-A")
def test_assigning_a_case_table_still_works_with_a_deprecation_warning() -> None:
    log = wise.running_p2p_log()
    table = log.cases.assign(region=["N", "N", "S", "S", "S"])

    with pytest.warns(DeprecationWarning, match="add_case_attribute"):
        log.cases = table
    table["region"] = "changed later"

    assert log.cases["region"].tolist() == ["N", "N", "S", "S", "S"]
    with pytest.warns(DeprecationWarning, match="EventLog.cases"), pytest.raises(LogSchemaError, match="index"):
        log.cases = table.iloc[:3]


@pytest.mark.spec("IV-D")
def test_an_assigned_case_table_keeps_the_record_of_the_derived_attributes_it_leaves_alone(computations: list[str]) -> None:
    log = wise.running_p2p_log()
    log.derive([N_GR])

    with pytest.warns(DeprecationWarning, match="EventLog.cases"):
        log.cases = log.cases.assign(region="N")
    result = wise.score(log, norm_on(N_GR))
    with pytest.warns(DeprecationWarning, match="EventLog.cases"):
        log.cases = log.cases.assign(n_gr=0.0)

    assert computations == ["n_gr"]
    assert result.log is log
    with pytest.raises(NormError, match="'n_gr'"):
        wise.score(log, norm_on(N_GR))


# ------------------------------------------------------------------ explicit use keeps changing the log
@pytest.mark.spec("IV-C.norm")
def test_derive_and_add_case_attribute_change_the_log_they_are_called_on() -> None:
    log = wise.running_p2p_log()

    assert log.derive([N_GR, GR_SHARE]) == ["n_gr", "gr_share"]
    log.add_case_attribute("budget", np.array([1, 2, 3, 4, 5]))

    assert log.cases["n_gr"].tolist() == N_GR_VALUES
    assert log.attribute("budget").tolist() == [1, 2, 3, 4, 5]
    assert log.case_attributes == ["flow_type", "company", "vendor", "n_gr", "gr_share", "budget"]


@pytest.mark.spec("IV-A")
def test_with_case_attributes_returns_a_new_log_and_leaves_this_one_alone() -> None:
    log = wise.running_p2p_log()

    other = log.with_case_attributes(
        {"budget": pd.Series([5, 4, 3, 2, 1], index=log.case_ids), "flag": np.array([1, 0, 0, 0, 0])}
    )

    assert other.attribute("budget").tolist() == [5, 4, 3, 2, 1]
    assert other.case_attributes == ["flow_type", "company", "vendor", "budget", "flag"]
    assert other.count(GR).tolist() == log.count(GR).tolist()
    assert other.case_index is log.case_index
    assert "budget" not in log.cases.columns
    assert log.case_attributes == ["flow_type", "company", "vendor"]
    with pytest.raises(LogSchemaError, match="reserved"):
        log.with_case_attributes({"n_events": np.zeros(5)})


@pytest.mark.spec("IV-A")
def test_adding_an_attribute_to_a_shallow_copy_leaves_the_original_alone() -> None:
    log = wise.running_p2p_log()
    duplicate = copy.copy(log)

    duplicate.add_case_attribute("budget", np.array([1, 2, 3, 4, 5]))

    assert "budget" in duplicate.cases.columns
    assert "budget" not in log.cases.columns
    assert log.case_attributes == ["flow_type", "company", "vendor"]
