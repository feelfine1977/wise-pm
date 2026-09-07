"""One kernel, two evaluators: the case-only special case (O10, O09, O03).

The claim is narrow and checkable. Take the paper's running example, express it
as an object log where each case is an object, and state the two obligations
the native catalogue can express — the goods-receipt-to-invoice lag and the
delivery-fragmentation limit — as a :class:`~wise.oc.constraints.CrossObjectLag`
reached *through a relation* and a
:class:`~wise.oc.constraints.RelatedObjectCardinality` counting *objects*. The
violations, the effective weights, the layer contributions and the scores must
then be identical to what :func:`wise.score` produces, in both modes and both
views, because both went through :mod:`wise._aggregation`.

The rest of the file pins the contract of the typed result: it is not a
:class:`~wise.scoring.ScoreResult`, it never carries an object log where a case
log is promised, it refuses to rank two unit types as one backlog, and it hands
the existing prioritisation path a frame it already knows how to read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _oc_fixtures import (
    case_shaped,
    case_shaped_spec,
    invoice_spec,
    legacy_case_norm,
    legacy_object_norm,
    partial_deliveries,
    shared_payment,
    wide_vendor,
)

import wise
from wise import oc
from wise.errors import OCConstraintError, OCUnitError
from wise.evidence.models import EvaluationRecord, QualificationCode, ReasonCode
from wise.norm import Layer, View

MODES = ("flat", "layer_balanced")


def identical(a, b) -> None:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.array_equal(np.isnan(a), np.isnan(b)), "the NaN mask moved"
    np.testing.assert_array_max_ulp(a[~np.isnan(a)], b[~np.isnan(b)], maxulp=0)


@pytest.fixture
def legacy():
    """The same five cases, once as a case log and once as an object log."""
    oc_log = case_shaped()
    spec = case_shaped_spec()
    return wise.running_p2p_log(), oc_log, spec, oc.build_units(oc_log, spec)


# ================================================================= O10, the parity
@pytest.mark.parametrize("mode", MODES)
def test_the_native_path_reproduces_the_legacy_violations_exactly(legacy, mode):
    case_log, oc_log, spec, units = legacy
    case_result = wise.score(case_log, legacy_case_norm(mode))
    oc_result = oc.score_units(oc_log, legacy_object_norm(mode), units, spec=spec)
    assert list(oc_result.violations.columns) == list(case_result.violations.columns)
    identical(oc_result.violations.to_numpy(), case_result.violations.to_numpy())


@pytest.mark.parametrize("mode", MODES)
def test_the_native_path_reproduces_the_legacy_scores_exactly(legacy, mode):
    case_log, oc_log, spec, units = legacy
    case_result = wise.score(case_log, legacy_case_norm(mode))
    oc_result = oc.score_units(oc_log, legacy_object_norm(mode), units, spec=spec)
    assert oc_result.views == case_result.views
    identical(oc_result.scores.to_numpy(), case_result.scores.to_numpy())
    for view in case_result.views:
        identical(oc_result.contributions[view].to_numpy(), case_result.contributions[view].to_numpy())
        identical(oc_result.effective_weights(view).to_numpy(), case_result.effective_weights(view).to_numpy())
        identical(oc_result.penalties(view).to_numpy(), case_result.penalties(view).to_numpy())


def test_the_reproduced_numbers_are_the_ones_a_reader_can_check(legacy):
    """Spelled out, so the parity is not two wrongs agreeing."""
    _, oc_log, spec, units = legacy
    result = oc.score_units(oc_log, legacy_object_norm("flat"), units, spec=spec)
    violations = result.violations.round(6).to_dict()
    assert violations["c2"] == {
        "case_review:A": 0.75,  # 25 days against delta 10, width 20
        "case_review:B": 0.0,  # 8 days
        "case_review:C": 0.0,  # 7 days
        "case_review:D": 0.0,  # 5 days
        "case_review:E": 1.0,  # no invoice at all, under missing_response='violate'
    }
    assert violations["c5"] == {
        "case_review:A": 0.0,
        "case_review:B": 0.666667,  # four goods receipts against k=2, K=3
        "case_review:C": 0.0,
        "case_review:D": 0.0,
        "case_review:E": 0.0,
    }


def test_the_lag_reached_its_activation_through_a_relation_not_a_case_column(legacy):
    """The gate: the native check must actually use the object relations."""
    _, oc_log, spec, units = legacy
    unit = next(u for u in units if u.anchor_id == "B")
    assert len(unit.role("receipts")) == 4, "four goods-receipt objects, reached by traversal"
    witness = unit.witnesses_for("receipts")[0]
    assert witness.describe().startswith("B <-[receipt for]-"), "the binding is justified by the relation it walked"
    result = oc.score_units(oc_log, legacy_object_norm("flat"), units, spec=spec)
    record = result.record("case_review:B", "c2")
    assert record.policies["match"] == "first_after"
    assert {w.event_id for w in record.witnesses if w.event_id} == {"e008", "e012"}


def test_the_decomposition_holds_on_both_sides(legacy):
    case_log, oc_log, spec, units = legacy
    for mode in MODES:
        assert wise.score(case_log, legacy_case_norm(mode)).check_decomposition() <= 1e-12
        assert oc.score_units(oc_log, legacy_object_norm(mode), units, spec=spec).check_decomposition() <= 1e-12


def test_a_backlog_over_the_object_result_matches_the_backlog_over_the_case_result(legacy):
    """The same comparator, the same shrinkage, the same ranking."""
    case_log, oc_log, spec, units = legacy
    case_result = wise.score(case_log, legacy_case_norm("flat"))
    oc_result = oc.score_units(oc_log, legacy_object_norm("flat"), units, spec=spec)
    case_backlog = wise.prioritize(case_result, by="case", view="Finance", as_index=False)
    oc_backlog = oc.object_backlog(oc_result, by="anchor_id", view="Finance", as_index=False)
    identical(case_backlog["stable_PI"].to_numpy(), oc_backlog["stable_PI"].to_numpy())
    assert list(case_backlog["case"]) == list(oc_backlog["anchor_id"])
    assert oc_backlog.attrs["unit_type"] == "case_review"
    assert oc_backlog.attrs["oc_run_id"] == oc_result.run_id


# ======================================================== the typed result contract
def invoice_catalogue(**changes) -> oc.ObjectNorm:
    payload = {
        "constraints": (
            oc.ObjectConstraint(
                "m1",
                "matching",
                oc.RelatedObjectCardinality(role="order", minimum=1, completeness="assumed_complete"),
                description="every invoice belongs to an order",
            ),
            oc.ObjectConstraint("h1", "handling", oc.RelatedObjectCardinality(role="receipts", maximum=1, width=3)),
        ),
        "layers": (Layer("matching"), Layer("handling")),
        "views": (View("Finance", constraint_weights={"m1": 0.7, "h1": 0.3}),),
        "name": "invoice review",
    }
    payload.update(changes)
    return oc.ObjectNorm(**payload)


def test_the_object_result_is_not_a_score_result_and_carries_no_case_log():
    log = partial_deliveries(3)
    units = oc.build_units(log, invoice_spec(), strict=False)
    result = oc.score_units(log, invoice_catalogue(), units)
    assert not isinstance(result, wise.ScoreResult)
    assert isinstance(result.oc_log, oc.OCEventLog)
    assert not hasattr(result, "log"), "ScoreResult.log promises an EventLog; this result does not have one to give"
    case_result = wise.score(wise.running_p2p_log(), wise.running_p2p_norm())
    assert isinstance(case_result.log, wise.EventLog)
    assert not isinstance(case_result.log, oc.OCEventLog)


def test_the_result_reports_its_identities_and_its_shape():
    log = partial_deliveries(3)
    units = oc.build_units(log, invoice_spec(), strict=False)
    norm = invoice_catalogue()
    result = oc.score_units(log, norm, units, run_id="ocrun-fixed")
    assert result.run_id == "ocrun-fixed"
    assert result.norm_fingerprint == norm.fingerprint()
    assert result.log_fingerprint == log.content_fingerprint()
    assert result.wise_version == wise.__version__
    assert result.unit_types == ("invoice_review",)
    assert result.complete_share() == 1.0
    assert "invoice_review" in repr(result)
    assert list(result.units.columns) == [
        "unit_type",
        "anchor_id",
        "anchor_type",
        "complete",
        "n_events",
        "n_bound",
        "evaluation_time",
    ]


def test_the_frame_is_the_shape_the_prioritisation_path_already_reads():
    log = partial_deliveries(3)
    units = oc.build_units(log, invoice_spec(), strict=False)
    result = oc.score_units(log, invoice_catalogue(), units)
    frame = result.frame("Finance")
    assert "score" in frame.columns
    assert {"contrib__matching", "contrib__handling"} <= set(frame.columns)
    assert frame.index.name == "unit_id"
    every_view = result.frame()
    assert "score__Finance" in every_view.columns
    assert "contrib__Finance__matching" in every_view.columns
    backlog = wise.prioritize(frame, by="anchor_type", view=None, as_index=False)
    assert backlog["n_cases"].sum() == len(units)


def test_ranking_stays_inside_one_unit_type_by_default():
    """O09: a count of invoices and a count of orders are not one volume."""
    log = partial_deliveries(3)
    invoices = oc.build_units(log, invoice_spec(), strict=False)
    order_spec = oc.UnitSpec(
        unit_type="order_review",
        anchor_type="purchase_order",
        roles=(oc.RolePath("receipts", (oc.PathStep("receipt for", target_type="goods_receipt", direction="reverse"),)),),
    )
    orders = oc.build_units(log, order_spec, strict=False)
    norm = invoice_catalogue(
        constraints=(oc.ObjectConstraint("h1", "handling", oc.RelatedObjectCardinality(role="receipts", maximum=1, width=3)),),
        layers=(Layer("handling"),),
        views=(View("Finance", constraint_weights={"h1": 1.0}),),
    )
    mixed = oc.score_units(log, norm, [*invoices, *orders])
    assert mixed.unit_types == ("invoice_review", "order_review")
    with pytest.raises(OCConstraintError, match="Ranking across them would treat"):
        mixed.frame("Finance")
    assert len(mixed.frame("Finance", unit_type="invoice_review")) == 1
    assert len(mixed.frame("Finance", allow_mixed=True)) == 2, "pooling is possible, but only on purpose"
    codes = {q.code for q in mixed.qualifications()}
    assert QualificationCode.HETEROGENEOUS_UNIT_TYPES in codes
    with pytest.raises(OCConstraintError, match="unknown unit type"):
        mixed.frame("Finance", unit_type="nope")


def test_the_backlog_helper_validates_the_grouping_before_reusing_prioritize():
    log = partial_deliveries(3)
    units = oc.build_units(log, invoice_spec(), strict=False)
    result = oc.score_units(log, invoice_catalogue(), units)
    with pytest.raises(OCConstraintError, match="unknown slice key"):
        oc.object_backlog(result, by="department", view="Finance")
    with pytest.raises(OCConstraintError, match="volume='exposure' needs an exposure column"):
        oc.object_backlog(result, by="anchor_type", view="Finance", volume="exposure")


def test_a_backlog_over_truncated_contexts_says_so():
    """The qualification must survive the step into the artefact a reader receives.

    One vendor whose forty invoices were cut to five ranks through
    ``object_backlog``. The frame's ``complete`` column knows the context was
    cut; ``attrs`` is what ``wise.prioritize`` hands on, so the count travels
    there or it travels nowhere.
    """
    log = wide_vendor(40)
    spec = oc.UnitSpec(
        unit_type="vendor_review",
        anchor_type="vendor",
        roles=(oc.RolePath("invoices", (oc.PathStep("billed by", target_type="invoice", direction="reverse"),)),),
        limits=oc.TraversalLimits(max_fan_out=5),
    )
    units = oc.build_units(log, spec)
    norm = oc.ObjectNorm(
        constraints=(
            # one check the cut leaves evaluable, so the unit is still scored
            # and still ranks — and one it does not
            oc.ObjectConstraint("m1", "matching", oc.RelatedObjectCardinality(role="invoices", minimum=1)),
            oc.ObjectConstraint("h1", "handling", oc.RelatedObjectCardinality(role="invoices", maximum=10)),
        ),
        layers=(Layer("matching"), Layer("handling")),
        views=(View("Finance", constraint_weights={"m1": 0.5, "h1": 0.5}),),
        name="vendor review",
    )
    result = oc.score_units(log, norm, units, spec=spec)
    assert not result.units["complete"].all()
    assert result.record("vendor_review:v1", "h1").reason_code is ReasonCode.BUDGET_TRUNCATED

    backlog = oc.object_backlog(result, by="anchor_type", view="Finance", as_index=False)
    assert backlog["mean_score"].iloc[0] == pytest.approx(1.0), "the surviving check passes, so the slice looks clean"
    assert backlog.attrs["unit_type"] == "vendor_review"
    assert backlog.attrs["oc_run_id"] == result.run_id
    assert backlog.attrs["context_truncated"] == 1, "the cut context is counted in the artefact a reader receives"
    assert backlog.attrs["context_truncated_share"] == pytest.approx(1.0)


def test_a_backlog_over_whole_contexts_reports_none_truncated():
    log = partial_deliveries(3)
    units = oc.build_units(log, invoice_spec(), strict=False)
    result = oc.score_units(log, invoice_catalogue(), units)
    backlog = oc.object_backlog(result, by="anchor_type", view="Finance")
    assert backlog.attrs["context_truncated"] == 0
    assert backlog.attrs["context_truncated_share"] == pytest.approx(0.0)


def test_an_exposure_column_comes_from_the_accounting_contract():
    log = shared_payment((100.0, 60.0, 40.0))
    at = "2024-06-01T00:00:00Z"
    units = oc.build_units(log, invoice_spec(), at=at, strict=False)
    payment = oc.QuantityRecord.from_reading(log.value_at("pay1", "amount", at), "EUR")
    assert payment is not None
    invoices = {u.unit_id: oc.QuantityRecord.from_reading(log.value_at(u.anchor_id, "amount", at), "EUR") for u in units}
    report = oc.allocate(
        [payment],
        [oc.Allocation(payment.record_id, uid, rec.amount / payment.amount) for uid, rec in invoices.items()],
    )
    assert report.complete
    result = oc.score_units(log, invoice_catalogue(), units, exposure=report.allocated)
    frame = result.frame("Finance")
    assert frame["exposure"].sum() == pytest.approx(200.0), "the backlog's volume is the payment, once"
    backlog = oc.object_backlog(result, by="anchor_type", view="Finance", volume="exposure")
    assert backlog["volume"].sum() == pytest.approx(200.0)


def test_an_out_of_scope_check_is_recorded_as_out_of_scope_not_as_satisfied():
    log = partial_deliveries(3)
    units = oc.build_units(log, invoice_spec(), strict=False)
    norm = invoice_catalogue(
        constraints=(
            oc.ObjectConstraint(
                "m1", "matching", oc.RelatedObjectCardinality(role="order", minimum=1), unit_types=("order_review",)
            ),
            oc.ObjectConstraint("h1", "handling", oc.RelatedObjectCardinality(role="receipts", maximum=1, width=3)),
        )
    )
    result = oc.score_units(log, norm, units)
    assert result.in_scope.loc["invoice_review:inv1", "m1"] is np.False_ or not result.in_scope.loc["invoice_review:inv1", "m1"]
    assert pd.isna(result.violations.loc["invoice_review:inv1", "m1"])
    record = result.record("invoice_review:inv1", "m1")
    assert record.reason_code is ReasonCode.OUT_OF_SCOPE
    assert record.violation is None
    assert result.scores.loc["invoice_review:inv1", "Finance"] == pytest.approx(1.0 - 2 / 3), (
        "the only applicable check carries the whole weight; the out-of-scope one is dropped, not satisfied"
    )


def test_every_evaluation_produces_a_record_that_can_stand_alone():
    log = partial_deliveries(3)
    units = oc.build_units(log, invoice_spec(), strict=False)
    result = oc.score_units(log, invoice_catalogue(), units)
    assert len(result.records) == len(units) * len(result.norm.constraints)
    assert all(isinstance(r, EvaluationRecord) for r in result.records)
    assert {r.run_id for r in result.records} == {result.run_id}
    assert {r.unit_type for r in result.records} == {"invoice_review"}
    frame = result.records_frame()
    assert set(frame["constraint_id"]) == {"m1", "h1"}
    assert frame["reason_code"].tolist() == ["observed", "observed"]
    round_tripped = [EvaluationRecord.from_dict(r.to_dict()) for r in result.records]
    assert round_tripped == list(result.records)
    with pytest.raises(OCConstraintError, match="no record for unit"):
        result.record("invoice_review:nope", "m1")


def test_a_unit_with_no_applicable_check_is_unscored_and_not_zero():
    log = partial_deliveries(3)
    units = oc.build_units(log, invoice_spec(), strict=False)
    norm = invoice_catalogue(
        constraints=(
            oc.ObjectConstraint(
                "m1", "matching", oc.RelatedObjectCardinality(role="order", minimum=1), unit_types=("order_review",)
            ),
        ),
        layers=(Layer("matching"),),
        views=(View("Finance", constraint_weights={"m1": 1.0}),),
    )
    result = oc.score_units(log, norm, units)
    assert result.scores["Finance"].isna().all()
    assert list(result.unscored("Finance")) == ["invoice_review:inv1"]
    assert result.summary().loc["Finance", "n_scored"] == 0


# ============================================================ the catalogue envelope
def test_the_catalogue_has_its_own_versioned_envelope_and_round_trips():
    norm = invoice_catalogue()
    payload = norm.to_dict()
    assert payload["schema"] == "wise-oc-norm/1"
    assert oc.ObjectNorm.from_dict(payload).fingerprint() == norm.fingerprint()
    assert norm.dumps().startswith("{")
    assert norm.fingerprint() != wise.running_p2p_norm().fingerprint()


def test_the_catalogue_refuses_a_foreign_envelope_and_stray_keys():
    with pytest.raises(OCConstraintError, match="is not 'wise-oc-norm/1'"):
        oc.ObjectNorm.from_dict({"schema": "wise-norm/2", "layers": [], "views": [], "constraints": []})
    with pytest.raises(OCConstraintError, match="unknown top-level key"):
        oc.ObjectNorm.from_dict({**invoice_catalogue().to_dict(), "scoring_engine": "fast"})
    with pytest.raises(OCConstraintError, match="unknown key"):
        oc.ObjectConstraint.from_dict({"id": "m1", "layer": "l", "type": "related_object_cardinality", "params": {}, "x": 1})


def test_the_catalogue_validates_itself_the_way_the_norm_does():
    good = oc.RelatedObjectCardinality(role="order", minimum=1)
    with pytest.raises(OCConstraintError, match="duplicate object constraint id"):
        oc.ObjectNorm(
            constraints=(oc.ObjectConstraint("m1", "l", good), oc.ObjectConstraint("m1", "l", good)),
            layers=(Layer("l"),),
            views=(View("V", constraint_weights={"m1": 1.0}),),
        )
    with pytest.raises(OCConstraintError, match="undeclared layer"):
        oc.ObjectNorm(
            constraints=(oc.ObjectConstraint("m1", "elsewhere", good),),
            layers=(Layer("l"),),
            views=(View("V", constraint_weights={"m1": 1.0}),),
        )
    with pytest.raises(OCConstraintError, match="weights unknown check"):
        oc.ObjectNorm(
            constraints=(oc.ObjectConstraint("m1", "l", good),),
            layers=(Layer("l"),),
            views=(View("V", constraint_weights={"m9": 1.0}),),
        )
    with pytest.raises(OCConstraintError, match="needs at least one layer"):
        oc.ObjectNorm(constraints=(), layers=(), views=(View("V", constraint_weights={"m1": 1.0}),))
    with pytest.raises(OCConstraintError, match="scoring_mode must be one of"):
        oc.ObjectNorm(
            constraints=(oc.ObjectConstraint("m1", "l", good),),
            layers=(Layer("l"),),
            views=(View("V", constraint_weights={"m1": 1.0}),),
            scoring_mode="fastest",
        )


def test_layer_weighted_views_work_the_way_they_do_in_a_case_norm():
    good = oc.RelatedObjectCardinality(role="order", minimum=1)
    other = oc.RelatedObjectCardinality(role="receipts", maximum=1)
    norm = oc.ObjectNorm(
        constraints=(
            oc.ObjectConstraint("m1", "matching", good, weight=3.0),
            oc.ObjectConstraint("m2", "matching", other, weight=1.0),
        ),
        layers=(Layer("matching"),),
        views=(View("Finance", layer_weights={"matching": 1.0}),),
    )
    assert norm.raw_weights("Finance") == pytest.approx({"m1": 0.75, "m2": 0.25})


def test_a_catalogue_is_refused_against_a_unit_type_it_cannot_read():
    log = partial_deliveries(3)
    spec = oc.UnitSpec(
        unit_type="order_review",
        anchor_type="purchase_order",
        roles=(oc.RolePath("receipts", (oc.PathStep("receipt for", target_type="goods_receipt", direction="reverse"),)),),
    )
    units = oc.build_units(log, spec)
    with pytest.raises(OCConstraintError, match="indistinguishable"):
        oc.score_units(log, invoice_catalogue(), units, spec=spec)
    assert invoice_catalogue().check(spec) == ("check 'm1': role 'order' is not declared by unit type 'order_review'",)


def test_two_units_cannot_share_an_identity():
    log = partial_deliveries(1)
    units = oc.build_units(log, invoice_spec(), strict=False)
    with pytest.raises(OCUnitError, match="two units share an id"):
        oc.evaluate_units(log, invoice_catalogue(), [*units, *units])


def test_an_unknown_view_or_mode_is_refused_by_name():
    log = partial_deliveries(1)
    units = oc.build_units(log, invoice_spec(), strict=False)
    with pytest.raises(OCConstraintError, match="unknown view"):
        oc.score_units(log, invoice_catalogue(), units, views="Nobody")
    with pytest.raises(OCConstraintError, match="mode must be one of"):
        oc.score_units(log, invoice_catalogue(), units, mode="quickest")
    result = oc.score_units(log, invoice_catalogue(), units)
    with pytest.raises(OCConstraintError, match="unknown view"):
        result.frame("Nobody")


def test_the_unit_table_can_be_built_on_its_own():
    log = partial_deliveries(2)
    units = oc.build_units(log, invoice_spec(), strict=False)
    frame = oc.unit_frame(units, exposure={"invoice_review:inv1": 100.0})
    assert frame.loc["invoice_review:inv1", "exposure"] == 100.0
    assert oc.unit_frame([]).index.name == "unit_id"
