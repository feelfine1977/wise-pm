"""Typed, bounded assessment units (O02).

Three claims are under test here, and they are the ones that make an
object-centric context safe to compute on:

1. **A path is a declaration, not a query.** Types, directions and qualifiers
   are named; a name that does not occur in the log is refused rather than
   quietly selecting nothing.
2. **A bound is a bound.** Depth, fan-out, bindings and total visits all cut,
   and every cut is visible: :attr:`AssessmentUnit.complete` is ``False``, the
   truncation says which limit bit, a qualification travels with the unit and
   :meth:`AssessmentUnit.require_complete` refuses.
3. **Time is declared, not guessed.** An attribute is read at the evaluation
   instant. Relations are filtered by validity only when the source actually
   supplies intervals; otherwise the unit says that it could not.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wise import oc
from wise.errors import OCUnitError
from wise.evidence.models import QualificationCode
from wise.oc.model import E2O, O2O, OCEvent, OCEventLog, OCObject, SourceMetadata
from wise.oc.units import AssessmentUnit, PathStep, RolePath, TraversalLimits, UnitScope, UnitSpec, build_units, context_report

MINI = Path(__file__).resolve().parents[1] / "fixtures" / "ocel" / "p2p_mini.json"


def ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text)


@pytest.fixture
def mini() -> OCEventLog:
    return oc.read_ocel2_json(MINI)


def invoice_spec(**changes) -> UnitSpec:
    """One invoice, its order, that order's items, its goods receipts and payment."""
    payload = {
        "unit_type": "invoice_review",
        "anchor_type": "invoice",
        "roles": (
            RolePath("order", (PathStep("belongs to", target_type="purchase_order"),)),
            RolePath(
                "items",
                (
                    PathStep("belongs to", target_type="purchase_order"),
                    PathStep("has item", target_type="item"),
                ),
            ),
            RolePath(
                "receipts",
                (
                    PathStep("belongs to", target_type="purchase_order"),
                    PathStep("receipt for", target_type="goods_receipt", direction="reverse"),
                ),
            ),
            RolePath("payment", (PathStep("settles", target_type="payment", direction="reverse"),)),
        ),
        "scope": UnitScope(roles=("order", "receipts", "payment")),
    }
    payload.update(changes)
    return UnitSpec(**payload)


def ring(n: int = 6, *, semantics: str = "atemporal") -> OCEventLog:
    """A cycle of ``n`` teams, each ``next to`` the following one."""
    return OCEventLog.build(
        events=[OCEvent("e1", "Touch", ts("2024-01-01T00:00:00Z"))],
        objects=[OCObject(f"t{i}", "team") for i in range(n)],
        e2o=[E2O("e1", "t0", "team")],
        o2o=[O2O(f"t{i}", f"t{(i + 1) % n}", "next to") for i in range(n)],
        source=SourceMetadata(interchange="in-memory", relation_time_semantics=semantics),
    )


def star(n: int) -> OCEventLog:
    """One vendor shared by ``n`` invoices — the join that eats the business."""
    return OCEventLog.build(
        events=[OCEvent("e1", "Touch", ts("2024-01-01T00:00:00Z"))],
        objects=[OCObject("v1", "vendor")] + [OCObject(f"i{i}", "invoice") for i in range(n)],
        e2o=[E2O("e1", "i0", "invoice")],
        o2o=[O2O(f"i{i}", "v1", "supplied by") for i in range(n)],
    )


# ------------------------------------------------------------ the declaration
def test_a_step_names_its_type_direction_and_qualifier():
    step = PathStep("belongs to", target_type="purchase_order")
    assert step.direction == "forward" and step.transitive is False
    assert step.describe() == "-[belongs to]->purchase_order"
    assert PathStep(None, target_type="x", direction="reverse").describe() == "<-[*]-x"


def test_a_step_with_an_unknown_direction_is_refused():
    with pytest.raises(OCUnitError, match="direction must be"):
        PathStep("q", target_type="t", direction="sideways")


def test_a_role_needs_at_least_one_step():
    with pytest.raises(OCUnitError, match="declares no steps"):
        RolePath("order", ())


def test_role_names_are_unique_inside_a_unit_type():
    with pytest.raises(OCUnitError, match="unique"):
        UnitSpec(
            unit_type="u",
            anchor_type="invoice",
            roles=(
                RolePath("order", (PathStep("q", target_type="t"),)),
                RolePath("order", (PathStep("q", target_type="t"),)),
            ),
        )


def test_a_path_longer_than_the_depth_limit_is_a_configuration_error():
    with pytest.raises(OCUnitError, match="cannot be walked"):
        UnitSpec(
            unit_type="u",
            anchor_type="invoice",
            roles=(RolePath("deep", tuple(PathStep("q", target_type="t") for _ in range(4))),),
            limits=TraversalLimits(max_depth=3),
        )


def test_the_observation_scope_may_only_name_declared_roles():
    with pytest.raises(OCUnitError, match="undeclared role"):
        UnitSpec(
            unit_type="u",
            anchor_type="invoice",
            roles=(RolePath("order", (PathStep("q", target_type="t"),)),),
            scope=UnitScope(roles=("nope",)),
        )


def test_a_scope_that_excludes_the_anchor_must_name_a_role():
    with pytest.raises(OCUnitError, match="must name at least one role"):
        UnitScope(include_anchor=False)


def test_every_limit_must_be_positive():
    for name in ("max_depth", "max_fan_out", "max_bindings", "max_visited"):
        with pytest.raises(OCUnitError, match=f"{name} must be at least 1"):
            TraversalLimits(**{name: 0})


def test_a_path_that_names_something_the_log_does_not_have_is_refused(mini):
    spec = UnitSpec(
        unit_type="u",
        anchor_type="invoice",
        roles=(RolePath("order", (PathStep("belongs to", target_type="purchase_order"),)),),
    )
    assert spec.check(mini) == ()
    typo = UnitSpec(
        unit_type="u",
        anchor_type="invoice",
        roles=(RolePath("order", (PathStep("belongs-to", target_type="purchase_order"),)),),
    )
    assert typo.check(mini) == ("role 'order' step 1: qualifier 'belongs-to' does not occur",)
    with pytest.raises(OCUnitError, match="not evidence of absence"):
        build_units(mini, typo)
    assert build_units(mini, typo, strict=False)[0].role("order") == ()


def test_an_unknown_anchor_type_is_refused(mini):
    spec = UnitSpec(unit_type="u", anchor_type="contract")
    with pytest.raises(OCUnitError, match="anchor type 'contract' does not occur"):
        build_units(mini, spec)


def test_an_anchor_of_the_wrong_type_is_refused(mini):
    with pytest.raises(OCUnitError, match="not the declared"):
        build_units(mini, invoice_spec(), anchors=["po1"])


# ---------------------------------------------------------------- the bindings
def test_the_declared_roles_bind_the_objects_the_path_reaches(mini):
    units = build_units(mini, invoice_spec(), at="2024-03-01T00:00:00Z")
    assert [u.unit_id for u in units] == ["invoice_review:inv1", "invoice_review:inv2"]
    unit = units[0]
    assert unit.anchor_id == "inv1" and unit.anchor_type == "invoice"
    assert unit.role("order") == ("po1",)
    assert unit.role("items") == ("it1", "it2", "it3")
    assert unit.role("receipts") == ("gr1", "gr2")
    assert unit.role("payment") == ("pay1",)
    assert unit.object_ids == ("gr1", "gr2", "inv1", "it1", "it2", "it3", "pay1", "po1")


def test_one_payment_serving_two_invoices_binds_to_both_without_being_shared_away(mini):
    units = build_units(mini, invoice_spec(), at="2024-03-01T00:00:00Z")
    assert [u.role("payment") for u in units] == [("pay1",), ("pay1",)]
    assert units[0].role("order") == units[1].role("order") == ("po1",)


def test_every_binding_carries_the_subgraph_that_justifies_it(mini):
    unit = build_units(mini, invoice_spec(), at="2024-03-01T00:00:00Z")[0]
    witness = unit.witnesses_for("items")[0]
    assert witness.object_id == "it1" and witness.object_type == "item" and witness.depth == 2
    assert witness.describe() == "inv1 -[belongs to]-> po1 po1 -[has item]-> it1"
    assert [h.qualifier for h in witness.hops] == ["belongs to", "has item"]
    assert len(unit.witnesses) == sum(len(ids) for ids in unit.bindings.values())


def test_asking_for_a_role_that_is_not_declared_is_an_error(mini):
    unit = build_units(mini, invoice_spec(), at="2024-03-01T00:00:00Z")[0]
    with pytest.raises(OCUnitError, match="undeclared role"):
        unit.role("vendor")


def test_a_role_that_binds_nothing_says_so_and_is_not_an_absence(mini):
    spec = UnitSpec(
        unit_type="u",
        anchor_type="goods_receipt",
        roles=(RolePath("payment", (PathStep("settles", target_type="payment", direction="reverse"),)),),
    )
    unit = build_units(mini, spec, at="2024-03-01T00:00:00Z")[0]
    assert unit.role("payment") == () and unit.unbound_roles == ("payment",)
    assert unit.complete is True, "nothing was cut; the role is empty because the log says so"


def test_direction_is_not_symmetric(mini):
    forward = UnitSpec(
        unit_type="f",
        anchor_type="purchase_order",
        roles=(RolePath("invoices", (PathStep("belongs to", target_type="invoice"),)),),
    )
    reverse = UnitSpec(
        unit_type="r",
        anchor_type="purchase_order",
        roles=(RolePath("invoices", (PathStep("belongs to", target_type="invoice", direction="reverse"),)),),
    )
    assert build_units(mini, forward)[0].role("invoices") == ()
    assert build_units(mini, reverse)[0].role("invoices") == ("inv1", "inv2")


# ------------------------------------------------------------------- the bounds
def test_a_shared_vendor_is_cut_by_the_fan_out_limit_and_the_unit_says_so():
    log = star(40)
    spec = UnitSpec(
        unit_type="invoice_review",
        anchor_type="invoice",
        roles=(
            RolePath(
                "siblings",
                (
                    PathStep("supplied by", target_type="vendor"),
                    PathStep("supplied by", target_type="invoice", direction="reverse"),
                ),
            ),
        ),
        limits=TraversalLimits(max_fan_out=10, max_bindings=100),
    )
    unit = build_units(log, spec, anchors=["i0"])[0]
    assert len(unit.role("siblings")) == 9, "ten neighbours survived the cut, and one of them was the anchor itself"
    assert "i0" not in unit.role("siblings")
    assert unit.complete is False
    assert unit.truncation.fan_out_limited == ("siblings",)
    assert unit.truncation.seen["siblings"] == 41, "one vendor, then 40 invoices seen before the cut"
    codes = [q.code for q in unit.qualifications]
    assert QualificationCode.CONTEXT_TRUNCATED in codes and QualificationCode.FAN_OUT_LIMIT_REACHED in codes


def test_the_binding_limit_cuts_after_the_fan_out_limit_has_let_things_through():
    log = star(40)
    spec = UnitSpec(
        unit_type="invoice_review",
        anchor_type="invoice",
        roles=(
            RolePath(
                "siblings",
                (
                    PathStep("supplied by", target_type="vendor"),
                    PathStep("supplied by", target_type="invoice", direction="reverse"),
                ),
            ),
        ),
        limits=TraversalLimits(max_fan_out=100, max_bindings=5),
    )
    unit = build_units(log, spec, anchors=["i0"])[0]
    assert len(unit.role("siblings")) == 5
    assert unit.truncation.binding_limited == ("siblings",)
    assert unit.truncation.kept["siblings"] == 5 and unit.truncation.seen["siblings"] == 41
    assert QualificationCode.BINDING_LIMIT_REACHED in [q.code for q in unit.qualifications]


def test_a_cycle_terminates_and_the_anchor_is_never_bound_to_its_own_role():
    log = ring(6)
    spec = UnitSpec(
        unit_type="team_context",
        anchor_type="team",
        roles=(RolePath("reachable", (PathStep("next to", target_type="team", transitive=True),)),),
        limits=TraversalLimits(max_depth=10),
    )
    unit = build_units(log, spec, anchors=["t0"])[0]
    assert unit.role("reachable") == ("t1", "t2", "t3", "t4", "t5")
    assert "t0" not in unit.role("reachable"), "coming back to the anchor is a cycle, not a related object"
    assert unit.complete is True


def test_a_transitive_step_stops_at_the_depth_limit_and_flags_it():
    log = ring(20)
    spec = UnitSpec(
        unit_type="team_context",
        anchor_type="team",
        roles=(RolePath("reachable", (PathStep("next to", target_type="team", transitive=True),)),),
        limits=TraversalLimits(max_depth=3),
    )
    unit = build_units(log, spec, anchors=["t0"])[0]
    assert unit.role("reachable") == ("t1", "t2", "t3")
    assert unit.complete is False and unit.truncation.depth_limited == ("reachable",)
    assert QualificationCode.DEPTH_LIMIT_REACHED in [q.code for q in unit.qualifications]


def test_the_visit_budget_bounds_the_whole_unit():
    log = star(40)
    spec = UnitSpec(
        unit_type="invoice_review",
        anchor_type="invoice",
        roles=(
            RolePath(
                "siblings",
                (
                    PathStep("supplied by", target_type="vendor"),
                    PathStep("supplied by", target_type="invoice", direction="reverse"),
                ),
            ),
        ),
        limits=TraversalLimits(max_fan_out=100, max_bindings=100, max_visited=6),
    )
    unit = build_units(log, spec, anchors=["i0"])[0]
    assert unit.truncation.visit_limited is True and unit.complete is False
    assert len(unit.role("siblings")) <= 6


def test_a_truncated_context_refuses_to_pose_as_a_complete_assessment():
    log = star(40)
    spec = UnitSpec(
        unit_type="invoice_review",
        anchor_type="invoice",
        roles=(
            RolePath(
                "siblings",
                (
                    PathStep("supplied by", target_type="vendor"),
                    PathStep("supplied by", target_type="invoice", direction="reverse"),
                ),
            ),
        ),
        limits=TraversalLimits(max_fan_out=3),
    )
    unit = build_units(log, spec, anchors=["i0"])[0]
    with pytest.raises(OCUnitError, match="cannot be presented as a complete assessment"):
        unit.require_complete()
    assert "lower bound" in unit.qualifications[0].message


def test_a_complete_context_passes_the_same_gate(mini):
    unit = build_units(mini, invoice_spec(), at="2024-03-01T00:00:00Z")[0]
    assert unit.require_complete() is unit


def test_the_context_report_separates_complete_from_truncated_units():
    log = star(40)
    spec = UnitSpec(
        unit_type="invoice_review",
        anchor_type="invoice",
        roles=(
            RolePath(
                "siblings",
                (
                    PathStep("supplied by", target_type="vendor"),
                    PathStep("supplied by", target_type="invoice", direction="reverse"),
                ),
            ),
        ),
        limits=TraversalLimits(max_fan_out=10),
    )
    report = context_report(build_units(log, spec))
    assert report["n_units"] == 40 and report["n_truncated"] == 40 and report["complete_share"] == 0.0
    assert "max_fan_out" in " ".join(report["truncation_reasons"])
    assert context_report([])["complete_share"] is None


# ------------------------------------------------------------ events in scope
def test_the_events_in_scope_come_from_the_anchor_and_the_named_roles(mini):
    unit = build_units(mini, invoice_spec(), at="2024-03-01T00:00:00Z")[0]
    assert unit.event_ids == ("e1", "e2", "e3", "e4", "e5", "e6")
    anchor_only = build_units(mini, invoice_spec(scope=UnitScope()), at="2024-03-01T00:00:00Z")[0]
    assert anchor_only.event_ids == ("e4", "e6")


def test_two_events_with_equal_timestamps_are_both_in_scope_and_ordered_stably(mini):
    unit = build_units(mini, invoice_spec(), at="2024-03-01T00:00:00Z")[0]
    equal = [e for e in unit.events(mini) if e.activity == "Record Invoice Receipt"]
    assert [e.event_id for e in equal] == ["e4", "e5"]
    assert equal[0].timestamp == equal[1].timestamp


def test_the_scope_can_narrow_by_activity_qualifier_and_window(mini):
    by_activity = build_units(
        mini, invoice_spec(scope=UnitScope(roles=("order",), activities=("Record Goods Receipt",))), at="2024-03-01T00:00:00Z"
    )[0]
    assert by_activity.event_ids == ("e2", "e3")
    by_qualifier = build_units(mini, invoice_spec(scope=UnitScope(qualifiers=("invoice",))), at="2024-03-01T00:00:00Z")[0]
    assert by_qualifier.event_ids == ("e4", "e6")
    windowed = build_units(
        mini,
        invoice_spec(
            scope=UnitScope(roles=("order",), window_start=ts("2024-01-04T00:00:00Z"), window_end=ts("2024-01-06T00:00:00Z"))
        ),
        at="2024-03-01T00:00:00Z",
    )[0]
    assert windowed.event_ids == ("e3", "e4", "e5")


def test_an_activity_the_log_does_not_have_is_refused_in_the_scope_too(mini):
    with pytest.raises(OCUnitError, match="activity 'Nope' does not occur"):
        build_units(mini, invoice_spec(scope=UnitScope(activities=("Nope",))))


# ------------------------------------------------------------- attributes and time
def test_an_attribute_is_read_at_the_units_evaluation_time(mini):
    early = build_units(mini, invoice_spec(), at="2024-01-10T00:00:00Z")[0]
    late = build_units(mini, invoice_spec(), at="2024-02-10T00:00:00Z")[0]
    assert early.attribute(mini, "amount").value == 100.0
    assert late.attribute(mini, "amount").value == 120.0
    assert early.attribute(mini, "owner").value == "team-a"
    assert late.attribute(mini, "owner").value == "team-b"


def test_a_unit_without_an_evaluation_time_refuses_to_read_as_of_one(mini):
    unit = build_units(mini, invoice_spec())[0]
    assert unit.evaluation_time is None
    with pytest.raises(OCUnitError, match="without an evaluation time"):
        unit.attribute(mini, "amount")
    latest = unit.attribute(mini, "amount", policy="latest_known")
    assert latest.value == 120.0 and latest.qualifications == ()


def test_an_attribute_of_a_bound_object_is_readable_and_a_stranger_is_not(mini):
    unit = build_units(mini, invoice_spec(), at="2024-03-01T00:00:00Z")[0]
    assert unit.attribute(mini, "vendor", object_id="po1").value == "V1"
    with pytest.raises(OCUnitError, match="is not part of this unit"):
        unit.attribute(mini, "amount", object_id="inv2")


def test_an_atemporal_source_says_that_it_cannot_place_its_relations_in_time(mini):
    unit = build_units(mini, invoice_spec(), at="2024-03-01T00:00:00Z")[0]
    relation = [q for q in unit.qualifications if q.code is QualificationCode.RELATION_VALIDITY_UNKNOWN]
    assert len(relation) == 1
    assert "not filtered" in relation[0].message and "no validity interval has been inferred" in relation[0].message


def test_a_source_that_supplies_intervals_has_them_used_not_merely_stored():
    log = OCEventLog.build(
        events=[OCEvent("e1", "Touch", ts("2024-01-01T00:00:00Z"))],
        objects=[OCObject("i1", "invoice"), OCObject("t1", "team"), OCObject("t2", "team")],
        e2o=[E2O("e1", "i1", "invoice")],
        o2o=[
            O2O("i1", "t1", "owned by", valid_from=ts("2024-01-01T00:00:00Z"), valid_to=ts("2024-06-01T00:00:00Z")),
            O2O("i1", "t2", "owned by", valid_from=ts("2024-06-01T00:00:00Z")),
        ],
        source=SourceMetadata(interchange="in-memory", relation_time_semantics="interval"),
    )
    spec = UnitSpec(
        unit_type="ownership",
        anchor_type="invoice",
        roles=(RolePath("owner", (PathStep("owned by", target_type="team"),)),),
    )
    assert build_units(log, spec, at="2024-03-01T00:00:00Z")[0].role("owner") == ("t1",)
    assert build_units(log, spec, at="2024-09-01T00:00:00Z")[0].role("owner") == ("t2",)
    at_the_boundary = build_units(log, spec, at="2024-06-01T00:00:00Z")[0]
    assert at_the_boundary.role("owner") == ("t2",), "half-open: [valid_from, valid_to)"
    assert [q.code for q in at_the_boundary.qualifications] == []


def test_an_interval_source_used_without_a_time_says_no_filter_was_applied():
    log = ring(4, semantics="interval")
    spec = UnitSpec(
        unit_type="team_context",
        anchor_type="team",
        roles=(RolePath("next", (PathStep("next to", target_type="team"),)),),
    )
    unit = build_units(log, spec, anchors=["t0"])[0]
    assert [q.code for q in unit.qualifications] == [QualificationCode.RELATION_VALIDITY_UNKNOWN]
    assert "no interval filter was applied" in unit.qualifications[0].message


# --------------------------------------------------------------------- exports
def test_a_unit_describes_itself_completely_enough_to_be_recorded(mini):
    unit = build_units(mini, invoice_spec(), at="2024-03-01T00:00:00Z")[0]
    payload = unit.to_dict()
    assert payload["unit_id"] == "invoice_review:inv1"
    assert payload["bindings"]["items"] == ["it1", "it2", "it3"]
    assert payload["complete"] is True and payload["truncation"]["reasons"] == []
    assert payload["evaluation_time"] == "2024-03-01T00:00:00+00:00"
    assert payload["n_events"] == 6 and payload["limits"]["max_fan_out"] == 25
    assert [q["code"] for q in payload["qualifications"]] == ["relation_validity_unknown"]
    assert payload["witnesses"][0]["hops"][0]["qualifier"] == "belongs to"


def test_the_spec_describes_itself_too(mini):
    payload = invoice_spec().to_dict()
    assert payload["anchor_type"] == "invoice"
    assert payload["roles"][1]["describes"] == "items: anchor-[belongs to]->purchase_order-[has item]->item"
    assert payload["attribute_policy"] == "as_of"


def test_the_unit_repr_never_hides_a_truncation():
    log = star(40)
    spec = UnitSpec(
        unit_type="invoice_review",
        anchor_type="invoice",
        roles=(
            RolePath(
                "siblings",
                (
                    PathStep("supplied by", target_type="vendor"),
                    PathStep("supplied by", target_type="invoice", direction="reverse"),
                ),
            ),
        ),
        limits=TraversalLimits(max_fan_out=4),
    )
    assert "truncated" in repr(build_units(log, spec, anchors=["i0"])[0])
    assert isinstance(build_units(log, spec, anchors=["i0"])[0], AssessmentUnit)


def test_building_units_twice_gives_the_same_answer(mini):
    first = build_units(mini, invoice_spec(), at="2024-03-01T00:00:00Z")
    second = build_units(mini, invoice_spec(), at="2024-03-01T00:00:00Z")
    assert [u.to_dict() for u in first] == [u.to_dict() for u in second]


def test_a_step_that_the_depth_budget_can_no_longer_reach_is_a_limit_not_an_absence():
    """A transitive step that eats the budget must not let the next step walk anyway."""
    log = OCEventLog.build(
        events=[OCEvent("e1", "Touch", ts("2024-01-01T00:00:00Z"))],
        objects=[OCObject(f"t{i}", "team") for i in range(5)] + [OCObject("i1", "invoice")],
        e2o=[E2O("e1", "t0", "team")],
        o2o=[O2O(f"t{i}", f"t{i + 1}", "next to") for i in range(4)] + [O2O("t4", "i1", "owns")],
    )
    spec = UnitSpec(
        unit_type="chain",
        anchor_type="team",
        roles=(
            RolePath(
                "far",
                (
                    PathStep("next to", target_type="team", transitive=True),
                    PathStep("owns", target_type="invoice"),
                ),
            ),
        ),
        limits=TraversalLimits(max_depth=2),
    )
    unit = build_units(log, spec, anchors=["t0"])[0]
    assert unit.role("far") == (), "the invoice is four hops away and the budget was two"
    assert unit.complete is False and unit.truncation.depth_limited == ("far",)
    assert unit.truncation.limited_roles == ("far",)


def test_the_spec_reports_an_object_type_the_log_does_not_have(mini):
    spec = UnitSpec(
        unit_type="u",
        anchor_type="invoice",
        roles=(RolePath("order", (PathStep("belongs to", target_type="contract"),)),),
    )
    assert spec.check(mini) == ("role 'order' step 1: object type 'contract' does not occur",)
    with pytest.raises(OCUnitError, match="undeclared role"):
        spec.role("nope")


# ---------------------------------------------------------------------- example
def test_the_end_to_end_example_runs_and_says_what_it_claims(tmp_path):
    """The stage's checkpoint, executed: read, repair, bound, export, refuse."""
    import json
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[2] / "examples" / "oc_units_review.py"
    out = tmp_path / "review"
    proc = subprocess.run([sys.executable, str(script), str(out)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "duplicate_e2o_row" in proc.stdout, "the repaired duplicate is reported, not hidden"
    assert "at 2024-01-01T00:00:00Z: no value in force" in proc.stdout
    assert "at 2024-01-10T00:00:00Z: 'team-a'" in proc.stdout
    assert "OCEL 2.0 JSON round-trip identical: True" in proc.stdout
    assert "native round-trip identical:       True" in proc.stdout
    assert "require_complete() refuses" in proc.stdout
    written = json.loads((out / "p2p_example.ocel.json").read_text(encoding="utf-8"))
    assert sorted(written) == ["eventTypes", "events", "objectTypes", "objects"]
    assert len(written["events"]) == 6 and len(written["objects"]) == 9
