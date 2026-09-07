"""Native evaluation against two documented projections, on a known truth (O05).

The fixture is three invoices whose right answers a reader can check by eye
(:func:`_oc_fixtures.comparison_log`). The same five obligations are stated
twice: natively over objects and relations, and as the case constraints an
analyst would reach for after flattening the log onto invoices. Nothing is a
straw man — the projected norm uses the paper's own constraint types,
correctly parameterised.

What the comparison shows, and what every assertion below pins:

* the **naive projection** invents six violations, misses two, and claims to
  know four things the data cannot support;
* the **identity-preserving projection** makes exactly the same five mistakes —
  keeping event identity does not restore a relation — but its duplication is
  measurable and its witnesses are real source event ids;
* the **native evaluation** matches the truth in all fifteen cells, and where
  the truth is "not knowable" it says so rather than producing a number.

The duplicated amount, the witness correctness, the runtime and the peak memory
are measured, not asserted from memory. The fixture is small and simulated: it
is a demonstration of a representational difference, not evidence of industrial
benefit, and no object link of any public case log is reconstructed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _oc_fixtures import (
    comparison_case_norm,
    comparison_log,
    comparison_object_norm,
    invoice_spec,
    repeated_rows,
    wide_vendor,
)

import wise
from wise import oc
from wise.evaluation.ocel import (
    REPRESENTATIONS,
    Expectation,
    Measured,
    compare_representations,
    identity_preserving_projection,
    measure,
    naive_projection,
    score_representation,
)
from wise.evidence.models import SourceIdentity
from wise.oc.accounting import Allocation, QuantityRecord, allocate

AT = "2024-06-01T00:00:00Z"
CHECKS = ("match", "three_way", "gr_to_invoice", "settled", "paid_late")

#: What a human says is true of this fixture. ``None`` means *not knowable from
#: this data*, which is a different answer from zero and from one.
TRUTH = (
    Expectation(
        unit_id="invoice_review:inv1",
        violations={"match": 0.0, "three_way": 0.25, "gr_to_invoice": 0.6, "settled": 0.0, "paid_late": 0.4},
        amount=100.0,
        witnesses={"paid_late": ("e_inv1", "e_pay")},
        note="an order, three receipts of 30 against an invoiced 100, one shared payment",
    ),
    Expectation(
        unit_id="invoice_review:inv2",
        violations={"match": None, "three_way": None, "gr_to_invoice": None, "settled": 0.0, "paid_late": 0.35},
        amount=100.0,
        witnesses={"paid_late": ("e_inv2", "e_pay")},
        note="no order is recorded, so three of the five answers are unknown, not zero",
    ),
    Expectation(
        unit_id="invoice_review:inv3",
        violations={"match": None, "three_way": None, "gr_to_invoice": None, "settled": 1.0, "paid_late": 0.55},
        amount=None,
        witnesses={"paid_late": ("e_inv3", "e_pay2")},
        note="two payments settle one invoice, which the case model cannot see as an excess",
    ),
)


# -------------------------------------------------------------------- helpers
def native_run():
    log = comparison_log()
    spec = invoice_spec()
    units = oc.build_units(log, spec, at=AT, strict=False)
    return log, spec, units, oc.score_units(log, comparison_object_norm(), units, spec=spec)


def projected_run(projection):
    log = comparison_log()
    events, report = projection(log, case_type="invoice", amount_attribute="amount")
    result = wise.score(events, comparison_case_norm(), evidence="full")
    violations = result.violations.copy()
    violations.index = pd.Index([f"invoice_review:{case}" for case in violations.index], name="unit_id")
    return log, events, report, result, violations


def projected_witnesses(result) -> dict[tuple[str, str], tuple[str, ...]]:
    """Source event ids the projected evaluation can actually name."""
    out: dict[tuple[str, str], tuple[str, ...]] = {}
    for record in result.evidence.records:
        ids = tuple(w.event_id for w in record.witnesses if w.identity is SourceIdentity.SOURCE and w.event_id)
        out[(f"invoice_review:{record.unit_id}", record.constraint_id)] = ids
    return out


def native_witnesses(result) -> dict[tuple[str, str], tuple[str, ...]]:
    return {(r.unit_id, r.constraint_id): tuple(w.event_id for w in r.witnesses if w.event_id) for r in result.records}


def projected_amounts(events: wise.EventLog) -> dict[str, float]:
    """What each case would report as the money it settled."""
    frame = events.events
    payments = frame[frame["activity"] == "Execute Payment"]
    return {f"invoice_review:{case}": float(total) for case, total in payments.groupby("case")["amount"].sum().items()}


def native_amounts(log, units) -> dict[str, float]:
    """The same money, through the conservation contract."""
    payment = QuantityRecord.from_reading(log.value_at("pay1", "amount", AT), "EUR")
    assert payment is not None
    sharing = [u for u in units if "pay1" in u.role("payment")]
    report = allocate(
        [payment],
        [Allocation(payment.record_id, u.unit_id, 1.0 / len(sharing)) for u in sharing],
    )
    assert report.complete
    return report.allocated


# ================================================================ the projections
def test_a_projection_says_what_it_duplicated_and_what_it_could_not_carry():
    log = comparison_log()
    _, report = naive_projection(log, case_type="invoice", amount_attribute="amount")
    assert report.n_source_events == 10
    assert report.n_cases == 3
    assert report.n_rows == 7, "six invoice-touching events, one of them copied into a second case"
    assert report.duplicated_events == 1 and report.duplicated_rows == 1
    assert report.duplicated_amount == 200.0, "the shared payment, counted a second time"
    assert report.unassigned_events == 4, "the order and its three goods receipts touch no invoice at all"
    assert report.lost_o2o == 8, "every object-to-object relation is simply gone"
    assert report.lost_object_types == ("purchase_order", "goods_receipt", "payment")
    assert report.keeps_event_identity is False
    assert "source identity is dropped" in report.notes[0]
    assert report.to_dict()["duplicated_amount"] == 200.0


def test_the_identity_preserving_projection_duplicates_just_as_much_but_visibly():
    log = comparison_log()
    naive_events, naive_report = naive_projection(log, case_type="invoice", amount_attribute="amount")
    kept_events, kept_report = identity_preserving_projection(log, case_type="invoice", amount_attribute="amount")
    assert kept_report.duplicated_amount == naive_report.duplicated_amount == 200.0
    assert kept_report.n_rows == naive_report.n_rows
    assert kept_report.keeps_event_identity is True
    assert "event_id" not in naive_events.events.columns
    assert "event_id" in kept_events.events.columns
    shared = kept_events.events.groupby("event_id")["case"].nunique()
    assert int((shared > 1).sum()) == 1, "the copied payment is detectable because its identity survived"
    assert set(kept_events.events.loc[kept_events.events["shared_event"], "event_id"]) == {"e_pay"}
    assert kept_events.event_id_col == "event_id"
    assert naive_events.event_id_col is None, "the naive projection has nothing to detect the copy with"


def test_a_projection_needs_a_case_notion_that_exists():
    with pytest.raises(ValueError, match="no object of type 'department'"):
        naive_projection(comparison_log(), case_type="department")


# ================================================================== the comparison
@pytest.fixture(scope="module")
def comparison():
    """All three representations, measured once."""
    rows = []

    native = measure(native_run)
    log, _, units, native_result = native.value
    rows.append(
        score_representation(
            "native",
            native_result.violations,
            TRUTH,
            amounts=native_amounts(log, units),
            witnesses=native_witnesses(native_result),
            seconds=native.seconds,
            peak_bytes=native.peak_bytes,
            notes=("relations preserved; unknown answers reported as unknown",),
        )
    )
    for name, projection in (
        ("naive_projection", naive_projection),
        ("identity_preserving_projection", identity_preserving_projection),
    ):
        run = measure(lambda projection=projection: projected_run(projection))
        _, events, report, result, violations = run.value
        rows.append(
            score_representation(
                name,
                violations,
                TRUTH,
                amounts=projected_amounts(events),
                witnesses=projected_witnesses(result),
                seconds=run.seconds,
                peak_bytes=run.peak_bytes,
                notes=(*report.notes, f"duplicated_amount={report.duplicated_amount}"),
            )
        )
    return compare_representations(rows), native_result


def test_the_native_evaluation_matches_the_manual_truth_in_every_cell(comparison):
    table, _ = comparison
    native = table.loc["native"]
    assert native["n_cells"] == 15
    assert native["exact_matches"] == 15
    assert native["false_violations"] == 0
    assert native["missed_obligations"] == 0
    assert native["unsupported_claims"] == 0


def test_both_projections_invent_the_same_six_violations_and_miss_the_same_two(comparison):
    table, _ = comparison
    for name in ("naive_projection", "identity_preserving_projection"):
        row = table.loc[name]
        assert row["false_violations"] == 6, "no purchase order and no goods receipt is in an invoice case"
        assert row["missed_obligations"] == 2, "the receipt lag and the second payment are invisible"
        assert row["unsupported_claims"] == 4, "four answers are produced where the data supports none"
        assert row["exact_matches"] == 7
    assert table.loc["naive_projection", "false_violations"] == table.loc["identity_preserving_projection", "false_violations"], (
        "preserving event identity does not restore a relation"
    )


def test_the_shared_payment_is_counted_twice_by_both_projections_and_once_natively(comparison):
    table, _ = comparison
    assert table.loc["native", "duplicated_quantity"] == 0.0
    assert table.loc["naive_projection", "duplicated_quantity"] == 200.0
    assert table.loc["identity_preserving_projection", "duplicated_quantity"] == 200.0


def test_only_the_representations_that_kept_an_identity_can_name_their_witnesses(comparison):
    table, _ = comparison
    assert table.loc["native", "witness_precision"] == 1.0
    assert table.loc["native", "witness_recall"] == 1.0
    assert table.loc["identity_preserving_projection", "witness_precision"] == 1.0
    assert table.loc["identity_preserving_projection", "witness_recall"] == 1.0
    naive = table.loc["naive_projection"]
    assert naive["witness_recall"] == 0.0, "its witnesses are snapshot row numbers, not source events"
    assert np.isnan(naive["witness_precision"]), "it made no source-identified claim, so precision is undefined"


def test_the_comparison_table_carries_the_cost_of_each_answer(comparison):
    table, _ = comparison
    assert list(table.index) == ["native", "naive_projection", "identity_preserving_projection"]
    assert set(REPRESENTATIONS) == set(table.index)
    assert (table["seconds"] > 0).all()
    assert (table["peak_bytes"] >= 0).all()
    assert table.loc["naive_projection", "notes"][-1] == "duplicated_amount=200.0"
    assert {
        "n_units",
        "n_cells",
        "false_violations",
        "missed_obligations",
        "unsupported_claims",
        "duplicated_quantity",
        "witness_precision",
        "witness_recall",
        "seconds",
        "peak_bytes",
    } <= set(table.columns)


def test_the_native_result_reports_the_three_answers_it_declined_to_give(comparison):
    _, native_result = comparison
    unknown = native_result.violations.isna().sum().sum()
    assert unknown == 6, "three checks on inv2 and three on inv3 have no evaluable answer"
    for unit in ("invoice_review:inv2", "invoice_review:inv3"):
        record = native_result.record(unit, "match")
        assert record.violation is None
        assert record.reason_code.value == "unverified_absence"
    assert native_result.record("invoice_review:inv3", "settled").violation == 1.0


# ====================================================== the scoring of a comparison
def test_score_representation_counts_the_three_failures_separately():
    truth = (Expectation("u1", {"a": 0.0, "b": 1.0, "c": None}),)
    observed = pd.DataFrame({"a": [0.5], "b": [0.0], "c": [0.25]}, index=["u1"])
    row = score_representation("test", observed, truth)
    assert row.false_violations == 2, "a above the truth, and a number where none was knowable"
    assert row.missed_obligations == 1
    assert row.unsupported_claims == 1
    assert row.exact_matches == 0
    assert row.n_cells == 3


def test_a_representation_that_declines_where_the_truth_declines_is_exactly_right():
    truth = (Expectation("u1", {"c": None}),)
    observed = pd.DataFrame({"c": [np.nan]}, index=["u1"])
    row = score_representation("test", observed, truth)
    assert row.exact_matches == 1
    assert row.unsupported_claims == 0
    assert row.false_violations == 0


def test_a_missing_row_costs_only_where_the_truth_expects_a_violation():
    truth = (Expectation("u1", {"a": 0.0, "b": 1.0}),)
    row = score_representation("test", pd.DataFrame(), truth)
    assert row.missed_obligations == 1
    assert row.false_violations == 0
    assert row.exact_matches == 0


def test_measure_reports_one_honest_observation():
    out = measure(lambda: sum(range(1000)))
    assert isinstance(out, Measured)
    assert out.value == 499500
    assert out.seconds > 0
    assert out.peak_bytes >= 0
    assert set(out.to_dict()) == {"seconds", "peak_bytes"}


def test_an_empty_comparison_is_an_empty_frame():
    assert compare_representations([]).empty


# ============================================= the awkward rows the handoff names
def test_duplicate_relation_rows_are_deduplicated_with_a_report():
    events, objects, e2o, o2o = repeated_rows()
    with pytest.raises(wise.errors.OCValidationError, match="dangling_o2o_target"):
        oc.OCEventLog.build(events=events, objects=objects, e2o=e2o, o2o=o2o)
    log = oc.OCEventLog.build(events=events, objects=objects, e2o=e2o, o2o=o2o, on_dangling="drop")
    assert len(log.events) == 1 and len(log.objects) == 2
    assert len(log.e2o) == 1 and len(log.o2o) == 1
    assert log.validation.count("duplicate_o2o_row") == 1
    assert log.validation.count("dangling_o2o_target") == 1
    assert not log.validation.clean, "a repaired log is not a clean one, and says which"


def test_missing_identifiers_and_times_are_refused_rather_than_defaulted():
    with pytest.raises(wise.errors.OCValidationError, match="event_id"):
        oc.OCEvent("", "A", pd.Timestamp("2024-01-01T00:00:00Z"))
    with pytest.raises(wise.errors.OCValidationError, match="timestamp"):
        oc.OCEvent("e1", "A", None)
    with pytest.raises(wise.errors.OCValidationError, match="object_type"):
        oc.OCObject("o1", "")


def test_excessive_fan_out_is_cut_natively_and_simply_absent_after_a_projection():
    """The difference is not that one is complete: it is that one says so."""
    log = wide_vendor(40)
    spec = oc.UnitSpec(
        unit_type="vendor_review",
        anchor_type="vendor",
        roles=(oc.RolePath("invoices", (oc.PathStep("billed by", target_type="invoice", direction="reverse"),)),),
        limits=oc.TraversalLimits(max_fan_out=10),
    )
    unit = oc.build_units(log, spec)[0]
    assert not unit.complete
    assert unit.truncation.seen["invoices"] == 40 and unit.truncation.kept["invoices"] == 10
    _, report = naive_projection(log, case_type="invoice")
    assert report.lost_o2o == 40, "after the projection there is no vendor to fan out from at all"
    assert report.n_cases == 1, "only one invoice is touched by an event; the other 39 vanish"


# ------------------------------------------------------------------- example
def test_the_end_to_end_example_runs_and_says_what_it_claims():
    """The stage's checkpoint, executed: check, score, allocate, project."""
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "examples" / "oc_invoice_review.py"
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "unverified_absence" in proc.stdout, "a missing order is reported as unknown, not as a violation"
    assert "overlapping_groups_not_summed" in proc.stdout
    assert "duplicated_amount=200.00" in proc.stdout
    assert "keeps_event_identity=False" in proc.stdout and "keeps_event_identity=True" in proc.stdout
    assert "layer decomposition error: 5.6e-17" in proc.stdout or "layer decomposition error" in proc.stdout
    assert "complete=True  total=200.00 EUR  residual=0.00" in proc.stdout
