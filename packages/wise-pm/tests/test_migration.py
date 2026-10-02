"""Migration of a norm document from schema version 2 to 3 (``wise.migrate``, decision 0001, sections 3 and 4).

The expected documents and reports are written out by hand from ``LEGACY``
below: every rewrite the decision prescribes occurs in it once, and every
other part of it must come out unchanged.
"""

from __future__ import annotations

import copy
import json
import math
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

import wise
from wise import norm_document
from wise.errors import NormError
from wise.semantics import CORRECTED_SEMANTICS

PKG = Path(__file__).resolve().parents[1]

#: A schema version 2 document with each rewrite of the migration once, and numbers that a reader would normalise.
LEGACY: dict[str, Any] = {
    "schema_version": 2,
    "name": "legacy",
    "version": "7",
    "description": "",
    "scoring_mode": "flat",
    "metadata": {"threshold_note": 0.30000000000000004, "big": 12345678901234567890},
    "layers": [{"id": "L", "name": "", "description": ""}],
    "views": [{"name": "v", "description": "", "constraint_weights": {"bare": 1, "open": 2, "bounded": 0.1, "p": 3}}],
    "derived_attributes": [
        {"name": "events", "kind": "eval", "expr": "n_events"},
        {"name": "scaled", "kind": "quantile_scale", "attribute": "amount", "q": 0.9},
        {"name": "value", "kind": "eval", "expr": "`order value`"},
    ],
    "constraints": [
        {"id": "bare", "layer": "L", "type": "lag", "params": {"a": "A", "b": "B"}, "weight": 2},
        {"id": "open", "layer": "L", "type": "lag", "params": {"a": "A", "b": "B", "delta": None, "unit": "h"}},
        {"id": "bounded", "layer": "L", "type": "lag", "params": {"a": "A", "b": "B", "delta": 10, "width": 1e-3}},
        {"id": "p", "layer": "L", "type": "presence", "params": {"activity": "A", "m": 1}, "applicability": {"x": {"ne": 1}}},
    ],
}

#: ``LEGACY`` migrated, written out by hand.
MIGRATED: dict[str, Any] = {
    "schema_version": 3,
    "name": "legacy",
    "version": "7",
    "description": "",
    "scoring_mode": "flat",
    "missing_values": "three_valued",
    "metadata": {"threshold_note": 0.30000000000000004, "big": 12345678901234567890},
    "layers": [{"id": "L", "name": "", "description": ""}],
    "views": [{"name": "v", "description": "", "constraint_weights": {"bare": 1, "open": 2, "bounded": 0.1, "p": 3}}],
    "derived_attributes": [
        {"name": "events", "kind": "column", "attribute": "n_events"},
        {
            "name": "scaled",
            "kind": "quantile_scale",
            "attribute": "amount",
            "q": 0.9,
            "calibration": {"status": "unfitted", "quantile": 0.9},
        },
        {"name": "value", "kind": "column", "attribute": "order value"},
    ],
    "constraints": [
        {"id": "bare", "layer": "L", "type": "lag", "params": {"a": "A", "b": "B", "delta": 0, "width": 0}, "weight": 2},
        {"id": "open", "layer": "L", "type": "lag", "params": {"a": "A", "b": "B", "unbounded": True, "unit": "h"}},
        {"id": "bounded", "layer": "L", "type": "lag", "params": {"a": "A", "b": "B", "delta": 10, "width": 1e-3}},
        {"id": "p", "layer": "L", "type": "presence", "params": {"activity": "A", "m": 1}, "applicability": {"x": {"ne": 1}}},
    ],
}

#: The report of migrating ``LEGACY``, written out by hand: one entry per rewrite, in document order.
REPORT: dict[str, Any] = {
    "norm": "legacy",
    "source_version": 2,
    "target_version": 3,
    "rewrites": [
        {
            "rule": "missing_values",
            "path": "",
            "subject": "legacy",
            "before": {},
            "after": {"missing_values": "three_valued"},
            "note": "the missing-value policy of applicability is named: a case whose attribute is missing is unknown "
            "under every operator and left out of the denominator",
            "corrections": ["C4", "C25"],
            "recommendation": None,
        },
        {
            "rule": "eval_column",
            "path": "/derived_attributes/0",
            "subject": "events",
            "before": {"kind": "eval", "expr": "n_events"},
            "after": {"kind": "column", "attribute": "n_events"},
            "note": "an eval recipe over the bare case-table column 'n_events' becomes a column recipe",
            "corrections": ["C16"],
            "recommendation": None,
        },
        {
            "rule": "calibration",
            "path": "/derived_attributes/1",
            "subject": "scaled",
            "before": {},
            "after": {"calibration": {"status": "unfitted", "quantile": 0.9}},
            "note": "the quantile_scale recipe carries an unfitted calibration record",
            "corrections": ["C15"],
            "recommendation": "fit the record on the reference log (wise.fit_calibration) before scoring: scoring refuses "
            "an unfitted record, and 0.1.0 divided by the quantile of whatever cases were scored",
        },
        {
            "rule": "eval_column",
            "path": "/derived_attributes/2",
            "subject": "value",
            "before": {"kind": "eval", "expr": "`order value`"},
            "after": {"kind": "column", "attribute": "order value"},
            "note": "an eval recipe over the bare case-table column 'order value' becomes a column recipe",
            "corrections": ["C16"],
            "recommendation": None,
        },
        {
            "rule": "lag_bound",
            "path": "/constraints/0/params",
            "subject": "bare",
            "before": {},
            "after": {"delta": 0, "width": 0},
            "note": "the lag had no bound; the bound 0.1.0 read it with, delta 0 and width 0, is written explicitly",
            "corrections": ["C2"],
            "recommendation": "review the bound: with delta 0 and width 0 any delay between the activities is a full "
            "violation; state the intended delta and width, or 'unbounded': true",
        },
        {
            "rule": "lag_unbounded",
            "path": "/constraints/1/params",
            "subject": "open",
            "before": {"delta": None},
            "after": {"unbounded": True},
            "note": "the lag without a time bound ('delta': null) is written 'unbounded': true",
            "corrections": [],
            "recommendation": None,
        },
    ],
}


def legacy(**overrides: Any) -> dict[str, Any]:
    return {**copy.deepcopy(LEGACY), **overrides}


def migrate_quietly(source: Any) -> wise.Migration:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return wise.migrate(source)


# ----------------------------------------------------------------------------- rewrites
@pytest.mark.spec("decision:0001/3")
def test_migrating_a_v2_document_writes_each_rewrite_and_reports_it() -> None:
    migration = migrate_quietly(legacy())
    assert json.dumps(migration.document) == json.dumps(MIGRATED)
    assert migration.report.to_dict() == REPORT
    assert migration.norm.schema_version == 3
    assert migration.norm.semantics == CORRECTED_SEMANTICS
    assert migration.norm == wise.Norm.from_dict(MIGRATED)


@pytest.mark.spec("decision:0001/3")
def test_each_rule_reports_one_entry_per_rewrite() -> None:
    rules = [r.rule for r in migrate_quietly(legacy()).report.rewrites]
    assert rules == ["missing_values", "eval_column", "calibration", "eval_column", "lag_bound", "lag_unbounded"]
    assert migrate_quietly(legacy()).report.corrections == ("C4", "C25", "C16", "C15", "C2")


@pytest.mark.spec("decision:0001/3")
def test_a_document_without_anything_to_rewrite_gets_only_its_missing_value_policy() -> None:
    document = legacy(derived_attributes=[], constraints=[LEGACY["constraints"][2], LEGACY["constraints"][3]])
    document["views"] = [{"name": "v", "constraint_weights": {"bounded": 1, "p": 1}}]
    migration = migrate_quietly(document)
    assert [r.rule for r in migration.report.rewrites] == ["missing_values"]
    expected = {k: v for k, v in document.items() if k != "schema_version"}
    out = dict(migration.document)
    assert (out.pop("schema_version"), out.pop("missing_values")) == (3, "three_valued")
    assert out == expected


@pytest.mark.spec("decision:0001/3")
def test_a_lag_given_only_its_width_gets_delta_0_and_keeps_its_width() -> None:
    document = legacy(constraints=[{"id": "bare", "layer": "L", "type": "lag", "params": {"a": "A", "b": "B", "width": 5}}])
    document["views"] = [{"name": "v", "constraint_weights": {"bare": 1}}]
    migration = migrate_quietly(document)
    assert migration.document["constraints"][0]["params"] == {"a": "A", "b": "B", "width": 5, "delta": 0}
    (lag,) = [r for r in migration.report.rewrites if r.rule == "lag_bound"]
    assert (lag.before, lag.after) == ({}, {"delta": 0})


@pytest.mark.spec("decision:0001/3")
def test_a_quantile_scale_without_q_is_calibrated_at_its_default_quantile_and_a_record_is_kept() -> None:
    fitted = {"status": "fitted", "quantile": 0.95, "divisor": 2.5, "log_fingerprint": "f", "size": 3, "window": [None, None]}
    recipes: list[dict[str, Any]] = [
        {"name": "s", "kind": "quantile_scale", "attribute": "amount"},
        {"name": "t", "kind": "quantile_scale", "attribute": "amount", "calibration": None},
        {"name": "u", "kind": "quantile_scale", "attribute": "amount", "calibration": fitted},
    ]
    migration = migrate_quietly(legacy(derived_attributes=recipes))
    assert [dict(r) for r in migration.document["derived_attributes"]] == [
        {**recipes[0], "calibration": {"status": "unfitted", "quantile": 0.95}},
        {**recipes[1], "calibration": {"status": "unfitted", "quantile": 0.95}},
        recipes[2],
    ]
    calibrations = [r for r in migration.report.rewrites if r.rule == "calibration"]
    assert [(r.subject, r.before) for r in calibrations] == [("s", {}), ("t", {"calibration": None})]


@pytest.mark.spec("decision:0001/3")
@pytest.mark.parametrize("expr", ["n_events * 2", "amount.sum()", "a + b", "1"])
def test_any_other_eval_stops_the_migration_with_the_recipe_named(expr: str) -> None:
    document = legacy(derived_attributes=[{"name": "twice", "kind": "eval", "expr": expr}])
    with pytest.raises(
        NormError,
        match=r"cannot migrate norm 'legacy' to schema version 3: derived attribute 'twice' \(eval\): expr .* is not a "
        r"single case-table column.*schema version 3 has no eval recipe; rewrite 'twice' as a typed recipe",
    ):
        wise.migrate(document)


# ----------------------------------------------------------------------------- numbers, report
@pytest.mark.spec("decision:0001/4")
def test_numbers_in_the_document_are_preserved_exactly() -> None:
    text = json.dumps(migrate_quietly(legacy()).document)
    for number in ("0.30000000000000004", "12345678901234567890", '"weight": 2', '"bounded": 0.1', "0.001", '"delta": 10,'):
        assert number in text
    assert '"delta": 10.0' not in text
    assert '"weight": 2.0' not in text


@given(
    weight=st.one_of(st.integers(0, 10**6), st.floats(0, 1e6, allow_nan=False)),
    delta=st.one_of(st.integers(0, 10**6), st.floats(0, 1e6, allow_nan=False)),
    note=st.floats(allow_nan=False, allow_infinity=False),
)
def test_numbers_survive_the_migration_bit_for_bit(weight: float, delta: float, note: float) -> None:
    document = legacy(metadata={"note": note})
    document["constraints"] = copy.deepcopy(LEGACY["constraints"])
    document["constraints"][2]["params"]["delta"] = delta
    document["constraints"][0]["weight"] = weight
    out = migrate_quietly(document).document
    for got, given_ in ((out["metadata"]["note"], note), (out["constraints"][2]["params"]["delta"], delta)):
        assert type(got) is type(given_)
        assert math.copysign(1, got) == math.copysign(1, given_)
        assert repr(got) == repr(given_)
    assert repr(out["constraints"][0]["weight"]) == repr(weight)


@pytest.mark.spec("decision:0001/4")
def test_the_report_is_serialisable_and_stable() -> None:
    first, second = migrate_quietly(legacy()).report, migrate_quietly(legacy()).report
    assert first == second
    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == REPORT
    assert wise.MigrationReport.from_dict(json.loads(first.to_json())) == first
    assert first.to_json() == json.dumps(REPORT, indent=2, sort_keys=True, ensure_ascii=False)


def test_the_migration_is_read_only_and_leaves_its_input_unchanged() -> None:
    document = legacy()
    migration = migrate_quietly(document)
    assert document == LEGACY
    with pytest.raises(TypeError):
        migration.document["name"] = "other"  # type: ignore[index]
    with pytest.raises(TypeError):
        migration.document["constraints"][0]["params"]["delta"] = 1


def test_the_migrated_document_reads_back_to_the_migrated_norm_and_dumps_as_its_text() -> None:
    migration = migrate_quietly(legacy())
    assert wise.Norm.loads(migration.dumps()) == migration.norm
    assert migration.dumps() == json.dumps(MIGRATED, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------------- sources
@pytest.mark.spec("decision:0001/4")
def test_a_norm_is_migrated_through_its_document(p2p_norm: wise.Norm) -> None:
    from_norm = migrate_quietly(p2p_norm)
    from_document = migrate_quietly(p2p_norm.to_dict())
    assert from_norm.document == from_document.document
    assert from_norm.report == from_document.report
    assert from_norm.norm.schema_version == 3
    assert p2p_norm.schema_version == 2


@pytest.mark.spec("decision:0001/4")
def test_a_schema_version_1_document_migrates_from_version_1() -> None:
    migration = migrate_quietly(legacy(schema_version=1))
    assert migration.report.source_version == 1
    assert migration.document["schema_version"] == 3
    assert json.dumps({**migration.document, "schema_version": 3}) == json.dumps(MIGRATED)


@pytest.mark.spec("decision:0001/4")
def test_a_v3_document_or_norm_is_not_migrated_again() -> None:
    v3 = MIGRATED
    with pytest.raises(NormError, match=r"norm 'legacy' has schema_version 3 already; migrate\(\) takes schema version 1 or 2"):
        wise.migrate(v3)
    with pytest.raises(NormError, match="has schema_version 3 already"):
        wise.migrate(wise.Norm.from_dict(v3))


@pytest.mark.spec("decision:0001/4")
@pytest.mark.parametrize(
    ("version", "message"),
    [
        (None, r"norm schema_version None is not an integer; supported: \[1, 2, 3\]"),
        (4, r"norm schema_version 4 is newer than supported \(3\); supported: \[1, 2, 3\]"),
        (0, r"norm schema_version 0 is not a known schema version; supported: \[1, 2, 3\]"),
        ("two", r"norm schema_version 'two' is not an integer; supported: \[1, 2, 3\]"),
    ],
)
def test_an_unknown_schema_version_is_refused_with_the_supported_versions(version: Any, message: str) -> None:
    with pytest.raises(NormError, match=message):
        wise.migrate(legacy(schema_version=version))
    with pytest.raises(NormError, match=message):
        wise.Norm.from_dict(legacy(schema_version=version))


@pytest.mark.spec("decision:0001/4")
def test_a_missing_schema_version_is_refused_with_the_supported_versions() -> None:
    document = legacy()
    del document["schema_version"]
    with pytest.raises(NormError, match=r"norm document has no 'schema_version'; give one of \[1, 2, 3\]"):
        wise.migrate(document)


def test_a_malformed_source_is_refused() -> None:
    with pytest.raises(NormError, match="a norm must be a mapping, got list"):
        wise.migrate([1, 2])  # type: ignore[arg-type]
    with pytest.raises(NormError, match="unknown top-level keys in norm: \\['extra'\\]"):
        wise.migrate(legacy(extra=1))
    with pytest.raises(NormError, match="unknown top-level keys in norm: \\['missing_values'\\]"):
        wise.migrate(legacy(missing_values="three_valued"))
    with pytest.raises(NormError, match="malformed norm: "):
        wise.migrate(legacy(constraints=[5]))


def test_a_report_names_only_the_rules_of_the_migration() -> None:
    document = copy.deepcopy(REPORT)
    document["rewrites"][0]["rule"] = "rename"
    with pytest.raises(NormError, match=r"unknown migration rule 'rename'; known \['missing_values', 'eval_column', "):
        wise.MigrationReport.from_dict(document)


def test_migrating_does_not_announce_the_legacy_semantics() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wise.migrate(legacy())
    assert caught == []


def test_migrate_is_the_document_module_operation() -> None:
    assert wise.migrate is norm_document.migrate
    assert wise.MigrationReport is norm_document.MigrationReport
    assert wise.Migration is norm_document.Migration


# ----------------------------------------------------------------------------- shipped norms
@pytest.mark.spec("decision:0001/4")
def test_the_running_example_migrates_to_its_committed_v3_copy() -> None:
    source = PKG / "examples" / "running_p2p_norm.json"
    migration = migrate_quietly(json.loads(source.read_text(encoding="utf-8")))
    assert migration.dumps() + "\n" == (PKG / "examples" / "running_p2p_norm_v3.json").read_text(encoding="utf-8")
    assert [r.rule for r in migration.report.rewrites] == ["missing_values"]


@pytest.mark.spec("decision:0001/4")
def test_the_bpic19_norm_migrates() -> None:
    migration = migrate_quietly(json.loads((PKG / "examples" / "bpic19_norm.json").read_text(encoding="utf-8")))
    assert [(r.rule, r.subject) for r in migration.report.rewrites] == [
        ("missing_values", "WISE BPIC'19 norm"),
        ("eval_column", "total_events"),
        ("calibration", "exposure_norm"),
    ]
    recipes = {r["name"]: dict(r) for r in migration.document["derived_attributes"]}
    assert recipes["total_events"] == {"name": "total_events", "kind": "column", "attribute": "n_events"}
    assert recipes["exposure_norm"]["calibration"] == {"status": "unfitted", "quantile": 0.95}


def bpic19_shaped_log() -> wise.EventLog:
    """Six purchase-order items with the activities, resources, net worth and flow types the BPIC'19 norm reads."""
    gr, inv, vinv, clear, rpb = (
        "Record Goods Receipt",
        "Record Invoice Receipt",
        "Vendor creates invoice",
        "Clear Invoice",
        "Remove Payment Block",
    )
    rows = [
        ("c1", gr, 0, "user_001", 100.0, "DF1"),
        ("c1", inv, 5, "user_002", 120.0, "DF1"),
        ("c1", clear, 40, "batch_00", 120.0, "DF1"),
        ("c2", gr, 0, "user_001", 5000.0, "DF1"),
        ("c2", gr, 1, "user_003", 5000.0, "DF1"),
        ("c2", gr, 2, "NONE", 5000.0, "DF1"),
        ("c2", vinv, 20, "user_004", 5200.0, "DF1"),
        ("c2", "Change Price", 21, "user_004", 5300.0, "DF1"),
        ("c2", "Vendor creates debit memo", 22, "user_005", 5300.0, "DF1"),
        ("c3", inv, 0, "user_006", 800.0, "DF2"),
        ("c3", gr, 3, "batch_01", 800.0, "DF2"),
        ("c3", rpb, 9, "user_006", 800.0, "DF2"),
        ("c3", clear, 15, "batch_01", 800.0, "DF2"),
        ("c4", inv, 0, "user_007", 300.0, "2-way"),
        ("c4", gr, 1, "user_007", 300.0, "2-way"),
        ("c4", clear, 50, "user_008", 300.0, "2-way"),
        ("c4", "Cancel Invoice Receipt", 51, "user_009", 300.0, "2-way"),
        ("c5", gr, 0, "user_010", 50.0, "Consignment"),
        ("c5", clear, 3, "user_010", 50.0, "Consignment"),
        ("c5", "Change Quantity", 4, "user_011", 70.0, "Consignment"),
        ("c6", gr, 0, "user_012", 10.0, "other"),
    ]
    df = pd.DataFrame(rows, columns=["case", "activity", "day", "event org:resource", "event Cumulative net worth (EUR)", "flow_type"])
    df["time"] = pd.Timestamp("2018-01-01") + pd.to_timedelta(df["day"], unit="D")
    return wise.EventLog(
        df,
        case_col="case",
        activity_col="activity",
        timestamp_col="time",
        case_attributes=["flow_type"],
        exposure_col="event Cumulative net worth (EUR)",
        exposure_agg="max",
    )


@pytest.mark.spec("decision:0001/4")
@pytest.mark.filterwarnings("ignore::wise.LegacySemanticsWarning")
@pytest.mark.parametrize("mode", ["flat", "layer_balanced"])
def test_the_migrated_bpic19_norm_scores_as_the_original_once_fitted_on_the_scored_log(mode: str) -> None:
    """The report names C4/C25 (no rule of the norm reads a missing attribute here), C16 (a column recipe reads what the
    eval read) and C15 (a record fitted on the scored log divides by the quantile 0.1.0 took): no row may differ."""
    path = PKG / "examples" / "bpic19_norm.json"
    log = bpic19_shaped_log()
    migration = migrate_quietly(json.loads(path.read_text(encoding="utf-8")))
    with pytest.raises(NormError, match=r"'exposure_norm' \(quantile_scale\) has an unfitted calibration record"):
        wise.score(log, migration.norm, mode=mode)
    fitted = wise.fit_calibration(migration.norm, log)
    legacy, corrected = wise.score(log, wise.Norm.load(path), mode=mode), wise.score(log, fitted, mode=mode)
    assert corrected.cases["exposure_norm"].gt(0).all()
    assert corrected.violations.notna().any().sum() > 10
    for name in ("cases", "violations", "in_scope", "scores"):
        pd.testing.assert_frame_equal(getattr(legacy, name), getattr(corrected, name), check_exact=True, obj=name)
    for view in legacy.views:
        pd.testing.assert_frame_equal(legacy.contributions[view], corrected.contributions[view], check_exact=True)
