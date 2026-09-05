"""Constraint semantics, checked against the worked examples in paper Sec. IV-B."""

import numpy as np
import pandas as pd
import pytest
from conftest import evaluate, make_log

import wise
from wise.constraints import constraint_from_dict
from wise.errors import NormError


def test_sat_paper_examples():
    assert wise.sat(8, 10, 20) == 0.0
    assert wise.sat(15, 10, 20) == pytest.approx(0.25)
    assert wise.sat(20, 10, 20) == pytest.approx(0.5)
    assert wise.sat(30, 10, 20) == 1.0
    s = wise.sat(pd.Series([8, 15, np.nan, 30]), 10, 20)
    assert s.tolist()[:2] == [0.0, 0.25] and np.isnan(s.iloc[2]) and s.iloc[3] == 1.0
    assert wise.sat(10, 10, 0) == 0.0 and wise.sat(10.1, 10, 0) == 1.0  # zero width = step
    assert wise.sat(np.array([0, 40]), 10, 20).tolist() == [0.0, 1.0]
    assert isinstance(wise.sat(np.int64(15), 10, 20), float)


def test_as_labels_accepts_a_single_string():
    assert wise.as_labels("Record Goods Receipt") == ("Record Goods Receipt",)
    assert wise.as_labels(["a", "b"]) == ("a", "b")
    with pytest.raises(NormError):
        wise.as_labels([])
    assert wise.Presence("abc").activity == ("abc",)


def test_presence_indicator_and_partial():
    log = make_log([("a", "INV", 0, 1, "F"), ("a", "INV", 1, 1, "F"), ("b", "GR", 0, 1, "F")])
    v = evaluate(log, wise.Presence("INV"))
    assert v["a"] == 0.0 and v["b"] == 1.0
    v = evaluate(log, wise.Presence("INV", m=4))  # 1 − min(2/4, 1)
    assert v["a"] == pytest.approx(0.5) and v["b"] == 1.0


def test_exclusion_plain_and_scoped():
    log = make_log(
        [
            ("a", "CINV", 0, 1, "F"),
            ("b", "GR", 0, 1, "F"),
            ("c", "Change Price", 0, 1, "F"),
            ("c", "GR", 1, 1, "F"),
            ("c", "Change Price", 2, 1, "F"),
            ("d", "Change Price", 0, 1, "F"),
            ("d", "GR", 1, 1, "F"),
            ("e", "Change Price", 0, 1, "F"),
        ]
    )
    v = evaluate(log, wise.Exclusion("CINV"))
    assert v["a"] == 1.0 and v["b"] == 0.0
    after = evaluate(log, wise.Exclusion("Change Price", after="GR"))
    assert after["c"] == 1.0 and after["d"] == 0.0 and np.isnan(after["e"])  # e has no GR anchor
    before = evaluate(log, wise.Exclusion("Change Price", before="GR"))
    assert before["c"] == 1.0 and before["d"] == 1.0 and before["b"] == 0.0


def test_singularity_paper_example_and_scope():
    rows = []
    for case, n in [("c2", 2), ("c3", 3), ("c4", 4), ("c5", 5), ("c6", 6)]:
        rows += [(case, "GR", d, 1, "F") for d in range(n)]
    v = evaluate(make_log(rows), wise.Singularity("GR", k=2, K=3))
    assert v["c2"] == 0.0
    assert v["c3"] == pytest.approx(1 / 3)
    assert v["c4"] == pytest.approx(2 / 3)
    assert v["c5"] == 1.0 and v["c6"] == 1.0
    log = make_log([("x", "PO", 0, 1, "F"), ("x", "GR", 1, 1, "F"), ("x", "GR", 2, 1, "F"), ("x", "GR", 3, 1, "F")])
    assert evaluate(log, wise.Singularity("GR", k=1, K=2, after="PO"))["x"] == pytest.approx(1.0)


def test_lag_paper_example_and_first_b_after_a():
    rows = [
        ("l10", "GR", 0, 1, "F"),
        ("l10", "INV", 10, 1, "F"),
        ("l20", "GR", 0, 1, "F"),
        ("l20", "INV", 20, 1, "F"),
        ("l30", "GR", 0, 1, "F"),
        ("l30", "INV", 30, 1, "F"),
        ("pre", "INV", -3, 1, "F"),
        ("pre", "GR", 0, 1, "F"),
        ("pre", "INV", 5, 1, "F"),
        ("only_pre", "INV", -3, 1, "F"),
        ("only_pre", "GR", 0, 1, "F"),
        ("no_a", "INV", 0, 1, "F"),
        ("no_b", "GR", 0, 1, "F"),
    ]
    log = make_log(rows)
    v = evaluate(log, wise.Lag("GR", "INV", delta=10, width=20))
    assert v["l10"] == 0.0 and v["l20"] == pytest.approx(0.5) and v["l30"] == 1.0
    assert v["pre"] == 0.0  # the first INV *after* GR counts
    assert v["only_pre"] == 1.0 and v["no_b"] == 1.0 and v["no_a"] == 1.0  # undefined endpoints → ν = 1
    skip_a = evaluate(log, wise.Lag("GR", "INV", delta=10, width=20, missing_a="skip"))
    assert np.isnan(skip_a["no_a"]) and skip_a["no_b"] == 1.0
    skip = evaluate(log, wise.Lag("GR", "INV", delta=10, width=20, missing_b="skip"))
    assert np.isnan(skip["no_b"]) and np.isnan(skip["only_pre"]) and skip["l20"] == pytest.approx(0.5)
    shorthand = evaluate(log, wise.Lag("GR", "INV", delta=10, width=20, missing="violate"))
    assert shorthand["no_a"] == 1.0 and shorthand["no_b"] == 1.0
    assert "missing" not in wise.Lag("GR", "INV", missing="skip").params()


def test_lag_censor_uses_observation_window():
    rows = [(f"c{i}", "GR", i, 1, "F") for i in range(600)] + [(f"c{i}", "INV", i + 5, 1, "F") for i in range(600)]
    rows += [("open", "GR", 590, 1, "F"), ("late", "GR", 0, 1, "F"), ("late", "X", 3000, 1, "F")]
    log = make_log(rows)  # observation window ends at day 604, the raw maximum at day 3000
    with pytest.warns(UserWarning, match="observation window"):
        v = evaluate(log, wise.Lag("GR", "INV", delta=10, width=20, missing_b="censor"))
    assert v["open"] == pytest.approx(0.2)  # 14 days open so far → sat(14; 10, 20)
    explicit = make_log(rows, window=("2024-01-01", "2024-01-31"))
    assert evaluate(explicit, wise.Lag("GR", "INV", delta=10, width=20, missing_b="censor"))["late"] == pytest.approx(1.0)


def test_lag_response_first_overall_and_censor():
    rows = [
        ("pre", "INV", -3, 1, "F"),
        ("pre", "GR", 0, 1, "F"),
        ("pre", "INV", 5, 1, "F"),
        ("open", "GR", 0, 1, "F"),
        ("late", "GR", 0, 1, "F"),
        ("late", "INV", 25, 1, "F"),
    ]
    log = make_log(rows, window=("2024-01-01", "2024-01-26"))
    first_overall = evaluate(log, wise.Lag("GR", "INV", delta=10, width=20, response="first_overall"))
    assert first_overall["pre"] == 1.0  # first INV precedes GR → response missing → violate
    assert first_overall["late"] == pytest.approx(0.75)
    censor = evaluate(log, wise.Lag("GR", "INV", delta=10, width=20, missing_b="censor"))
    assert censor["open"] == pytest.approx(0.75)  # 25 days open so far → sat(25; 10, 20)
    assert censor["late"] == pytest.approx(0.75)


def test_lag_activation_each_and_last():
    rows = [
        ("x", "GR", 0, 1, "F"),
        ("x", "GR", 10, 1, "F"),
        ("x", "INV", 12, 1, "F"),
        ("y", "GR", 0, 1, "F"),
        ("y", "INV", 2, 1, "F"),
        ("y", "GR", 20, 1, "F"),
    ]
    log = make_log(rows)
    each = evaluate(log, wise.Lag("GR", "INV", delta=10, width=20, activation="each"))
    assert each["x"] == pytest.approx((0.1 + 0.0) / 2)  # activations: 12 days → 0.1, 2 days → 0
    assert each["y"] == pytest.approx((0.0 + 1.0) / 2)  # second GR has no INV after it → 1
    last = evaluate(log, wise.Lag("GR", "INV", delta=10, width=20, activation="last"))
    assert last["x"] == 0.0 and last["y"] == 1.0
    each_skip = evaluate(log, wise.Lag("GR", "INV", delta=10, width=20, activation="each", missing_b="skip"))
    assert each_skip["y"] == 0.0


def test_lag_units_and_unbounded():
    log = make_log([("h", "A", 0, 1, "F"), ("h", "B", 0.5, 1, "F")])  # 12 hours
    assert evaluate(log, wise.Lag("A", "B", delta=6, width=12, unit="h"))["h"] == pytest.approx(0.5)
    assert evaluate(log, wise.Lag("A", "B", delta=360, width=720, unit="min"))["h"] == pytest.approx(0.5)
    assert wise.Lag("A", "B", unit="H").unit == "h" and wise.Lag("A", "B", unit="T").unit == "min"
    with pytest.raises(NormError):
        wise.Lag("A", "B", unit="fortnight")
    # delta=None: "b must follow a", any positive lag is fine
    follow = wise.Lag("A", "B", delta=None)
    v = evaluate(
        make_log([("ok", "A", 0, 1, "F"), ("ok", "B", 5, 1, "F"), ("bad", "B", 0, 1, "F"), ("bad", "A", 1, 1, "F")]), follow
    )
    assert v["ok"] == 0.0 and v["bad"] == 1.0


def test_precedence():
    rows = [
        ("ok", "GR", 0, 1, "F"),
        ("ok", "INV", 5, 1, "F"),
        ("pre", "INV", -3, 1, "F"),
        ("pre", "GR", 0, 1, "F"),
        ("pre", "INV", 5, 1, "F"),
        ("two_pre", "INV", -3, 1, "F"),
        ("two_pre", "INV", -2, 1, "F"),
        ("two_pre", "GR", 0, 1, "F"),
        ("no_a", "INV", 0, 1, "F"),
        ("no_b", "GR", 0, 1, "F"),
    ]
    log = make_log(rows)
    v = evaluate(log, wise.Precedence("GR", "INV"))
    assert v["ok"] == 0.0 and v["pre"] == 1.0 and v["two_pre"] == 1.0
    assert np.isnan(v["no_a"]) and v["no_b"] == 0.0  # vacuous truth without any b
    graded = evaluate(log, wise.Precedence("GR", "INV", k=0, K=2))
    assert graded["pre"] == pytest.approx(0.5) and graded["two_pre"] == 1.0
    strict = evaluate(log, wise.Precedence("GR", "INV", missing_a="violate", missing_b="skip"))
    assert strict["no_a"] == 1.0 and np.isnan(strict["no_b"])


def test_balance_paper_semantics_and_agg():
    rows = [
        ("exact", "GR", 0, 100, "F"),
        ("exact", "INV", 1, 100, "F"),
        ("mis", "GR", 0, 100, "F"),
        ("mis", "INV", 1, 82, "F"),
        ("multi", "GR", 0, 40, "F"),
        ("multi", "GR", 1, 60, "F"),
        ("multi", "INV", 2, 100, "F"),
        ("missing", "GR", 0, 100, "F"),
        ("empty", "CLR", 0, 5, "F"),
    ]
    log = make_log(rows)
    v = evaluate(log, wise.Balance("amount", "INV", "amount", "GR", tau=0.05, width=0.20))
    assert v["exact"] == 0.0
    assert v["mis"] == pytest.approx((0.18 - 0.05) / 0.20)  # 0.65 as in Table VI
    assert v["multi"] == 0.0 and v["missing"] == 1.0 and v["empty"] == 0.0
    cumulative = evaluate(log, wise.Balance("amount", "INV", "amount", "GR", tau=0.05, width=0.50, agg="max"))
    assert cumulative["multi"] == pytest.approx((0.4 - 0.05) / 0.5)  # max(40, 60) = 60 vs 100
    neg = make_log([("n", "GR", 0, -100, "F"), ("n", "INV", 1, 100, "F")])
    with pytest.raises(NormError):
        evaluate(neg, wise.Balance("amount", "INV", "amount", "GR"))


def test_metric_extension():
    log = make_log([("a", "X", 0, 1, "F"), ("b", "X", 0, 1, "F"), ("c", "X", 0, 1, "F")])
    log.add_case_attribute("touches", pd.Series({"a": 2, "b": 8}))
    v = evaluate(log, wise.Metric("touches", threshold=4, width=8))
    assert v["a"] == 0.0 and v["b"] == pytest.approx(0.5) and np.isnan(v["c"])
    v = evaluate(log, wise.Metric("touches", threshold=4, width=2, direction="low"))
    assert v["a"] == 1.0 and v["b"] == 0.0


def test_applicability_masks_to_nan():
    log = make_log([("a", "INV", 0, 1, "DF1"), ("b", "GR", 0, 1, "DF2")])
    v = evaluate(log, wise.Presence("INV"), applicability={"flow_type": ["DF1"]})
    assert v["a"] == 0.0 and np.isnan(v["b"])
    v = evaluate(log, wise.Presence("INV"), applicability={"flow_type": "DF1"})  # bare string, not characters
    assert v["a"] == 0.0 and np.isnan(v["b"])
    with pytest.raises(NormError):
        evaluate(log, wise.Presence("INV"), applicability={"nope": ["x"]})


def test_activity_lists_merge_labels():
    log = make_log([("a", "Vendor creates invoice", 0, 1, "F"), ("b", "GR", 0, 1, "F")])
    v = evaluate(log, wise.Presence(["Record Invoice Receipt", "Vendor creates invoice"]))
    assert v["a"] == 0.0 and v["b"] == 1.0


def test_constraint_from_dict_aliases_coercion_and_validation():
    c = constraint_from_dict("pres", {"activity": "INV", "m": "2"})
    assert isinstance(c, wise.Presence) and c.m == 2
    assert isinstance(constraint_from_dict("sing", {"activity": "GR", "k": 1, "K": "2"}), wise.Singularity)
    assert isinstance(constraint_from_dict("order", {"a": "GR", "b": "INV"}), wise.Precedence)
    with pytest.raises(NormError):
        constraint_from_dict("presence", {"activity": "INV", "bogus": 1})
    with pytest.raises(NormError):
        constraint_from_dict("unknown_type", {})
    for bad in (
        lambda: wise.Lag("A", "B", missing="maybe"),
        lambda: wise.Lag("A", "B", delta=-1),
        lambda: wise.Lag("A", "B", width=-5),
        lambda: wise.Lag("A", "B", activation="each", response="first_overall"),
        lambda: wise.Singularity("A", k=-1),
        lambda: wise.Singularity("A", K=0),
        lambda: wise.Presence("A", m=0),
        lambda: wise.Balance("x", "A", "y", "B", tau=2.0),
        lambda: wise.Metric("x", width=-1),
        lambda: wise.Metric("", 1, 1),
        lambda: wise.Precedence("A", "B", k=1.5),
        lambda: wise.Lag("A", "B", delta=None, missing_b="censor"),
        lambda: wise.Lag("A", "B", delta=1, width=1, missing_b="censor", response="first_overall"),
    ):
        with pytest.raises(NormError):
            bad()


def test_constraints_are_hashable_and_comparable():
    assert wise.Presence("a") == wise.Presence(["a"])
    assert hash(wise.Presence("a")) == hash(wise.Presence(["a"]))
    assert wise.Lag("a", "b", delta=1, width=2).params()["a"] == ["a"]
    assert wise.Exclusion("a").describe().startswith("exclusion(")
