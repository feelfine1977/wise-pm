"""EventLog construction, robustness, primitives, and derived attributes."""

import numpy as np
import pandas as pd
import pytest
from conftest import make_log

import wise
from wise.errors import LogSchemaError, NormError


def test_missing_columns_and_defaults(p2p_log):
    df = p2p_log.events
    with pytest.raises(LogSchemaError, match="missing columns"):
        wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="nope")
    pm = df.rename(columns={"case": "case:concept:name", "activity": "concept:name", "time": "time:timestamp"})
    log = wise.EventLog(pm)  # pm4py defaults
    assert len(log) == 5 and log.count("Record Goods Receipt")["B"] == 4
    assert set(log.to_pm4py().columns) >= {"case:concept:name", "concept:name", "time:timestamp"}
    assert "EventLog(" in repr(log) and log.activity_labels


def test_categorical_and_filtered_case_ids(p2p_log):
    df = p2p_log.events.copy()
    df["case"] = pd.Categorical(df["case"], categories=list("ABCDEFG"))
    log = wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="time")
    assert list(log.case_ids) == list("ABCDE")  # unused categories are not cases
    filtered = df[df["case"] != "E"]
    log2 = wise.EventLog(filtered, case_col="case", activity_col="activity", timestamp_col="time")
    assert "E" not in log2.case_ids
    df["activity"] = df["activity"].astype("category")
    log3 = wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="time")
    assert log3.count("Record Goods Receipt")["B"] == 4
    assert isinstance(log3.events["activity"].dtype, pd.CategoricalDtype)


def test_null_case_ids_and_int_ids(p2p_log):
    df = p2p_log.events.copy()
    df.loc[0, "case"] = None
    with pytest.raises(LogSchemaError, match="null case id"):
        wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="time")
    ints = p2p_log.events.assign(case=lambda d: d["case"].map({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}))
    log = wise.EventLog(ints, case_col="case", activity_col="activity", timestamp_col="time")
    assert log.count("Record Goods Receipt")[2] == 4 and log.cases.index.dtype.kind == "i"


def test_bare_string_arguments(p2p_log):
    assert p2p_log.count("Record Goods Receipt")["B"] == 4
    assert p2p_log.first_ts("Record Goods Receipt")["A"] == pd.Timestamp("2024-01-01")
    log = wise.EventLog(p2p_log.events, case_col="case", activity_col="activity", timestamp_col="time", case_attributes="company")
    assert log.case_attributes == ["company"]


def test_timestamps_missing_tz_and_strings(p2p_log):
    df = p2p_log.events.copy()
    df.loc[0, "time"] = pd.NaT
    with pytest.raises(LogSchemaError, match="null or unparseable"):
        wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="time")
    kept = wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="time", missing_timestamps="keep")
    assert kept.n_missing_timestamps == 1 and len(kept) == 5
    dropped = wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="time", missing_timestamps="drop")
    assert len(dropped.events) == len(df) - 1
    strings = p2p_log.events.assign(time=lambda d: d["time"].astype(str))
    assert pd.api.types.is_datetime64_any_dtype(
        wise.EventLog(strings, case_col="case", activity_col="activity", timestamp_col="time").events["time"]
    )
    aware = p2p_log.events.assign(time=lambda d: d["time"].dt.tz_localize("Europe/Amsterdam"))
    log_tz = wise.EventLog(
        aware, case_col="case", activity_col="activity", timestamp_col="time", case_attributes=["company", "flow_type"]
    )
    res = wise.score(log_tz, wise.running_p2p_norm())
    pd.testing.assert_frame_equal(res.scores, wise.score(p2p_log, wise.running_p2p_norm()).scores)
    rc = wise.right_censored(log_tz, "Clear Invoice", window="40D", window_end=pd.Timestamp("2024-01-31"))
    assert rc["D"] and not rc["A"]
    mixed = strings.copy()
    mixed.loc[0, "time"] = "2024-01-01 00:00:00+01:00"
    with pytest.raises(LogSchemaError):
        wise.EventLog(mixed, case_col="case", activity_col="activity", timestamp_col="time")
    utc = wise.EventLog(mixed, case_col="case", activity_col="activity", timestamp_col="time", utc=True)
    assert str(utc.tz) == "UTC"


def test_order_col_ties_lifecycle_dedupe_keep_columns():
    rows = [
        {"case": "x", "activity": "INV", "time": "2024-01-01", "seq": 2, "lc": "complete", "extra": 1},
        {"case": "x", "activity": "GR", "time": "2024-01-01", "seq": 1, "lc": "complete", "extra": 1},
        {"case": "x", "activity": "GR", "time": "2024-01-01", "seq": 1, "lc": "start", "extra": 1},
    ]
    df = pd.DataFrame(rows)
    log = wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="time", order_col="seq", lifecycle_col="lc")
    assert log.events["activity"].tolist() == ["GR", "INV"]  # start row dropped, seq breaks the tie
    one = wise.EventLog(
        df, case_col="case", activity_col="activity", timestamp_col="time", lifecycle_col="lc", keep_transitions="complete"
    )
    assert len(one.events) == 2
    t_a, t_b = log.first_after("GR", "INV")
    assert t_b["x"] == pd.Timestamp("2024-01-01")
    # events with equal timestamps are simultaneous for every lag reading
    nc = wise.NormConstraint("c", "L", wise.Lag("GR", "INV", delta=10, width=20, activation="each"))
    assert wise.evaluate_constraint(log, nc)["x"] == 0.0
    assert wise.evaluate_constraint(log, wise.NormConstraint("c", "L", wise.Precedence("GR", "INV")))["x"] == 0.0
    dup = pd.concat([df, df]).drop(columns="lc")
    assert len(wise.EventLog(dup, case_col="case", activity_col="activity", timestamp_col="time", dedupe=True).events) == 2
    slim = wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="time", keep_columns="seq")
    assert "extra" not in slim.events.columns and "seq" in slim.events.columns
    euro = pd.DataFrame({"case": ["x"], "activity": ["A"], "time": ["05/01/2024"]})
    assert wise.EventLog(euro, case_col="case", activity_col="activity", timestamp_col="time", dayfirst=True).cases[
        "first_ts"
    ].iloc[0] == pd.Timestamp("2024-01-05")
    assert wise.EventLog(euro, case_col="case", activity_col="activity", timestamp_col="time", timestamp_format="%d/%m/%Y").cases[
        "first_ts"
    ].iloc[0] == pd.Timestamp("2024-01-05")


def test_primitives(p2p_log):
    log = p2p_log
    assert log.count(["Record Goods Receipt", "Record Invoice Receipt"])["B"] == 5
    assert log.last_ts("Record Goods Receipt")["B"] == pd.Timestamp("2024-01-04")
    before = log.count_before("Record Goods Receipt", "Record Invoice Receipt")
    assert before["B"] == 4 and np.isnan(before["E"])
    after = log.count_after("Clear Invoice", "Record Invoice Receipt")
    assert after["A"] == 1 and after["D"] == 0
    assert log.total("amount", "Record Goods Receipt")["B"] == 100
    assert log.total("amount", "Record Goods Receipt", agg="max")["B"] == 25
    with pytest.raises(LogSchemaError):
        log.total("nope", "X")
    with pytest.raises(LogSchemaError):
        log.attribute("nope")
    with pytest.raises(NormError):
        log.count([])
    assert len(log.trace("B")) == 7 and log.trace("B")["activity"].iloc[-1] == "Clear Invoice"
    pairs = log.activation_lags("Record Goods Receipt", "Record Invoice Receipt")
    assert len(pairs) == 8 and pairs["t_b"].isna().sum() == 1  # E's GR has no invoice


def test_reserved_attribute_names(p2p_log):
    df = p2p_log.events.assign(n_events=1)
    with pytest.raises(LogSchemaError, match="reserved"):
        wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="time", case_attributes=["n_events"])
    with pytest.raises(LogSchemaError, match="reserved"):
        p2p_log.add_case_attribute("contrib__x", np.arange(5))


def test_add_case_attribute_alignment(p2p_log):
    p2p_log.add_case_attribute("m", {"A": 1, "B": 2})
    assert p2p_log.cases.loc["B", "m"] == 2 and np.isnan(p2p_log.cases.loc["C", "m"])
    p2p_log.add_case_attribute("arr", np.arange(5))
    assert p2p_log.cases.loc["E", "arr"] == 4
    with pytest.raises(LogSchemaError):
        p2p_log.add_case_attribute("bad", pd.Series({1: 1, 2: 2}))
    with pytest.raises(LogSchemaError):
        p2p_log.add_case_attribute("bad", np.arange(3))
    assert "m" in p2p_log.case_attributes


def test_exposure_and_window(p2p_log):
    df = p2p_log.events
    log = wise.EventLog(
        df, case_col="case", activity_col="activity", timestamp_col="time", exposure_col="amount", exposure_agg="sum"
    )
    assert log.cases.loc["B", "exposure"] == pytest.approx(200.0)
    with pytest.raises(LogSchemaError, match="negative"):
        wise.EventLog(
            df.assign(amount=-df["amount"]), case_col="case", activity_col="activity", timestamp_col="time", exposure_col="amount"
        )
    win = wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="time", window=("2024-01-01", "2024-02-01"))
    assert win.window_end == pd.Timestamp("2024-02-01") and win.observation_window() == win.window
    with pytest.raises(ValueError):
        wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="time", window=("2024-02-01", "2024-01-01"))
    start, end = p2p_log.observation_window(q=0.0)
    assert start == pd.Timestamp("2023-12-31") and end == pd.Timestamp("2024-01-31")
    assert p2p_log.observation_window(q=0.1) == (start, end)  # too few cases to trim
    small = make_log([(f"c{i}", "A", 20 + i, 1, "F") for i in range(15)] + [("out", "A", 5000, 1, "F")])
    assert small.observation_window(q=0.05)[1] == pd.Timestamp("2024-01-01") + pd.Timedelta(days=34)


def test_validate_report():
    rows = [("a", "GR", 0, 1, "F"), ("a", "INV", 0, 1, "F"), ("b", "GR", 0, 1, "G"), ("b", "INV", 400, 1, "H")]
    log = make_log(rows)
    rep = log.validate(q=0.0)
    assert rep["n_cases"] == 2 and rep["tied_events_share"] == pytest.approx(0.5)
    assert rep["cases_with_ties_share"] == pytest.approx(0.5)
    assert rep["varying_within_case[flow_type]"] == 1
    assert rep["timestamp_outliers"] == 0


def test_derive_recipes(p2p_log):
    log = p2p_log
    names = log.derive(
        [
            {"name": "n_gr", "kind": "count", "activities": ["Record Goods Receipt"]},
            {"name": "n_after_inv", "kind": "count", "activities": "Record Goods Receipt", "after": "Record Invoice Receipt"},
            {"name": "n_gr_events", "kind": "count_events", "where": {"column": "activity", "in": ["Record Goods Receipt"]}},
            {"name": "n_labels", "kind": "nunique", "column": "activity"},
            {"name": "max_amount", "kind": "agg", "column": "amount", "agg": "max"},
            {"name": "amount_cv", "kind": "cv", "column": "amount"},
            {"name": "gr_to_inv", "kind": "lag", "a": "Record Goods Receipt", "b": "Record Invoice Receipt"},
            {"name": "share", "kind": "ratio", "numerator": "n_gr", "denominator": "n_events"},
            {"name": "amount_norm", "kind": "quantile_scale", "attribute": "max_amount", "q": 0.5},
            {"name": "cancel_x", "kind": "indicator_times", "activities": ["Cancel Invoice Receipt"], "attribute": "amount_norm"},
            {"name": "evald", "kind": "eval", "expr": "n_gr / n_events"},
            {"name": "regex", "kind": "count_events", "where": {"column": "activity", "regex": "^Record"}},
        ]
    )
    c = log.cases
    assert names[0] == "n_gr" and c.loc["B", "n_gr"] == 4 and c.loc["B", "n_after_inv"] == 0
    assert c.loc["B", "n_gr_events"] == 4 and c.loc["B", "n_labels"] == 4 and c.loc["B", "max_amount"] == 100
    assert c.loc["A", "amount_cv"] == 0.0 and c.loc["B", "amount_cv"] > 0
    assert c.loc["A", "gr_to_inv"] == 25 and np.isnan(c.loc["E", "gr_to_inv"])
    assert c.loc["B", "share"] == pytest.approx(4 / 7) and c.loc["B", "evald"] == pytest.approx(4 / 7)
    assert c.loc["D", "cancel_x"] > 0 and c.loc["A", "cancel_x"] == 0
    assert c.loc["A", "regex"] == 2
    with pytest.raises(NormError):
        log.derive([{"name": "x", "kind": "nope"}])
    with pytest.raises(LogSchemaError):
        log.derive([{"name": "x", "kind": "agg", "column": "nope", "agg": "sum"}])
    assert log.derive([{"name": "n_gr", "kind": "count", "activities": ["Record Goods Receipt"]}], overwrite=False) == ["n_gr"]
    assert c.loc["B", "n_gr"] == 4
    log.derive([{"name": "n_gr", "kind": "eval", "expr": "0"}], overwrite=False)  # a different recipe is recomputed
    assert log.cases.loc["B", "n_gr"] == 0


def test_from_csv(tmp_path, p2p_log):
    p = tmp_path / "log.csv"
    p2p_log.events.to_csv(p, index=False)
    log = wise.EventLog.from_csv(
        str(p), case_col="case", activity_col="activity", timestamp_col="time", case_attributes=["company"]
    )
    assert len(log) == 5
    pq = tmp_path / "log.parquet"
    try:
        p2p_log.events.to_parquet(pq)
    except ImportError:
        pytest.skip("no parquet engine")
    assert len(wise.EventLog.from_csv(str(pq), case_col="case", activity_col="activity", timestamp_col="time")) == 5
