"""Governance-loop diagnostics: censoring, truncation, replication, gap retention."""

import numpy as np
import pandas as pd
import pytest
from conftest import make_log

import wise
from wise.errors import NotScoredError


def test_right_censored(p2p_log):
    log = p2p_log
    # window end = day 30 (A clears on day 30). 10-day window: only A is active late and it is closed.
    assert not wise.right_censored(log, "Clear Invoice", window="10D", window_end="2024-01-31").any()
    rc = wise.right_censored(log, ["Clear Invoice"], window="40D", window_end="2024-01-31")
    assert rc["D"] and rc["E"] and not rc["A"]
    rc3 = wise.right_censored(log, "Clear Invoice", window="40D", opened_by="Record Invoice Receipt", window_end="2024-01-31")
    assert rc3["D"] and not rc3["E"]
    assert not wise.right_censored(log, "Clear Invoice", window="40D", window_end="2024-03-15").any()


def test_right_censored_uses_observation_window_end():
    rows = []
    for i in range(20):  # closed cases ending on days 20..39
        rows += [(f"n{i}", "INV", i, 1, "F"), (f"n{i}", "CLR", 20 + i, 1, "F")]
    rows += [("b", "INV", 35, 1, "F"), ("c", "INV", 10, 1, "F"), ("c", "X", 3000, 1, "F")]  # b open; c has a placeholder date
    log = make_log(rows)
    with pytest.warns(UserWarning, match="observation window"):
        rc = wise.right_censored(log, "CLR", window="10D", q=0.05)
    assert rc["b"] and not rc["n0"] and not rc["n19"]
    win = make_log(rows, window=("2024-01-01", "2024-02-09"))
    assert wise.right_censored(win, "CLR", window="10D")["b"]
    out = wise.timestamp_outliers(log, window=("2024-01-01", "2024-03-01"))
    assert out.sum() == 1


def test_left_truncated():
    rows = [("a", "PO", 0, 1, "F"), ("a", "GR", 5, 1, "F"), ("b", "GR", 2, 1, "F"), ("c", "GR", 200, 1, "F")]
    log = make_log(rows)
    lt = wise.left_truncated(log, "PO", window="30D", window_start="2024-01-01")
    assert lt["b"] and not lt["a"] and not lt["c"]
    assert wise.observation_window(log, q=0.0)[0] == pd.Timestamp("2024-01-01")


def test_event_replication(p2p_log):
    rep = wise.event_replication(p2p_log)
    assert rep.loc["A", "replication_ratio"] == 1.0 and rep.loc["A", "replicated_share"] == 0.0
    dup = pd.concat([p2p_log.events, p2p_log.events[p2p_log.events["case"] == "B"]])
    rep2 = wise.event_replication(wise.EventLog(dup, case_col="case", activity_col="activity", timestamp_col="time"))
    assert rep2.loc["B", "replication_ratio"] == 2.0 and rep2.loc["B", "replicated_share"] == 1.0
    assert rep2.loc["B", "n_distinct_ts"] == 7


def test_cross_case_replication():
    rows = [
        ("d1_i1", "PO", 0, 1, "F"),
        ("d1_i1", "INV", 5, 1, "F"),
        ("d1_i2", "PO", 0, 1, "F"),
        ("d1_i2", "GR", 2, 1, "F"),
        ("d2_i1", "PO", 0, 1, "F"),
    ]
    df = pd.DataFrame(rows, columns=["case", "activity", "day", "amount", "flow_type"])
    df["time"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(df["day"], unit="D")
    df["doc"] = df["case"].str.split("_").str[0]
    log = wise.EventLog(df, case_col="case", activity_col="activity", timestamp_col="time", case_attributes=["doc"])
    share = wise.cross_case_replication(log, "doc")
    assert share["d1_i1"] == pytest.approx(0.5) and share["d1_i2"] == pytest.approx(0.5) and share["d2_i1"] == 0.0
    assert wise.cross_case_replication(log, "doc", keys="activity")["d1_i1"] == pytest.approx(0.5)


def test_gap_retained_uses_fixed_baseline(p2p_result):
    exclude = pd.Series({"E": True})
    out = wise.gap_retained(p2p_result, "Finance", ["company"], exclude=exclude)
    assert out.loc["B", "n_kept"] == 2
    assert out.loc["B", "stable_gap_kept"] < out.loc["B", "stable_gap"] and 0 <= out.loc["B", "retained"] < 1
    assert out.loc["A", "stable_gap_kept"] == 0.0 and np.isnan(out.loc["A", "retained"])  # baseline unchanged
    with pytest.raises(NotScoredError):
        wise.gap_retained(p2p_result, "Finance", ["company"], exclude=pd.Series(True, index=p2p_result.cases.index))


def test_validation_table(p2p_result, p2p_log):
    rc = wise.right_censored(p2p_log, "Clear Invoice", window="40D", window_end="2024-01-31")
    rep = wise.event_replication(p2p_log)
    tab = wise.validation_table(p2p_result, "Finance", ["company"], censored=rc, replication=rep, top=5)
    assert set(tab.columns) >= {"n_cases", "censored_share", "replicated_share", "retained", "reading"}
    assert tab.loc["B", "censored_share"] == pytest.approx(2 / 3)
    assert tab["reading"].str.len().gt(0).all()
    tab2 = wise.validation_table(p2p_result, "Finance", "company", replication=pd.Series(True, index=p2p_result.cases.index))
    assert (tab2["replicated_share"] == 1.0).all() and tab2["reading"].str.contains("replication").all()
