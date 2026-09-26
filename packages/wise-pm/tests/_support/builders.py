"""Builders for small event logs and single-constraint evaluations used across the suite."""

from __future__ import annotations

import pandas as pd

import wise


def make_log(rows, attrs=("flow_type",), **kwargs) -> wise.EventLog:
    """rows: (case, activity, day, amount, flow_type) tuples → EventLog.

    ``day`` may be fractional; timestamps start at 2024-01-01.
    """
    df = pd.DataFrame(rows, columns=["case", "activity", "day", "amount", "flow_type"])
    df["time"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(df["day"], unit="D")
    return wise.EventLog(
        df, case_col="case", activity_col="activity", timestamp_col="time", case_attributes=list(attrs), **kwargs
    )


def evaluate(log, constraint, applicability=None, layer="L"):
    nc = wise.NormConstraint("c", layer, constraint, applicability=applicability or {})
    return wise.evaluate_constraint(log, nc)
