import pandas as pd
import pytest

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


@pytest.fixture
def p2p_norm():
    return wise.running_p2p_norm()


@pytest.fixture
def p2p_log():
    return wise.running_p2p_log()


@pytest.fixture
def p2p_result(p2p_log, p2p_norm):
    return wise.score(p2p_log, p2p_norm)


@pytest.fixture
def helpers():
    return make_log, evaluate
