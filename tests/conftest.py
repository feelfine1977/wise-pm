import numpy as np
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


# --------------------------------------------------------------------------
# Stage-2 (wise.explain) populations: deliberately awkward, because the
# explanation contract is about what happens at the edges.
# --------------------------------------------------------------------------
@pytest.fixture
def previous_period_result(p2p_norm):
    """A *different* population under the same norm, view and mode.

    Cases A, B and C only, and B's invoice arrives two days later, so the
    comparator frozen from this run is not the current population's mean. A
    historical comparator is expected to come from another population; that is
    the point of it.
    """
    events = wise.running_p2p_events()
    events = events[events["case"].isin(["A", "B", "C"])].copy()
    late = (events["case"] == "B") & (events["activity"] == "Record Invoice Receipt")
    events.loc[late, "time"] = events.loc[late, "time"] + pd.Timedelta(days=2)
    log = wise.EventLog(
        events,
        case_col="case",
        activity_col="activity",
        timestamp_col="time",
        case_attributes=["flow_type", "company", "vendor"],
    )
    return wise.score(log, p2p_norm)


def _split_view_norm() -> wise.Norm:
    """A norm whose two views score different cases.

    ``c_one`` applies to DF1 only and ``c_two`` to DF2 only; view ``Left``
    weights only ``c_one`` and view ``Right`` only ``c_two``. A DF1 case is
    therefore scored under ``Left`` and unscored under ``Right``, which is the
    "deliberately different scored populations" case of the acceptance list.
    """
    layers = (wise.Layer("one"), wise.Layer("two"))
    constraints = (
        wise.NormConstraint("c_one", "one", wise.Presence("A", m=1), applicability={"flow_type": ["DF1"]}),
        wise.NormConstraint("c_two", "two", wise.Presence("B", m=1), applicability={"flow_type": ["DF2"]}),
    )
    views = (
        wise.View("Left", constraint_weights={"c_one": 1.0, "c_two": 0.0}),
        wise.View("Right", constraint_weights={"c_one": 0.0, "c_two": 1.0}),
    )
    return wise.Norm(constraints, layers, views, name="split", version="1", scoring_mode="flat")


@pytest.fixture
def split_population_result():
    """Scored result whose ``Left`` and ``Right`` views score different cases."""
    rows = []
    for case, flow, activities in [
        ("d1", "DF1", ["A", "Z"]),
        ("d2", "DF1", ["Z"]),
        ("d3", "DF2", ["B", "Z"]),
        ("d4", "DF2", ["Z"]),
    ]:
        for day, activity in enumerate(activities):
            rows.append(
                {
                    "case": case,
                    "activity": activity,
                    "time": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                    "flow_type": flow,
                    "team": "T1" if case in ("d1", "d2") else "T2",
                }
            )
    log = wise.EventLog(
        pd.DataFrame(rows),
        case_col="case",
        activity_col="activity",
        timestamp_col="time",
        case_attributes=["flow_type", "team"],
    )
    return wise.score(log, _split_view_norm())


@pytest.fixture
def null_key_result(p2p_norm):
    """The running example where one case has no ``company`` at all."""
    events = wise.running_p2p_events().copy()
    events.loc[events["case"] == "E", "company"] = np.nan
    log = wise.EventLog(
        events,
        case_col="case",
        activity_col="activity",
        timestamp_col="time",
        case_attributes=["flow_type", "company", "vendor"],
    )
    return wise.score(log, p2p_norm)


@pytest.fixture
def exposure_result(p2p_norm):
    """The running example with an exposure column that is zero for company B."""
    events = wise.running_p2p_events().copy()
    events["value"] = np.where(events["company"] == "A", 100.0, 0.0)
    log = wise.EventLog(
        events,
        case_col="case",
        activity_col="activity",
        timestamp_col="time",
        case_attributes=["flow_type", "company", "vendor"],
        exposure_col="value",
    )
    return wise.score(log, p2p_norm)
