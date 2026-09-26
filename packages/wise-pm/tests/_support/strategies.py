"""Hypothesis strategies: small random event logs and valid random norms.

Sizes are kept small (at most 40 cases x 8 events, 3-7 constraints) so that
one example scores in milliseconds. Amounts are multiples of 0.25 and day
offsets multiples of 0.5, so sums are exact in floating point and timestamp
ties occur naturally.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from hypothesis import note
from hypothesis import strategies as st

import wise

ACTIVITIES = ["PO", "GR", "INV", "CLR", "CINV", "CHG"]
FLOWS = ["DF1", "DF2"]
REGIONS = ["X", "Y", "Z"]
CONSTRAINT_TYPES = ("presence", "exclusion", "singularity", "lag", "precedence", "balance", "metric")
VIEW_BY_LAYER = "by_layer"
VIEW_BY_CONSTRAINT = "by_constraint"
VIEWS = (VIEW_BY_LAYER, VIEW_BY_CONSTRAINT)
AMOUNT_TOTAL_RECIPE = {"kind": "agg", "column": "amount", "agg": "sum", "name": "amount_total"}


def log_kwargs() -> dict[str, Any]:
    """Keyword arguments that rebuild a strategy log from its ``events`` frame."""
    return {
        "case_col": "case",
        "activity_col": "activity",
        "timestamp_col": "time",
        "case_attributes": ["flow", "region"],
        "exposure_col": "amount",
        "exposure_agg": "sum",
    }


# ------------------------------------------------------------------ event logs
_activity = st.sampled_from(ACTIVITIES)
_day = st.integers(0, 120).map(lambda v: v / 2)  # [0, 60] days in half-day steps
_amount = st.integers(0, 400).map(lambda v: v / 4)  # [0, 100] in quarter steps
_event = st.tuples(_activity, _day, _amount)
_case = st.tuples(st.sampled_from(FLOWS), st.sampled_from(REGIONS), st.lists(_event, min_size=1, max_size=8))


def build_log(cases: list[tuple[str, str, list[tuple[str, float, float]]]]) -> wise.EventLog:
    """``[(flow, region, [(activity, day, amount), ...]), ...]`` -> EventLog."""
    rows = [
        (f"c{i:02d}", act, day, amount, flow, region)
        for i, (flow, region, events) in enumerate(cases)
        for act, day, amount in events
    ]
    df = pd.DataFrame(rows, columns=["case", "activity", "day", "amount", "flow", "region"])
    df["time"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(df["day"], unit="D")
    return wise.EventLog(df, **log_kwargs())


@st.composite
def event_logs(draw: st.DrawFn) -> wise.EventLog:
    """1-40 cases, 1-8 events each, attributes ``flow``/``region`` constant per case."""
    cases = draw(st.lists(_case, min_size=1, max_size=40))
    note(f"log cases: {cases}")
    return build_log(cases)


# ----------------------------------------------------------------------- norms
_labels = st.one_of(_activity, st.lists(_activity, min_size=2, max_size=2, unique=True))
_anchor = st.none() | _activity
_weight = st.floats(0.05, 2.0, allow_nan=False, allow_infinity=False)


def _unit(lo: float, hi: float) -> st.SearchStrategy[float]:
    return st.floats(lo, hi, allow_nan=False, allow_infinity=False)


@st.composite
def constraints(draw: st.DrawFn, kind: str) -> wise.Constraint:
    """One valid constraint of the given type with random parameters."""
    if kind == "presence":
        return wise.Presence(draw(_labels), m=draw(st.integers(1, 3)))
    if kind == "exclusion":
        return wise.Exclusion(draw(_labels), after=draw(_anchor), before=draw(_anchor))
    if kind == "singularity":
        return wise.Singularity(
            draw(_labels), k=draw(st.integers(0, 3)), K=draw(_unit(0.5, 5.0)), after=draw(_anchor), before=draw(_anchor)
        )
    if kind == "lag":
        missing_b: Literal["violate", "skip", "censor"] = draw(st.sampled_from(["violate", "skip", "censor"]))
        activation: Literal["first", "last", "each"] = draw(st.sampled_from(["first", "last", "each"]))
        fixed_response = activation == "each" or missing_b == "censor"
        response: Literal["first_after", "first_overall"] = (
            "first_after" if fixed_response else draw(st.sampled_from(["first_after", "first_overall"]))
        )
        return wise.Lag(
            draw(_labels),
            draw(_labels),
            delta=draw(_unit(0.0, 20.0)),
            width=draw(_unit(0.0, 30.0)),
            missing_a=draw(st.sampled_from(["violate", "skip"])),
            missing_b=missing_b,
            activation=activation,
            response=response,
        )
    if kind == "precedence":
        return wise.Precedence(
            draw(_labels),
            draw(_labels),
            k=draw(st.integers(0, 2)),
            K=draw(_unit(0.5, 3.0)),
            missing_a=draw(st.sampled_from(["skip", "violate"])),
            missing_b=draw(st.sampled_from(["satisfy", "skip"])),
        )
    if kind == "balance":
        # first/last depend on the tie order of events (order_col), so only order-free aggregations
        return wise.Balance(
            "amount",
            "INV",
            "amount",
            "GR",
            tau=draw(_unit(0.0, 1.0)),
            width=draw(_unit(0.0, 2.0)),
            agg=draw(st.sampled_from(["sum", "max", "min", "mean"])),
        )
    if kind == "metric":
        return wise.Metric(
            "amount_total",
            threshold=draw(_unit(0.0, 400.0)),
            width=draw(_unit(0.0, 200.0)),
            direction=draw(st.sampled_from(["high", "low"])),
        )
    raise ValueError(kind)  # pragma: no cover


@st.composite
def norms(draw: st.DrawFn, *, one_per_layer: bool = False) -> wise.Norm:
    """3-7 constraints of random types, 2-4 layers, two views, random applicability.

    ``one_per_layer=True`` gives every constraint its own layer (the case in
    which flat and layer-balanced scoring coincide by construction).
    """
    kinds = draw(st.lists(st.sampled_from(CONSTRAINT_TYPES), min_size=3, max_size=7))
    n_layers = len(kinds) if one_per_layer else draw(st.integers(2, 4))
    layer_ids = [f"L{i + 1}" for i in range(n_layers)]
    cons = []
    for i, kind in enumerate(kinds):
        layer = layer_ids[i] if one_per_layer else draw(st.sampled_from(layer_ids))
        applicability = {"flow": ["DF1"]} if draw(st.booleans()) else {}
        cons.append(
            wise.NormConstraint(f"c{i}", layer, draw(constraints(kind)), weight=draw(_weight), applicability=applicability)
        )
    views = (
        wise.View(VIEW_BY_LAYER, layer_weights={lid: draw(_weight) for lid in layer_ids}),
        wise.View(VIEW_BY_CONSTRAINT, constraint_weights={c.id: draw(_weight) for c in cons}),
    )
    return wise.Norm(
        tuple(cons),
        tuple(wise.Layer(lid) for lid in layer_ids),
        views,
        scoring_mode=draw(st.sampled_from(["layer_balanced", "flat"])),
        derived_attributes=(AMOUNT_TOTAL_RECIPE,),
    )
