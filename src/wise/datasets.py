"""Bundled example data: the paper's running purchase-to-pay example.

:func:`running_p2p_norm` builds the norm of Table V (six constraints in five
layers, views Finance and Logistics). :func:`running_p2p_log` builds an
event log whose five cases σA … σE reproduce the key facts of Table VI, so
that scoring them yields exactly the scores printed in the paper.
"""

from __future__ import annotations

import pandas as pd

from .constraints import Balance, Exclusion, Lag, Presence, Singularity
from .log import EventLog
from .norm import Layer, Norm, NormConstraint, View

GR = "Record Goods Receipt"
INV = "Record Invoice Receipt"
CLR = "Clear Invoice"
CINV = "Cancel Invoice Receipt"
PO = "Create Purchase Order Item"


def running_p2p_norm() -> Norm:
    """Norm ``N`` of Table V with the Finance and Logistics views.

    Constraint ``c4`` (ordered vs. received amount) is declared applicable to
    flow type ``DF1`` only; the illustrative cases are ``DF2`` items, for
    which the paper assumes ``c4`` not applicable. The scoring mode is
    ``flat``: Table VI computes the scores as weighted sums of the
    applicable raw weights.
    """
    layers = (
        Layer("completeness", "Core completeness"),
        Layer("lead_times", "Working-capital lead times"),
        Layer("match", "Match consistency"),
        Layer("handling", "Handling discipline"),
        Layer("exceptions", "Exception discipline"),
    )
    constraints = (
        NormConstraint("c1", "completeness", Presence(INV, m=1), description="Require an invoice receipt"),
        NormConstraint(
            "c2",
            "lead_times",
            Lag(GR, INV, delta=10, width=20, unit="D"),
            description="GR → INV within 10 days",
        ),
        NormConstraint(
            "c3", "match", Balance("amount", INV, "amount", GR, tau=0.05, width=0.20), description="Invoiced vs received amount"
        ),
        NormConstraint(
            "c4",
            "match",
            Balance("amount", PO, "amount", GR, tau=0.01, width=0.02),
            applicability={"flow_type": ["DF1"]},
            description="Ordered vs received amount",
        ),
        NormConstraint("c5", "handling", Singularity(GR, k=2, K=3), description="Limit delivery fragmentation"),
        NormConstraint("c6", "exceptions", Exclusion(CINV), description="Avoid invoice cancellations"),
    )
    views = (
        View("Finance", constraint_weights={"c1": 0.20, "c2": 0.45, "c3": 0.20, "c4": 0.05, "c5": 0.05, "c6": 0.10}),
        View("Logistics", constraint_weights={"c1": 0.25, "c2": 0.15, "c3": 0.05, "c4": 0.25, "c5": 0.45, "c6": 0.10}),
    )
    return Norm(constraints, layers, views, name="Running P2P example (paper Table V)", version="1", scoring_mode="flat")


def running_p2p_events() -> pd.DataFrame:
    """Events of the five illustrative cases of Table VI."""
    t0 = pd.Timestamp("2024-01-01")
    rows: list[dict[str, object]] = []

    def ev(case: str, act: str, day: float, amount: float | None = None, company: str = "A", vendor: str = "V1") -> None:
        rows.append(
            {
                "case": case,
                "activity": act,
                "time": t0 + pd.Timedelta(days=day),
                "amount": amount,
                "flow_type": "DF2",
                "company": company,
                "vendor": vendor,
            }
        )

    for case, company, vendor in [("A", "A", "V1"), ("B", "A", "V2"), ("C", "B", "V1"), ("D", "B", "V1"), ("E", "B", "V2")]:
        ev(case, PO, -1, None, company=company, vendor=vendor)
    # σA: lag GR→INV 25 days, one GR, amounts match, no cancellation
    ev("A", GR, 0, 100)
    ev("A", INV, 25, 100)
    ev("A", CLR, 30)
    # σB: lag 8 days, four goods receipts (fragmentation), amounts match
    for d in range(4):
        ev("B", GR, d, 25, vendor="V2")
    ev("B", INV, 8, 100, vendor="V2")
    ev("B", CLR, 12, vendor="V2")
    # σC: lag 7 days, one GR, invoiced 82 vs received 100 → mismatch 0.18
    ev("C", GR, 0, 100, company="B")
    ev("C", INV, 7, 82, company="B")
    ev("C", CLR, 9, company="B")
    # σD: lag 5 days, one GR, amounts match, invoice cancelled
    ev("D", GR, 0, 100, company="B")
    ev("D", INV, 5, 100, company="B")
    ev("D", CINV, 6, company="B")
    # σE: invoice missing → lag undefined, mismatch 1
    ev("E", GR, 0, 100, company="B", vendor="V2")
    return pd.DataFrame(rows)


def running_p2p_log() -> EventLog:
    return EventLog(
        running_p2p_events(),
        case_col="case",
        activity_col="activity",
        timestamp_col="time",
        case_attributes=["flow_type", "company", "vendor"],
    )
