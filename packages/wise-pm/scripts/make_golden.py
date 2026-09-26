"""Golden dataset and pipeline snapshot for wise-pm.

The golden tier of the test suite (``tests/golden/test_pipeline_golden.py``)
runs the full pipeline on a committed synthetic purchase-to-pay log and
compares every output with the files written by this script. Regenerate
only on purpose and review the diff::

    python scripts/make_golden.py --write

Every golden frame is written as a parquet file at full precision: that is
the artefact of record. The summary frames (backlogs, drivers, penalty mass,
view agreement, hotspots, validation table) additionally get a human-readable
CSV rounded to ``CSV_DECIMALS`` decimals with LF line endings, so that
reviewers can read the diff; the manifest lists both files per frame
(``record`` and ``readable``). The golden test compares the recomputed frame
with the parquet record exactly (``assert_frame_equal(check_exact=True)``, no
tolerance) and each CSV with the record rounded to ``CSV_DECIMALS``.

Exact-parity policy: unchanged classic arithmetic must stay bit-identical with
the record under the pinned environments (verified across Python 3.13 / 3.10
and the floor pins of numpy and pandas); only a new numerical method may
declare a tolerance, next to its own golden. Everything is derived from
``numpy.random.default_rng(SEED)``; nothing here reads the clock.

``--check`` proves determinism: the script runs twice into temporary
directories, every file (parquet records included) must be byte-identical
and every frame equal, and the committed files are compared with the fresh
run (exit status 1 when ``--write`` would change them: bytes first, then the
golden test's own exact comparison for records whose bytes differ, as when
the parquet metadata names other library versions; stale committed files
that the generator no longer produces are reported too).

Planted ground truth (``tests/data/p2p_synthetic_2k.ground_truth.json``):

* company ``C03`` has systematically longer GR -> INV lags and the largest
  volume (a *reservoir* hotspot);
* vendor ``V017`` (only in company ``C07``, half of its items) has a high
  invoice cancellation rate, frequent amount mismatches and slow clearing
  under payment blocks, at small volume (a *severity* hotspot, at vendor and
  at company level);
* company ``C05`` processes receipts and invoices mostly by hand (an
  *effort* mechanism);
* 60 right-censored cases (3 %): GR in the last 20 days of the window and no
  clearing; 40 of them have no invoice yet ("open receipts"), 20 have an
  invoice after the GR ("open invoices", the ones ``right_censored`` with
  ``opened_by=INV`` must flag);
* 40 cases (2 %) whose every event is extracted twice (event replication);
* about 1 % of events with a missing timestamp (kept via
  ``missing_timestamps="keep"``);
* two cases whose post-invoice events carry the placeholder date 2099-12-31,
  so that the robust observation window matters when no explicit window is
  given (the pipeline sets one).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

import wise
from wise.datasets import CINV, CLR, GR, INV, PO

# --------------------------------------------------------------------------- constants
SEED = 20260926
N_CASES = 2000
T0 = pd.Timestamp("2024-01-01")
HORIZON_DAYS = 365.0  # 2024-01-01 .. 2024-12-31 00:00 (leap year)
WINDOW = ("2024-01-01", "2024-12-31")
PLACEHOLDER_TS = pd.Timestamp("2099-12-31")

CHANGE_PRICE = "Change Price"
REMOVE_BLOCK = "Remove Payment Block"
ACTIVITIES = (PO, GR, INV, CLR, CINV, CHANGE_PRICE, REMOVE_BLOCK)

COMPANIES = [f"C{i:02d}" for i in range(1, 9)]
COMPANY_SHARE = np.array([0.12, 0.10, 0.28, 0.11, 0.09, 0.10, 0.05, 0.15])
VENDORS = [f"V{i:03d}" for i in range(1, 61)]
RESERVOIR_COMPANY = "C03"
MANUAL_COMPANY = "C05"
SEVERITY_COMPANY = "C07"
SEVERITY_VENDOR = "V017"

N_OPEN_RECEIPTS = 40
N_OPEN_INVOICES = 20
N_REPLICATED = 40
N_PLACEHOLDER = 2
N_NO_INVOICE = 30
MISSING_TS_SHARE = 0.01
CENSOR_LAST_DAYS = 20.0

RATES: dict[str, float] = {
    "flow_type_df1": 0.75,
    "df2_invoice_first": 0.30,
    "df1_invoice_first": 0.04,
    "gr_batch_tie": 0.50,
    "split_invoice": 0.35,
    "invoice_mismatch": 0.15,
    "invoice_mismatch_severity_vendor": 0.50,
    "cancellation": 0.04,
    "cancellation_severity_vendor": 0.60,
    "reinvoice_after_cancellation": 0.70,
    "price_change": 0.20,
    "price_change_after_gr": 0.35,
    "payment_block": 0.35,
    "payment_block_severity_vendor": 0.80,
    "payment_block_tied_to_invoice": 0.40,
    "severity_vendor_share_in_company": 0.50,
    "user_gr": 0.45,
    "user_inv": 0.40,
    "user_clr": 0.10,
    "user_gr_manual_company": 0.90,
    "user_inv_manual_company": 0.85,
}
MISMATCH_SIGMA = 0.15  # log-normal spread of the invoiced/received ratio when amounts disagree
MISMATCH_SIGMA_SEVERITY_VENDOR = 0.30
LAG_MEDIANS_DAYS: dict[str, float] = {
    "po_to_first_gr": 10.0,
    "gr_to_inv": 8.0,
    "gr_to_inv_reservoir_company": 25.0,
    "inv_to_clr": 20.0,
    "inv_to_clr_severity_vendor": 60.0,
    "inv_to_cancellation": 2.0,
    "cancellation_to_reinvoice": 3.0,
}
GR_COUNTS = np.array([1, 2, 3, 4, 5, 6])
GR_COUNT_P = np.array([0.42, 0.26, 0.15, 0.09, 0.05, 0.03])

# --------------------------------------------------------------------------- pipeline parameters
LOG_KWARGS: dict[str, Any] = {
    "case_col": "case",
    "activity_col": "activity",
    "timestamp_col": "time",
    "case_attributes": ["company", "vendor", "flow_type"],
    "order_col": "eventID",
    "event_id_col": "eventID",
    "missing_timestamps": "keep",
    "window": WINDOW,
}
VIEW = "Finance"
VIEW_NAMES = ("Finance", "Operations", "Compliance")
MODES = ("flat", "layer_balanced")
GAMMA = 20.0
TOP_K = 20
CORRELATION: Final[Literal["pearson", "spearman", "kendall"]] = "pearson"
AGREEMENT_BY = "vendor"
CENSORING: dict[str, Any] = {"closure": CLR, "opened_by": INV, "window": "60D"}
VALIDATION_MODE = "layer_balanced"
CSV_DECIMALS = 12  # rounding of the human-readable CSV companions; the parquet records keep full precision

DATASET_FILE = "p2p_synthetic_2k.parquet"
GROUND_TRUTH_FILE = "p2p_synthetic_2k.ground_truth.json"
GOLDEN_SUBDIR = "golden"
NORM_FILE = "norm.json"
MANIFEST_FILE = "manifest.json"
QUALITY_FILE = "quality_report.json"
RECORD_FORMAT = "parquet"  # every golden frame: the artefact of record, compared exactly
READABLE_FORMAT = "csv"  # summary frames only: rounded to CSV_DECIMALS so that reviewers can read the diff

MODE_OUTPUTS: tuple[str, ...] = (
    "violations",
    "in_scope",
    "scores",
    *(f"contributions_{v}" for v in VIEW_NAMES),
    "backlog_company",
    "backlog_company_vendor",
    "layer_drivers_company",
    "constraint_drivers_all",
    "penalty_mass_vendor",
    "view_agreement",
    "hotspot_company",
)
COMMON_OUTPUTS: tuple[str, ...] = ("event_replication", "right_censored", "validation_table_company")
READABLE_OUTPUTS: frozenset[str] = frozenset(
    {
        "backlog_company",
        "backlog_company_vendor",
        "layer_drivers_company",
        "constraint_drivers_all",
        "penalty_mass_vendor",
        "view_agreement",
        "hotspot_company",
        "validation_table_company",
    }
)
assert READABLE_OUTPUTS.issubset({*MODE_OUTPUTS, *COMMON_OUTPUTS})

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "tests" / "data"


# --------------------------------------------------------------------------- norm
def golden_norm() -> wise.Norm:
    """The golden norm: six layers, eleven constraints, three views."""
    layers = (
        wise.Layer("completeness", "Core completeness"),
        wise.Layer("lead_times", "Working-capital lead times"),
        wise.Layer("match", "Match consistency"),
        wise.Layer("handling", "Handling discipline"),
        wise.Layer("exceptions", "Exception discipline"),
        wise.Layer("effort", "Manual effort"),
    )
    constraints = (
        wise.NormConstraint("inv_present", "completeness", wise.Presence(INV), description="Require an invoice receipt"),
        wise.NormConstraint(
            "clr_present",
            "completeness",
            wise.Presence(CLR),
            applicability={"flow_type": ["DF1"]},
            description="Three-way-match items must be cleared",
        ),
        wise.NormConstraint(
            "gr_inv_lag",
            "lead_times",
            wise.Lag(GR, INV, delta=10, width=20),
            description="First GR to first INV within 10 days",
        ),
        wise.NormConstraint(
            "gr_inv_lag_each",
            "lead_times",
            wise.Lag(GR, INV, delta=10, width=20, activation="each"),
            weight=0.5,
            description="Every GR followed by an INV within 10 days",
        ),
        wise.NormConstraint(
            "inv_clr_lag",
            "lead_times",
            wise.Lag(INV, CLR, delta=30, width=30, missing_b="censor"),
            description="INV to CLR within 30 days; open invoices are scored by their age",
        ),
        wise.NormConstraint(
            "inv_gr_balance",
            "match",
            wise.Balance("amount", INV, "amount", GR, tau=0.05, width=0.2),
            description="Invoiced vs received amount",
        ),
        wise.NormConstraint(
            "gr_before_inv",
            "match",
            wise.Precedence(GR, INV),
            applicability={"flow_type": ["DF1"]},
            description="No invoice before the first GR in a three-way match",
        ),
        wise.NormConstraint(
            "gr_singular", "handling", wise.Singularity(GR, k=2, K=3), description="Limit delivery fragmentation"
        ),
        wise.NormConstraint("no_cancel", "exceptions", wise.Exclusion(CINV), description="Avoid invoice cancellations"),
        wise.NormConstraint(
            "no_price_change_after_gr",
            "exceptions",
            wise.Exclusion(CHANGE_PRICE, after=GR),
            description="No price change after the goods receipt",
        ),
        wise.NormConstraint(
            "manual_effort",
            "effort",
            wise.Metric("manual_touches", threshold=3, width=3),
            description="At most three manual touches per item",
        ),
    )
    views = (
        wise.View(
            "Finance",
            layer_weights={
                "completeness": 0.25,
                "lead_times": 0.30,
                "match": 0.20,
                "handling": 0.05,
                "exceptions": 0.15,
                "effort": 0.05,
            },
        ),
        wise.View(
            "Operations",
            layer_weights={
                "completeness": 0.15,
                "lead_times": 0.15,
                "match": 0.10,
                "handling": 0.30,
                "exceptions": 0.10,
                "effort": 0.20,
            },
        ),
        wise.View(
            "Compliance",
            constraint_weights={
                "inv_present": 0.10,
                "clr_present": 0.10,
                "gr_inv_lag": 0.05,
                "gr_inv_lag_each": 0.05,
                "inv_clr_lag": 0.05,
                "inv_gr_balance": 0.20,
                "gr_before_inv": 0.20,
                "gr_singular": 0.05,
                "no_cancel": 0.10,
                "no_price_change_after_gr": 0.10,
            },
            description="Constraint weights; manual effort deliberately carries no weight",
        ),
    )
    derived = ({"name": "manual_touches", "kind": "count_events", "where": {"column": "org:resource", "regex": "^user"}},)
    return wise.Norm(
        constraints,
        layers,
        views,
        name="Golden P2P norm",
        version="1",
        description="Norm of the golden pipeline snapshot on the synthetic 2k-case purchase-to-pay log",
        scoring_mode="layer_balanced",
        derived_attributes=derived,
        metadata={"dataset": DATASET_FILE, "seed": SEED},
    )


# --------------------------------------------------------------------------- generator
def _logn(rng: np.random.Generator, median: float, sigma: float, lo: float, hi: float) -> float:
    return float(np.clip(rng.lognormal(math.log(median), sigma), lo, hi))


def _split_amount(rng: np.random.Generator, total: float, n: int) -> list[float]:
    """Split ``total`` into ``n`` non-negative parts rounded to cents that sum exactly to ``total``."""
    if n == 1:
        return [total]
    parts = np.round(rng.dirichlet(np.full(n, 2.0)) * total, 2)
    parts[-1] = round(total - float(parts[:-1].sum()), 2)
    if parts[-1] < 0:
        parts[0] = round(parts[0] + parts[-1], 2)
        parts[-1] = 0.0
    return [float(p) for p in parts]


class _Generator:
    """Builds the synthetic log case by case from one random generator."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        others = [v for v in VENDORS if v != SEVERITY_VENDOR]
        weights = 1.0 / np.arange(1, len(others) + 1) ** 0.8
        self.other_vendors = others
        self.vendor_p = rng.permutation(weights / weights.sum())
        self.rows: list[dict[str, Any]] = []
        self.cases: list[dict[str, Any]] = []

    # -- helpers ---------------------------------------------------------------
    def _user(self) -> str:
        return f"user_{int(self.rng.integers(1, 41)):02d}"

    def _batch(self) -> str:
        return f"batch_{int(self.rng.integers(1, 6)):02d}"

    def _pick(self, p_user: float) -> str:
        return self._user() if self.rng.random() < p_user else self._batch()

    # -- one case --------------------------------------------------------------
    def case(self, cid: str, kind: str) -> None:
        rng = self.rng
        company = COMPANIES[int(rng.choice(len(COMPANIES), p=COMPANY_SHARE))]
        if company == SEVERITY_COMPANY and rng.random() < RATES["severity_vendor_share_in_company"]:
            vendor = SEVERITY_VENDOR
        else:
            vendor = self.other_vendors[int(rng.choice(len(self.other_vendors), p=self.vendor_p))]
        flow = "DF1" if rng.random() < RATES["flow_type_df1"] else "DF2"
        manual = company == MANUAL_COMPANY
        open_kind = kind in ("open_receipt", "open_invoice")
        events: list[tuple[str, float, float, str]] = []  # activity, days since PO, amount, resource

        po_amount = round(_logn(rng, 3000.0, 1.0, 20.0, 500000.0), 2)
        events.append((PO, 0.0, po_amount, self._user()))

        n_gr = int(rng.choice(GR_COUNTS, p=GR_COUNT_P))
        first_gr = _logn(rng, LAG_MEDIANS_DAYS["po_to_first_gr"], 0.5, 0.5, 90.0)
        gr_tied = n_gr > 1 and rng.random() < RATES["gr_batch_tie"]
        if n_gr == 1:
            gr_times = [first_gr]
        elif gr_tied:
            gr_times = [first_gr] * n_gr
        else:
            gaps = rng.exponential(3.0, n_gr - 1) + 0.25
            gr_times = [first_gr, *(first_gr + np.cumsum(gaps)).tolist()]
        parts = _split_amount(rng, po_amount, n_gr)
        p_user_gr = RATES["user_gr_manual_company"] if manual else RATES["user_gr"]
        shared_resource = self._pick(p_user_gr) if gr_tied else None
        for t, amount in zip(gr_times, parts):
            events.append((GR, t, amount, shared_resource or self._pick(p_user_gr)))
        last_gr = gr_times[-1]

        n_price_after_gr = 0
        if rng.random() < RATES["price_change"]:
            for _ in range(int(rng.choice([1, 2, 3], p=[0.7, 0.2, 0.1]))):
                if not open_kind and rng.random() < RATES["price_change_after_gr"]:
                    t = first_gr + rng.uniform(0.5, 10.0)
                    n_price_after_gr += 1
                else:
                    t = rng.uniform(0.05, 0.95) * first_gr
                events.append((CHANGE_PRICE, t, math.nan, self._user()))

        info: dict[str, Any] = {
            "case": cid,
            "kind": kind,
            "company": company,
            "vendor": vendor,
            "flow_type": flow,
            "n_gr": n_gr,
            "gr_tied": gr_tied,
            "n_price_change_after_gr": n_price_after_gr,
            "invoice_first": False,
            "split_invoice": False,
            "mismatch": False,
            "cancelled": False,
            "reinvoiced": False,
            "gr_to_inv_days": math.nan,
        }
        p_user_inv = RATES["user_inv_manual_company"] if manual else RATES["user_inv"]

        if kind in ("open_receipt", "no_invoice"):
            self._finish(cid, kind, events, info, anchor=last_gr)
            return
        if kind == "open_invoice":
            t_inv = last_gr + rng.uniform(0.25, 6.0)
            events.append((INV, t_inv, po_amount, self._pick(p_user_inv)))
            info["gr_to_inv_days"] = t_inv - first_gr
            self._finish(cid, kind, events, info, anchor=last_gr)
            return

        p_first = RATES["df2_invoice_first"] if flow == "DF2" else RATES["df1_invoice_first"]
        invoice_first = rng.random() < p_first
        median = LAG_MEDIANS_DAYS["gr_to_inv_reservoir_company" if company == RESERVOIR_COMPANY else "gr_to_inv"]
        split = n_gr >= 2 and not invoice_first and rng.random() < RATES["split_invoice"]
        invoices: list[tuple[float, float]]
        if invoice_first:
            invoices = [(rng.uniform(0.3, 0.9) * first_gr, po_amount)]
        elif split:
            half = math.ceil(n_gr / 2)
            t1 = gr_times[half - 1] + _logn(rng, median, 0.55, 0.25, 120.0)
            t2 = last_gr + _logn(rng, median, 0.55, 0.25, 120.0)
            invoices = sorted([(t1, round(sum(parts[:half]), 2)), (t2, round(sum(parts[half:]), 2))])
        else:
            invoices = [(last_gr + _logn(rng, median, 0.55, 0.25, 120.0), po_amount)]
        severe = vendor == SEVERITY_VENDOR
        p_mismatch = RATES["invoice_mismatch_severity_vendor"] if severe else RATES["invoice_mismatch"]
        if rng.random() < p_mismatch:
            sigma = MISMATCH_SIGMA_SEVERITY_VENDOR if severe else MISMATCH_SIGMA
            factor = float(np.clip(math.exp(rng.normal(0.0, sigma)), 0.5, 1.6))
            t0, a0 = invoices[0]
            invoices[0] = (t0, round(a0 * factor, 2))
            info["mismatch"] = True
        info["invoice_first"] = invoice_first
        info["split_invoice"] = split
        for t, amount in invoices:
            events.append((INV, t, amount, self._pick(p_user_inv)))
        inv_after_gr = [t for t, _ in invoices if t >= first_gr]
        info["gr_to_inv_days"] = (min(inv_after_gr) - first_gr) if inv_after_gr else math.nan

        p_cancel = RATES["cancellation_severity_vendor"] if severe else RATES["cancellation"]
        to_clear = list(invoices)
        if kind != "placeholder" and rng.random() < p_cancel:
            info["cancelled"] = True
            t_inv0, a0 = invoices[0]
            t_cinv = t_inv0 + _logn(rng, LAG_MEDIANS_DAYS["inv_to_cancellation"], 0.5, 0.1, 30.0)
            events.append((CINV, t_cinv, math.nan, self._user()))
            if rng.random() < RATES["reinvoice_after_cancellation"]:
                info["reinvoiced"] = True
                t_re = t_cinv + _logn(rng, LAG_MEDIANS_DAYS["cancellation_to_reinvoice"], 0.5, 0.2, 30.0)
                events.append((INV, t_re, a0, self._pick(p_user_inv)))
                to_clear = [(t_re, a0), *invoices[1:]]
            else:
                to_clear = invoices[1:]

        clear_times: list[float] = []
        clr_median = LAG_MEDIANS_DAYS["inv_to_clr_severity_vendor" if severe else "inv_to_clr"]
        for t_i, _ in to_clear:
            base = max(t_i, last_gr) if invoice_first else t_i
            t_c = base + _logn(rng, clr_median, 0.5, 0.5, 120.0)
            events.append((CLR, t_c, math.nan, self._pick(RATES["user_clr"])))
            clear_times.append(t_c)
        p_block = RATES["payment_block_severity_vendor"] if severe else RATES["payment_block"]
        if to_clear and rng.random() < p_block:
            t_i, t_c = to_clear[0][0], clear_times[0]
            t_b = t_i if rng.random() < RATES["payment_block_tied_to_invoice"] else t_i + rng.uniform(0.1, 0.9) * (t_c - t_i)
            events.append((REMOVE_BLOCK, t_b, math.nan, self._user()))
        self._finish(cid, kind, events, info, anchor=None)

    def _finish(
        self,
        cid: str,
        kind: str,
        events: list[tuple[str, float, float, str]],
        info: dict[str, Any],
        *,
        anchor: float | None,
    ) -> None:
        """Place the case in the year and emit its rows."""
        rng = self.rng
        span = max(t for _, t, _, _ in events)
        if kind == "open_receipt":
            start = rng.uniform(HORIZON_DAYS - CENSOR_LAST_DAYS, HORIZON_DAYS - 0.1) - float(anchor or 0.0)
        elif kind == "open_invoice":
            start = rng.uniform(HORIZON_DAYS - CENSOR_LAST_DAYS, HORIZON_DAYS - 8.0) - float(anchor or 0.0)
        else:
            if span >= HORIZON_DAYS - 1.0:
                raise RuntimeError(f"case {cid} spans {span:.1f} days; loosen the clips")
            start = rng.uniform(0.0, HORIZON_DAYS - span - 0.01)
        if start < 0.0:
            raise RuntimeError(f"case {cid} would start before the window")
        t_inv_first = min((t for a, t, _, _ in events if a == INV), default=math.inf)
        for seq, (activity, t, amount, resource) in enumerate(sorted(events, key=lambda e: (e[1], e[0]))):
            self.rows.append(
                {
                    "case": cid,
                    "seq": seq,
                    "activity": activity,
                    "day": start + t,
                    "amount": amount,
                    "org:resource": resource,
                    "company": info["company"],
                    "vendor": info["vendor"],
                    "flow_type": info["flow_type"],
                    "placeholder": kind == "placeholder" and t >= t_inv_first,
                }
            )
        info["start_day"] = start
        info["n_events"] = len(events)
        self.cases.append(info)


def _assign_kinds(rng: np.random.Generator) -> list[str]:
    kinds = ["normal"] * N_CASES
    perm = rng.permutation(N_CASES)
    pos = 0
    for kind, n in (
        ("open_receipt", N_OPEN_RECEIPTS),
        ("open_invoice", N_OPEN_INVOICES),
        ("replicated", N_REPLICATED),
        ("placeholder", N_PLACEHOLDER),
        ("no_invoice", N_NO_INVOICE),
    ):
        for j in perm[pos : pos + n]:
            kinds[int(j)] = kind
        pos += n
    return kinds


def generate(seed: int = SEED) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The synthetic event log (one row per event, shuffled) and its ground truth."""
    rng = np.random.default_rng(seed)
    kinds = _assign_kinds(rng)
    gen = _Generator(rng)
    for i, kind in enumerate(kinds, start=1):
        gen.case(f"POI{i:05d}", kind)

    ev = pd.DataFrame(gen.rows)
    ev["time"] = (T0 + pd.to_timedelta(ev["day"], unit="D")).dt.floor("s").astype("datetime64[ns]")
    ev = ev.sort_values(["time", "case", "seq"], kind="mergesort").reset_index(drop=True)
    ev["eventID"] = np.arange(1, len(ev) + 1, dtype=np.int64)

    cases = pd.DataFrame(gen.cases).set_index("case")
    kind_of = cases["kind"]
    replicated = ev["case"].map(kind_of).eq("replicated").to_numpy()
    ev = pd.concat([ev, ev[replicated]], ignore_index=True)

    placeholder_mask = ev["placeholder"].to_numpy()
    ev.loc[placeholder_mask, "time"] = PLACEHOLDER_TS

    normal_rows = np.flatnonzero(ev["case"].map(kind_of).eq("normal").to_numpy())
    n_missing = round(MISSING_TS_SHARE * len(ev))
    missing_rows = np.sort(rng.choice(normal_rows, size=n_missing, replace=False))
    ev.loc[missing_rows, "time"] = pd.NaT

    # Event ids of the planted defects, captured *before* the shuffle: ``missing_rows`` and
    # ``placeholder_mask`` are positions in the current frame and mean nothing after it.
    missing_ids = sorted(int(x) for x in ev.loc[missing_rows, "eventID"])
    placeholder_ids = sorted(int(x) for x in ev.loc[placeholder_mask, "eventID"].unique())

    ev = ev.iloc[rng.permutation(len(ev))].reset_index(drop=True)
    events = ev[["eventID", "case", "activity", "time", "amount", "org:resource", "company", "vendor", "flow_type"]].copy()
    events["time"] = events["time"].astype("datetime64[ns]")

    truth = _ground_truth(events, cases, missing_ids, placeholder_ids)
    return events, truth


def _ground_truth(
    events: pd.DataFrame,
    cases: pd.DataFrame,
    missing_ids: list[int],
    placeholder_ids: list[int],
) -> dict[str, Any]:
    assert events.loc[events["time"].isna(), "eventID"].sort_values().tolist() == missing_ids
    assert events.loc[events["time"] >= PLACEHOLDER_TS, "eventID"].sort_values().tolist() == placeholder_ids

    def ids(kind: str) -> list[str]:
        return sorted(cases.index[cases["kind"] == kind].tolist())

    normal = cases[cases["kind"].isin(["normal", "replicated", "placeholder"])]
    lag = normal["gr_to_inv_days"]
    by_company = normal.groupby("company")
    sev = cases[cases["vendor"] == SEVERITY_VENDOR]
    other = cases[(cases["vendor"] != SEVERITY_VENDOR) & cases["kind"].isin(["normal", "replicated"])]
    manual_touch = events["org:resource"].str.startswith("user").groupby(events["case"]).sum().groupby(cases["company"]).mean()
    return {
        "seed": SEED,
        "generator": "scripts/make_golden.py",
        "n_cases": len(cases),
        "n_events": len(events),
        "window": list(WINDOW),
        "activities": list(ACTIVITIES),
        "companies": COMPANIES,
        "company_share": COMPANY_SHARE.tolist(),
        "n_vendors": len(VENDORS),
        "rates": RATES,
        "mismatch_sigma": {"default": MISMATCH_SIGMA, "severity_vendor": MISMATCH_SIGMA_SEVERITY_VENDOR},
        "lag_medians_days": LAG_MEDIANS_DAYS,
        "gr_count_distribution": dict(zip((str(k) for k in GR_COUNTS.tolist()), GR_COUNT_P.tolist())),
        "planted": {
            "reservoir_company": {
                "id": RESERVOIR_COMPANY,
                "mechanism": "GR -> INV lag median 25 days instead of 8, largest volume",
                "n_cases": int((cases["company"] == RESERVOIR_COMPANY).sum()),
                "realised_gr_to_inv_median_days": float(lag[normal["company"] == RESERVOIR_COMPANY].median()),
                "realised_gr_to_inv_median_days_other": float(lag[normal["company"] != RESERVOIR_COMPANY].median()),
                "realised_gr_to_inv_median_days_by_company": {
                    k: float(v) for k, v in by_company["gr_to_inv_days"].median().items()
                },
            },
            "severity_vendor": {
                "id": SEVERITY_VENDOR,
                "company": SEVERITY_COMPANY,
                "mechanism": (
                    "cancellation rate 0.60 instead of 0.04, mismatch rate 0.50 instead of 0.15, "
                    "INV -> CLR median 60 days instead of 20 under payment blocks; small volume"
                ),
                "n_cases": len(sev),
                "n_cancelled": int(sev["cancelled"].sum()),
                "n_mismatch": int(sev["mismatch"].sum()),
                "realised_cancellation_rate": float(sev["cancelled"].mean()),
                "realised_cancellation_rate_other_vendors": float(other["cancelled"].mean()),
                "realised_mismatch_rate": float(sev["mismatch"].mean()),
                "realised_mismatch_rate_other_vendors": float(other["mismatch"].mean()),
            },
            "manual_company": {
                "id": MANUAL_COMPANY,
                "mechanism": "receipts and invoices posted by user_* resources instead of batch_*",
                "realised_mean_manual_touches_by_company": {k: float(v) for k, v in manual_touch.items()},
            },
            "right_censored": {
                "definition": "last GR in the last 20 days of the window, no clearing",
                "open_receipts": ids("open_receipt"),
                "open_invoices": ids("open_invoice"),
            },
            "replicated_cases": ids("replicated"),
            "no_invoice_cases": ids("no_invoice"),
            "placeholder_timestamp": {
                "value": PLACEHOLDER_TS.isoformat(),
                "cases": ids("placeholder"),
                "event_ids": placeholder_ids,
            },
            "missing_timestamp_event_ids": missing_ids,
        },
        "realised": {
            "events_per_activity": {a: int((events["activity"] == a).sum()) for a in ACTIVITIES},
            "cases_per_company": {k: int(v) for k, v in cases["company"].value_counts().sort_index().items()},
            "cases_per_flow_type": {k: int(v) for k, v in cases["flow_type"].value_counts().sort_index().items()},
            "n_cancelled": int(cases["cancelled"].sum()),
            "n_reinvoiced": int(cases["reinvoiced"].sum()),
            "n_mismatch": int(cases["mismatch"].sum()),
            "n_invoice_first_df1": int((cases["invoice_first"] & (cases["flow_type"] == "DF1")).sum()),
            "n_invoice_first_df2": int((cases["invoice_first"] & (cases["flow_type"] == "DF2")).sum()),
            "n_split_invoice": int(cases["split_invoice"].sum()),
            "n_cases_with_price_change_after_gr": int((cases["n_price_change_after_gr"] > 0).sum()),
            "n_gr_tied_cases": int(cases["gr_tied"].sum()),
            "n_missing_timestamps": len(missing_ids),
            "n_placeholder_events": len(placeholder_ids),
        },
    }


# --------------------------------------------------------------------------- pipeline
def make_event_log(events: pd.DataFrame) -> wise.EventLog:
    """The EventLog exactly as the golden test builds it."""
    return wise.EventLog(events, **LOG_KWARGS)


def load_event_log(path: Path) -> wise.EventLog:
    return make_event_log(pd.read_parquet(path))


def mode_outputs(result: wise.ScoreResult) -> dict[str, pd.DataFrame]:
    """Every mode-dependent golden frame, keyed by output name."""
    backlog_company = wise.prioritize(result, "company", view=VIEW, gamma=GAMMA)
    drivers = wise.layer_drivers(result, "company", view=VIEW)
    out: dict[str, pd.DataFrame] = {
        "violations": result.violations,
        "in_scope": result.in_scope,
        "scores": result.scores,
        **{f"contributions_{v}": result.contributions[v] for v in VIEW_NAMES},
        "backlog_company": backlog_company,
        "backlog_company_vendor": wise.prioritize(result, ["company", "vendor"], view=VIEW, gamma=GAMMA),
        "layer_drivers_company": drivers,
        "constraint_drivers_all": wise.constraint_drivers(result, VIEW),
        "penalty_mass_vendor": wise.penalty_mass(result, VIEW, "vendor"),
        "view_agreement": wise.view_agreement(result, AGREEMENT_BY, k=TOP_K, gamma=GAMMA, method=CORRELATION),
        "hotspot_company": wise.hotspot_table(backlog_company, drivers=drivers),
    }
    assert set(out) == set(MODE_OUTPUTS)
    return out


def common_outputs(log: wise.EventLog, result: wise.ScoreResult) -> dict[str, pd.DataFrame]:
    """Mode-independent diagnostics (``result`` is the ``VALIDATION_MODE`` result)."""
    censored = wise.right_censored(log, **CENSORING)
    replication = wise.event_replication(log)
    out = {
        "event_replication": replication,
        "right_censored": censored.to_frame(),
        "validation_table_company": wise.validation_table(
            result, VIEW, "company", censored=censored, replication=replication, gamma=GAMMA
        ),
    }
    assert set(out) == set(COMMON_OUTPUTS)
    return out


def quality_report(log: wise.EventLog) -> dict[str, Any]:
    """``log.validate()`` as a JSON-ready mapping."""
    return {str(k): _jsonable(v) for k, v in log.validate().items()}


# --------------------------------------------------------------------------- files
def golden_stem(name: str, mode: str | None) -> str:
    """File stem of a golden frame: ``<name>_<mode>`` for mode-dependent outputs, ``<name>`` otherwise."""
    return f"{name}_{mode}" if mode is not None else name


def record_file(name: str, mode: str | None) -> str:
    """The parquet record of a golden frame (every frame has one)."""
    return f"{golden_stem(name, mode)}.{RECORD_FORMAT}"


def readable_file(name: str, mode: str | None) -> str | None:
    """The human-readable CSV of a golden frame, or ``None`` for frames that have none."""
    return f"{golden_stem(name, mode)}.{READABLE_FORMAT}" if name in READABLE_OUTPUTS else None


def write_record(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, engine="pyarrow")


def write_readable(frame: pd.DataFrame, path: Path) -> None:
    frame.round(CSV_DECIMALS).to_csv(path, lineterminator="\n")


def read_record(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def read_readable(path: Path, like: pd.DataFrame) -> pd.DataFrame:
    """Read a CSV back; ``like`` (the record) supplies the index depth and string dtypes a CSV cannot carry."""
    df = pd.read_csv(path, index_col=list(range(like.index.nlevels)), float_precision="round_trip")
    strings = {c: like[c].dtype for c in df.columns if c in like.columns and not pd.api.types.is_numeric_dtype(like[c])}
    return df.astype(strings) if strings else df


def uniform_nulls(frame: pd.DataFrame) -> pd.DataFrame:
    """Represent every missing value of an object column as NaN.

    ``layer_drivers`` writes ``None`` into ``dominant_layer`` while a file round trip yields NaN; pandas 2.3 warns
    about comparing the two and ``filterwarnings = error`` would turn that warning into a failure.
    """
    out = frame.copy()
    for col in [c for c, dtype in out.dtypes.items() if pd.api.types.is_object_dtype(dtype)]:
        out[col] = out[col].where(out[col].notna(), np.nan)
    return out


def assert_frames_exact(actual: pd.DataFrame, expected: pd.DataFrame, *, obj: str) -> None:
    """The golden comparison, shared by the golden test and ``--check``: exact values, no tolerance.

    Index and column order, dtypes and every value must agree (``check_exact=True``, no ``rtol``/``atol``; NaN
    equals NaN). Unchanged classic arithmetic keeps bit-for-bit parity with the record; only a new numerical method
    may declare a tolerance, and it does so next to its own golden rather than here.
    """
    pd.testing.assert_frame_equal(uniform_nulls(actual), uniform_nulls(expected), check_exact=True, obj=obj)


REPORT_ATOL = 1e-9


def assert_frames_report(actual: pd.DataFrame, expected: pd.DataFrame, *, obj: str) -> str:
    """Portability comparison for environments that are not the record environment.

    Values must agree within ``REPORT_ATOL`` (structure exactly); the returned text summarises the largest
    absolute difference so a CI log shows environment-induced drift (BLAS, SIMD reduction order) without
    labelling it a semantic change. The record environment (see the golden manifest) uses ``assert_frames_exact``.
    """
    a, e = uniform_nulls(actual), uniform_nulls(expected)
    pd.testing.assert_frame_equal(a, e, check_exact=False, rtol=0, atol=REPORT_ATOL, obj=obj)
    worst = 0.0
    for col in a.columns:
        if pd.api.types.is_float_dtype(a[col]):
            diff = a[col].to_numpy(dtype=float) - e[col].to_numpy(dtype=float)
            diff = diff[np.isfinite(diff)]
            if diff.size:
                worst = max(worst, float(np.abs(diff).max()))
    return f"{obj}: max |actual - record| = {worst:.3e} (report mode, atol {REPORT_ATOL:g})"


def _jsonable(v: Any) -> Any:
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, np.integer | int) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, np.floating | float):
        return round(float(v), CSV_DECIMALS)
    if isinstance(v, np.bool_ | bool):
        return bool(v)
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_jsonable(x) for x in v]
    return v


def write_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(_jsonable(obj), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(base: Path) -> dict[str, pd.DataFrame]:
    """Generate the dataset and every golden file under ``base``; return the frames by relative path."""
    golden = base / GOLDEN_SUBDIR
    golden.mkdir(parents=True, exist_ok=True)
    events, truth = generate()
    events.to_parquet(base / DATASET_FILE, engine="pyarrow", index=False)
    write_json(truth, base / GROUND_TRUTH_FILE)

    norm = golden_norm()
    norm.dump(golden / NORM_FILE)
    log = load_event_log(base / DATASET_FILE)
    assert norm.check(log) == [], norm.check(log)

    outputs: list[tuple[str, str | None, pd.DataFrame]] = []  # name, mode, frame
    results = {mode: wise.score(log, norm, mode=mode) for mode in MODES}
    for mode, result in results.items():
        result.check_decomposition()
        outputs.extend((name, mode, frame) for name, frame in mode_outputs(result).items())
    outputs.extend((name, None, frame) for name, frame in common_outputs(log, results[VALIDATION_MODE]).items())

    frames: dict[str, pd.DataFrame] = {}  # keyed by the record's path relative to ``base``
    listing: dict[str, dict[str, Any]] = {}
    for name, frame_mode, frame in outputs:
        record_rel = f"{GOLDEN_SUBDIR}/{record_file(name, frame_mode)}"
        readable = readable_file(name, frame_mode)
        readable_rel = f"{GOLDEN_SUBDIR}/{readable}" if readable else None
        write_record(frame, base / record_rel)
        if readable_rel:
            write_readable(frame, base / readable_rel)
        frames[record_rel] = frame
        listing[golden_stem(name, frame_mode)] = {
            "record": record_rel,
            "readable": readable_rel,
            "rows": len(frame),
            "columns": [str(c) for c in frame.columns],
        }
    write_json(quality_report(log), golden / QUALITY_FILE)

    manifest = {
        "dataset": {
            "file": DATASET_FILE,
            "sha256": sha256(base / DATASET_FILE),
            "n_events": len(events),
            "n_cases": len(log),
            "seed": SEED,
        },
        "norm": {
            "file": f"{GOLDEN_SUBDIR}/{NORM_FILE}",
            "fingerprint": norm.fingerprint(),
            "name": norm.name,
            "version": norm.version,
            "scoring_mode": norm.scoring_mode,
        },
        "log_kwargs": {k: list(v) if isinstance(v, tuple) else v for k, v in LOG_KWARGS.items()},
        "parameters": {
            "modes": list(MODES),
            "view": VIEW,
            "gamma": GAMMA,
            "top_k": TOP_K,
            "correlation": CORRELATION,
            "agreement_by": AGREEMENT_BY,
            "censoring": CENSORING,
            "validation_table_mode": VALIDATION_MODE,
            "record_format": RECORD_FORMAT,
            "readable_format": READABLE_FORMAT,
            "csv_float_decimals": CSV_DECIMALS,
        },
        "versions": {
            "wise": wise.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": _pyarrow_version(),
            "python": ".".join(str(x) for x in sys.version_info[:2]),
        },
        "frames": listing,
    }
    write_json(manifest, golden / MANIFEST_FILE)
    return frames


def _pyarrow_version() -> str:
    import pyarrow

    return str(pyarrow.__version__)


# --------------------------------------------------------------------------- CLI
def _files_under(base: Path) -> list[Path]:
    """Every regular file below ``base`` (relative paths, sorted); dotfiles such as ``.DS_Store`` are ignored."""
    return sorted(p.relative_to(base) for p in base.rglob("*") if p.is_file() and not p.name.startswith("."))


def _report(base: Path, frames: Mapping[str, pd.DataFrame]) -> None:
    size = (base / DATASET_FILE).stat().st_size
    print(f"dataset: {base / DATASET_FILE} ({size / 1024:.1f} KB)")
    files = _files_under(base)
    total = sum((base / rel).stat().st_size for rel in files)
    n_readable = sum(1 for rel in files if rel.suffix == f".{READABLE_FORMAT}")
    print(f"golden files: {len(frames)} parquet records, {n_readable} readable CSVs, {total / 1024:.1f} KB in total under {base}")


def _records_equal(fresh: Path, old: Path) -> bool:
    """Whether two parquet records hold the same frame under the golden test's exact comparison."""
    try:
        assert_frames_exact(read_record(fresh), read_record(old), obj=fresh.name)
    except AssertionError:
        return False
    return True


def _committed_changes(committed: Path, fresh: Path, frames: Mapping[str, pd.DataFrame]) -> list[str]:
    """Committed files that ``--write`` would change, and committed files the generator no longer produces.

    Files are compared byte for byte first. When the bytes differ, a parquet record counts as changed only if its
    frame differs under the golden test's exact comparison (the parquet metadata names the writing libraries, so
    another environment changes the bytes but not the frame), a JSON file only if its content differs (the
    manifest's recorded library versions are reported but tolerated), and anything else (a readable CSV, the
    dataset parquet whose sha256 the manifest pins) is changed as soon as its bytes are.
    """
    fresh_files = _files_under(fresh)
    changed: list[str] = []
    for path_rel in fresh_files:
        rel = str(path_rel)
        new_file, old_file = fresh / path_rel, committed / path_rel
        if not old_file.exists():
            changed.append(f"{rel} (missing)")
        elif new_file.read_bytes() == old_file.read_bytes():
            continue
        elif rel in frames:
            if not _records_equal(new_file, old_file):
                changed.append(rel)
        elif path_rel.suffix == ".json":
            new_obj, old_obj = read_json(new_file), read_json(old_file)
            if path_rel.name == MANIFEST_FILE and new_obj.pop("versions") != old_obj.pop("versions"):
                print(f"note: {rel} was written with other library versions")
            if new_obj != old_obj:
                changed.append(rel)
        else:
            changed.append(rel)
    stale = sorted(set(_files_under(committed)) - set(fresh_files))
    changed.extend(f"{path_rel} (stale: the generator does not produce it)" for path_rel in stale)
    return changed


def check(committed: Path) -> int:
    """Two runs must be byte-identical; report whether the committed files would change."""
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "run1", Path(tmp) / "run2"
        frames_a, frames_b = run(a), run(b)
        files_a, files_b = _files_under(a), _files_under(b)
        if files_a != files_b:
            raise AssertionError("the two runs produced different file sets")
        for path_rel in files_a:
            if (a / path_rel).read_bytes() != (b / path_rel).read_bytes():
                raise AssertionError(f"{path_rel} differs between two runs: generation is not deterministic")
        for rel, frame in frames_a.items():
            assert_frames_exact(frame, frames_b[rel], obj=rel)
            assert_frames_exact(read_record(a / rel), read_record(b / rel), obj=rel)
        print(f"determinism: {len(files_a)} files byte-identical and {len(frames_a)} frames exactly equal across two runs")
        _report(a, frames_a)

        if not (committed / GOLDEN_SUBDIR / MANIFEST_FILE).exists():
            print(f"no committed golden files under {committed}; run with --write first")
            return 0
        changed = _committed_changes(committed, a, frames_a)
        if changed:
            print("committed golden files that --write would change:")
            for line in changed:
                print(f"  {line}")
            return 1
        print(f"committed golden files under {committed} are up to date")
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="write the dataset and golden files")
    group.add_argument("--check", action="store_true", help="prove determinism and compare with the committed files")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"target directory (default {DEFAULT_OUT})")
    args = parser.parse_args(argv)
    if args.check:
        return check(args.out)
    frames = run(args.out)
    _report(args.out, frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
