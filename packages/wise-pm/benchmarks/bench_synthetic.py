"""End-to-end benchmark of wise-pm on a synthetic purchase-to-pay log.

The script builds a deterministic log of 250 000 cases / 1 600 000 events
(``numpy.random.default_rng(0)``) and times the classic pipeline step by
step: :class:`wise.EventLog` construction, :func:`wise.score` with a
10-constraint, 4-view ``layer_balanced`` norm, :func:`wise.prioritize`, the
driver tables, :func:`wise.view_agreement`, the observation diagnostics and
:func:`wise.validation_table`. Peak memory is taken from :mod:`tracemalloc`.

It prints one JSON object to stdout::

    {"build": 1.5, "score": 1.4, ..., "peak_gb": 0.46,
     "versions": {"python": ..., "numpy": ..., "pandas": ..., "wise": ...}}

Run ``python benchmarks/bench_synthetic.py`` for the full size and
``--scale 0.05`` for a 5 % log (12 500 cases / 80 000 events) as a quick
check. ``tests/perf/test_benchmarks.py`` runs it at ``--scale 1`` and asserts
the budgets in ``benchmarks/README.md`` when ``WISE_RUN_BENCH=1`` is set.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
import tracemalloc
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import wise

FULL_CASES = 250_000
FULL_EVENTS = 1_600_000
DEFAULT_SEED = 0
T0 = pd.Timestamp("2018-01-01")
HORIZON_DAYS = 400.0

N_COMPANIES = 20
N_VENDORS = 2_000
N_USERS = 50

# The running example's vocabulary (paper Table V), extended with two BPIC'19 labels.
PO = "Create Purchase Order Item"
GR = "Record Goods Receipt"
INV = "Record Invoice Receipt"
CLR = "Clear Invoice"
CINV = "Cancel Invoice Receipt"
CHP = "Change Price"
VINV = "Vendor creates invoice"
ACTIVITIES: tuple[str, ...] = (PO, GR, INV, CLR, CINV, CHP, VINV)

#: Every case gets this skeleton; the remaining events are drawn from ``EXTRA_P``.
SKELETON: tuple[str, ...] = (PO, GR, INV, CLR)
#: Sampling weights of the extra events, aligned with ``ACTIVITIES``
#: (fragmented receipts, duplicate invoices, cancellations, price changes, vendor invoices).
EXTRA_P: tuple[float, ...] = (0.05, 0.35, 0.10, 0.05, 0.10, 0.15, 0.20)

SLICE = ("company", "vendor")
VIEW = "Finance"
GAMMA = 20.0
TOP_K = 20

STEPS: tuple[str, ...] = ("build", "score", "prioritize", "drivers", "agreement", "diagnostics", "validation_table")


# ----------------------------------------------------------------------------- data
def synthetic_events(n_cases: int, n_events: int, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """A shuffled event table with ``n_cases`` integer case ids and exactly ``n_events`` rows.

    Columns: ``case`` (int), ``activity``, ``time`` (naive ns timestamps in
    ``[2018-01-01, 2018-01-01 + 400 d)``), ``amount`` (uniform, 2 decimals),
    ``company`` (20 values), ``vendor`` (2 000 values), ``flow_type``
    (``DF1``/``DF2``, constant per case) and ``resource`` (``user_NN`` or
    ``batch_NN``, the input of the ``count_events`` recipe).
    """
    if n_cases < 1:
        raise ValueError("n_cases must be >= 1")
    if n_events < n_cases:
        raise ValueError("n_events must be >= n_cases so that every case has an event")
    rng = np.random.default_rng(seed)
    act_index = {a: i for i, a in enumerate(ACTIVITIES)}

    per_case = min(len(SKELETON), n_events // n_cases)
    n_extra = n_events - per_case * n_cases
    case = np.concatenate([np.repeat(np.arange(n_cases), per_case), rng.integers(0, n_cases, n_extra)])
    skeleton_codes = np.array([act_index[a] for a in SKELETON[:per_case]], dtype=np.int64)
    act = np.concatenate([np.tile(skeleton_codes, n_cases), rng.choice(len(ACTIVITIES), n_extra, p=EXTRA_P)])

    company_of_case = rng.integers(0, N_COMPANIES, n_cases)
    vendor_of_case = rng.integers(0, N_VENDORS, n_cases)
    flow_of_case = rng.integers(0, 2, n_cases)
    is_user = rng.random(n_events) < 0.35
    resource_id = rng.integers(0, N_USERS, n_events)

    companies = np.array([f"C{i:02d}" for i in range(N_COMPANIES)])
    vendors = np.array([f"V{i:04d}" for i in range(N_VENDORS)])
    flows = np.array(["DF1", "DF2"])
    users = np.array([f"user_{i:02d}" for i in range(N_USERS)])
    batches = np.array([f"batch_{i:02d}" for i in range(N_USERS)])

    order = rng.permutation(n_events)
    frame = pd.DataFrame(
        {
            "case": case[order],
            "activity": np.asarray(ACTIVITIES, dtype=object)[act[order]],
            "time": T0 + pd.to_timedelta(rng.uniform(0.0, HORIZON_DAYS, n_events), unit="D"),
            "amount": np.round(rng.uniform(10.0, 10_000.0, n_events), 2),
            "company": companies[company_of_case[case[order]]],
            "vendor": vendors[vendor_of_case[case[order]]],
            "flow_type": flows[flow_of_case[case[order]]],
            "resource": np.where(is_user[order], users[resource_id[order]], batches[resource_id[order]]),
        }
    )
    return frame


# ----------------------------------------------------------------------------- norm
def benchmark_norm() -> wise.Norm:
    """The running example norm (six constraints, five layers) plus four extensions.

    ``c7`` is a ``Lag`` with ``activation="each"``, ``c8`` a ``Precedence``,
    ``c9`` an ``Exclusion`` scoped to events after the goods receipt, and
    ``c10`` a ``Metric`` on the ``count_events`` recipe ``manual_touches``.
    Four views (two by constraint weights, two by layer weights), scoring
    mode ``layer_balanced``.
    """
    base = wise.running_p2p_norm()
    constraints = (
        *base.constraints,
        wise.NormConstraint(
            "c7",
            "lead_times",
            wise.Lag(GR, CLR, delta=30, width=30, unit="D", activation="each"),
            description="Every goods receipt cleared within 30 days",
        ),
        wise.NormConstraint("c8", "match", wise.Precedence(GR, INV), description="Invoice must not precede the goods receipt"),
        wise.NormConstraint(
            "c9", "exceptions", wise.Exclusion(CHP, after=GR), description="No price change after the goods receipt"
        ),
        wise.NormConstraint(
            "c10",
            "handling",
            wise.Metric("manual_touches", threshold=3, width=5),
            description="Limit manual touches",
        ),
    )
    views = (
        wise.View(
            "Finance",
            constraint_weights={
                "c1": 0.15,
                "c2": 0.30,
                "c3": 0.15,
                "c4": 0.05,
                "c5": 0.05,
                "c6": 0.10,
                "c7": 0.10,
                "c8": 0.05,
                "c9": 0.03,
                "c10": 0.02,
            },
        ),
        wise.View(
            "Logistics",
            constraint_weights={
                "c1": 0.20,
                "c2": 0.10,
                "c3": 0.05,
                "c4": 0.20,
                "c5": 0.25,
                "c6": 0.05,
                "c7": 0.05,
                "c8": 0.05,
                "c9": 0.03,
                "c10": 0.02,
            },
        ),
        wise.View(
            "Procurement",
            layer_weights={"completeness": 0.20, "lead_times": 0.20, "match": 0.30, "handling": 0.15, "exceptions": 0.15},
        ),
        wise.View("Audit", layer_weights={"completeness": 0.10, "lead_times": 0.10, "match": 0.30, "exceptions": 0.50}),
    )
    return wise.Norm(
        constraints,
        base.layers,
        views,
        name="Synthetic benchmark norm (running example + 4)",
        version="1",
        scoring_mode="layer_balanced",
        derived_attributes=(
            {"name": "manual_touches", "kind": "count_events", "where": {"column": "resource", "regex": "^user"}},
        ),
    )


# ----------------------------------------------------------------------------- timing
class Stopwatch:
    """Collects ``perf_counter`` durations per named step."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        yield
        self.seconds[name] = time.perf_counter() - start


def _peak_rss_gb() -> float:
    """Peak resident set size of this process (macOS reports bytes, Linux kibibytes)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss * 1024 / 1e9


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "wise": wise.__version__,
    }


@dataclass
class Outputs:
    """Everything one pipeline pass produces (kept for the consistency checks)."""

    log: wise.EventLog
    result: wise.ScoreResult
    backlog: pd.DataFrame
    layers: pd.DataFrame
    drivers: pd.DataFrame
    agreement: pd.DataFrame
    table: pd.DataFrame


def pipeline(events: pd.DataFrame, norm: wise.Norm, watch: Stopwatch) -> Outputs:
    """One cold pass over the classic API, each step timed by ``watch``."""
    with watch.step("build"):
        log = wise.EventLog(
            events,
            case_col="case",
            activity_col="activity",
            timestamp_col="time",
            case_attributes=["company", "flow_type", "vendor"],
            exposure_col="amount",
            exposure_agg="max",
        )
    with watch.step("score"):
        result = wise.score(log, norm)
    with watch.step("prioritize"):
        backlog = wise.prioritize(result, list(SLICE), view=VIEW, gamma=GAMMA)
    with watch.step("drivers"):
        layers = wise.layer_drivers(result, list(SLICE), view=VIEW)
        company, vendor = backlog.index[0]
        drivers = wise.constraint_drivers(result, VIEW, {"company": company, "vendor": vendor})
    with watch.step("agreement"):
        agreement = wise.view_agreement(result, list(SLICE), k=TOP_K, gamma=GAMMA)
    with watch.step("diagnostics"):
        censored = wise.right_censored(log, CLR, window="60D", opened_by=INV)
        replication = wise.event_replication(log)
    with watch.step("validation_table"):
        table = wise.validation_table(result, VIEW, list(SLICE), censored=censored, replication=replication, gamma=GAMMA)
    return Outputs(log, result, backlog, layers, drivers, agreement, table)


def _check(out: Outputs, norm: wise.Norm, n_cases: int, n_events: int) -> float:
    """Cheap consistency checks (a benchmark that computes nothing is not a benchmark); returns the decomposition error."""
    if len(out.log) != n_cases or len(out.log.events) != n_events:
        raise RuntimeError(f"log has {len(out.log)} cases / {len(out.log.events)} events, expected {n_cases} / {n_events}")
    scores, violations = out.result.scores, out.result.violations
    if scores.shape != (n_cases, len(norm.views)) or violations.shape != (n_cases, len(norm.constraints)):
        raise RuntimeError(f"unexpected result shapes {scores.shape} / {violations.shape}")
    if out.backlog.empty or out.layers.empty or out.drivers.empty or out.agreement.empty or out.table.empty:
        raise RuntimeError("an output table is empty")
    return float(out.result.check_decomposition())


def run(scale: float = 1.0, *, seed: int = DEFAULT_SEED, trace_memory: bool = True) -> dict[str, Any]:
    """Generate the log, time every step once, measure peak memory, and return the report.

    ``scale`` multiplies both the case and the event count of the full size
    (``1.0`` is 250 000 cases / 1 600 000 events).

    Timing and memory are measured in two separate passes: with tracemalloc
    active, the pyarrow-backed string columns of pandas 3 make the build
    step about five times slower, so the timed pass runs without it and a
    second, untimed pass under tracemalloc yields ``peak_gb``. The input
    frame is allocated before tracing starts and is reported separately as
    ``input_gb``.
    """
    if scale <= 0:
        raise ValueError("scale must be > 0")
    n_cases = max(1, round(FULL_CASES * scale))
    n_events = max(n_cases, round(FULL_EVENTS * scale))
    events = synthetic_events(n_cases, n_events, seed=seed)
    norm = benchmark_norm()

    watch = Stopwatch()
    out = pipeline(events, norm, watch)
    decomposition_error = _check(out, norm, n_cases, n_events)
    n_slices = len(out.backlog)
    scored_share = float(out.result.scores[VIEW].notna().mean())
    del out

    peak_gb: float | None = None
    if trace_memory:
        tracemalloc.start()
        try:
            _check(pipeline(events, norm, Stopwatch()), norm, n_cases, n_events)
            peak_gb = tracemalloc.get_traced_memory()[1] / 1e9
        finally:
            tracemalloc.stop()

    report: dict[str, Any] = {step: round(watch.seconds[step], 4) for step in STEPS}
    report["total"] = round(sum(watch.seconds.values()), 4)
    report["peak_gb"] = None if peak_gb is None else round(peak_gb, 4)
    report["peak_rss_gb"] = round(_peak_rss_gb(), 4)
    report["input_gb"] = round(float(events.memory_usage(deep=True).sum()) / 1e9, 4)
    report["scale"] = scale
    report["seed"] = seed
    report["n_cases"] = n_cases
    report["n_events"] = n_events
    report["n_slices"] = n_slices
    report["scored_share"] = round(scored_share, 4)
    report["decomposition_error"] = decomposition_error
    report["platform"] = platform.platform()
    report["versions"] = _versions()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="fraction of the full size (1 = 250 000 cases / 1 600 000 events, 0.05 = 12 500 / 80 000)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed of numpy.random.default_rng (default 0)")
    parser.add_argument(
        "--no-tracemalloc", action="store_true", help="skip the second, untimed pass under tracemalloc (peak_gb becomes null)"
    )
    parser.add_argument("--json-out", metavar="PATH", help="also write the JSON report to this file")
    args = parser.parse_args(argv)

    report = run(args.scale, seed=args.seed, trace_memory=not args.no_tracemalloc)
    text = json.dumps(report, indent=2, sort_keys=False)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
