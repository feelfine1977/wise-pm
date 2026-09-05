"""WISE on the BPI Challenge 2019 purchase-to-pay log (paper Section V).

Usage:
    python examples/bpic19_evaluation.py /path/to/BPI_Challenge_2019.csv

The challenge CSV is available from https://icpmconference.org/2019/icpm-2019/contests-challenges/bpi-challenge-2019/
(1,595,923 events, 251,734 PO items). The norm in ``examples/bpic19_norm.json``
holds the 29 constraints, seven layers and four views of the evaluation.
The script prints the outputs behind the paper's Tables X–XII and Figures 3–5:
mean scores per view, layer attribution, the company × spend-area backlog,
document-backlog concentration, view agreement, and the validation
diagnostics for the focus slices.

The norm file records the settings of the evaluation: ``scoring_mode`` is
``layer_balanced``; lag constraints use ``response = "first_overall"`` and
are not applicable when an endpoint is missing (``missing_a = missing_b =
"skip"``, which differs from the library default), the missing milestone
being penalised by the closure layer; order constraints are precedence
checks that are not applicable without both activities. Change these fields
in the norm file to score the log under other readings.
"""

import sys
import time
from pathlib import Path

import pandas as pd

import wise

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

FLOW_TYPES = {
    "3-way match, invoice after GR": "DF1",
    "3-way match, invoice before GR": "DF2",
    "2-way match": "2-way",
    "Consignment": "Consignment",
}
CASE, ACTIVITY, TIMESTAMP = "case concept:name", "event concept:name", "event time:timestamp"
COMPANY, SPEND, VENDOR, ITEM, DOC = (
    "case Company",
    "case Spend area text",
    "case Vendor",
    "case Item Type",
    "case Purchasing Document",
)
INVOICE = ["Record Invoice Receipt", "Vendor creates invoice"]
CLEAR = "Clear Invoice"
FOCUS = [("companyID_0000", "Packaging"), ("companyID_0000", "Logistics"), ("companyID_0003", "Real Estate")]


def load(path: str) -> wise.EventLog:
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    ts = pd.to_datetime(df[TIMESTAMP], format="%d-%m-%Y %H:%M:%S.%f", errors="coerce")
    missing = ts.isna() & df[TIMESTAMP].notna()
    if missing.any():
        ts.loc[missing] = pd.to_datetime(df.loc[missing, TIMESTAMP], dayfirst=True, format="mixed", errors="coerce")
    df[TIMESTAMP] = ts
    df["flow_type"] = df["case Item Category"].map(FLOW_TYPES).fillna("other")
    for col in (COMPANY, SPEND, VENDOR, ITEM):
        df[col] = df[col].fillna("(missing)")
    df["event Cumulative net worth (EUR)"] = df["event Cumulative net worth (EUR)"].abs()
    return wise.EventLog(
        df,
        case_col=CASE,
        activity_col=ACTIVITY,
        timestamp_col=TIMESTAMP,
        case_attributes=["flow_type", COMPANY, SPEND, VENDOR, ITEM, DOC, "case Document Type"],
        exposure_col="event Cumulative net worth (EUR)",
        exposure_agg="max",
        order_col="eventID",
        missing_timestamps="keep",
    )


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    t0 = time.time()
    log = load(argv[1])
    log.add_case_attribute("case_start_quarter", log.cases["first_ts"].dt.to_period("Q").astype(str))
    print(log, f"({time.time() - t0:.1f}s)")
    print(log.validate().to_string(), "\n")

    norm = wise.Norm.load(Path(__file__).with_name("bpic19_norm.json"))
    print(norm)
    for issue in norm.check(log):
        print("  -", issue)

    t0 = time.time()
    result = wise.score(log, norm)
    print(
        f"\nscored {len(result.scores):,} cases in {time.time() - t0:.1f}s; "
        f"applicability density {result.applicability_density():.3f} (evaluated), {result.applicability_density(scope=True):.3f} (in scope)"
    )
    print("\nMean scores and layer attribution per view (Fig. 3, Fig. 5)\n", result.summary().round(4).to_string())

    by = [COMPANY, SPEND]
    for view in result.views:
        backlog = wise.prioritize(result, by, view=view, gamma=20)
        drivers = wise.layer_drivers(result, by, view=view)
        print(
            f"\nTop slices, {view} view (γ = 20)\n",
            wise.hotspot_table(backlog, top=8, drivers=drivers)[
                ["n_cases", "stable_gap", "stable_PI", "hotspot", "dominant_layer"]
            ]
            .round(4)
            .to_string(),
        )

    docs = {v: wise.prioritize(result, DOC, view=v, gamma=20) for v in result.views}
    print(
        "\nDocument backlog concentration, Compliance view (Fig. 4)\n",
        wise.concentration(docs["Compliance"]).round(4).to_string(),
    )
    print(
        "\nView agreement on document backlogs (Table X)\n", wise.view_agreement(result, DOC, k=20, gamma=20).round(3).to_string()
    )

    window_end = log.cases["last_ts"].quantile(0.999)
    censored = wise.right_censored(log, CLEAR, window="60D", opened_by=INVOICE, window_end=window_end)
    replication = wise.event_replication(log)
    table = wise.validation_table(result, "Automation", by, censored=censored, replication=replication, gamma=20)
    print("\nValidation diagnostics for the focus slices, Automation view (Table XII)\n", table.loc[FOCUS].round(4).to_string())

    packaging = {COMPANY: "companyID_0000", SPEND: "Packaging"}
    vendors = wise.penalty_mass(result, "Automation", VENDOR, where=packaging)
    k80 = int((vendors["cum_share"] < 0.8).sum()) + 1
    print(
        f"\nVendor Pareto inside companyID_0000 × Packaging (Fig. 9): {k80} of {len(vendors)} vendors carry 80% of the Automation penalty mass"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
