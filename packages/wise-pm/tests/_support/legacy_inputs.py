"""The synthetic golden log written as the input files of the 0.1.0 examples.

``examples/bpic19_evaluation.py`` reads the BPI Challenge 2019 CSV, which is not
available in CI; :func:`write_bpic19_csv` writes the committed 2 000-case synthetic
purchase-to-pay log in the challenge's layout (column names, timestamp format, item
categories, the companies and spend areas of the focus slices), so the example runs
end to end on it. :func:`write_readme_csv` writes the ``log.csv`` the quick start of
the 0.1.0 README reads. Pandas only, so the same files can be written next to any
version of the library.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SOURCE = Path(__file__).resolve().parents[1] / "data" / "p2p_synthetic_2k.parquet"
ITEM_CATEGORY = {"DF1": "3-way match, invoice after GR", "DF2": "3-way match, invoice before GR"}
#: The two largest companies carry the identifiers of the focus slices of the evaluation.
COMPANY = {"C03": "companyID_0000", "C08": "companyID_0003"}
SPEND_AREAS = ("Packaging", "Logistics", "Real Estate")


def write_bpic19_csv(path: Path) -> Path:
    events = pd.read_parquet(SOURCE)
    others = sorted(set(events["company"]) - set(COMPANY))
    company = {**COMPANY, **{c: f"companyID_{i + 10:04d}" for i, c in enumerate(others)}}
    vendor_number = events["vendor"].str.removeprefix("V").astype(int)
    case_number = events["case"].str.removeprefix("POI").astype(int)
    frame = pd.DataFrame(
        {
            "eventID": events["eventID"],
            "case concept:name": events["case"],
            "event concept:name": events["activity"],
            "event time:timestamp": events["time"].dt.strftime("%d-%m-%Y %H:%M:%S.000"),
            "event org:resource": events["org:resource"],
            "event Cumulative net worth (EUR)": events["amount"],
            "case Company": events["company"].map(company),
            "case Spend area text": [SPEND_AREAS[v % len(SPEND_AREAS)] for v in vendor_number],
            "case Vendor": events["vendor"],
            "case Item Type": "Standard",
            "case Item Category": events["flow_type"].map(ITEM_CATEGORY),
            "case Purchasing Document": [f"doc_{n // 4:04d}" for n in case_number],
            "case Document Type": "Standard PO",
        }
    )
    frame.to_csv(path, index=False, encoding="latin-1")
    return path


def write_readme_csv(path: Path) -> Path:
    """The columns of the 0.1.0 README quick start; company ``C03`` is its company ``A``.
    The quick start reads a log without missing timestamps, so the untimed events are left out."""
    events = pd.read_parquet(SOURCE)
    events = events[events["time"].notna()]
    vendor_number = events["vendor"].str.removeprefix("V").astype(int)
    frame = pd.DataFrame(
        {
            "case": events["case"],
            "activity": events["activity"],
            "time": events["time"],
            "amount": events["amount"],
            "net_worth": events["amount"].abs(),
            "flow_type": events["flow_type"],
            "company": events["company"].replace({"C03": "A"}),
            "spend_area": [SPEND_AREAS[v % len(SPEND_AREAS)] for v in vendor_number],
            "vendor": events["vendor"],
        }
    )
    frame.to_csv(path, index=False)
    return path
