"""Independent bounded input-contract fixtures and raw-XES witness checks."""

import argparse
import gzip
import json
from pathlib import Path

import bpic2013_transfer as app
import pandas as pd
from defusedxml.ElementTree import iterparse

p = argparse.ArgumentParser()
p.add_argument("--results", type=Path, required=True)
p.add_argument("--input-dir", type=Path, required=True)
args = p.parse_args()
cfg = json.loads((Path(__file__).parent / "illustrative_norm.json").read_text())
checks = []


def ck(name, condition):
    checks.append({"check": name, "passed": bool(condition)})
    assert condition, name


# Constructed records retain raw-key form; expectations below are hand-specified.
def fixture(teams, times=None, fam="incidents", statuses=None):
    n = len(teams)
    status = statuses or ["Accepted"] * n
    rows = []
    for i in range(n):
        row = {
            "case_id": "synthetic",
            "trace_position": 1,
            "event_position": i,
            "event::concept:name": status[i],
            "event::lifecycle:transition": "In Progress",
            "event::time:timestamp": (times or [f"2012-01-{j + 1:02d}T00:00:00Z" for j in range(n)])[i],
            "event::product": "TEST",
            "event::impact": "Medium",
            "event::org:group": teams[i],
        }
        if fam == "incidents":
            row["event::organization involved"] = "Org line TEST"
        rows.append(row)
    return app.features(pd.DataFrame(rows), fam, cfg)[0]


f = fixture(["A", "A", "B", "A"])
ck("compressed_team_changes_two", f.team_changes.iloc[0] == 2)
ck("compressed_immediate_return_one", f.team_returns.iloc[0] == 1)
v, a, r = app.assess(f, "incidents", cfg)
ck("C01_threshold_boundary_zero", v.C01.iloc[0] == 0)
ck("C02_one_return_half", v.C02.iloc[0] == 0.5)
f = fixture(["A", None, "B", "A"])
ck("missing_team_never_bridged", pd.isna(f.team_returns.iloc[0]))
ck("missing_team_reason", f.state_team.iloc[0] == "missing_team_value")
f = fixture(["A", "B"], times=["2012-01-01T00:00:00Z"] * 2)
ck(
    "different_tied_teams_unevaluable",
    f.state_team.iloc[0] == "ambiguous_timestamp_tie",
)
f = fixture(["A", "B"], times=["invalid", "2012-01-02T00:00:00Z"])
ck("invalid_timestamp_unevaluable", f.state_team.iloc[0] == "invalid_timestamp")
f = fixture(["A", "A", "A"], statuses=["Accepted", "Completed", "Accepted"])
ck("post_completed_active_detected", f.post_completed_active.iloc[0] == 1)
f = fixture(["Org line A", "Org line B"], fam="open_problems")
v, a, r = app.assess(f, "open_problems", cfg)
ck(
    "open_XES_team_missing_not_zero",
    pd.isna(v.C01.iloc[0]) and r.C01.iloc[0] == "missing_XES_team_field",
)
ck("open_WaitUser_outside_scope", not a.C04.iloc[0] and pd.isna(v.C04.iloc[0]))
ck("open_reactivation_outside_scope", not a.C06.iloc[0])
# Independent raw-trace traversal and counts, without the production feature adapter.
for fam in app.FAMILIES:
    dest = args.results / fam
    wit = json.loads((dest / "witness.json").read_text())
    cid = wit["case_id"]
    events = None
    with gzip.open(args.input_dir / f"bpi_challenge_2013_{fam}.xes.gz", "rb") as stream:
        for _, node in iterparse(
            stream,
            events=["end"],
            forbid_dtd=True,
            forbid_entities=True,
            forbid_external=True,
        ):
            if node.tag.rsplit("}", 1)[-1] != "trace":
                continue
            trace_id = next(
                (x.attrib.get("value") for x in node if x.attrib.get("key") == "concept:name"),
                None,
            )
            if trace_id == cid:
                events = [
                    {x.attrib["key"]: x.attrib.get("value") for x in e} for e in node if e.tag.rsplit("}", 1)[-1] == "event"
                ]
                break
            node.clear()
    ck(fam + ":witness_found_raw", events is not None)
    f = pd.read_parquet(dest / "case_features.parquet").loc[cid]
    ck(fam + ":raw_event_count", len(events) == f.n_events)
    sub = [e.get("lifecycle:transition") for e in events]
    ck(fam + ":repeated_substatus", len(sub) - len(set(sub)) == f.repeated_substatus)
    stamps = pd.to_datetime([e["time:timestamp"] for e in events], utc=True, format="ISO8601")
    ck(
        fam + ":raw_span_days",
        abs((max(stamps) - min(stamps)).total_seconds() / 86400 - f.span_days) < 1e-12,
    )
    if fam != "open_problems":
        key = "org:group" if fam == "incidents" else "organization involved"
        pairs = sorted(zip(stamps, range(len(events)), events), key=lambda x: (x[0], x[1]))
        teams = [x[2][key] for x in pairs]
        runs = []
        for t in teams:
            if not runs or runs[-1] != t:
                runs.append(t)
        ck(fam + ":team_changes_from_raw", len(runs) - 1 == f.team_changes)
        ck(
            fam + ":team_returns_from_raw",
            sum(runs[i] == runs[i - 2] for i in range(2, len(runs))) == f.team_returns,
        )
    else:
        ck(
            fam + ":raw_XES_no_team_field",
            all("organization involved" not in e for e in events),
        )
    assessment = pd.read_csv(dest / "witness_assessment.csv", index_col=0)
    ck(
        fam + ":witness_weighted_sum",
        abs(assessment.penalty.sum() - (1 - wit["score"])) < 1e-12,
    )
report = {
    "status": "passed",
    "checks": len(checks),
    "passed": sum(x["passed"] for x in checks),
    "checks_detail": checks,
    "meaning": "12 synthetic contract assertions plus independent raw-source checks for three selected witnesses; not domain validation.",
}
report["provenance"] = {
    "verifier_sha256": app.sha(Path(__file__)),
    "adapter_sha256": app.sha(Path(__file__).parent / "bpic2013_transfer.py"),
    "norm_sha256": app.sha(Path(__file__).parent / "illustrative_norm.json"),
    "primary_manifest_sha256": app.sha(args.results / "manifest.json"),
    "raw_inputs": {fam: app.sha(args.input_dir / f"bpi_challenge_2013_{fam}.xes.gz") for fam in app.FAMILIES},
}
app.dump(args.results / "contract_verification.json", report)
print(json.dumps({k: report[k] for k in ["status", "checks", "passed"]}))
