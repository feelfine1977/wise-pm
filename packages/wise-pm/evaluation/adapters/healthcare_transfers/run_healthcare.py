#!/usr/bin/env python3
"""Safe, nonclinical pathway/format transfer tests. No patient IDs/clinical values exported."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
import zipfile
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from defusedxml import ElementTree as ET

sys.dont_write_bytecode = True


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        )
    )


def localtag(e):
    return e.tag.rsplit("}", 1)[-1]


def ramp(x, c):
    return float(np.clip((x - c["threshold"]) / c["width"], 0, 1)) if pd.notna(x) else np.nan


def parse_sepsis(path):
    root = ET.parse(path, forbid_dtd=True, forbid_entities=True, forbid_external=True).getroot()
    traces = []
    ids = set()
    attrs = Counter()
    activities = Counter()
    orgs = set()
    invalid = 0
    for ti, tr in enumerate([x for x in root if localtag(x) == "trace"], 1):
        td = {x.get("key"): x.get("value") for x in tr if localtag(x) != "event"}
        original = td.get("concept:name")
        if not original or original in ids:
            raise ValueError("Missing or duplicate case ID")
        ids.add(original)
        events = []
        for ei, ev in enumerate([x for x in tr if localtag(x) == "event"], 1):
            d = {x.get("key"): x.get("value") for x in ev}
            attrs.update(d.keys())
            if not d.get("concept:name"):
                raise ValueError("Missing activity")
            t = pd.to_datetime(d.get("time:timestamp"), utc=True, errors="coerce")
            invalid += int(pd.isna(t))
            events.append({"activity": d["concept:name"], "time": t, "position": ei})
            activities[d["concept:name"]] += 1
            orgs.add(d.get("org:group"))
        traces.append({"case_id": f"S{ti:04d}", "source_trace_ordinal": ti, "events": events})
    return traces, {
        "cases": len(traces),
        "events": sum(len(t["events"]) for t in traces),
        "activity_counts": dict(activities),
        "event_attribute_names": sorted(attrs),
        "org_group_distinct_including_missing": len(orgs),
        "invalid_or_missing_timestamps": invalid,
        "case_unit": "recorded hospital pathway",
        "export_policy": "No original case IDs, lab values, diagnoses, demographics or absolute timestamps; witnesses link by source hash and trace ordinal.",
    }


def parse_workbook(path):
    # Read the supplied simple XLSX using data XML only. No macros, formula execution or external links.
    with zipfile.ZipFile(path) as z:
        if any("vbaProject" in x for x in z.namelist()):
            raise ValueError("Unexpected macro content")
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        props = next((e for e in wb if localtag(e) == "workbookPr"), None)
        date1904 = props is not None and props.get("date1904") in ("1", "true")
        sheets = [e for e in wb.iter() if localtag(e) == "sheet"]
        if len(sheets) != 1:
            raise ValueError("Expected one worksheet")
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")):
                strings.append("".join(t.text or "" for t in si.iter() if localtag(t) == "t"))
        root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in [e for e in root.iter() if localtag(e) == "row"]:
            vals = {}
            for c in row:
                if any(localtag(e) == "f" for e in c):
                    raise ValueError("Formula input unsupported")
                col = "".join(x for x in c.get("r", "") if x.isalpha())
                typ = c.get("t")
                v = next((x.text for x in c if localtag(x) == "v"), None)
                if typ == "s":
                    v = strings[int(v)]
                elif typ == "inlineStr":
                    v = "".join(x.text or "" for x in c.iter() if localtag(x) == "t")
                vals[col] = v
            rows.append((int(row.get("r")), vals))
    if [rows[0][1].get(c) for c in ["A", "B", "C"]] != [
        "Patient_id",
        "Activity",
        "Timestamp",
    ]:
        raise ValueError("Unexpected worksheet schema")
    grouped = OrderedDict()
    acts = Counter()
    invalid = 0
    for ri, d in rows[1:]:
        if not any(d.values()):
            continue
        if not d.get("A") or not d.get("B"):
            raise ValueError("Missing case/activity")
        try:
            t = pd.Timestamp("1904-01-01" if date1904 else "1899-12-30") + pd.to_timedelta(float(d["C"]), unit="D")
        except (ValueError, TypeError, OverflowError):
            t = pd.NaT
        invalid += int(pd.isna(t))
        acts[d["B"]] += 1
        grouped.setdefault(d["A"], []).append({"activity": d["B"], "time": t, "position": ri})
    traces = [{"case_id": f"W{i:03d}", "source_case_ordinal": i, "events": ev} for i, ev in enumerate(grouped.values(), 1)]
    return traces, {
        "cases": len(traces),
        "events": sum(map(len, grouped.values())),
        "sheet": sheets[0].get("name"),
        "headers": ["Patient_id", "Activity", "Timestamp"],
        "activity_counts": dict(acts),
        "invalid_or_missing_timestamps": invalid,
        "date_system": "1904" if date1904 else "1900",
        "case_unit": "supplied Patient_id grouped history; identity/provenance unknown",
        "export_policy": "No original IDs or calendar timestamps; worksheet row positions retained in witnesses.",
    }


def derive(traces, kind, norm):
    features = []
    vs = []
    scopes = []
    reasons = []
    for tr in traces:
        ev = tr["events"]
        counts = Counter(e["activity"] for e in ev)
        ts = [e["time"] for e in ev]
        valid = bool(ts) and all(pd.notna(t) for t in ts)
        span = (max(ts) - min(ts)).total_seconds() / 86400 if len(ts) >= 2 and valid else np.nan
        if kind == "sepsis":
            excluded = set(norm["excluded_measurement_activities"])
            repeated = sum(max(n - 1, 0) for a, n in counts.items() if a not in excluded)
            regs = [e for e in ev if e["activity"] == "ER Registration"]
            rels = [e for e in ev if e["activity"].startswith("Release ")]
            labels = sorted({e["activity"] for e in rels})
            group = " + ".join(labels) if labels else "No recorded release"
            interval = np.nan
            reason = "outside_scope:no_registration"
            if regs:
                reason = "unevaluable:no_recorded_release"
                if rels:
                    if any(pd.isna(e["time"]) for e in regs + rels):
                        reason = "unevaluable:missing_endpoint_timestamp"
                    else:
                        x = (min(e["time"] for e in rels) - min(e["time"] for e in regs)).total_seconds() / 86400
                        if x < 0:
                            reason = "unevaluable:negative_endpoint_order"
                        else:
                            interval = x
                            reason = "evaluated"
            raw = {
                "S01": repeated,
                "S02": span,
                "S03": interval,
                "S04": int(counts["Return ER"] > 0),
            }
            scope = {"S01": True, "S02": True, "S03": bool(regs), "S04": True}
            why = {
                "S01": "evaluated",
                "S02": "evaluated" if pd.notna(span) else "unevaluable:missing_time_or_short_trace",
                "S03": reason,
                "S04": "evaluated",
            }
        else:
            raw = {"W01": sum(max(n - 1, 0) for n in counts.values()), "W02": span * 24}
            scope = dict.fromkeys(raw, True)
            why = {
                "W01": "evaluated",
                "W02": "evaluated" if pd.notna(span) else "unevaluable:missing_time_or_short_trace",
            }
            group = "All supplied workbook cases"
        val = {k: float(x) if norm["criteria"][k].get("binary") else ramp(x, norm["criteria"][k]) for k, x in raw.items()}
        for k in val:
            if not scope[k]:
                val[k] = np.nan
        features.append(
            {
                "case_id": tr["case_id"],
                "group": group,
                "event_count": len(ev),
                "span_days": span,
                **raw,
            }
        )
        vs.append(val)
        scopes.append(scope)
        reasons.append(why)
    f = pd.DataFrame(features).set_index("case_id")
    ix = f.index
    return (
        f,
        pd.DataFrame(vs, index=ix),
        pd.DataFrame(scopes, index=ix, dtype=bool),
        pd.DataFrame(reasons, index=ix),
    )


def evaluate(ref, f, v, scope, norm, view, mode="flat"):
    layers = {k: c["layer"] for k, c in norm["criteria"].items()}
    weights = pd.Series({k: norm["views"][view][layers[k]] for k in v}, dtype=float)
    s = ref.score_assessments(v, scope, weights, layers, mode=mode)
    g = ref.group_priorities(s["score"], f["group"])
    signed = ref.signed_components(s["penalties"], f["group"])
    co = ref.coco_same_target(s["penalty"], f["group"])
    residuals = {
        "weights": float((s["effective_weights"].sum(axis=1) - 1).abs().max()),
        "case_reconstruction": float((s["penalties"].sum(axis=1) - s["penalty"]).abs().max()),
        "signed_priority": float((signed.sum(axis=1).clip(lower=0) - g.PI).abs().max()),
        "coco_identity": float((co.numerator.clip(lower=0) - g.PI).abs().max()),
        "completion_bounds": float(
            ((s["flat_completion_bounds"].penalty_max - s["flat_completion_bounds"].penalty_min) - (1 - s["coverage"]))
            .abs()
            .max()
        ),
    }
    return s, g, signed, co, residuals


def positive(g, k):
    return list(g.index[g.PI > 1e-12][:k])


def independent_witnesses(traces, f, v, scope, norm, kind):
    selected = [f.index[0], f.span_days.idxmax()]
    if kind == "sepsis":
        missing = f.index[f["group"] == "No recorded release"]
        selected += list(missing[:1])
        selected += [f.S01.idxmax()]
    else:
        selected = list(f.index)
    selected = list(dict.fromkeys(selected))
    rows = []
    checks = []
    for tr in traces:
        if tr["case_id"] not in selected:
            continue
        ev = tr["events"]
        cid = tr["case_id"]
        tvalid = all(pd.notna(e["time"]) for e in ev)
        times = [e["time"] for e in ev]
        duration = (max(times) - min(times)).total_seconds() / 86400 if tvalid and len(ev) > 1 else None

        def sev(x, k):
            c = norm["criteria"][k]
            return None if x is None else min(1.0, max(0.0, (x - c["threshold"]) / c["width"]))

        seen = set()
        extra = 0
        for e in ev:
            a = e["activity"]
            if kind == "sepsis" and a in norm["excluded_measurement_activities"]:
                continue
            if a in seen:
                extra += 1
            seen.add(a)
        if kind == "sepsis":
            r = [e for e in ev if e["activity"] == "ER Registration"]
            releases = [e for e in ev if e["activity"].startswith("Release ")]
            delta = None
            if r and releases and all(pd.notna(e["time"]) for e in r + releases):
                z = (sorted(e["time"] for e in releases)[0] - sorted(e["time"] for e in r)[0]).total_seconds() / 86400
                if z >= 0:
                    delta = z
            expect = {
                "S01": sev(extra, "S01"),
                "S02": sev(duration, "S02"),
                "S03": sev(delta, "S03"),
                "S04": float(any(e["activity"] == "Return ER" for e in ev)),
            }
            sc = {"S01": True, "S02": True, "S03": bool(r), "S04": True}
        else:
            expect = {
                "W01": sev(extra, "W01"),
                "W02": sev(None if duration is None else duration * 24, "W02"),
            }
            sc = dict.fromkeys(expect, True)
        for k, x in expect.items():
            ok = (pd.isna(v.loc[cid, k]) if x is None else math.isclose(v.loc[cid, k], x, abs_tol=1e-12)) and bool(
                scope.loc[cid, k]
            ) == sc[k]
            checks.append(
                {
                    "case_id": cid,
                    "criterion": k,
                    "passed": ok,
                    "expected_severity": x,
                    "in_scope": sc[k],
                }
            )
        start = min((e["time"] for e in ev if pd.notna(e["time"])), default=None)
        for e in ev:
            rows.append(
                {
                    "case_id": cid,
                    "source_trace_ordinal": tr.get("source_trace_ordinal"),
                    "source_sheet": None if kind == "sepsis" else "01_hospital_data",
                    "source_event_or_row_position": e["position"],
                    "activity": e["activity"],
                    "elapsed_hours_from_first_event": None
                    if start is None or pd.isna(e["time"])
                    else (e["time"] - start).total_seconds() / 3600,
                }
            )
    return checks, rows


def fixtures(norm):
    t = pd.Timestamp("2020-01-01", tz="UTC")

    def trace(acts):
        return [
            {
                "case_id": "F001",
                "events": [{"activity": a, "time": ts, "position": i + 1} for i, (a, ts) in enumerate(acts)],
            }
        ]

    specs = [
        ("missing_release", [("ER Registration", t)], "S03", True, None),
        ("missing_registration", [("Release A", t)], "S03", False, None),
        (
            "missing_release_timestamp",
            [("ER Registration", t), ("Release A", pd.NaT)],
            "S03",
            True,
            None,
        ),
        (
            "invalid_repeated_endpoint",
            [
                ("ER Registration", t),
                ("Release A", t + pd.Timedelta(days=4)),
                ("Release A", pd.NaT),
            ],
            "S03",
            True,
            None,
        ),
        (
            "negative_order",
            [("Release A", t), ("ER Registration", t + pd.Timedelta(days=1))],
            "S03",
            True,
            None,
        ),
        ("same_time", [("ER Registration", t), ("Release A", t)], "S03", True, 0),
        (
            "threshold_boundary",
            [("ER Registration", t), ("Release A", t + pd.Timedelta(days=3))],
            "S03",
            True,
            0,
        ),
        ("return_observed", [("Return ER", t), ("Return ER", t)], "S04", True, 1),
        ("measurement_repeats_excluded", [("CRP", t)] * 20, "S01", True, 0),
        (
            "no_assumed_return",
            [("ER Registration", t), ("Release A", t)],
            "S04",
            True,
            0,
        ),
    ]
    out = []
    for name, acts, k, ins, val in specs:
        _, v, a, _ = derive(trace(acts), "sepsis", norm)
        got = v.iloc[0][k]
        passed = bool(a.iloc[0][k]) == ins and (pd.isna(got) if val is None else math.isclose(float(got), val, abs_tol=1e-12))
        out.append({"check": name, "passed": passed})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--project-root", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--norm", type=Path, default=Path(__file__).with_name("norm_frozen.json"))
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    norm = json.loads(args.norm.read_text())
    refpath = args.project_root / "code/wise_reference.py"
    spec = importlib.util.spec_from_file_location("wise_reference_healthcare", refpath)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    src = args.data_root
    sp = src / "archive (2)" / "Sepsis Cases - Event Log.xes"
    wp = src / "01_hospital_data_update.xlsx"
    traces_s, schema_s = parse_sepsis(sp)
    traces_w, schema_w = parse_workbook(wp)
    raw = {
        str(p.relative_to(src)): sha(p)
        for p in [
            sp,
            wp,
            src / "archive (2)" / "readme.txt",
            src / "archive (2)" / "DATA.xml",
            src / "archive (2).zip",
            src / "archive (7).zip",
        ]
    }
    with zipfile.ZipFile(src / "archive (7).zip") as z:
        duplicate_w = z.read(wp.name) == wp.read_bytes()
    with zipfile.ZipFile(src / "archive (2).zip") as z:
        matches_s = {n: z.read(n) == (src / "archive (2)" / n).read_bytes() for n in ["readme.txt", "DATA.xml", sp.name]}
    triage = {
        "sepsis": {
            "status": "executed_nonclinical_pathway_transfer",
            "provenance": "Mannhardt 2016; DOI 10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460; anonymised and time-shifted with within-case durations preserved",
            "archive_matches": matches_s,
            "primary_metadata_md5_matches": {
                f: hashlib.md5((src / "archive (2)" / f).read_bytes()).hexdigest() == h
                for f, h in {
                    "readme.txt": "d990049d11124aeb5d68f2db6a2981d4",
                    "DATA.xml": "c0bf23210eb3c7c9e2d7f2b8837cd37d",
                }.items()
            },
            "exact_published_xes_identity": "not independently checksum-verified against decompressed official archive",
        },
        "hospital_workbook": {
            "status": "executed_technical_fixture_only",
            "provenance": "unknown; filename searches did not identify primary documentation",
            "cases": len(traces_w),
            "events": schema_w["events"],
            "archive7_identical_to_standalone": duplicate_w,
            "limitation": "No business grouping or normative context. Three cases do not establish empirical transfer; no original IDs exported.",
        },
    }
    save(out / "triage.json", triage)
    save(out / "executed_norm.json", norm)
    allsummary = []
    allchecks = []
    allsensitivity = []
    wc = {}
    for kind, traces, schema in [
        ("sepsis", traces_s, schema_s),
        ("hospital_workbook", traces_w, schema_w),
    ]:
        d = out / kind
        d.mkdir(exist_ok=True)
        n = norm[kind]
        f, v, sc, r = derive(traces, kind, n)
        save(d / "schema.json", schema)
        f.to_csv(d / "case_features.csv")
        v.to_csv(d / "criterion_severity.csv")
        sc.to_csv(d / "criterion_scope.csv")
        r.to_csv(d / "criterion_state.csv")
        base = None
        for view in n["views"]:
            s, g, sg, co, res = evaluate(ref, f, v, sc, n, view)
            g.to_csv(d / f"groups_{view}.csv")
            sg.to_csv(d / f"signed_components_{view}.csv")
            co.to_csv(d / f"coco_{view}.csv")
            pd.DataFrame(
                {
                    k: s[k]
                    for k in [
                        "score",
                        "penalty",
                        "coverage",
                        "in_scope_count",
                        "evaluated_count",
                    ]
                }
            ).to_csv(d / f"case_scores_{view}.csv")
            for name, df in ref.coverage_diagnostics(s, f["group"]).items():
                df.to_csv(d / f"coverage_{name}_{view}.csv")
            pos = positive(g, n["top_k"])
            allsummary.append(
                {
                    "dataset": kind,
                    "view": view,
                    "unit": schema["case_unit"],
                    "cases": len(f),
                    "events": schema["events"],
                    "scored": int(s["score"].notna().sum()),
                    "groups": len(g),
                    "mean_penalty": float(s["penalty"].mean()),
                    "mean_coverage": float(s["coverage"].mean()),
                    "positive_groups": int((g.PI > 1e-12).sum()),
                    "selected_positive_top_k": len(pos),
                    "lead_group": str(g.index[0]) if pos else None,
                    "lead_PI": float(g.PI.iloc[0]) if pos else 0.0,
                }
            )
            allchecks.append(
                {
                    "dataset": kind,
                    "view": view,
                    "passed": max(res.values()) < 1e-10,
                    "residuals": res,
                }
            )
            if view == "Balanced":
                base = (s, g)
        checks, rows = independent_witnesses(traces, f, v, sc, n, kind)
        save(d / "source_witness_checks.json", checks)
        pd.DataFrame(rows).to_csv(d / "source_witness_events.csv", index=False)
        wc[kind] = {
            "cases": len({x["case_id"] for x in checks}),
            "scalar_checks": len(checks),
            "passed": all(x["passed"] for x in checks),
            "method": "independent scalar loops on source-parsed activity/time records; witnesses identified by source ordinal/worksheet row, not direct identifiers",
        }
        scenarios = []
        for view in n["views"]:
            scenarios.append((view, "valuation", f, v, sc, n, view, "flat"))
        scenarios.append(("layer_balanced", "assessment", f, v, sc, n, "Balanced", "layer_balanced"))
        mask = base[0]["coverage"] >= 1 - 1e-12
        if mask.any():
            scenarios.append(
                (
                    "complete_in_scope_evidence",
                    "population_change",
                    f.loc[mask],
                    v.loc[mask],
                    sc.loc[mask],
                    n,
                    "Balanced",
                    "flat",
                )
            )
        if kind == "sepsis":
            for th in norm["sensitivity"]["sepsis_S02_threshold_days"]:
                nn = copy.deepcopy(n)
                nn["criteria"]["S02"]["threshold"] = th
                ff, vv, aa, _ = derive(traces, kind, nn)
                scenarios.append(
                    (
                        f"S02_threshold_{th}",
                        "assessment",
                        ff,
                        vv,
                        aa,
                        nn,
                        "Balanced",
                        "flat",
                    )
                )
        bset = set(positive(base[1], n["top_k"]))
        for name, typ, ff, vv, aa, nn, view, mode in scenarios:
            ss, gg, _, _, _ = evaluate(ref, ff, vv, aa, nn, view, mode)
            oset = set(positive(gg, n["top_k"]))
            allsensitivity.append(
                {
                    "dataset": kind,
                    "scenario": name,
                    "type": typ,
                    "cases": len(ff),
                    "mean_coverage": float(ss["coverage"].mean()),
                    "requested_k": n["top_k"],
                    "base_positive_candidates": int((base[1].PI > 1e-12).sum()),
                    "other_positive_candidates": int((gg.PI > 1e-12).sum()),
                    "base_selected": len(bset),
                    "other_selected": len(oset),
                    "shared": len(bset & oset),
                    "jaccard": len(bset & oset) / len(bset | oset) if bset | oset else None,
                }
            )
        states = []
        for k in v:
            states.append(
                {
                    "criterion": k,
                    "in_scope": int(sc[k].sum()),
                    "evaluated": int(v[k].notna().sum()),
                    "positive": int((v[k] > 0).sum()),
                }
            )
        pd.DataFrame(states).to_csv(d / "criterion_summary.csv", index=False)
    fix = fixtures(norm["sepsis"])
    verified = (
        all(c["passed"] for c in allchecks)
        and all(c["passed"] for c in wc.values())
        and all(c["passed"] for c in fix)
        and duplicate_w
        and all(matches_s.values())
    )
    pd.DataFrame(allsummary).to_csv(out / "view_summary.csv", index=False)
    pd.DataFrame(allsensitivity).to_csv(out / "sensitivity.csv", index=False)
    verification = {
        "status": "passed" if verified else "failed",
        "numerical_checks": allchecks,
        "source_witnesses": wc,
        "contract_fixtures": fix,
        "meaning": "Execution agreement with author-defined transformations; no clinical validation, human study or performance superiority.",
    }
    save(out / "verification.json", verification)
    outputs = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    save(
        out / "manifest.json",
        {
            "status": verification["status"],
            "executed_utc": datetime.now(timezone.utc).isoformat(),
            "reference_sha256": sha(refpath),
            "script_sha256": sha(__file__),
            "config_sha256": sha(args.norm),
            "raw_sha256": raw,
            "output_sha256": outputs,
            "reference_path": str(refpath),
            "runtime": {
                "python": sys.version.split()[0],
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
            "results": "two executions: one empirical nonclinical pathway application and one three-case technical fixture; duplicate workbook not new dataset",
        },
    )
    print(
        json.dumps(
            {
                "status": verification["status"],
                "views": allsummary,
                "witness_checks": wc,
            },
            indent=2,
        )
    )
    if not verified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
