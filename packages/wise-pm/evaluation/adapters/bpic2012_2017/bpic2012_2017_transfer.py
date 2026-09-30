#!/usr/bin/env python3
"""Full-source BPIC2012/2017 illustration. No source mutation or model fitting.
CLI: --data-root PATH --project-root PATH --output PATH
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from defusedxml import ElementTree as ET

EXPECTED_CORE = "87a25cbfb36eb0546ed2602cf4048e6aa09afe3224ae47354fcefb7ea26282c4"
HERE = Path(__file__).resolve().parent
NUMERIC = {
    "RequestedAmount",
    "case:RequestedAmount",
    "AMOUNT_REQ",
    "FirstWithdrawalAmount",
    "NumberOfTerms",
    "MonthlyCost",
    "CreditScore",
    "OfferedAmount",
}
BOOL = {"Accepted", "Selected"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + "\n")


def tag(el):
    return el.tag.rsplit("}", 1)[-1]


def attrs(el):
    children = [x for x in el if x.get("key") is not None]
    keys = [x.get("key") for x in children]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate direct XES attribute keys are ambiguous")
    return {x.get("key"): x.get("value") for x in children}


def iter_traces(path):
    """DTD/entities forbidden; direct attributes only; never inherit metadata defaults."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as stream:
        it = ET.iterparse(stream, events=("start", "end"), forbid_dtd=True, forbid_entities=True, forbid_external=True)
        _, root = next(it)
        if tag(root) != "log":
            raise ValueError("Expected XES log")
        for event, el in it:
            if event == "end" and tag(el) == "trace":
                tr = attrs(el)
                rows = []
                for child in el:
                    if tag(child) == "event":
                        row = attrs(child)
                        nested = [ET.tostring(a, encoding="unicode") for a in child if len(a)]
                        if nested:
                            row["__nested_attributes_xml"] = json.dumps(nested)
                        rows.append(row)
                yield tr, rows
                root.remove(el)


def instant(x):
    try:
        d = datetime.fromisoformat(str(x).replace("Z", "+00:00"))
        return d.astimezone(timezone.utc) if d.tzinfo is not None else None
    except (ValueError, TypeError, OverflowError):
        return None


def present(v, norm):
    return v is not None and str(v).strip().upper() not in norm["missing_markers"]


def number(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else np.nan
    except (ValueError, TypeError):
        return np.nan


def amount_band(x, norm):
    if not math.isfinite(x) or x < 0:
        return "<unknown>"
    return norm["amount_labels"][int(np.searchsorted(norm["amount_bins"], x, side="right") - 1)]


def feature_row(tr, events, family, norm):
    cfg = norm["families"][family]
    names = [e.get("concept:name") for e in events]
    labels_ok = all(present(x, norm) for x in names) and all(present(e.get("lifecycle:transition"), norm) for e in events)
    starts = [
        e
        for e in events
        if str(e.get("concept:name", "")).startswith("W_") and str(e.get("lifecycle:transition", "")).lower() == "start"
    ]
    start_counts = Counter(e["concept:name"] for e in starts)
    dt = [instant(e.get("time:timestamp")) for e in events]
    time_ok = len(dt) >= 2 and all(x is not None for x in dt)
    known = [x for x in dt if x is not None]
    of = [e for e in events if str(e.get("concept:name", "")).startswith("O_")]
    creation = [
        e
        for e in events
        if e.get("concept:name") == cfg["offer_creation_activity"]
        and str(e.get("lifecycle:transition", "")).lower() == "complete"
    ]
    offers_ok = False
    offer_n = np.nan
    ids = []
    state_ids = []
    if family == "bpic2017":
        ids = [e.get("EventID") for e in creation]
        created = [
            e
            for e in events
            if e.get("concept:name") == "O_Created" and str(e.get("lifecycle:transition", "")).lower() == "complete"
        ]
        state_ids = [e.get("OfferID") for e in created]
        other_ids = [e.get("OfferID") for e in of if e.get("concept:name") != "O_Create Offer"]
        offers_ok = (
            bool(ids)
            and all(present(i, norm) for i in ids + state_ids + other_ids)
            and len(ids) == len(set(ids))
            and len(state_ids) == len(set(state_ids))
            and set(ids) == set(state_ids)
            and set(other_ids) <= set(ids)
        )
        if labels_ok and offers_ok:
            offer_n = len(set(ids))
    else:
        offers_ok = bool(creation)
        if labels_ok and offers_ok:
            offer_n = len(creation)
    amount = number(tr.get("RequestedAmount" if family == "bpic2017" else "AMOUNT_REQ"))
    resources_ok = all(present(e.get("org:resource"), norm) for e in starts)
    f = {
        "case_id": str(tr["concept:name"]),
        "event_count": len(events),
        "raw_trace_attributes_json": json.dumps(tr, sort_keys=True),
        "labels_complete": labels_ok,
        "valid_timestamps": len(known),
        "timestamp_ties": len(known) - len(set(known)),
        "source_order_timestamp_descents": sum(b < a for a, b in pairwise(known)) if len(known) == len(dt) else -1,
        "observed_span_days": (max(dt) - min(dt)).total_seconds() / 86400 if time_ok else np.nan,
        "span_evaluable": time_ok,
        "start_month_utc": min(known).strftime("%Y-%m") if time_ok else "<unknown>",
        "work_start_records": len(starts),
        "work_start_resource_missing": sum(not present(e.get("org:resource"), norm) for e in starts),
        "repeated_work_starts": sum(max(n - 1, 0) for n in start_counts.values()) if labels_ok else np.nan,
        "work_start_resource_breadth": len({e["org:resource"] for e in starts}) if labels_ok and resources_ok else np.nan,
        "has_offer_records": bool(of),
        "offer_creation_records": len(creation),
        "offer_identity_consistent": offers_ok if of else False,
        "offer_multiplicity": offer_n,
        "offer_ids_json": json.dumps(ids),
        "offer_created_state_ids_json": json.dumps(state_ids),
        "incomplete_state_records": sum(
            e.get("concept:name") == "A_Incomplete" and str(e.get("lifecycle:transition", "")).lower() == "complete"
            for e in events
        )
        if labels_ok
        else np.nan,
        "requested_amount": amount,
        "amount_band": amount_band(amount, norm),
        "application_type": tr.get("ApplicationType", "<not supplied>"),
        "loan_goal": tr.get("LoanGoal", "<not supplied>"),
        "has_recorded_cancellation": any(n in ("A_CANCELLED", "A_Cancelled") for n in names),
        "has_recorded_decline": any(n in ("A_DECLINED", "A_Denied") for n in names),
    }
    return f


def assessments(features, cfg, span_threshold=None):
    V = pd.DataFrame(index=features.index)
    A = pd.DataFrame(index=features.index)
    R = pd.DataFrame(index=features.index)
    for cid, c in cfg["criteria"].items():
        scope = features["has_offer_records"].astype(bool) if cid == "C04" else pd.Series(True, index=features.index)
        x = features[c["feature"]].astype(float)
        threshold = span_threshold if cid == "C01" and span_threshold is not None else c["threshold"]
        V[cid] = ((x - threshold) / c["width"]).clip(0, 1).where(scope)
        A[cid] = scope
        R[cid] = np.where(~scope, "out_of_scope", np.where(x.notna(), "evaluated", "unevaluable"))
    return V, A, R


def normalized(k, v):
    if v is None or v == "":
        return None
    if k == "time:timestamp":
        d = instant(v)
        return d.isoformat() if d else str(v)
    if k in NUMERIC:
        x = number(v)
        return x if math.isfinite(x) else str(v)
    if k in BOOL:
        return str(v).lower()
    return str(v)


def parse_family(path, family, norm, csv_path=None):
    features = []
    tc = Counter()
    ec = Counter()
    activities = Counter()
    lifecycle = Counter()
    al = Counter()
    nested = Counter()
    all_offer_ids = Counter()
    csv_mismatch = Counter()
    examples = []
    csvrows = 0
    seen = set()
    csv_handle = csv_path.open(newline="") if csv_path else None
    reader = csv.DictReader(csv_handle) if csv_handle else None
    fields = reader.fieldnames if reader else []
    for tr, events in iter_traces(path):
        cid = tr.get("concept:name")
        if not present(cid, norm) or cid in seen:
            raise ValueError("Missing/duplicate trace identity")
        seen.add(cid)
        tc.update(tr.keys())
        features.append(feature_row(tr, events, family, norm))
        if family == "bpic2017":
            all_offer_ids.update(json.loads(features[-1]["offer_ids_json"]))
        for pos, e in enumerate(events, 1):
            ec.update(k for k in e if not k.startswith("__"))
            activities[e.get("concept:name")] += 1
            lifecycle[e.get("lifecycle:transition")] += 1
            al[f"{e.get('concept:name')} | {e.get('lifecycle:transition')}"] += 1
            if "__nested_attributes_xml" in e:
                nested["events"] += 1
            if reader:
                row = next(reader, None)
                if row is None:
                    raise ValueError("Cleaned CSV exhausted before XES")
                csvrows += 1
                expected = {**e, **{f"case:{k}": v for k, v in tr.items()}}
                missing = set(expected) - set(fields)
                if missing:
                    raise ValueError(f"Cleaned CSV omits source columns: {missing}")
                for key in fields:
                    if normalized(key, expected.get(key)) != normalized(key, row.get(key)):
                        csv_mismatch[key] += 1
                        if len(examples) < 5:
                            examples.append(
                                {
                                    "case_id": cid,
                                    "event_position": pos,
                                    "field": key,
                                    "xes": expected.get(key),
                                    "csv": row.get(key),
                                }
                            )
    if reader:
        extra = sum(1 for _ in reader)
        csv_handle.close()
    else:
        extra = 0
    f = pd.DataFrame(features).set_index("case_id")
    f.index = f.index.astype(str)
    schema = {
        "cases": len(f),
        "events": int(f.event_count.sum()),
        "trace_attribute_presence": dict(tc),
        "event_attribute_presence": dict(ec),
        "activities": dict(activities),
        "lifecycle": dict(lifecycle),
        "activity_lifecycle": dict(al),
        "nested_event_attributes": dict(nested),
        "offer_ids_unique_global": len(all_offer_ids),
        "offer_creation_ids_repeated_across_or_within_applications": sum(n - 1 for n in all_offer_ids.values()),
        "cases_offer_identity_inconsistent": int((f.has_offer_records & ~f.offer_identity_consistent).sum()),
        "timestamp_invalid_events": int((f.event_count - f.valid_timestamps).sum()),
        "cases_with_timestamp_ties": int(f.timestamp_ties.gt(0).sum()),
        "timestamp_tie_surplus_events": int(f.timestamp_ties.sum()),
        "cases_with_timestamp_descents": int(f.source_order_timestamp_descents.gt(0).sum()),
        "csv_representation": {
            "checked": bool(reader),
            "rows_compared_in_order": csvrows,
            "extra_csv_rows": extra,
            "field_mismatch_counts": dict(csv_mismatch),
            "first_mismatches": examples,
            "columns": fields,
            "normalisation": "UTC instants; finite numeric equivalents; boolean lowercase; blank/missing equality; all other strings exact; no event removal, reordering or deduplication",
            "equivalent": bool(reader) and not csv_mismatch and extra == 0,
        },
    }
    return f, schema


def group_keys(f, cols):
    return f[cols].fillna("<unknown>").astype(str).agg(" | ".join, axis=1).rename("group")


def top(g, k, cutoff, column="PI"):
    return g.loc[g[column] > cutoff].sort_index().sort_values([column, "n_cases"], ascending=False, kind="stable").head(k)


def overlap(a, b):
    x = set(a.index)
    y = set(b.index)
    return len(x & y) / len(x | y) if x | y else None


def evaluate(f, schema, cfg, norm, out, api):
    out.mkdir(parents=True, exist_ok=True)
    V, A, R = assessments(f, cfg)
    layers = {k: v["layer"] for k, v in cfg["criteria"].items()}
    keys = group_keys(f, cfg["primary_group"])
    other = group_keys(f, cfg["secondary_group"])
    f.to_parquet(out / "case_features.parquet")
    V.to_parquet(out / "assessments.parquet")
    A.to_parquet(out / "in_scope.parquet")
    R.to_parquet(out / "evidence_states.parquet")
    records = []
    ver = []
    scores = pd.DataFrame(index=f.index)
    view_results = {}
    signed_all = {}
    sensitivity = []
    for view, w in cfg["views"].items():
        r = api.score_assessments(V, A, pd.Series(w, dtype=float), layers, mode=norm["mode"])
        view_results[view] = r
        g = api.group_priorities(r["score"], keys)
        co = api.coco_same_target(r["penalty"], keys)
        signed = api.signed_components(r["penalties"], keys)
        signed_all[view] = signed
        gl = api.coverage_diagnostics(r, keys)["groups"]
        g = g.join(gl[["coverage_mean", "coverage_min"]])
        g.to_csv(out / f"groups_{view}.csv")
        api.group_priorities(r["score"], other).to_csv(out / f"secondary_groups_{view}.csv")
        co.to_csv(out / f"coco_{view}.csv")
        signed.to_csv(out / f"signed_criteria_{view}.csv")
        signed.T.groupby(pd.Series(layers)).sum().T.to_csv(out / f"signed_layers_{view}.csv")
        t = top(g, norm["top_k"], norm["positive_cutoff"])
        t.to_csv(out / f"top10_{view}.csv")
        for name in ("score", "penalty", "coverage", "in_scope_count", "evaluated_count"):
            scores[f"{view}__{name}"] = r[name]
        for name in ("penalty_min", "penalty_max"):
            scores[f"{view}__{name}"] = r["flat_completion_bounds"][name]
        r["penalties"].to_parquet(out / f"case_components_{view}.parquet")
        residuals = {
            "case_additivity": float((r["penalties"].sum(axis=1, min_count=1) - r["penalty"]).abs().max()),
            "signed_PI_reconstruction": float((signed.sum(axis=1).clip(lower=0) - g.PI.reindex(signed.index)).abs().max()),
            "same_target_coco_identity": float((co.D * co.CoCo.clip(lower=0) - g.PI.reindex(co.index)).abs().max()),
            "flat_bound_width": float(
                (r["flat_completion_bounds"].eval("penalty_max-penalty_min") - (1 - r["coverage"])).abs().max()
            ),
        }
        if any(x > 1e-9 for x in residuals.values()):
            raise AssertionError(residuals)
        ver.append(
            {
                "view": view,
                "residual_units": {
                    "case_additivity": "case penalty",
                    "signed_PI_reconstruction": "penalty times application count",
                    "same_target_coco_identity": "penalty times application count",
                    "flat_bound_width": "case penalty",
                },
                "residuals": residuals,
                "passed": True,
            }
        )
        records.append(
            {
                "view": view,
                "unit": "application trace",
                "cases": len(f),
                "events": int(f.event_count.sum()),
                "scored": int(r["score"].notna().sum()),
                "groups": len(g),
                "mean_penalty": float(r["penalty"].mean()),
                "mean_coverage": float(r["coverage"].mean()),
                "positive_groups": int(g.PI.gt(norm["positive_cutoff"]).sum()),
                "top_k_requested": norm["top_k"],
                "top_k_selected": len(t),
                "lead_group": str(t.index[0]) if len(t) else None,
                "lead_PI": float(t.PI.iloc[0]) if len(t) else None,
            }
        )
        for scenario, params in [("gamma20", {"gamma": 20}), ("support50", {"min_cases": 50})]:
            gg = api.group_priorities(r["score"], keys, **params)
            col = "stable_PI" if scenario == "gamma20" else "PI"
            tt = top(gg, norm["top_k"], norm["positive_cutoff"], col)
            sensitivity.append(
                {
                    "view": view,
                    "scenario": scenario,
                    "same_case_target_and_reference": True,
                    "scored_cases": int(r["score"].notna().sum()),
                    "eligible_groups": len(gg),
                    "eligible_scored_cases": int(gg.n_cases.sum()),
                    "positive_candidates": int(gg[col].gt(norm["positive_cutoff"]).sum()),
                    "selected_k": len(tt),
                    "top10_jaccard": overlap(t, tt),
                    "mean_coverage": float(r["coverage"].mean()),
                    "mean_penalty": float(r["penalty"].mean()),
                }
            )
        for scenario, vv, mode in [("layer_balanced", V, "layer_balanced")] + [
            (f"span_threshold_{d}", assessments(f, cfg, d)[0], "flat") for d in norm["sensitivities"]["span_threshold_days"]
        ]:
            rr = api.score_assessments(vv, A, pd.Series(w, dtype=float), layers, mode=mode)
            gg = api.group_priorities(rr["score"], keys)
            tt = top(gg, norm["top_k"], norm["positive_cutoff"])
            sensitivity.append(
                {
                    "view": view,
                    "scenario": scenario,
                    "same_case_target_and_reference": False,
                    "scored_cases": int(rr["score"].notna().sum()),
                    "eligible_groups": len(gg),
                    "eligible_scored_cases": int(gg.n_cases.sum()),
                    "positive_candidates": int(gg.PI.gt(norm["positive_cutoff"]).sum()),
                    "selected_k": len(tt),
                    "top10_jaccard": overlap(t, tt),
                    "mean_coverage": float(rr["coverage"].mean()),
                    "mean_penalty": float(rr["penalty"].mean()),
                }
            )
    scores.to_parquet(out / "case_scores.parquet")
    pd.DataFrame(sensitivity).to_csv(out / "sensitivity.csv", index=False)
    states = pd.DataFrame(
        [
            {
                "criterion": c,
                "in_scope": int(A[c].sum()),
                "evaluated": int(V[c].notna().sum()),
                "positive": int(V[c].gt(0).sum()),
                "zero": int(V[c].eq(0).sum()),
                "unevaluable": int((A[c] & V[c].isna()).sum()),
                "out_of_scope": int((~A[c]).sum()),
            }
            for c in V
        ]
    )
    states.to_csv(out / "criterion_states.csv", index=False)
    comparisons = []
    for view in list(cfg["views"])[1:]:
        qa = view_results["Balanced"]["penalty"]
        qb = view_results[view]["penalty"]
        ga = api.group_priorities(view_results["Balanced"]["score"], keys)
        gb = api.group_priorities(view_results[view]["score"], keys)
        comparisons.append(
            {
                "base_view": "Balanced",
                "other_view": view,
                "same_population": bool(qa.notna().equals(qb.notna())),
                "max_case_penalty_difference": float((qa - qb).abs().max()),
                "spearman_case_penalty": float(qa.corr(qb, method="spearman")),
                "top10_positive_jaccard": overlap(top(ga, 10, 1e-12), top(gb, 10, 1e-12)),
            }
        )
    pd.DataFrame(comparisons).to_csv(out / "view_comparisons.csv", index=False)
    # Deterministic examples: positive severity closest to positive median, then lexical ID;
    # zero, missing, and out-of-scope first lexical ID. No representativeness claim.
    selections = []
    for c in V:
        for state, mask in [
            ("positive", V[c].gt(0)),
            ("zero", V[c].eq(0)),
            ("unevaluable", A[c] & V[c].isna()),
            ("out_of_scope", ~A[c]),
        ]:
            ix = V.index[mask]
            if len(ix):
                median = float(V.loc[ix, c].median()) if state == "positive" else 0
                pick = sorted(ix, key=lambda i: (abs(V.loc[i, c] - median) if state == "positive" else 0, str(i)))[0]
                selections.append(
                    {
                        "criterion": c,
                        "state": state,
                        "case_id": pick,
                        "severity": None if pd.isna(V.loc[pick, c]) else float(V.loc[pick, c]),
                        "raw_feature": None
                        if pd.isna(f.loc[pick, cfg["criteria"][c]["feature"]])
                        else float(f.loc[pick, cfg["criteria"][c]["feature"]]),
                    }
                )
    dump(out / "witness_selection.json", selections)
    return records, ver, selections


def witness_audit(path, family, norm, cfg, features, selections, out):
    ids = {s["case_id"] for s in selections}
    rows = []
    checks = []
    for tr, ev in iter_traces(path):
        cid = tr["concept:name"]
        if cid not in ids:
            continue
        for p, e in enumerate(ev, 1):
            rows.append(
                {
                    "case_id": cid,
                    "source_event_position_1based": p,
                    "raw_link": f"{path.name}#trace={cid}&event={p}",
                    "attributes_json": json.dumps(e, sort_keys=True),
                    **{
                        k: e.get(k)
                        for k in ("concept:name", "lifecycle:transition", "time:timestamp", "org:resource", "EventID", "OfferID")
                    },
                }
            )
        # Independent expected feature calculations from raw witness events.
        starts = [
            e for e in ev if e.get("concept:name", "").startswith("W_") and e.get("lifecycle:transition", "").upper() == "START"
        ]
        expected_repeat = len(starts) - len({e["concept:name"] for e in starts})
        expected_breadth = (
            len({e.get("org:resource") for e in starts}) if all(present(e.get("org:resource"), norm) for e in starts) else np.nan
        )
        stamps = [instant(e.get("time:timestamp")) for e in ev]
        expected_span = (
            (max(stamps) - min(stamps)).total_seconds() / 86400
            if all(d is not None for d in stamps) and len(stamps) > 1
            else np.nan
        )
        offer_rows = [
            e
            for e in ev
            if e.get("concept:name") == cfg["offer_creation_activity"] and e.get("lifecycle:transition", "").upper() == "COMPLETE"
        ]
        if family == "bpic2012":
            expected_offer = len(offer_rows) if offer_rows else np.nan
        else:
            created_ids = [e.get("EventID") for e in offer_rows]
            confirmed_ids = [
                e.get("OfferID")
                for e in ev
                if e.get("concept:name") == "O_Created" and e.get("lifecycle:transition", "").upper() == "COMPLETE"
            ]
            used_ids = [
                e.get("OfferID")
                for e in ev
                if e.get("concept:name", "").startswith("O_") and e.get("concept:name") != "O_Create Offer"
            ]
            consistent = (
                bool(created_ids)
                and all(present(x, norm) for x in created_ids + confirmed_ids + used_ids)
                and Counter(created_ids) == Counter(confirmed_ids)
                and all(n == 1 for n in Counter(created_ids).values())
                and all(i in created_ids for i in used_ids)
            )
            expected_offer = len(created_ids) if consistent else np.nan
        expected_incomplete = len(
            [e for e in ev if e.get("concept:name") == "A_Incomplete" and e.get("lifecycle:transition", "").upper() == "COMPLETE"]
        )
        checks.extend(
            [
                {
                    "case_id": cid,
                    "check": name,
                    "passed": bool((pd.isna(actual) and pd.isna(expected)) or np.isclose(actual, expected)),
                }
                for name, actual, expected in [
                    ("repeated_START_records", features.loc[cid, "repeated_work_starts"], expected_repeat),
                    ("START_resource_breadth", features.loc[cid, "work_start_resource_breadth"], expected_breadth),
                    ("observed_span_days", features.loc[cid, "observed_span_days"], expected_span),
                    ("offer_multiplicity", features.loc[cid, "offer_multiplicity"], expected_offer),
                    ("incomplete_state_records", features.loc[cid, "incomplete_state_records"], expected_incomplete),
                ]
            ]
        )
        for s in selections:
            if s["case_id"] != cid:
                continue
            c = cfg["criteria"][s["criterion"]]
            x = s["raw_feature"]
            v = None if x is None or s["state"] == "out_of_scope" else min(max((x - c["threshold"]) / c["width"], 0), 1)
            checks.append({"case_id": cid, "check": f"{s['criterion']}_{s['state']}_severity", "passed": v == s["severity"]})
    if {r["case_id"] for r in rows} != ids or not all(c["passed"] for c in checks):
        raise AssertionError("Raw witness reconstruction failed")
    pd.DataFrame(rows).to_parquet(out / "witness_events.parquet")
    pd.DataFrame(checks).to_csv(out / "source_witness_checks.csv", index=False)
    return {
        "selected_unique_cases": len(ids),
        "raw_events": len(rows),
        "assertions": len(checks),
        "assertion_unit": "raw-linked case feature or selected criterion severity equality",
        "passed": True,
    }


def fixture_checks(norm, api, out):
    import tempfile

    results = []

    def check(name, test):
        if not bool(test):
            raise AssertionError(name)
        results.append({"check": name, "passed": True})

    def e(name, life="complete", day=1, resource="R1", **extras):
        return {
            "concept:name": name,
            "lifecycle:transition": life,
            "time:timestamp": f"2016-01-{day:02d}T00:00:00Z",
            "org:resource": resource,
            **extras,
        }

    def f(events, fam="bpic2017"):
        ff = pd.DataFrame(
            [feature_row({"concept:name": "constructed", "RequestedAmount": "1000", "AMOUNT_REQ": "1000"}, events, fam, norm)]
        ).set_index("case_id")
        vv, aa, _ = assessments(ff, norm["families"][fam])
        return ff, vv, aa

    ff, v, a = f([e("W_Task", "schedule"), e("W_Task", "start"), e("W_Task", "complete")])
    check("lifecycle_triple_is_one_start", ff.work_start_records.iloc[0] == 1)
    check("lifecycle_triple_not_repeated_start", ff.repeated_work_starts.iloc[0] == 0)
    ff, v, a = f([e("W_Task", "start"), e("W_Task", "resume"), e("W_Task", "suspend"), e("W_Task", "start")])
    check("resume_not_start", ff.work_start_records.iloc[0] == 2)
    check("two_starts_one_repetition", ff.repeated_work_starts.iloc[0] == 1)
    ff, v, a = f([e("W_Task", "start", resource=None), e("W_Task", "complete")])
    check("missing_resource_is_unknown", pd.isna(v.C03.iloc[0]) and a.C03.iloc[0])
    r = api.score_assessments(
        v,
        a,
        pd.Series(norm["families"]["bpic2017"]["views"]["Balanced"], dtype=float),
        {k: c["layer"] for k, c in norm["families"]["bpic2017"]["criteria"].items()},
        mode="flat",
    )
    check("missing_resource_lowers_coverage", r["coverage"].iloc[0] < 1)
    events = [
        e("O_Create Offer", EventID="offerA"),
        e("O_Created", OfferID="offerA"),
        e("O_Create Offer", EventID="offerB"),
        e("O_Created", OfferID="offerB"),
        e("O_Accepted", OfferID="offerA"),
    ]
    ff, v, a = f(events)
    check("five_offer_events_two_offers", ff.offer_multiplicity.iloc[0] == 2)
    check("offer_threshold_contract", np.isclose(v.C04.iloc[0], 1 / 3))
    events[1]["OfferID"] = "other"
    ff, v, a = f(events)
    check("unreconciled_offer_ids_unknown", pd.isna(v.C04.iloc[0]) and a.C04.iloc[0])
    ff, v, a = f([e("A_Cancelled"), e("A_Denied")])
    check("legitimate_outcomes_unpenalised", v.drop(columns="C04").fillna(99).to_numpy().sum() == 0)
    check("no_offer_out_of_scope", not a.C04.iloc[0] and pd.isna(v.C04.iloc[0]))
    ev = [e("A_Create Application"), e("A_Concept")]
    ev[1]["time:timestamp"] = "bad"
    ff, v, a = f(ev)
    check("bad_timestamp_span_unknown", pd.isna(v.C01.iloc[0]))
    ev[1]["time:timestamp"] = "2016-01-02T00:00:00"
    ff, v, a = f(ev)
    check("timezone_absence_not_silently_UTC", pd.isna(v.C01.iloc[0]))
    ff, v, a = f([e("O_CREATED"), e("O_CREATED")], "bpic2012")
    check("2012_creation_records_not_identity", ff.offer_multiplicity.iloc[0] == 2)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "nested.xes"
        p.write_text(
            '<log><trace><string key="concept:name" value="case"/><event><string key="concept:name" value="A"><string key="concept:name" value="child"/></string></event></trace></log>'
        )
        _tr, ev = next(iter_traces(p))
        check("nested_attribute_not_parent_override", ev[0]["concept:name"] == "A")
        check("nested_attribute_retained", bool(ev[0].get("__nested_attributes_xml")))
        p.write_text('<!DOCTYPE log [<!ENTITY bad "x">]><log/>')
        try:
            list(iter_traces(p))
            rejected = False
        except Exception as ex:
            rejected = type(ex).__name__ == "DTDForbidden"
        check("DTD_rejected", rejected)
    report = {
        "synthetic_feature_scenarios": 9,
        "parser_scenarios": 2,
        "assertions": len(results),
        "assertion_unit": "named deterministic constructed contract assertion",
        "checks": results,
        "passed": True,
        "limits": "Operational consistency of author-defined contracts, not domain preference validity.",
    }
    dump(out / "contract_checks.json", report)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    normfile = HERE / "illustrative_norm.json"
    norm = json.loads(normfile.read_text())
    core = args.project_root / "code/wise_reference.py"
    if sha(core) != EXPECTED_CORE:
        raise RuntimeError("Unexpected reference code revision; re-audit before changing pinned hash")
    sys.path.insert(0, str(core.parent))
    import wise_reference as api

    files = {}
    paths = {}
    zipaudit = {}
    for year in ("2012", "2017"):
        folder = args.data_root / f"BPI Challenge {year}_1_all"
        path = next(folder.glob("*.xes.gz"))
        paths[f"bpic{year}"] = path
        for p in (path, folder / "DATA.xml", args.data_root / f"BPI Challenge {year}_1_all.zip"):
            files[str(p.relative_to(args.data_root))] = {"sha256": sha(p), "bytes": p.stat().st_size}
        za = []
        with zipfile.ZipFile(args.data_root / f"BPI Challenge {year}_1_all.zip") as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                h = hashlib.sha256()
                with z.open(info) as stream:
                    for b in iter(lambda: stream.read(2**20), b""):
                        h.update(b)
                target = folder / Path(info.filename).name
                member_matches = target.is_file() and h.hexdigest() == sha(target)
                za.append(
                    {
                        "member": info.filename,
                        "bytes": info.file_size,
                        "sha256": h.hexdigest(),
                        "matches_extracted_file": member_matches,
                    }
                )
        zipaudit[year] = za
    csvpath = args.data_root / "bpi_2017_cleaned.csv"
    files[csvpath.name] = {"sha256": sha(csvpath), "bytes": csvpath.stat().st_size}
    summaries = []
    schemas = {}
    verification = {}
    witnesses = {}
    fixtures = fixture_checks(norm, api, out)
    for family, path in paths.items():
        print("Parsing", family, flush=True)
        f, schema = parse_family(path, family, norm, csvpath if family == "bpic2017" else None)
        metadata = ET.parse(path.parent / "DATA.xml", forbid_dtd=True).getroot()
        declared = {"cases": int(metadata.findtext("number_of_traces")), "events": int(metadata.findtext("number_of_events"))}
        if any(schema[k] != v for k, v in declared.items()):
            raise AssertionError("Parsed counts differ from supplied DATA.xml")
        schema["supplied_metadata"] = {
            "doi": metadata.findtext("doi"),
            "description": metadata.findtext("description"),
            "declared_counts": declared,
            "counts_match": True,
        }
        schemas[family] = schema
        dest = out / family
        cfg = norm["families"][family]
        rec, ver, selected = evaluate(f, schema, cfg, norm, dest, api)
        for r in rec:
            r["dataset"] = family
        summaries.extend(rec)
        verification[family] = ver
        witnesses[family] = witness_audit(path, family, norm, cfg, f, selected, dest)
        print(family, len(f), schema["events"], "finished", flush=True)
    pd.DataFrame(summaries).to_csv(out / "view_summary.csv", index=False)
    dump(out / "schema_audit.json", schemas)
    dump(out / "zip_representation_audit.json", zipaudit)
    verify = {
        "passed": True,
        "numeric_checks": verification,
        "numeric_check_unit": "four maximum absolute reconstruction residuals per dataset-view; not independent business observations",
        "source_witnesses": witnesses,
        "constructed_contracts": {
            "assertions": fixtures["assertions"],
            "synthetic_feature_scenarios": fixtures["synthetic_feature_scenarios"],
            "parser_scenarios": fixtures["parser_scenarios"],
            "passed": True,
        },
        "cleaned_2017_equivalent": schemas["bpic2017"]["csv_representation"]["equivalent"],
        "zip_members_match": all(x["matches_extracted_file"] for a in zipaudit.values() for x in a),
    }
    if not verify["cleaned_2017_equivalent"] or not verify["zip_members_match"]:
        raise AssertionError("Representation parity failed; inspect schema audit")
    dump(out / "verification.json", verify)
    dump(
        out / "manifest.json",
        {
            "status": "passed",
            "reference_sha256": sha(core),
            "reference_path": str(core),
            "config_sha256": sha(normfile),
            "script_sha256": sha(Path(__file__)),
            "raw_inputs": files,
            "norm_id": norm["norm_id"],
            "dataset_scope": "Complete independent application logs; never pooled; CSV representation check only",
            "mode": norm["mode"],
            "tie_policy": norm["rank_ties"],
            "positive_cutoff": norm["positive_cutoff"],
            "source_order": "retained in raw witness links; no order-dependent measures",
            "dependencies": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__},
        },
    )
    print("PASSED", out, flush=True)


if __name__ == "__main__":
    main()
