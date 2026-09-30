#!/usr/bin/env python3
"""Read-only raw dataset audit and bounded WISE transfers; frozen author norms.
No source writes, no pickles, no graph repair, no old derived outputs as input.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import argparse
import ast
import collections
import hashlib
import json
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from defusedxml import ElementTree as ET


def sha(p, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(p, v):
    def enc(x):
        if isinstance(x, (np.integer, np.floating, np.bool_)):
            return x.item()
        if isinstance(x, (Path, pd.Timestamp)):
            return str(x)
        raise TypeError(type(x).__name__)

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, indent=2, default=enc, allow_nan=False) + "\n")


def readcsv(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False).apply(lambda c: c.str.strip())


def sat(z, t, w):
    return ((z - t) / w).clip(0, 1)


def dt(x):
    return pd.to_datetime(x, format="mixed", errors="coerce", utc=True)


def scalar_time(x):
    return datetime.fromisoformat(x.replace("Z", "+00:00"))


def assert_near(x, y, tol=1e-10):
    assert np.isclose(x, y, atol=tol, rtol=0), (x, y)


def common_run(name, f, V, A, reason, raw, norm, out, wr, sensitivities):
    out.mkdir(parents=True, exist_ok=True)
    f.to_csv(out / "case_features.csv")
    long = []
    for cid in V:
        x = pd.DataFrame(
            {
                "case_id": f.index,
                "criterion": cid,
                "in_scope": A[cid].values,
                "state": reason[cid].values,
                "raw": raw[cid].values,
                "severity": V[cid].values,
            }
        )
        long.append(x)
    pd.concat(long).to_csv(out / "criterion_assessments.csv", index=False)
    profiles = []
    for cid in V:
        for state, n in reason[cid].value_counts().items():
            profiles.append({"criterion": cid, "state": state, "cases": int(n)})
    pd.DataFrame(profiles).to_csv(out / "criterion_states.csv", index=False)
    layers = {c["id"]: c["layer"] for c in norm["criteria"]}
    scores = f[["group"]].copy()
    checks = []
    summaries = []
    tables = {}
    for view, ww in norm["views"].items():
        w = pd.Series(ww, index=V.columns, dtype=float)
        s = wr.score_assessments(V, A, w, layers, mode="flat")
        table = wr.group_priorities(s["score"], f.group)
        signed = wr.signed_components(s["penalties"], f.group)
        cc = wr.coco_same_target(s["penalty"], f.group)
        errors = {
            "effective_weights": float((s["effective_weights"].sum(axis=1)[s["score"].notna()] - 1).abs().max()),
            "penalty_components": float((s["penalties"].sum(axis=1, min_count=1) - s["penalty"]).abs().max()),
            "signed_priority": float((signed.sum(axis=1).clip(lower=0) - table.PI).abs().max()),
            "coco_identity": float((cc.CoCo.clip(lower=0) * cc.D - table.PI).abs().max())
            if cc.normalisation_defined.all()
            else None,
        }
        assert all(v is None or v < 1e-9 for v in errors.values())
        checks.append({"view": view, "passed": True, "max_residuals": errors})
        table.to_csv(out / f"groups_{view}.csv")
        signed.to_csv(out / f"signed_components_{view}.csv")
        cc.to_csv(out / f"coco_{view}.csv")
        for k in ["score", "penalty", "coverage", "in_scope_count", "evaluated_count"]:
            scores[k + "__" + view] = s[k]
        summary = {
            "dataset": name,
            "view": view,
            "cases": len(f),
            "scored": int(s["score"].notna().sum()),
            "mean_penalty": float(s["penalty"].mean()),
            "mean_coverage": float(s["coverage"].mean()),
            "groups": len(table),
            "positive_groups": int(table.PI.gt(1e-12).sum()),
            "total_priority": float(table.PI.sum()),
            "top_group": str(table.index[0]),
            "top_group_PI": float(table.PI.iloc[0]),
        }
        summaries.append(summary)
        tables[view] = table
        if view == "Balanced":
            primary = s
    scores.to_csv(out / "case_scores.csv")
    sr = []
    base = tables["Balanced"]
    k = min(5, int(base.PI.gt(1e-12).sum()))
    base_top = set(base.loc[base.PI.gt(1e-12)].head(k).index)
    for label, ff, vv, aa, mode in sensitivities:
        sc = wr.score_assessments(
            vv,
            aa,
            pd.Series(norm["views"]["Balanced"], index=vv.columns, dtype=float),
            layers,
            mode=mode,
        )
        tab = wr.group_priorities(sc["score"], ff.group)
        top = set(tab.loc[tab.PI.gt(1e-12)].head(k).index)
        union = base_top | top
        sr.append(
            {
                "sensitivity": label,
                "cases": len(ff),
                "scored": int(sc["score"].notna().sum()),
                "mean_penalty": float(sc["penalty"].mean()),
                "mean_coverage": float(sc["coverage"].mean()),
                "top_k_requested": k,
                "selected_positive_groups": len(top),
                "primary_positive_groups_selected": len(base_top),
                "intersection": len(top & base_top),
                "jaccard": len(top & base_top) / len(union) if union else None,
                "top_group": str(tab.index[0]),
                "top_PI": float(tab.PI.iloc[0]),
            }
        )
        tab.to_csv(out / f"sensitivity_{label}.csv")
    pd.DataFrame(sr).to_csv(out / "sensitivity.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "view_summary.csv", index=False)
    dump(out / "checks.json", checks)
    return summaries, primary


def audit_ocel(folder, out):
    rawpaths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix in [".json", ".xml", ".sqlite", ".csv"])
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p), "md5": sha(p, "md5")} for p in rawpaths}
    d = json.loads((folder / "ocel2-p2p.json").read_text())
    obs = {x["id"]: x for x in d["objects"]}
    evs = {x["id"]: x for x in d["events"]}
    objects = {(o["id"], o["type"]) for o in d["objects"]}
    events = {(e["id"], e["type"], e["time"]) for e in d["events"]}
    parsed_times = pd.to_datetime([e["time"] for e in d["events"]], format="ISO8601", errors="coerce", utc=True)
    assert parsed_times.notna().all(), "Invalid OCEL event timestamp"
    assert all(e["time"].endswith("Z") for e in d["events"]), "This adapter requires explicit UTC source times"
    er = [(e["id"], r["objectId"], r["qualifier"]) for e in d["events"] for r in e["relationships"]]
    oo = [(o["id"], r["objectId"], r["qualifier"]) for o in d["objects"] for r in o["relationships"]]
    bad_er = [r for r in er if r[1] not in obs]
    bad_oo = [r for r in oo if r[1] not in obs]
    attrs = [(o["id"], a["name"], str(a["value"]), a["time"]) for o in d["objects"] for a in o["attributes"]]
    audit = {
        "files": files,
        "events": len(evs),
        "objects": len(obs),
        "event_types": dict(collections.Counter(e["type"] for e in d["events"])),
        "object_types": dict(collections.Counter(o["type"] for o in d["objects"])),
        "event_object_rows": len(er),
        "object_object_rows": len(oo),
        "duplicate_event_ids": len(d["events"]) - len(evs),
        "duplicate_object_ids": len(d["objects"]) - len(obs),
        "duplicate_event_object_triples": len(er) - len(set(er)),
        "duplicate_object_object_triples": len(oo) - len(set(oo)),
        "dangling_event_object_rows": len(bad_er),
        "dangling_object_object_rows": len(bad_oo),
        "distinct_dangling_object_ids": len({r[1] for r in bad_oo}),
        "object_attribute_records": len(attrs),
        "invalid_event_timestamps": int(parsed_times.isna().sum()),
        "all_event_timestamps_explicit_UTC": True,
        "time_min": min(e["time"] for e in d["events"]),
        "time_max": max(e["time"] for e in d["events"]),
        "checks": {},
    }
    assert not bad_er
    assert not audit["duplicate_event_ids"]
    assert not audit["duplicate_object_ids"]
    checks = audit["checks"]
    for name in ["ocel2-p2p.xml", "ocel2-export.xml"]:
        root = ET.parse(folder / name).getroot()
        xo = {(o.attrib["id"], o.attrib["type"]) for o in root.find("objects")}
        xe = {(e.attrib["id"], e.attrib["type"], e.attrib["time"]) for e in root.find("events")}
        xer = [
            (e.attrib["id"], r.attrib["object-id"], r.attrib["qualifier"]) for e in root.find("events") for r in e.find("objects")
        ]
        xoo = [
            (o.attrib["id"], r.attrib["object-id"], r.attrib["qualifier"])
            for o in root.find("objects")
            for r in o.find("objects")
        ]
        checks[name] = {
            "object_identity_types_equal": xo == objects,
            "event_identity_type_time_equal": xe == events,
            "event_object_multisets_equal": collections.Counter(xer) == collections.Counter(er),
            "object_object_multisets_equal": collections.Counter(xoo) == collections.Counter(oo),
            "dangling_object_object_rows": sum(r[1] not in obs for r in xoo),
        }
    con = sqlite3.connect("file:" + str(folder / "ocel2-p2p.sqlite") + "?mode=ro", uri=True)
    so = set(con.execute("select ocel_id,ocel_type from object"))
    se = set(con.execute("select ocel_id,ocel_type from event"))
    ser = list(con.execute("select ocel_event_id,ocel_object_id,ocel_qualifier from event_object"))
    soo = list(con.execute("select ocel_source_id,ocel_target_id,ocel_qualifier from object_object"))
    checks["sqlite"] = {
        "object_identity_types_equal": so == objects,
        "event_identity_types_equal": se == {(e[0], e[1]) for e in events},
        "event_object_multisets_equal": collections.Counter(ser) == collections.Counter(er),
        "object_object_multisets_equal": collections.Counter(soo) == collections.Counter(oo),
        "dangling_object_object_rows": sum(r[1] not in obs for r in soo),
    }
    sql_ev = []
    for typ, mapping in con.execute("select ocel_type,ocel_type_map from event_map_type"):
        table = "event_" + mapping
        quoted = '"' + table.replace('"', '""') + '"'
        for row in con.execute("select ocel_id,ocel_time from " + quoted):
            sql_ev.append((row[0], typ, pd.Timestamp(row[1]).isoformat()))
    checks["sqlite"]["event_times_equal"] = {(i, t, pd.Timestamp(tm).isoformat()) for i, t, tm in events} == set(sql_ev)
    con.close()
    ec = readcsv(folder / "ocel2.ocel.events.csv")
    oc = readcsv(folder / "ocel2.ocel.objects.csv")
    rc = readcsv(folder / "ocel2.ocel.relationships.events-objects.csv")
    ac = readcsv(folder / "ocel2.ocel.objects.attributes.csv")
    checks["csv_tables"] = {
        "events_equal": set(ec[["id", "type", "time"]].itertuples(index=False, name=None)) == events,
        "objects_equal": set(oc[["id", "type"]].itertuples(index=False, name=None)) == objects,
        "event_objects_equal": collections.Counter(rc[["eventId", "objectId", "qualifier"]].itertuples(index=False, name=None))
        == collections.Counter(er),
        "object_attributes_equal": collections.Counter(ac[["id", "name", "value", "time"]].itertuples(index=False, name=None))
        == collections.Counter(attrs),
        "object_object_table_present": False,
    }
    flat = readcsv(folder / "ocel2-p2p.csv")
    fp = []
    for row in flat.to_dict("records"):
        for c, v in row.items():
            if c.startswith("ocel:type:") and v:
                for oid in ast.literal_eval(v):
                    fp.append((row["ocel:eid"], oid))
    checks["flat_csv"] = {
        "event_identity_activity_time_equal": {
            (
                r["ocel:eid"],
                r["ocel:activity"],
                pd.Timestamp(r["ocel:timestamp"]).isoformat(),
            )
            for r in flat.to_dict("records")
        }
        == {(i, t, pd.Timestamp(tm).isoformat()) for i, t, tm in events},
        "event_object_pairs_equal": set(fp) == {(e, o) for e, o, q in er},
        "qualifiers_and_object_history_absent": True,
    }
    audit["format_scope"] = (
        "Identity/type/time and relationship multisets checked. XML dynamic attribute timestamp interpretation and all SQLite dynamic attribute records not asserted equal. CSV tables omit O2O; flat CSV loses qualifiers and object history."
    )
    published = {
        "ocel2-p2p.json": "26a8a29333c191d239a7b0fdc44b818f",
        "ocel2-p2p.sqlite": "1a4238260019939239488b0b4befb515",
        "ocel2-p2p.xml": "cb50e3ef18bda7f843dad00f4ec42123",
        "ocel2-export.xml": "cb50e3ef18bda7f843dad00f4ec42123",
    }
    audit["zenodo_md5_matches"] = {n: files[n]["md5"] == h for n, h in published.items()}
    audit["provenance"] = {
        "source": "https://zenodo.org/records/8412920",
        "doi": "10.5281/zenodo.8412920",
        "kind": "simulated SAP-style P2P log",
        "accessed": "2026-09-30",
        "ground_truth_labels": "No per-case fault labels established in this audit",
    }
    dump(out / "schema_audit.json", audit)
    pd.DataFrame(bad_oo, columns=["source_object", "missing_target_object", "qualifier"]).to_csv(
        out / "dangling_object_relations.csv", index=False
    )
    dump(
        out / "integrity_witness.json",
        {
            "missing_target_present_in_object_tables": False,
            "witness_triples": bad_oo[:6],
            "same_defect_in_xml_and_sqlite": all(
                checks[k]["dangling_object_object_rows"] == len(bad_oo) for k in ["ocel2-p2p.xml", "ocel2-export.xml", "sqlite"]
            ),
            "not_used_by_direct_projection": True,
        },
    )
    return d, obs, evs, audit


def run_ocel(folder, norm, out, wr):
    out.mkdir(parents=True, exist_ok=True)
    d, obs, evs, audit = audit_ocel(folder, out)
    poids = sorted(o for o in obs if obs[o]["type"] == "purchase_order")
    direct = collections.defaultdict(set)
    event_po = {}
    payment_events = collections.defaultdict(set)
    for eid, e in evs.items():
        ids = {r["objectId"] for r in e["relationships"]}
        pos = {x for x in ids if obs[x]["type"] == "purchase_order"}
        event_po[eid] = pos
        for po in pos:
            direct[po].add(eid)
        if e["type"] == "Execute Payment":
            for oid in ids:
                if obs[oid]["type"] == "payment":
                    payment_events[oid].add(eid)
    rows = []
    for po in poids:
        events = [evs[e] for e in direct[po]]
        row = {"case_id": po, "n_event_associations": len(events)}
        init = {a["name"]: a["value"] for a in obs[po]["attributes"] if a["time"].startswith("1970-01-01")}
        row["vendor_initial"] = init.get("Vendor (EKKO-LIFNR)", "(missing)")
        row["purchasing_group_initial"] = init.get("Purchasing Group (EKKO-EKGRP)", "(missing)")
        row["group"] = row["vendor_initial"]
        for key, act in {
            "create": "Create Purchase Order",
            "approve": "Approve Purchase Order",
            "goods": "Create Goods Receipt",
            "pay": "Execute Payment",
        }.items():
            ee = [e for e in events if e["type"] == act]
            row["n_" + key] = len(ee)
            row["first_" + key] = min((e["time"] for e in ee), default=None)
        ps = {
            r["objectId"]
            for e in events
            if e["type"] == "Execute Payment"
            for r in e["relationships"]
            if obs[r["objectId"]]["type"] == "payment"
        }
        row["max_payment_executions"] = max(
            (
                sum(e["type"] == "Execute Payment" and any(r["objectId"] == p for r in e["relationships"]) for e in events)
                for p in ps
            ),
            default=np.nan,
        )
        row["shared_payment_associations"] = sum(e["type"] == "Execute Payment" and len(event_po[e["id"]]) > 1 for e in events)
        rows.append(row)
    f = pd.DataFrame(rows).set_index("case_id")
    for c in ["create", "approve", "goods", "pay"]:
        f["first_" + c] = dt(f["first_" + c])
    f["approval_days"] = (f.first_approve - f.first_create).dt.total_seconds() / 86400
    f["po_payment_days"] = (f.first_pay - f.first_create).dt.total_seconds() / 86400
    ids = [x["id"] for x in norm["criteria"]]
    A = pd.DataFrame(True, index=f.index, columns=ids)
    V = pd.DataFrame(np.nan, index=f.index, columns=ids)
    raw = V.copy()
    reason = pd.DataFrame("evaluated", index=f.index, columns=ids)
    raw["O01"] = f.approval_days
    raw["O02"] = (f.first_goods - f.first_approve).dt.total_seconds() / 86400
    raw["O03"] = f.max_payment_executions
    raw["O04"] = f.po_payment_days
    rules = {c["id"]: c for c in norm["criteria"]}
    for cid in ["O01", "O03", "O04"]:
        V[cid] = sat(raw[cid], rules[cid]["threshold"], rules[cid]["width"])
    V.O02 = (raw.O02 < 0).astype(float)
    for c in ids:
        bad = raw[c].isna() | (raw[c].lt(0) if c in ["O01", "O04"] else False)
        V.loc[bad, c] = np.nan
        reason.loc[bad, c] = "missing_or_invalid_prerequisite"
    projection = {
        "case_type": "purchase_order",
        "cases": len(f),
        "all_source_events": len(evs),
        "unique_projected_events": len(set().union(*direct.values())),
        "event_case_associations": int(f.n_event_associations.sum()),
        "shared_source_events": sum(len(v) > 1 for v in event_po.values()),
        "shared_payment_source_events": sum(len(v) > 1 and evs[k]["type"] == "Execute Payment" for k, v in event_po.items()),
        "POs_with_shared_payment_evidence": int(f.shared_payment_associations.gt(0).sum()),
        "unique_payment_objects": len(payment_events),
        "payment_execution_events": sum(e["type"] == "Execute Payment" for e in evs.values()),
        "unit": "per-PO concern, not unique events/actions or monetary exposure",
        "censoring": "No certified extraction closure. Missing endpoints unevaluable, not failed completion.",
        "object_history": "1970 initial attributes used as baseline grouping metadata; later blank updates are not forward-filled into asserted event-time truth.",
    }
    dump(out / "projection.json", projection)
    alternative = V.copy()
    alternative.O01 = sat(raw.O01, 5, rules["O01"]["width"]).where(V.O01.notna())
    masked = V.copy()
    masked.loc[f.shared_payment_associations.gt(0), ["O03", "O04"]] = np.nan
    sens = [
        ("layer_balanced", f, V, A, "layer_balanced"),
        ("approval_threshold_5_days", f, alternative, A, "flat"),
        ("exclude_shared_payment_evidence", f, masked, A, "flat"),
    ]
    summaries, _primary = common_run("OCEL_simulated_P2P", f, V, A, reason, raw, norm, out, wr, sens)
    # Independent scalar source witnesses: raw JSON relationship scan, not feature columns.
    chosen = set()
    for c in ids:
        for mask in [V[c].gt(0), V[c].eq(0), V[c].isna()]:
            if mask.any():
                chosen.add(str(V.index[mask][0]))
    witness = []
    for po in sorted(chosen):
        ee = [e for e in d["events"] if any(r["objectId"] == po for r in e["relationships"])]

        def first(act, events=ee):
            return min((scalar_time(e["time"]) for e in events if e["type"] == act), default=None)

        tc, ta, tg, tp = map(
            first,
            [
                "Create Purchase Order",
                "Approve Purchase Order",
                "Create Goods Receipt",
                "Execute Payment",
            ],
        )
        calc = {
            "O01": None
            if tc is None or ta is None or ta < tc
            else min(
                max(
                    ((ta - tc).total_seconds() / 86400 - rules["O01"]["threshold"]) / rules["O01"]["width"],
                    0,
                ),
                1,
            ),
            "O02": None if tg is None or ta is None else float(tg < ta),
            "O04": None
            if tc is None or tp is None or tp < tc
            else min(
                max(
                    ((tp - tc).total_seconds() / 86400 - rules["O04"]["threshold"]) / rules["O04"]["width"],
                    0,
                ),
                1,
            ),
        }
        pay_ids = {
            r["objectId"]
            for e in ee
            if e["type"] == "Execute Payment"
            for r in e["relationships"]
            if obs[r["objectId"]]["type"] == "payment"
        }
        counts = [
            sum(e["type"] == "Execute Payment" and any(r["objectId"] == p for r in e["relationships"]) for e in ee)
            for p in pay_ids
        ]
        calc["O03"] = (
            None
            if not counts
            else min(
                max((max(counts) - rules["O03"]["threshold"]) / rules["O03"]["width"], 0),
                1,
            )
        )
        for c, val in calc.items():
            if val is None:
                assert pd.isna(V.loc[po, c])
            else:
                assert_near(V.loc[po, c], val)
        witness.append(
            {
                "case_id": po,
                "criterion_expected": calc,
                "passed": True,
                "event_ids": [e["id"] for e in sorted(ee, key=lambda e: (e["time"], e["id"]))],
                "raw_events": [
                    {"id": e["id"], "type": e["type"], "time": e["time"]} for e in sorted(ee, key=lambda e: (e["time"], e["id"]))
                ],
            }
        )
    dump(
        out / "source_witnesses.json",
        {
            "cases": len(witness),
            "checks": 4 * len(witness),
            "passed": True,
            "selection": "first lexical positive, zero and missing case for each criterion; union",
            "witnesses": witness,
        },
    )
    return {
        "views": summaries,
        "audit": {
            k: audit[k]
            for k in [
                "events",
                "objects",
                "dangling_object_object_rows",
                "distinct_dangling_object_ids",
                "zenodo_md5_matches",
            ]
        },
        "projection": projection,
        "source_witness_cases": len(witness),
    }


def audit_icpm(folder, out):
    tables = {}
    summary = []
    keys = {
        "Purchase Orders": ["Purchasing Document", "Purchasing Document Item No"],
        "Purchase Order Activities": [
            "Purchasing Document",
            "Purchasing Document Item No",
        ],
        "Sales Orders": ["Sales Document Number", "Sales Document Item"],
        "Sales Order Activities": ["Sales Document Number", "Sales Document Item"],
        "Stock Movements": ["Movement ID"],
        "Material Master": ["SKU ID"],
        "Inventory": ["SKU ID", "Inventory Date"],
        "Material Master History": ["SKU ID"],
    }
    files = {}
    for path in sorted(folder.glob("ICPM Data - *.csv")):
        name = path.stem.replace("ICPM Data - ", "")
        t = readcsv(path)
        tables[name] = t
        files[path.name] = {"bytes": path.stat().st_size, "sha256": sha(path)}
        summary.append(
            {
                "table": name,
                "rows": len(t),
                "columns": list(t.columns),
                "key_columns": keys[name],
                "distinct_key_values": len(t[keys[name]].drop_duplicates()),
                "exact_repeated_rows_after_whitespace_trim": int(t.duplicated().sum()),
                "blank_fields": {c: int(t[c].eq("").sum()) for c in t},
            }
        )
    zipmatch = {}
    archive = folder / "OneDrive_2026-02-11.zip"
    with zipfile.ZipFile(archive) as z:
        for item in z.namelist():
            if Path(item).name in files:
                zipmatch[Path(item).name] = hashlib.sha256(z.read(item)).hexdigest() == files[Path(item).name]["sha256"]
    joins = {}
    for master, events, kk in [
        ("Purchase Orders", "Purchase Order Activities", keys["Purchase Orders"]),
        ("Sales Orders", "Sales Order Activities", keys["Sales Orders"]),
    ]:
        m = set(map(tuple, tables[master][kk].itertuples(index=False, name=None)))
        e = set(map(tuple, tables[events][kk].itertuples(index=False, name=None)))
        joins[master] = {
            "master_keys": len(m),
            "activity_keys": len(e),
            "master_without_activity": len(m - e),
            "activity_without_master": len(e - m),
        }
    material = set(tables["Material Master"]["SKU ID"])
    joins["SKU_coverage"] = {
        n: {
            "distinct_SKUs": t["SKU ID"].nunique(),
            "SKUs_absent_from_material_master": len(set(t["SKU ID"]) - material),
        }
        for n, t in tables.items()
        if "SKU ID" in t
    }
    date_quality = {}
    for table, t in tables.items():
        for col in t.columns:
            if col.lower() in [
                "timestamp",
                "inventory date",
                "posting date",
                "created on",
                "order item created on",
                "scheduled delivery date",
                "latest confirmation date",
                "requested delivery date",
            ]:
                fmt = "%m/%d/%Y %H:%M:%S" if col.lower() == "timestamp" else "%m/%d/%Y"
                parsed = pd.to_datetime(t[col], format=fmt, errors="coerce")
                date_quality[table + " / " + col] = {
                    "rows": len(t),
                    "blank": int(t[col].eq("").sum()),
                    "unparseable_nonblank": int((parsed.isna() & t[col].ne("")).sum()),
                    "min": str(parsed.min()),
                    "max": str(parsed.max()),
                }
    stock = tables["Stock Movements"]
    for label, doc, item, master, mdoc, mitem in [
        (
            "purchase",
            "Purchase Order Number",
            "Purchase Document Item",
            "Purchase Orders",
            "Purchasing Document",
            "Purchasing Document Item No",
        ),
        (
            "sales",
            "Sales Document Number",
            "Sales Document Item",
            "Sales Orders",
            "Sales Document Number",
            "Sales Document Item",
        ),
    ]:
        attached = stock[doc].ne("")
        known = set(zip(tables[master][mdoc], tables[master][mitem]))
        link = list(zip(stock.loc[attached, doc], stock.loc[attached, item]))
        joins["stock_" + label + "_exact_string_keys"] = {
            "rows_with_document_reference": len(link),
            "rows_matching_master_key": sum(k in known for k in link),
            "distinct_referenced_keys": len(set(link)),
            "unmatched_distinct_keys": len(set(link) - known),
            "normalisation": "surrounding whitespace only; leading zeroes retained",
        }
    pa = tables["Purchase Order Activities"].copy()
    pa["case_key"] = pa["Purchasing Document"] + "|" + pa["Purchasing Document Item No"]
    shared = pa.groupby(["Purchasing Document", "Activity", "Timestamp"]).case_key.transform("nunique").gt(1)
    joins["purchase_document_level_record_sharing"] = {
        "activity_rows_sharing_document_activity_timestamp_across_items": int(shared.sum()),
        "interpretation": "Observed repeated document-level record pattern, not proof of duplicate real actions",
    }
    audit = {
        "files": files,
        "tables": summary,
        "joins": joins,
        "date_field_quality": date_quality,
        "archive_byte_matches": zipmatch,
        "excluded_inputs": "All derived folders and pickles excluded. Archive only compared against raw CSV files; not an additional dataset.",
        "event_identity": "Activity ID is activity class, not a unique event ID. Raw activity source row number is retained as record identity.",
        "financial_units": "Indexed order value is not interpreted as actual currency, spend loss or recoverable benefit. No cross-table receipt/invoice quantity balance is assumed.",
        "provenance": {
            "organiser": "https://pretix.eu/PI360/icpm-2026-hack/",
            "statement": "Organiser identifies ABB as provider of a real business challenge and data. Local file identity is documented by hashes and matching supplied ZIP; licence and exact public release metadata not established.",
            "accessed": "2026-09-30",
        },
    }
    dump(out / "schema_audit.json", audit)
    pd.DataFrame([{k: v for k, v in x.items() if k not in ["columns", "blank_fields"]} for x in summary]).to_csv(
        out / "table_counts.csv", index=False
    )
    return tables, audit


def icpm_assess(m, a, norm, width=None):
    m = m.copy()
    a = a.copy()
    m["case_id"] = m["Purchasing Document"] + "|" + m["Purchasing Document Item No"]
    a["case_id"] = a["Purchasing Document"] + "|" + a["Purchasing Document Item No"]
    assert not m.case_id.duplicated().any()
    universe = pd.Index(sorted(set(m.case_id) | set(a.case_id)), name="case_id")
    f = m.set_index("case_id").reindex(universe)
    f["master_present"] = f["SKU ID"].notna()
    f["n_activity_rows"] = a.groupby("case_id").size().reindex(universe, fill_value=0)
    f["group"] = f["Vendor ID"].fillna("(missing master)") + " | " + f["Vendor Country"].fillna("(missing master)")
    ts = pd.to_datetime(a.Timestamp, format="%m/%d/%Y %H:%M:%S", errors="coerce")
    a["parsed_time"] = ts
    gr = a.loc[a.Activity.eq("Goods receipt")].groupby("case_id").parsed_time.min()
    f["first_goods_receipt"] = gr.reindex(universe)
    for label, col in [
        ("scheduled", "Scheduled delivery date"),
        ("confirmed", "Latest confirmation date"),
        ("created", "Created On"),
    ]:
        f[label] = pd.to_datetime(f[col], format="%m/%d/%Y", errors="coerce")
    for cid in ["I03", "I04"]:
        acts = next(c["activities"] for c in norm["criteria"] if c["id"] == cid)
        f["count_" + cid] = a.loc[a.Activity.isin(acts)].groupby("case_id").size().reindex(universe, fill_value=0)
    ids = [c["id"] for c in norm["criteria"]]
    A = pd.DataFrame(True, index=universe, columns=ids)
    A.loc[f["Deletion indicator"].eq("L"), ["I01", "I02"]] = False
    raw = pd.DataFrame(index=universe)
    raw["I01"] = (f.confirmed - f.scheduled).dt.days.astype(float)
    raw["I02"] = (f.first_goods_receipt.dt.normalize() - f.scheduled).dt.days.astype(float)
    raw["I03"] = f.count_I03.astype(float)
    raw["I04"] = f.count_I04.astype(float)
    oq = pd.to_numeric(f["Order Quantity"], errors="coerce")
    cq = pd.to_numeric(f["Quantity confirmed"], errors="coerce")
    raw["I05"] = ((oq - cq).clip(lower=0) / oq).where(np.isfinite(oq) & np.isfinite(cq) & oq.gt(0) & cq.ge(0))
    V = raw.copy()
    rules = {c["id"]: c for c in norm["criteria"]}
    for cid in ["I01", "I02", "I03", "I04"]:
        active_width = width if width is not None and cid in ["I01", "I02"] else rules[cid]["width"]
        V[cid] = sat(raw[cid], rules[cid]["threshold"], active_width)
    V.I05 = raw.I05.clip(0, 1)
    V.loc[f.n_activity_rows.eq(0), ["I03", "I04"]] = np.nan
    reason = pd.DataFrame("evaluated", index=universe, columns=ids)
    reason = reason.mask(V.isna(), "missing_prerequisite")
    reason.loc[~f.master_present, ["I01", "I02", "I05"]] = "missing_master_context"
    V = V.where(A)
    reason = reason.mask(~A, "known_deleted_out_of_scope")
    assert V.notna().equals(reason.eq("evaluated"))
    return f, V, A, reason, raw, a


def run_icpm(folder, norm, out, wr):
    out.mkdir(parents=True, exist_ok=True)
    tabs, _audit = audit_icpm(folder, out)
    m, a = tabs["Purchase Orders"], tabs["Purchase Order Activities"]
    f, V, A, reason, raw, events = icpm_assess(m, a, norm)
    rules = {c["id"]: c for c in norm["criteria"]}
    deletion_state = f["Deletion indicator"].fillna("(missing master)").replace("", "not_deleted_in_snapshot")
    diag = pd.DataFrame(
        {
            "deletion_state": deletion_state,
            "I05_evaluated": V.I05.notna(),
            "I05_positive": V.I05.gt(0),
            "I05_severity": V.I05,
        }
    )
    diag.groupby("deletion_state").agg(
        cases=("I05_evaluated", "size"),
        evaluated=("I05_evaluated", "sum"),
        positive=("I05_positive", "sum"),
        mean_severity=("I05_severity", "mean"),
    ).to_csv(out / "I05_by_deletion_state.csv")
    fd, vd, ad, _, _, _ = icpm_assess(m, a.drop_duplicates(), norm)
    fw, vw, aw, _, _, _ = icpm_assess(m, a, norm, width=60)
    keep = f.master_present
    sens = [
        ("collapse_exact_activity_rows", fd, vd, ad, "flat"),
        ("master_context_only", f.loc[keep], V.loc[keep], A.loc[keep], "flat"),
        ("layer_balanced", f, V, A, "layer_balanced"),
        ("schedule_width_60_days", fw, vw, aw, "flat"),
    ]
    summaries, primary = common_run("ICPM_ABB_PO_items", f, V, A, reason, raw, norm, out, wr, sens)
    projection = {
        "cases": len(f),
        "master_cases": int(f.master_present.sum()),
        "orphan_activity_cases_retained": int((~f.master_present).sum()),
        "activity_rows": len(a),
        "distinct_document_keys": a["Purchasing Document"].nunique(),
        "parsed_activity_times": int(events.parsed_time.notna().sum()),
        "activity_min": str(events.parsed_time.min()),
        "activity_max": str(events.parsed_time.max()),
        "known_deleted_cases": int(f["Deletion indicator"].eq("L").sum()),
        "future_scheduled_relative_to_last_activity": int(f.scheduled.gt(events.parsed_time.max().normalize()).sum()),
        "exact_rows_removed_only_in_sensitivity": int(a.duplicated().sum()),
        "quantity_snapshot_shortfalls": int(raw.I05.gt(0).sum()),
        "censoring": "No certified extraction boundary. Future and missing receipt cases are not labelled failed completion; receipt-lag criterion unavailable without receipt. Snapshot scheduled date is not established historical commitment.",
        "exposure": "One PO item per case. No monetary scaling. Shared document actions may appear at multiple items; those are item-record associations.",
    }
    dump(out / "projection.json", projection)
    f.loc[~f.master_present, ["n_activity_rows", "group"]].to_csv(out / "orphan_activity_cases.csv")
    # Alternate SKU grouping is context, not a new case population.
    sku = f["SKU ID"].fillna("(missing master)").rename("SKU")
    wr.group_priorities(primary["score"], sku).to_csv(out / "groups_Balanced_by_SKU.csv")
    chosen = set()
    for c in V:
        for mask in [V[c].gt(0), V[c].eq(0), V[c].isna()]:
            if mask.any():
                chosen.add(str(V.index[mask][0]))
    md = {r["Purchasing Document"] + "|" + r["Purchasing Document Item No"]: r for r in m.to_dict("records")}
    ar = collections.defaultdict(list)
    for i, r in enumerate(a.to_dict("records"), start=2):
        ar[r["Purchasing Document"] + "|" + r["Purchasing Document Item No"]].append({"source_csv_line": i, **r})
    witnesses = []
    for case in sorted(chosen):
        master = md.get(case)
        ee = ar[case]
        calc = {}
        deleted = master is not None and master["Deletion indicator"] == "L"

        def date(s):
            return datetime.strptime(s, "%m/%d/%Y")

        def clamp(v):
            return min(max(v, 0), 1)

        calc["I01"] = (
            None
            if master is None or deleted
            else clamp(
                (
                    (date(master["Latest confirmation date"]) - date(master["Scheduled delivery date"])).days
                    - rules["I01"]["threshold"]
                )
                / rules["I01"]["width"]
            )
        )
        gr = [
            datetime.strptime(e["Timestamp"], "%m/%d/%Y %H:%M:%S").replace(hour=0, minute=0, second=0)
            for e in ee
            if e["Activity"] == "Goods receipt"
        ]
        calc["I02"] = (
            None
            if master is None or deleted or not gr
            else clamp(
                ((min(gr) - date(master["Scheduled delivery date"])).days - rules["I02"]["threshold"]) / rules["I02"]["width"]
            )
        )
        for cid in ["I03", "I04"]:
            acts = next(x["activities"] for x in norm["criteria"] if x["id"] == cid)
            calc[cid] = (
                None
                if not ee
                else clamp((sum(e["Activity"] in acts for e in ee) - rules[cid]["threshold"]) / rules[cid]["width"])
            )
        calc["I05"] = (
            None
            if master is None
            else clamp((float(master["Order Quantity"]) - float(master["Quantity confirmed"])) / float(master["Order Quantity"]))
        )
        for c, val in calc.items():
            if val is None:
                assert pd.isna(V.loc[case, c]), (case, c)
            else:
                assert_near(V.loc[case, c], val)
        witnesses.append(
            {
                "case_id": case,
                "passed": True,
                "criterion_expected": calc,
                "master_row": master,
                "activity_records": ee,
            }
        )
    dump(
        out / "source_witnesses.json",
        {
            "cases": len(witnesses),
            "checks": 5 * len(witnesses),
            "passed": True,
            "selection": "first lexical positive, zero and unavailable case per criterion; union",
            "witnesses": witnesses,
        },
    )
    return {
        "views": summaries,
        "projection": projection,
        "raw_table_rows": {k: len(t) for k, t in tabs.items()},
        "source_witness_cases": len(witnesses),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--norm", type=Path, default=Path(__file__).with_name("norms.json"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.project_root / "code"))
    import wise_reference as wr

    cfg = json.loads(args.norm.read_text())
    for nn in ["ocel", "icpm"]:
        for rule in cfg[nn]["criteria"]:
            if "width" in rule:
                assert np.isfinite(rule["threshold"])
                assert np.isfinite(rule["width"])
                assert rule["width"] > 0
    dump(args.output / "executed_norm.json", cfg)
    norm_hash = sha(args.norm)
    source_hash = sha(args.project_root / "code/wise_reference.py")
    dump(
        args.output / "manifest.json",
        {
            "status": "running",
            "norm_sha256": norm_hash,
            "reference_sha256": source_hash,
        },
    )
    results = {}
    results["ocel"] = run_ocel(args.data_root / "OCEL", cfg["ocel"], args.output / "ocel", wr)
    results["icpm"] = run_icpm(args.data_root / "ICPM Hackathon", cfg["icpm"], args.output / "icpm", wr)
    assert sha(args.norm) == norm_hash
    assert sha(args.project_root / "code/wise_reference.py") == source_hash
    dump(args.output / "summary.json", results)
    dump(
        args.output / "manifest.json",
        {
            "status": "passed",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "data_root": str(args.data_root),
            "project_root": str(args.project_root),
            "norm_sha256": norm_hash,
            "reference_sha256": source_hash,
            "script_sha256": sha(Path(__file__)),
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "source_writes": False,
            "pickles_loaded": False,
            "derived_input_folders_used": False,
            "norm_frozen_before_ranking": True,
            "results_files": {
                str(p.relative_to(args.output)): sha(p)
                for p in args.output.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            },
        },
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
