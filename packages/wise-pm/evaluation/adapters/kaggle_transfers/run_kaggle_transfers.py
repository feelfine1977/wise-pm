#!/usr/bin/env python3
"""Read-only raw CSV transfers; no archive code or deserialization is executed."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib.util
import json
import time
import zipfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPECS = {
    "incident_template": (
        "Incident_Management_CSV.csv",
        ";",
        "Case ID",
        "Event",
        "Timestamp",
        "%d/%m/%Y %H:%M",
    ),
    "uci_incidents": (
        "incident_event_log.csv",
        ",",
        "number",
        "incident_state",
        "sys_updated_at",
        "%d/%m/%Y %H:%M",
    ),
    "amr_orders": (
        "REGISTRO-EVENTOS-AMR-26-2.csv",
        ",",
        "ORDER",
        "ACTIVITY",
        "DATE-TIME",
        "%d/%m/%y %H:%M",
    ),
    "p2p": (
        "archive (5)/Procure-to-Pay.csv",
        ",",
        "Case ID",
        "Activity",
        "Start Timestamp",
        "%d/%m/%Y %H:%M:%S",
    ),
}
AMEND = [
    "Amend Purchase Requisition",
    "Amend Request for Quotation Requester",
    "Amend Request for Quotation Requester Manager",
]
DISPUTE = [
    "Settle dispute with supplier Purchasing Agent",
    "Settle dispute with supplier Financial Manager",
]
CHANGE = ["CAMBIO DE CENTRO", "CAMBIO DATOS", "CAMBIO DE SITUACION"]


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            default=lambda x: x.item() if hasattr(x, "item") else str(x),
            allow_nan=False,
        )
        + "\n"
    )


def csv(p, d):
    d.to_csv(p, index=True, index_label=d.index.name or "case_id", float_format="%.15g")


def read_csv(p, sep=","):
    d = pd.read_csv(p, sep=sep, dtype=str, keep_default_na=False)
    d["_source_file"] = str(p.name)
    d["_source_row"] = np.arange(2, len(d) + 2)
    return d


def load(root, name):
    path, sep, key, act, t, fmt = SPECS[name]
    d = read_csv(root / path, sep)
    if name == "p2p":
        e = read_csv(root / "archive (5)/Procure-to-Pay 2.csv")
        assert not set(d[key]) & set(e[key]), "P2P primary files overlap in case identity"
        d = pd.concat([d, e], ignore_index=True)
    d["_case"] = d[key].str.strip()
    d["_act"] = d[act].str.strip()
    assert d["_case"].ne("").all(), "Blank case identity"
    d["_t"] = pd.to_datetime(d[t], format=fmt, errors="coerce")
    d["_complete"] = pd.to_datetime(d["Complete Timestamp"], format=fmt, errors="coerce") if name == "p2p" else d["_t"]
    if name == "uci_incidents":
        d["_counter"] = pd.to_numeric(d["sys_mod_count"], errors="coerce")
    else:
        d["_counter"] = 0
    return d.sort_values(["_case", "_t", "_counter", "_source_file", "_source_row"], kind="stable")


def constant(d, col, index):
    # Missing values anywhere are unknown evidence, not silently ignored.
    s = d[col].str.strip().replace({"": "<missing>", "?": "<missing>"})
    z = pd.DataFrame({"case": d["_case"], "v": s}).groupby("case")["v"]
    out = z.first().reindex(index)
    return out.where(z.nunique().reindex(index).eq(1) & out.ne("<missing>"))


def extract(d, name):
    g = d.groupby("_case", sort=True)
    ix = pd.Index(g.size().index, name="case_id")
    f = pd.DataFrame(index=ix)
    f["source_files"] = g["_source_file"].apply(lambda x: " | ".join(sorted(set(x))))
    f["raw_records"] = g.size()
    f["missing_activity_records"] = g["_act"].apply(lambda x: int(x.eq("").sum()))
    f["unparseable_time_records"] = g["_t"].apply(lambda x: int(x.isna().sum()))
    f["first_record_time"] = g["_t"].min()
    f["last_record_time"] = g["_t"].max()

    def cnt(labels):
        return (
            d["_act"].isin(labels).groupby(d["_case"]).sum().reindex(ix).astype(float).where(f["missing_activity_records"].eq(0))
        )

    def time_at(labels, col, how):
        return d.loc[d["_act"].isin(labels)].groupby("_case")[col].agg(how).reindex(ix)

    def duration(start, end, unit):
        f["timing_start"] = start
        f["timing_end"] = end
        q = (end - start).dt.total_seconds() / unit
        f["timing_evidence_state"] = np.select(
            [start.isna() | end.isna(), q.lt(0), f["unparseable_time_records"].gt(0)],
            [
                "missing_endpoint",
                "negative_endpoint_interval",
                "unparseable_event_time",
            ],
            default="evaluated",
        )
        return q.where(q.ge(0) & f["unparseable_time_records"].eq(0))

    if name == "incident_template":
        f["group"] = (
            constant(d, "Issue Type", ix).fillna("<unknown>") + " | " + constant(d, "Report Channel", ix).fillna("<unknown>")
        )
        f["elapsed_hours"] = duration(
            time_at(["Ticket created"], "_t", "min"),
            time_at(["Ticket closed"], "_t", "max"),
            3600,
        )
        f["reopen_events"] = cnt(["Ticket reopened by customer"])
        sat = pd.to_numeric(constant(d, "Customer Satisfaction", ix), errors="coerce")
        f["satisfaction_shortfall"] = (4 - sat).clip(lower=0).where(sat.between(1, 5) & sat.mod(1).eq(0))
    elif name == "uci_incidents":
        # Row-ordered initial category, not a claim that category caused the outcome.
        f["group"] = g["category"].first().replace({"?": "<unknown>", "": "<unknown>"})
        a = pd.to_datetime(constant(d, "opened_at", ix), format="%d/%m/%Y %H:%M", errors="coerce")
        b = pd.to_datetime(constant(d, "resolved_at", ix), format="%d/%m/%Y %H:%M", errors="coerce")
        f["elapsed_hours"] = duration(a, b, 3600)
        for col, target in [
            ("reassignment_count", "reassignments"),
            ("reopen_count", "reopens"),
        ]:
            num = pd.to_numeric(d[col], errors="coerce")
            valid = np.isfinite(num) & num.ge(0) & num.mod(1).eq(0)
            f[target] = num.groupby(d["_case"]).max().reindex(ix).where(valid.groupby(d["_case"]).all().reindex(ix))
        f["category_changes_observed"] = g["category"].nunique() - 1
    elif name == "amr_orders":
        f["group"] = constant(d, "SERVICE TYPE", ix).fillna("<unknown>")
        f["elapsed_days"] = duration(time_at(["CAPTURO"], "_t", "min"), time_at(["TERMINO"], "_t", "max"), 86400)
        f["assignment_events"] = cnt(["ASIGNO"])
        f["change_events"] = cnt(CHANGE)
    else:
        f["group"] = constant(d, "Country", ix).fillna("<unknown>")
        f["elapsed_days"] = duration(
            time_at(["Create Purchase Requisition"], "_t", "min"),
            time_at(["Pay invoice"], "_complete", "max"),
            86400,
        )
        f["amendment_events"] = cnt(AMEND)
        f["dispute_events"] = cnt(DISPUTE)
        bad = d["_complete"].isna() | d["_complete"].lt(d["_t"])
        f["invalid_interval_records"] = bad.groupby(d["_case"]).sum().reindex(ix)
        f["elapsed_days"] = f["elapsed_days"].where(f["invalid_interval_records"].eq(0))
        f.loc[f["invalid_interval_records"].gt(0), "timing_evidence_state"] = "invalid_activity_interval"
    return f


def assess(f, norm):
    v = pd.DataFrame(index=f.index)
    for rule in norm["criteria"]:
        assert np.isfinite(rule["threshold"])
        assert np.isfinite(rule["width"])
        assert rule["width"] > 0
        v[rule["id"]] = ((f[rule["feature"]] - rule["threshold"]) / rule["width"]).clip(0, 1)
    return (
        v,
        pd.DataFrame(True, index=v.index, columns=v.columns),
        {r["id"]: r["layer"] for r in norm["criteria"]},
    )


def scalar_features(rows, name):
    # Independent, per-case witnesses use source strings and Python datetime.
    from datetime import datetime

    _, _, _, ac, tc, fmt = SPECS[name]
    labels = [str(r[ac]).strip() for r in rows]

    def ts(s, format=fmt):
        try:
            return datetime.strptime(s, format)
        except (ValueError, TypeError):
            return None

    times = [ts(r[tc]) for r in rows]

    def count(acts):
        return float(sum(a in acts for a in labels)) if all(labels) else np.nan

    def lag(start, end, scale, endcol=None):
        a = [ts(r[tc]) for r, label in zip(rows, labels) if label == start]
        b = [ts(r[endcol or tc]) for r, label in zip(rows, labels) if label == end]
        if not a or not b or None in times or None in a or None in b:
            return np.nan
        value = (max(b) - min(a)).total_seconds() / scale
        return value if value >= 0 else np.nan

    if name == "incident_template":
        vals = {r["Customer Satisfaction"].strip() for r in rows}
        try:
            s = float(next(iter(vals))) if len(vals) == 1 else np.nan
        except ValueError:
            s = np.nan
        return [
            lag("Ticket created", "Ticket closed", 3600),
            count(["Ticket reopened by customer"]),
            max(4 - s, 0) if np.isfinite(s) and 1 <= s <= 5 and s.is_integer() else np.nan,
        ]
    if name == "uci_incidents":

        def one(c):
            vals = {r[c].strip() for r in rows}
            return ts(next(iter(vals)), "%d/%m/%Y %H:%M") if len(vals) == 1 else None

        a, b = one("opened_at"), one("resolved_at")
        q = (b - a).total_seconds() / 3600 if a and b and None not in times else np.nan

        def counter(c):
            try:
                vals = [float(r[c]) for r in rows]
                return max(vals) if all(np.isfinite(v) and v >= 0 and v.is_integer() for v in vals) else np.nan
            except ValueError:
                return np.nan

        return [
            q if q >= 0 else np.nan,
            counter("reassignment_count"),
            counter("reopen_count"),
        ]
    if name == "amr_orders":
        return [lag("CAPTURO", "TERMINO", 86400), count(["ASIGNO"]), count(CHANGE)]
    bad = any(
        ts(r["Complete Timestamp"]) is None
        or ts(r["Start Timestamp"]) is None
        or ts(r["Complete Timestamp"]) < ts(r["Start Timestamp"])
        for r in rows
    )
    return [
        np.nan if bad else lag("Create Purchase Requisition", "Pay invoice", 86400, "Complete Timestamp"),
        count(AMEND),
        count(DISPUTE),
    ]


def witness(d, f, v, norm, name):
    # Fixed lexical first, missing-evidence, positive-count and duplicate-case examples.
    ids = [f.index[0]]
    for criterion in v:
        for cond in [v[criterion].isna(), v[criterion].gt(0)]:
            matches = f.index[cond]
            if len(matches):
                ids.append(matches[0])
    ids = list(dict.fromkeys(ids))
    out = []
    errors = []
    rawcols = [c for c in d if not c.startswith("_")]
    for case in ids:
        z = d[d["_case"].eq(case)]
        records = z[rawcols].to_dict("records")
        values = scalar_features(records, name)
        derived = []
        for j, rule in enumerate(norm["criteria"]):
            x = values[j]
            expected = min(1, max(0, (x - rule["threshold"]) / rule["width"])) if np.isfinite(x) else np.nan
            actual = v.loc[case, rule["id"]]
            ok = (np.isnan(expected) and np.isnan(actual)) or abs(actual - expected) < 1e-12
            assert ok, (name, case, rule, expected, actual)
            if np.isfinite(expected):
                errors.append(abs(actual - expected))
            derived.append(
                {
                    "criterion": rule["id"],
                    "raw_feature": None if np.isnan(x) else x,
                    "severity": None if np.isnan(expected) else expected,
                    "passed": bool(ok),
                }
            )
        # Source row locator and minimum sufficient fields, not personal reporter names.
        keep = list(
            dict.fromkeys(
                [SPECS[name][2], SPECS[name][3], SPECS[name][4]]
                + (
                    {
                        "uci_incidents": [
                            "opened_at",
                            "resolved_at",
                            "reassignment_count",
                            "reopen_count",
                            "category",
                        ],
                        "incident_template": [
                            "Issue Type",
                            "Report Channel",
                            "Customer Satisfaction",
                        ],
                        "amr_orders": ["SERVICE TYPE"],
                        "p2p": ["Complete Timestamp", "Country"],
                    }[name]
                )
            )
        )
        out.append(
            {
                "case_id": case,
                "checks": derived,
                "source_records": z[["_source_file", "_source_row", *keep]].to_dict("records"),
            }
        )
    return out, {
        "cases": len(ids),
        "criterion_checks": len(ids) * len(norm["criteria"]),
        "max_severity_residual": max(errors, default=0),
    }


def raw_audit(root):
    pairs = []
    hashes = {}
    archives = []
    for zn in [
        "archive (1).zip",
        "archive (3).zip",
        "archive (4).zip",
        "archive (5).zip",
        "archive (6).zip",
    ]:
        zp = root / zn
        hashes[zn] = sha(zp)
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                archives.append({"archive": zn, "member": info.filename, "bytes": info.file_size})
                if info.is_dir():
                    continue
                target = root / ("archive (5)" if zn == "archive (5).zip" else "") / info.filename
                if target.exists() and target.is_file():
                    with z.open(info) as stream:
                        h = hashlib.file_digest(stream, "sha256").hexdigest()
                    local = sha(target)
                    hashes[str(target.relative_to(root))] = local
                    pairs.append(
                        {
                            "archive": zn,
                            "member": info.filename,
                            "raw_file": str(target.relative_to(root)),
                            "sha256": local,
                            "byte_identical": h == local,
                        }
                    )
                    assert h == local, "Archive/extracted mismatch"
    a = pd.read_csv(root / "archive (5)/Procure-to-Pay.csv", dtype=str, keep_default_na=False)
    b = pd.read_csv(root / "archive (5)/Procure-to-Pay 2.csv", dtype=str, keep_default_na=False)
    c = pd.read_csv(root / "archive (5)/Procure-to-Pay 3.csv", dtype=str, keep_default_na=False)
    aset = set(map(tuple, a.values))
    assert all(tuple(row) in aset for row in c.values)
    assert not set(a["Case ID"]) & set(b["Case ID"])
    mapping = pd.read_csv(root / "archive (5)/Phases_of_activities.csv", sep=";", dtype=str)
    assert mapping["Activity"].is_unique
    unknown = sorted((set(a.Activity) | set(b.Activity)) - set(mapping.Activity))
    assert not unknown
    return {
        "archive_members": archives,
        "archive_raw_identity": pairs,
        "raw_sha256": hashes,
        "p2p": {
            "main_rows": len(a),
            "second_rows": len(b),
            "third_rows": len(c),
            "third_is_exact_row_subset_of_main": True,
            "third_cases": c["Case ID"].nunique(),
            "main_second_case_overlap": 0,
            "unique_primary_rows": len(a) + len(b),
            "unique_primary_cases": a["Case ID"].nunique() + b["Case ID"].nunique(),
            "phase_mapping_rows": len(mapping),
            "unmapped_activities": unknown,
        },
        "excluded_applications": [
            {
                "source": "archive (1).zip / bpi_2017_cleaned.csv",
                "reason": "BPIC17 representation, evaluated by BPIC adapter; not additional data",
            },
            {
                "source": "Procure-to-Pay 3.csv",
                "reason": "All rows are in Procure-to-Pay.csv",
            },
            {
                "source": "Phases_of_activities.csv",
                "reason": "Activity dictionary, not events; full mapping audited, not a normative ordering model",
            },
        ],
    }


def run(root, project, output, norm_path):
    output = output.resolve()
    root = root.resolve()
    project = project.resolve()
    assert output != root, "Output must be outside immutable source/project roots"
    assert root not in output.parents, "Output must be outside immutable source/project roots"
    assert project not in output.parents, "Output must be outside immutable source/project roots"
    output.mkdir(parents=True, exist_ok=True)
    assert not any(output.iterdir()), "Output directory must be empty"
    start = time.monotonic()
    core = project / "code/wise_reference.py"
    core_hash = sha(core)
    spec = importlib.util.spec_from_file_location("wise_reference", core)
    wise = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wise)
    norms = json.loads(norm_path.read_text())
    dump(output / "executed_norm.json", norms)
    triage = raw_audit(root)
    dump(output / "triage.json", triage)
    allrows = []
    verification = {}
    schemas = {}
    for name, norm in norms["datasets"].items():
        dest = output / name
        dest.mkdir()
        d = load(root, name)
        f = extract(d, name)
        v, a, layers = assess(f, norm)
        keys = f["group"].rename("group")
        rawcols = [c for c in d if not c.startswith("_")]
        duplicated = d.duplicated(rawcols)
        schema = {
            "rows": len(d),
            "cases": len(f),
            "activities": d["_act"].nunique(),
            "blank_activity_rows": int(d["_act"].eq("").sum()),
            "invalid_timestamp_rows": int(d["_t"].isna().sum()),
            "exact_duplicate_rows": int(duplicated.sum()),
            "time_min": str(d["_t"].min()),
            "time_max": str(d["_t"].max()),
            "columns": rawcols,
            "unknown_group_cases": int(f["group"].str.contains("<unknown>", regex=False).sum()),
        }
        schema["timing_evidence_states"] = {
            str(k): int(v) for k, v in f["timing_evidence_state"].value_counts().sort_index().items()
        }
        if name == "uci_incidents":
            schema["cases_category_changed"] = int(f["category_changes_observed"].gt(0).sum())
            schema["unknown_state_code_rows"] = int(d["_act"].eq("-100").sum())
            for c in ["reassignment_count", "reopen_count"]:
                schema[c + "_cases_with_observed_decrease"] = int(
                    pd.to_numeric(d[c]).groupby(d["_case"]).diff().lt(0).groupby(d["_case"]).any().sum()
                )
        csv(dest / "case_features.csv", f)
        csv(dest / "assessments.csv", v)
        states = pd.DataFrame(
            [
                {
                    "criterion": c,
                    "in_scope": len(v),
                    "evaluated": int(v[c].notna().sum()),
                    "missing_evidence": int(v[c].isna().sum()),
                    "positive": int(v[c].gt(0).sum()),
                }
                for c in v
            ]
        )
        states.to_csv(dest / "criterion_states.csv", index=False)
        dump(dest / "schema.json", schema)
        schemas[name] = schema
        witnesses, wcheck = witness(d, f, v, norm, name)
        dump(dest / "source_witnesses.json", witnesses)
        viewchecks = {}
        wide = pd.DataFrame(index=v.index)
        base = None
        for view, weights in norm["views"].items():
            w = pd.Series(weights, index=v.columns, dtype=float)
            r = wise.score_assessments(v, a, w, layers, mode="flat")
            groups = wise.group_priorities(r["score"], keys, gamma=0, min_cases=1)
            signed = wise.signed_components(r["penalties"], keys)
            co = wise.coco_same_target(r["penalty"], keys)
            csv(dest / f"groups_{view}.csv", groups)
            csv(dest / f"signed_{view}.csv", signed)
            csv(dest / f"coco_{view}.csv", co)
            wide[f"penalty__{view}"] = r["penalty"]
            wide[f"coverage__{view}"] = r["coverage"]
            direct = (v.mul(w).sum(axis=1, min_count=1) / v.notna().mul(w).sum(axis=1)).where(v.notna().mul(w).sum(axis=1) > 0)
            residuals = {
                "weighted_penalty": float((direct - r["penalty"]).abs().max()),
                "effective_weight_sum": float((r["effective_weights"].sum(axis=1) - 1).abs().max()),
                "signed_PI": float((signed.sum(axis=1).clip(lower=0) - groups["PI"]).abs().max()),
                "group_mean_PI": float(
                    (groups["n_cases"] * (groups["mean_penalty"] - r["penalty"].mean()).clip(lower=0) - groups["PI"]).abs().max()
                ),
            }
            D = float(co["D"].iloc[0])
            residuals["coco_identity"] = float((co["D"] * co["CoCo"].clip(lower=0) - groups["PI"]).abs().max()) if D > 0 else None
            assert all(x is None or x < 1e-8 for x in residuals.values()), residuals
            coverage = v.notna().mul(w).sum(axis=1) / w.sum()
            assert np.allclose(coverage, r["coverage"])
            row = {
                "dataset": name,
                "view": view,
                "unit": norm["unit"],
                "cases": len(f),
                "events": len(d),
                "scored": int(r["score"].notna().sum()),
                "groups": len(groups),
                "positive_groups": int(groups.PI.gt(1e-12).sum()),
                "mean_penalty": float(r["penalty"].mean()),
                "mean_coverage": float(r["coverage"].mean()),
                "lead_group": str(groups.index[0]),
                "lead_PI": float(groups.PI.iloc[0]),
            }
            allrows.append(row)
            viewchecks[view] = {
                "status": "passed",
                "residuals": residuals,
                "coverage_identity": True,
                "coco_D": D,
            }
            if view == "Balanced":
                base = (r, groups)
        csv(dest / "case_scores.csv", wide)
        sensitivities = []
        # Postfreeze reporting of declared alternatives, no norm tuning.
        alternatives = [
            ("layer_balanced", v, a, "layer_balanced"),
            ("timing_width_doubled", v.copy(), a, "flat"),
        ]
        first = norm["criteria"][0]
        alternatives[1][1][first["id"]] = ((f[first["feature"]] - first["threshold"]) / (first["width"] * 2)).clip(0, 1)
        if duplicated.any():
            ff = extract(d.loc[~duplicated], name)
            vv, aa, _ = assess(ff, norm)
            alternatives.append(("exact_row_duplicates_collapsed", vv, aa, "flat"))
        for label, vv, aa, mode in alternatives:
            w = pd.Series(norm["views"]["Balanced"], index=vv.columns, dtype=float)
            rr = wise.score_assessments(vv, aa, w, layers, mode=mode)
            gg = wise.group_priorities(rr["score"], keys)
            bset = set(base[1].loc[base[1].PI.gt(1e-12)].head(5).index)
            sset = set(gg.loc[gg.PI.gt(1e-12)].head(5).index)
            sensitivities.append(
                {
                    "policy": label,
                    "cases": len(vv),
                    "mean_penalty": float(rr["penalty"].mean()),
                    "mean_coverage": float(rr["coverage"].mean()),
                    "positive_top5_jaccard": len(bset & sset) / len(bset | sset) if bset | sset else np.nan,
                    "cases_penalty_changed": int((rr["penalty"] - base[0]["penalty"]).abs().gt(1e-12).sum()),
                    "lead_group": str(gg.index[0]),
                    "lead_PI": float(gg.PI.iloc[0]),
                }
            )
            csv(dest / f"groups_sensitivity_{label}.csv", gg)
        pd.DataFrame(sensitivities).to_csv(dest / "sensitivity.csv", index=False, float_format="%.15g")
        verification[name] = {
            "status": "passed",
            "witnesses": wcheck,
            "views": viewchecks,
        }
    assert sha(core) == core_hash, "Reference core changed"
    for path, hash in triage["raw_sha256"].items():
        assert sha(root / path) == hash, "Raw source changed"
    pd.DataFrame(allrows).to_csv(output / "view_summary.csv", index=False, float_format="%.15g")
    dump(output / "verification.json", verification)
    dump(output / "schema_summary.json", schemas)
    manifest = {
        "status": "passed",
        "reference_sha256": core_hash,
        "script_sha256": sha(Path(__file__)),
        "norm_sha256": sha(norm_path),
        "raw_sha256": triage["raw_sha256"],
        "raw_preserved": True,
        "core_preserved": True,
        "datasets": list(norms["datasets"]),
        "cases_total_across_different_units": sum(s["cases"] for s in schemas.values()),
        "elapsed_seconds": time.monotonic() - start,
        "data_root": str(root),
        "project_root": str(project),
        "output": str(output),
        "output_sha256": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
    }
    dump(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "passed",
                "applications": len(schemas),
                "cases": manifest["cases_total_across_different_units"],
                "source_witness_checks": sum(x["witnesses"]["criterion_checks"] for x in verification.values()),
                "output": str(output),
            }
        )
    )
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--norm", type=Path, default=HERE / "norms.json")
    a = p.parse_args()
    run(a.data_root, a.project_root, a.output, a.norm)
