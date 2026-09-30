#!/usr/bin/env python3
"""Separate full-XES BPIC2013 transfer tests. No source mutation or pickle use."""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from defusedxml.ElementTree import iterparse

HERE = Path(__file__).resolve().parent
FAMILIES = ["incidents", "open_problems", "closed_problems"]
MAX_BYTES = 256 * 1024 * 1024


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def dump(p, x):
    def enc(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o).__name__)

    Path(p).write_text(json.dumps(x, indent=2, default=enc, allow_nan=False) + "\n")


def local(tag):
    return tag.rsplit("}", 1)[-1]


def attrs(node):
    d = {}
    for x in node:
        tag = local(x.tag)
        if tag == "event":
            continue
        if tag not in {"string", "date", "int", "float", "boolean", "id"} or len(x):
            raise ValueError("Unsupported nested XES attribute; map explicitly")
        k = x.attrib["key"]
        if k in d:
            raise ValueError("Duplicate XES key " + k)
        d[k] = x.attrib.get("value", "")
    return d


def parse_xes(path):
    with gzip.open(path, "rb") as stream:
        payload = stream.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError("Decompressed XES exceeds declared 256MiB safety limit")
    rows = []
    traces = []
    tc = collections.Counter()
    ec = collections.Counter()
    for _, node in iterparse(
        io.BytesIO(payload),
        events=["end"],
        forbid_dtd=True,
        forbid_entities=True,
        forbid_external=True,
    ):
        if local(node.tag) != "trace":
            continue
        t = attrs(node)
        cid = t.get("concept:name")
        if not cid:
            raise ValueError("Trace has no case identifier")
        tc.update(t.keys())
        trace_no = len(traces) + 1
        traces.append({"case_id": cid, **{"trace::" + k: v for k, v in t.items()}})
        for j, e in enumerate(x for x in node if local(x.tag) == "event"):
            a = attrs(e)
            ec.update(a.keys())
            rows.append(
                {
                    "case_id": cid,
                    "trace_position": trace_no,
                    "event_position": j,
                    **{"event::" + k: v for k, v in a.items()},
                }
            )
        node.clear()
    t = pd.DataFrame(traces)
    d = pd.DataFrame(rows)
    if not t.case_id.is_unique:
        raise ValueError("Duplicate trace IDs; never merge silently")
    if d.empty or set(d.case_id) != set(t.case_id):
        raise ValueError("Empty or eventless trace")
    schema = {
        "cases": len(t),
        "events": len(d),
        "trace_attributes": dict(tc),
        "event_attributes": dict(ec),
        "compressed_sha256": sha(path),
        "decompressed_sha256": hashlib.sha256(payload).hexdigest(),
        "decompressed_bytes": len(payload),
    }
    schema["attribute_profiles"] = {
        c: {
            "present": int(d[c].notna().sum()),
            "distinct": int(d[c].nunique(dropna=True)),
            "top10": [[str(k), int(v)] for k, v in d[c].value_counts(dropna=False).head(10).items()],
        }
        for c in d
        if c.startswith("event::")
    }
    return t, d, schema


def usable(s, cfg):
    return s.notna() & ~s.astype(str).str.strip().str.lower().isin(cfg["missing_markers"])


def stable(s, cfg):
    if not usable(s, cfg).all():
        return "(missing attribute)"
    return str(s.iloc[0]) if s.nunique() == 1 else "(multiple values)"


def sequence_values(values):
    compressed = [v for i, v in enumerate(values) if i == 0 or v != values[i - 1]]
    return max(0, len(compressed) - 1), sum(compressed[i] == compressed[i - 2] for i in range(2, len(compressed)))


def features(d, family, cfg):
    mapping = cfg["field_mapping"][family]
    d = d.copy()
    for dest, key in [
        ("team", mapping["team"]),
        ("organization", mapping["organization"]),
        ("country", mapping["country"]),
        ("status", "concept:name"),
        ("substatus", "lifecycle:transition"),
        ("product", "product"),
        ("impact", "impact"),
    ]:
        d[dest] = (
            d.get("event::" + str(key), pd.Series(pd.NA, index=d.index, dtype="string"))
            if key
            else pd.Series(pd.NA, index=d.index, dtype="string")
        )
    d["timestamp"] = pd.to_datetime(d.get("event::time:timestamp"), format="ISO8601", utc=True, errors="coerce")
    rows = []
    for cid, g in d.groupby("case_id", sort=True):
        original = g.sort_values("event_position")
        valid = bool(g.timestamp.notna().all())
        g = g.sort_values(["timestamp", "event_position"], kind="stable", na_position="last")

        def tie_amb(cols, frame=g):
            return bool(
                any(z[cols].drop_duplicates().shape[0] > 1 for _, z in frame.groupby("timestamp", dropna=False) if len(z) > 1)
            )

        team_ok = bool(usable(g.team, cfg).all())
        sub_ok = bool(usable(g.substatus, cfg).all())
        status_ok = bool(usable(g.status, cfg).all())
        team_tie = tie_amb(["team"])
        status_tie = tie_amb(["status"])
        tchange, treturn = sequence_values(g.team.tolist()) if team_ok and valid and not team_tie else (np.nan, np.nan)
        state_team = (
            "evaluated"
            if team_ok and valid and not team_tie
            else (
                "missing_XES_team_field"
                if mapping["team"] is None
                else "missing_team_value"
                if not team_ok
                else "invalid_timestamp"
                if not valid
                else "ambiguous_timestamp_tie"
            )
        )
        active = np.nan
        if status_ok and valid and not status_tie:
            seen = False
            active = 0
            for s in g.status:
                if seen and s in {"Accepted", "Queued"}:
                    active = 1
                if s == "Completed":
                    seen = True
        span = (g.timestamp.max() - g.timestamp.min()).total_seconds() / 86400 if valid and len(g) >= 2 else np.nan
        rows.append(
            {
                "case_id": cid,
                "n_events": len(g),
                "product_group": stable(g["product"], cfg),
                "organization_group": stable(g.organization, cfg),
                "impact_group": stable(g.impact, cfg),
                "first_timestamp": g.timestamp.min(),
                "last_timestamp": g.timestamp.max(),
                "all_timestamps_valid": valid,
                "source_timestamp_order_monotone": bool(original.timestamp.is_monotonic_increasing) if valid else False,
                "team_tie_ambiguous": team_tie,
                "status_tie_ambiguous": status_tie,
                "team_changes": tchange,
                "team_returns": treturn,
                "state_team": state_team,
                "repeated_substatus": len(g) - g.substatus.nunique() if sub_ok else np.nan,
                "wait_user_count": int(g.substatus.eq("Wait - User").sum()) if sub_ok else np.nan,
                "span_days": span,
                "post_completed_active": active,
                "has_recorded_closed": bool(g.substatus.eq("Closed").any()),
                "last_substatus": str(g.substatus.iloc[-1]) if valid else "(invalid timestamp)",
                "state_substatus": "evaluated" if sub_ok else "missing_substatus",
                "state_span": "evaluated"
                if np.isfinite(span)
                else "invalid_timestamp"
                if not valid
                else "insufficient_endpoints",
                "state_active": "evaluated"
                if np.isfinite(active)
                else "missing_status"
                if not status_ok
                else "invalid_timestamp"
                if not valid
                else "ambiguous_timestamp_tie",
            }
        )
    return pd.DataFrame(rows).set_index("case_id"), d


def assess(f, family, cfg, span_scale=1):
    ids = [c["id"] for c in cfg["criteria"]]
    V = pd.DataFrame(np.nan, index=f.index, columns=ids)
    A = pd.DataFrame(True, index=f.index, columns=ids)
    R = pd.DataFrame("", index=f.index, columns=ids)
    for c in cfg["criteria"]:
        cid = c["id"]
        scope = c["scope"] == "all" or c["scope"] == family
        A[cid] = scope
        if not scope:
            R[cid] = "out_of_scope"
            continue
        feature = c["feature"]
        source_state = (
            "state_team"
            if cid in ["C01", "C02"]
            else "state_substatus"
            if cid in ["C03", "C04"]
            else "state_span"
            if cid == "C05"
            else "state_active"
        )
        R[cid] = f[source_state]
        threshold = c.get("threshold", c.get("threshold_by_family", {}).get(family))
        width = c.get("width", c.get("width_by_family", {}).get(family))
        if cid == "C05":
            threshold *= span_scale
            width *= span_scale
        V[cid] = ((f[feature] - threshold) / width).clip(0, 1)
    return V, A, R


def pick(t, key="PI", k=10):
    t = t.reset_index()
    name = t.columns[0]
    t["_key"] = t[name].astype(str)
    t = t.sort_values([key, "n_cases", "_key"], ascending=[False, False, True], kind="stable")
    pos = t.loc[t[key] > 1e-12]
    top = pos.head(k)
    boundary = float(top[key].iloc[-1]) if len(top) else None
    ties = int(np.isclose(pos[key], boundary, atol=1e-12, rtol=0).sum()) if boundary is not None else 0
    crosses = (
        bool(len(pos) > len(top) and np.isclose(pos[key].iloc[len(top)], boundary, atol=1e-12, rtol=0))
        if boundary is not None
        else False
    )
    return top, {
        "selected_k": len(top),
        "positive_candidates": len(pos),
        "requested_k": k,
        "boundary_tied_candidates": ties,
        "boundary_tie_crosses_cutoff": crosses,
    }


def overlap(a, b):
    sa = set(a.iloc[:, 0].astype(str))
    sb = set(b.iloc[:, 0].astype(str))
    return len(sa & sb) / len(sa | sb) if sa | sb else None


def csv_audit(path, all_d):
    results = {}
    with zipfile.ZipFile(path) as z:
        for fam, d in all_d.items():
            name = "VINST cases " + fam.replace("_", " ") + ".csv"
            with z.open(name) as f:
                c = pd.read_csv(f, sep=";", encoding="cp1252", dtype=str, keep_default_na=False)
            idcol = "SR Number" if fam == "incidents" else "Problem Number"
            timecol = "Change Date+Time" if fam == "incidents" else "Problem Change Date+Time"
            teamcol = "Involved ST" if fam == "incidents" else "Problem Involved Owner ST"
            orgcol = "Involved Org line 3" if fam == "incidents" else "Problem Involved ST Org line 3"
            statuscol = "Status" if fam == "incidents" else "Problem Status"
            subcol = "Sub Status" if fam == "incidents" else "Problem Sub Status"
            xs = pd.DataFrame(
                {
                    "case_id": d.case_id,
                    "timestamp": d.timestamp.astype(str),
                    "status": d.status,
                    "substatus": d.substatus,
                }
            )
            cs = pd.DataFrame(
                {
                    "case_id": c[idcol],
                    "timestamp": pd.to_datetime(c[timecol], format="ISO8601", utc=True).astype(str),
                    "status": c[statuscol],
                    "substatus": c[subcol],
                }
            )

            def key(frame):
                return collections.Counter(frame.itertuples(index=False, name=None))

            results[fam] = {
                "member": name,
                "rows": len(c),
                "case_ids": c[idcol].nunique(),
                "case_id_sets_equal": set(c[idcol]) == set(d.case_id),
                "case_timestamp_status_substatus_multiset_equal": key(xs) == key(cs),
                "csv_team_nonempty_rows": int(c[teamcol].ne("").sum()),
                "csv_team_distinct": int(c[teamcol].nunique()),
                "XES_team_available": bool(d.team.notna().any()),
                "csv_org_distinct": int(c[orgcol].nunique()),
                "headers": list(c.columns),
            }
    return results


def run(args):
    cfg = json.loads((HERE / "illustrative_norm.json").read_text())
    norm_hash = sha(HERE / "illustrative_norm.json")
    core = args.project_root / "code/wise_reference.py"
    assert sha(core) == cfg["core_sha256"], "Canonical arithmetic changed; version/review required"
    sys.path.insert(0, str(args.project_root / "code"))
    from wise_reference import (
        coco_same_target,
        group_priorities,
        score_assessments,
        signed_components,
    )

    out = args.output.resolve()
    if out == args.input_dir.resolve() or args.input_dir.resolve() in out.parents:
        raise ValueError("Output must be outside the source dataset directory")
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "manifest.json", {"status": "incomplete", "norm_sha256": norm_hash})
    dump(out / "configuration.json", cfg)
    all_d = {}
    case_sets = {}
    summaries = []
    schemas = {}
    checks = []
    view_comparisons = []
    top_metadata = []
    for fam in FAMILIES:
        p = args.input_dir / f"bpi_challenge_2013_{fam}.xes.gz"
        assert sha(p) == cfg["expected"][fam]["sha256"], fam + " input changed"
        t, d, schema = parse_xes(p)
        assert len(t) == cfg["expected"][fam]["cases"]
        assert len(d) == cfg["expected"][fam]["events"]
        f, d = features(d, fam, cfg)
        all_d[fam] = d
        case_sets[fam] = set(f.index)
        schemas[fam] = schema
        dest = out / fam
        dest.mkdir(exist_ok=True)
        f.to_parquet(dest / "case_features.parquet")
        d.to_parquet(dest / "event_evidence.parquet", index=False)
        V, A, R = assess(f, fam, cfg)
        V.add_prefix("severity__").join(A.add_prefix("in_scope__")).join(R.add_prefix("state__")).to_parquet(
            dest / "assessments.parquet"
        )
        R.melt(ignore_index=False, var_name="criterion", value_name="state").groupby(["criterion", "state"]).size().rename(
            "cases"
        ).to_csv(dest / "criterion_states.csv")
        layers = {c["id"]: c["layer"] for c in cfg["criteria"]}
        scored_views = {}
        tables = {}
        tops = {}
        for view, weights_by_layer in cfg["views"].items():
            w = pd.Series({c["id"]: weights_by_layer[c["layer"]] for c in cfg["criteria"]})
            s = score_assessments(V, A, w, layers, mode=cfg["primary_mode"])
            scored_views[view] = s
            tab = group_priorities(s["score"], f.product_group)
            tables[view] = tab
            tab.to_csv(dest / f"groups_{view}.csv")
            group_priorities(s["score"], f.organization_group).to_csv(dest / f"organizations_{view}.csv")
            top, meta = pick(tab)
            tops[view] = top
            top_metadata.append({"dataset": fam, "view": view, **meta})
            top.to_csv(dest / f"top10_{view}.csv", index=False)
            layer = pd.DataFrame(
                {
                    layer_name: s["penalties"][[c for c in layers if layers[c] == layer_name]].sum(axis=1, min_count=1)
                    for layer_name in dict.fromkeys(layers.values())
                }
            )
            signed = signed_components(layer, f.product_group)
            signed.to_csv(dest / f"signed_layers_{view}.csv")
            residual = float((signed.sum(axis=1).clip(lower=0) - tab.PI).abs().max())
            coco = coco_same_target(s["penalty"], f.product_group)
            cr = float((tab.PI - coco.D * coco.CoCo.clip(lower=0)).abs().max())
            reconstruction = float((s["penalties"].sum(axis=1) - s["penalty"]).abs().max())
            weight_error = float((s["effective_weights"].sum(axis=1) - 1).abs().max())
            bound_error = float(
                ((s["flat_completion_bounds"].penalty_max - s["flat_completion_bounds"].penalty_min) - (1 - s["coverage"]))
                .abs()
                .max()
            )
            assert max(residual, cr, reconstruction, weight_error, bound_error) < 1e-8
            checks.append(
                {
                    "dataset": fam,
                    "view": view,
                    "penalty_residual": reconstruction,
                    "weight_residual": weight_error,
                    "signed_PI_residual": residual,
                    "coco_residual": cr,
                    "flat_bounds_width_residual": bound_error,
                }
            )
            lead = top.iloc[0] if len(top) else None
            summaries.append(
                {
                    "dataset": fam,
                    "view": view,
                    "cases": len(f),
                    "events": len(d),
                    "groups": len(tab),
                    "scored": int(s["score"].notna().sum()),
                    "mean_penalty": float(s["penalty"].mean()),
                    "mean_coverage": float(s["coverage"].mean()),
                    "positive_groups": meta["positive_candidates"],
                    "top_k": meta["selected_k"],
                    "lead_group": str(lead.iloc[0]) if lead is not None else None,
                    "lead_PI": float(lead.PI) if lead is not None else None,
                }
            )
        case_scores = pd.DataFrame(
            {v + "__" + k: scored_views[v][k] for v in scored_views for k in ["score", "penalty", "coverage"]}
        )
        for v in scored_views:
            for b in ["penalty_min", "penalty_max"]:
                case_scores[v + "__" + b] = scored_views[v]["flat_completion_bounds"][b]
        case_scores.to_parquet(dest / "case_scores.parquet")
        for v in ["Routing", "Timing"]:
            view_comparisons.append(
                {
                    "dataset": fam,
                    "comparison": "view_" + v,
                    "kind": "same input population, view weights change; reference recomputed",
                    "top10_jaccard": overlap(tops["Balanced"], tops[v]),
                }
            )
        base = scored_views["Balanced"]
        base_top = tops["Balanced"]
        sensitivity = []
        w = pd.Series({c["id"]: cfg["views"]["Balanced"][c["layer"]] for c in cfg["criteria"]})
        scenarios = [
            ("span_threshold_width_doubled", *assess(f, fam, cfg, 2)[:2], "flat", w),
            ("layer_balanced", V, A, "layer_balanced", w),
        ]
        omit = w.copy()
        omit["C03"] = 0
        scenarios.append(("omit_repeated_substatus", V, A, "flat", omit))
        for name, v, a, mode, weights in scenarios:
            s = score_assessments(v, a, weights, layers, mode=mode)
            tab = group_priorities(s["score"], f.product_group)
            top, meta = pick(tab)
            sensitivity.append(
                {
                    "scenario": name,
                    "kind": "assessment changes; reference recomputed",
                    "cases": len(f),
                    "scored_cases": int(s["score"].notna().sum()),
                    "unscored_cases": int(s["score"].isna().sum()),
                    "mean_penalty": float(s["penalty"].mean()),
                    "mean_coverage": float(s["coverage"].mean()),
                    "top10_jaccard": overlap(base_top, top),
                    **meta,
                }
            )
        tab = group_priorities(base["score"], f.product_group, gamma=20)
        top, meta = pick(tab, key="stable_PI")
        sensitivity.append(
            {
                "scenario": "gamma20",
                "kind": "same target population and reference; rank policy changes",
                "cases": len(f),
                "scored_cases": int(base["score"].notna().sum()),
                "unscored_cases": int(base["score"].isna().sum()),
                "mean_penalty": float(base["penalty"].mean()),
                "mean_coverage": float(base["coverage"].mean()),
                "top10_jaccard": overlap(base_top, top),
                **meta,
            }
        )
        pd.DataFrame(sensitivity).to_csv(dest / "sensitivity.csv", index=False)
        # Trace witness maximises positive Routing-C02 severity within the top Balanced product,
        # then largest Balanced penalty, then lexical ID; no claim of representativeness.
        lead = tops["Balanced"].iloc[0, 0]
        eligible = f.index[f.product_group.eq(lead)]
        choose = (
            pd.DataFrame(
                {
                    "case_id": eligible,
                    "C02": V.loc[eligible, "C02"].fillna(-1).to_numpy(),
                    "q": base["penalty"].loc[eligible].to_numpy(),
                }
            )
            .sort_values(["C02", "q", "case_id"], ascending=[False, False, True], kind="stable")
            .iloc[0]
            .case_id
        )
        d.loc[d.case_id.eq(choose)].to_csv(dest / "witness_events.csv", index=False)
        witness = pd.DataFrame(
            {
                "severity": V.loc[choose],
                "in_scope": A.loc[choose],
                "state": R.loc[choose],
                "effective_weight": base["effective_weights"].loc[choose],
                "penalty": base["penalties"].loc[choose],
            }
        )
        witness.to_csv(dest / "witness_assessment.csv")
        dump(
            dest / "witness.json",
            {
                "case_id": choose,
                "product_group": str(lead),
                "selection": "In highest Balanced-PI product: descending C02 severity (unknown last), then Balanced penalty, lexical case ID",
                "score": float(base["score"].loc[choose]),
                "coverage": float(base["coverage"].loc[choose]),
                "source_trace_position": int(d.loc[d.case_id.eq(choose), "trace_position"].iloc[0]),
            },
        )
        print(fam, len(f), len(d), "complete", flush=True)
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "verification.csv", index=False)
    pd.DataFrame(view_comparisons).to_csv(out / "view_comparisons.csv", index=False)
    pd.DataFrame(top_metadata).to_csv(out / "top10_metadata.csv", index=False)
    dump(out / "schema_audit.json", schemas)
    cross = {
        "open_closed_case_id_overlap": len(case_sets["open_problems"] & case_sets["closed_problems"]),
        "family_case_counts": {k: len(v) for k, v in case_sets.items()},
        "pooled": False,
        "open_source_cases_with_recorded_Closed": int(
            all_d["open_problems"].groupby("case_id").substatus.apply(lambda s: s.eq("Closed").any()).sum()
        ),
    }
    dump(out / "population_audit.json", cross)
    dump(out / "csv_xes_audit.json", csv_audit(args.input_dir / "csv_files.zip", all_d))
    assert sha(HERE / "illustrative_norm.json") == norm_hash
    assert sha(core) == cfg["core_sha256"]
    dump(
        out / "manifest.json",
        {
            "status": "passed",
            "norm_sha256": norm_hash,
            "script_sha256": sha(Path(__file__)),
            "wise_reference_sha256": sha(core),
            "input_directory": str(args.input_dir),
            "project_root": str(args.project_root),
            "expected_inputs": cfg["expected"],
            "supplementary_input_hashes": {
                n: sha(args.input_dir / n) for n in ["csv_files.zip", "vinst_data_set.pdf", "vinst_manual.pdf"]
            },
            "full_logs": True,
            "pooled": False,
            "checks": checks,
            "population_audit": cross,
            "scope": "Independent illustrative applications; means/PI values are not calibrated across logs",
            "runtime": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--project-root", type=Path, required=True)
    run(p.parse_args())
