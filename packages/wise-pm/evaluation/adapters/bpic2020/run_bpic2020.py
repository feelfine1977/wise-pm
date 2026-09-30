#!/usr/bin/env python3
"""BPIC2020 Domestic adapter: explicit assessment; unchanged WISE reference core.
No supplied Python/pickle is loaded. XML entities/DTDs are rejected.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import platform
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from defusedxml.common import DefusedXmlException
from defusedxml.ElementTree import fromstring, iterparse

HERE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def finite(value):
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def attrs(el):
    out = {}
    for child in el:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag in {"string", "date", "int", "float", "boolean", "id"}:
            key = child.attrib.get("key")
            if key is None or key in out:
                raise ValueError("Missing/duplicate XML attribute key")
            out[key] = child.attrib.get("value")
    return out


def parse_xes(path):
    traces = []
    events = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        for _, el in iterparse(
            f,
            events=["end"],
            forbid_dtd=True,
            forbid_entities=True,
            forbid_external=True,
        ):
            if el.tag.rsplit("}", 1)[-1] != "trace":
                continue
            a = attrs(el)
            case = a.get("concept:name")
            if not case:
                raise ValueError("Missing trace concept:name")
            traces.append({"case_id": case, **a})
            position = 0
            for child in el:
                if child.tag.rsplit("}", 1)[-1] != "event":
                    continue
                b = attrs(child)
                if not b.get("id") or not b.get("concept:name"):
                    raise ValueError("Missing event ID/activity label")
                events.append({"case_id": case, "source_order": position, **b})
                position += 1
            if position == 0:
                raise ValueError("Empty trace requires a separately declared policy")
            el.clear()
    c = pd.DataFrame(traces).set_index("case_id")
    e = pd.DataFrame(events)
    if not c.index.is_unique or e["id"].duplicated().any():
        raise ValueError("Duplicate trace/event identity")
    e["timestamp_utc"] = pd.to_datetime(e.get("time:timestamp"), utc=True, errors="coerce")
    schema = {
        "trace_fields": {
            k: {
                "present": int(c[k].notna().sum()),
                "distinct": int(c[k].nunique(dropna=False)),
            }
            for k in c
        },
        "event_fields": {
            k: {
                "present": int(e[k].notna().sum()),
                "distinct": int(e[k].nunique(dropna=False)),
            }
            for k in e
            if k not in ["timestamp_utc", "source_order"]
        },
        "activities": {str(k): int(v) for k, v in e["concept:name"].value_counts().items()},
        "event_resources": {str(k): int(v) for k, v in e["org:resource"].value_counts(dropna=False).items()},
        "event_roles": {str(k): int(v) for k, v in e["org:role"].value_counts(dropna=False).items()},
        "cases": len(c),
        "events": len(e),
        "invalid_or_missing_timestamps": int(e.timestamp_utc.isna().sum()),
        "timestamp_min_utc": str(e.timestamp_utc.min()),
        "timestamp_max_utc": str(e.timestamp_utc.max()),
    }
    return c, e, schema


def make_features(c, e, cfg):
    f = pd.DataFrame(index=c.index)
    g = e.groupby("case_id", sort=False)
    f["n_events"] = g.size().reindex(c.index)
    for family, labels in cfg["activity_families"].items():
        if family == "rejection_prefix":
            continue
        sub = e.loc[e["concept:name"].isin(labels)]
        s = sub.groupby("case_id", sort=False)
        f["n_" + family] = s.size().reindex(c.index, fill_value=0)
        f["first_" + family] = s.timestamp_utc.min().reindex(c.index)
        f["invalid_" + family] = sub.timestamp_utc.isna().groupby(sub.case_id).sum().reindex(c.index, fill_value=0)
    rejections = e.loc[e["concept:name"].str.startswith(cfg["activity_families"]["rejection_prefix"])]
    f["n_reject"] = rejections.groupby("case_id").size().reindex(c.index, fill_value=0)
    f["amount"] = pd.to_numeric(c.get("Amount"), errors="coerce")
    f["amount_band"] = "missing_or_invalid"
    a = f.amount
    f.loc[a.eq(0), "amount_band"] = "0: zero"
    edges = cfg["amount_band_edges"]
    for low, high in pairwise(edges):
        f.loc[a.gt(low) & a.le(high), "amount_band"] = f"({low}, {high}]"
    f.loc[np.isfinite(a) & a.gt(edges[-1]), "amount_band"] = f"> {edges[-1]}"
    valid_start = f.n_submit.gt(0) & f.invalid_submit.eq(0)
    f["first_submission_quarter"] = "missing_or_invalid_submission_time"
    f.loc[valid_start, "first_submission_quarter"] = (
        f.loc[valid_start, "first_submit"].dt.tz_localize(None).dt.to_period("Q").astype(str)
    )
    f["submission_year"] = f.first_submit.dt.year.where(valid_start)
    return f


def assess(f, cfg):
    ids = [x["id"] for x in cfg["criteria"]]
    v = pd.DataFrame(np.nan, index=f.index, columns=ids)
    raw = v.copy()
    scope = pd.DataFrame(True, index=f.index, columns=ids)
    state = pd.DataFrame("evaluated", index=f.index, columns=ids)
    for rule in cfg["criteria"]:
        cid = rule["id"]
        sc = f[rule["scope_feature"]].gt(0) if "scope_feature" in rule else pd.Series(True, index=f.index)
        scope[cid] = sc
        if rule["kind"] == "duration_ramp":
            st, en = rule["start"], rule["end"]
            missing = f["n_" + st].eq(0) | f["n_" + en].eq(0)
            invalidstamp = f["invalid_" + st].gt(0) | f["invalid_" + en].gt(0)
            val = (f["first_" + en] - f["first_" + st]).dt.total_seconds() / 86400
            raw[cid] = val
            state.loc[missing, cid] = "missing_prerequisite"
            state.loc[~missing & invalidstamp, cid] = "missing_or_invalid_timestamp"
            state.loc[~missing & ~invalidstamp & val.lt(0), cid] = "invalid_temporal_order"
            v[cid] = ((val - rule["threshold"]) / rule["width"]).clip(0, 1)
        elif rule["kind"] == "absence":
            raw[cid] = f[rule["feature"]].eq(0).astype(float)
            v[cid] = raw[cid]
        else:
            raw[cid] = f[rule["feature"]].astype(float)
            v[cid] = ((raw[cid] - rule["threshold"]) / rule["width"]).clip(0, 1)
        state.loc[~sc, cid] = "out_of_scope"
        v.loc[state[cid].ne("evaluated"), cid] = np.nan
    return v, scope, state, raw


def score(ref, assessment, cfg, view, mode="flat"):
    v, a, _, _ = assessment
    layers = {x["id"]: x["layer"] for x in cfg["criteria"]}
    w = pd.Series({cid: cfg["views"][view][layers[cid]] for cid in v.columns}, dtype=float)
    return ref.score_assessments(v, a, w, layers, mode=mode)


def top_positive(groups, k, field="PI", eps=1e-12):
    return groups.loc[groups[field].gt(eps)].sort_index().sort_values([field, "n_cases"], ascending=False, kind="stable").head(k)


def agreement(base, other, k):
    a = set(top_positive(base, k).index)
    b = set(top_positive(other, k).index)
    return {
        "requested_k": k,
        "base_positive_selected": len(a),
        "other_positive_selected": len(b),
        "shared": len(a & b),
        "jaccard": len(a & b) / len(a | b) if a | b else None,
        "base_positive_candidates": int(base.PI.gt(1e-12).sum()),
        "other_positive_candidates": int(other.PI.gt(1e-12).sum()),
    }


def export_assessment(assessment, results, path):
    v, a, s, raw = assessment
    blocks = [
        v.add_prefix("severity__"),
        a.add_prefix("in_scope__"),
        s.add_prefix("state__"),
        raw.add_prefix("raw__"),
    ]
    for view, res in results.items():
        blocks.extend(
            [
                res["penalties"].add_prefix("component__" + view + "__"),
                res["effective_weights"].add_prefix("weight__" + view + "__"),
            ]
        )
    pd.concat(blocks, axis=1).to_parquet(path, compression="zstd")


def scalar_expected(trace, rule, cfg):
    """Independent row-wise raw-event calculation; never reads case features."""
    acts = [x["concept:name"] for x in trace]
    counts = {
        fam: sum(a in labels for a in acts) for fam, labels in cfg["activity_families"].items() if fam != "rejection_prefix"
    }
    counts["reject"] = sum(a.startswith(cfg["activity_families"]["rejection_prefix"]) for a in acts)
    sc = True if "scope_feature" not in rule else counts[rule["scope_feature"][2:]] > 0
    if not sc:
        return False, "out_of_scope", None
    if rule["kind"] == "duration_ramp":
        ends = []
        for family in [rule["start"], rule["end"]]:
            rr = [x for x in trace if x["concept:name"] in cfg["activity_families"][family]]
            if not rr:
                return True, "missing_prerequisite", None
            ts = [pd.to_datetime(x.get("time:timestamp"), utc=True, errors="coerce") for x in rr]
            ends.append(ts)
        if any(pd.isna(t) for ts in ends for t in ts):
            return True, "missing_or_invalid_timestamp", None
        val = (min(ends[1]) - min(ends[0])).total_seconds() / 86400
        if val < 0:
            return True, "invalid_temporal_order", None
        return (
            True,
            "evaluated",
            min(1, max(0, (val - rule["threshold"]) / rule["width"])),
        )
    val = counts[rule["feature"][2:]]
    return (
        True,
        "evaluated",
        float(val == 0) if rule["kind"] == "absence" else min(1, max(0, (val - rule["threshold"]) / rule["width"])),
    )


def contract_fixtures(cfg, ref):
    sub, ap, req, paid = [cfg["activity_families"][f][0] for f in ["submit", "final_approval", "request", "handled"]]
    traces = {
        "F01": [(sub, 0), (ap, 1), (req, 2), (paid, 3)],
        "F02": [(sub, 0), (sub, 1), (sub, 2), (ap, 30), (req, 31), (paid, 33)],
        "F03": [(sub, 0)],
        "F04": [(sub, 0), (ap, 1), (req, 2)],
        "F05": [(sub, 0), (ap, 1), (req, 2), (paid, None)],
        "F06": [(sub, 0), (ap, 1), (req, 10), (paid, 5)],
        "F07": [(sub, 0), (ap, 1), (req, 2), (paid, 3), (paid, 4)],
        "F08": [(sub, 0), (sub, None), (ap, 2), (req, 3), (paid, 4)],
        "F09": [
            ("Declaration REJECTED by EMPLOYEE", 0),
            ("Declaration REJECTED by ADMINISTRATION", 1),
        ],
        "F10": [(sub, 0), (ap, 0), (req, 0), (paid, 0)],
        "F11": [(sub, 0), (ap, 1), (req, 2), (paid, None), (paid, 5)],
    }
    rows = []
    for cid, items in traces.items():
        for i, (act, d) in enumerate(items):
            rows.append(
                {
                    "case_id": cid,
                    "id": f"{cid}_{i}",
                    "source_order": i,
                    "concept:name": act,
                    "time:timestamp": None if d is None else str(pd.Timestamp("2018-01-01", tz="UTC") + pd.Timedelta(days=d)),
                }
            )
    e = pd.DataFrame(rows)
    e["timestamp_utc"] = pd.to_datetime(e["time:timestamp"], utc=True, errors="coerce")
    c = pd.DataFrame({"Amount": 100}, index=pd.Index(traces, name="case_id"))
    f = make_features(c, e, cfg)
    a = assess(f, cfg)
    res = score(ref, a, cfg, "Balanced")
    checks = []
    expected = [
        ("F01", "D03", "evaluated", 0),
        ("F02", "D01", "evaluated", 1),
        ("F02", "D03", "evaluated", 4 / 7),
        ("F03", "D03", "missing_prerequisite", None),
        ("F03", "D04", "out_of_scope", None),
        ("F03", "D06", "out_of_scope", None),
        ("F04", "D05", "missing_prerequisite", None),
        ("F04", "D06", "evaluated", 1),
        ("F05", "D05", "missing_or_invalid_timestamp", None),
        ("F05", "D06", "evaluated", 0),
        ("F06", "D05", "invalid_temporal_order", None),
        ("F07", "D07", "evaluated", 1),
        ("F08", "D03", "missing_or_invalid_timestamp", None),
        ("F08", "D01", "evaluated", 0.5),
        ("F09", "D02", "evaluated", 1),
        ("F10", "D05", "evaluated", 0),
        ("F11", "D05", "missing_or_invalid_timestamp", None),
        ("F11", "D06", "evaluated", 0),
        ("F11", "D07", "evaluated", 1),
    ]
    for cid, col, st, val in expected:
        ok = a[2].loc[cid, col] == st and (pd.isna(a[0].loc[cid, col]) if val is None else abs(a[0].loc[cid, col] - val) < 1e-12)
        checks.append(
            {
                "fixture": cid,
                "criterion": col,
                "expected_state": st,
                "expected_severity": val,
                "passed": bool(ok),
            }
        )
    for cid, key, val in [
        ("F01", "penalty", 0),
        ("F03", "coverage", 3 / 4),
        ("F04", "penalty", 1 / 6),
        ("F05", "coverage", 6 / 7),
        ("F07", "penalty", 1 / 7),
        ("F09", "penalty", 1 / 3),
    ]:
        checks.append(
            {
                "fixture": cid,
                "quantity": key,
                "expected": val,
                "passed": bool(abs(res[key].loc[cid] - val) < 1e-12),
            }
        )
    try:
        fromstring(
            b'<!DOCTYPE x [<!ENTITY e "boom">]><x>&e;</x>',
            forbid_dtd=True,
            forbid_entities=True,
            forbid_external=True,
        )
        blocked = False
    except DefusedXmlException:
        blocked = True
    checks.append({"fixture": "DTD_entity_rejection", "passed": blocked})
    assert all(x["passed"] for x in checks), checks
    return e, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--project-root", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=HERE / "norm_frozen.json")
    args = ap.parse_args()
    inp = args.input
    if inp.is_dir():
        choices = list(inp.glob("*.xes.gz")) + list(inp.glob("*.xes"))
        if len(choices) != 1:
            raise ValueError("Input directory must contain exactly one XES/XES.gz")
        inp = choices[0]
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text())
    reference_path = args.project_root / "code/wise_reference.py"
    reference_hash = sha(reference_path)
    spec = importlib.util.spec_from_file_location("wise_transfer_reference", reference_path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    if sha(inp) != cfg["expected_compressed_sha256"]:
        raise ValueError("Input hash differs from frozen dataset; declare a new configuration before running")
    write_json(
        out / "manifest.json",
        {
            "status": "running",
            "input": str(inp),
            "config_sha256": sha(args.config),
            "reference_sha256": reference_hash,
        },
    )
    c, e, schema = parse_xes(inp)
    assert len(c) == cfg["expected_cases"]
    assert len(e) == cfg["expected_events"]
    f = make_features(c, e, cfg)
    assessment = assess(f, cfg)
    v, scope, state, raw = assessment
    c.to_parquet(out / "cases.parquet", compression="zstd")
    e.to_parquet(out / "events.parquet", compression="zstd", index=False)
    f.to_parquet(out / "case_features.parquet", compression="zstd")
    write_json(out / "schema.json", schema)
    write_json(out / "configuration.json", cfg)
    rules = {x["id"]: x for x in cfg["criteria"]}
    results = {}
    groups_all = {}
    summaries = []
    checks = []
    allscores = []
    for view in cfg["views"]:
        res = score(ref, assessment, cfg, view)
        results[view] = res
        allscores.append(
            pd.DataFrame(
                {
                    k: res[k]
                    for k in [
                        "score",
                        "penalty",
                        "coverage",
                        "in_scope_count",
                        "evaluated_count",
                    ]
                }
            ).add_prefix(view + "__")
        )
        summaries.append(
            {
                "view": view,
                "cases": len(f),
                "scored": int(res["score"].notna().sum()),
                "mean_concern": float(res["penalty"].mean()),
                "mean_coverage": float(res["coverage"].mean()),
                "partial_evidence_cases": int(res["coverage"].lt(1 - 1e-12).sum()),
            }
        )
        checks.append(
            {
                "view": view,
                "check": "effective_weight_sum",
                "max_residual": float((res["effective_weights"].sum(axis=1) - 1).abs().max()),
            }
        )
        checks.append(
            {
                "view": view,
                "check": "case_penalty_reconstruction",
                "max_residual": float((res["penalties"].sum(axis=1) - res["penalty"]).abs().max()),
            }
        )
        for key in ["amount_band", "first_submission_quarter"]:
            groups = ref.group_priorities(res["score"], f[key])
            groups_all[(view, key)] = groups
            groups.to_csv(out / f"groups_{view}_{key}.csv")
            signed = ref.signed_components(res["penalties"], f[key])
            signed.to_csv(out / f"signed_criteria_{view}_{key}.csv")
            layer = pd.DataFrame(
                {ly: signed[[rid for rid in signed if rules[rid]["layer"] == ly]].sum(axis=1) for ly in cfg["views"][view]},
                index=signed.index,
            )
            layer.to_csv(out / f"signed_layers_{view}_{key}.csv")
            coco = ref.coco_same_target(res["penalty"], f[key])
            coco.to_csv(out / f"coco_{view}_{key}.csv")
            checks.append(
                {
                    "view": view,
                    "grouping": key,
                    "check": "signed_PI_reconstruction",
                    "max_residual": float((signed.sum(axis=1).clip(lower=0) - groups.PI.reindex(signed.index)).abs().max()),
                }
            )
            if coco.D.iloc[0] > 0:
                checks.append(
                    {
                        "view": view,
                        "grouping": key,
                        "check": "coco_identity",
                        "max_residual": float((coco.CoCo.clip(lower=0) * coco.D - groups.PI.reindex(coco.index)).abs().max()),
                    }
                )
            diag = ref.coverage_diagnostics(res, f[key])
            for kind, table in diag.items():
                table.to_csv(out / f"coverage_{kind}_{view}_{key}.csv")
    pd.DataFrame(summaries).to_csv(out / "view_summary.csv", index=False)
    pd.concat(allscores, axis=1).to_parquet(out / "case_scores.parquet", compression="zstd")
    export_assessment(assessment, results, out / "case_assessments.parquet")
    states = []
    for cid in v:
        for st, n in state[cid].value_counts().items():
            states.append({"criterion": cid, "state": st, "n": int(n)})
    pd.DataFrame(states).to_csv(out / "criterion_states.csv", index=False)
    stats = []
    for cid in v:
        vals = v[cid]
        stats.append(
            {
                "criterion": cid,
                "name": rules[cid]["name"],
                "in_scope": int(scope[cid].sum()),
                "evaluated": int(vals.notna().sum()),
                "positive": int(vals.gt(0).sum()),
                "mean_severity_when_evaluated": float(vals.mean()),
                "raw_min": finite(raw[cid].min()),
                "raw_median": finite(raw[cid].median()),
                "raw_max": finite(raw[cid].max()),
            }
        )
    pd.DataFrame(stats).to_csv(out / "criterion_summary.csv", index=False)
    context = f.assign(
        recorded_submission=f.n_submit.gt(0),
        recorded_final_approval=f.n_final_approval.gt(0),
        recorded_payment_request=f.n_request.gt(0),
        recorded_payment_handled=f.n_handled.gt(0),
        recorded_rejection=f.n_reject.gt(0),
        in_scope_count=scope.sum(axis=1),
        evaluated_count=v.notna().sum(axis=1),
        coverage=results["Balanced"]["coverage"],
    )
    profile = context.groupby("amount_band").agg(
        cases=("n_events", "size"),
        submitted=("recorded_submission", "sum"),
        final_approved=("recorded_final_approval", "sum"),
        requested=("recorded_payment_request", "sum"),
        handled=("recorded_payment_handled", "sum"),
        rejected=("recorded_rejection", "sum"),
        mean_in_scope=("in_scope_count", "mean"),
        mean_evaluated=("evaluated_count", "mean"),
        mean_coverage=("coverage", "mean"),
    )
    profile.to_csv(out / "amount_band_context.csv")
    rows = []
    for band, idx in f.groupby("amount_band").groups.items():
        for cid in v:
            rows.append(
                {
                    "amount_band": band,
                    "criterion": cid,
                    "cases": len(idx),
                    "in_scope": int(scope.loc[idx, cid].sum()),
                    "evaluated": int(v.loc[idx, cid].notna().sum()),
                    "positive": int(v.loc[idx, cid].gt(0).sum()),
                }
            )
    pd.DataFrame(rows).to_csv(out / "criterion_by_amount_band.csv", index=False)
    # Freeze-respecting sensitivity: no norm tuning after rankings.
    base = groups_all[("Balanced", "amount_band")]
    comparisons = []
    k = cfg["top_k"]
    for view in cfg["views"]:
        comparisons.append(
            {
                "scenario": view,
                "type": "valuation",
                "cases": len(f),
                "reference_score": float(results[view]["score"].mean()),
                **agreement(base, groups_all[(view, "amount_band")], k),
            }
        )
    for threshold in cfg["sensitivity"]["D03_threshold_days"]:
        alt = copy.deepcopy(cfg)
        next(x for x in alt["criteria"] if x["id"] == "D03")["threshold"] = threshold
        rr = score(ref, assess(f, alt), alt, "Balanced")
        gg = ref.group_priorities(rr["score"], f.amount_band)
        gg.to_csv(out / f"sensitivity_D03_threshold_{threshold}.csv")
        comparisons.append(
            {
                "scenario": f"D03_threshold_{threshold}",
                "type": "assessment",
                "cases": len(f),
                "reference_score": float(rr["score"].mean()),
                **agreement(base, gg, k),
            }
        )
    rr = score(ref, assessment, cfg, "Balanced", mode="layer_balanced")
    gg = ref.group_priorities(rr["score"], f.amount_band)
    gg.to_csv(out / "sensitivity_layer_balanced.csv")
    comparisons.append(
        {
            "scenario": "layer_balanced",
            "type": "assessment",
            "cases": len(f),
            "reference_score": float(rr["score"].mean()),
            **agreement(base, gg, k),
        }
    )
    for name, mask in {
        "submission_calendar_2018": f.submission_year.eq(2018),
        "recorded_payment_handled": f.n_handled.gt(0),
        "complete_in_scope_evidence": results["Balanced"]["coverage"].ge(1 - 1e-12),
    }.items():
        if not mask.any():
            continue
        gg = ref.group_priorities(results["Balanced"]["score"].loc[mask], f.amount_band.loc[mask])
        gg.to_csv(out / f"population_{name}.csv")
        comparisons.append(
            {
                "scenario": name,
                "type": "population_change",
                "cases": int(mask.sum()),
                "reference_score": float(results["Balanced"]["score"].loc[mask].mean()),
                **agreement(base, gg, k),
            }
        )
    gamma = cfg["sensitivity"]["gamma"]
    gg = ref.group_priorities(results["Balanced"]["score"], f.amount_band, gamma=gamma)
    gg.to_csv(out / f"policy_gamma_{gamma}.csv")
    gg2 = gg.copy()
    gg2["PI"] = gg.stable_PI
    comparisons.append(
        {
            "scenario": f"gamma_{gamma}",
            "type": "fixed_target_policy",
            "cases": len(f),
            "reference_score": float(results["Balanced"]["score"].mean()),
            **agreement(base, gg2, k),
        }
    )
    pd.DataFrame(comparisons).to_csv(out / "sensitivity.csv", index=False)
    # Witness: positive dominant component in the primary leading positive group.
    lead = top_positive(base, 1).index[0]
    signed = pd.read_csv(out / "signed_criteria_Balanced_amount_band.csv", index_col=0)
    dominant = signed.loc[lead].idxmax()
    candidates = f.index[f.amount_band.eq(lead) & v[dominant].gt(0)]
    ordered = results["Balanced"]["penalties"].loc[candidates, dominant].sort_index().sort_values(ascending=False, kind="stable")
    wcase = ordered.index[0]
    e.loc[e.case_id.eq(wcase)].to_csv(out / "trace_witness_events.csv", index=False)
    wa = pd.DataFrame(
        {
            "scope": scope.loc[wcase],
            "state": state.loc[wcase],
            "raw": raw.loc[wcase],
            "severity": v.loc[wcase],
            "effective_weight": results["Balanced"]["effective_weights"].loc[wcase],
            "component": results["Balanced"]["penalties"].loc[wcase],
        }
    )
    wa.index.name = "criterion"
    wa.to_csv(out / "trace_witness_assessment.csv")
    witness = {
        "case_id": wcase,
        "amount_band": lead,
        "dominant_group_criterion": dominant,
        "selection_rule": "Leading positive Balanced amount group; maximum contribution to its largest positive signed criterion; lexical case ties",
        "case_penalty": float(results["Balanced"]["penalty"].loc[wcase]),
        "case_coverage": float(results["Balanced"]["coverage"].loc[wcase]),
        "not_representative_or_causal": True,
    }
    write_json(out / "trace_witness.json", witness)
    # Reparse the original compressed input, independently calculating scalar expectations.
    selected = {wcase}
    example_rows = []
    for cid in v:
        for label, mask in [("positive", v[cid].gt(0)), ("zero", v[cid].eq(0))] + [
            (st, state[cid].eq(st)) for st in state[cid].unique() if st != "evaluated"
        ]:
            ids = sorted(v.index[mask])
            pick = ids[0] if ids else None
            if pick:
                selected.add(pick)
            example_rows.append(
                {
                    "criterion": cid,
                    "example_state": label,
                    "matching_cases": len(ids),
                    "case_id": pick,
                }
            )
    pd.DataFrame(example_rows).to_csv(out / "criterion_examples.csv", index=False)
    _, raw_events, _ = parse_xes(inp)
    audit = []
    for case in sorted(selected):
        rows = raw_events.loc[raw_events.case_id.eq(case)].to_dict("records")
        for rule in cfg["criteria"]:
            sc, st, val = scalar_expected(rows, rule, cfg)
            cid = rule["id"]
            got = v.loc[case, cid]
            ok = (
                sc == scope.loc[case, cid]
                and st == state.loc[case, cid]
                and (pd.isna(got) if val is None else abs(got - val) < 1e-12)
            )
            audit.append(
                {
                    "case_id": case,
                    "criterion": cid,
                    "expected_scope": sc,
                    "expected_state": st,
                    "expected_severity": val,
                    "passed": bool(ok),
                }
            )
    assert all(x["passed"] for x in audit)
    pd.DataFrame(audit).to_csv(out / "source_witness_checks.csv", index=False)
    raw_events.loc[raw_events.case_id.isin(selected)].to_parquet(
        out / "source_witness_events.parquet", compression="zstd", index=False
    )
    fix, fc = contract_fixtures(cfg, ref)
    fix.to_csv(out / "contract_fixture_events.csv", index=False)
    write_json(
        out / "contract_fixture_checks.json",
        {
            "label": "Synthetic explicit contract expectations, not business validation",
            "traces": int(fix.case_id.nunique()),
            "checks": len(fc),
            "passed": sum(x["passed"] for x in fc),
            "details": fc,
        },
    )
    pd.DataFrame(checks).to_csv(out / "numeric_checks.csv", index=False)
    assert max(x["max_residual"] for x in checks) < 1e-9
    assert sha(reference_path) == reference_hash
    summary = {
        "cases": len(c),
        "events": len(e),
        "primary_group_count": len(base),
        "grouping_limitation": "No department/team/vendor/project. BudgetNumber constant. Amount bands are descriptive strata, not organisational ownership/cost impact. Submission-quarter composition differs across years.",
        "criterion_summary": stats,
        "views": summaries,
        "primary_top_positive": [
            {**{"group": idx}, **{k: finite(z) for k, z in row.items()}} for idx, row in top_positive(base, k).iterrows()
        ],
        "sensitivity": comparisons,
        "witness": witness,
        "source_witness_cases": len(selected),
        "source_witness_scalar_checks": len(audit),
        "contract_traces": int(fix.case_id.nunique()),
        "contract_checks": len(fc),
        "max_numeric_residual": max(x["max_residual"] for x in checks),
    }
    write_json(out / "summary.json", summary)
    files = {x.name: sha(x) for x in sorted(out.iterdir()) if x.is_file() and x.name != "manifest.json"}
    with gzip.open(inp, "rb") if str(inp).endswith(".gz") else inp.open("rb") as stream:
        xml_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    readme = inp.parent / "README.txt"
    manifest = {
        "status": "completed",
        "dataset": "BPIC2020 Domestic Declarations",
        "experiment": cfg["id"],
        "input_path": str(inp),
        "compressed_sha256": sha(inp),
        "uncompressed_xml_sha256": xml_hash,
        "provided_readme_sha256": sha(readme) if readme.exists() else None,
        "norm_source_sha256": sha(args.config),
        "effective_configuration_sha256": sha(out / "configuration.json"),
        "script_sha256": sha(__file__),
        "reference_module_path": str(reference_path),
        "reference_sha256": reference_hash,
        "reference_modified": False,
        "cases": len(c),
        "events": len(e),
        "full_log": True,
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "output_sha256": files,
        "limitations": cfg["exclusions"],
    }
    write_json(out / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "completed",
                "cases": len(c),
                "events": len(e),
                "groups": len(base),
                "witness_cases": len(selected),
                "numeric_max_residual": summary["max_numeric_residual"],
                "output": str(out),
            }
        )
    )


if __name__ == "__main__":
    main()
