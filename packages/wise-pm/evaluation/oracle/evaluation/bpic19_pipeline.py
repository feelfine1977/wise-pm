"""Full-log BPIC19 demonstration of an explicit, illustrative assessment contract.

No network use, sample fallback, learned model, causal estimate or source mutation.
The fresh ten-rule norm is distinct from the historical 29-rule WISE experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "code"))
from wise_reference import (
    coco_same_target,
    group_priorities,
    score_assessments,
    signed_components,
)

REFERENCE_HASH_AT_IMPORT = hashlib.sha256((ROOT / "code/wise_reference.py").read_bytes()).hexdigest()
PIPELINE_HASH_AT_IMPORT = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

CASE = "case concept:name"
ACT = "event concept:name"
TS = "event time:timestamp"
DOC = "case Purchasing Document"
RESOURCE = "event org:resource"
GROUP_COLS = ["case Company", "case Spend area text"]
COLS = [
    "eventID",
    CASE,
    ACT,
    TS,
    DOC,
    RESOURCE,
    "case Item",
    "case Company",
    "case Spend area text",
    "case Vendor",
    "case Item Category",
]
DEFAULT_INPUT = None


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def dump_json(path, obj):
    def encode(x):
        if isinstance(x, (np.integer, np.floating, np.bool_)):
            return x.item()
        if isinstance(x, (pd.Timestamp, Path)):
            return str(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x).__name__)

    Path(path).write_text(json.dumps(obj, indent=2, default=encode, allow_nan=False) + "\n")


def read_events(path: Path, cfg: dict):
    if not path.is_file():
        raise FileNotFoundError(f"Full BPIC19 CSV required: {path}. Set --input explicitly; no fallback is used.")
    digest = sha256(path)
    if digest != cfg["expected_csv_sha256"]:
        raise ValueError(
            "CSV hash does not match the audited BPIC19 source. Review input and explicitly version a new configuration."
        )
    d = pd.read_csv(
        path,
        encoding="cp1252",
        dtype=str,
        usecols=lambda c: c.strip() in COLS,
        keep_default_na=False,
    )
    d.columns = d.columns.str.strip()
    if sorted(d.columns) != sorted(COLS):
        raise ValueError("Required BPIC19 fields unavailable")
    d["timestamp"] = pd.to_datetime(d[TS], format=cfg["timestamp_format"], errors="coerce")
    d["plausible_timestamp"] = d["timestamp"].where(
        d["timestamp"].ge(pd.Timestamp(cfg["timing_plausibility_start"]))
        & d["timestamp"].lt(pd.Timestamp(cfg["timing_plausibility_end_exclusive"]))
    )
    resource = d[RESOURCE].str.strip()
    # The audited source uses explicit anonymised user_### / batch_## identities.
    # Unknown nonempty identifiers are not assumed to represent human labour.
    d["human_record"] = resource.str.fullmatch(r"user_[0-9]+", case=False)
    d["resource_usable"] = d["human_record"] | resource.str.fullmatch(r"batch_[0-9]+", case=False)
    assert len(d) == cfg["expected_events"]
    assert d[CASE].nunique() == cfg["expected_cases"]
    assert d[DOC].nunique() == cfg["expected_documents"]
    assert d["eventID"].nunique() == len(d)
    assert (d[CASE] == d[DOC] + "_" + d["case Item"]).all()
    for col in [
        CASE,
        DOC,
        ACT,
        RESOURCE,
        "case Company",
        "case Spend area text",
        "case Vendor",
        "case Item Category",
    ]:
        d[col] = d[col].astype("category")
    return d, digest


def features_from_events(d: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    gb = d.groupby(CASE, observed=True, sort=False)
    attrs = [
        DOC,
        "case Company",
        "case Spend area text",
        "case Vendor",
        "case Item Category",
    ]
    f = gb[attrs].first().astype(str)
    for col in attrs:
        if (gb[col].nunique(dropna=False) > 1).any():
            raise ValueError(f"Case-level attribute varies within case: {col}")
    f.index = f.index.astype(str)
    f.index.name = "case_id"
    f["flow"] = f["case Item Category"].map(cfg["flow_map"])
    if f["flow"].isna().any():
        raise ValueError("Unmapped flow: do not silently assign scope")
    f["n_events"] = gb.size().to_numpy()
    f["first_timestamp"] = gb.timestamp.min().to_numpy()
    f["last_timestamp"] = gb.timestamp.max().to_numpy()
    f["last_plausible_timestamp"] = gb.plausible_timestamp.max().to_numpy()
    f["all_resources_usable"] = gb.resource_usable.all().to_numpy()
    f["human_touches_lower_bound"] = gb.human_record.sum().to_numpy()
    h = d.loc[d.human_record].groupby(CASE, observed=True)[RESOURCE].nunique()
    f["human_resources_lower_bound"] = h.reindex(f.index, fill_value=0)
    activities = {
        "invoice": ["Record Invoice Receipt"],
        "clear": ["Clear Invoice"],
        "receipt": ["Record Goods Receipt", "Record Service Entry Sheet"],
        "cancel_invoice": ["Cancel Invoice Receipt"],
        "edits": ["Change Price", "Change Quantity"],
    }
    for key, labels in activities.items():
        sub = d.loc[d[ACT].isin(labels)]
        sg = sub.groupby(CASE, observed=True)
        f[f"n_{key}"] = sg.size().reindex(f.index, fill_value=0)
        # First recorded timestamp is selected before plausibility filtering; do not
        # silently replace a bad first endpoint with a later valid event.
        first = sg.timestamp.min().reindex(f.index)
        f[f"first_{key}"] = first
        f[f"usable_{key}"] = first.ge(pd.Timestamp(cfg["timing_plausibility_start"])) & first.lt(
            pd.Timestamp(cfg["timing_plausibility_end_exclusive"])
        )
    f["invoice_clear_days"] = (f.first_clear - f.first_invoice).dt.total_seconds() / 86400
    f["mature_heuristic"] = f.last_plausible_timestamp.le(
        pd.Timestamp(cfg["maturity_reference_date"]) - pd.Timedelta(days=cfg["maturity_days"])
    )
    f["recorded_clear"] = f.n_clear.gt(0)
    f["group"] = f[GROUP_COLS].replace("", cfg["missing_group_label"]).agg(" | ".join, axis=1)
    f["vendor"] = f["case Vendor"].replace("", cfg["missing_group_label"])
    return f


def assessment_matrices(
    f: pd.DataFrame,
    cfg: dict,
    *,
    lag_threshold=None,
    resource_policy="complete",
    temporal_filter=True,
):
    ids = [c["id"] for c in cfg["criteria"]]
    rules = {c["id"]: c for c in cfg["criteria"]}
    for cid in ["C04", "C07", "C08", "C09", "C10"]:
        if not np.isfinite(rules[cid]["threshold"]) or not np.isfinite(rules[cid]["width"]) or rules[cid]["width"] <= 0:
            raise ValueError(f"{cid}: threshold must be finite and width positive")
    if lag_threshold is None:
        lag_threshold = rules["C04"]["threshold"]
    scope = pd.DataFrame(True, index=f.index, columns=ids)
    nonconsign = f.flow.ne("Consignment")
    scope.loc[:, ["C01", "C02", "C04", "C08"]] = np.broadcast_to(nonconsign.to_numpy()[:, None], (len(f), 4))
    scope["C03"] = f.flow.isin(["DF1", "DF2"])
    scope["C05"] = f.flow.eq("DF1")
    V = pd.DataFrame(np.nan, index=f.index, columns=ids)
    reason = pd.DataFrame("evaluated", index=f.index, columns=ids)
    raw = pd.DataFrame(index=f.index)
    raw["C01"] = f.n_invoice.eq(0).astype(float)
    raw["C02"] = f.n_clear.eq(0).astype(float)
    raw["C03"] = f.n_receipt.eq(0).astype(float)
    raw["C04"] = f.invoice_clear_days
    raw["C05"] = (f.first_invoice - f.first_receipt).dt.total_seconds() / 86400
    raw["C06"] = f.n_cancel_invoice.astype(float)
    raw["C07"] = f.n_edits.astype(float)
    raw["C08"] = f.n_invoice.astype(float)
    raw["C09"] = f.human_touches_lower_bound.astype(float)
    raw["C10"] = f.human_resources_lower_bound.astype(float)
    V["C01"], V["C02"], V["C03"] = raw.C01, raw.C02, raw.C03
    V["C04"] = ((raw.C04 - lag_threshold) / rules["C04"]["width"]).clip(0, 1)
    V["C05"] = raw.C05.lt(0).astype(float)
    V["C06"] = raw.C06.gt(0).astype(float)
    for cid in ["C07", "C08", "C09", "C10"]:
        V[cid] = ((raw[cid] - rules[cid]["threshold"]) / rules[cid]["width"]).clip(0, 1)
    for criterion, endpoints in {
        "C04": ("invoice", "clear"),
        "C05": ("invoice", "receipt"),
    }.items():
        missing = f[f"n_{endpoints[0]}"].eq(0) | f[f"n_{endpoints[1]}"].eq(0)
        invalid = f[f"first_{endpoints[0]}"].isna() | f[f"first_{endpoints[1]}"].isna()
        if temporal_filter:
            invalid |= ~f[f"usable_{endpoints[0]}"] | ~f[f"usable_{endpoints[1]}"]
        if criterion == "C04":
            invalid |= f.invoice_clear_days.lt(0)
        reason.loc[invalid, criterion] = "invalid_temporal_evidence"
        reason.loc[missing, criterion] = "missing_prerequisite"
        V.loc[missing | invalid, criterion] = np.nan
    if resource_policy == "complete":
        reason.loc[~f.all_resources_usable, ["C09", "C10"]] = "missing_resource_evidence"
        V.loc[~f.all_resources_usable, ["C09", "C10"]] = np.nan
    elif resource_policy != "lower_bound":
        raise ValueError(resource_policy)
    V = V.where(scope)
    reason = reason.mask(~scope, "out_of_scope")
    assert V.notna().equals(reason.eq("evaluated"))
    return V, scope, reason, raw


def weights_for(cfg, view):
    return pd.Series({c["id"]: cfg["views"][view][c["layer"]] for c in cfg["criteria"]}, dtype=float)


def rank_keys(table, key="PI", k=10):
    t = table.reset_index()
    t["_key"] = t.iloc[:, 0].astype(str)
    t = t.sort_values([key, "n_cases", "_key"], ascending=[False, False, True], kind="mergesort")
    return t["_key"].head(min(k, len(t))).tolist()


def compare_tables(base, other, *, label, kind, key="PI"):
    common = base.index.intersection(other.index)
    a, b = base.loc[common, key], other.loc[common, key]
    corr = float(spearmanr(a, b).statistic) if len(common) > 1 and a.nunique() > 1 and b.nunique() > 1 else None
    topa, topb = set(rank_keys(base, key)), set(rank_keys(other, key))
    return {
        "comparison": label,
        "kind": kind,
        "common_groups": len(common),
        "base_scored_cases": int(base.n_cases.sum()),
        "other_scored_cases": int(other.n_cases.sum()),
        "spearman_on_common": corr,
        "top10_jaccard": len(topa & topb) / len(topa | topb) if topa | topb else None,
        "base_positive_groups": int(base[key].gt(0).sum()),
        "other_positive_groups": int(other[key].gt(0).sum()),
        "base_reference": float(base.reference.iloc[0]),
        "other_reference": float(other.reference.iloc[0]),
    }


def criterion_group_profiles(V, scope, reason, groups):
    rows = []
    for criterion in V:
        x = pd.DataFrame(
            {
                "scope": scope[criterion].astype(int),
                "evaluable": V[criterion].notna().astype(int),
                "positive": V[criterion].gt(0).astype(int),
                "severity": V[criterion],
                "group": groups,
            }
        )
        g = x.groupby("group", sort=True).agg(
            n_cases=("scope", "size"),
            in_scope=("scope", "sum"),
            evaluated=("evaluable", "sum"),
            positive=("positive", "sum"),
            mean_evaluated_severity=("severity", "mean"),
        )
        g["criterion"] = criterion
        g["scope_share"] = g.in_scope / g.n_cases
        g["evaluable_within_scope"] = g.evaluated / g.in_scope.replace(0, np.nan)
        rows.append(g.reset_index())
    profiles = pd.concat(rows, ignore_index=True)
    reasons = (
        reason.join(groups.rename("group"))
        .melt(id_vars="group", var_name="criterion", value_name="state")
        .groupby(["group", "criterion", "state"], sort=True)
        .size()
        .rename("n")
        .reset_index()
    )
    return profiles, reasons


def document_bootstrap(q: pd.Series, f: pd.DataFrame, cfg: dict):
    z = f.loc[q.dropna().index, [DOC, "group"]].copy()
    z["q"] = q.reindex(z.index)
    doc_codes, docs = pd.factorize(z[DOC], sort=True)
    grp_codes, grps = pd.factorize(z.group, sort=True)
    counts = sparse.csr_matrix((np.ones(len(z)), (doc_codes, grp_codes)), shape=(len(docs), len(grps)))
    burden = sparse.csr_matrix((z.q.to_numpy(), (doc_codes, grp_codes)), shape=counts.shape)
    rng = np.random.default_rng(cfg["seed"])
    values, selection = [], np.zeros(len(grps), dtype=int)
    for _ in range(cfg["bootstrap_replicates"]):
        doc_multiplicity = np.bincount(rng.integers(len(docs), size=len(docs)), minlength=len(docs))
        n = np.asarray(doc_multiplicity @ counts).ravel()
        total = np.asarray(doc_multiplicity @ burden).ravel()
        avg = total.sum() / n.sum()
        pi = np.maximum(total - n * avg, 0)
        values.append(pi)
        order = np.lexsort((np.arange(len(grps)), -n, -pi))
        order = order[n[order] > 0]
        selection[order[: min(cfg["bootstrap_top_k"], len(order))]] += 1
    a = np.asarray(values)
    return pd.DataFrame(
        {
            "group": grps,
            "selection_frequency_top10": selection / len(a),
            "PI_p025": np.quantile(a, 0.025, axis=0),
            "PI_median": np.median(a, axis=0),
            "PI_p975": np.quantile(a, 0.975, axis=0),
            "replicates": len(a),
            "resampling_unit": "purchasing document",
        }
    )


def event_sharing_diagnostics(d):
    keys = [DOC, ACT, TS]
    shared = d.groupby(keys, observed=True, sort=False)[CASE].transform("nunique").gt(1)
    dup = d.duplicated([CASE, ACT, TS], keep="first")
    return {
        "duplicate_key_rows_beyond_first": int(dup.sum()),
        "duplicate_key_policy": "same case/activity/raw timestamp; not certified duplicates",
        "events_with_same_document_activity_timestamp_in_multiple_items": int(shared.sum()),
        "share_same_document_key_in_multiple_items": float(shared.mean()),
        "sharing_interpretation": "Document-conditioned shared-key proxy; cannot establish extraction replication or duplicate cost",
        "timestamp_parse_failures": int(d.timestamp.isna().sum()),
        "timestamp_outside_plausibility_range": int((d.timestamp.notna() & d.plausible_timestamp.isna()).sum()),
        "events_missing_resource_evidence": int((~d.resource_usable).sum()),
    }


def execute(input_path=DEFAULT_INPUT, output_dir=None, *, bootstrap_replicates=None):
    started = time.time()
    out = Path(output_dir) if output_dir else HERE / "results"
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((HERE / "illustrative_norm.json").read_text())
    config_hash_at_start = sha256(HERE / "illustrative_norm.json")
    dump_json(
        out / "run_manifest.json",
        {
            "status": "running",
            "experiment": cfg["id"],
            "full_log": True,
            "pipeline_sha256": PIPELINE_HASH_AT_IMPORT,
            "reference_sha256": REFERENCE_HASH_AT_IMPORT,
            "config_sha256": config_hash_at_start,
        },
    )
    if bootstrap_replicates is not None:
        cfg["bootstrap_replicates"] = int(bootstrap_replicates)
        if cfg["bootstrap_replicates"] < 1:
            raise ValueError("At least one bootstrap replicate is required")
    print("Reading verified full BPIC19 CSV", flush=True)
    d, data_hash = read_events(Path(input_path), cfg)
    f = features_from_events(d, cfg)
    V, scope, reason, raw = assessment_matrices(f, cfg)
    layers = {c["id"]: c["layer"] for c in cfg["criteria"]}
    dump_json(out / "configuration.json", cfg)
    f.to_parquet(out / "case_features.parquet")
    V.add_prefix("severity__").join(scope.add_prefix("in_scope__")).join(reason.add_prefix("state__")).join(
        raw.add_prefix("raw__")
    ).to_parquet(out / "case_assessments.parquet")
    group_profiles, group_reasons = criterion_group_profiles(V, scope, reason, f.group)
    group_profiles.to_csv(out / "criterion_group_coverage.csv", index=False)
    group_reasons.to_csv(out / "criterion_group_states.csv", index=False)
    global_states = (
        reason.melt(var_name="criterion", value_name="state").groupby(["criterion", "state"]).size().rename("n").reset_index()
    )
    global_states.to_csv(out / "criterion_states.csv", index=False)
    print("Scoring explicit views and verifying identities", flush=True)
    tables, summaries, checks, comparative = {}, [], [], []
    case_scores = f[["group", DOC, "flow"]].copy()
    balanced = None
    for view in cfg["views"]:
        scored = score_assessments(V, scope, weights_for(cfg, view), layers, mode="flat")
        tab = group_priorities(scored["score"], f.group, gamma=0)
        tables[view] = tab
        tab.to_csv(out / f"groups_{view}.csv")
        case_scores[f"penalty__{view}"] = scored["penalty"]
        case_scores[f"coverage__{view}"] = scored["coverage"]
        case_scores[f"score__{view}"] = scored["score"]
        for grouping in ["vendor", DOC]:
            group_priorities(scored["score"], f[grouping], gamma=0).to_csv(
                out / f"groups_{view}_{'document' if grouping == DOC else grouping}.csv"
            )
        valid = scored["score"].notna()
        residual = float((scored["penalties"].sum(axis=1)[valid] - scored["penalty"][valid]).abs().max())
        weight_residual = float((scored["effective_weights"].sum(axis=1)[valid] - 1).abs().max())
        cc = coco_same_target(scored["penalty"], f.group)
        expected_pi = cc.D * cc.CoCo.clip(lower=0)
        coco_residual = float((tab.PI - expected_pi).abs().max())
        signed = signed_components(scored["penalties"], f.group)
        signed.to_csv(out / f"signed_criterion_excess_{view}.csv")
        signed_layer = signed.T.groupby(pd.Series(layers)).sum().T
        signed_layer.to_csv(out / f"signed_layer_excess_{view}.csv")
        signed_residual = float((signed.sum(axis=1).clip(lower=0) - tab.PI).abs().max())
        assert residual < 1e-10
        assert weight_residual < 1e-10
        assert coco_residual < 1e-7
        assert signed_residual < 1e-7
        assert scored["penalty"].dropna().between(-1e-12, 1 + 1e-12).all()
        cc.to_csv(out / f"coco_identity_{view}.csv")
        checks.append(
            {
                "view": view,
                "penalty_reconstruction_max_residual": residual,
                "effective_weight_sum_max_residual": weight_residual,
                "coco_identity_max_residual": coco_residual,
                "signed_PI_reconstruction_max_residual": signed_residual,
            }
        )
        summaries.append(
            {
                "view": view,
                "cases": len(f),
                "scored": int(valid.sum()),
                "unscored": int((~valid).sum()),
                "mean_penalty": float(scored["penalty"].mean()),
                "mean_coverage": float(scored["coverage"].mean()),
                "coverage_below_one_share": float(scored["coverage"].lt(1 - 1e-12).mean()),
                "positive_groups": int(tab.PI.gt(0).sum()),
                "sum_absolute_burden": float(tab.total_penalty.sum()),
                "sum_relative_priority": float(tab.PI.sum()),
            }
        )
        profile = pd.DataFrame(
            {
                "flow": f.flow,
                "coverage": scored["coverage"],
                "penalty": scored["penalty"],
                "signature": V.notna().astype(int).astype(str).agg("".join, axis=1),
            }
        )
        profile.groupby(["flow", "signature"], sort=True).agg(
            n=("coverage", "size"),
            mean_coverage=("coverage", "mean"),
            min_coverage=("coverage", "min"),
            mean_penalty=("penalty", "mean"),
        ).to_csv(out / f"evidence_signatures_{view}.csv")
        pd.DataFrame(
            {
                "group": f.group,
                "coverage": scored["coverage"],
                "low": scored["coverage"].lt(1 - 1e-12),
            }
        ).groupby("group").agg(
            n=("coverage", "size"),
            mean_coverage=("coverage", "mean"),
            minimum_coverage=("coverage", "min"),
            low_coverage_share=("low", "mean"),
        ).to_csv(out / f"group_coverage_{view}.csv")
        if view == "Balanced":
            balanced = scored
    case_scores.to_parquet(out / "case_scores.parquet")
    pd.DataFrame(summaries).to_csv(out / "view_summary.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "verification_checks.csv", index=False)
    base = tables["Balanced"]
    for view in ["Completion", "Operations"]:
        comparative.append(
            compare_tables(
                base,
                tables[view],
                label=f"view_{view}",
                kind="target and reference change; groups/ranker fixed",
            )
        )
    print("Running controlled evidence and policy sensitivities", flush=True)

    def sensitivity(name, vv, aa, ff=f, *, mode="flat"):
        scored = score_assessments(vv, aa, weights_for(cfg, "Balanced"), layers, mode=mode)
        table = group_priorities(scored["score"], ff.group, gamma=0)
        table.to_csv(out / f"sensitivity_{name}.csv")
        row = compare_tables(
            base,
            table,
            label=name,
            kind="assessment/evidence sensitivity; reference recomputed",
        )
        row["mean_coverage"] = float(scored["coverage"].mean())
        row["mean_penalty"] = float(scored["penalty"].mean())
        comparative.append(row)
        return scored, table

    sensitivity("layer_balanced", V, scope, mode="layer_balanced")
    for threshold in [15, 60]:
        vv, aa, _, _ = assessment_matrices(f, cfg, lag_threshold=threshold)
        sensitivity(f"lag_threshold_{threshold}", vv, aa)
    vv, aa, _, _ = assessment_matrices(f, cfg, resource_policy="lower_bound")
    sensitivity("resources_recorded_lower_bound", vv, aa)
    vv, aa, _, _ = assessment_matrices(f, cfg, temporal_filter=False)
    sensitivity("timestamps_without_plausibility_filter", vv, aa)
    for name, keep in {
        "complete_evidence_cohort": balanced["coverage"].ge(1 - 1e-12),
        "mature_heuristic_cohort": f.mature_heuristic,
        "recorded_clear_cohort": f.recorded_clear,
    }.items():
        sensitivity(name, V.loc[keep], scope.loc[keep], f.loc[keep])
    cohort_rows = []
    for name, mask in {
        "all_cases": pd.Series(True, index=f.index),
        "complete_evidence_cohort": balanced["coverage"].ge(1 - 1e-12),
        "mature_heuristic_cohort": f.mature_heuristic,
        "recorded_clear_cohort": f.recorded_clear,
    }.items():
        for flow in cfg["flow_map"].values():
            cohort_rows.append(
                {
                    "cohort": name,
                    "flow": flow,
                    "cases": int((mask & f.flow.eq(flow)).sum()),
                }
            )
    pd.DataFrame(cohort_rows).to_csv(out / "cohort_flow_counts.csv", index=False)
    dedup = d.drop_duplicates([CASE, ACT, TS], keep="first")
    fd = features_from_events(dedup, cfg)
    vd, ad, _, _ = assessment_matrices(fd, cfg)
    sensitivity("hypothetical_duplicate_key_collapse", vd, ad, fd)
    # Remove one criterion without recoding it as out of scope: a zero-weight policy.
    for cid in ["C07", "C08"]:
        ww = weights_for(cfg, "Balanced")
        ww[cid] = 0
        ss = score_assessments(V, scope, ww, layers, mode="flat")
        tt = group_priorities(ss["score"], f.group, gamma=0)
        tt.to_csv(out / f"sensitivity_leave_out_{cid}.csv")
        comparative.append(
            compare_tables(
                base,
                tt,
                label=f"leave_out_{cid}",
                kind="criterion-weight omission; target/reference change",
            )
        )
    pd.DataFrame(comparative).to_csv(out / "assessment_sensitivity.csv", index=False)
    # Ranking policies use precisely the same Balanced penalty, groups and population.
    policy = base.copy()
    policy["mean_concern"] = policy.mean_penalty
    policy["total_concern"] = policy.total_penalty
    regularised = group_priorities(balanced["score"], f.group, gamma=20)
    policy["gamma20_PI"] = regularised.stable_PI
    fixed_reference = group_priorities(balanced["score"], f.group, baseline=0.9)
    policy["external_adequacy_0p9_PI"] = fixed_reference.PI
    policy.to_csv(out / "fixed_target_policies.csv")
    policy_rows = []
    for key in [
        "mean_concern",
        "total_concern",
        "gamma20_PI",
        "external_adequacy_0p9_PI",
    ]:
        a = set(rank_keys(policy, "PI"))
        b = set(rank_keys(policy, key))
        policy_rows.append(
            {
                "policy": key,
                "spearman_vs_PI": float(spearmanr(policy.PI, policy[key]).statistic),
                "top10_jaccard": len(a & b) / len(a | b),
                "target_population": "same Balanced full-log scored cases",
                "reference_change": key.startswith("external"),
            }
        )
    pd.DataFrame(policy_rows).to_csv(out / "policy_comparison.csv", index=False)
    support_rows = []
    for grouping in ["group", "vendor", DOC]:
        for threshold in [1, 30, 100]:
            tab = group_priorities(balanced["score"], f[grouping], min_cases=threshold)
            support_rows.append(
                {
                    "grouping": grouping,
                    "minimum_support": threshold,
                    "eligible_groups": len(tab),
                    "represented_scored_cases": int(tab.n_cases.sum()),
                    "reference": float(tab.reference.iloc[0]) if len(tab) else None,
                }
            )
    pd.DataFrame(support_rows).to_csv(out / "support_sensitivity.csv", index=False)
    print(
        f"Document-cluster bootstrap: {cfg['bootstrap_replicates']} replicates",
        flush=True,
    )
    boot = document_bootstrap(balanced["penalty"], f, cfg)
    boot.to_csv(out / "document_bootstrap.csv", index=False)
    # A deterministic trace, selected by index then case ID, has no inferred owner.
    selected_group = rank_keys(base, k=1)[0]
    candidates = balanced["penalty"].loc[f.group.eq(selected_group)].rename("penalty").reset_index()
    candidates.columns = ["case_id", "penalty"]
    selected_case = candidates.sort_values(["penalty", "case_id"], ascending=[False, True], kind="mergesort").iloc[0].case_id
    trace = d.loc[
        d[CASE].astype(str).eq(selected_case),
        [*COLS, "timestamp", "resource_usable", "human_record"],
    ].copy()
    trace["event_order_id"] = pd.to_numeric(trace.eventID, errors="raise")
    trace.sort_values(["timestamp", "event_order_id"], kind="mergesort").to_csv(out / "selected_case_events.csv", index=False)
    example = pd.DataFrame(
        {
            "criterion": V.columns,
            "in_scope": scope.loc[selected_case].values,
            "state": reason.loc[selected_case].values,
            "raw_measure": raw.loc[selected_case].values,
            "severity": V.loc[selected_case].values,
            "effective_weight": balanced["effective_weights"].loc[selected_case].values,
            "penalty_component": balanced["penalties"].loc[selected_case].values,
        }
    )
    example.to_csv(out / "selected_case_assessment.csv", index=False)
    dump_json(
        out / "selected_example.json",
        {
            "group": selected_group,
            "case_id": selected_case,
            "selection_rule": "highest Balanced relative-priority group; highest case penalty; lexical ties",
            "group_PI": float(base.loc[selected_group, "PI"]),
            "case_penalty": float(balanced["penalty"].loc[selected_case]),
            "case_coverage": float(balanced["coverage"].loc[selected_case]),
            "source_rows": len(trace),
            "owner": None,
            "interpretation": "Recorded concerns under the author-defined norm; possible follow-up hypothesis, not established cause or remedy",
        },
    )
    # Per-rule positive, zero, unavailable and out-of-scope witnesses, deterministic.
    witnesses = []
    for cid in V:
        masks = {
            "positive": V[cid].gt(0),
            "zero": V[cid].eq(0),
            "unevaluable": scope[cid] & V[cid].isna(),
            "out_of_scope": ~scope[cid],
        }
        for state, mask in masks.items():
            ix = sorted(V.index[mask])
            witnesses.append(
                {
                    "criterion": cid,
                    "example_kind": state,
                    "matching_cases": len(ix),
                    "case_id": ix[0] if ix else None,
                    "reason": reason.loc[ix[0], cid] if ix else "no such case in this log",
                    "raw_measure": float(raw.loc[ix[0], cid]) if ix and pd.notna(raw.loc[ix[0], cid]) else None,
                    "severity": float(V.loc[ix[0], cid]) if ix and pd.notna(V.loc[ix[0], cid]) else None,
                }
            )
    pd.DataFrame(witnesses).to_csv(out / "criterion_examples.csv", index=False)
    diagnostics = event_sharing_diagnostics(d)
    dump_json(out / "data_diagnostics.json", diagnostics)
    assert sha256(Path(__file__)) == PIPELINE_HASH_AT_IMPORT, "Pipeline changed during execution; rerun"
    assert sha256(ROOT / "code/wise_reference.py") == REFERENCE_HASH_AT_IMPORT, "Reference changed during execution; rerun"
    assert sha256(HERE / "illustrative_norm.json") == config_hash_at_start, "Configuration changed during execution; rerun"
    manifest = {
        "status": "completed",
        "experiment": cfg["id"],
        "full_log": True,
        "input_path": str(Path(input_path).resolve()),
        "input_sha256": data_hash,
        "events": len(d),
        "cases": len(f),
        "documents": int(f[DOC].nunique()),
        "activities": int(d[ACT].nunique()),
        "group_count": len(base),
        "flow_counts": f.flow.value_counts().to_dict(),
        "elapsed_seconds": time.time() - started,
        "bootstrap_replicates": cfg["bootstrap_replicates"],
        "config_sha256": sha256(HERE / "illustrative_norm.json"),
        "effective_config_sha256": sha256(out / "configuration.json"),
        "pipeline_sha256": sha256(Path(__file__)),
        "reference_sha256": sha256(ROOT / "code/wise_reference.py"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "pyarrow"]},
        "checks": checks,
        "limitations": cfg["claim_boundary"],
    }
    dump_json(out / "run_manifest.json", manifest)
    write_summary(out, manifest, summaries, diagnostics)
    print(
        f"Completed full log in {manifest['elapsed_seconds']:.1f}s. Outputs: {out}",
        flush=True,
    )
    return manifest


def write_summary(out, manifest, summaries, diagnostics):
    s = [
        "# Fresh BPIC19 evaluation — computed results",
        "",
        "This is the fresh ten-criterion illustrative norm, not a reproduction of the historical 29-rule experiment.",
        "",
        f"Processed **{manifest['events']:,} events, {manifest['cases']:,} PO items, {manifest['documents']:,} purchasing documents**, and {manifest['activities']} activities. All cases are retained in the primary run. All criteria, thresholds and views are author-defined.",
        "",
        "| View | Mean penalty | Mean weighted evidence coverage | Cases below full coverage | Positive-priority groups |",
        "|---|---:|---:|---:|---:|",
    ]
    for x in summaries:
        s.append(
            f"| {x['view']} | {x['mean_penalty']:.6f} | {x['mean_coverage']:.4f} | {x['coverage_below_one_share']:.2%} | {x['positive_groups']} |"
        )
    cohorts = pd.read_csv(out / "cohort_flow_counts.csv")
    complete = cohorts.loc[cohorts.cohort.eq("complete_evidence_cohort") & cohorts.cases.gt(0)]
    profile_text = ", ".join(f"{int(r.cases):,} {r.flow}" for r in complete.itertuples())
    s += [
        "",
        f"**Complete-evidence selection:** only {int(complete.cases.sum()):,} cases have all in-scope evidence available: {profile_text}. In this run that cohort is entirely Consignment. Complete here means every in-scope criterion, not all ten criteria: six criteria are outside Consignment's scope. A complete-case analysis therefore changes the process-flow population and is not representative validation. Exact cohort denominators are in `cohort_flow_counts.csv`.",
        "",
        "C08 counts repeated recorded invoice receipts. Multiple invoices can be legitimate for rent/logistics and other item histories; its positive severity is an illustrative investigation concern, not established defective rework. First-invoice-to-first-clearing lag is a trace proxy, not individual invoice-instance matching.",
        "",
        "Contribution sums, effective weights, signed priority decomposition and same-target CoCo identity passed numerical assertions. Residuals are exported in `verification_checks.csv`; CoCo agreement is an algebraic relationship, not a performance advantage.",
        "",
        f"Data diagnostics found {diagnostics['duplicate_key_rows_beyond_first']:,} repeated case/activity/timestamp keys beyond their first row, and {diagnostics['events_with_same_document_activity_timestamp_in_multiple_items']:,} events with a same-document/activity/timestamp key shared across items. These are proxy diagnostics, not evidence that events should be deleted or costs added/subtracted.",
        "",
        "Assessment sensitivity, fixed-target policy contrasts, evidence coverage, document-cluster resampling and selected raw trace are available in the accompanying tables. Resampling uses purchasing documents, not independent PO-item draws. Percentile dispersion is descriptive and is not a calibrated confidence interval for effects.",
        "",
        "No recorded invoice/clearing is an observation about the available history; it does not establish failure to complete. Effort criteria count recorded resource proxies, not labour time or avoidable work. Anonymised groups are not validated decision owners. No stakeholder study or intervention outcome is included.",
        "",
        "The main result is that changing explicit assessment/evidence choices changes some priorities under a fixed ranker, while decomposition retains the path back to criteria and events. Whether these are the right choices and whether the outputs improve review decisions remain separate empirical questions.",
    ]
    (out / "RESULTS.md").write_text("\n".join(s) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    args = parser.parse_args()
    execute(args.input, args.output, bootstrap_replicates=args.bootstrap_replicates)
