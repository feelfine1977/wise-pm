#!/usr/bin/env python3
"""Replay prepared assessments through installed WISE; verify against frozen results.

The optional differential oracle is an explicitly supplied test dependency. The
native execution always uses wise.Metric, wise.Norm, wise.score and prioritize.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from native_assessments import prepare_native, weighted_coverage, write_prepared

import wise

sys.dont_write_bytecode = True

DATASETS = [
    "bpic2019",
    "bpic2013_incidents",
    "bpic2013_open_problems",
    "bpic2013_closed_problems",
    "bpic2020",
    "ocel",
    "icpm",
    "bpic2012",
    "bpic2017",
    "incident_template",
    "uci_incidents",
    "amr_orders",
    "p2p",
    "sepsis",
    "hospital_workbook",
]


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, x):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False))


def require_same_labels(actual, expected, label):
    """Reject missing, extra, duplicate or type-changed labels before alignment."""
    if not actual.is_unique or not expected.is_unique:
        raise AssertionError(f"{label}: duplicate identities")
    actual_keys = {(type(value).__name__, repr(value)) for value in actual.tolist()}
    expected_keys = {(type(value).__name__, repr(value)) for value in expected.tolist()}
    if len(actual) != len(expected) or actual_keys != expected_keys:
        raise AssertionError(f"{label}: exact identity mismatch")


def string_index(index):
    """Normalize file identifiers only when the conversion is injective."""
    converted = pd.Index([str(value) for value in index], dtype=object, name=index.name)
    if not index.is_unique or not converted.is_unique:
        raise ValueError("Identifier string conversion would merge distinct or duplicate identities")
    return converted


def table(p):
    if p.suffix == ".parquet":
        d = pd.read_parquet(p)
    else:
        col = pd.read_csv(p, nrows=0).columns[0]
        d = pd.read_csv(p, dtype={col: str}, index_col=0)
    d.index = string_index(d.index)
    d.index.name = "case_id"
    if not d.index.is_unique:
        raise ValueError(f"Duplicate case/group identifiers in {p.name}")
    return d


def read_bundle(root, adapter_root, name):
    used = []

    def read(p):
        used.append(p)
        return table(p)

    def config(*paths):
        for p in paths:
            if p.exists():
                used.append(p)
                return json.loads(p.read_text())
        raise FileNotFoundError("Configuration missing: " + ", ".join(str(p) for p in paths))

    def family(fam):
        for p in [root / fam / "results", root / fam, root / "data/prepared" / fam / "results", root / "data/prepared" / fam]:
            if p.is_dir() and any(p.glob("*.json")):
                return p
        raise FileNotFoundError(f"Cannot find prepared results for {fam}")

    expected_weights = {}
    expected_components = {}
    if name in ["bpic2019", "bpic2020"] or name.startswith("bpic2013_"):
        fam = "bpic2013" if name.startswith("bpic2013_") else name
        b = family(fam)
        cfg = config(b / "configuration.json")
        d = b / name.removeprefix("bpic2013_") if fam == "bpic2013" else b
        a = read(d / ("assessments.parquet" if fam == "bpic2013" else "case_assessments.parquet"))
        v = a.filter(regex="^severity__").rename(columns=lambda c: c.split("__", 1)[1])
        scope = a.filter(regex="^in_scope__").rename(columns=lambda c: c.split("__", 1)[1])
        f = read(d / "case_features.parquet")
        scores = read(d / "case_scores.parquet")
        key = {"bpic2013": "product_group", "bpic2019": "group", "bpic2020": "amount_band"}[fam]
        groups = f[key]
        criteria = cfg["criteria"]
        layers = {c["id"]: c["layer"] for c in criteria}
        views = {n: {c: float(w[layers[c]]) for c in v} for n, w in cfg["views"].items()}
        expected = {
            n: read(d / f"groups_{n}_amount_band.csv") if fam == "bpic2020" else read(d / f"groups_{n}.csv") for n in views
        }
        for n in views:
            wc = [f"weight__{n}__{c}" for c in v]
            pc = [f"component__{n}__{c}" for c in v]
            if all(c in a for c in wc):
                expected_weights[n] = a[wc].set_axis(v.columns, axis=1)
            if all(c in a for c in pc):
                expected_components[n] = a[pc].set_axis(v.columns, axis=1)
    elif name in ["bpic2012", "bpic2017"]:
        b = family("bpic2012_2017")
        d = b / name
        cfg = config(
            b / "configuration.json",
            b / "executed_norm.json",
            adapter_root / "bpic2012_2017/illustrative_norm.json",
            root / "bpic2012_2017/illustrative_norm.json",
        )
        cfg = cfg["families"][name]
        v = read(d / "assessments.parquet")
        scope = read(d / "in_scope.parquet")
        f = read(d / "case_features.parquet")
        scores = read(d / "case_scores.parquet")
        layers = {c: r["layer"] for c, r in cfg["criteria"].items()}
        views = cfg["views"]
        groups = (
            f["amount_band"] if name == "bpic2012" else f["application_type"].astype(str) + " | " + f["loan_goal"].astype(str)
        )
        expected = {n: read(d / f"groups_{n}.csv") for n in views}
        for n in views:
            expected_components[n] = read(d / f"case_components_{n}.parquet")
    elif name in ["ocel", "icpm"]:
        b = family("ocel_icpm")
        d = b / name
        cfg = config(b / "executed_norm.json")[name]
        p = d / "criterion_assessments.csv"
        used.append(p)
        a = pd.read_csv(p, dtype={"case_id": str})
        v = a.pivot(index="case_id", columns="criterion", values="severity")
        scope = a.pivot(index="case_id", columns="criterion", values="in_scope")
        f = read(d / "case_features.csv")
        scores = read(d / "case_scores.csv")
        groups = f["group"]
        layers = {c["id"]: c["layer"] for c in cfg["criteria"]}
        views = {n: dict(zip([c["id"] for c in cfg["criteria"]], w)) for n, w in cfg["views"].items()}
        expected = {n: read(d / f"groups_{n}.csv") for n in views}
    elif name in ["sepsis", "hospital_workbook"]:
        b = family("healthcare_transfers")
        d = b / name
        cfg = config(b / "executed_norm.json")[name]
        v = read(d / "criterion_severity.csv")
        scope = read(d / "criterion_scope.csv")
        f = read(d / "case_features.csv")
        groups = f["group"]
        layers = {c: r["layer"] for c, r in cfg["criteria"].items()}
        views = {n: {c: w[layers[c]] for c in v} for n, w in cfg["views"].items()}
        scores = pd.DataFrame(index=v.index)
        for n in views:
            one = read(d / f"case_scores_{n}.csv")
            require_same_labels(one.index, v.index, "case score rows")
            scores[f"score__{n}"] = one["score"]
            scores[f"coverage__{n}"] = one["coverage"]
        expected = {n: read(d / f"groups_{n}.csv") for n in views}
    else:
        b = family("kaggle_transfers")
        d = b / name
        cfg = config(b / "executed_norm.json")["datasets"][name]
        v = read(d / "assessments.csv")
        scope = pd.DataFrame(True, index=v.index, columns=v.columns)
        f = read(d / "case_features.csv")
        scores = read(d / "case_scores.csv")
        groups = f["group"]
        layers = {c["id"]: c["layer"] for c in cfg["criteria"]}
        views = {n: dict(zip([c["id"] for c in cfg["criteria"]], w)) for n, w in cfg["views"].items()}
        expected = {n: read(d / f"groups_{n}.csv") for n in views}
        # In-scope-all is an explicit adapter declaration, checked against its exported counts.
        st = pd.read_csv(d / "criterion_states.csv")
        used.append(d / "criterion_states.csv")
        if not (st["in_scope"] == len(v)).all():
            raise ValueError("Stored catalogue no longer has all-case scope")
    v.index = string_index(v.index)
    scope.index = string_index(scope.index)
    require_same_labels(v.index, scope.index, "scope rows")
    require_same_labels(v.columns, scope.columns, "scope columns")
    require_same_labels(v.index, groups.index, "group case rows")
    require_same_labels(v.index, scores.index, "score rows")
    scope = scope.reindex(index=v.index, columns=v.columns)
    groups = groups.reindex(v.index).rename("assessment_group")
    scores = scores.reindex(v.index)
    if not all(pd.api.types.is_bool_dtype(t) for t in scope.dtypes):
        raise ValueError("Scope output must contain booleans")
    expected_scores = {}
    expected_coverage = {}
    for view in views:
        possibilities = [f"score__{view}", f"{view}__score"]
        col = next((c for c in possibilities if c in scores), None)
        expected_scores[view] = scores[col] if col else 1 - scores[f"penalty__{view}"]
        col = next(c for c in [f"coverage__{view}", f"{view}__coverage"] if c in scores)
        expected_coverage[view] = scores[col]
    return (
        v,
        scope,
        layers,
        views,
        groups,
        expected_scores,
        expected_coverage,
        expected,
        expected_weights,
        expected_components,
        used,
    )


def residual(actual, expected):
    if isinstance(actual, pd.DataFrame) != isinstance(expected, pd.DataFrame):
        raise AssertionError("Comparison container mismatch")
    require_same_labels(actual.index, expected.index, "comparison rows")
    if isinstance(actual, pd.DataFrame):
        require_same_labels(actual.columns, expected.columns, "comparison columns")
        expected = expected.reindex(index=actual.index, columns=actual.columns)
    else:
        expected = expected.reindex(actual.index)
    x = actual.to_numpy(dtype=float)
    y = expected.to_numpy(dtype=float)
    if x.shape != y.shape or not np.array_equal(np.isnan(x), np.isnan(y)):
        raise AssertionError("Missingness or shape mismatch")
    if not np.isfinite(x[~np.isnan(x)]).all() or not np.isfinite(y[~np.isnan(y)]).all():
        raise AssertionError("Nonfinite observed value")
    delta = np.abs(x - y)
    return float(np.nanmax(delta)) if np.isfinite(delta).any() else 0.0


def group_residual(native, expected):
    require_same_labels(native.index, expected.index, "group population")
    cols = ["n_cases", "volume", "mean_score", "gap", "PI", "stable_mean", "stable_gap", "stable_PI"]
    return {c: residual(native[c], expected[c]) for c in cols}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evaluation-root", type=Path, required=True)
    p.add_argument("--adapter-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--norm-output", type=Path)
    p.add_argument("--oracle", type=Path)
    p.add_argument("--dataset", action="append", choices=DATASETS)
    a = p.parse_args()
    root = a.evaluation_root
    adapters = a.adapter_root or root / "adapters"
    a.output.mkdir(parents=True, exist_ok=True)
    oracle = None
    if a.oracle:
        spec = importlib.util.spec_from_file_location("evaluation_test_oracle", a.oracle)
        oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oracle)
    reports = []
    source_hashes = {}
    for name in a.dataset or DATASETS:
        (
            v,
            scope,
            layers,
            views,
            groups,
            expected_scores,
            expected_cov,
            expected_groups,
            expected_weights,
            expected_components,
            files,
        ) = read_bundle(root, adapters, name)
        log, norm, cases = prepare_native(v, scope, layers, views, groups, name=f"{name}: prepared assessment norm")
        d = a.output / name
        norm = write_prepared(d, cases, norm)
        if a.norm_output:
            a.norm_output.mkdir(parents=True, exist_ok=True)
            norm.dump(a.norm_output / f"{name}.json")
        for f in files:
            try:
                key = str(f.relative_to(root))
            except ValueError:
                key = "adapter/" + str(f.relative_to(adapters))
            source_hashes[key] = digest(f)
        primary = wise.score(log, norm, derive=False)
        checks = {}
        checks["severity"] = residual(primary.violations, v)
        require_same_labels(primary.in_scope.index, scope.index, "native scope rows")
        require_same_labels(primary.in_scope.columns, scope.columns, "native scope columns")
        if not primary.in_scope.equals(scope.reindex(primary.in_scope.index)):
            raise AssertionError("Native scope mismatch")
        checks["native_layer_decomposition"] = primary.check_decomposition(atol=1e-10)
        differential = []
        primary_signed = {}
        wide = pd.DataFrame(index=primary.scores.index)
        for view in views:
            ng = wise.prioritize(primary, by="assessment_group", view=view)
            ng.to_csv(d / f"groups_{view}.csv")
            drivers = wise.layer_drivers(primary, by="assessment_group", view=view)
            drivers.to_csv(d / f"layer_drivers_{view}.csv")
            signed = pd.DataFrame({layer: drivers[f"{layer}__delta"] * drivers["n_cases"] for layer in norm.layer_ids})
            signed.to_csv(d / f"signed_layer_excess_{view}.csv")
            primary_signed[view] = signed
            vc = {
                "native_layer_priority": residual(signed.sum(axis=1).clip(lower=0), ng["PI"]),
                "scores": residual(primary.scores[view], expected_scores[view]),
                "coverage": residual(weighted_coverage(primary, view), expected_cov[view]),
                **{f"group_{c}": r for c, r in group_residual(ng, expected_groups[view]).items()},
            }
            if view in expected_weights:
                vc["stored_effective_weights"] = residual(
                    primary.effective_weights(view).where(primary.scores[view].notna()), expected_weights[view]
                )
            if view in expected_components:
                vc["stored_components"] = residual(
                    primary.penalties(view).where(primary.scores[view].notna()), expected_components[view]
                )
            checks[view] = vc
            wide[f"score__{view}"] = primary.scores[view]
            wide[f"coverage__{view}"] = weighted_coverage(primary, view)
        wide.to_parquet(d / "native_case_scores.parquet")
        if oracle:
            for mode in ["flat", "layer_balanced"]:
                native = primary if mode == "flat" else wise.score(log, norm, mode=mode, derive=False)
                for view, w in views.items():
                    reference = oracle.score_assessments(
                        v, scope, pd.Series(w).reindex(v.columns).astype(float), layers, mode=mode
                    )
                    rr = {
                        "scores": residual(native.scores[view], reference["score"]),
                        "weights": residual(
                            native.effective_weights(view).where(native.scores[view].notna()), reference["effective_weights"]
                        ),
                        "components": residual(native.penalties(view).where(native.scores[view].notna()), reference["penalties"]),
                        "coverage": residual(weighted_coverage(native, view), reference["coverage"]),
                    }
                    if mode == "flat":
                        layer_penalties = pd.DataFrame(
                            {
                                layer: reference["penalties"][[c for c in v if layers[c] == layer]].sum(axis=1, min_count=1)
                                for layer in norm.layer_ids
                            }
                        )
                        reference_signed = oracle.signed_components(layer_penalties, groups)
                        rr["native_layer_signed_excess"] = residual(primary_signed[view], reference_signed)
                    for gamma in [0.0, 20.0]:
                        ng = wise.prioritize(native, by="assessment_group", view=view, gamma=gamma)
                        eg = oracle.group_priorities(reference["score"], groups, gamma=gamma)
                        rr.update({f"gamma{gamma:g}_{c}": z for c, z in group_residual(ng, eg).items()})
                    differential.append({"mode": mode, "view": view, "residuals": rr})

        def values(x):
            if isinstance(x, dict):
                for y in x.values():
                    yield from values(y)
            elif isinstance(x, (int, float)):
                yield x

        maximum = max(list(values(checks)) + [z for r in differential for z in r["residuals"].values()])
        passed = maximum < 1e-8
        report = {
            "dataset": name,
            "cases": len(v),
            "criteria": len(v.columns),
            "views": len(views),
            "status": "passed" if passed else "failed",
            "max_residual": maximum,
            "norm_fingerprint": norm.fingerprint(),
            "stored_output_checks": checks,
            "differential_checks": differential,
            "meaning": "Native score/prioritization parity on prepared severities. Source parsing and raw operator semantics are separate evidence.",
        }
        save(d / "verification.json", report)
        reports.append(report)
        print(json.dumps({k: report[k] for k in ["dataset", "cases", "criteria", "views", "status", "max_residual"]}), flush=True)
        if not passed:
            raise AssertionError(f"{name}: residual {maximum}")
        del log, primary, cases, v, scope
        gc.collect()
    module_root = Path(wise.__file__).parent
    lib_hashes = {x.name: digest(x) for x in sorted(module_root.glob("*.py"))}
    output_hashes = {
        str(f.relative_to(a.output)): digest(f) for f in sorted(a.output.rglob("*")) if f.is_file() and f.name != "manifest.json"
    }
    summary = {
        "status": "passed",
        "wise_version": wise.__version__,
        "datasets": len(reports),
        "case_applications": sum(r["cases"] for r in reports),
        "views": sum(r["views"] for r in reports),
        "max_residual": max(r["max_residual"] for r in reports),
        "oracle_sha256": digest(a.oracle) if a.oracle else None,
        "native_library_sha256": lib_hashes,
        "prepared_source_sha256": source_hashes,
        "code_sha256": {
            Path(__file__).name: digest(__file__),
            "native_assessments.py": digest(Path(__file__).with_name("native_assessments.py")),
        },
        "output_sha256": output_hashes,
        "scope": "Prepared assessment parity, not raw-native operator equivalence. Case-application count includes overlapping populations and the three-case workbook fixture.",
    }
    save(a.output / "manifest.json", summary)


if __name__ == "__main__":
    main()
