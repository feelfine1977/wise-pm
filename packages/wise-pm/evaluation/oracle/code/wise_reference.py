"""Transparent assessment reference; illustrations, not a new ranking method.

This module is an independent numerical oracle for prepared assessments.
It does not infer business scope, valid evidence, causes, or monetary benefit.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _aligned(value, index, name):
    if not value.index.is_unique or not index.is_unique:
        raise ValueError(f"{name}: duplicate case identifiers")
    if not value.index.equals(index):
        raise ValueError(f"{name}: case index/order must exactly match")
    return value


def _finite_nonnegative(x, name):
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError(f"{name} must be finite and nonnegative")


def _stable_mean(series):
    observed = series.dropna()
    if observed.empty:
        return np.nan
    anchor = float(observed.iloc[0])
    return anchor + float((observed - anchor).mean())


def score_assessments(violations, in_scope, weights, layers, *, mode):
    """Score explicit assessments; undefined violations use NaN.

    Finite values outside scope are rejected rather than interpreted as zeros.
    NaN inside scope is unevaluable. Reason codes/evidence links belong to the
    caller. Coverage uses raw weights before renormalisation. Layer weights are
    the sums of the supplied raw criterion weights, matching native semantics.
    """
    if mode not in {"flat", "layer_balanced"}:
        raise ValueError("mode must explicitly be flat or layer_balanced")
    if not isinstance(violations, pd.DataFrame) or violations.empty:
        raise ValueError("violations must be a nonempty DataFrame")
    if not violations.columns.is_unique or not violations.index.is_unique:
        raise ValueError("duplicate case or criterion identifiers")
    if not isinstance(in_scope, pd.DataFrame):
        raise ValueError("in_scope must be a boolean DataFrame")
    _aligned(in_scope, violations.index, "in_scope")
    if not in_scope.columns.equals(violations.columns):
        raise ValueError("in_scope columns/order must exactly match")
    if in_scope.isna().any().any() or not all(pd.api.types.is_bool_dtype(t) for t in in_scope.dtypes):
        raise ValueError("in_scope must contain booleans without missing values")
    if not isinstance(weights, pd.Series) or not weights.index.equals(violations.columns):
        raise ValueError("weights must have the exact criterion index/order")
    v = violations.to_numpy(dtype=float)
    a = in_scope.to_numpy(dtype=bool)
    w = weights.to_numpy(dtype=float)
    _finite_nonnegative(w, "weights")
    if not np.any(w > 0):
        raise ValueError("at least one criterion weight must be positive")
    if np.isinf(v).any() or ((v < 0) | (v > 1)).any():
        raise ValueError("evaluated severities must be finite in [0,1]")
    m = ~np.isnan(v)
    if (m & ~a).any():
        raise ValueError("out-of-scope severity must be NaN, not zero")
    if not isinstance(layers, Mapping) or set(layers) != set(violations.columns):
        raise ValueError("layers must map each criterion exactly once")
    layer_names = list(dict.fromkeys(layers[c] for c in violations.columns))
    if any(not isinstance(layer, str) or not layer for layer in layer_names):
        raise ValueError("layer names must be nonempty strings")
    available = m @ w
    scoped = a @ w
    scored = available > 0
    eff = np.zeros_like(v)
    if mode == "flat":
        np.divide(m * w, available[:, None], out=eff, where=available[:, None] > 0)
    else:
        active_layer_weight = np.zeros(len(v))
        for layer in layer_names:
            ix = np.array([layers[c] == layer for c in violations.columns])
            alpha = w[ix].sum()
            bw = m[:, ix] * w[ix]
            bw_sum = bw.sum(axis=1)
            within = np.zeros_like(bw)
            np.divide(bw, bw_sum[:, None], out=within, where=bw_sum[:, None] > 0)
            eff[:, ix] = within * alpha
            active_layer_weight += (bw_sum > 0) * alpha
        np.divide(
            eff,
            active_layer_weight[:, None],
            out=eff,
            where=active_layer_weight[:, None] > 0,
        )
    penalties = eff * np.nan_to_num(v, nan=0.0)
    # Roundoff at the endpoints must not make our own score invalid downstream.
    # Raw evaluated inputs were already range-checked; this is not data repair.
    q = np.clip(penalties.sum(axis=1), 0.0, 1.0)
    q[~scored] = np.nan
    penalties[~scored] = np.nan
    eff[~scored] = np.nan
    coverage = np.full(len(v), np.nan)
    np.divide(available, scoped, out=coverage, where=scoped > 0)
    q_min = np.full(len(v), np.nan)
    q_max = np.full(len(v), np.nan)
    known = (np.nan_to_num(v, nan=0.0) * w).sum(axis=1)
    np.divide(known, scoped, out=q_min, where=scoped > 0)
    np.divide(known + scoped - available, scoped, out=q_max, where=scoped > 0)
    idx, cols = violations.index, violations.columns

    def series(x, name):
        return pd.Series(x, index=idx, name=name)

    result = {
        "score": series(1 - q, "score"),
        "penalty": series(q, "penalty"),
        "effective_weights": pd.DataFrame(eff, index=idx, columns=cols),
        "penalties": pd.DataFrame(penalties, index=idx, columns=cols),
        "coverage": series(coverage, "coverage"),
        "in_scope": in_scope.copy(),
        "evaluated": pd.DataFrame(m, index=idx, columns=cols),
        "in_scope_count": series(a.sum(axis=1), "in_scope_count"),
        "evaluated_count": series(m.sum(axis=1), "evaluated_count"),
        "positive_weight_scope_count": series((a & (w > 0)).sum(axis=1), "positive_weight_scope_count"),
        "positive_weight_evaluated_count": series((m & (w > 0)).sum(axis=1), "positive_weight_evaluated_count"),
        "flat_completion_bounds": pd.DataFrame({"penalty_min": q_min, "penalty_max": q_max}, index=idx),
        "weights": weights.copy(),
        "layers": dict(layers),
        "mode": mode,
    }
    return result


def _grouped_frame(data, group_keys):
    if isinstance(group_keys, pd.Series):
        group_keys = group_keys.to_frame()
    if not isinstance(group_keys, pd.DataFrame) or group_keys.shape[1] == 0:
        raise ValueError("group_keys must be a Series or nonempty-column DataFrame")
    _aligned(group_keys, data.index, "group_keys")
    if not group_keys.columns.is_unique:
        raise ValueError("group key names must be unique")
    names = list(group_keys.columns)
    internal = [f"__wise_group_{i}" for i in range(len(names))]
    if set(internal) & set(data.columns):
        raise ValueError("reserved group key column")
    keys = group_keys.copy()
    keys.columns = internal
    return pd.concat([keys, data], axis=1), internal, names


def _set_group_names(frame, names):
    frame.index.names = names
    return frame


def _score_series(score):
    if not isinstance(score, pd.Series) or not score.index.is_unique:
        raise ValueError("score must be a Series with unique case identifiers")
    x = score.to_numpy(dtype=float)
    if np.isinf(x).any() or ((x < 0) | (x > 1)).any():
        raise ValueError("score must be in [0,1] or NaN")
    return score.astype(float)


def _params(gamma, baseline, min_cases=1):
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative")
    if baseline is not None and (not np.isfinite(baseline) or not 0 <= baseline <= 1):
        raise ValueError("baseline must be finite in [0,1]")
    if isinstance(min_cases, bool) or not isinstance(min_cases, (int, np.integer)) or min_cases < 1:
        raise ValueError("min_cases must be a positive integer")


def _exposure(exposure, index):
    if exposure is None:
        return pd.Series(1.0, index=index, name="volume")
    if not isinstance(exposure, pd.Series):
        raise ValueError("exposure must be an aligned Series")
    _aligned(exposure, index, "exposure")
    x = exposure.to_numpy(dtype=float)
    _finite_nonnegative(x, "exposure")
    return exposure.rename("volume").astype(float)


def group_priorities(score, group_keys, *, gamma=0.0, baseline=None, exposure=None, min_cases=1):
    """Rank groups; reference uses all scored cases before support filtering.

    Unscored cases do not enter means, volume, or the reference. Groups with no
    scored cases are absent from the ranked output; coverage reports keep them.
    Missing grouping values remain an explicit group. Ties: larger n, key order.
    """
    _params(gamma, baseline, min_cases)
    score = _score_series(score)
    data = pd.DataFrame({"score": score, "volume": _exposure(exposure, score.index)})
    frame, keys, names = _grouped_frame(data, group_keys)
    raw_n = frame.groupby(keys, dropna=False, observed=True, sort=True).size()
    d = frame.loc[score.notna()]
    if d.empty:
        raise ValueError("no scored cases to aggregate")
    reference = _stable_mean(score) if baseline is None else float(baseline)
    g = d.groupby(keys, dropna=False, observed=True, sort=True)
    out = g["score"].agg(n_cases="size", mean_score="mean")
    out["n_total"] = raw_n.reindex(out.index)
    out["n_unscored"] = out["n_total"] - out["n_cases"]
    out["volume"] = g["volume"].sum()
    out["mean_penalty"] = 1 - out["mean_score"]
    out["total_penalty"] = out["n_cases"] * out["mean_penalty"]
    out["gap"] = (reference - out["mean_score"]).clip(lower=0)
    if baseline is None and score.dropna().nunique() == 1:
        out["gap"] = 0.0
    out["PI"] = out["volume"] * out["gap"]
    shrink = out["n_cases"] / (out["n_cases"] + gamma)
    out["stable_mean"] = shrink * out["mean_score"] + (1 - shrink) * reference
    out["stable_gap"] = shrink * out["gap"]
    out["stable_PI"] = shrink * out["PI"]
    out["reference"] = reference
    out = out.loc[out["n_cases"] >= min_cases]
    out = out.sort_index().sort_values(["stable_PI", "n_cases"], ascending=False, kind="stable")
    _set_group_names(out, names)
    out.attrs.update(
        gamma=float(gamma),
        baseline=reference,
        reference_kind="observed_mean" if baseline is None else "external",
        min_cases=int(min_cases),
        volume="cases" if exposure is None else "exposure",
        scored_population=int(score.notna().sum()),
        total_population=len(score),
        filter_after_reference=True,
        tie_policy="stable_PI desc,n_cases desc,key asc",
    )
    return out


def signed_components(
    penalties,
    group_keys,
    *,
    reference_profile=None,
    baseline=None,
    exposure=None,
    gamma=0.0,
):
    """Signed amounts that sum to pre-clipping priority excess.

    Clip the ROW SUM, never individual components. An external baseline needs a
    matching reference profile; arbitrary scalar references do not identify it.
    This is algebraic accounting, not causal attribution.
    """
    _params(gamma, baseline)
    if not isinstance(penalties, pd.DataFrame) or not penalties.columns.is_unique:
        raise ValueError("penalties must have unique component columns")
    x = penalties.to_numpy(dtype=float)
    partial = np.isnan(x).any(axis=1) & ~np.isnan(x).all(axis=1)
    if partial.any() or np.isinf(x).any() or (x < 0).any():
        raise ValueError("penalty rows must be finite nonnegative or entirely NaN")
    q = penalties.sum(axis=1, min_count=1)
    if (q > 1 + 1e-12).any():
        raise ValueError("component penalties cannot sum above 1")
    if not q.notna().any():
        raise ValueError("no scored cases to explain")
    if baseline is not None and reference_profile is None:
        raise ValueError("external baseline requires an explicit component reference profile")
    if reference_profile is None:
        ref = penalties.mean(axis=0)
        reference = 1 - float(q.mean())
    else:
        if not isinstance(reference_profile, pd.Series) or not reference_profile.index.equals(penalties.columns):
            raise ValueError("reference_profile must have exact component index/order")
        ref = reference_profile.astype(float)
        _finite_nonnegative(ref.to_numpy(), "reference_profile")
        if ref.sum() > 1 + 1e-12:
            raise ValueError("reference profile sum cannot exceed 1")
        reference = 1 - float(ref.sum()) if baseline is None else float(baseline)
        if not np.isclose(ref.sum(), 1 - reference, atol=1e-12, rtol=0):
            raise ValueError("reference profile must sum to 1-baseline")
    volumes = _exposure(exposure, penalties.index)
    frame, keys, names = _grouped_frame(penalties.assign(__volume=volumes), group_keys)
    frame = frame.loc[q.notna()]
    g = frame.groupby(keys, dropna=False, observed=True, sort=True)
    n = g.size()
    means = g[list(penalties.columns)].mean()
    vol = g["__volume"].sum()
    multiplier = vol * n / (n + gamma)
    out = (means - ref).mul(multiplier, axis=0)
    _set_group_names(out, names)
    out.attrs.update(
        reference=reference,
        reference_profile=ref.to_dict(),
        gamma=float(gamma),
        volume="cases" if exposure is None else "exposure",
        clip="after_row_sum",
    )
    return out


def coco_same_target(penalty, group_keys):
    """CoCo as a fraction. D=0 -> NaN with explicit undefined state."""
    q = _score_series(penalty)
    d, keys, names = _grouped_frame(q.rename("penalty").to_frame(), group_keys)
    d = d.loc[q.notna()]
    if d.empty:
        raise ValueError("no scored cases for CoCo")
    mean = _stable_mean(q)
    denominator = float((q - mean).clip(lower=0).sum())
    g = d.groupby(keys, dropna=False, observed=True, sort=True)["penalty"]
    out = g.agg(n_cases="size", mean_penalty="mean")
    out["numerator"] = out["n_cases"] * (out["mean_penalty"] - mean)
    if q.dropna().nunique() == 1:
        out["numerator"] = 0.0
    out["D"] = denominator
    out["CoCo"] = out["numerator"] / denominator if denominator > 0 else np.nan
    out["normalisation_defined"] = denominator > 0
    _set_group_names(out, names)
    out.attrs.update(reference_penalty=mean, normaliser=denominator, units="fraction_not_percent")
    return out


def coverage_diagnostics(result, group_keys):
    """Return group coverage, criterion counts, and evidence-signature counts."""
    scalar = pd.DataFrame({k: result[k] for k in ["score", "coverage", "in_scope_count", "evaluated_count"]})
    scalar["scored"] = scalar["score"].notna().astype(int)
    frame, keys, names = _grouped_frame(scalar, group_keys)
    g = frame.groupby(keys, dropna=False, observed=True, sort=True)
    groups = g.agg(
        n_total=("scored", "size"),
        n_scored=("scored", "sum"),
        coverage_mean=("coverage", "mean"),
        coverage_min=("coverage", "min"),
        in_scope_count_mean=("in_scope_count", "mean"),
        evaluated_count_mean=("evaluated_count", "mean"),
    )
    groups["n_unscored"] = groups["n_total"] - groups["n_scored"]
    groups["coverage_q10"] = g["coverage"].quantile(0.1)
    groups["unscored_share"] = groups["n_unscored"] / groups["n_total"]
    _set_group_names(groups, names)
    counts = {}
    for label, matrix in [
        ("in_scope", result["in_scope"]),
        ("evaluated", result["evaluated"]),
    ]:
        f, ks, ns = _grouped_frame(matrix.astype(int), group_keys)
        counts[label] = _set_group_names(
            f.groupby(ks, dropna=False, observed=True, sort=True)[list(matrix.columns)].sum(),
            ns,
        )
    masks = result["evaluated"].to_numpy(dtype=bool)
    unique, inverse = np.unique(masks, axis=0, return_inverse=True)
    signatures = ["|".join(str(c) for c, yes in zip(result["evaluated"].columns, row) if yes) or "<none>" for row in unique]
    sig = pd.Series([signatures[i] for i in inverse], index=result["score"].index, name="signature")
    sf, sk, sn = _grouped_frame(sig.to_frame(), group_keys)
    signature_counts = sf.groupby([*sk, "signature"], dropna=False, observed=True, sort=True).size().rename("n_cases").to_frame()
    signature_counts.index.names = [*sn, "signature"]
    return {
        "groups": groups,
        "criterion_in_scope": counts["in_scope"],
        "criterion_evaluated": counts["evaluated"],
        "signatures": signature_counts,
    }


def balance_from_observed_totals(x, y, valid_x, valid_y, *, threshold=0.0, width=1.0, eps=1e-9):
    """Illustrative evidence-safe balance helper, not a native WISE patch.

    Caller decides what establishes an observed total. Explicit observed zeros
    are valid; missing/nonnumeric values remain NaN rather than becoming zeros.
    """
    for obj, name in [(y, "y"), (valid_x, "valid_x"), (valid_y, "valid_y")]:
        _aligned(obj, x.index, name)
    if not np.isfinite([threshold, width, eps]).all() or threshold < 0 or width <= 0 or eps <= 0:
        raise ValueError("invalid balance parameters")
    xx, yy = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
    valid = valid_x & valid_y & np.isfinite(xx) & np.isfinite(yy)
    if ((xx[valid] < 0) | (yy[valid] < 0)).any():
        raise ValueError("observed totals must be nonnegative; specify credit treatment")
    denom = np.maximum(np.maximum(xx, yy), eps)
    value = (((xx - yy).abs() / denom - threshold) / width).clip(0, 1)
    return value.where(valid).rename("balance_severity")


def toy_assessments():
    """Constructed four-case illustration; never described as BPIC observations."""
    ix = pd.Index(["toy_A1", "toy_A2", "toy_B1", "toy_B2"], name="case_id")
    v = pd.DataFrame(
        {
            "delay": [0.8, 0.6, 0.1, 0.1],
            "approval": [0, 0, 1, 0.8],
            "control": [0.2, np.nan, 0.2, np.nan],
        },
        index=ix,
    )
    scope = pd.DataFrame(True, index=ix, columns=v.columns)
    weights = pd.Series({"delay": 0.5, "approval": 0.25, "control": 0.25})
    layers = {"delay": "Timeliness", "approval": "Controls", "control": "Controls"}
    groups = pd.Series(["A", "A", "B", "B"], index=ix, name="group")
    return v, scope, weights, layers, groups
