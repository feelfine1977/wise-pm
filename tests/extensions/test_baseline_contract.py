"""Executable regression contract for the pre-extension library (stage S0, item E05).

Every fact below was measured on the unchanged base commit
``df5db50b839cc124b489a269894f5a2bfe7dc634`` and is the reference the
extension stages must not move.  The file is deliberately literal: it repeats
values that other tests derive, because its purpose is to fail loudly when a
later stage changes a number, a column order, a mask or a signature.

Conventions
-----------
* **Strict equality** for identities, orderings, masks, dtypes, column and
  index names, fingerprints and reason-carrying structure.  Those cannot drift
  by floating-point noise, so any change is a real change.
* **``rtol=0, atol=1e-12``** (``TOL``) for floating-point values whose operation
  order is unchanged by an extension.  The base results are exact to ~1e-16;
  1e-12 leaves room for a different but equivalent summation order without
  admitting a semantic change.  Existing benchmark tolerances elsewhere
  (``tests/test_bpic19.py``: ``abs=1e-3``/``5e-4``/``1.0``, ``tests/test_scoring.py``:
  ``abs=5e-5``) are deliberately **not** replaced by this tolerance.
* **Signatures** are pinned as an ordered prefix: the recorded
  positional-or-keyword parameters must stay exactly as they are, and every
  recorded keyword-only parameter must keep its default.  Appending a *new*
  keyword-only parameter is allowed — that is the extension rule of the master
  prompt ("new APIs are keyword-only additions") — while renaming, reordering,
  removing or re-defaulting an existing one fails here.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import wise
from wise import datasets, scoring
from wise import norm as norm_module
from wise.errors import NotScoredError

BASE_COMMIT = "df5db50b839cc124b489a269894f5a2bfe7dc634"
TOL = {"rtol": 0, "atol": 1e-12}
REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"
EMPTY = inspect.Parameter.empty


# --------------------------------------------------------------------- helpers
def _params(fn):
    return list(inspect.signature(fn).parameters.values())


def assert_signature(fn, positional, keyword_only, *, returns=None):
    """Pin the stable part of a public signature (see the module docstring)."""
    params = _params(fn)
    seen_pos = [(p.name, p.default) for p in params if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert seen_pos == positional, f"{fn.__qualname__}: positional parameters changed"
    seen_kw = {p.name: p.default for p in params if p.kind is inspect.Parameter.KEYWORD_ONLY}
    for name, default in keyword_only:
        assert name in seen_kw, f"{fn.__qualname__}: keyword-only parameter {name!r} disappeared"
        assert seen_kw[name] == default or (seen_kw[name] is default), (
            f"{fn.__qualname__}: default of {name!r} changed to {seen_kw[name]!r}"
        )
    if returns is not None:
        assert str(inspect.signature(fn).return_annotation) == returns


def close(actual, expected):
    np.testing.assert_allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), **TOL)


# ------------------------------------------------------------- public surface
BASE_EXPORTS = (
    "CONSTRAINT_TYPES",
    "SCHEMA_VERSION",
    "Balance",
    "Constraint",
    "EventLog",
    "Exclusion",
    "Lag",
    "Layer",
    "LogSchemaError",
    "Metric",
    "Norm",
    "NormConstraint",
    "NormError",
    "NotScoredError",
    "Precedence",
    "Presence",
    "ScoreResult",
    "Singularity",
    "View",
    "WiseError",
    "__version__",
    "as_labels",
    "compare_periods",
    "concentration",
    "constraint_drivers",
    "constraint_from_dict",
    "cross_case_replication",
    "datasets",
    "estimate_gamma",
    "evaluate_constraint",
    "event_replication",
    "gap_retained",
    "hotspot_table",
    "layer_drivers",
    "left_truncated",
    "observation_window",
    "pareto",
    "penalty_mass",
    "prioritize",
    "right_censored",
    "running_p2p_events",
    "running_p2p_log",
    "running_p2p_norm",
    "sat",
    "score",
    "timestamp_outliers",
    "top_k_overlap",
    "validation_table",
    "view_agreement",
    "violation_matrix",
)


def test_public_exports_are_preserved():
    assert set(BASE_EXPORTS) <= set(wise.__all__)
    for name in BASE_EXPORTS:
        assert hasattr(wise, name), f"public export {name!r} disappeared"
    # new exports may be interleaved (ruff RUF022 keeps __all__ isort-sorted), but the
    # relative order of the base exports must not change
    kept = [name for name in wise.__all__ if name in set(BASE_EXPORTS)]
    assert kept == list(BASE_EXPORTS), "existing exports must keep their relative order"


def test_module_constants():
    assert wise.SCHEMA_VERSION == 2
    assert norm_module.SCORING_MODES == ("layer_balanced", "flat")
    assert norm_module.DEFAULT_SCORING_MODE == "layer_balanced"
    assert sorted(wise.CONSTRAINT_TYPES) == [
        "balance",
        "exclusion",
        "lag",
        "metric",
        "precedence",
        "presence",
        "singularity",
    ]


def test_importing_wise_needs_no_optional_package_or_network():
    code = (
        "import sys, socket\n"
        "socket.socket = None\n"
        "import wise\n"
        "assert 'scipy' not in sys.modules and 'pm4py' not in sys.modules\n"
        "assert 'requests' not in sys.modules and 'urllib.request' not in sys.modules\n"
        "print(wise.__version__)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == wise.__version__


# ------------------------------------------------------------------ signatures
def test_eventlog_signature():
    assert_signature(
        wise.EventLog.__init__,
        [("self", EMPTY), ("events", EMPTY)],
        [
            ("case_col", "case:concept:name"),
            ("activity_col", "concept:name"),
            ("timestamp_col", "time:timestamp"),
            ("case_attributes", ()),
            ("exposure_col", None),
            ("exposure_agg", "max"),
            ("order_col", None),
            ("event_id_col", None),
            ("lifecycle_col", None),
            ("keep_transitions", ("complete",)),
            ("utc", False),
            ("timestamp_format", None),
            ("dayfirst", False),
            ("missing_timestamps", "raise"),
            ("window", None),
            ("keep_columns", None),
            ("dedupe", False),
        ],
    )
    assert_signature(wise.EventLog.derive, [("self", EMPTY), ("recipes", EMPTY)], [("overwrite", True)])
    assert_signature(wise.EventLog.from_csv.__func__, [("cls", EMPTY), ("path", EMPTY)], [("read_kwargs", None)])


def test_scoring_signatures():
    assert_signature(wise.evaluate_constraint, [("log", EMPTY), ("nc", EMPTY)], [], returns="pd.Series")
    assert_signature(wise.violation_matrix, [("log", EMPTY), ("norm", EMPTY)], [("return_scope", False)], returns="Any")
    assert_signature(
        wise.score,
        [("log", EMPTY), ("norm", EMPTY), ("views", None)],
        [("mode", None), ("derive", True)],
        returns="ScoreResult",
    )
    assert_signature(wise.ScoreResult.frame, [("self", EMPTY), ("view", None)], [])
    assert_signature(wise.ScoreResult.effective_weights, [("self", EMPTY), ("view", EMPTY)], [])
    assert_signature(wise.ScoreResult.penalties, [("self", EMPTY), ("view", EMPTY)], [])
    assert_signature(wise.ScoreResult.applicability_density, [("self", EMPTY), ("scope", False)], [])
    assert_signature(wise.ScoreResult.check_decomposition, [("self", EMPTY), ("atol", 1e-9)], [])
    assert_signature(wise.ScoreResult.worst_cases, [("self", EMPTY), ("view", EMPTY), ("n", 20), ("where", None)], [])
    assert_signature(wise.ScoreResult.trace, [("self", EMPTY), ("case_id", EMPTY)], [])


def test_prioritization_signatures():
    assert_signature(
        wise.prioritize,
        [
            ("data", EMPTY),
            ("by", EMPTY),
            ("view", None),
            ("gamma", 0.0),
            ("volume", "cases"),
            ("min_cases", 1),
            ("z", None),
            ("score_col", "score"),
            ("baseline", None),
            ("as_index", True),
        ],
        [],
    )
    assert_signature(
        wise.layer_drivers,
        [("data", EMPTY), ("by", EMPTY), ("view", None), ("layers", None), ("as_index", True)],
        [],
    )
    assert_signature(wise.constraint_drivers, [("result", EMPTY), ("view", EMPTY), ("where", None)], [])
    assert_signature(
        wise.penalty_mass, [("result", EMPTY), ("view", EMPTY), ("by", EMPTY), ("where", None), ("as_index", True)], []
    )
    assert_signature(wise.pareto, [("backlog", EMPTY), ("metric", "stable_PI")], [])
    assert_signature(wise.concentration, [("backlog", EMPTY), ("thresholds", (0.8, 0.95)), ("metric", "stable_PI")], [])
    assert_signature(
        wise.top_k_overlap,
        [("backlog_a", EMPTY), ("backlog_b", EMPTY), ("k", 20), ("metric", "stable_PI"), ("by", None)],
        [],
    )
    assert_signature(
        wise.view_agreement,
        [("result", EMPTY), ("by", EMPTY), ("k", 20), ("gamma", 0.0), ("method", "pearson"), ("metric", "stable_PI")],
        [],
    )
    assert_signature(wise.hotspot_table, [("backlog", EMPTY), ("top", 12), ("drivers", None)], [])
    assert_signature(wise.compare_periods, [("previous", EMPTY), ("current", EMPTY)], [])
    assert_signature(wise.estimate_gamma, [("data", EMPTY), ("by", EMPTY), ("view", None), ("score_col", "score")], [])


def test_norm_and_diagnostics_signatures():
    assert_signature(wise.Norm.to_dict, [("self", EMPTY)], [], returns="dict[str, Any]")
    assert_signature(wise.Norm.fingerprint, [("self", EMPTY)], [], returns="str")
    assert_signature(wise.Norm.dumps, [("self", EMPTY), ("indent", 2)], [])
    assert_signature(wise.Norm.from_dict.__func__, [("cls", EMPTY), ("d", EMPTY)], [])
    assert_signature(wise.Norm.load.__func__, [("cls", EMPTY), ("path", EMPTY)], [])
    assert_signature(wise.Norm.weight_vector, [("self", EMPTY), ("view", EMPTY)], [])
    assert_signature(
        wise.validation_table,
        [
            ("result", EMPTY),
            ("view", EMPTY),
            ("by", EMPTY),
            ("censored", None),
            ("replication", None),
            ("gamma", 0.0),
            ("ratio_flag", 2.0),
            ("top", None),
        ],
        [],
    )
    assert_signature(wise.observation_window, [("log", EMPTY), ("q", 0.001)], [])


BASE_DATACLASS_FIELDS = {
    "ScoreResult": [
        "norm",
        "cases",
        "violations",
        "in_scope",
        "scores",
        "contributions",
        "mode",
        "log",
        "norm_fingerprint",
        "wise_version",
        "_weights",
        "_eff_cache",
    ],
    "Norm": [
        "constraints",
        "layers",
        "views",
        "name",
        "version",
        "description",
        "scoring_mode",
        "derived_attributes",
        "metadata",
    ],
    "NormConstraint": ["id", "layer", "constraint", "weight", "applicability", "description"],
    "Layer": ["id", "name", "description"],
    "View": ["name", "layer_weights", "constraint_weights", "description"],
}


@pytest.mark.parametrize("name", sorted(BASE_DATACLASS_FIELDS))
def test_dataclass_field_order_is_a_prefix(name):
    """New fields may only be appended, and must carry a default."""
    cls = getattr(wise, name)
    fields = list(dataclasses.fields(cls))
    expected = BASE_DATACLASS_FIELDS[name]
    assert [f.name for f in fields[: len(expected)]] == expected
    for extra in fields[len(expected) :]:
        assert extra.default is not dataclasses.MISSING or extra.default_factory is not dataclasses.MISSING, (
            f"new {name} field {extra.name!r} must have a default"
        )


def test_scoreresult_defaults_and_manual_construction(p2p_result):
    defaults = {f.name: f.default for f in dataclasses.fields(wise.ScoreResult)}
    assert defaults["mode"] == "flat"
    assert defaults["log"] is None
    assert defaults["norm_fingerprint"] == ""
    assert defaults["wise_version"] == wise.__version__
    manual = wise.ScoreResult(
        p2p_result.norm,
        p2p_result.cases,
        p2p_result.violations,
        p2p_result.in_scope,
        p2p_result.scores,
        p2p_result.contributions,
    )
    assert manual.mode == "flat" and manual.log is None and manual.views == ["Finance", "Logistics"]
    with pytest.raises(NotScoredError):
        manual.trace("A")


# ------------------------------------------------------------- norm identities
P2P_FINGERPRINT = "b76e8f39df8510658c4ffe45d07c360b97a834b1655ed70fddc4a758873ba995"
BPIC19_FINGERPRINT = "e17ca18ed3c1a30114694c29455b86def27480bb34000e797e1e06a82dc644b0"
P2P_JSON_SHA256 = "7acda57c1c7e35fd4de92c5a3fd38e8bf0f2bbcc4e2e2761355e178e4c71b4bb"


def test_shipped_norm_fingerprints():
    assert datasets.running_p2p_norm().fingerprint() == P2P_FINGERPRINT
    assert wise.Norm.load(EXAMPLES / "running_p2p_norm.json").fingerprint() == P2P_FINGERPRINT
    assert wise.Norm.load(EXAMPLES / "bpic19_norm.json").fingerprint() == BPIC19_FINGERPRINT


def test_norm_serialisation_is_byte_stable():
    """``examples/quickstart.py`` rewrites this file on every run; it must not move."""
    on_disk = (EXAMPLES / "running_p2p_norm.json").read_bytes()
    assert hashlib.sha256(on_disk).hexdigest() == P2P_JSON_SHA256
    assert datasets.running_p2p_norm().dumps() == on_disk.decode("utf-8"), "canonical serialisation changed"
    assert json.loads(on_disk)["schema_version"] == 2


@pytest.mark.parametrize("path", ["running_p2p_norm.json", "bpic19_norm.json"])
def test_schema_2_round_trip_keeps_the_fingerprint(path):
    original = wise.Norm.load(EXAMPLES / path)
    assert wise.Norm.loads(original.dumps()).fingerprint() == original.fingerprint()
    assert wise.Norm.from_dict(original.to_dict()).fingerprint() == original.fingerprint()
    assert original.to_dict()["schema_version"] == 2


def test_norm_catalogue_identities():
    p2p = datasets.running_p2p_norm()
    assert p2p.scoring_mode == "flat"
    assert p2p.constraint_ids == ["c1", "c2", "c3", "c4", "c5", "c6"]
    assert p2p.layer_ids == ["completeness", "lead_times", "match", "handling", "exceptions"]
    assert p2p.view_names == ["Finance", "Logistics"]
    bpic = wise.Norm.load(EXAMPLES / "bpic19_norm.json")
    assert bpic.scoring_mode == "layer_balanced"
    assert len(bpic.constraints) == 29
    assert bpic.view_names == ["Finance", "Logistics", "Compliance", "Automation"]
    assert bpic.layer_ids == [
        "L1_closure_completeness",
        "L2_flow_discipline",
        "L3_timeliness_ageing",
        "L4_rework_instability",
        "L5_exceptions_corrections",
        "L6_value_commercial",
        "L7_effort_automation",
    ]
    assert wise.Norm(p2p.constraints, p2p.layers, p2p.views).scoring_mode == "layer_balanced"


# ------------------------------------------------------- running example frames
CASES = ["A", "B", "C", "D", "E"]
CONSTRAINTS = ["c1", "c2", "c3", "c4", "c5", "c6"]
LAYERS = ["completeness", "lead_times", "match", "handling", "exceptions"]
VIOLATIONS = {
    "c1": [0.0, 0.0, 0.0, 0.0, 1.0],
    "c2": [0.75, 0.0, 0.0, 0.0, 1.0],
    "c3": [0.0, 0.0, 0.65, 0.0, 1.0],
    "c4": [np.nan] * 5,
    "c5": [0.0, 2 / 3, 0.0, 0.0, 0.0],
    "c6": [0.0, 0.0, 0.0, 1.0, 0.0],
}
SCORES_FLAT = {
    "Finance": [0.6625, 0.9666666666666667, 0.87, 0.9, 0.15],
    "Logistics": [0.8875, 0.7, 0.9675, 0.9, 0.55],
}
SCORES_LAYER_BALANCED = {
    "Finance": [0.6785714285714286, 0.9682539682539683, 0.8452380952380952, 0.9047619047619048, 0.1428571428571428],
    "Logistics": [0.91, 0.76, 0.844, 0.92, 0.44],
}
CONTRIB_FINANCE = {
    "completeness": [0.0, 0.0, 0.0, 0.0, 0.2],
    "lead_times": [0.3375, 0.0, 0.0, 0.0, 0.45],
    "match": [0.0, 0.0, 0.13, 0.0, 0.2],
    "handling": [0.0, 1 / 30, 0.0, 0.0, 0.0],
    "exceptions": [0.0, 0.0, 0.0, 0.1, 0.0],
}


def test_log_shape_index_and_dtypes(p2p_log):
    assert len(p2p_log) == 5 and len(p2p_log.events) == 21
    cases = p2p_log.cases
    assert cases.index.name == "case"
    assert list(cases.index) == CASES
    assert list(cases.columns) == ["n_events", "first_ts", "last_ts", "flow_type", "company", "vendor"]
    assert cases["n_events"].dtype == np.dtype("int64")
    assert pd.api.types.is_datetime64_any_dtype(cases["first_ts"])
    assert pd.api.types.is_datetime64_any_dtype(cases["last_ts"])
    for col in ("flow_type", "company", "vendor"):
        # pandas 2 gives object, pandas 3 gives the string dtype; both are string-like
        assert pd.api.types.is_string_dtype(cases[col]) or cases[col].dtype == object
    assert list(p2p_log.activity_labels) == [
        "Create Purchase Order Item",
        "Record Goods Receipt",
        "Record Invoice Receipt",
        "Clear Invoice",
        "Cancel Invoice Receipt",
    ]
    assert cases["n_events"].tolist() == [4, 7, 4, 4, 2]


def test_violation_matrix_shape_masks_and_values(p2p_result):
    V = p2p_result.violations
    assert V.shape == (5, 6)
    assert V.index.name == "case" and list(V.index) == CASES
    assert list(V.columns) == CONSTRAINTS
    assert set(V.dtypes) == {np.dtype("float64")}
    for cid, expected in VIOLATIONS.items():
        mask = [v is np.nan or (isinstance(v, float) and np.isnan(v)) for v in expected]
        assert V[cid].isna().tolist() == mask, f"NaN mask of {cid} changed"
        close(V[cid].fillna(0.0), np.nan_to_num(np.asarray(expected, dtype=float)))
    # c4 is out of scope for every DF2 case: not applicable, not "satisfied"
    assert V["c4"].isna().all() and not p2p_result.in_scope["c4"].any()


def test_in_scope_is_boolean_and_distinct_from_evaluated(p2p_result):
    S = p2p_result.in_scope
    assert list(S.columns) == CONSTRAINTS and S.index.name == "case"
    assert set(S.dtypes) == {np.dtype("bool")}
    expected = pd.DataFrame({c: [c != "c4"] * 5 for c in CONSTRAINTS}, index=pd.Index(CASES, name="case"))
    pd.testing.assert_frame_equal(S, expected)
    pd.testing.assert_frame_equal(p2p_result.applicable, S)  # they coincide here, but are separate concepts
    close([p2p_result.applicability_density(), p2p_result.applicability_density(scope=True)], [5 / 6, 5 / 6])


def test_violation_matrix_return_scope_is_exactly_two_results(p2p_log, p2p_norm):
    out = wise.violation_matrix(p2p_log, p2p_norm, return_scope=True)
    assert isinstance(out, tuple) and len(out) == 2, "return_scope must stay a two-item return"
    V, S = out
    assert isinstance(V, pd.DataFrame) and isinstance(S, pd.DataFrame)
    single = wise.violation_matrix(p2p_log, p2p_norm)
    assert isinstance(single, pd.DataFrame)
    pd.testing.assert_frame_equal(single, V)


def test_running_example_scores_flat(p2p_result):
    assert p2p_result.mode == "flat"
    assert list(p2p_result.scores.columns) == ["Finance", "Logistics"]
    assert list(p2p_result.scores.index) == CASES
    assert p2p_result.scores.index.name == "case"
    assert set(p2p_result.scores.dtypes) == {np.dtype("float64")}
    assert not p2p_result.scores.isna().to_numpy().any()
    for view, expected in SCORES_FLAT.items():
        close(p2p_result.scores[view], expected)


def test_running_example_scores_layer_balanced(p2p_log, p2p_norm):
    res = wise.score(p2p_log, p2p_norm, mode="layer_balanced")
    assert res.mode == "layer_balanced"
    for view, expected in SCORES_LAYER_BALANCED.items():
        close(res.scores[view], expected)
    # the mode changes the score, never the observed violations or the NaN mask
    pd.testing.assert_frame_equal(res.violations, wise.score(p2p_log, p2p_norm).violations)
    pd.testing.assert_frame_equal(res.scores.isna(), wise.score(p2p_log, p2p_norm).scores.isna())
    # partial applicability is exactly where the two modes legitimately differ
    assert res.scores.loc["C", "Finance"] != pytest.approx(0.87, abs=1e-6)


def test_default_mode_comes_from_the_norm(p2p_log, p2p_norm):
    assert p2p_norm.scoring_mode == "flat"
    assert wise.score(p2p_log, p2p_norm).mode == "flat"
    stored = p2p_norm.replace(scoring_mode="layer_balanced")
    assert wise.score(p2p_log, stored).mode == "layer_balanced"
    # a runtime override does not rewrite the norm
    assert wise.score(p2p_log, p2p_norm, mode="layer_balanced").mode == "layer_balanced"
    assert p2p_norm.scoring_mode == "flat"


def test_layer_contributions_and_decomposition(p2p_result):
    contrib = p2p_result.contributions["Finance"]
    assert list(contrib.columns) == LAYERS and list(contrib.index) == CASES
    for layer, expected in CONTRIB_FINANCE.items():
        close(contrib[layer], expected)
    assert p2p_result.check_decomposition(atol=1e-12) < 1e-12
    for view in p2p_result.views:
        close(p2p_result.contributions[view].sum(axis=1), 1.0 - p2p_result.scores[view])
        close(p2p_result.penalties(view).sum(axis=1), 1.0 - p2p_result.scores[view])


def test_effective_weights_renormalise_over_the_applicable_set(p2p_result):
    w = p2p_result.effective_weights("Finance")
    assert list(w.columns) == CONSTRAINTS and list(w.index) == CASES
    assert not w.isna().to_numpy().any(), "effective weights are nan_to_num'ed to 0.0 in the public frame"
    close(w["c4"], [0.0] * 5)
    close(w.sum(axis=1), [1.0] * 5)
    close(w.loc["A"], [0.20, 0.45, 0.20, 0.0, 0.05, 0.10])


def test_frame_and_summary_column_order(p2p_result):
    case_cols = ["n_events", "first_ts", "last_ts", "flow_type", "company", "vendor"]
    assert list(p2p_result.frame("Finance").columns) == [
        *case_cols,
        "score",
        *[f"contrib__{layer}" for layer in LAYERS],
    ]
    assert list(p2p_result.frame().columns) == [
        *case_cols,
        "score__Finance",
        *[f"contrib__Finance__{layer}" for layer in LAYERS],
        "score__Logistics",
        *[f"contrib__Logistics__{layer}" for layer in LAYERS],
    ]
    s = p2p_result.summary()
    assert s.index.name == "view" and list(s.index) == ["Finance", "Logistics"]
    assert list(s.columns) == ["n_scored", "mean_score", *[f"contrib__{layer}" for layer in LAYERS]]
    assert s["n_scored"].tolist() == [5, 5]
    close(s["mean_score"], [0.7098333333333334, 0.801])


def test_unscored_stays_unscored(p2p_log, p2p_norm):
    """All applicable weights removed → NaN, not a satisfied zero-penalty case."""
    nowhere = p2p_norm.map_constraints(lambda c: c.replace(applicability={"flow_type": ["DF1"]}))
    res = wise.score(p2p_log, nowhere)
    assert res.scores.isna().to_numpy().all()
    assert res.contributions["Finance"].isna().to_numpy().all()
    assert not res.in_scope.to_numpy().any()
    # nothing evaluated is density 0.0 here (NaN is reserved for an empty log)
    assert res.applicability_density() == 0.0 and res.applicability_density(scope=True) == 0.0
    with pytest.raises(NotScoredError):
        wise.prioritize(res, "company", view="Finance")


# ------------------------------------------------------------- prioritisation
BACKLOG_COLUMNS = [
    "n_cases",
    "mean_score",
    "volume",
    "gap",
    "PI",
    "stable_mean",
    "stable_gap",
    "stable_PI",
    "global_mean",
]
BACKLOG_COLUMNS_WITH_Z = [
    "n_cases",
    "mean_score",
    "volume",
    "gap",
    "PI",
    "stable_mean",
    "stable_gap",
    "stable_PI",
    "se",
    "gap_lower",
    "PI_lower",
    "global_mean",
]
GLOBAL_MEAN_FINANCE = 0.7098333333333334


def test_backlog_columns_order_and_values(p2p_result):
    b = wise.prioritize(p2p_result, "company", view="Finance", gamma=1.0)
    assert list(b.columns) == BACKLOG_COLUMNS
    assert b.index.name == "company"
    assert list(b.index) == ["B", "A"], "ranking order changed"
    assert b["n_cases"].dtype == np.dtype("int64")
    assert set(b[BACKLOG_COLUMNS[1:]].dtypes) == {np.dtype("float64")}
    assert b["n_cases"].tolist() == [3, 2]
    close(b["mean_score"], [0.64, 0.8145833333333334])
    close(b["volume"], [3.0, 2.0])
    close(b["gap"], [0.06983333333333341, 0.0])
    close(b["PI"], [0.20950000000000024, 0.0])
    close(b["stable_mean"], [0.6574583333333334, 0.7796666666666667])
    close(b["stable_gap"], [0.05237500000000006, 0.0])
    close(b["stable_PI"], [0.15712500000000018, 0.0])
    close(b["global_mean"], [GLOBAL_MEAN_FINANCE] * 2)
    assert b.attrs == {
        "view": "Finance",
        "gamma": 1.0,
        "baseline": GLOBAL_MEAN_FINANCE,
        "volume": "cases",
        "by": ["company"],
    }


def test_backlog_z_columns_and_position(p2p_result):
    b = wise.prioritize(p2p_result, "company", view="Finance", gamma=1.0, z=1.96)
    assert list(b.columns) == BACKLOG_COLUMNS_WITH_Z, "the z columns must stay before global_mean"


def test_baseline_argument_replaces_the_global_mean(p2p_result):
    b = wise.prioritize(p2p_result, "company", view="Finance", baseline=1.0)
    assert list(b.index) == ["B", "A"]
    close(b["global_mean"], [1.0, 1.0])
    close(b["gap"], [0.36, 0.18541666666666656])
    assert b.attrs["baseline"] == 1.0


def test_case_level_backlog_ordering(p2p_result):
    b = wise.prioritize(p2p_result, "case", view="Finance")
    assert list(b.index) == ["E", "A", "B", "C", "D"]
    close(b["PI"], [0.5598333333333333, 0.04733333333333334, 0.0, 0.0, 0.0])
    close(b["stable_PI"], b["PI"])  # gamma = 0 leaves the raw index untouched


def test_tie_break_is_stable_and_input_order_independent():
    df = pd.DataFrame({"slice": list("bbbaac"), "score": [0.5] * 6})
    assert list(wise.prioritize(df, "slice").index) == ["b", "a", "c"]
    assert list(wise.prioritize(df.iloc[::-1], "slice").index) == ["b", "a", "c"]


def test_layer_drivers_columns_order_and_deltas(p2p_result):
    ld = wise.layer_drivers(p2p_result, "company", view="Finance")
    assert list(ld.columns) == [
        "n_cases",
        *LAYERS,
        *[f"{layer}__delta" for layer in LAYERS],
        "dominant_layer",
    ]
    assert list(ld.index) == ["A", "B"], "layer_drivers keeps the sorted group order, not the PI order"
    assert ld["n_cases"].tolist() == [2, 3]
    close(ld["lead_times"], [0.16874999999999996, 0.15])
    close(ld["match__delta"], [-0.066, 0.044])
    assert ld["dominant_layer"].tolist() == ["lead_times", "match"]
    # the deltas contrast against the current scored population
    for layer in LAYERS:
        weighted = float((ld["n_cases"] * ld[layer]).sum() / ld["n_cases"].sum())
        close(ld[layer] - ld[f"{layer}__delta"], [weighted] * 2)


def test_constraint_drivers_columns_and_ordering(p2p_result):
    cd = wise.constraint_drivers(p2p_result, "Finance", {"company": "B"})
    assert cd.index.name == "constraint"
    assert list(cd.columns) == [
        "layer",
        "type",
        "mean_penalty",
        "mean_violation",
        "share_violated",
        "share_in_scope",
        "share_evaluated",
        "description",
    ]
    assert list(cd.index) == ["c2", "c3", "c1", "c6", "c5", "c4"]
    close(cd["mean_penalty"], [0.15, 0.11, 1 / 15, 1 / 30, 0.0, 0.0])
    # an out-of-scope constraint keeps NaN measures and zero shares, never a zero violation
    assert np.isnan(cd.loc["c4", "mean_violation"]) and np.isnan(cd.loc["c4", "share_violated"])
    assert cd.loc["c4", "share_in_scope"] == 0.0 and cd.loc["c4", "share_evaluated"] == 0.0
    assert list(wise.constraint_drivers(p2p_result, "Finance").index) == ["c2", "c3", "c1", "c6", "c5", "c4"]


def test_penalty_mass_columns_and_ordering(p2p_result):
    pm = wise.penalty_mass(p2p_result, "Finance", "vendor")
    assert list(pm.columns) == ["n_cases", "penalty_mass", "mean_penalty", "share", "cum_share", "rank"]
    assert list(pm.index) == ["V2", "V1"]
    close(pm["penalty_mass"], [0.8833333333333332, 0.5674999999999999])
    close(pm["cum_share"], [0.6088454910970706, 1.0])
    assert pm["rank"].tolist() == [1, 2]


def test_worst_cases_ordering(p2p_result):
    assert list(p2p_result.worst_cases("Finance").index) == ["E", "A", "C", "D", "B"]
    assert list(p2p_result.worst_cases("Finance", n=2).index) == ["E", "A"]
    assert list(p2p_result.worst_cases("Finance", where={"company": "B"}).index) == ["E", "C", "D"]


def test_view_agreement_and_hotspots(p2p_result):
    va = wise.view_agreement(p2p_result, "case", k=2)
    assert list(va.columns) == ["top2_overlap", "score_correlation"]
    assert list(va.index) == [("Finance", "Logistics")]
    close(va["top2_overlap"], [1 / 3])
    close(va["score_correlation"], [0.6812808143845632])
    ht = wise.hotspot_table(wise.prioritize(p2p_result, "case", view="Finance"), top=5)
    assert list(ht.columns) == [*BACKLOG_COLUMNS, "hotspot"]
    assert list(ht.index) == ["E", "A"]
    assert ht["hotspot"].tolist() == ["severity", "reservoir"]


def test_pareto_and_concentration_columns(p2p_result):
    b = wise.prioritize(p2p_result, "case", view="Finance")
    p = wise.pareto(b)
    assert list(p.columns) == [*BACKLOG_COLUMNS, "rank", "share", "cum_share"]
    close(p["cum_share"].iloc[-1], 1.0)
    c = wise.concentration(b)
    assert c.index.name == "threshold" and list(c.columns) == ["top_k", "share_of_slices"]


def test_estimate_gamma_returns_inf_when_slices_do_not_differ(p2p_result):
    assert wise.estimate_gamma(p2p_result, "company", view="Finance") == float("inf")


def test_validation_table_columns(p2p_result):
    vt = wise.validation_table(p2p_result, "Finance", "company")
    assert list(vt.columns) == ["n_cases", "stable_gap", "stable_PI", "reading"]
    assert list(vt.index) == ["B", "A"]


# ------------------------------------------------------------------------- CLI
def test_cli_default_stdout_csv_header(tmp_path):
    log = tmp_path / "log.csv"
    datasets.running_p2p_events().to_csv(log, index=False)
    norm = tmp_path / "norm.json"
    datasets.running_p2p_norm().dump(norm)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from wise.cli import main; sys.exit(main(sys.argv[1:]))",
            "score",
            str(norm),
            str(log),
            "--case",
            "case",
            "--activity",
            "activity",
            "--timestamp",
            "time",
            "--attr",
            "company",
            "--attr",
            "flow_type",
            "--by",
            "company",
            "--view",
            "Finance",
            "--gamma",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert lines[0] == "company," + ",".join(BACKLOG_COLUMNS), "default CLI CSV columns changed"
    assert lines[1].startswith("B,3,")
    assert proc.stderr == "", "diagnostic chatter must not be mixed into the stdout CSV"


# ------------------------------------------------------ private aggregation kernel
def test_private_weight_kernel_is_the_single_scoring_formula(p2p_norm):
    """S4 will reuse this helper for object units; pin its current contract."""
    M = np.array([[1.0, 1.0, 1.0, 0.0, 1.0, 1.0]])
    w = p2p_norm.weight_vector("Finance").to_numpy(dtype=float)
    flat = scoring._effective_weights(M, w, p2p_norm, "flat")
    close(flat, [[0.20, 0.45, 0.20, 0.0, 0.05, 0.10]])
    balanced = scoring._effective_weights(M, w, p2p_norm, "layer_balanced")
    close(balanced.sum(axis=1), [1.0])
    close(balanced[0, 2], 0.25 / 1.05)  # the whole 'match' layer weight falls on c3
    none_applicable = scoring._effective_weights(np.zeros((1, 6)), w, p2p_norm, "flat")
    assert np.isnan(none_applicable).all(), "no applicable constraint must stay NaN, not 0.0"
