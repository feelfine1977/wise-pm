"""Golden pipeline snapshot: the full pipeline on the synthetic 2k-case P2P log.

Every output that ``scripts/make_golden.py`` writes under ``tests/data/golden``
is recomputed here from the committed dataset and norm and compared with the
committed parquet record exactly (``pandas.testing.assert_frame_equal`` with
``check_exact=True``, no ``rtol``/``atol``; index and column order included),
and every human-readable CSV must equal its record rounded to 12 decimals.
Exact parity is the policy for unchanged classic arithmetic; only a new
numerical method may declare a tolerance, next to its own golden. A failure
means the library's numbers moved: decide whether the change is a listed fix,
then regenerate only with

    /Users/ula/code/PhD/WISE/wise-next/.venv/bin/python scripts/make_golden.py --write

and review the diff. Never edit golden files by hand.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

import wise

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "tests" / "data"


def _load_generator() -> ModuleType:
    """Import ``scripts/make_golden.py`` so that the test runs the very same pipeline."""
    path = REPO / "scripts" / "make_golden.py"
    spec = importlib.util.spec_from_file_location("make_golden", path)
    assert spec is not None, path
    assert spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mg = _load_generator()
GOLDEN = DATA / mg.GOLDEN_SUBDIR
#: WISE_GOLDEN_MODE: "auto" (default) compares exactly when this environment matches the record host named in the
#: golden manifest and within mg.REPORT_ATOL otherwise; "exact" and "report" force either. Exactness across hosts is
#: not a property of IEEE arithmetic (SIMD reduction order, fused multiply-add, BLAS kernels move the last bit);
#: differences beyond REPORT_ATOL fail in every mode.
GOLDEN_MODE_REQUESTED = os.environ.get("WISE_GOLDEN_MODE", "auto")
ENVIRONMENT_DIFFERENCES = mg.environment_differences(mg.read_json(GOLDEN / mg.MANIFEST_FILE).get("environment"))
GOLDEN_MODE = (
    GOLDEN_MODE_REQUESTED if GOLDEN_MODE_REQUESTED in {"exact", "report"} else ("report" if ENVIRONMENT_DIFFERENCES else "exact")
)
PLACEHOLDER_FROM = pd.Timestamp("2099-01-01")
# Every frame that has a readable CSV next to its parquet record, as (name, mode) pairs.
READABLE = [(n, m) for m in mg.MODES for n in mg.MODE_OUTPUTS if n in mg.READABLE_OUTPUTS] + [
    (n, None) for n in mg.COMMON_OUTPUTS if n in mg.READABLE_OUTPUTS
]

pytestmark = pytest.mark.spec("golden")


# ------------------------------------------------------------------ fixtures (module scope: the pipeline runs once)
@pytest.fixture(scope="module")
def log() -> wise.EventLog:
    return mg.load_event_log(DATA / mg.DATASET_FILE)


@pytest.fixture(scope="module")
def norm() -> wise.Norm:
    return wise.Norm.load(GOLDEN / mg.NORM_FILE)


@pytest.fixture(scope="module")
def results(log: wise.EventLog, norm: wise.Norm) -> dict[str, wise.ScoreResult]:
    return {mode: wise.score(log, norm, mode=mode) for mode in mg.MODES}


@pytest.fixture(scope="module")
def mode_frames(results: dict[str, wise.ScoreResult]) -> dict[str, dict[str, pd.DataFrame]]:
    return {mode: mg.mode_outputs(result) for mode, result in results.items()}


@pytest.fixture(scope="module")
def common_frames(log: wise.EventLog, results: dict[str, wise.ScoreResult]) -> dict[str, pd.DataFrame]:
    return mg.common_outputs(log, results[mg.VALIDATION_MODE])


@pytest.fixture(scope="module")
def events() -> pd.DataFrame:
    """The committed dataset as written, before ``EventLog`` touches it."""
    return pd.read_parquet(DATA / mg.DATASET_FILE)


@pytest.fixture(scope="module")
def truth() -> dict[str, Any]:
    return mg.read_json(DATA / mg.GROUND_TRUTH_FILE)


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return mg.read_json(GOLDEN / mg.MANIFEST_FILE)


def test_golden_mode_is_recorded(request: pytest.FixtureRequest) -> None:
    """The resolved comparison mode and the environment differences go into the junit report of every CI cell."""
    request.node.user_properties.append(("golden_mode", GOLDEN_MODE))
    request.node.user_properties.append(("environment_differences", "; ".join(ENVIRONMENT_DIFFERENCES) or "record host"))
    assert GOLDEN_MODE in {"exact", "report"}
    if GOLDEN_MODE_REQUESTED == "auto" and not ENVIRONMENT_DIFFERENCES:
        assert GOLDEN_MODE == "exact"


def _assert_golden(actual: pd.DataFrame, name: str, mode: str | None) -> None:
    """The recomputed frame must equal the parquet record exactly.

    ``mg.assert_frames_exact`` is ``assert_frame_equal(check_exact=True)`` after the null normalisation of object
    columns; ``--check`` uses the very same function, so the script and this test cannot drift apart.
    """
    path = GOLDEN / mg.record_file(name, mode)
    assert path.exists(), f"{path} is missing; regenerate with scripts/make_golden.py --write"
    if GOLDEN_MODE == "report":
        print(mg.assert_frames_report(actual, mg.read_record(path), obj=path.name))  # drift summary for CI logs
        return
    mg.assert_frames_exact(actual, mg.read_record(path), obj=path.name)


# ------------------------------------------------------------------ provenance
def test_dataset_and_norm_match_manifest(log: wise.EventLog, norm: wise.Norm, manifest: dict[str, Any]) -> None:
    dataset = DATA / mg.DATASET_FILE
    assert dataset.stat().st_size < 600_000
    assert manifest["dataset"]["sha256"] == mg.sha256(dataset)
    assert manifest["dataset"]["n_cases"] == len(log) == 2000
    assert manifest["dataset"]["n_events"] == len(log.events)
    assert 12_000 <= len(log.events) <= 18_000
    assert manifest["norm"]["fingerprint"] == norm.fingerprint() == mg.golden_norm().fingerprint()
    assert norm.check(log) == []


def test_manifest_lists_every_output(
    mode_frames: dict[str, dict[str, pd.DataFrame]], common_frames: dict[str, pd.DataFrame], manifest: dict[str, Any]
) -> None:
    listed = manifest["frames"]
    computed: dict[tuple[str, str | None], pd.DataFrame] = {
        **{(n, m): f for m, frames in mode_frames.items() for n, f in frames.items()},
        **{(n, None): f for n, f in common_frames.items()},
    }
    assert set(listed) == {mg.golden_stem(n, m) for n, m in computed}
    prefix = f"{mg.GOLDEN_SUBDIR}/"
    for (name, mode), frame in computed.items():
        record_rel = f"{prefix}{mg.record_file(name, mode)}"
        readable = mg.readable_file(name, mode)
        readable_rel = f"{prefix}{readable}" if readable else None
        entry = {"record": record_rel, "readable": readable_rel, "rows": len(frame), "columns": [str(c) for c in frame.columns]}
        assert listed[mg.golden_stem(name, mode)] == entry, (name, mode)
        assert (DATA / record_rel).exists(), record_rel
        assert readable_rel is None or (DATA / readable_rel).exists(), readable_rel
    # ... and nothing else lies under tests/data/golden: a stale file left behind by a rename would go unnoticed otherwise.
    expected_files = {e[k].removeprefix(prefix) for e in listed.values() for k in ("record", "readable") if e[k]}
    expected_files |= {mg.NORM_FILE, mg.MANIFEST_FILE, mg.QUALITY_FILE}
    on_disk = {p.name for p in GOLDEN.iterdir() if p.is_file() and not p.name.startswith(".")}
    assert on_disk == expected_files
    assert not [p for p in GOLDEN.iterdir() if p.is_dir()], "no subdirectories are expected under tests/data/golden"


# ------------------------------------------------------------------ mode-dependent outputs
@pytest.mark.parametrize("mode", mg.MODES)
def test_violations(mode: str, mode_frames: dict[str, dict[str, pd.DataFrame]]) -> None:
    _assert_golden(mode_frames[mode]["violations"], "violations", mode)


@pytest.mark.parametrize("mode", mg.MODES)
def test_in_scope(mode: str, mode_frames: dict[str, dict[str, pd.DataFrame]]) -> None:
    _assert_golden(mode_frames[mode]["in_scope"], "in_scope", mode)


@pytest.mark.parametrize("mode", mg.MODES)
def test_scores(mode: str, mode_frames: dict[str, dict[str, pd.DataFrame]]) -> None:
    _assert_golden(mode_frames[mode]["scores"], "scores", mode)


@pytest.mark.parametrize("view", mg.VIEW_NAMES)
@pytest.mark.parametrize("mode", mg.MODES)
def test_contributions(mode: str, view: str, mode_frames: dict[str, dict[str, pd.DataFrame]]) -> None:
    _assert_golden(mode_frames[mode][f"contributions_{view}"], f"contributions_{view}", mode)


@pytest.mark.parametrize("mode", mg.MODES)
def test_backlog_company(mode: str, mode_frames: dict[str, dict[str, pd.DataFrame]]) -> None:
    _assert_golden(mode_frames[mode]["backlog_company"], "backlog_company", mode)


@pytest.mark.parametrize("mode", mg.MODES)
def test_backlog_company_vendor(mode: str, mode_frames: dict[str, dict[str, pd.DataFrame]]) -> None:
    _assert_golden(mode_frames[mode]["backlog_company_vendor"], "backlog_company_vendor", mode)


@pytest.mark.parametrize("mode", mg.MODES)
def test_layer_drivers_company(mode: str, mode_frames: dict[str, dict[str, pd.DataFrame]]) -> None:
    _assert_golden(mode_frames[mode]["layer_drivers_company"], "layer_drivers_company", mode)


@pytest.mark.parametrize("mode", mg.MODES)
def test_constraint_drivers_all(mode: str, mode_frames: dict[str, dict[str, pd.DataFrame]]) -> None:
    _assert_golden(mode_frames[mode]["constraint_drivers_all"], "constraint_drivers_all", mode)


@pytest.mark.parametrize("mode", mg.MODES)
def test_penalty_mass_vendor(mode: str, mode_frames: dict[str, dict[str, pd.DataFrame]]) -> None:
    _assert_golden(mode_frames[mode]["penalty_mass_vendor"], "penalty_mass_vendor", mode)


@pytest.mark.parametrize("mode", mg.MODES)
def test_view_agreement(mode: str, mode_frames: dict[str, dict[str, pd.DataFrame]]) -> None:
    _assert_golden(mode_frames[mode]["view_agreement"], "view_agreement", mode)


@pytest.mark.parametrize("mode", mg.MODES)
def test_hotspot_company(mode: str, mode_frames: dict[str, dict[str, pd.DataFrame]]) -> None:
    _assert_golden(mode_frames[mode]["hotspot_company"], "hotspot_company", mode)


# ------------------------------------------------------------------ mode-independent outputs
def test_event_replication(common_frames: dict[str, pd.DataFrame]) -> None:
    _assert_golden(common_frames["event_replication"], "event_replication", None)


def test_right_censored(common_frames: dict[str, pd.DataFrame]) -> None:
    _assert_golden(common_frames["right_censored"], "right_censored", None)


def test_validation_table_company(common_frames: dict[str, pd.DataFrame]) -> None:
    _assert_golden(common_frames["validation_table_company"], "validation_table_company", None)


def test_quality_report(log: wise.EventLog) -> None:
    """``log.validate()`` equals the committed JSON exactly (both sides round floats to ``CSV_DECIMALS``)."""
    assert mg.quality_report(log) == mg.read_json(GOLDEN / mg.QUALITY_FILE)


# ------------------------------------------------------------------ readable CSV companions
@pytest.mark.parametrize(("name", "mode"), READABLE, ids=[mg.golden_stem(n, m) for n, m in READABLE])
def test_readable_csv_equals_record_rounded(name: str, mode: str | None) -> None:
    """Each CSV is its parquet record rounded to ``CSV_DECIMALS`` with LF line endings; a stale CSV fails here."""
    record = mg.read_record(GOLDEN / mg.record_file(name, mode))
    readable = mg.readable_file(name, mode)
    assert readable is not None
    csv = GOLDEN / readable
    assert b"\r" not in csv.read_bytes(), f"{csv.name} must use LF line endings"
    mg.assert_frames_exact(mg.read_readable(csv, like=record), record.round(mg.CSV_DECIMALS), obj=csv.name)


# ------------------------------------------------------------------ planted ground truth (known answers)
def test_planted_event_ids_match_dataset(events: pd.DataFrame, truth: dict[str, Any]) -> None:
    """The truth's event-level ids point at the rows of the committed parquet that actually carry the defect."""
    planted = truth["planted"]
    missing = events.loc[events["time"].isna(), "eventID"]
    assert set(planted["missing_timestamp_event_ids"]) == set(missing.tolist())
    assert len(planted["missing_timestamp_event_ids"]) == len(missing) == truth["realised"]["n_missing_timestamps"]
    assert missing.is_unique  # no missing timestamp on a replicated event: the ids identify rows one to one

    placeholder = planted["placeholder_timestamp"]
    rows = events[events["time"] >= PLACEHOLDER_FROM]
    assert (rows["time"] == pd.Timestamp(placeholder["value"])).all()
    assert sorted(rows["eventID"].tolist()) == placeholder["event_ids"]
    assert sorted(rows["case"].unique().tolist()) == placeholder["cases"]
    assert len(rows) == truth["realised"]["n_placeholder_events"]
    # eventIDs repeat only through the planted replication (every event of those cases is extracted twice)
    duplicated_cases = set(events.loc[events["eventID"].duplicated(), "case"].tolist())
    assert duplicated_cases == set(planted["replicated_cases"])


def test_planted_censoring_is_flagged(common_frames: dict[str, pd.DataFrame], truth: dict[str, Any]) -> None:
    flagged = common_frames["right_censored"]["right_censored"]
    planted = truth["planted"]["right_censored"]
    assert flagged.loc[planted["open_invoices"]].all()  # open invoices: GR and INV in the last 20 days, no CLR
    assert not flagged.loc[planted["open_receipts"]].any()  # no invoice yet, so not "opened by" INV


def test_planted_replication_is_detected(common_frames: dict[str, pd.DataFrame], truth: dict[str, Any]) -> None:
    rep = common_frames["event_replication"]
    planted = truth["planted"]["replicated_cases"]
    assert (rep.loc[planted, "replication_ratio"] >= 2.0).all()
    assert (rep.loc[planted, "replicated_share"] == 1.0).all()
    assert (rep.drop(index=planted)["replicated_share"] < 1.0).all()


@pytest.mark.parametrize("mode", mg.MODES)
def test_planted_hotspots(mode: str, mode_frames: dict[str, dict[str, pd.DataFrame]], truth: dict[str, Any]) -> None:
    frames = mode_frames[mode]
    reservoir = truth["planted"]["reservoir_company"]["id"]
    severity = truth["planted"]["severity_vendor"]
    assert frames["backlog_company"].index[0] == reservoir
    hotspots = frames["hotspot_company"]["hotspot"]
    assert hotspots.loc[reservoir] == "reservoir"
    assert hotspots.loc[severity["company"]] == "severity"
    assert frames["backlog_company_vendor"].index[0] == (severity["company"], severity["id"])
    drivers = frames["layer_drivers_company"]["dominant_layer"]
    assert drivers.loc[reservoir] == "lead_times"
    assert drivers.loc[truth["planted"]["manual_company"]["id"]] == "effort"
