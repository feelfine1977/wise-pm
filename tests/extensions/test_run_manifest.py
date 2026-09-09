"""The run record: actual mode, identities, horizons, explicit hashing (E01)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import wise
from wise.errors import EvidenceError
from wise.evidence import RunManifest, fingerprint_events, fingerprint_result, new_run_id


@pytest.fixture
def log():
    return wise.datasets.running_p2p_log()


@pytest.fixture
def norm():
    return wise.datasets.running_p2p_norm()


# ------------------------------------------------------------- actual mode
def test_the_manifest_records_the_mode_actually_used_not_the_norm_default(log, norm):
    """F07: a runtime override is the run's mode, whatever the norm says."""
    default = wise.score(log, norm)
    assert default.manifest.mode == "flat"
    assert default.manifest.mode_source == "norm_default"
    assert default.manifest.norm_default_mode == "flat"

    override = wise.score(log, norm, mode="layer_balanced")
    assert override.manifest.mode == "layer_balanced"
    assert override.manifest.mode_source == "call_override"
    assert override.manifest.norm_default_mode == "flat"
    assert override.mode == "layer_balanced"


def test_a_manually_constructed_result_has_no_run_record(log, norm):
    """ScoreResult.mode defaults to 'flat' whatever the norm says; the run
    record is the thing that knows, and a hand-built result simply has none."""
    real = wise.score(log, norm, mode="layer_balanced")
    manual = wise.ScoreResult(real.norm, real.cases, real.violations, real.in_scope, real.scores, real.contributions)
    assert manual.mode == "flat"
    assert manual.manifest is None and manual.evidence is None


def test_the_selected_views_are_recorded(log, norm):
    assert wise.score(log, norm).manifest.views == ("Finance", "Logistics")
    assert wise.score(log, norm, views="Finance").manifest.views == ("Finance",)


# ------------------------------------------------------------- identities
def test_configuration_fingerprint_is_deterministic_and_run_id_is_not(log, norm):
    first, second = wise.score(log, norm), wise.score(log, norm)
    assert first.manifest.config_fingerprint() == second.manifest.config_fingerprint()
    assert first.manifest.run_id != second.manifest.run_id
    assert first.manifest.created_at != "" and first.manifest.run_id not in first.manifest.config_fingerprint()


def test_a_different_mode_or_view_is_a_different_configuration(log, norm):
    base = wise.score(log, norm).manifest.config_fingerprint()
    assert wise.score(log, norm, mode="layer_balanced").manifest.config_fingerprint() != base
    assert wise.score(log, norm, views="Finance").manifest.config_fingerprint() != base


def test_run_ids_are_unique():
    assert new_run_id() != new_run_id()
    assert new_run_id("evidence").startswith("evidence-")


def test_the_norm_fingerprint_is_the_shipped_one(log, norm):
    assert wise.score(log, norm).manifest.norm_fingerprint == norm.fingerprint()


# --------------------------------------------------------- input and options
def test_the_manifest_records_the_preparation_options_and_tie_order_policy(log, norm):
    manifest = wise.score(log, norm).manifest
    options = manifest.preprocessing
    assert options["case_col"] == "case" and options["timestamp_col"] == "time"
    assert options["tie_order_policy"] == "input_row_order"
    assert options["dedupe"] is False and options["missing_timestamps"] == "raise"
    assert manifest.input.n_events == 21 and manifest.input.n_cases == 5


def test_an_order_column_changes_the_recorded_tie_order_policy(norm):
    events = wise.datasets.running_p2p_events()
    events["seq"] = range(len(events))
    ordered = wise.EventLog(
        events, case_col="case", activity_col="activity", timestamp_col="time", case_attributes=["flow_type"], order_col="seq"
    )
    assert ordered.options()["tie_order_policy"] == "order_col:seq"


def test_source_identity_is_recorded_as_unavailable_when_no_event_id_column(log, norm):
    manifest = wise.score(log, norm).manifest
    assert manifest.input.event_id_col is None
    assert manifest.input.source_identity_available is False


def test_the_data_digest_is_absent_until_it_is_explicitly_computed(log, norm):
    manifest = wise.score(log, norm).manifest
    assert manifest.input.data_digest is None
    assert manifest.input.digest_algorithm is None

    digest, canonicalisation = fingerprint_events(log)
    with_digest = manifest.with_data_digest(digest, algorithm="sha256", canonicalisation=canonicalisation)
    assert with_digest.input.data_digest == digest
    assert "stored row order" in with_digest.input.digest_canonicalisation
    assert manifest.input.data_digest is None, "the original record is immutable"


def test_the_data_digest_depends_on_semantic_row_order():
    """Sorting to get a convenient order-insensitive hash would erase a real
    difference: with no order column, timestamp ties are broken by row order,
    and the two orders are different logs."""
    tied = pd.DataFrame(
        {
            "case": ["a", "a", "a"],
            "activity": ["GR", "INV", "CLR"],
            "time": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
            "flow_type": ["DF2"] * 3,
        }
    )
    kwargs = {"case_col": "case", "activity_col": "activity", "timestamp_col": "time", "case_attributes": ["flow_type"]}
    forward = wise.EventLog(tied, **kwargs)
    swapped = wise.EventLog(tied.iloc[[1, 0, 2]].reset_index(drop=True), **kwargs)
    assert list(forward.events["activity"]) == ["GR", "INV", "CLR"]
    assert list(swapped.events["activity"]) == ["INV", "GR", "CLR"], "the tie is broken by input row order"
    assert fingerprint_events(forward)[0] != fingerprint_events(swapped)[0]


def test_an_unknown_hash_algorithm_is_refused(log):
    with pytest.raises(EvidenceError, match="unknown hash algorithm"):
        fingerprint_events(log, algorithm="not-a-hash")


def test_the_result_fingerprint_is_separate_from_the_configuration(log, norm):
    result = wise.score(log, norm)
    digest = fingerprint_result(result.scores, result.violations)
    assert digest != result.manifest.config_fingerprint()
    assert digest == fingerprint_result(result.scores, result.violations)
    other = wise.score(log, norm, mode="layer_balanced")
    assert fingerprint_result(other.scores, other.violations) != digest


def test_git_metadata_stays_unknown_unless_supplied(log, norm):
    manifest = wise.score(log, norm).manifest
    assert manifest.environment.git_commit is None
    assert manifest.with_commit("deadbeef").environment.git_commit == "deadbeef"
    assert manifest.environment.wise_version == wise.__version__


def test_importing_wise_reads_no_git_state_and_opens_no_socket():
    import subprocess
    import sys

    code = (
        "import socket, subprocess, sys\n"
        "socket.socket = None\n"
        "subprocess.run = lambda *a, **k: (_ for _ in ()).throw(AssertionError('no subprocess at import'))\n"
        "import wise, wise.evidence\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


# ----------------------------------------------------------- observation scope
def test_the_observation_policy_distinguishes_an_explicit_from_a_derived_window(norm):
    events = wise.datasets.running_p2p_events()
    kwargs = {
        "case_col": "case",
        "activity_col": "activity",
        "timestamp_col": "time",
        "case_attributes": ["flow_type", "company", "vendor"],
    }
    derived = wise.score(wise.EventLog(events, **kwargs), norm).manifest.observation
    assert derived.window_source == "derived_quantile"
    assert derived.policy_id.startswith("derived-quantile-window")

    explicit = wise.score(wise.EventLog(events, window=("2024-01-01", "2024-02-01"), **kwargs), norm).manifest.observation
    assert explicit.window_source == "explicit"
    assert explicit.window_start.startswith("2024-01-01")
    assert explicit.policy_id.startswith("explicit-window")


def test_resolved_horizons_are_recorded_when_a_censoring_policy_resolves_one():
    events = wise.datasets.running_p2p_events()
    log = wise.EventLog(
        events,
        case_col="case",
        activity_col="activity",
        timestamp_col="time",
        case_attributes=["flow_type"],
        window=("2024-01-01", "2024-02-01"),
    )
    censoring = wise.Norm(
        constraints=(
            wise.NormConstraint(
                "c", "L", wise.Lag("Record Goods Receipt", "Record Invoice Receipt", delta=10, width=20, missing_b="censor")
            ),
        ),
        layers=(wise.Layer("L"),),
        views=(wise.View("V", constraint_weights={"c": 1.0}),),
    )
    without = wise.score(log, censoring)
    assert without.manifest.observation.resolved_horizons is None, "no horizon is resolved when nothing captured one"
    with_capture = wise.score(log, censoring, evidence="summary")
    assert with_capture.manifest.observation.resolved_horizons == {"c": "2024-02-01T00:00:00"}


# ------------------------------------------------------------------- stages
def test_a_score_manifest_is_not_a_backlog_manifest(log, norm):
    result = wise.score(log, norm)
    manifest = result.manifest
    assert manifest.stage == "score"
    assert manifest.priority is None
    assert manifest.is_complete_backlog_run is False

    backlog = wise.prioritize(result, "company", view="Finance", gamma=1.0)
    final = manifest.finalize(
        grouping=["company"],
        view="Finance",
        gamma=1.0,
        comparator="current_population_mean",
        comparator_value=float(backlog.attrs["baseline"]),
        min_cases=1,
        volume="cases",
    )
    assert final.stage == "backlog" and final.is_complete_backlog_run
    assert final.priority.grouping == ("company",) and final.priority.gamma == 1.0
    assert final.run_id == manifest.run_id
    assert manifest.stage == "score", "finalising returns a new record and leaves the score record alone"
    assert final.config_fingerprint() != manifest.config_fingerprint()


def test_a_backlog_stage_without_a_priority_configuration_is_refused(log, norm):
    manifest = wise.score(log, norm).manifest
    with pytest.raises(EvidenceError, match="needs its priority configuration"):
        RunManifest(**{**{f: getattr(manifest, f) for f in RunManifest.__dataclass_fields__}, "stage": "backlog"})


def test_an_unknown_stage_or_mode_source_is_refused(log, norm):
    fields = {f: getattr(wise.score(log, norm).manifest, f) for f in RunManifest.__dataclass_fields__}
    with pytest.raises(EvidenceError, match="stage must be"):
        RunManifest(**{**fields, "stage": "guessed"})
    with pytest.raises(EvidenceError, match="mode_source"):
        RunManifest(**{**fields, "mode_source": "probably_the_norm"})


# --------------------------------------------------------------------- JSON
def test_manifest_json_is_utf8_explicit_and_finite(log, norm):
    manifest = wise.score(log, norm, evidence="summary").manifest
    text = manifest.to_json()
    payload = json.loads(text)
    assert payload["mode"] == "flat" and payload["mode_source"] == "norm_default"
    assert payload["input"]["data_digest"] is None
    assert payload["environment"]["git_commit"] is None
    assert payload["is_complete_backlog_run"] is False
    assert "NaN" not in text and "Infinity" not in text
    pd.Timestamp(payload["created_at"])  # a normalised, parseable timestamp


def test_json_refuses_a_non_finite_number_rather_than_writing_nan(log, norm):
    manifest = wise.score(log, norm).manifest
    broken = RunManifest(
        **{**{f: getattr(manifest, f) for f in RunManifest.__dataclass_fields__}, "preprocessing": {"q": float("nan")}}
    )
    with pytest.raises(EvidenceError, match="non-finite"):
        broken.to_json()


def test_json_refuses_an_unsupported_type_rather_than_stringifying_it(log, norm):
    manifest = wise.score(log, norm).manifest
    broken = RunManifest(
        **{**{f: getattr(manifest, f) for f in RunManifest.__dataclass_fields__}, "preprocessing": {"x": object()}}
    )
    with pytest.raises(EvidenceError, match="no conversion policy"):
        broken.to_json()


# ------------------------------------------------- the object run's own unit type
def object_run():
    """The running example as an object log, scored natively (5 ``case_review`` units)."""
    from _oc_fixtures import case_shaped, case_shaped_spec, legacy_object_norm

    from wise import oc

    oc_log = case_shaped()
    spec = case_shaped_spec()
    units = oc.build_units(oc_log, spec)
    return oc.score_units(oc_log, legacy_object_norm("flat"), units, spec=spec)


def test_an_object_run_records_its_own_unit_type():
    """An object run joins the run-record contract instead of being refused by it.

    ``RunManifest.unit_type`` was ``"case"`` by construction, and both
    :func:`wise.evidence.capture_evidence` and :func:`wise.explain_priority`
    refused an :class:`~wise.oc.evaluation.OCScoreResult` by type — so the one
    kind of run that has a unit type other than ``case`` was the one kind that
    could not say so.
    """
    from wise.evidence import capture_evidence

    result = object_run()
    manifest = result.manifest
    assert manifest is not None
    assert manifest.unit_type == "case_review"
    assert manifest.stage == "score" and manifest.priority is None
    assert manifest.run_id == result.run_id
    assert manifest.norm_fingerprint == result.norm.fingerprint()
    assert manifest.views == ("Finance", "Logistics")
    assert manifest.preprocessing["input_model"] == "object_centric"
    assert manifest.preprocessing["n_units"] == 5
    assert manifest.input.n_cases is None, "an object log has no case table to count"
    assert manifest.input.n_events == 21
    assert manifest.input.n_objects == 5 + 8, "five case objects and the eight goods receipts"
    assert json.loads(manifest.to_json())["unit_type"] == "case_review"

    packet = capture_evidence(result)
    assert packet.unit_type == "case_review"
    assert packet.manifest.unit_type == "case_review"
    assert packet.coverage.unit_type == "case_review"
    assert {r.unit_type for r in packet.records} == {"case_review"}
    assert packet.run_id == result.run_id

    explanation = wise.explain_priority(result, "n_bound", 1, view="Finance", evidence=packet)
    assert explanation.unit_type == "case_review"
    assert explanation.run_id == result.run_id
    assert explanation.priority.n_units == 4
    assert explanation.run["unit_type"] == "case_review"

    # the control: a case run still says exactly what it always said
    case_manifest = wise.score(wise.datasets.running_p2p_log(), wise.datasets.running_p2p_norm()).manifest
    assert case_manifest.unit_type == "case"
    assert case_manifest.input.n_cases == 5 and case_manifest.input.n_objects is None
