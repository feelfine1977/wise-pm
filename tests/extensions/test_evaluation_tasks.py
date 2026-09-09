"""The offline boundary harness (L04): what it covers, and that it can fail.

The harness scores the boundary, not a model, so these tests are about the
harness's own honesty:

* the six refusals L3 established and L4 repaired are each a scenario, and each
  runs against the real gateway, policy, draft reader and evidence packet;
* a negative control is served, so a harness that refused everything would fail
  here rather than look perfect;
* a declared outcome that is wrong is reported as ``not_as_declared`` — the
  harness is capable of red;
* a scenario needing a model is skipped with its environment variable named,
  and cannot be made to run by setting the variable alone;
* the whole run opens no socket, which is asserted in a subprocess rather than
  promised in a docstring.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from wise.errors import LLMError
from wise.evaluation.llm import (
    HANDLERS,
    BoundaryMaterial,
    HarnessReport,
    Outcome,
    RecordedTask,
    Status,
    _markers,
    load_tasks,
    run_tasks,
)

TASKS = Path(__file__).resolve().parents[1] / "fixtures" / "llm_tasks"

#: The refusals the brief requires the harness to cover, and the scenario each
#: of them is. "A stale fingerprint" exists in two places and both are covered.
REQUIRED_FAMILIES = {
    "unauthorised_population",
    "invented_reference",
    "unknown_tool",
    "poisoned_document",
    "oversize_reply",
    "stale_norm_fingerprint",
    "stale_snapshot_fingerprint",
}


@pytest.fixture(scope="module")
def material():
    return BoundaryMaterial.running_example()


@pytest.fixture(scope="module")
def tasks():
    return load_tasks(TASKS)


@pytest.fixture(scope="module")
def report(tasks, material):
    return run_tasks(tasks, material=material, env={})


# ------------------------------------------------------------------- coverage
def test_the_shipped_scenarios_cover_every_refusal_the_brief_names(tasks):
    families = {task.family for task in tasks}
    assert families >= REQUIRED_FAMILIES
    assert families <= set(HANDLERS), "a scenario must name a family the runner can dispatch"
    assert "authorised_population" in families, "the negative control is part of the coverage, not an extra"


def test_every_scenario_behaves_as_the_library_declares(report):
    assert report.ok, "\n".join(f"{o.task_id}: {o.detail or o.happened}" for o in report.failures())
    counts = report.counts()
    assert counts == {"as_declared": 9, "not_as_declared": 0, "skipped": 1}


def test_each_line_says_what_was_asked_what_happened_and_whether_that_is_declared(report):
    text = report.render()
    for outcome in report.outcomes:
        assert outcome.ask and outcome.ask in text
        if outcome.status == Status.SKIPPED.value:
            assert outcome.skip_reason in text
        else:
            assert outcome.happened and outcome.happened in text
            assert outcome.observed in (o.value for o in Outcome)
    assert "9 as declared, 0 not as declared, 1 skipped" in text


def test_the_negative_control_is_actually_served(report):
    """A harness that refuses everything passes every refusal test."""
    served = report.of("03_authorised_population")
    assert served.observed == Outcome.SERVED.value and served.as_declared
    assert "43 facts" in served.happened


def test_the_refusals_are_the_ones_l3_established_and_l4_repaired(report):
    assert report.of("01_unauthorised_population").observed == Outcome.REFUSED.value
    assert "authorised for no unit" in report.of("01_unauthorised_population").happened
    assert report.of("02_relabelled_comparator").observed == Outcome.SERVED_RELABELLED.value
    assert "withheld" in report.of("02_relabelled_comparator").happened
    assert report.of("04_unknown_tool").observed == Outcome.REFUSED.value
    assert report.of("05_invented_fact_id").observed == Outcome.REFUSED.value
    assert report.of("06_poisoned_document").observed == Outcome.CONTAINED.value
    assert report.of("07_oversize_reply").observed == Outcome.DETERMINISTIC_ONLY.value
    assert report.of("08_stale_norm_fingerprint").observed == Outcome.REFUSED.value
    assert report.of("09_stale_snapshot_fingerprint").observed == Outcome.REFUSED.value


def test_the_same_material_gives_the_same_report(tasks, material):
    first = run_tasks(tasks, material=material, env={})
    second = run_tasks(tasks, material=material, env={})
    assert first.to_dict() == second.to_dict()


def test_the_stale_snapshot_scenario_leaves_the_shared_material_alone(tasks, material, report):
    """It tampers with a log; it must not be the one the other scenarios read."""
    assert material.log.events.loc[0, "activity"] != "Tampered By The Harness"
    again = run_tasks(tasks, material=material, env={})
    assert again.of("09_stale_snapshot_fingerprint").as_declared


# --------------------------------------------------------------- it can fail
def test_a_wrong_declared_outcome_is_reported_red(tasks, material):
    """The harness is capable of red: this is that capability, asserted."""
    wrong = [dataclasses.replace(t, expected_outcome=Outcome.SERVED.value) for t in tasks if t.task_id.startswith("01_")]
    broken = run_tasks(wrong, material=material, env={})
    assert not broken.ok
    failure = broken.of("01_unauthorised_population")
    assert failure.status == Status.NOT_AS_DECLARED.value
    assert failure.expected == "served" and failure.observed == "refused"
    assert broken.counts()["not_as_declared"] == 1


def test_a_leaked_forbidden_marker_fails_even_when_the_outcome_matches(tasks, material):
    """The population mean must not travel, whatever the outcome is called."""
    leaking = [
        dataclasses.replace(t, forbidden_markers=("0.7098333333333334",))
        for t in tasks
        if t.task_id.startswith("03_")  # the control, which legitimately carries it
    ]
    broken = run_tasks(leaking, material=material, env={})
    assert not broken.ok
    detail = broken.of("03_authorised_population").detail
    assert "forbidden marker(s) present" in detail and "0.7098333333333334" in detail


@pytest.mark.parametrize("value", ["0.7098333333333334", "0.7098333333333333", "7.098333333333333e-1"])
@pytest.mark.parametrize("template", ['{"baseline": %s}', '{"report": "baseline is %s"}'])
def test_numeric_markers_detect_both_required_and_forbidden_platform_representations(tasks, value, template):
    task = next(t for t in tasks if t.task_id.startswith("03_"))
    marker = "0.7098333333333334"
    transcript = template % value
    required = dataclasses.replace(task, expected_markers=(marker,), forbidden_markers=())
    assert _markers(required, transcript) == ""
    forbidden = dataclasses.replace(task, expected_markers=(), forbidden_markers=(marker,))
    assert "forbidden marker(s) present" in _markers(forbidden, transcript)


@pytest.mark.parametrize("value", ["0.64", "0.7098333333", "10.7098333333333334", "id0.7098333333333334", "null", "NaN"])
def test_a_missing_or_different_number_cannot_satisfy_a_numeric_marker(tasks, value):
    task = next(t for t in tasks if t.task_id.startswith("03_"))
    required = dataclasses.replace(task, expected_markers=("0.7098333333333334",), forbidden_markers=())
    assert "expected marker(s) absent" in _markers(required, value)


@pytest.mark.parametrize("transcript", ["id0.7098333333333334", "10.7098333333333334"])
def test_literal_forbidden_markers_remain_detectable_even_inside_larger_tokens(tasks, transcript):
    task = next(t for t in tasks if t.task_id.startswith("03_"))
    forbidden = dataclasses.replace(task, expected_markers=(), forbidden_markers=("0.7098333333333334",))
    assert "forbidden marker(s) present" in _markers(forbidden, transcript)


def test_a_missing_expected_marker_fails_too(tasks, material):
    demanding = [
        dataclasses.replace(t, expected_markers=("a sentence no refusal contains",)) for t in tasks if t.task_id.startswith("04_")
    ]
    broken = run_tasks(demanding, material=material, env={})
    assert not broken.ok
    assert "expected marker(s) absent" in broken.of("04_unknown_tool").detail


# ------------------------------------------------------------- the model gate
def test_a_scenario_needing_a_model_is_skipped_with_its_variable_named(report):
    skipped = report.skipped()
    assert [s.task_id for s in skipped] == ["10_live_model_quality"]
    assert "ALLOW_LIVE_LLM_TESTS" in skipped[0].skip_reason
    assert skipped[0].observed == "", "a skipped scenario observed nothing"
    assert report.ok, "a skip is neither a pass nor a failure, and does not make the run red"


def test_setting_the_variable_alone_still_does_not_reach_a_model(tasks, material):
    """The variable enables the scenario; only the caller can supply a provider."""
    enabled = run_tasks(tasks, material=material, env={"ALLOW_LIVE_LLM_TESTS": "1"})
    reason = enabled.of("10_live_model_quality").skip_reason
    assert enabled.of("10_live_model_quality").status == Status.SKIPPED.value
    assert "ALLOW_LIVE_LLM_TESTS" in reason and "live_provider" in reason


def test_the_gate_is_read_from_the_supplied_environment_not_the_process(tasks, material):
    for value in ("", "0", "false"):
        run = run_tasks(tasks, material=material, env={"ALLOW_LIVE_LLM_TESTS": value})
        assert run.of("10_live_model_quality").status == Status.SKIPPED.value


# ------------------------------------------------------------- strict loading
def test_a_task_file_is_parsed_strictly(tmp_path):
    good = json.loads((TASKS / "04_unknown_tool.json").read_text(encoding="utf-8"))

    (tmp_path / "a.json").write_text(json.dumps({**good, "surprise": 1}), encoding="utf-8")
    with pytest.raises(LLMError, match="unknown key"):
        load_tasks(tmp_path)

    (tmp_path / "a.json").write_text(json.dumps({**good, "family": "make_it_up"}), encoding="utf-8")
    with pytest.raises(LLMError, match="unknown family"):
        load_tasks(tmp_path)

    (tmp_path / "a.json").write_text(json.dumps({**good, "expected_outcome": "fine"}), encoding="utf-8")
    with pytest.raises(ValueError, match="fine"):
        load_tasks(tmp_path)

    (tmp_path / "a.json").write_text(json.dumps({**good, "schema_version": "wise-llm-task/9"}), encoding="utf-8")
    with pytest.raises(LLMError, match="is not 'wise-llm-task/1'"):
        load_tasks(tmp_path)

    (tmp_path / "a.json").write_text("not json at all", encoding="utf-8")
    with pytest.raises(LLMError, match="not a JSON task"):
        load_tasks(tmp_path)


def test_an_empty_or_missing_directory_is_refused(tmp_path):
    with pytest.raises(LLMError, match="holds no"):
        load_tasks(tmp_path)
    with pytest.raises(LLMError, match="not a directory"):
        load_tasks(tmp_path / "nowhere")


def test_the_shipped_tasks_round_trip_through_their_own_dictionary(tasks):
    for task in tasks:
        assert RecordedTask.from_dict(task.to_dict()) == task
        assert task.notes, "a scenario states why it exists"


def test_an_unknown_policy_name_is_refused(material):
    with pytest.raises(LLMError, match="unknown harness policy"):
        material.policy("root")


# ------------------------------------------------------------------- offline
def test_the_whole_harness_runs_without_opening_a_socket():
    """Offline by construction, asserted in a subprocess rather than promised."""
    code = (
        "import socket, sys\n"
        "class NoSockets:\n"
        "    def __init__(self, *a, **k):\n"
        "        raise AssertionError('the harness opened a socket')\n"
        "socket.socket = NoSockets\n"
        "socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(AssertionError('the harness dialled out'))\n"
        "from wise.evaluation.llm import load_tasks, run_tasks\n"
        f"report = run_tasks(load_tasks({str(TASKS)!r}), env={{}})\n"
        "print(report.counts()['as_declared'], report.counts()['skipped'], report.ok)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "9 1 True"


def test_the_report_serialises_to_a_readable_record(report):
    payload = report.to_dict()
    assert payload["schema_version"] == "wise-llm-harness/1"
    assert payload["ok"] is True
    assert len(payload["scenarios"]) == len(report) == 10
    assert set(payload["families"]) == {t["family"] for t in payload["scenarios"]}
    table = report.table()
    assert list(table.index) == [o.task_id for o in report.outcomes]
    assert {"family", "expected", "observed", "status"} <= set(table.columns)
    assert isinstance(HarnessReport(), HarnessReport) and HarnessReport().ok
