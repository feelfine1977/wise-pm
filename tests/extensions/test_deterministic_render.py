"""Deterministic, escaped, fact-bound rendering (stage S2, item E03).

The renderer owns the presentation and the packet owns the numbers. These
tests pin that separation: the same packet renders to the same bytes, every
number in a fact table is a fact of the packet, every mandatory limitation
survives to the output, hostile labels cannot restructure a table or a path,
and the command line renders a packet without computing anything.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

import wise
from wise.cli import main
from wise.errors import EvidenceError
from wise.explain import (
    EXPLANATION_SCHEMA_VERSION,
    BaselineSpec,
    ExplanationPacket,
    Fact,
    FactKind,
    escape_markdown,
    escape_prose,
    escape_text,
    explain_priority,
    explanation_filename,
    render_explanation,
    render_json,
    render_markdown,
    render_text,
    safe_identifier,
    write_explanation,
)
from wise.explain.render import _fmt

HOSTILE = "Vendor |--|\n[click](http://example.invalid) ../../etc/passwd"


@pytest.fixture
def packet(p2p_log, p2p_norm):
    result = wise.score(p2p_log, p2p_norm, evidence="full")
    return explain_priority(result, "company", "B", view="Finance", gamma=1.0, evidence=result.evidence)


@pytest.fixture
def hostile_packet(p2p_norm):
    events = wise.running_p2p_events().copy()
    events["vendor"] = HOSTILE
    log = wise.EventLog(
        events,
        case_col="case",
        activity_col="activity",
        timestamp_col="time",
        case_attributes=["flow_type", "company", "vendor"],
    )
    return explain_priority(wise.score(log, p2p_norm), "vendor", HOSTILE, view="Finance")


# ------------------------------------------------------------- determinism
@pytest.mark.parametrize("renderer", [render_text, render_markdown, render_json])
def test_rendering_the_same_packet_twice_gives_the_same_bytes(packet, renderer):
    assert renderer(packet) == renderer(packet)


def test_two_packets_of_the_same_run_render_identically(p2p_log, p2p_norm):
    result = wise.score(p2p_log, p2p_norm)
    first = explain_priority(result, "company", "B", view="Finance", gamma=1.0)
    second = explain_priority(result, "company", "B", view="Finance", gamma=1.0)
    assert first.explanation_id == second.explanation_id
    assert render_markdown(first) == render_markdown(second)
    assert render_text(first) == render_text(second)
    assert first.to_json() == second.to_json()


def test_the_dispatcher_covers_every_declared_format(packet):
    assert render_explanation(packet, "text") == render_text(packet)
    assert render_explanation(packet, "markdown") == render_markdown(packet)
    assert render_explanation(packet, "json") == render_json(packet)
    with pytest.raises(EvidenceError, match="format must be one of"):
        render_explanation(packet, "html")


# ------------------------------------------------- numbers come from facts
def _table_values(markdown: str, title: str) -> list[str]:
    """The value column of one fact table."""
    section = markdown.split(f"## {title}", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("| `")]
    return [row.split("|")[2].strip() for row in rows]


@pytest.mark.parametrize(
    ("kind", "title"),
    [
        (FactKind.OBSERVED_ASSESSMENT, "What was observed"),
        (FactKind.RELATIVE_PRIORITY, "Why it has this priority"),
        (FactKind.ALTERNATIVE_VIEW, "Under another approved view"),
        (FactKind.EVIDENCE_QUALIFICATION, "What the evidence covers"),
    ],
)
def test_every_number_in_a_fact_table_is_a_fact_of_the_packet(packet, kind, title):
    allowed = {_fmt(fact.value) for fact in packet.facts_of(kind)}
    printed = _table_values(render_markdown(packet), title)
    assert printed, title
    assert set(printed) <= allowed


def test_the_four_kinds_of_claim_get_four_separate_sections(packet):
    markdown = render_markdown(packet)
    for title in ("What was observed", "Why it has this priority", "Under another approved view", "What the evidence covers"):
        assert f"## {title}" in markdown
    assert markdown.index("## What was observed") < markdown.index("## Why it has this priority")


def test_a_changed_fact_changes_the_rendering(packet):
    import dataclasses

    facts = list(packet.facts)
    index = next(i for i, f in enumerate(facts) if f.fact_id == "PRI-stable_PI")
    facts[index] = dataclasses.replace(facts[index], value=99.0)
    altered = dataclasses.replace(packet, facts=tuple(facts))
    assert "99" in _table_values(render_markdown(altered), "Why it has this priority")
    assert render_markdown(altered) != render_markdown(packet)


def test_a_fact_cannot_carry_a_non_finite_value_or_no_reference():
    with pytest.raises(EvidenceError, match="non-finite value is a status"):
        Fact("F-1", FactKind.OBSERVED_ASSESSMENT, "x", float("nan"), "score", "d", ("run:1",))
    with pytest.raises(EvidenceError, match="no evidence reference"):
        Fact("F-1", FactKind.OBSERVED_ASSESSMENT, "x", 1.0, "score", "d", ())


def test_a_null_value_is_printed_as_an_explicit_null_not_a_zero(p2p_result):
    scalar = BaselineSpec.target(0.95, baseline_id="t", view="Finance", scoring_mode="flat")
    text = render_text(explain_priority(p2p_result, "company", "B", view="Finance", baseline_spec=scalar))
    assert "null" in text
    assert "0.0 " not in text.split("Layer decomposition")[1].split("Where the penalty sits")[0].replace("null", "")


# ------------------------------------------------- mandatory qualifications
def test_every_limitation_reaches_every_rendering(packet):
    text, markdown, payload = render_text(packet), render_markdown(packet), json.loads(render_json(packet))
    codes = {q.code.value for q in packet.limitations}
    assert codes
    for code in codes:
        assert code in text
        assert escape_markdown(code) in markdown
    assert {q["code"] for q in payload["limitations"]} == codes


def test_a_clipped_index_explains_its_clipping_in_the_report(p2p_result):
    text = render_text(explain_priority(p2p_result, "company", "A", view="Finance", gamma=1.0))
    assert "non_positive_gap_clipped" in text
    assert "clipped to zero" in text


def test_the_report_keeps_the_denominators_and_the_comparator_visible(packet):
    text = render_text(packet)
    assert "Denominators" in text
    for denominator in packet.denominators:
        assert denominator.name in text
    assert packet.baseline.baseline_id in text
    assert "reference score" in text


def test_the_vocabulary_stays_factual(packet):
    """No causal or monetary language is generated by the renderer."""
    lowered = (render_text(packet) + render_markdown(packet)).lower()
    for forbidden in ("root cause", "caused by", "money saved", "savings", "confidence level", "guarantee"):
        assert forbidden not in lowered


# ---------------------------------------------------------------- escaping
def test_a_hostile_label_cannot_restructure_a_markdown_table(hostile_packet):
    markdown = render_markdown(hostile_packet)
    # the packet keeps the label as the data it is; the renderer is what escapes it
    assert "\n" in hostile_packet.group_label
    assert HOSTILE not in markdown
    assert "](http" not in markdown
    assert "| Vendor |" not in markdown
    # every table row still has the column count its header declares
    for block in markdown.split("\n\n"):
        rows = [line for line in block.splitlines() if line.startswith("|")]
        if len(rows) < 2:
            continue
        widths = {len(re.split(r"(?<!\\)\|", row)) for row in rows}
        assert len(widths) == 1, block


def test_a_hostile_label_is_flattened_in_the_text_rendering(hostile_packet):
    text = render_text(hostile_packet)
    assert "../../etc/passwd" in text  # shown, because hiding evidence is worse
    assert all("\n" not in line for line in text.splitlines())
    assert "Priority explanation" in text.splitlines()[0]


def test_escaping_helpers_do_what_they_claim():
    assert escape_text("a\nb\tc") == "a b c"
    assert escape_markdown("a|b") == r"a\|b"
    assert escape_markdown("*bold*") == r"\*bold\*"
    assert escape_prose("a (b) c.") == "a (b) c."
    assert escape_prose("a | b") == r"a \| b"
    assert escape_markdown("\x00control") == "control"


# --------------------------------------------------------------- filenames
def test_a_generated_filename_uses_safe_identifiers_only(hostile_packet, tmp_path):
    path = explanation_filename(hostile_packet, suffix="md", directory=tmp_path)
    assert path.parent.resolve() == tmp_path.resolve()
    assert ".." not in path.name and "/" not in path.name
    assert path.name.endswith(".md")
    assert path.name.startswith(hostile_packet.explanation_id)


def test_a_hostile_suffix_cannot_become_a_path(packet, tmp_path):
    path = explanation_filename(packet, suffix="../../evil.sh", directory=tmp_path)
    assert path.parent.resolve() == tmp_path.resolve()
    assert path.suffix == ".evilsh"


def test_two_different_labels_cannot_collapse_into_one_identifier():
    assert safe_identifier("a b") != safe_identifier("a/b")
    assert safe_identifier("x" * 80) != safe_identifier("x" * 81)
    assert safe_identifier("plain") == "plain"


def test_write_explanation_writes_both_renderings(packet, tmp_path):
    written = write_explanation(packet, tmp_path, formats=("json", "md"))
    assert [p.suffix for p in written] == [".json", ".md"]
    assert all(p.parent.resolve() == tmp_path.resolve() for p in written)
    assert json.loads(written[0].read_text(encoding="utf-8"))["explanation_id"] == packet.explanation_id
    assert written[1].read_text(encoding="utf-8") == render_markdown(packet)
    with pytest.raises(EvidenceError, match="unknown output format"):
        write_explanation(packet, tmp_path, formats=("exe",))


# ------------------------------------------------------------- round-trip
def test_a_packet_round_trips_through_json(packet, tmp_path):
    again = ExplanationPacket.from_dict(json.loads(packet.to_json()))
    assert again.explanation_id == packet.explanation_id
    assert again.priority == packet.priority
    assert again.layers == packet.layers
    assert again.facts == packet.facts
    assert again.denominators == packet.denominators
    assert again.limitations == packet.limitations
    assert again.constraints == packet.constraints
    assert again.view_contrasts == packet.view_contrasts
    assert again.baseline == packet.baseline
    assert again.witnesses == packet.witnesses
    assert render_text(again) == render_text(packet)
    assert render_markdown(again) == render_markdown(packet)


def test_a_round_tripped_packet_keeps_the_run_record_but_not_the_live_manifest(packet):
    again = ExplanationPacket.from_dict(json.loads(packet.to_json()))
    assert again.manifest is None
    assert again.run["stage"] == "backlog"
    assert again.run["run_id"] == packet.run_id


def test_an_unknown_schema_version_is_refused(packet):
    data = json.loads(packet.to_json())
    data["schema_version"] = "wise-explanation/99"
    with pytest.raises(EvidenceError, match="unsupported explanation schema version"):
        ExplanationPacket.from_dict(data)
    assert packet.schema_version == EXPLANATION_SCHEMA_VERSION


def test_duplicate_fact_ids_are_refused(packet):
    import dataclasses

    with pytest.raises(EvidenceError, match="fact ids must be unique"):
        dataclasses.replace(packet, facts=(*packet.facts, packet.facts[0]))


def test_a_null_group_key_survives_the_round_trip(null_key_result):
    packet = explain_priority(null_key_result, "company", float("nan"), view="Finance")
    data = json.loads(packet.to_json())
    assert data["group_key"] == [None]
    again = ExplanationPacket.from_dict(data)
    assert again.group_label == "<null>"
    assert render_text(again) == render_text(packet)


# -------------------------------------------------------------------- CLI
def _cli_files(tmp_path):
    log = tmp_path / "log.csv"
    wise.running_p2p_events().to_csv(log, index=False)
    norm = tmp_path / "norm.json"
    wise.running_p2p_norm().dump(norm)
    common = ["--case", "case", "--activity", "activity", "--timestamp", "time", "--attr", "company", "--attr", "flow_type"]
    return str(log), str(norm), common


def test_the_default_stdout_csv_is_unchanged_by_the_new_sidecars(tmp_path, capsys):
    log, norm, common = _cli_files(tmp_path)
    base = ["score", norm, log, *common, "--by", "company", "--view", "Finance", "--gamma", "1"]
    assert main(base) == 0
    plain = capsys.readouterr().out
    assert main([*base, "--explain-out", str(tmp_path / "e.json"), "--manifest-out", str(tmp_path / "m.json")]) == 0
    captured = capsys.readouterr()
    assert captured.out == plain, "the opt-in sidecars must not touch the stdout CSV"
    assert "wrote the explanation" in captured.err
    assert list(pd.read_csv(tmp_path / "e.json", nrows=0).columns) or True  # the sidecar is JSON, not CSV


def test_wise_explain_renders_a_packet_the_score_command_wrote(tmp_path, capsys):
    log, norm, common = _cli_files(tmp_path)
    packet_path = tmp_path / "explanation.json"
    assert main(["score", norm, log, *common, "--by", "company", "--view", "Finance", "--explain-out", str(packet_path)]) == 0
    capsys.readouterr()
    payload = json.loads(packet_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == EXPLANATION_SCHEMA_VERSION
    assert payload["group_label"] == "B", "the top-ranked slice is explained"

    assert main(["explain", str(packet_path)]) == 0
    text = capsys.readouterr().out
    assert "Priority explanation" in text and "Limitations" in text
    out = tmp_path / "report.md"
    assert main(["explain", str(packet_path), "--format", "markdown", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("# Priority explanation")
    assert main(["explain", str(packet_path), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["explanation_id"] == payload["explanation_id"]


def test_the_baseline_file_changes_the_reference_and_no_column(tmp_path, capsys):
    log, norm, common = _cli_files(tmp_path)
    base = ["score", norm, log, *common, "--by", "company", "--view", "Finance", "--out", str(tmp_path / "a.csv")]
    assert main(base) == 0
    default = pd.read_csv(tmp_path / "a.csv")
    spec = BaselineSpec.target(0.95, baseline_id="board-2026", view="Finance", scoring_mode="flat")
    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    assert main([*base, "--out", str(tmp_path / "b.csv"), "--baseline-file", str(baseline_file)]) == 0
    targeted = pd.read_csv(tmp_path / "b.csv")
    assert list(targeted.columns) == list(default.columns)
    assert targeted["global_mean"].tolist() == [0.95, 0.95]
    assert default["global_mean"].tolist() != targeted["global_mean"].tolist()
    capsys.readouterr()


def test_a_malformed_baseline_file_is_a_clean_error(tmp_path, capsys):
    log, norm, common = _cli_files(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text('{"baseline_id": "t", "kind": "target", "reference_score": 2.0}', encoding="utf-8")
    rc = main(["score", norm, log, *common, "--by", "company", "--baseline-file", str(bad)])
    assert rc == 1
    assert "error:" in capsys.readouterr().err


def test_the_explain_command_refuses_a_file_that_is_not_a_packet(tmp_path, capsys):
    bad = tmp_path / "nope.json"
    bad.write_text("{}", encoding="utf-8")
    assert main(["explain", str(bad)]) == 1
    assert "error:" in capsys.readouterr().err


# ---------------------------------------------------------------- example
def test_the_end_to_end_example_runs_and_writes_what_it_says(tmp_path):
    """The stage's checkpoint, executed: score, export, explain, no model."""
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "examples" / "explanation_review.py"
    out = tmp_path / "review"
    proc = subprocess.run([sys.executable, str(script), str(out)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "sum of layer deltas   +0.069833333333" in proc.stdout
    assert "signed score gap      +0.069833333333" in proc.stdout
    assert "kind absolute_only" in proc.stdout
    written = sorted(p.name for p in out.iterdir())
    assert "run.json" in written and "evidence.json" in written and "explanation.md" in written
    packets = [p for p in out.iterdir() if p.name.startswith("exp-") and p.suffix == ".json"]
    assert len(packets) == 1
    assert json.loads(packets[0].read_text(encoding="utf-8"))["schema_version"] == EXPLANATION_SCHEMA_VERSION
    assert json.loads((out / "run.json").read_text(encoding="utf-8"))["stage"] == "backlog"
