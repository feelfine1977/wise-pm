"""Command-line interface."""

import json

import pandas as pd
import pytest

import wise
from wise.cli import main

# Schema version 2 documents are read here; each read announces the 0.1.0 semantics (LegacySemanticsWarning).
pytestmark = pytest.mark.filterwarnings("ignore::wise.LegacySemanticsWarning")


def _files(tmp_path):
    log = tmp_path / "log.csv"
    wise.running_p2p_events().to_csv(log, index=False)
    norm = tmp_path / "norm.json"
    wise.running_p2p_norm().dump(norm)
    return log, norm


def test_validate_and_describe(tmp_path, capsys):
    _, norm = _files(tmp_path)
    assert main(["validate", str(norm)]) == 0
    assert "OK" in capsys.readouterr().out
    assert main(["describe", str(norm)]) == 0
    assert "Raw weights" in capsys.readouterr().out


def test_check_and_score(tmp_path, capsys):
    log, norm = _files(tmp_path)
    common = ["--case", "case", "--activity", "activity", "--timestamp", "time", "--attr", "company", "--attr", "flow_type"]
    assert main(["check", str(norm), str(log), *common]) == 0
    assert "No issues" in capsys.readouterr().out
    out = tmp_path / "backlog.csv"
    cases = tmp_path / "cases.csv"
    rc = main(
        [
            "score",
            str(norm),
            str(log),
            *common,
            "--by",
            "company",
            "--view",
            "Finance",
            "--gamma",
            "1",
            "--out",
            str(out),
            "--cases-out",
            str(cases),
        ]
    )
    assert rc == 0
    b = pd.read_csv(out).set_index("company")
    assert b.loc["B", "PI"] > 0
    assert len(pd.read_csv(cases)) == 5
    assert main(["score", str(norm), str(log), *common, "--by", "company"]) == 0
    assert "stable_PI" in capsys.readouterr().out
    assert (
        main(["score", str(norm), str(log), "--case", "case", "--activity", "activity", "--timestamp", "time", "--by", "company"])
        == 1
    )
    assert "error:" in capsys.readouterr().err


BARE_LAG_NORM = {
    "schema_version": 2,
    "name": "bare",
    "views": [{"name": "v", "constraint_weights": {"l": 1}}],
    "constraints": [{"id": "l", "layer": "L", "type": "lag", "params": {"a": "A", "b": "B"}}],
}


@pytest.mark.spec("decision:0001/4")
def test_migrate_writes_the_migrated_norm_beside_its_input_and_prints_the_report(tmp_path, capsys):
    source = tmp_path / "norm.json"
    source.write_text(json.dumps(BARE_LAG_NORM), encoding="utf-8")
    before = source.read_bytes()
    assert main(["migrate", str(source)]) == 0
    captured = capsys.readouterr()
    target = tmp_path / "norm_v3.json"
    assert source.read_bytes() == before
    migrated = json.loads(target.read_text(encoding="utf-8"))
    assert migrated["schema_version"] == 3
    assert migrated["constraints"][0]["params"] == {"a": "A", "b": "B", "delta": 0, "width": 0}
    report = json.loads(captured.out)
    assert [r["rule"] for r in report["rewrites"]] == ["missing_values", "lag_bound"]
    assert report["rewrites"][1]["recommendation"].startswith("review the bound")
    assert captured.out == wise.migrate(BARE_LAG_NORM).report.to_json() + "\n"
    assert target.read_text(encoding="utf-8") == wise.migrate(BARE_LAG_NORM).dumps() + "\n"
    assert f"wrote the schema version 3 norm to {target}" in captured.err
    assert "LegacySemanticsWarning" not in captured.err


def test_migrate_writes_to_the_given_path(tmp_path, capsys):
    source = tmp_path / "norm.json"
    source.write_text(json.dumps(BARE_LAG_NORM), encoding="utf-8")
    target = tmp_path / "out" / "migrated.json"
    target.parent.mkdir()
    assert main(["migrate", str(source), "--out", str(target)]) == 0
    assert wise.Norm.load(target).schema_version == 3


@pytest.mark.spec("decision:0001/4")
def test_migrate_never_overwrites_its_input(tmp_path, capsys, monkeypatch):
    source = tmp_path / "norm.json"
    source.write_text(json.dumps(BARE_LAG_NORM), encoding="utf-8")
    before = source.read_bytes()
    link = tmp_path / "link.json"
    link.symlink_to(source)
    monkeypatch.chdir(tmp_path)
    hard = tmp_path / "hard.json"
    hard.hardlink_to(source)
    for out in (str(source), "norm.json", "./norm.json", str(link), str(hard)):
        assert main(["migrate", str(source), "--out", out]) == 1
        assert "error: migrate does not overwrite its input" in capsys.readouterr().err
    assert source.read_bytes() == before


def test_migrate_reports_a_refused_norm_and_writes_nothing(tmp_path, capsys):
    source = tmp_path / "norm.json"
    source.write_text(json.dumps({**BARE_LAG_NORM, "derived_attributes": [{"name": "x", "kind": "eval", "expr": "a + b"}]}))
    assert main(["migrate", str(source)]) == 1
    assert "error: cannot migrate norm 'bare' to schema version 3: derived attribute 'x' (eval)" in capsys.readouterr().err
    assert not (tmp_path / "norm_v3.json").exists()
    source.write_text(json.dumps({k: v for k, v in BARE_LAG_NORM.items() if k != "schema_version"}))
    assert main(["migrate", str(source)]) == 1
    assert "give one of [1, 2, 3]" in capsys.readouterr().err
