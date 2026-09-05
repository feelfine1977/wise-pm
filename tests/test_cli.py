"""Command-line interface."""

import pandas as pd

import wise
from wise.cli import main


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
    assert b.loc["B", "PI"] > 0 and len(pd.read_csv(cases)) == 5
    assert main(["score", str(norm), str(log), *common, "--by", "company"]) == 0
    assert "stable_PI" in capsys.readouterr().out
    assert (
        main(["score", str(norm), str(log), "--case", "case", "--activity", "activity", "--timestamp", "time", "--by", "company"])
        == 1
    )
    assert "error:" in capsys.readouterr().err
