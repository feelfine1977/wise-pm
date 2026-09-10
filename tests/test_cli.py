"""Command-line interface."""

import contextlib
import csv
import io
import os

import pandas as pd
import pytest

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


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["posix", "windows"])
@pytest.mark.parametrize("destination", ["stdout", "file"])
@pytest.mark.parametrize("company", ["B", 'B,\n"branch"'], ids=["plain", "quoted-multiline"])
def test_score_csv_has_no_double_translation(tmp_path, monkeypatch, newline, destination, company):
    events = wise.running_p2p_events()
    events.loc[events["company"] == "B", "company"] = company
    log, norm = _files(tmp_path)
    events.to_csv(log, index=False, lineterminator="\n")
    output = tmp_path / "backlog.csv"
    original_open = open

    def output_stream():
        return io.TextIOWrapper(original_open(output, "wb"), encoding="utf-8", newline=newline)

    def translated_open(path, mode="r", *args, **kwargs):
        if path == str(output) and mode == "w":
            return output_stream()
        return original_open(path, mode, *args, **kwargs)

    args = [
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
    ]
    # Exercise the real CLI with both Windows newline defaults: pandas' CSV
    # terminator and a translating TextIOWrapper. No CSV/scoring function is mocked.
    with monkeypatch.context() as patch:
        patch.setattr(os, "linesep", newline)
        if destination == "stdout":
            with output_stream() as stream, contextlib.redirect_stdout(stream):
                assert main(args) == 0
        else:
            patch.setattr("builtins.open", translated_open)
            assert main([*args, "--out", str(output)]) == 0

    payload = output.read_bytes()
    assert b"\r\r\n" not in payload
    # newline='' preserves the bytes' newline forms inside quoted fields;
    # csv.reader must see exactly a header and two records, without filtering blanks.
    rows = list(csv.reader(io.StringIO(payload.decode("utf-8"), newline="")))
    assert len(rows) == 3
    assert rows[0] == [
        "company",
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
    assert rows[1][:2] == [company.replace("\n", newline), "3"]
    assert rows[2][:2] == ["A", "2"]
    assert all(len(row) == len(rows[0]) for row in rows)
