"""Error hierarchy (``wise.errors``) and command-line interface (``wise.cli``).

Regression tests pin the desired behaviour for defects C9, C11 and C18 of the
findings ledger and are ``xfail(strict=True)`` until the fix lands. The plain
tests characterise behaviour that is already correct and must stay.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys

import pytest

import wise
import wise.errors
from wise.cli import main
from wise.errors import LogSchemaError, NormError, NotScoredError, WiseError

_PUBLIC_EXCEPTIONS = [
    cls
    for name, cls in inspect.getmembers(wise.errors, inspect.isclass)
    if issubclass(cls, BaseException) and not name.startswith("_") and cls.__module__ == wise.errors.__name__
]
_LOG_ARGS = ["--case", "case", "--activity", "activity", "--timestamp", "time", "--attr", "company", "--attr", "flow_type"]
_MALFORMED_NORMS = {
    "empty_mapping": {},
    "no_constraints": {"name": "broken", "constraints": []},
    "unknown_constraint_type": {"name": "broken", "constraints": [{"id": "c1", "layer": "L1", "type": "nope"}]},
    "not_a_mapping": [1, 2],
}


def _builtin_bases(cls: type) -> list[type]:
    """Builtin classes in ``cls.__mro__`` other than the universal ``Exception`` chain."""
    return [b for b in cls.__mro__ if b.__module__ == "builtins" and b not in (Exception, BaseException, object)]


def _write_norm(tmp_path, payload=None):
    path = tmp_path / "norm.json"
    if payload is None:
        wise.running_p2p_norm().dump(path)
    else:
        path.write_text(json.dumps(payload))
    return path


def _write_p2p_files(tmp_path):
    log = tmp_path / "log.csv"
    wise.running_p2p_events().to_csv(log, index=False)
    return log, _write_norm(tmp_path)


# --- regression: error hierarchy (C11 / C9) --------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="C11: LogSchemaError subclasses KeyError, so any `except KeyError:` swallows schema errors",
)
def test_log_schema_error_is_not_a_key_error():
    assert not issubclass(wise.LogSchemaError, KeyError)


@pytest.mark.regression
@pytest.mark.parametrize("cls", [NormError, NotScoredError, LogSchemaError])
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="C11/C9: WiseError subclasses also inherit builtin ValueError/KeyError, leaking out of the hierarchy",
)
def test_wise_error_subclass_inherits_no_builtin_besides_exception(cls):
    assert _builtin_bases(cls) == []


# --- regression: CLI (C18) ---------------------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="C18: main() calls sys.stdout/sys.stderr.reconfigure(encoding='utf-8') for the whole process",
)
def test_main_leaves_process_stream_encoding_unchanged(tmp_path):
    norm = _write_norm(tmp_path)
    code = (
        "import sys, wise.cli as c; before = sys.stdout.encoding; "
        f"c.main(['validate', {str(norm)!r}]); print(before == sys.stdout.encoding)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONIOENCODING": "latin-1"},
    )
    assert proc.stdout.splitlines()[-1] == "True"


# --- characterisation: error hierarchy ---------------------------------------------------------------------------


def test_errors_module_exposes_the_documented_exceptions():
    names = {cls.__name__ for cls in _PUBLIC_EXCEPTIONS}
    assert {"WiseError", "NormError", "LogSchemaError", "NotScoredError"} <= names


@pytest.mark.parametrize("cls", _PUBLIC_EXCEPTIONS, ids=lambda cls: cls.__name__)
def test_public_exception_derives_from_wise_error(cls):
    assert issubclass(cls, WiseError)


def test_log_schema_error_message_is_not_quoted():
    assert str(LogSchemaError("missing column 'case'")) == "missing column 'case'"


# --- characterisation: CLI ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("payload", list(_MALFORMED_NORMS.values()), ids=list(_MALFORMED_NORMS))
def test_validate_on_malformed_norm_returns_1_and_reports_error_on_stderr(tmp_path, capsys, payload):
    norm = _write_norm(tmp_path, payload)
    assert main(["validate", str(norm)]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error:")
    assert "OK" not in captured.out


def test_version_flag_prints_package_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"wise {wise.__version__}"


@pytest.mark.parametrize("by", [["company"], ["company", "flow_type"]], ids=["one_key", "two_keys"])
def test_score_writes_backlog_csv_with_slice_keys_as_leading_columns(tmp_path, by):
    log, norm = _write_p2p_files(tmp_path)
    out = tmp_path / "backlog.csv"
    by_args = [arg for key in by for arg in ("--by", key)]
    assert main(["score", str(norm), str(log), *_LOG_ARGS, *by_args, "--out", str(out)]) == 0
    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith(",".join(by) + ",")


def test_score_to_stdout_starts_with_slice_key_header(tmp_path, capsys):
    log, norm = _write_p2p_files(tmp_path)
    assert main(["score", str(norm), str(log), *_LOG_ARGS, "--by", "company"]) == 0
    first_line = capsys.readouterr().out.splitlines()[0]
    assert first_line.startswith("company,")
