"""Why a case has no value for a constraint, kept on the score result and in the tables written from it.

The expectations are derived by hand from the three-case log ``reason_log``:
case ``in`` (``flow_type="DF1"``, A then B), case ``out`` (``flow_type="DF2"``,
A only) and case ``unknown`` (no ``flow_type``, A only). Constraint ``c``
(B present) applies to ``flow_type in ["DF1"]``; constraint ``d`` (A present)
applies always.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import wise
import wise.compat
from wise.cli import main
from wise.family import EVALUATED, OUT_OF_SCOPE, SCOPE_UNKNOWN

CASES = ["in", "out", "unknown"]


def reason_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": ["in", "in", "out", "unknown"],
            "activity": ["A", "B", "A", "A"],
            "time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-01"]),
            "flow_type": ["DF1", "DF1", "DF2", None],
        }
    )


@pytest.fixture
def reason_log() -> wise.EventLog:
    return wise.EventLog(
        reason_events(), case_col="case", activity_col="activity", timestamp_col="time", case_attributes=["flow_type"]
    )


def reason_document(version: int) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": version,
        "name": "reasons",
        "scoring_mode": "flat",
        "layers": [{"id": "L"}],
        "views": [{"name": "v", "constraint_weights": {"c": 1, "d": 1}}],
        "constraints": [
            {"id": "c", "layer": "L", "type": "presence", "params": {"activity": "B"}, "applicability": {"flow_type": ["DF1"]}},
            {"id": "d", "layer": "L", "type": "presence", "params": {"activity": "A"}},
        ],
    }
    if version == 3:
        document["missing_values"] = "three_valued"
    return document


def read(version: int) -> wise.Norm:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wise.LegacySemanticsWarning)
        return wise.Norm.from_dict(reason_document(version))


@pytest.mark.spec("decision:0001/3")
def test_the_reason_table_tells_unknown_scope_from_out_of_scope_under_schema_3(reason_log: wise.EventLog) -> None:
    result = wise.score(reason_log, read(3))
    assert result.reasons is not None
    assert result.reasons.index.equals(result.violations.index)
    assert list(result.reasons.columns) == ["c", "d"]
    assert result.reasons["c"].astype(str).tolist() == [EVALUATED, OUT_OF_SCOPE, SCOPE_UNKNOWN]
    assert result.reasons["d"].astype(str).tolist() == [EVALUATED, EVALUATED, EVALUATED]
    # the two cases without a value keep the same pair of verdict and scope as before
    assert result.violations["c"].tolist() == pytest.approx([0.0, np.nan, np.nan], nan_ok=True)
    assert result.in_scope["c"].tolist() == [True, False, False]


@pytest.mark.spec("decision:0001/4")
def test_under_schema_2_a_missing_attribute_is_out_of_scope(reason_log: wise.EventLog) -> None:
    """The 0.1.0 profile has no unknown scope: a membership rule leaves a missing attribute out of scope."""
    result = wise.score(reason_log, read(2))
    assert result.reasons is not None
    assert result.reasons["c"].astype(str).tolist() == [EVALUATED, OUT_OF_SCOPE, OUT_OF_SCOPE]


@pytest.mark.spec("decision:0001/3")
def test_the_verdict_table_carries_the_reason_beside_the_value(reason_log: wise.EventLog) -> None:
    table = wise.score(reason_log, read(3)).verdicts()
    assert table.index.names == ["case", "constraint"]
    assert list(table.columns) == ["value", "in_scope", "reason"]
    assert table.index.tolist() == [(case, cid) for case in CASES for cid in ("c", "d")]
    assert table["value"].tolist() == pytest.approx([0.0, 0.0, np.nan, 0.0, np.nan, 0.0], nan_ok=True)
    assert table["in_scope"].tolist() == [True, True, False, True, False, True]
    assert table["reason"].astype(str).tolist() == [
        EVALUATED,
        EVALUATED,
        OUT_OF_SCOPE,
        EVALUATED,
        SCOPE_UNKNOWN,
        EVALUATED,
    ]


@pytest.mark.spec("decision:0001/3")
def test_the_verdict_table_written_by_the_command_line_keeps_the_reasons(tmp_path: Path) -> None:
    log = tmp_path / "log.csv"
    reason_events().to_csv(log, index=False)
    norm = tmp_path / "norm.json"
    norm.write_text(json.dumps(reason_document(3)), encoding="utf-8")
    verdicts = tmp_path / "verdicts.csv"
    common = ["--case", "case", "--activity", "activity", "--timestamp", "time", "--attr", "flow_type"]
    rc = main(["score", str(norm), str(log), *common, "--by", "flow_type", "--out", str(tmp_path / "b.csv")])
    assert rc == 0
    assert not verdicts.exists()
    rc = main(
        [
            "score",
            str(norm),
            str(log),
            *common,
            "--by",
            "flow_type",
            "--out",
            str(tmp_path / "b.csv"),
            "--verdicts-out",
            str(verdicts),
        ]
    )
    assert rc == 0
    written = pd.read_csv(verdicts, keep_default_na=False).set_index(["case", "constraint"])
    assert written.loc[("out", "c"), "reason"] == OUT_OF_SCOPE
    assert written.loc[("unknown", "c"), "reason"] == SCOPE_UNKNOWN
    assert written.loc[("in", "c"), "reason"] == EVALUATED
    assert written.loc[("out", "c"), "value"] == written.loc[("unknown", "c"), "value"] == ""


def test_a_result_built_without_reasons_has_none_and_refuses_the_verdict_table(reason_log: wise.EventLog) -> None:
    """A caller that builds a result by hand from the 0.1.0 attributes is unaffected; only the new table needs the reasons."""
    scored = wise.score(reason_log, read(3))
    shell = wise.ScoreResult(
        norm=scored.norm,
        cases=scored.cases,
        violations=scored.violations,
        in_scope=scored.in_scope,
        scores=pd.DataFrame(index=scored.violations.index),
        contributions={},
        mode=scored.mode,
        _weights={"v": np.array([1.0, 1.0])},
    )
    assert shell.reasons is None
    assert shell.effective_weights("v").to_numpy().tolist() == [[0.5, 0.5], [0.0, 1.0], [0.0, 1.0]]
    with pytest.raises(wise.NotScoredError, match="reasons"):
        shell.verdicts()


def test_replacing_the_log_keeps_the_reasons(reason_log: wise.EventLog) -> None:
    result = wise.score(reason_log, read(3))
    detached = dataclasses.replace(result, log=None)
    assert detached.reasons is result.reasons
    pd.testing.assert_frame_equal(detached.verdicts(), result.verdicts())


@pytest.mark.parametrize(
    "table",
    [
        wise.ScoreResult.frame,
        wise.ScoreResult.worst_cases,
        wise.ScoreResult.summary,
        wise.ScoreResult.penalties,
        wise.ScoreResult.effective_weights,
        wise.violation_matrix,
        wise.prioritize,
        wise.compat.prioritize,
        wise.layer_drivers,
        wise.constraint_drivers,
        wise.penalty_mass,
        wise.case_slices,
        wise.view_agreement,
        wise.gap_retained,
        wise.validation_table,
    ],
    ids=lambda f: f"{f.__module__}.{f.__qualname__}",
)
def test_every_table_written_from_a_result_says_where_the_reasons_are(table: Any) -> None:
    doc = " ".join((table.__doc__ or "").split())
    assert "verdicts" in doc or "reasons`" in doc
