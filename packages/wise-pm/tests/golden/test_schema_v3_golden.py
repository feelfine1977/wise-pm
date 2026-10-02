"""The schema version 3 copy of the running example norm, scored beside the schema version 2 original.

``examples/running_p2p_norm_v3.json`` is ``examples/running_p2p_norm.json``
with ``schema_version`` 3 and the named missing-value policy; it is evaluated
under the corrected semantics (decision 0001, section 3), the original under
the 0.1.0 semantics (section 4). Both are scored on the paper's running
example and on the golden dataset in both scoring modes. A cell that differs
must carry the correction that explains it (a case whose applicability is
unknown, C4/C25); the delta report ``docs/semantics/schema-v3-delta.md``,
referenced from the CHANGELOG, lists the rows that differ. There are none:
the norm's only applicability rule is a membership rule, under which 0.1.0
already left a case without the attribute out of scope, its lag states its
bound, and it has no recipes.

The committed copy is what ``wise.migrate`` makes of the original, and the
golden norm (``tests/data/golden/norm.json``) is migrated here and scored
beside its schema version 2 record in the same way; the migration report
names the corrections that may explain a differing row.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

import wise
from wise.family import SCOPE_UNKNOWN

PKG = Path(__file__).resolve().parents[2]
EXAMPLES = PKG / "examples"
DATA = PKG / "tests" / "data"
REPORT = PKG.parents[1] / "docs" / "semantics" / "schema-v3-delta.md"

pytestmark = [
    pytest.mark.spec("golden"),
    # The original is a schema version 2 document: it evaluates under the 0.1.0 semantics, announced on every read.
    pytest.mark.filterwarnings("ignore::wise.LegacySemanticsWarning"),
]

#: Paper Table VI: (Finance, Logistics) per case.
TABLE_VI = {
    "A": (0.6625, 0.8875),
    "B": (0.9667, 0.7000),
    "C": (0.8700, 0.9675),
    "D": (0.9000, 0.9000),
    "E": (0.1500, 0.5500),
}
#: The declared delta of the v3 copy: (dataset, mode) → cases whose rows differ from the v2 original.
DECLARED_DELTA: dict[tuple[str, str], list[str]] = {
    (dataset, mode): [] for dataset in ("running_example", "golden") for mode in ("flat", "layer_balanced")
}


def _generator() -> ModuleType:
    path = PKG / "scripts" / "make_golden.py"
    spec = importlib.util.spec_from_file_location("make_golden", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def logs() -> dict[str, wise.EventLog]:
    mg = _generator()
    return {"running_example": wise.running_p2p_log(), "golden": mg.load_event_log(DATA / mg.DATASET_FILE)}


@pytest.fixture(scope="module")
def norms() -> dict[int, wise.Norm]:
    return {2: wise.Norm.load(EXAMPLES / "running_p2p_norm.json"), 3: wise.Norm.load(EXAMPLES / "running_p2p_norm_v3.json")}


def test_the_v3_copy_is_the_migrated_original() -> None:
    migration = wise.migrate(json.loads((EXAMPLES / "running_p2p_norm.json").read_text(encoding="utf-8")))
    assert migration.dumps() + "\n" == (EXAMPLES / "running_p2p_norm_v3.json").read_text(encoding="utf-8")
    assert migration.report.corrections == ("C4", "C25")


def test_the_v3_copy_differs_from_the_original_only_in_its_schema(norms: dict[int, wise.Norm]) -> None:
    original, copy = norms[2].to_dict(), norms[3].to_dict()
    assert (original.pop("schema_version"), copy.pop("schema_version")) == (2, 3)
    assert copy.pop("missing_values") == "three_valued"
    assert copy == original
    assert norms[2].semantics.name == "0.1.0"
    assert norms[3].semantics.name == "corrected"


@pytest.mark.spec("VI")
def test_the_v3_copy_reproduces_table_vi(norms: dict[int, wise.Norm], logs: dict[str, wise.EventLog]) -> None:
    result = wise.score(logs["running_example"], norms[3])
    for case, (finance, logistics) in TABLE_VI.items():
        assert result.scores.loc[case, "Finance"] == pytest.approx(finance, abs=5e-5)
        assert result.scores.loc[case, "Logistics"] == pytest.approx(logistics, abs=5e-5)


def _differing_rows(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    assert left.index.equals(right.index)
    assert left.columns.equals(right.columns)
    same = (left == right) | (left.isna() & right.isna())
    return [str(case) for case in left.index[~same.all(axis=1)]]


def _explained_rows(norm: wise.Norm, log: wise.EventLog | None, corrections: tuple[str, ...]) -> list[str]:
    """The cases whose rows a correction named in a migration report may change: under three-valued applicability
    (C4, C25) the cases whose scope is unknown; no other correction changes a row of a norm that scores."""
    assert log is not None
    if not {"C4", "C25"} & set(corrections):
        return []
    unknown = pd.DataFrame(
        {nc.id: wise.evaluate(nc, log, semantics=norm.semantics).reason == SCOPE_UNKNOWN for nc in norm.constraints}
    )
    return [str(case) for case in unknown.index[unknown.any(axis=1)]]


def _assert_rows_differ_only_by_correction(
    legacy: wise.ScoreResult, corrected: wise.ScoreResult, explained: list[str], declared: list[str]
) -> None:
    frames = [("violations", legacy.violations, corrected.violations), ("in_scope", legacy.in_scope, corrected.in_scope)]
    frames += [("scores", legacy.scores, corrected.scores)]
    frames += [(f"contributions[{v}]", legacy.contributions[v], corrected.contributions[v]) for v in legacy.views]
    differing: set[str] = set()
    for name, left, right in frames:
        rows = _differing_rows(left, right)
        assert set(rows) <= set(explained), f"{name}: rows {sorted(set(rows) - set(explained))} differ without a correction"
        differing.update(rows)
        if not rows:
            pd.testing.assert_frame_equal(left, right, check_exact=True, obj=name)
    assert sorted(differing) == declared


@pytest.mark.parametrize("mode", ["flat", "layer_balanced"])
def test_the_migrated_golden_norm_differs_from_its_v2_record_only_by_named_corrections(
    logs: dict[str, wise.EventLog], mode: str
) -> None:
    """The golden tier proves the v2 norm reproduces the committed record; its migrated norm is compared with it here."""
    mg = _generator()
    migration = wise.migrate(json.loads((DATA / mg.GOLDEN_SUBDIR / mg.NORM_FILE).read_text(encoding="utf-8")))
    assert [r.rule for r in migration.report.rewrites] == ["missing_values"]
    log = logs["golden"]
    legacy = wise.score(log, wise.Norm.load(DATA / mg.GOLDEN_SUBDIR / mg.NORM_FILE), mode=mode)
    corrected = wise.score(log, migration.norm, mode=mode)
    explained = _explained_rows(migration.norm, corrected.log, migration.report.corrections)
    _assert_rows_differ_only_by_correction(legacy, corrected, explained, declared=[])


@pytest.mark.parametrize("mode", ["flat", "layer_balanced"])
@pytest.mark.parametrize("dataset", ["running_example", "golden"])
def test_rows_that_differ_from_v2_carry_their_correction(
    norms: dict[int, wise.Norm], logs: dict[str, wise.EventLog], dataset: str, mode: str
) -> None:
    log = logs[dataset]
    legacy, corrected = (wise.score(log, norms[v], mode=mode) for v in (2, 3))
    explained = _explained_rows(norms[3], corrected.log, ("C4", "C25"))
    _assert_rows_differ_only_by_correction(legacy, corrected, explained, DECLARED_DELTA[(dataset, mode)])


@pytest.mark.skipif(not REPORT.exists(), reason="the changelog and the delta report are not part of the package")
def test_the_delta_report_is_referenced_from_the_changelog() -> None:
    report = REPORT.read_text(encoding="utf-8")
    assert "running_p2p_norm_v3.json" in report
    assert "docs/semantics/schema-v3-delta.md" in (PKG.parents[1] / "CHANGELOG.md").read_text(encoding="utf-8")
