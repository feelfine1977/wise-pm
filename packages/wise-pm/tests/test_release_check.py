"""The release check refuses a tag that disagrees with the package, CITATION.cff or the changelog.

The check lives at the repository root (``scripts/check_release.py``); these tests are skipped where
only the package is present, as in an unpacked sdist.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType

import pytest

import wise

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "check_release.py"
VERSION_FILE = Path("packages") / "wise-pm" / "src" / "wise" / "_version.py"

pytestmark = pytest.mark.skipif(not SCRIPT.exists(), reason="repository scripts are not part of this tree")


def _load(script: Path = SCRIPT) -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_release", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(root: Path, *, package: str, citation: str, changelog: str) -> None:
    version_file = root / VERSION_FILE
    version_file.parent.mkdir(parents=True)
    version_file.write_text(f'__version__ = "{package}"\n', encoding="utf-8")
    (root / "CITATION.cff").write_text(f"cff-version: 1.2.0\nversion: {citation}\n", encoding="utf-8")
    (root / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    ledger = root / "docs" / "decisions" / "known-defects.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({"schema": 1, "entries": []}), encoding="utf-8")


@pytest.fixture
def check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The check pointed at an empty tree under ``tmp_path``, without the golden reproduction."""
    module = _load()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "VERSION_FILES", [tmp_path / VERSION_FILE])
    monkeypatch.setattr(module, "golden_exact_problem", lambda: None)
    return module


RELEASED = "# Changelog\n\n## [Unreleased]\n\n## [0.2.0] — 2026-12-01\n\n- Notes.\n\n## [0.1.0] — 2026-09-05\n\nFirst release.\n"
DEVELOPING = "# Changelog\n\n## [Unreleased]\n\n- Work in progress.\n\n## [0.1.0] — 2026-09-05\n\nFirst release.\n"


def test_a_tag_that_agrees_everywhere_passes(check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _tree(tmp_path, package="0.2.0", citation="0.2.0", changelog=RELEASED)
    assert check.main(["check_release.py", "v0.2.0"]) == 0
    assert capsys.readouterr().out.strip() == "version=0.2.0"


def test_a_development_version_refuses_the_release_tag(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _tree(tmp_path, package="0.2.0.dev0", citation="0.1.0", changelog=DEVELOPING)
    assert check.main(["check_release.py", "v0.2.0"]) == 1
    err = capsys.readouterr().err
    assert "_version.py has 0.2.0.dev0, tag says 0.2.0" in err
    assert "CITATION.cff has 0.1.0, tag says 0.2.0" in err
    assert "CHANGELOG.md first release heading is 0.1.0, tag says 0.2.0" in err
    assert "CHANGELOG.md still has entries under [Unreleased]" in err


def test_the_previous_release_tag_is_refused_on_a_development_line(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _tree(tmp_path, package="0.2.0.dev0", citation="0.1.0", changelog=DEVELOPING)
    assert check.main(["check_release.py", "v0.1.0"]) == 1
    err = capsys.readouterr().err
    assert "_version.py has 0.2.0.dev0, tag says 0.1.0" in err
    assert "CITATION.cff" not in err


def test_a_tag_without_the_v_prefix_is_refused(check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _tree(tmp_path, package="0.2.0", citation="0.2.0", changelog=RELEASED)
    assert check.main(["check_release.py", "0.2.0"]) == 1
    assert "does not start with 'v'" in capsys.readouterr().err


def test_this_checkout_refuses_a_tag_that_is_not_its_version(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load()
    monkeypatch.setattr(module, "golden_exact_problem", lambda: None)
    assert module.read_version_file(module.VERSION_FILES[0]) == wise.__version__
    assert module.main(["check_release.py", "v99.0.0"]) == 1
    assert f"_version.py has {wise.__version__}, tag says 99.0.0" in capsys.readouterr().err


@pytest.mark.parametrize("heading", ["## [0.2.0] — YYYY-MM-DD", "## [0.2.0]", "## [0.2.0] — 2026-13-01"])
def test_a_release_heading_without_a_real_date_is_refused(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str], heading: str
) -> None:
    _tree(tmp_path, package="0.2.0", citation="0.2.0", changelog=RELEASED.replace("## [0.2.0] — 2026-12-01", heading))
    assert check.main(["check_release.py", "v0.2.0"]) == 1
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1
    assert err[0].startswith("release check: CHANGELOG.md heading of 0.2.0 has no release date")
    assert err[0].endswith("enter the date of the tag")


def test_the_version_file_of_every_package_of_the_workspace_is_read(tmp_path: Path) -> None:
    other = tmp_path / "packages" / "companion" / "src" / "companion" / "_version.py"
    for path in (tmp_path / VERSION_FILE, other):
        path.parent.mkdir(parents=True)
        path.write_text('__version__ = "0.2.0"\n', encoding="utf-8")
    script = tmp_path / "scripts" / "check_release.py"
    script.parent.mkdir()
    shutil.copyfile(SCRIPT, script)
    module = _load(script)
    assert [tmp_path / VERSION_FILE, other] == module.VERSION_FILES
