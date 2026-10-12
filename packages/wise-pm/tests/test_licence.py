"""wise-pm is MIT as a whole: the licence files agree and the built artefacts carry them.

The repository's licence documents and ``scripts/check_dist.py`` live outside the
package; these tests are skipped where only the package is present, as in an
unpacked sdist.
"""

from __future__ import annotations

import importlib.util
import io
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "check_dist.py"
PACKAGE_LICENCE = REPO / "packages" / "wise-pm" / "LICENSE"

pytestmark = pytest.mark.skipif(not SCRIPT.exists(), reason="the repository's licence documents are not part of this tree")

#: Words that belonged to the licence policy wise-pm does not have, as halves so that this file does not hold them.
_HALVES = {"any case": [("poly", "form"), ("non", "commercial"), ("rights-", "controlled")], "exact": [("MIT", " AND")]}
WITHDRAWN = re.compile(
    "(?i:" + "|".join(a + b for a, b in _HALVES["any case"]) + ")|" + "|".join(a + b for a, b in _HALVES["exact"])
)

#: Records that quote an earlier state and say so at their top.
QUOTING = {Path("docs/decisions/0001-canonical-library-and-migration-map.md")}


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_dist", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tracked() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [Path(name) for name in out.decode().split("\0") if name]


#: The MIT licence as published by the Open Source Initiative, with the owner's copyright line.
MIT_TEXT = """MIT License

Copyright (c) 2026 Urszula Jessen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def test_the_root_and_the_package_licence_are_the_same_mit_text() -> None:
    root = (REPO / "LICENSE").read_bytes()
    assert root == PACKAGE_LICENCE.read_bytes()
    assert root == MIT_TEXT.encode()


def test_the_notice_names_the_copyright_holder_and_the_mit_licence() -> None:
    notice = (REPO / "NOTICE").read_text(encoding="utf-8")
    assert notice.startswith("Copyright (c) 2026 Urszula Jessen\n")
    assert "wise-pm is distributed under the MIT licence" in notice


def test_no_tracked_file_speaks_of_another_licence_for_wise_pm() -> None:
    hits: list[str] = []
    for path in _tracked():
        if path in QUOTING:
            continue
        try:
            text = (REPO / path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        hits.extend(f"{path}:{m.group(0)}" for m in WITHDRAWN.finditer(text))
    assert hits == []


def test_a_record_that_quotes_the_earlier_state_says_so_at_its_top() -> None:
    for path in QUOTING:
        head = (REPO / path).read_text(encoding="utf-8")[:1200]
        assert "quote the state of 2026-09-27" in head, path


# ------------------------------------------------------------------ the built artefacts
METADATA = "Metadata-Version: 2.4\nName: wise-pm\nVersion: 0.2.0\nLicense-Expression: {expression}\nLicense-File: LICENSE\n{extra}\nReadme.\n"


def _artefacts(dist: Path, *, expression: str = "MIT", licence: bytes | None = None, extra: str = "") -> None:
    text = PACKAGE_LICENCE.read_bytes() if licence is None else licence
    metadata = METADATA.format(expression=expression, extra=extra).encode()
    with zipfile.ZipFile(dist / "wise_pm-0.2.0-py3-none-any.whl", "w") as wheel:
        wheel.writestr("wise_pm-0.2.0.dist-info/METADATA", metadata)
        wheel.writestr("wise_pm-0.2.0.dist-info/licenses/LICENSE", text)
    with tarfile.open(dist / "wise_pm-0.2.0.tar.gz", "w:gz") as sdist:
        for name, data in (("PKG-INFO", metadata), ("LICENSE", text)):
            info = tarfile.TarInfo(f"wise_pm-0.2.0/{name}")
            info.size = len(data)
            sdist.addfile(info, io.BytesIO(data))


def test_artefacts_with_the_mit_text_and_expression_pass(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _artefacts(tmp_path)
    assert _load().main(["check_dist.py", str(tmp_path)]) == 0
    assert capsys.readouterr().out.count("carries the MIT licence text and License-Expression: MIT") == 2


def test_a_combined_expression_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _artefacts(tmp_path, expression="MIT OR Apache-2.0")
    assert _load().main(["check_dist.py", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "wise_pm-0.2.0-py3-none-any.whl: License-Expression is ['MIT OR Apache-2.0'], expected ['MIT']" in err
    assert "wise_pm-0.2.0.tar.gz: License-Expression is ['MIT OR Apache-2.0'], expected ['MIT']" in err


def test_another_licence_text_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _artefacts(tmp_path, licence=b"Some other licence\n")
    assert _load().main(["check_dist.py", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "wise_pm-0.2.0-py3-none-any.whl: the licence file is not the MIT text of LICENSE" in err
    assert "wise_pm-0.2.0.tar.gz: the licence file is not the MIT text of LICENSE" in err


def test_a_directory_without_both_artefacts_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _load().main(["check_dist.py", str(tmp_path)]) == 1
    assert "does not hold both a wise_pm wheel and a wise_pm sdist" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "problem"),
    [
        ("License: MIT\n", "carries a License field besides License-Expression"),
        ("Classifier: License :: OSI Approved :: MIT License\n", "carries licence classifiers"),
        ("License-File: NOTICE\n", "License-File is ['LICENSE', 'NOTICE'], expected ['LICENSE']"),
    ],
)
def test_a_further_licence_field_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: str, problem: str) -> None:
    _artefacts(tmp_path, extra=extra)
    assert _load().main(["check_dist.py", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert f"wise_pm-0.2.0-py3-none-any.whl: {problem}" in err
    assert f"wise_pm-0.2.0.tar.gz: {problem}" in err
