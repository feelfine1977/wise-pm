"""Verify that the built wise-pm wheel and sdist carry the MIT licence text and expression.

Usage: python scripts/check_dist.py dist/
Each ``wise_pm-*.whl`` and ``wise_pm-*.tar.gz`` in the directory must hold the licence
file byte for byte as ``LICENSE`` and ``packages/wise-pm/LICENSE`` (which must agree),
and its metadata must declare ``License-Expression: MIT`` and no other licence field.
Prints one line per checked artefact and exits 1 with one line per problem otherwise.
Standard library only.
"""

from __future__ import annotations

import sys
import tarfile
import zipfile
from email.parser import BytesHeaderParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LICENCE_FILES = [ROOT / "LICENSE", ROOT / "packages" / "wise-pm" / "LICENSE"]


def metadata_problems(name: str, raw: bytes) -> list[str]:
    """What is wrong with the licence fields of one METADATA or PKG-INFO file."""
    headers = BytesHeaderParser().parsebytes(raw)
    problems = []
    expressions = headers.get_all("License-Expression") or []
    if expressions != ["MIT"]:
        problems.append(f"{name}: License-Expression is {expressions or 'missing'}, expected ['MIT']")
    if headers.get_all("License"):
        problems.append(f"{name}: carries a License field besides License-Expression")
    classifiers = [c for c in headers.get_all("Classifier") or [] if c.startswith("License ::")]
    if classifiers:
        problems.append(f"{name}: carries licence classifiers {classifiers}")
    if headers.get_all("License-File") != ["LICENSE"]:
        problems.append(f"{name}: License-File is {headers.get_all('License-File')}, expected ['LICENSE']")
    return problems


def wheel_members(path: Path) -> tuple[bytes | None, bytes | None]:
    """(METADATA, licence file) of a wheel; None for a member that is missing."""
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        metadata = next((n for n in names if n.endswith(".dist-info/METADATA")), None)
        licence = next((n for n in names if n.endswith(".dist-info/licenses/LICENSE")), None)
        return (wheel.read(metadata) if metadata else None, wheel.read(licence) if licence else None)


def sdist_members(path: Path) -> tuple[bytes | None, bytes | None]:
    """(PKG-INFO, licence file) at the top of an sdist; None for a member that is missing."""
    with tarfile.open(path) as sdist:
        top = sdist.getnames()[0].split("/")[0]

        def read(name: str) -> bytes | None:
            try:
                member = sdist.extractfile(f"{top}/{name}")
            except KeyError:
                return None
            return member.read() if member else None

        return read("PKG-INFO"), read("LICENSE")


def check(dist: Path, licence: bytes) -> list[str]:
    problems: list[str] = []
    artefacts = sorted(dist.glob("wise_pm-*.whl")) + sorted(dist.glob("wise_pm-*.tar.gz"))
    if not any(a.suffix == ".whl" for a in artefacts) or not any(a.name.endswith(".tar.gz") for a in artefacts):
        problems.append(f"{dist} does not hold both a wise_pm wheel and a wise_pm sdist")
    for artefact in artefacts:
        metadata, text = wheel_members(artefact) if artefact.suffix == ".whl" else sdist_members(artefact)
        found = metadata_problems(artefact.name, metadata) if metadata is not None else [f"{artefact.name}: no metadata file"]
        if text != licence:
            found.append(f"{artefact.name}: the licence file is {'missing' if text is None else 'not the MIT text of LICENSE'}")
        if not found:
            print(f"dist check: {artefact.name} carries the MIT licence text and License-Expression: MIT")
        problems.extend(found)
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_dist.py <dist directory>", file=sys.stderr)
        return 2
    texts = [p.read_bytes() for p in LICENCE_FILES]
    problems = [] if texts[0] == texts[1] else ["LICENSE and packages/wise-pm/LICENSE differ"]
    if not texts[0].startswith(b"MIT License\n"):
        problems.append("LICENSE is not the MIT licence text")
    problems.extend(check(Path(argv[1]), texts[1]))
    for p in problems:
        print(f"dist check: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
