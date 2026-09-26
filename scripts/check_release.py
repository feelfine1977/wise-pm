"""Verify that a release tag, the package versions, CITATION.cff and CHANGELOG.md agree.

Usage: python scripts/check_release.py v0.2.0
Prints ``version=0.2.0`` on success (for GITHUB_OUTPUT) and exits 1 with one line
per problem otherwise. Standard library only, so it runs before any sync.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILES = [
    ROOT / "packages" / "wise-pm" / "src" / "wise" / "_version.py",
    ROOT / "packages" / "wise-pm-actionability" / "src" / "wise_actionability" / "_version.py",
]
SEMVER = r"\d+\.\d+\.\d+(?:[-.]?(?:a|b|rc|dev)\d+)?"


def read_version_file(path: Path) -> str | None:
    if not path.exists():
        return None
    m = re.search(r'__version__\s*=\s*"([^"]+)"', path.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def changelog_versions(text: str) -> tuple[str | None, bool]:
    """Return (first released version heading, unreleased section is empty)."""
    headings = list(re.finditer(rf"^## \[({SEMVER}|Unreleased)\]", text, flags=re.M))
    first_release = next((h.group(1) for h in headings if h.group(1) != "Unreleased"), None)
    unreleased_empty = True
    for i, h in enumerate(headings):
        if h.group(1) == "Unreleased":
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            body = text[h.end() : end]
            unreleased_empty = not re.sub(r"\s+", "", body)
    return first_release, unreleased_empty


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_release.py v<version>", file=sys.stderr)
        return 2
    tag = argv[1]
    problems: list[str] = []
    if not tag.startswith("v"):
        problems.append(f"tag {tag!r} does not start with 'v'")
    version = tag[1:]

    for path in VERSION_FILES:
        found = read_version_file(path)
        if found is None and path == VERSION_FILES[0]:
            problems.append(f"missing or unreadable {path.relative_to(ROOT)}")
        elif found is not None and found != version:
            problems.append(f"{path.relative_to(ROOT)} has {found}, tag says {version}")

    cff = ROOT / "CITATION.cff"
    m = re.search(r"^version:\s*\"?([^\"\n]+)\"?\s*$", cff.read_text(encoding="utf-8"), flags=re.M)
    if not m:
        problems.append("CITATION.cff has no version line")
    elif m.group(1).strip() != version:
        problems.append(f"CITATION.cff has {m.group(1).strip()}, tag says {version}")

    first_release, unreleased_empty = changelog_versions((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
    if first_release != version:
        problems.append(f"CHANGELOG.md first release heading is {first_release}, tag says {version}")
    if not unreleased_empty:
        problems.append("CHANGELOG.md still has entries under [Unreleased]")

    for p in problems:
        print(f"release check: {p}", file=sys.stderr)
    if problems:
        return 1
    print(f"version={version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
