"""Print the CHANGELOG.md section of one version (for GitHub release notes).

Usage: python scripts/changelog_section.py 0.2.0
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def section(text: str, version: str) -> str | None:
    pattern = rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)"
    m = re.search(pattern, text, flags=re.M | re.S)
    return m.group(1).strip() + "\n" if m else None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: changelog_section.py <version>", file=sys.stderr)
        return 2
    text = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text(encoding="utf-8")
    body = section(text, argv[1])
    if body is None:
        print(f"no CHANGELOG section for {argv[1]}", file=sys.stderr)
        return 1
    sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
