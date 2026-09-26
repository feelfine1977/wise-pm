"""Keep the strict expected failures and the known-defect ledger in step.

Every ``xfail`` under this directory must cite a ledger id (``C<n>``) that is still
open, and no open marker may remain for a ledger entry recorded as fixed
(``docs/decisions/known-defects.json``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

LEDGER = Path(__file__).resolve().parents[4] / "docs" / "decisions" / "known-defects.json"
ID = re.compile(r"\b(C\d+)[a-z]?\b")  # C3a / C3b are sub-cases of C3


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
    if not LEDGER.exists():  # packaged sdist run without the workspace docs
        return
    ledger = {e["id"]: e for e in json.loads(LEDGER.read_text(encoding="utf-8"))["entries"]}
    here = Path(__file__).resolve().parent
    problems: list[str] = []
    cited: set[str] = set()
    for item in items:
        if not str(item.fspath).startswith(str(here)):
            continue
        for marker in item.iter_markers("xfail"):
            reason = str(marker.kwargs.get("reason", ""))
            ids = ID.findall(reason)
            if not ids:
                problems.append(f"{item.nodeid}: xfail reason cites no ledger id: {reason!r}")
            for cid in ids:
                cited.add(cid)
                entry = ledger.get(cid)
                if entry is None:
                    problems.append(f"{item.nodeid}: {cid} is not in the known-defect ledger")
                elif entry["status"] != "open":
                    problems.append(f"{item.nodeid}: {cid} is recorded as {entry['status']} but still carries an xfail")
    if problems:
        raise pytest.UsageError("known-defect ledger out of step:\n  " + "\n  ".join(problems))
