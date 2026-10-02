"""The changelog's section for the coming release is complete.

It has the sections Added, Changed, Deprecated and Fixed; it closes each fixed
entry of the defect ledger with exactly one ``Fixed: C<n>`` item that states the
delta on the golden dataset, and no open one; and its migration note names every
0.1.0 name of ``wise.compat.PATHS``. The changelog and the ledger live in the
repository, not in the package.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import wise
from wise.compat import PATHS

REPO = Path(wise.__file__).resolve().parents[4]
CHANGELOG = REPO / "CHANGELOG.md"
LEDGER = REPO / "docs" / "decisions" / "known-defects.json"

pytestmark = pytest.mark.skipif(
    not (CHANGELOG.exists() and LEDGER.exists()), reason="the changelog and the ledger are not part of the package"
)


def _unreleased() -> str:
    text = CHANGELOG.read_text(encoding="utf-8")
    match = re.search(r"^## \[Unreleased\]\n(.*?)(?=^## \[)", text, flags=re.M | re.S)
    assert match is not None
    return match.group(1)


def _sections() -> dict[str, str]:
    parts = re.split(r"^### (.+)$", _unreleased(), flags=re.M)
    return {parts[i].strip(): parts[i + 1] for i in range(1, len(parts), 2)}


def _items(section: str) -> list[str]:
    return [item.strip() for item in re.split(r"^- ", section, flags=re.M)[1:]]


def test_the_section_has_the_kinds_of_change_and_a_migration_note() -> None:
    assert list(_sections()) == ["Changed", "Added", "Deprecated", "Fixed", "Migrating from 0.1.0"]


@pytest.mark.spec("decision:0001")
def test_each_fixed_ledger_entry_has_one_fixed_item_with_its_delta_and_no_open_entry_has_one() -> None:
    entries = json.loads(LEDGER.read_text(encoding="utf-8"))["entries"]
    fixed_items: dict[str, list[str]] = {}
    for item in _items(_sections()["Fixed"]):
        if m := re.match(r"Fixed: (C\d+) ", item):
            fixed_items.setdefault(m.group(1), []).append(item)
    for entry in entries:
        items = fixed_items.pop(entry["id"], [])
        if entry["status"] == "fixed":
            assert len(items) == 1, entry["id"]
            assert "delta" in items[0].lower(), entry["id"]
        else:
            assert items == [], entry["id"]
    assert fixed_items == {}  # no item closes an entry the ledger does not know


@pytest.mark.spec("decision:0001/2")
def test_the_migration_note_names_every_changed_0_1_0_name() -> None:
    note = _sections()["Migrating from 0.1.0"]
    assert "wise.compat" in note
    for name in PATHS:
        spelled = name.rpartition(".")[2]
        assert spelled in note, name
    for item in ("wise.compat.prioritize", "wise.compat.norm_from_dict", "_rule_attributes"):
        assert item in _sections()["Added"]
