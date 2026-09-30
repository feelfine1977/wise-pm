"""Constructed byte fixtures exercise import safety without real datasets."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path, PureWindowsPath

import pytest

SPEC = importlib.util.spec_from_file_location("evaluation_data_manager", Path(__file__).resolve().parents[1] / "data_manager.py")
assert SPEC is not None
assert SPEC.loader is not None
dm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dm)


def fixture_catalogue(payload: bytes = b"case,activity\n1,start\n") -> dict:
    return {
        "datasets": [
            {
                "id": "constructed",
                "files": [
                    {
                        "source_relative_path": "OCEL/events.csv",
                        "path": "data/raw/constructed/events.csv",
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
        ]
    }


def prepare(tmp_path: Path) -> tuple[Path, Path, bytes, dict]:
    source, raw = tmp_path / "source", tmp_path / "raw"
    payload = b"case,activity\n1,start\n"
    path = source / "OCEL" / "events.csv"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    return source, raw, payload, fixture_catalogue(payload)


def test_import_verification_and_idempotent_reuse(tmp_path):
    source, raw, payload, catalogue = prepare(tmp_path)
    first = dm.import_data(source, raw, catalogue)
    second = dm.import_data(source, raw, catalogue)
    assert first["copied"] == 1
    assert first["reused"] == 0
    assert second["copied"] == 0
    assert second["reused"] == 1
    assert (source / "OCEL/events.csv").read_bytes() == payload
    assert (raw / "constructed/events.csv").read_bytes() == payload
    assert dm.verify_data(raw, catalogue)["status"] == "passed"


def test_mismatched_existing_destination_is_preserved(tmp_path):
    source, raw, payload, catalogue = prepare(tmp_path)
    target = raw / "constructed/events.csv"
    target.parent.mkdir(parents=True)
    original = b"existing user data must survive"
    target.write_bytes(original)
    with pytest.raises(ValueError, match="mismatch"):
        dm.import_data(source, raw, catalogue)
    assert target.read_bytes() == original
    assert (source / "OCEL/events.csv").read_bytes() == payload
    assert list(raw.rglob(".import-*")) == []


def test_source_hash_mismatch_fails_before_creating_destination(tmp_path):
    source, raw, payload, catalogue = prepare(tmp_path)
    (source / "OCEL/events.csv").write_bytes(payload.replace(b"start", b"other"))
    with pytest.raises(ValueError, match="mismatch"):
        dm.import_data(source, raw, catalogue)
    assert not raw.exists()


@pytest.mark.parametrize("unsafe", ["../outside", "/outside", "C:/outside", "folder\\outside"])
def test_traversal_and_absolute_catalogue_paths_rejected(unsafe):
    catalogue = fixture_catalogue()
    catalogue["datasets"][0]["files"][0]["source_relative_path"] = unsafe
    with pytest.raises(ValueError, match=r"[Pp]ath"):
        dm.entries(catalogue)


@pytest.mark.parametrize("relative", [Path("OCEL/events.csv"), PureWindowsPath("OCEL/events.csv")])
def test_contained_path_accepts_native_path_objects(tmp_path, relative):
    assert dm.contained_path(tmp_path, relative) == tmp_path.resolve() / "OCEL" / "events.csv"


@pytest.mark.parametrize(
    "unsafe",
    ["../outside", "/outside", "C:/outside", "folder\\outside", PureWindowsPath("../outside"), PureWindowsPath("C:/outside")],
)
def test_contained_path_rejects_unsafe_inputs(tmp_path, unsafe):
    with pytest.raises(ValueError, match=r"[Pp]ath"):
        dm.contained_path(tmp_path, unsafe)


def test_symlink_parent_cannot_escape_destination_root(tmp_path):
    source, raw, _, catalogue = prepare(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    raw.mkdir()
    try:
        (raw / "constructed").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Creating symlinks requires a Windows privilege")
        raise
    with pytest.raises(ValueError, match="escapes"):
        dm.import_data(source, raw, catalogue)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("original_directory", ["OCEL", "OCEL "])
def test_legacy_context_normalizes_ocel_and_removes_only_temporary_tree(tmp_path, original_directory):
    source, raw, payload, catalogue = prepare(tmp_path)
    dm.import_data(source, raw, catalogue)
    catalogue["datasets"][0]["files"][0]["source_relative_path"] = f"{original_directory}/events.csv"
    with dm.legacy_layout(raw, catalogue) as layout:
        assert (layout / "OCEL/events.csv").read_bytes() == payload
        assert {entry.name for entry in layout.iterdir()} == {"OCEL"}
        temporary_path = layout
    assert not temporary_path.exists()
    assert (raw / "constructed/events.csv").read_bytes() == payload
    assert (source / "OCEL/events.csv").read_bytes() == payload


def test_legacy_copy_fallback_preserves_input(tmp_path, monkeypatch):
    source, raw, payload, catalogue = prepare(tmp_path)
    dm.import_data(source, raw, catalogue)

    def unavailable_link(*args, **kwargs):
        raise OSError("constructed cross-device link failure")

    monkeypatch.setattr(dm.os, "link", unavailable_link)
    with dm.legacy_layout(raw, catalogue) as layout:
        target = layout / "OCEL/events.csv"
        assert target.read_bytes() == payload
        assert not target.samefile(raw / "constructed/events.csv")
    assert (raw / "constructed/events.csv").read_bytes() == payload
