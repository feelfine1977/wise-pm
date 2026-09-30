"""Import and verify local benchmark bytes using the portable data catalogue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_CATALOG = HERE / "data_catalog.json"
DEFAULT_DATA_ROOT = Path(os.environ.get("WISE_EVALUATION_DATA_ROOT", str(HERE / "data" / "raw")))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def relative_path(value: str) -> Path:
    """Accept portable relative paths, never absolute paths or traversal."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
        raise ValueError("Invalid relative catalogue path")
    parts = PurePosixPath(value).parts
    if PurePosixPath(value).is_absolute() or not parts or any(p in {"..", "."} for p in parts):
        raise ValueError("Catalogue path must remain relative")
    return Path(*parts)


def contained_path(root: Path, relative: str | Path) -> Path:
    root = root.resolve()
    rel = relative_path(str(relative))
    result = root / rel
    if not result.resolve().is_relative_to(root):
        raise ValueError("Catalogue path escapes its root through a link")
    return result


def load_catalogue(path: Path = DEFAULT_CATALOG) -> dict[str, Any]:
    catalogue = json.loads(path.read_text())
    entries(catalogue)  # Validate before any filesystem mutation.
    return catalogue


def entries(catalogue: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    targets: set[str] = set()
    legacy: set[str] = set()
    for dataset in catalogue["datasets"]:
        for original in dataset["files"]:
            row = dict(original)
            source = relative_path(row["source_relative_path"])
            path = relative_path(row["path"])
            if path.parts[:2] != ("data", "raw") or len(path.parts) < 3:
                raise ValueError("Catalogue destinations must be below data/raw")
            target = Path(*path.parts[2:])
            if str(target) in targets or str(source) in legacy:
                raise ValueError("Duplicate catalogue destination or source identity")
            if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
                raise ValueError("Invalid SHA-256")
            if isinstance(row["bytes"], bool) or not isinstance(row["bytes"], int) or row["bytes"] < 0:
                raise ValueError("Invalid source size")
            targets.add(str(target))
            legacy.add(str(source))
            row["dataset"] = dataset["id"]
            row["relative_data_path"] = target.as_posix()
            rows.append(row)
    if not rows:
        raise ValueError("Catalogue contains no data files")
    return rows


def check_file(path: Path, entry: dict[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing catalogue input: {entry['path']}")
    if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
        raise ValueError(f"Input size/hash mismatch: {entry['path']}")


def verify_data(data_root: Path, catalogue: dict[str, Any]) -> dict[str, Any]:
    rows = entries(catalogue)
    for row in rows:
        check_file(contained_path(data_root, row["relative_data_path"]), row)
    return {"status": "passed", "files": len(rows), "bytes": sum(r["bytes"] for r in rows)}


def import_data(source_root: Path, data_root: Path, catalogue: dict[str, Any]) -> dict[str, Any]:
    """Copy exact inputs; reject an existing mismatched destination unchanged."""
    source_root, data_root = source_root.resolve(), data_root.resolve()
    if source_root == data_root or data_root.is_relative_to(source_root):
        raise ValueError("Destination must not be inside the supplied source root")
    rows = entries(catalogue)
    # Preflight every source and existing destination before copying anything.
    for row in rows:
        source = contained_path(source_root, row["source_relative_path"])
        target = contained_path(data_root, row["relative_data_path"])
        check_file(source, row)
        if target.exists() or target.is_symlink():
            check_file(target, row)
    copied = reused = 0
    for row in rows:
        source = contained_path(source_root, row["source_relative_path"])
        target = contained_path(data_root, row["relative_data_path"])
        if target.exists():
            check_file(target, row)
            reused += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".import-", dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(source, temporary)
            check_file(temporary, row)
            try:
                # Publish without overwriting a destination created after preflight.
                os.link(temporary, target)
                copied += 1
            except FileExistsError:
                check_file(target, row)
                reused += 1
            except OSError:
                # Some filesystems do not support hardlinks even within a directory.
                created = False
                try:
                    with target.open("xb") as destination, temporary.open("rb") as source_stream:
                        created = True
                        shutil.copyfileobj(source_stream, destination)
                    check_file(target, row)
                    copied += 1
                except FileExistsError:
                    check_file(target, row)
                    reused += 1
                except Exception:
                    if created:
                        target.unlink(missing_ok=True)
                    raise
        finally:
            temporary.unlink(missing_ok=True)
    report = verify_data(data_root, catalogue)
    report.update(copied=copied, reused=reused, source_modified=False)
    return report


@contextmanager
def legacy_layout(data_root: Path, catalogue: dict[str, Any]) -> Iterator[Path]:
    """Temporary legacy names for read-only adapters, using links or copies.

    OCEL's original directory contains a trailing space. The temporary layout
    intentionally calls it OCEL, matching the portable adapter configuration.
    The context removes only its temporary tree and never changes permissions.
    """
    verify_data(data_root, catalogue)
    with tempfile.TemporaryDirectory(prefix="wise-evaluation-inputs-") as directory:
        layout = Path(directory)
        for row in entries(catalogue):
            original = relative_path(row["source_relative_path"])
            parts = list(original.parts)
            if parts[0] == "OCEL ":
                parts[0] = "OCEL"
            target = contained_path(layout, Path(*parts))
            target.parent.mkdir(parents=True, exist_ok=True)
            source = contained_path(data_root, row["relative_data_path"])
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
            check_file(target, row)
        try:
            yield layout
        finally:
            # Detect any unexpected mutation by the read-only adapter contract.
            verify_data(data_root, catalogue)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["import", "verify"])
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()
    catalogue = load_catalogue(args.catalog)
    if args.command == "import":
        if args.source_root is None:
            parser.error("import requires --source-root")
        report = import_data(args.source_root, args.data_root, catalogue)
    else:
        report = verify_data(args.data_root, catalogue)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
