"""The norm document format: a :class:`~wise.norm.Norm` as a JSON document and back.

A document names its format in the ``schema_version`` key, which is required.
Each schema version is read and written by an adapter; :mod:`wise.norm_document.v2`
reads the layout of 0.1.0 (schema versions 1 and 2). The norm keeps the
version it was read with (``Norm.schema_version``) and :func:`write` emits it
again, so reading and writing a document never changes its version.

The fingerprint of a norm is defined here, beside the format: the SHA-256 of
the canonical JSON (sorted keys, no whitespace) of the written document
without its free-form parts, the ``metadata`` and the descriptions. Two norms
that measure the same thing have the same fingerprint, whatever their notes
say.

``Norm.load``, ``loads``, ``dump``, ``dumps``, ``from_dict``, ``to_dict``,
``from_json``, ``to_json`` and ``fingerprint`` delegate to this module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._json import jsonable
from ..errors import NormError
from ..norm import SCHEMA_VERSION, SCHEMA_VERSIONS, Norm, NormConstraint
from . import v2


@dataclass(frozen=True)
class Adapter:
    """How the documents of some schema versions are read and written."""

    versions: tuple[int, ...]
    #: The norm a document describes; called with the document's schema version.
    read: Callable[[Mapping[str, Any], int], Norm]
    #: The document of a norm, in plain JSON containers.
    write: Callable[[Norm], dict[str, Any]]
    #: One entry of the constraint list.
    write_constraint: Callable[[NormConstraint], dict[str, Any]]
    #: A written document without its free-form parts (what the fingerprint covers).
    identity: Callable[[Mapping[str, Any]], dict[str, Any]]


V2 = Adapter(v2.VERSIONS, v2.read, v2.write, v2.write_constraint, v2.identity)
ADAPTERS: Mapping[int, Adapter] = dict.fromkeys(V2.versions, V2)


def adapter(schema_version: int) -> Adapter:
    """The adapter of a schema version."""
    try:
        return ADAPTERS[schema_version]
    except KeyError:
        raise NormError(f"no document format for norm schema_version {schema_version!r}; known: {sorted(ADAPTERS)}") from None


def schema_version_of(document: Mapping[str, Any]) -> int:
    """The ``schema_version`` of a document; refused when absent, not an integer, or unknown."""
    if "schema_version" not in document:
        raise NormError(f"norm document has no 'schema_version'; give one of {list(SCHEMA_VERSIONS)} (current: {SCHEMA_VERSION})")
    raw = document["schema_version"]
    if isinstance(raw, bool) or not isinstance(raw, int | str):
        raise NormError(f"norm schema_version {raw!r} is not an integer")
    try:
        version = int(raw)
    except ValueError:
        raise NormError(f"norm schema_version {raw!r} is not an integer") from None
    if version > SCHEMA_VERSION:
        raise NormError(f"norm schema_version {version} is newer than supported ({SCHEMA_VERSION})")
    if version not in ADAPTERS:
        raise NormError(f"norm schema_version {version} is not a known schema version; known: {sorted(ADAPTERS)}")
    return version


# ----------------------------------------------------------------------------- documents
def read(document: Mapping[str, Any]) -> Norm:
    """The norm a document describes. Malformed documents raise :class:`NormError`."""
    if not isinstance(document, Mapping):
        raise NormError(f"a norm must be a mapping, got {type(document).__name__}")
    version = schema_version_of(document)
    try:
        return adapter(version).read(document, version)
    except NormError:
        raise
    except (TypeError, KeyError, AttributeError, ValueError) as exc:
        raise NormError(f"malformed norm: {exc}") from exc


def write(norm: Norm) -> dict[str, Any]:
    """The document of ``norm`` in the format of its schema version."""
    return adapter(norm.schema_version).write(norm)


def write_constraint(nc: NormConstraint, schema_version: int = SCHEMA_VERSION) -> dict[str, Any]:
    """One entry of the constraint list of a document."""
    return adapter(schema_version).write_constraint(nc)


# ----------------------------------------------------------------------------- text and files
def _json_default(obj: Any) -> Any:
    out = jsonable(obj)
    if out is obj:
        raise NormError(f"norm metadata holds a value of type {type(obj).__name__} that cannot be serialised to JSON")
    return out


def _parse_json(text: str, source: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise NormError(f"{source} is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})") from exc


def dumps(norm: Norm, indent: int = 2) -> str:
    """The document as JSON text."""
    return json.dumps(write(norm), indent=indent, ensure_ascii=False, default=_json_default)


def dump(norm: Norm, path: str | Path, indent: int = 2) -> Path:
    """Write the JSON text and a final newline to ``path``."""
    p = Path(path)
    p.write_text(dumps(norm, indent) + "\n", encoding="utf-8")
    return p


def loads(text: str, *, source: str = "norm text") -> Norm:
    """The norm of a JSON text."""
    return read(_parse_json(text, source))


def load(path: str | Path) -> Norm:
    """The norm of a JSON file."""
    return loads(Path(path).read_text(encoding="utf-8"), source=str(path))


# ----------------------------------------------------------------------------- fingerprint
def fingerprint(norm: Norm) -> str:
    """SHA-256 of the canonical JSON of the document without its free-form parts (metadata, descriptions)."""
    format_ = adapter(norm.schema_version)
    canonical = json.dumps(
        format_.identity(format_.write(norm)), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_json_default
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "ADAPTERS",
    "V2",
    "Adapter",
    "adapter",
    "dump",
    "dumps",
    "fingerprint",
    "load",
    "loads",
    "read",
    "schema_version_of",
    "write",
    "write_constraint",
]
