"""Immutable run records and canonical fingerprints.

A norm fingerprint identifies a *configuration*, not a run. Reproducing a
score also needs the data that was read, how it was prepared, which mode was
actually used (a call can override the norm's default), which views were
selected, how the observation window was resolved, which derived recipes and
frozen calibrations were applied, and which library versions computed it.
:class:`RunManifest` records all of that.

Two identities are deliberately separate:

``config_fingerprint()``
    deterministic over the configuration only — two runs of the same
    configuration on the same input share it.
``run_id`` / ``created_at``
    per execution — they differ every time, and they are excluded from the
    configuration fingerprint.

Nothing here reads Git or hashes the event data. The commit is ``None``
unless a caller supplies it (an installed wheel has no repository to ask),
and a data digest is computed only by an explicit call to
:func:`fingerprint_events`, which records the algorithm and the
canonicalisation it used.

A manifest produced by :func:`wise.score` describes a **score** run. The
grouping, comparator, minimum support and gamma of a backlog are not known at
that point, so :meth:`RunManifest.finalize` returns a *new* manifest at stage
``"backlog"``; a score-only snapshot is never labelled a complete backlog run.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .._version import __version__
from ..errors import EvidenceError

if TYPE_CHECKING:  # pragma: no cover
    from ..log import EventLog

MANIFEST_SCHEMA_VERSION = "wise-run/1"

#: Stage of a run record. A score run knows nothing about the backlog
#: configuration, so the two are different records.
STAGES = ("score", "backlog")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(obj: Any) -> Any:
    """Plain JSON types, with explicit nulls and normalised timestamps."""
    if obj is None or isinstance(obj, bool | int | str):
        return obj
    if isinstance(obj, float):
        if not np.isfinite(obj):
            raise EvidenceError(f"non-finite number {obj!r} cannot be exported; use null with a reason instead")
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return _canonical(float(obj))
    if isinstance(obj, pd.Timestamp | datetime):
        ts = pd.Timestamp(obj)
        return None if pd.isna(ts) else str(ts.isoformat())
    if obj is pd.NaT or obj is pd.NA:
        return None
    if isinstance(obj, Mapping):
        return {str(k): _canonical(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        return [_canonical(v) for v in obj]
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return _canonical(obj.to_dict())
    raise EvidenceError(
        f"no conversion policy for {type(obj).__name__} in a run record; convert it explicitly rather than relying on str()"
    )


def _digest(obj: Any) -> str:
    canonical = json.dumps(_canonical(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def only_fields(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    """``data`` restricted to the dataclass's own fields.

    ``to_dict`` also exports derived values — ``config_fingerprint``,
    ``evaluated_share``, ``witnesses_truncated``, ``complete``. They are
    *outputs* of a record, never inputs: accepting one back would let an
    exported file assert a fingerprint its own fields do not produce, which is
    the opposite of a verifiable identity. They are dropped here and recomputed.
    """
    names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    return {k: v for k, v in data.items() if k in names}


# --------------------------------------------------------------------- parts
@dataclass(frozen=True)
class EnvironmentInfo:
    """Which code produced a run. ``git_commit`` is never read from Git."""

    wise_version: str = __version__
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    numpy_version: str = field(default_factory=lambda: str(np.__version__))
    pandas_version: str = field(default_factory=lambda: str(pd.__version__))
    platform: str = field(default_factory=platform.platform)
    git_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "wise_version": self.wise_version,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "pandas_version": self.pandas_version,
            "platform": self.platform,
            "git_commit": self.git_commit,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EnvironmentInfo:
        """Rebuild from :meth:`to_dict`; the environment of the *export* is not substituted."""
        return cls(**only_fields(cls, data))


@dataclass(frozen=True)
class InputIdentity:
    """What was read, and how far its identity is actually established."""

    n_events: int
    n_cases: int
    event_columns: tuple[str, ...]
    case_columns: tuple[str, ...]
    event_id_col: str | None
    source_identity_available: bool
    snapshot_fingerprint: str
    source: str | None = None
    data_digest: str | None = None
    digest_algorithm: str | None = None
    digest_canonicalisation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "n_events": int(self.n_events),
            "n_cases": int(self.n_cases),
            "event_columns": list(self.event_columns),
            "case_columns": list(self.case_columns),
            "event_id_col": self.event_id_col,
            "source_identity_available": bool(self.source_identity_available),
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "data_digest": self.data_digest,
            "digest_algorithm": self.digest_algorithm,
            "digest_canonicalisation": self.digest_canonicalisation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InputIdentity:
        """Rebuild from :meth:`to_dict`."""
        payload = only_fields(cls, data)
        for name in ("event_columns", "case_columns"):
            if name in payload:
                payload[name] = tuple(str(c) for c in payload[name])
        return cls(**payload)


@dataclass(frozen=True)
class ObservationScope:
    """The observation window actually used, and the horizons resolved from it.

    ``resolved_horizons`` is ``None`` when no horizon was resolved during the
    run (no censoring policy applied, or evidence capture was off); it is a
    mapping of constraint id to ISO timestamp when horizons were resolved.
    """

    window_source: str
    window_start: str | None
    window_end: str | None
    quantile: float
    missing_timestamps: str
    timezone: str | None = None
    resolved_horizons: dict[str, str] | None = None

    @property
    def policy_id(self) -> str:
        """A short, stable identifier of the observation policy."""
        if self.window_source == "explicit":
            return f"explicit-window[{self.window_start}..{self.window_end}]"
        return f"derived-quantile-window[q={self.quantile:g}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "window_source": self.window_source,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "quantile": float(self.quantile),
            "missing_timestamps": self.missing_timestamps,
            "timezone": self.timezone,
            "resolved_horizons": None if self.resolved_horizons is None else dict(self.resolved_horizons),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ObservationScope:
        """Rebuild from :meth:`to_dict`; ``policy_id`` is derived, not read."""
        return cls(**only_fields(cls, data))


@dataclass(frozen=True)
class PriorityConfig:
    """The backlog configuration — unknown while only scoring has happened."""

    grouping: tuple[str, ...]
    view: str | None = None
    volume: str = "cases"
    comparator: str = "current_population_mean"
    comparator_value: float | None = None
    min_cases: int = 1
    gamma: float = 0.0
    z: float | None = None
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "grouping": list(self.grouping),
            "view": self.view,
            "volume": self.volume,
            "comparator": self.comparator,
            "comparator_value": self.comparator_value,
            "min_cases": int(self.min_cases),
            "gamma": float(self.gamma),
            "z": self.z,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PriorityConfig:
        """Rebuild from :meth:`to_dict`."""
        payload = only_fields(cls, data)
        payload["grouping"] = tuple(str(g) for g in payload.get("grouping", ()))
        payload["diagnostics"] = tuple(str(d) for d in payload.get("diagnostics", ()))
        return cls(**payload)


# ------------------------------------------------------------------ manifest
@dataclass(frozen=True)
class RunManifest:
    """A complete record of one run: configuration, input, environment, identity."""

    run_id: str
    mode: str
    mode_source: str
    norm_fingerprint: str
    input: InputIdentity
    observation: ObservationScope
    preprocessing: dict[str, Any]
    views: tuple[str, ...] = ()
    norm_name: str = ""
    norm_version: str = ""
    norm_default_mode: str = ""
    derived_recipes: tuple[dict[str, Any], ...] = ()
    calibrations: tuple[dict[str, Any], ...] = ()
    environment: EnvironmentInfo = field(default_factory=EnvironmentInfo)
    stage: str = "score"
    priority: PriorityConfig | None = None
    unit_type: str = "case"
    created_at: str = field(default_factory=_now)
    schema_version: str = MANIFEST_SCHEMA_VERSION
    result_fingerprint: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise EvidenceError(f"stage must be one of {STAGES}, got {self.stage!r}")
        if self.mode_source not in ("norm_default", "call_override"):
            raise EvidenceError(f"mode_source must be 'norm_default' or 'call_override', got {self.mode_source!r}")
        if self.stage == "backlog" and self.priority is None:
            raise EvidenceError("a backlog manifest needs its priority configuration; use RunManifest.finalize")
        object.__setattr__(self, "views", tuple(str(v) for v in self.views))
        object.__setattr__(self, "preprocessing", dict(self.preprocessing))
        object.__setattr__(self, "derived_recipes", tuple(dict(r) for r in self.derived_recipes))
        object.__setattr__(self, "calibrations", tuple(dict(c) for c in self.calibrations))
        object.__setattr__(self, "notes", tuple(str(n) for n in self.notes))

    # ----------------------------------------------------------- identities
    def config_fingerprint(self) -> str:
        """SHA-256 over the deterministic configuration, excluding this execution.

        The run id, the timestamp, the machine and the result are *not* part of
        it; the norm, the mode actually used, the views, the input identity,
        the preparation options, the observation policy, the recipes and the
        frozen calibrations are. The library version is included, because the
        same configuration under a different library is a different experiment.
        """
        return _digest(
            {
                "schema_version": self.schema_version,
                "mode": self.mode,
                "views": list(self.views),
                "unit_type": self.unit_type,
                "norm_fingerprint": self.norm_fingerprint,
                "input": self.input.to_dict(),
                "preprocessing": self.preprocessing,
                "observation": self.observation.to_dict(),
                "derived_recipes": list(self.derived_recipes),
                "calibrations": list(self.calibrations),
                "priority": None if self.priority is None else self.priority.to_dict(),
                "wise_version": self.environment.wise_version,
            }
        )

    @property
    def is_complete_backlog_run(self) -> bool:
        """True only for a manifest finalised with a priority configuration."""
        return self.stage == "backlog" and self.priority is not None

    def finalize(
        self,
        *,
        grouping: Sequence[str],
        view: str | None = None,
        volume: str = "cases",
        comparator: str = "current_population_mean",
        comparator_value: float | None = None,
        min_cases: int = 1,
        gamma: float = 0.0,
        z: float | None = None,
        diagnostics: Iterable[str] = (),
    ) -> RunManifest:
        """A new manifest at stage ``"backlog"`` with the priority configuration.

        The score manifest is left untouched: the two are different records of
        different things.
        """
        config = PriorityConfig(
            grouping=tuple(str(g) for g in grouping),
            view=view,
            volume=volume,
            comparator=comparator,
            comparator_value=None if comparator_value is None else float(comparator_value),
            min_cases=int(min_cases),
            gamma=float(gamma),
            z=z,
            diagnostics=tuple(str(d) for d in diagnostics),
        )
        return replace(self, stage="backlog", priority=config)

    def with_data_digest(self, digest: str, *, algorithm: str, canonicalisation: str) -> RunManifest:
        """A copy whose input identity carries an explicitly computed digest."""
        return replace(
            self,
            input=replace(self.input, data_digest=digest, digest_algorithm=algorithm, digest_canonicalisation=canonicalisation),
        )

    def with_commit(self, commit: str | None) -> RunManifest:
        """A copy recording a caller-supplied commit; Git is never read here."""
        return replace(self, environment=replace(self.environment, git_commit=commit))

    # ------------------------------------------------------------- export
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "stage": self.stage,
            "config_fingerprint": self.config_fingerprint(),
            "unit_type": self.unit_type,
            "mode": self.mode,
            "mode_source": self.mode_source,
            "norm_default_mode": self.norm_default_mode,
            "views": list(self.views),
            "norm_name": self.norm_name,
            "norm_version": self.norm_version,
            "norm_fingerprint": self.norm_fingerprint,
            "input": self.input.to_dict(),
            "preprocessing": dict(self.preprocessing),
            "observation": self.observation.to_dict(),
            "derived_recipes": [dict(r) for r in self.derived_recipes],
            "calibrations": [dict(c) for c in self.calibrations],
            "priority": None if self.priority is None else self.priority.to_dict(),
            "environment": self.environment.to_dict(),
            "result_fingerprint": self.result_fingerprint,
            "is_complete_backlog_run": self.is_complete_backlog_run,
            "notes": list(self.notes),
        }

    def to_json(self, indent: int | None = 2) -> str:
        """UTF-8 JSON with explicit nulls, normalised timestamps and ``allow_nan=False``."""
        return json.dumps(_canonical(self.to_dict()), indent=indent, ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunManifest:
        """Rebuild a run record from :meth:`to_dict`.

        The two derived identities — ``config_fingerprint`` and
        ``is_complete_backlog_run`` — are recomputed from the fields rather
        than read, so a rebuilt manifest cannot claim a fingerprint its own
        configuration does not produce.
        """
        version = str(data.get("schema_version", MANIFEST_SCHEMA_VERSION))
        if version != MANIFEST_SCHEMA_VERSION:
            raise EvidenceError(f"unsupported run schema version {version!r}; this library reads {MANIFEST_SCHEMA_VERSION!r}")
        payload = only_fields(cls, data)
        payload["input"] = InputIdentity.from_dict(data["input"])
        payload["observation"] = ObservationScope.from_dict(data["observation"])
        payload["environment"] = EnvironmentInfo.from_dict(data.get("environment", {}))
        priority = data.get("priority")
        payload["priority"] = None if priority is None else PriorityConfig.from_dict(priority)
        payload["views"] = tuple(str(v) for v in data.get("views", ()))
        payload["derived_recipes"] = tuple(dict(r) for r in data.get("derived_recipes", ()))
        payload["calibrations"] = tuple(dict(c) for c in data.get("calibrations", ()))
        payload["notes"] = tuple(str(n) for n in data.get("notes", ()))
        payload["preprocessing"] = dict(data.get("preprocessing", {}))
        return cls(**payload)

    def __repr__(self) -> str:
        return (
            f"RunManifest({self.run_id!r}, stage={self.stage!r}, mode={self.mode!r} "
            f"({self.mode_source}), views={list(self.views)}, config={self.config_fingerprint()[:12]}…)"
        )


def new_run_id(prefix: str = "run") -> str:
    """A per-execution identity. Never a content fingerprint."""
    return f"{prefix}-{uuid.uuid4().hex}"


# ------------------------------------------------------------- explicit hashing
def fingerprint_events(
    log: EventLog,
    *,
    columns: Sequence[str] | None = None,
    algorithm: str = "sha256",
    chunk_size: int = 50_000,
) -> tuple[str, str]:
    """Digest of the event data — **explicit, never automatic**.

    Returns ``(digest, canonicalisation)``. The canonicalisation is recorded
    with the digest because the digest means nothing without it: the events are
    written as UTF-8 CSV in their stored row order, without the index, with
    ISO-8601 timestamps, in chunks of ``chunk_size`` rows.

    Stored row order is preserved on purpose. Sorting the data first would give
    an order-insensitive digest and would erase a real difference, because
    timestamp ties are broken by input row order unless ``order_col`` was given.
    """
    if algorithm not in hashlib.algorithms_available:
        raise EvidenceError(f"unknown hash algorithm {algorithm!r}")
    frame = log.events if columns is None else log.events[list(columns)]
    digest = hashlib.new(algorithm)
    header = ",".join(str(c) for c in frame.columns) + "\n"
    digest.update(header.encode("utf-8"))
    for start in range(0, len(frame), max(int(chunk_size), 1)):
        block = frame.iloc[start : start + chunk_size]
        text = block.to_csv(index=False, header=False, date_format="%Y-%m-%dT%H:%M:%S.%f%z")
        digest.update(text.encode("utf-8"))
    canonicalisation = (
        "UTF-8 CSV of the selected columns in stored row order, no index, header row first, timestamps as %Y-%m-%dT%H:%M:%S.%f%z"
    )
    return digest.hexdigest(), canonicalisation


def fingerprint_result(scores: pd.DataFrame, violations: pd.DataFrame) -> str:
    """Digest of the numeric result — explicit, and not part of the configuration.

    NaN is part of the result (it means *unscored* or *not evaluated*), so it is
    hashed as the bit pattern of the array rather than dropped.
    """
    parts = [
        ",".join(str(c) for c in scores.columns).encode("utf-8"),
        np.ascontiguousarray(scores.to_numpy(dtype=float)).tobytes(),
        ",".join(str(c) for c in violations.columns).encode("utf-8"),
        np.ascontiguousarray(violations.to_numpy(dtype=float)).tobytes(),
    ]
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()
