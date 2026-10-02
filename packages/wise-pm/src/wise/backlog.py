"""The ranked slice backlog as a record: the table plus the specification that produced it.

:func:`wise.prioritize` returns a :class:`Backlog`. Its :attr:`~Backlog.table`
is the ranked frame (paper Sec. IV-E); its :attr:`~Backlog.spec` is a frozen
:class:`BacklogSpec` holding the ranking parameters. The specification is a
field of the record, not an entry of ``DataFrame.attrs``, so it survives
copying and pickling and is never lost in a pandas operation on the table.

Compatibility (decision record 0001, section 2): in 0.1.0 :func:`wise.prioritize`
returned the frame itself with the parameters in ``DataFrame.attrs``. Using a
:class:`Backlog` as that frame (indexing, ``len``, iteration, membership,
attribute access such as ``backlog.loc`` or ``backlog.attrs``) still works and
emits one :class:`DeprecationWarning` per use naming the replacement. A frame
handed out on that path carries the 0.1.0 ``attrs``; the record's own table does
not. Column assignment (``backlog[col] = ...``) reaches the table; any other
in-place change on that path (``backlog.loc[...] = ...``, ``inplace=True``)
acts on a copy and must be made on ``backlog.table``. The path is removed in 1.0.
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

__all__ = ["Backlog", "BacklogSpec", "legacy_frame"]

DATAFRAME_DEPRECATION = (
    "using the backlog returned by prioritize() as a DataFrame is deprecated and will be removed in 1.0; "
    "use Backlog.table for the ranked frame and Backlog.spec for its parameters"
)


@dataclass(frozen=True)
class BacklogSpec:
    """The parameters a backlog was ranked with (paper Sec. IV-E).

    Attributes
    ----------
    by
        Slice key column(s), in order.
    view
        The view whose scores were ranked; ``None`` when the input was a
        per-case frame rather than a :class:`~wise.scoring.ScoreResult`.
    gamma
        Shrinkage constant ``γ``.
    volume
        ``"cases"``, ``"exposure"`` or the per-case column summed as volume.
    baseline
        The reference score as requested; ``None`` means the global mean of
        the scored cases.
    global_mean
        The reference score ``μ̄`` actually used (``baseline`` when given).
    min_cases
        Slices with fewer scored cases were dropped from the ranking.
    z
        Multiplier of the conservative lower bound; ``None`` when no bound was
        computed.
    norm_fingerprint
        Fingerprint of the norm the scores came from; ``None`` for a per-case
        frame.
    score_col
        The score column ranked (``"score"`` for a result).
    """

    by: tuple[str, ...]
    view: str | None
    gamma: float
    volume: str
    baseline: float | None
    global_mean: float
    min_cases: int
    z: float | None
    norm_fingerprint: str | None
    score_col: str = "score"

    def to_dict(self) -> dict[str, Any]:
        """The specification as a JSON-ready mapping (``by`` as a list)."""
        return {
            "by": list(self.by),
            "view": self.view,
            "gamma": self.gamma,
            "volume": self.volume,
            "baseline": self.baseline,
            "global_mean": self.global_mean,
            "min_cases": self.min_cases,
            "z": self.z,
            "norm_fingerprint": self.norm_fingerprint,
            "score_col": self.score_col,
        }

    def _legacy_attrs(self) -> dict[str, Any]:
        """The ``DataFrame.attrs`` that 0.1.0 attached to a backlog (``baseline`` is the realised ``μ̄``)."""
        return {"view": self.view, "gamma": self.gamma, "baseline": self.global_mean, "volume": self.volume, "by": list(self.by)}


@dataclass(frozen=True, eq=False)
class Backlog:
    """A ranked slice backlog: :attr:`table` and the :attr:`spec` it was ranked with.

    ``table`` is indexed by the slice keys (unless ``prioritize(as_index=False)``),
    sorted by ``stable_PI``, with the columns documented in :func:`wise.prioritize`.
    """

    table: pd.DataFrame
    spec: BacklogSpec

    def __repr__(self) -> str:
        s = self.spec
        return (
            f"Backlog({len(self.table):,} slices by {list(s.by)}, view={s.view!r}, gamma={s.gamma:g}, "
            f"volume={s.volume!r}, global_mean={s.global_mean:.6g})"
        )

    def __bool__(self) -> bool:
        return True

    if not TYPE_CHECKING:
        # The deprecated DataFrame path; invisible to type checkers so typed callers move to ``table``.

        def _legacy_frame(self) -> pd.DataFrame:
            warnings.warn(DATAFRAME_DEPRECATION, DeprecationWarning, stacklevel=3)
            return legacy_frame(self)

        def __getattr__(self, name: str) -> Any:
            if name.startswith("_") or name in ("table", "spec"):  # pickling and copying probe before the fields exist
                raise AttributeError(name)
            return getattr(self._legacy_frame(), name)

        def __getitem__(self, key: Any) -> Any:
            return self._legacy_frame()[key]

        def __setitem__(self, key: Any, value: Any) -> None:
            warnings.warn(DATAFRAME_DEPRECATION, DeprecationWarning, stacklevel=2)
            self.table[key] = value

        def __len__(self) -> int:
            return len(self._legacy_frame())

        def __iter__(self) -> Any:
            return iter(self._legacy_frame())

        def __contains__(self, key: Any) -> bool:
            return key in self._legacy_frame()

        def __array__(self, dtype: Any = None, copy: bool | None = None) -> Any:
            # numpy looks this up on the type; without it np.asarray would build an array of column names
            return self._legacy_frame().__array__(dtype, copy=copy)


def legacy_frame(backlog: Backlog) -> pd.DataFrame:
    """The 0.1.0 form of a backlog: a shallow copy of :attr:`Backlog.table` with the
    ranking parameters in ``DataFrame.attrs`` (``view``, ``gamma``, ``baseline`` = the
    realised global mean ``μ̄``, ``volume``, ``by``). For the compatibility paths
    only; removed in 1.0."""
    frame = backlog.table.copy(deep=False)
    attrs: dict[Hashable | None, Any] = dict(backlog.spec._legacy_attrs().items())
    frame.attrs = attrs
    return frame
