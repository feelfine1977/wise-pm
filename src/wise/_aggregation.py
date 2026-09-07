"""The shared numerical kernel behind every WISE score.

There is exactly one applicability-aware scoring formula in this library, and
it lives here. :func:`wise.scoring.score` uses it for cases; the object-centric
evaluator in :mod:`wise.oc.evaluation` uses it for typed assessment units. That
is the whole point of the module: a second implementation would drift, and two
numbers called "the score" that were computed differently are worse than one
number computed once.

The kernel is deliberately *pure array code*. It knows nothing about
:class:`~wise.log.EventLog`, :class:`~wise.norm.Norm`,
:class:`~wise.oc.model.OCEventLog` or assessment units. It takes

* ``violations`` — rows × checks, ``ν ∈ [0, 1]`` and ``NaN`` where the check
  was not applicable or not evaluable,
* ``weights`` — the raw weight ``w_c`` of every check, in column order,
* ``layers`` — a :class:`LayerAssignment` saying which layer each check
  belongs to, in a fixed layer order,
* ``row_ids`` — the identities of the rows, carried through so that a caller
  cannot silently misalign a result with its units,
* ``mode`` — ``"flat"`` or ``"layer_balanced"``, always explicit,

and returns scores, per-layer contributions and per-row effective weights with
the same missing-value conventions as before:

``NaN`` in ``violations`` means *not evaluated*, never *satisfied*. A row for
which no positively weighted check applies is **unscored**: its score, its
contributions and every one of its effective weights are ``NaN``, not zero.

The two modes
-------------
``"flat"`` renormalises the raw weights over the applicable set. In
``"layer_balanced"`` each layer first averages its own applicable checks with
their within-layer weights and the applicable layers are then averaged with
the layer weights, so an applicable layer keeps its full weight regardless of
how many of its checks could be evaluated. The two coincide when every check
of a layer applies.

>>> import numpy as np
>>> layers = LayerAssignment(("c1", "c2"), ("A", "B"), ("A", "B"))
>>> out = aggregate(np.array([[0.0, 1.0]]), np.array([0.5, 0.5]), layers, "flat")
>>> float(out.scores[0]), [float(x) for x in out.contributions[0]]
(0.5, [0.0, 0.5])
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import NormError

#: The two aggregation modes. Identical to :data:`wise.norm.SCORING_MODES`;
#: repeated here so that the kernel does not import the norm package.
AGGREGATION_MODES = ("layer_balanced", "flat")


@dataclass(frozen=True)
class LayerAssignment:
    """Which layer every check belongs to, with both orders pinned.

    ``constraint_ids`` fixes the column order of the violation matrix and the
    weight vector; ``layer_ids`` fixes the column order of the contributions.
    Both are part of the result's meaning, so both are data here rather than
    something the kernel derives from a set.

    >>> a = LayerAssignment(("c1", "c2", "c3"), ("fast", "correct"), ("fast", "correct", "correct"))
    >>> a.members("correct").tolist()
    [1, 2]
    >>> a.onehot.tolist()
    [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    """

    constraint_ids: tuple[str, ...]
    layer_ids: tuple[str, ...]
    layer_of: tuple[str, ...]

    _members: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False, compare=False)
    _onehot: np.ndarray = field(default_factory=lambda: np.empty((0, 0)), init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraint_ids", tuple(str(c) for c in self.constraint_ids))
        object.__setattr__(self, "layer_ids", tuple(str(layer) for layer in self.layer_ids))
        object.__setattr__(self, "layer_of", tuple(str(layer) for layer in self.layer_of))
        if len(self.layer_of) != len(self.constraint_ids):
            raise NormError(
                f"layer assignment: {len(self.constraint_ids)} check(s) but {len(self.layer_of)} layer label(s); "
                "every check must name exactly one layer"
            )
        if len(set(self.constraint_ids)) != len(self.constraint_ids):
            raise NormError("layer assignment: check ids must be unique")
        if len(set(self.layer_ids)) != len(self.layer_ids):
            raise NormError("layer assignment: layer ids must be unique")
        unknown = sorted(set(self.layer_of) - set(self.layer_ids))
        if unknown:
            raise NormError(f"layer assignment: check(s) placed in undeclared layer(s) {unknown}")
        members = {
            layer: np.array([i for i, owner in enumerate(self.layer_of) if owner == layer], dtype=int) for layer in self.layer_ids
        }
        object.__setattr__(self, "_members", members)
        onehot = np.zeros((len(self.constraint_ids), len(self.layer_ids)), dtype=float)
        for column, layer in enumerate(self.layer_ids):
            onehot[members[layer], column] = 1.0
        object.__setattr__(self, "_onehot", onehot)

    @classmethod
    def from_mapping(
        cls,
        constraint_ids: Sequence[str],
        layer_of: Mapping[Any, Any],
        layer_ids: Sequence[str] | None = None,
    ) -> LayerAssignment:
        """Build from ``{check_id: layer_id}``, keeping the given orders.

        ``layer_ids`` defaults to first-seen order, which is only ever right
        when the caller has no declared layer order of its own; both
        :class:`wise.norm.Norm` and :class:`wise.oc.evaluation.ObjectNorm` pass
        theirs explicitly.
        """
        cids = [str(c) for c in constraint_ids]
        try:
            owners = [str(layer_of[c]) for c in cids]
        except KeyError as exc:
            raise NormError(f"layer assignment: check {exc.args[0]!r} has no layer") from None
        layers = tuple(str(layer) for layer in layer_ids) if layer_ids is not None else tuple(dict.fromkeys(owners))
        return cls(tuple(cids), layers, tuple(owners))

    def __len__(self) -> int:
        return len(self.constraint_ids)

    def members(self, layer: str) -> np.ndarray:
        """Column positions of the checks in one layer, in column order."""
        try:
            return self._members[str(layer)]
        except KeyError:
            raise NormError(f"unknown layer {layer!r}; declared layers are {list(self.layer_ids)}") from None

    @property
    def onehot(self) -> np.ndarray:
        """Checks × layers indicator matrix, used to fold penalties into layers."""
        return self._onehot


@dataclass(frozen=True)
class Aggregation:
    """What the kernel computed, with the identities it computed it for.

    Every array is plain NumPy: the kernel does not decide whether its caller
    wants a frame indexed by case id or by unit id.
    """

    row_ids: tuple[Any, ...]
    constraint_ids: tuple[str, ...]
    layer_ids: tuple[str, ...]
    mode: str
    scores: np.ndarray
    contributions: np.ndarray
    effective_weights: np.ndarray
    penalties: np.ndarray
    unscored: np.ndarray

    @property
    def n_scored(self) -> int:
        """Rows that carry a score."""
        return int((~self.unscored).sum())

    def decomposition_error(self) -> float:
        """Max ``|Σ_λ Δ_λ − (1 − S)|`` over scored rows; 0.0 when none are."""
        if not len(self.scores) or self.unscored.all():
            return 0.0
        keep = ~self.unscored
        lhs = self.contributions[keep].sum(axis=1)
        rhs = 1.0 - self.scores[keep]
        return float(np.max(np.abs(lhs - rhs))) if len(lhs) else 0.0


def effective_weights(applicable: np.ndarray, weights: np.ndarray, layers: LayerAssignment, mode: str) -> np.ndarray:
    """Per-row effective weights ``w̃_{c,row}``; ``NaN`` for an unscored row.

    ``applicable`` is the 0/1 matrix of *evaluated* checks — in scope **and**
    evaluable. Rows of scored units sum to 1.
    """
    if mode == "flat":
        w_app = applicable @ weights
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(w_app[:, None] > 0, applicable * weights[None, :] / w_app[:, None], np.nan)
    if mode != "layer_balanced":
        raise NormError(f"mode must be one of {AGGREGATION_MODES}, got {mode!r}")
    return _layer_balanced(applicable, weights, layers)


def _layer_balanced(applicable: np.ndarray, weights: np.ndarray, layers: LayerAssignment) -> np.ndarray:
    n = applicable.shape[0]
    w_eff = np.zeros_like(applicable)
    a_app = np.zeros(n)
    for layer in layers.layer_ids:
        idx = layers.members(layer)
        a = float(weights[idx].sum()) if len(idx) else 0.0
        if len(idx) == 0 or a <= 0:
            continue
        b_app = applicable[:, idx] * weights[idx][None, :]
        b_sum = b_app.sum(axis=1)
        layer_app = b_sum > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            w_eff[:, idx] = np.where(b_sum[:, None] > 0, b_app / b_sum[:, None], 0.0) * a
        a_app += layer_app * a
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a_app[:, None] > 0, w_eff / a_app[:, None], np.nan)


def aggregate(
    violations: np.ndarray,
    weights: np.ndarray,
    layers: LayerAssignment,
    mode: str,
    *,
    row_ids: Sequence[Any] | None = None,
) -> Aggregation:
    """Score one view: violations and raw weights in, scores and layers out.

    ``violations`` carries ``NaN`` where a check was not applicable or not
    evaluable. That is not a zero: the check is dropped from the row's
    normalisation instead of contributing a satisfied observation.
    """
    V = np.asarray(violations, dtype=float)
    if V.ndim != 2:
        raise NormError(f"violations must be a rows × checks matrix, got {V.ndim} dimension(s)")
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(layers),):
        raise NormError(f"weights must have one entry per check ({len(layers)}), got {w.shape}")
    if V.shape[1] != len(layers):
        raise NormError(f"violations has {V.shape[1]} column(s) but the layer assignment declares {len(layers)} check(s)")
    ids = tuple(range(V.shape[0])) if row_ids is None else tuple(row_ids)
    if len(ids) != V.shape[0]:
        raise NormError(f"row_ids has {len(ids)} entr(ies) for {V.shape[0]} row(s); a misaligned result is a wrong result")

    missing = np.isnan(V)
    M = (~missing).astype(float)
    V0 = np.where(missing, 0.0, V)
    w_eff = effective_weights(M, w, layers, mode)
    penalties = np.nan_to_num(w_eff) * V0
    unscored = np.all(np.isnan(w_eff), axis=1)
    scores = 1.0 - penalties.sum(axis=1)
    scores[unscored] = np.nan
    contributions = penalties @ layers.onehot
    contributions[unscored, :] = np.nan
    return Aggregation(
        row_ids=ids,
        constraint_ids=layers.constraint_ids,
        layer_ids=layers.layer_ids,
        mode=mode,
        scores=scores,
        contributions=contributions,
        effective_weights=w_eff,
        penalties=penalties,
        unscored=unscored,
    )


__all__ = ["AGGREGATION_MODES", "Aggregation", "LayerAssignment", "aggregate", "effective_weights"]
