"""The shared numerical kernel is the *same* arithmetic, bit for bit (O03).

`src/wise/_aggregation.py` was extracted from `wise.scoring.score` so that the
case evaluator and the object evaluator cannot drift apart. An extraction that
moved a number would be a regression dressed up as a refactor, so the claim
under test here is exact equality, not closeness:

* `_reference_effective_weights` and `_reference_view` below are literal
  transcriptions of the pre-extraction code at the reviewed base
  `df5db50b839cc124b489a269894f5a2bfe7dc634`. Every assertion compares the
  kernel against them with `atol=0, rtol=0` and an identical NaN mask.
* The awkward shapes are deliberate: no applicable check at all, a whole layer
  inapplicable, a zero-weight view, one case, no case, and violations that are
  exactly 0 or exactly 1.
* `score` itself still produces the frames it produced before — the stage-0
  regression contract pins that from the outside; these tests pin the
  intermediate arrays as well.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import wise
from wise import scoring
from wise._aggregation import AGGREGATION_MODES, Aggregation, LayerAssignment, aggregate, effective_weights
from wise.errors import NormError

MODES = ("flat", "layer_balanced")


# --------------------------------------------------------- the reviewed base, transcribed
def _reference_layer_balanced(M: np.ndarray, w: np.ndarray, norm: wise.Norm) -> np.ndarray:
    n = M.shape[0]
    cids = norm.constraint_ids
    layer_of = norm.layer_of
    w_eff = np.zeros_like(M)
    a_app = np.zeros(n)
    for layer in norm.layer_ids:
        idx = np.array([i for i, cid in enumerate(cids) if layer_of[cid] == layer], dtype=int)
        a = float(w[idx].sum()) if len(idx) else 0.0
        if len(idx) == 0 or a <= 0:
            continue
        b_app = M[:, idx] * w[idx][None, :]
        b_sum = b_app.sum(axis=1)
        layer_app = b_sum > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            w_eff[:, idx] = np.where(b_sum[:, None] > 0, b_app / b_sum[:, None], 0.0) * a
        a_app += layer_app * a
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a_app[:, None] > 0, w_eff / a_app[:, None], np.nan)


def _reference_effective_weights(M: np.ndarray, w: np.ndarray, norm: wise.Norm, mode: str) -> np.ndarray:
    if mode == "flat":
        w_app = M @ w
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(w_app[:, None] > 0, M * w[None, :] / w_app[:, None], np.nan)
    return _reference_layer_balanced(M, w, norm)


def _reference_view(V: pd.DataFrame, w: np.ndarray, norm: wise.Norm, mode: str):
    """One view's scores and contributions, exactly as ``score`` computed them."""
    M = V.notna().to_numpy(dtype=float)
    V0 = V.fillna(0.0).to_numpy(dtype=float)
    cids = list(V.columns)
    layer_ids = norm.layer_ids
    onehot = np.array([[1.0 if norm.layer_of[cid] == layer else 0.0 for layer in layer_ids] for cid in cids])
    w_eff = _reference_effective_weights(M, w, norm, mode)
    penalty_c = np.nan_to_num(w_eff) * V0
    unscored = np.all(np.isnan(w_eff), axis=1)
    s = 1.0 - penalty_c.sum(axis=1)
    s[unscored] = np.nan
    contrib = penalty_c @ onehot
    contrib[unscored, :] = np.nan
    return s, contrib, w_eff, penalty_c, unscored


def identical(a: np.ndarray, b: np.ndarray) -> None:
    """Same shape, same NaN mask, and bit-identical finite entries."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.array_equal(np.isnan(a), np.isnan(b)), "the NaN mask moved"
    np.testing.assert_array_max_ulp(a[~np.isnan(a)], b[~np.isnan(b)], maxulp=0)


def layers_of(norm: wise.Norm) -> LayerAssignment:
    return LayerAssignment.from_mapping(norm.constraint_ids, norm.layer_of, norm.layer_ids)


def awkward_matrices(n_constraints: int) -> list[np.ndarray]:
    """Shapes chosen because they are where a scoring formula goes wrong."""
    rng = np.random.default_rng(20260907)
    out: list[np.ndarray] = [
        np.full((1, n_constraints), np.nan),  # nothing applicable at all
        np.zeros((1, n_constraints)),  # everything applicable and satisfied
        np.ones((1, n_constraints)),  # everything applicable and maximally violated
        np.empty((0, n_constraints)),  # no rows
    ]
    for _ in range(40):
        V = rng.random((7, n_constraints))
        V[rng.random((7, n_constraints)) < 0.35] = np.nan
        out.append(V)
    for column in range(n_constraints):  # exactly one check evaluated
        V = np.full((3, n_constraints), np.nan)
        V[:, column] = [0.0, 0.5, 1.0]
        out.append(V)
    return out


# ------------------------------------------------------------------ the kernel itself
@pytest.mark.parametrize("mode", MODES)
def test_the_kernel_reproduces_the_reviewed_base_arithmetic_bit_for_bit(p2p_norm, mode):
    layers = layers_of(p2p_norm)
    for view in p2p_norm.view_names:
        w = p2p_norm.weight_vector(view).to_numpy(dtype=float)
        for V in awkward_matrices(len(p2p_norm.constraint_ids)):
            frame = pd.DataFrame(V, columns=p2p_norm.constraint_ids)
            s, contrib, w_eff, penalties, unscored = _reference_view(frame, w, p2p_norm, mode)
            # exactly the array ``score`` hands the kernel, layout included
            out = aggregate(frame.to_numpy(dtype=float), w, layers, mode)
            identical(out.scores, s)
            identical(out.contributions, contrib)
            identical(out.effective_weights, w_eff)
            identical(out.penalties, penalties)
            assert np.array_equal(out.unscored, unscored)


@pytest.mark.parametrize("mode", MODES)
def test_the_private_scoring_seam_still_answers_for_itself(p2p_norm, mode):
    """``scoring._effective_weights`` is what the stage-0 contract calls."""
    w = p2p_norm.weight_vector("Finance").to_numpy(dtype=float)
    for V in awkward_matrices(len(p2p_norm.constraint_ids)):
        M = (~np.isnan(V)).astype(float)
        identical(scoring._effective_weights(M, w, p2p_norm, mode), _reference_effective_weights(M, w, p2p_norm, mode))
        identical(effective_weights(M, w, layers_of(p2p_norm), mode), _reference_effective_weights(M, w, p2p_norm, mode))


def test_the_layer_balanced_helper_kept_its_name_and_its_answer(p2p_norm):
    M = np.array([[1.0, 1.0, 1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    w = p2p_norm.weight_vector("Logistics").to_numpy(dtype=float)
    identical(scoring._layer_balanced_weights(M, w, p2p_norm), _reference_layer_balanced(M, w, p2p_norm))


def test_a_zero_weight_view_leaves_every_row_unscored(p2p_norm):
    """No positively weighted check applies, so nothing is scored — not zero."""
    layers = layers_of(p2p_norm)
    w = np.zeros(len(p2p_norm.constraint_ids))
    V = np.zeros((4, len(p2p_norm.constraint_ids)))
    for mode in MODES:
        out = aggregate(V, w, layers, mode)
        assert out.unscored.all()
        assert np.isnan(out.scores).all()
        assert np.isnan(out.contributions).all()
        assert out.n_scored == 0


def test_missing_is_dropped_from_the_normalisation_and_never_counted_as_satisfied():
    """The one semantic that a shared kernel must not lose."""
    layers = LayerAssignment(("c1", "c2"), ("L",), ("L", "L"))
    w = np.array([0.5, 0.5])
    missing = aggregate(np.array([[1.0, np.nan]]), w, layers, "flat")
    satisfied = aggregate(np.array([[1.0, 0.0]]), w, layers, "flat")
    assert float(missing.scores[0]) == 0.0, "the evaluated check carries the whole weight"
    assert float(satisfied.scores[0]) == 0.5, "a satisfied check halves the penalty"


def test_the_layer_decomposition_is_exact_on_the_running_example(p2p_norm, p2p_log):
    V = wise.violation_matrix(p2p_log, p2p_norm).to_numpy(dtype=float)
    for mode in MODES:
        for view in p2p_norm.view_names:
            out = aggregate(V, p2p_norm.weight_vector(view).to_numpy(dtype=float), layers_of(p2p_norm), mode)
            assert out.decomposition_error() <= 1e-12


def test_an_empty_result_has_no_decomposition_error_to_report():
    layers = LayerAssignment(("c1",), ("L",), ("L",))
    assert aggregate(np.empty((0, 1)), np.array([1.0]), layers, "flat").decomposition_error() == 0.0
    assert aggregate(np.array([[np.nan]]), np.array([1.0]), layers, "flat").decomposition_error() == 0.0


# ------------------------------------------------------------------ the assignment itself
def test_a_layer_assignment_pins_both_orders():
    a = LayerAssignment(("c1", "c2", "c3"), ("second", "first"), ("first", "second", "first"))
    assert a.members("first").tolist() == [0, 2]
    assert a.members("second").tolist() == [1]
    assert a.onehot.tolist() == [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], "layer column order is the declared one"
    assert len(a) == 3


def test_a_layer_assignment_refuses_to_guess(p2p_norm):
    with pytest.raises(NormError, match="every check must name exactly one layer"):
        LayerAssignment(("c1", "c2"), ("L",), ("L",))
    with pytest.raises(NormError, match="check ids must be unique"):
        LayerAssignment(("c1", "c1"), ("L",), ("L", "L"))
    with pytest.raises(NormError, match="layer ids must be unique"):
        LayerAssignment(("c1", "c2"), ("L", "L"), ("L", "L"))
    with pytest.raises(NormError, match="undeclared layer"):
        LayerAssignment(("c1",), ("L",), ("M",))
    with pytest.raises(NormError, match="unknown layer"):
        LayerAssignment(("c1",), ("L",), ("L",)).members("M")
    with pytest.raises(NormError, match="has no layer"):
        LayerAssignment.from_mapping(["c1"], {})


def test_from_mapping_defaults_to_first_seen_layer_order():
    a = LayerAssignment.from_mapping(["c1", "c2", "c3"], {"c1": "b", "c2": "a", "c3": "b"})
    assert a.layer_ids == ("b", "a")


def test_the_kernel_refuses_a_misaligned_call():
    layers = LayerAssignment(("c1", "c2"), ("L",), ("L", "L"))
    with pytest.raises(NormError, match="rows × checks"):
        aggregate(np.array([0.0, 1.0]), np.array([1.0, 1.0]), layers, "flat")
    with pytest.raises(NormError, match="one entry per check"):
        aggregate(np.zeros((2, 2)), np.array([1.0]), layers, "flat")
    with pytest.raises(NormError, match="the layer assignment declares"):
        aggregate(np.zeros((2, 3)), np.array([1.0, 1.0]), layers, "flat")
    with pytest.raises(NormError, match="a misaligned result is a wrong result"):
        aggregate(np.zeros((2, 2)), np.array([1.0, 1.0]), layers, "flat", row_ids=["only-one"])
    with pytest.raises(NormError, match="mode must be one of"):
        aggregate(np.zeros((1, 2)), np.array([1.0, 1.0]), layers, "nearest")


def test_the_mode_tuple_matches_the_norms():
    assert set(AGGREGATION_MODES) == set(wise.norm.SCORING_MODES)


def test_row_ids_travel_with_the_result(p2p_norm):
    out = aggregate(np.zeros((3, 6)), p2p_norm.weight_vector("Finance").to_numpy(dtype=float), layers_of(p2p_norm), "flat")
    assert out.row_ids == (0, 1, 2), "without identities the rows are their positions, and say so"
    named = aggregate(
        np.zeros((3, 6)),
        p2p_norm.weight_vector("Finance").to_numpy(dtype=float),
        layers_of(p2p_norm),
        "flat",
        row_ids=["A", "B", "C"],
    )
    assert named.row_ids == ("A", "B", "C")
    assert isinstance(named, Aggregation)
    assert named.constraint_ids == tuple(p2p_norm.constraint_ids)
    assert named.layer_ids == tuple(p2p_norm.layer_ids)
    assert named.mode == "flat"


# ------------------------------------------------------------------ score() is unmoved
@pytest.mark.parametrize("mode", MODES)
def test_score_produces_the_frames_the_reviewed_base_produced(p2p_log, p2p_norm, mode):
    result = wise.score(p2p_log, p2p_norm, mode=mode)
    V = result.violations
    for view in p2p_norm.view_names:
        w = p2p_norm.weight_vector(view).to_numpy(dtype=float)
        s, contrib, w_eff, _, _ = _reference_view(V, w, p2p_norm, mode)
        identical(result.scores[view].to_numpy(dtype=float), s)
        identical(result.contributions[view].to_numpy(dtype=float), contrib)
        identical(result.effective_weights(view).to_numpy(dtype=float), np.nan_to_num(w_eff))
    assert list(result.contributions["Finance"].columns) == p2p_norm.layer_ids
    assert list(result.scores.columns) == p2p_norm.view_names


def test_the_paper_numbers_are_still_the_paper_numbers(p2p_result):
    """Table VI, computed through the extracted kernel.

    Against decimal literals rather than a recomputation, so this uses the
    stage-0 contract's declared tolerance ``rtol=0, atol=1e-12``; bit-identity
    is asserted above, against the reviewed base's own arithmetic.
    """
    close = dict(rtol=0, atol=1e-12)
    np.testing.assert_allclose(p2p_result.scores["Finance"].to_numpy(), [0.6625, 0.9666666666666667, 0.87, 0.9, 0.15], **close)
    np.testing.assert_allclose(p2p_result.scores["Logistics"].to_numpy(), [0.8875, 0.7, 0.9675, 0.9, 0.55], **close)


def test_the_kernel_is_as_layout_sensitive_as_the_code_it_replaces(p2p_norm):
    """Stated rather than hidden: row sums of a C- and an F-ordered copy of the
    same matrix can differ in the last bit, because pairwise summation walks
    memory. ``score`` hands the kernel the same F-ordered array it always did,
    which is why the extraction moves nothing; a caller who builds its own
    matrix gets agreement to 1e-12 and not necessarily to the bit."""
    rng = np.random.default_rng(11)
    V = rng.random((9, len(p2p_norm.constraint_ids)))
    w = p2p_norm.weight_vector("Finance").to_numpy(dtype=float)
    layers = layers_of(p2p_norm)
    c_order = aggregate(np.ascontiguousarray(V), w, layers, "flat").scores
    f_order = aggregate(np.asfortranarray(V), w, layers, "flat").scores
    np.testing.assert_allclose(c_order, f_order, rtol=0, atol=1e-12)
