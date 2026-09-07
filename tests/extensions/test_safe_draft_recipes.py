"""An untrusted recipe is refused before anything is computed (row A01).

The failure this file guards against is precise: :func:`wise.derive.compute_recipe`
supports ``kind="eval"``, which reaches :meth:`pandas.DataFrame.eval`. That is
acceptable in a norm a maintainer wrote and unacceptable in one a model or a
document proposed.

Two properties are asserted here, and the second is the one that matters:

1. the untrusted path rejects the recipe, and
2. it rejects it *before* any evaluation happens — proved by making
   ``DataFrame.eval`` explode if it is ever reached during a preview.

The trusted path is asserted too, in the other direction: an explicitly
authored ``eval`` recipe still works exactly as it always did, because
removing that is a separate deprecation decision nobody has taken.
"""

from __future__ import annotations

import pandas as pd
import pytest

import wise
from wise.derive import KINDS, UNSAFE_RECIPE_KINDS, compute_recipe
from wise.errors import UnsafeDraft
from wise.llm.drafts import DraftLimits, NormDraft, check_recipe, check_regex, preview_norm_draft, validate_candidate


@pytest.fixture
def base_norm():
    return wise.running_p2p_norm()


def candidate_with(norm, recipes):
    payload = norm.to_dict()
    payload["derived_attributes"] = list(recipes)
    return payload


# ------------------------------------------------------------- the eval kind
def test_eval_is_the_declared_unsafe_kind_and_it_is_still_a_supported_kind():
    assert UNSAFE_RECIPE_KINDS == ("eval",)
    assert "eval" in KINDS, "the trusted catalogue is unchanged"


def test_an_eval_recipe_is_refused_in_the_untrusted_path(base_norm):
    with pytest.raises(UnsafeDraft, match="evaluates an expression"):
        validate_candidate(candidate_with(base_norm, [{"name": "x", "kind": "eval", "expr": "n_events * 2"}]))


@pytest.mark.parametrize(
    "expr",
    [
        "n_events * 2",
        "__import__('os').system('id')",
        "@py_call",
        "n_events.__class__.__mro__",
    ],
)
def test_no_expression_gets_a_second_chance(base_norm, expr):
    with pytest.raises(UnsafeDraft, match="evaluates an expression"):
        check_recipe({"name": "x", "kind": "eval", "expr": expr}, pointer="r", limits=DraftLimits())


def test_the_rejection_happens_before_any_evaluation(base_norm, monkeypatch):
    """Make ``DataFrame.eval`` fatal, then prove the preview never reaches it."""

    def explode(*args, **kwargs):
        raise AssertionError("DataFrame.eval was reached from the untrusted draft path")

    monkeypatch.setattr(pd.DataFrame, "eval", explode)
    draft = NormDraft(
        parent_norm_hash=base_norm.fingerprint(),
        candidate_norm=candidate_with(base_norm, [{"name": "x", "kind": "eval", "expr": "n_events"}]),
    )
    with pytest.raises(UnsafeDraft):
        preview_norm_draft(draft, base_norm, wise.running_p2p_log())


def test_the_rejection_happens_before_the_loader_is_even_asked(base_norm, monkeypatch):
    """Strict JSON loading is not the boundary, so it must not be the first check."""
    called: list[str] = []
    original = wise.Norm.from_dict

    def watched(cls_data):  # pragma: no cover - only runs if the order is wrong
        called.append("from_dict")
        return original(cls_data)

    monkeypatch.setattr(wise.Norm, "from_dict", staticmethod(watched))
    with pytest.raises(UnsafeDraft):
        validate_candidate(candidate_with(base_norm, [{"name": "x", "kind": "eval", "expr": "1"}]))
    assert called == [], "the loader ran before the unsafe kind was refused"


def test_the_trusted_path_is_untouched():
    log = wise.running_p2p_log()
    values = compute_recipe(log, {"name": "double", "kind": "eval", "expr": "n_events * 2"})
    assert list(values) == [n * 2 for n in log.cases["n_events"]]


def test_a_trusted_norm_with_an_eval_recipe_still_loads_and_scores():
    norm = wise.running_p2p_norm()
    payload = norm.to_dict()
    payload["derived_attributes"] = [*payload.get("derived_attributes", []), {"name": "d", "kind": "eval", "expr": "n_events"}]
    trusted = wise.Norm.from_dict(payload)
    result = wise.score(wise.running_p2p_log(), trusted)
    assert "d" in result.cases.columns


# ------------------------------------------------- unknown kinds and keys
@pytest.mark.parametrize("kind", ["exec", "python", "sql", "shell", "", None, 7])
def test_an_unknown_recipe_kind_is_refused(kind):
    with pytest.raises(UnsafeDraft, match="unknown recipe kind"):
        check_recipe({"name": "x", "kind": kind}, pointer="r", limits=DraftLimits())


def test_an_unrecognised_key_is_refused_rather_than_ignored():
    with pytest.raises(UnsafeDraft, match="unrecognised key"):
        check_recipe({"name": "x", "kind": "count", "activities": ["A"], "module": "os"}, pointer="r", limits=DraftLimits())


def test_an_unknown_evaluator_name_is_refused():
    with pytest.raises(UnsafeDraft, match="unknown evaluator name"):
        check_recipe({"name": "x", "kind": "agg", "column": "amount", "agg": "eval"}, pointer="r", limits=DraftLimits())


def test_a_derived_attribute_name_must_be_a_plain_identifier():
    for name in ["../x", "os.system", "a b", "", None]:
        with pytest.raises(UnsafeDraft, match="plain identifier"):
            check_recipe({"name": name, "kind": "count", "activities": ["A"]}, pointer="r", limits=DraftLimits())


def test_a_missing_required_key_is_refused():
    with pytest.raises(UnsafeDraft, match="needs"):
        check_recipe({"name": "x", "kind": "ratio", "numerator": "a"}, pointer="r", limits=DraftLimits())


# ------------------------------------------------------- unsafe regex shapes
@pytest.mark.parametrize("pattern", ["(a+)+$", "(a*)*b", "(x|x)+y"])
def test_a_catastrophically_backtracking_pattern_is_refused(pattern):
    with pytest.raises(UnsafeDraft, match="backtrack"):
        check_regex(pattern, pointer="r", limits=DraftLimits())


def test_an_enormous_or_uncompilable_pattern_is_refused():
    with pytest.raises(UnsafeDraft, match="past the limit"):
        check_regex("a" * 500, pointer="r", limits=DraftLimits())
    with pytest.raises(UnsafeDraft, match="repetition count"):
        check_regex("a{99999}", pointer="r", limits=DraftLimits())
    with pytest.raises(UnsafeDraft, match="does not compile"):
        check_regex("(unclosed", pointer="r", limits=DraftLimits())


def test_an_ordinary_pattern_is_accepted():
    check_regex(r"INV-\d{4,8}", pointer="r", limits=DraftLimits())


def test_an_unsafe_pattern_inside_a_where_clause_is_refused():
    with pytest.raises(UnsafeDraft, match="backtrack"):
        check_recipe(
            {"name": "x", "kind": "count_events", "where": {"column": "activity", "regex": "(a+)+$"}},
            pointer="r",
            limits=DraftLimits(),
        )


def test_an_unknown_where_operator_is_refused():
    with pytest.raises(UnsafeDraft, match="unrecognised key"):
        check_recipe(
            {"name": "x", "kind": "count_events", "where": {"column": "activity", "query": "1==1"}},
            pointer="r",
            limits=DraftLimits(),
        )


# ------------------------------------------------------ resource shapes
@pytest.mark.parametrize("q", [0, -1, 2, "0.9", True, None])
def test_a_quantile_outside_its_range_is_refused(q):
    with pytest.raises(UnsafeDraft, match="q must be"):
        check_recipe({"name": "x", "kind": "quantile_scale", "attribute": "a", "q": q}, pointer="r", limits=DraftLimits())


def test_a_non_positive_epsilon_is_refused():
    with pytest.raises(UnsafeDraft, match="eps must be"):
        check_recipe({"name": "x", "kind": "cv", "column": "amount", "eps": 0}, pointer="r", limits=DraftLimits())


def test_an_enormous_activity_list_is_refused():
    with pytest.raises(UnsafeDraft, match="at most"):
        check_recipe(
            {"name": "x", "kind": "count", "activities": [f"A{i}" for i in range(500)]},
            pointer="r",
            limits=DraftLimits(),
        )


def test_a_safe_recipe_passes_every_check():
    check_recipe(
        {"name": "manual_touches", "kind": "count", "activities": ["Change Quantity"], "after": ["Record Invoice Receipt"]},
        pointer="r",
        limits=DraftLimits(),
    )
