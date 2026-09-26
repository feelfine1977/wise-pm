"""Defects C1, C2 and C9 in the constraint definitions (``wise.constraints``).

Regression tests pin the desired behaviour and are ``xfail(strict=True)``
until the fix lands; characterisation tests guard behaviour that is already
right. See ``01-findings-classic-main.md`` and decision 3 of 2026-09-26.
"""

from typing import Any

import pytest

import wise
from _support.builders import evaluate
from wise.constraints import constraint_from_dict
from wise.errors import NormError

GR = "Record Goods Receipt"
INV = "Record Invoice Receipt"

#: Minimal valid parameters per constraint type; the lag carries an explicit
#: time bound so the case stays valid once C2 is fixed.
MINIMAL_PARAMS: dict[str, dict[str, Any]] = {
    "presence": {"activity": "A"},
    "exclusion": {"activity": "A"},
    "singularity": {"activity": "A"},
    "lag": {"a": "A", "b": "B", "delta": 10, "width": 20},
    "precedence": {"a": "A", "b": "B"},
    "balance": {"attr_x": "x", "activities_x": "A", "attr_y": "y", "activities_y": "B"},
    "metric": {"attribute": "amount"},
}


# --- C1: constraint_from_dict truncates non-integer m / k ------------------------------


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="C1: constraint_from_dict truncates non-integer m/k with int() before the dataclass validation runs",
)
@pytest.mark.parametrize(
    ("type_name", "params"),
    [
        pytest.param("presence", {"activity": "A", "m": 1.5}, id="presence-m-1.5"),
        pytest.param("presence", {"activity": "A", "m": "2.7"}, id="presence-m-str-2.7"),
        pytest.param("singularity", {"activity": "A", "k": 0.9}, id="singularity-k-0.9"),
        pytest.param("precedence", {"a": "A", "b": "B", "k": 1.5}, id="precedence-k-1.5"),
    ],
)
def test_constraint_from_dict_rejects_non_integer_count_parameters(type_name, params):
    with pytest.raises(NormError):
        constraint_from_dict(type_name, params)


@pytest.mark.parametrize("m", [2.0, "2", 2], ids=["float", "numeric-string", "int"])
def test_constraint_from_dict_accepts_integer_valued_m(m):
    constraint = constraint_from_dict("presence", {"activity": "A", "m": m})
    assert isinstance(constraint, wise.Presence)
    assert constraint.m == 2
    assert isinstance(constraint.m, int)


# --- C2: Lag without a time bound silently defaults to delta=0, width=0 ------------------


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="C2: Lag defaults to delta=0, width=0; decision 3 (2026-09-26) requires delta or the unbounded form",
)
def test_lag_without_delta_is_rejected():
    with pytest.raises(NormError):
        wise.Lag("A", "B")


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="C2: with the silent default the 8-day GR->INV lag of running-example case B scores as a full violation (1.0)",
)
def test_default_lag_cannot_be_evaluated_on_running_example(p2p_log):
    # Case B: four goods receipts on days 0-3, invoice on day 8. A lag with
    # no declared bound must not be evaluable at all, rather than turning
    # that ordinary delay into nu = 1.0.
    with pytest.raises(NormError):
        evaluate(p2p_log, wise.Lag(GR, INV))


@pytest.mark.spec("decision:2026-09-26/3")
def test_lag_unbounded_form_is_accepted():
    follow = wise.Lag("A", "B", delta=None)
    assert follow.delta is None
    assert follow.params()["delta"] is None


@pytest.mark.spec("decision:2026-09-26/3")
def test_lag_with_explicit_bound_is_accepted():
    lag = wise.Lag("A", "B", delta=10, width=20)
    assert (lag.delta, lag.width, lag.unit) == (10.0, 20.0, "D")


# --- C9: non-numeric parameters escape as TypeError / ValueError ------------------------


def _c9(raises, what):
    return [
        pytest.mark.regression,
        pytest.mark.xfail(strict=True, raises=raises, reason=f"C9: {what} escapes as {raises.__name__}, not NormError"),
    ]


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: wise.Presence("A", m="2"), id="presence-m-str", marks=_c9(TypeError, "Presence(m='2')")),  # type: ignore[arg-type]  # C9: deliberate wrong type
        pytest.param(
            lambda: wise.Balance("x", "A", "y", "B", tau="x"),  # type: ignore[arg-type]  # C9: deliberate wrong type
            id="balance-tau-str",
            marks=_c9(ValueError, "Balance(tau='x')"),
        ),
        pytest.param(lambda: wise.Precedence("A", "B", k="1"), id="precedence-k-str", marks=_c9(TypeError, "Precedence(k='1')")),  # type: ignore[arg-type]  # C9: deliberate wrong type
        pytest.param(lambda: wise.Singularity("A", k="1"), id="singularity-k-str", marks=_c9(TypeError, "Singularity(k='1')")),  # type: ignore[arg-type]  # C9: deliberate wrong type
        pytest.param(lambda: wise.Singularity("A", K="big"), id="singularity-K-str"),  # type: ignore[arg-type]  # C9: deliberate wrong type
    ],
)
def test_constructor_rejects_non_numeric_parameter_with_norm_error(build):
    with pytest.raises(NormError):
        build()


# --- characterisation: constraint_from_dict contract ------------------------------------


def test_constraint_from_dict_rejects_unknown_type():
    with pytest.raises(NormError, match="unknown constraint type"):
        constraint_from_dict("bogus", {"activity": "A"})


def test_constraint_from_dict_rejects_unknown_parameter():
    with pytest.raises(NormError, match="unknown parameters"):
        constraint_from_dict("presence", {"activity": "A", "zzz": 1})


def test_constraint_from_dict_rejects_non_numeric_string_for_numeric_parameter():
    with pytest.raises(NormError, match="not numeric"):
        constraint_from_dict("presence", {"activity": "A", "m": "abc"})


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("pres", wise.Presence),
        ("excl", wise.Exclusion),
        ("sing", wise.Singularity),
        ("bal", wise.Balance),
        ("order", wise.Precedence),
    ],
)
def test_type_alias_resolves_to_constraint_class(alias, expected):
    constraint = constraint_from_dict(alias, MINIMAL_PARAMS[expected.type])
    assert type(constraint) is expected
    assert constraint.type == expected.type


def test_minimal_params_cover_every_registered_constraint_type():
    assert set(MINIMAL_PARAMS) == set(wise.CONSTRAINT_TYPES)


@pytest.mark.parametrize("type_name", sorted(MINIMAL_PARAMS))
def test_params_round_trip_through_constraint_from_dict(type_name):
    original = constraint_from_dict(type_name, MINIMAL_PARAMS[type_name])
    rebuilt = constraint_from_dict(type_name, original.params())
    assert rebuilt == original
    assert rebuilt.params() == original.params()
