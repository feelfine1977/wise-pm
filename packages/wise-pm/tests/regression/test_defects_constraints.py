"""Defects C1, C2 and C9 in the constraint definitions (``wise.constraints``).

Regression tests pin the desired behaviour and are ``xfail(strict=True)``
until the fix lands; characterisation tests guard behaviour that is already
right. See ``01-findings-classic-main.md`` and decision 3 of 2026-09-26.
"""

import json
from typing import Any

import pytest

import wise
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


# --- C1: constraint_from_dict truncated non-integer m / k ------------------------------
# C1 closed 2026-10-02: the counts m (Presence) and k (Singularity, Precedence) of paper
# Sec. IV-B are integers; a document and a constructor refuse a non-integer value with the
# same NormError instead of truncating it (2.7 used to become m = 2).

NON_INTEGER_COUNTS = [
    pytest.param("presence", {"activity": "A", "m": 1.5}, "presence: m must be an integer", id="presence-m-1.5"),
    pytest.param("presence", {"activity": "A", "m": "2.7"}, "presence: m must be an integer", id="presence-m-str-2.7"),
    pytest.param("singularity", {"activity": "A", "k": 0.9}, "singularity: k must be an integer", id="singularity-k-0.9"),
    pytest.param("precedence", {"a": "A", "b": "B", "k": 1.5}, "precedence: k must be an integer", id="precedence-k-1.5"),
]


@pytest.mark.regression
@pytest.mark.spec("IV-B")
@pytest.mark.parametrize(("type_name", "params", "message"), NON_INTEGER_COUNTS)
def test_constraint_from_dict_rejects_non_integer_count_parameters(type_name, params, message):
    with pytest.raises(NormError, match=message):
        constraint_from_dict(type_name, params)


@pytest.mark.regression
@pytest.mark.spec("IV-B")
@pytest.mark.parametrize(("type_name", "params", "message"), NON_INTEGER_COUNTS)
def test_norm_document_rejects_non_integer_count_parameters(type_name, params, message):
    doc = {
        "schema_version": 2,
        "constraints": [{"id": "c", "layer": "L", "type": type_name, "params": params}],
        "views": [{"name": "v", "layer_weights": {"L": 1}}],
    }
    with pytest.raises(NormError, match=message):
        wise.Norm.from_dict(doc)


@pytest.mark.regression
@pytest.mark.spec("IV-B")
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: wise.Presence("A", m=1.5), id="presence-m-1.5"),  # type: ignore[arg-type]  # C1: deliberate non-integer
        pytest.param(lambda: wise.Singularity("A", k=0.9), id="singularity-k-0.9"),  # type: ignore[arg-type]  # C1: deliberate non-integer
        pytest.param(lambda: wise.Precedence("A", "B", k=1.5), id="precedence-k-1.5"),  # type: ignore[arg-type]  # C1: deliberate non-integer
    ],
)
def test_constructor_rejects_non_integer_count_parameters(build):
    with pytest.raises(NormError, match="must be an integer"):
        build()


@pytest.mark.parametrize("m", [2.0, "2", 2], ids=["float", "numeric-string", "int"])
def test_constraint_from_dict_accepts_integer_valued_m(m):
    constraint = constraint_from_dict("presence", {"activity": "A", "m": m})
    assert isinstance(constraint, wise.Presence)
    assert constraint.m == 2
    assert isinstance(constraint.m, int)


# --- C2: Lag without a time bound silently defaults to delta=0, width=0 ------------------
# C2 closed 2026-10-02 for schema version 3 documents (decision 0001, section 3): a v3 lag states its bound, so a lag
# without one is refused when the document is read and nothing is scored with it; migrate() writes the bound 0.1.0
# read a bare lag with, delta 0 and width 0, into the document and recommends reviewing it. A schema version 2 document
# keeps the 0.1.0 reading under the legacy profile (section 4), and so does a Lag built in Python, whose default cannot
# be told apart from an explicit delta=0.

C2_LAG = {"id": "lag", "layer": "L", "type": "lag", "params": {"a": GR, "b": INV}}
C2_DOCUMENT: dict[str, Any] = {
    "schema_version": 2,
    "name": "c2",
    "views": [{"name": "v", "constraint_weights": {"lag": 1}}],
    "constraints": [C2_LAG],
}


@pytest.mark.regression
@pytest.mark.spec("decision:0001/3")
@pytest.mark.parametrize(
    "params", [{"a": GR, "b": INV}, {"a": GR, "b": INV, "width": 5}, {"a": GR, "b": INV, "delta": None}], ids=str
)
def test_lag_without_delta_is_rejected(params):
    document = {**C2_DOCUMENT, "schema_version": 3, "missing_values": "three_valued"}
    document["constraints"] = [{**C2_LAG, "params": params}]
    with pytest.raises(NormError, match=r"constraint 'lag' \(lag\): a lag states its bound in schema version 3"):
        wise.Norm.from_dict(document)


@pytest.mark.regression
@pytest.mark.spec("decision:0001/3")
@pytest.mark.filterwarnings("ignore::wise.LegacySemanticsWarning")
def test_default_lag_cannot_be_evaluated_on_running_example(p2p_log):
    # Case B: four goods receipts on days 0-3, invoice on day 8. Under 0.1.0 a lag with no declared bound turns that
    # ordinary delay into nu = 1.0; schema version 2 keeps that reading.
    legacy = wise.score(p2p_log, wise.Norm.from_dict(C2_DOCUMENT))
    assert legacy.violations.loc["B", "lag"] == 1.0
    # The migration does not keep the default silent: it writes it into the document and asks for a review.
    migration = wise.migrate(C2_DOCUMENT)
    assert migration.document["constraints"][0]["params"] == {"a": GR, "b": INV, "delta": 0, "width": 0}
    (rewrite,) = [r for r in migration.report.rewrites if r.rule == "lag_bound"]
    assert (rewrite.subject, rewrite.corrections) == ("lag", ("C2",))
    assert rewrite.recommendation is not None
    assert "any delay between the activities is a full violation" in rewrite.recommendation
    # The bound the review settles on is what a v3 norm evaluates: here an unbounded lag, which B satisfies.
    reviewed = json.loads(migration.dumps())
    reviewed["constraints"][0]["params"] = {"a": GR, "b": INV, "unbounded": True}
    assert wise.score(p2p_log, wise.Norm.from_dict(reviewed)).violations.loc["B", "lag"] == 0.0


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


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: wise.Presence("A", m="2"), id="presence-m-str"),  # type: ignore[arg-type]  # C9: deliberate wrong type
        pytest.param(
            lambda: wise.Balance("x", "A", "y", "B", tau="x"),  # type: ignore[arg-type]  # C9: deliberate wrong type
            id="balance-tau-str",
        ),
        pytest.param(lambda: wise.Precedence("A", "B", k="1"), id="precedence-k-str"),  # type: ignore[arg-type]  # C9: deliberate wrong type
        pytest.param(lambda: wise.Singularity("A", k="1"), id="singularity-k-str"),  # type: ignore[arg-type]  # C9: deliberate wrong type
        pytest.param(lambda: wise.Singularity("A", K="big"), id="singularity-K-str"),  # type: ignore[arg-type]  # C9: deliberate wrong type
    ],
)
def test_constructor_rejects_non_numeric_parameter_with_norm_error(build):
    # C9 closed 2026-09-27: constructors coerce once and raise NormError for any non-numeric parameter (decision record 0001)
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
