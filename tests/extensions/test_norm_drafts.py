"""The draft envelope, the deep check and the preview that applies nothing (N01).

Acceptance rows A01 (unsafe recipe — its own file), A02 (a model's approval is
not authority), A03 (a stale parent fingerprint is a conflict) and A04 (a
preview leaves the original untouched), plus the catalogue introspection the
whole check is derived from.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import wise
from wise.errors import DraftConflict, DraftError, UnsafeDraft
from wise.llm.drafts import (
    DRAFT_ENVELOPE_VERSION,
    DraftLimits,
    NormDraft,
    ProposedExample,
    Provenance,
    Severity,
    assumption_diff,
    isolated_log,
    preview_norm_draft,
    validate_candidate,
)
from wise.llm.schemas import NORM_DRAFT_SCHEMA, validate_against
from wise.schema import (
    CATALOGUE_VERSION,
    catalogue_digest,
    check_parameters,
    constraint_catalogue,
    describe_catalogue,
    recipe_catalogue,
    verify_catalogue,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "norm_drafts"


@pytest.fixture
def approved():
    return wise.running_p2p_norm()


@pytest.fixture
def log():
    return wise.running_p2p_log()


def slower_lag(norm, delta=6.0):
    """A candidate identical to the approved norm but for one lag threshold."""
    payload = norm.to_dict()
    for entry in payload["constraints"]:
        if entry["type"] == "lag":
            entry["params"]["delta"] = delta
            break
    return payload


# ------------------------------------------------------- catalogue (N01)
def test_the_catalogue_is_read_off_the_implementation():
    catalogue = constraint_catalogue()
    assert set(catalogue) == set(wise.CONSTRAINT_TYPES)
    for name, cls in wise.CONSTRAINT_TYPES.items():
        import dataclasses

        assert catalogue[name].parameter_names == tuple(f.name for f in dataclasses.fields(cls))


def test_the_declared_bounds_cannot_drift_from_the_implementation():
    verify_catalogue()
    for name, spec in constraint_catalogue().items():
        for parameter in spec.parameters:
            if parameter.numeric:
                assert parameter.bound is not None, f"{name}.{parameter.name} has no declared range"


def test_the_catalogue_digest_is_stable_and_describes_itself():
    assert catalogue_digest() == catalogue_digest()
    described = describe_catalogue()
    assert described["catalogue_version"] == CATALOGUE_VERSION
    assert described["catalogue_digest"] == catalogue_digest()
    assert "eval" in {r["kind"] for r in described["recipes"]}
    assert next(r for r in described["recipes"] if r["kind"] == "eval")["evaluates_expression"] is True


def test_recipe_keys_come_from_the_one_maintained_table():
    assert recipe_catalogue()["lag"].required == ("a", "b")
    assert "unit" in recipe_catalogue()["lag"].optional


@pytest.mark.parametrize(
    "type_name,params,expected",
    [
        ("presence", {"activity": "A", "m": 0}, "outside the supported range"),
        ("balance", {"attr_x": "a", "activities_x": "A", "attr_y": "b", "activities_y": "B", "tau": 2.0}, "outside"),
        ("lag", {"a": "A", "b": "B", "unit": "fortnight"}, "not one of"),
        ("metric", {"attribute": "x", "width": float("inf")}, "non-finite"),
        ("presence", {"activity": "A", "colour": "red"}, "unknown parameter"),
        ("presence", {}, "required parameter"),
        ("nonsense", {}, "unknown constraint type"),
    ],
)
def test_parameter_checks_reflect_the_real_ranges(type_name, params, expected):
    findings = check_parameters(type_name, params)
    assert findings and any(expected in f for f in findings)


# ---------------------------------------------------------- the envelope
def test_the_envelope_matches_the_roadmap_contract(approved):
    draft = NormDraft(
        parent_norm_hash=approved.fingerprint(),
        candidate_norm=None,
        source_refs=("policy-span-01",),
        unresolved_questions=("Which business calendar applies?",),
        proposed_examples=(ProposedExample("EX-001", "An open invoice.", "unknown"),),
    )
    validate_against(NORM_DRAFT_SCHEMA, draft.to_contract_dict())
    assert set(draft.to_contract_dict()) == set(NORM_DRAFT_SCHEMA["required"])


def test_draft_metadata_stays_outside_the_norm_schema(approved):
    draft = NormDraft(parent_norm_hash=approved.fingerprint(), candidate_norm=slower_lag(approved))
    assert "validation_findings" not in draft.to_contract_dict()
    assert "validation_findings" in draft.to_dict()
    candidate = wise.Norm.from_dict(draft.candidate_norm)
    assert set(candidate.to_dict()) == set(approved.to_dict()), "no draft key leaked into schema-2 norm JSON"


def test_the_roadmap_example_round_trips():
    payload = json.loads((FIXTURES / "roadmap_example.json").read_text(encoding="utf-8"))
    draft = NormDraft.from_dict(payload)
    assert draft.to_contract_dict() == payload
    assert draft.envelope_version == DRAFT_ENVELOPE_VERSION


def test_a_draft_serialises_and_reads_back(approved):
    draft = NormDraft(
        parent_norm_hash=approved.fingerprint(),
        candidate_norm=slower_lag(approved),
        assumptions=("the business calendar is ignored",),
        draft_id="D-001",
    )
    again = NormDraft.from_dict(json.loads(draft.to_json()))
    assert again.parent_norm_hash == draft.parent_norm_hash
    assert again.candidate_norm == draft.candidate_norm
    assert again.draft_id == "D-001"


def test_a_draft_without_a_parent_is_refused():
    with pytest.raises(DraftError, match="fingerprint of the norm"):
        NormDraft(parent_norm_hash="", candidate_norm=None)


# ------------------------------------------------------ A02 not an approval
def test_a_model_cannot_confirm_an_owner_or_an_approval(approved):
    payload = {
        "schema_version": "0.1-proposal",
        "parent_norm_hash": approved.fingerprint(),
        "review_status": "pending_human_review",
        "candidate_norm": None,
        "source_refs": [],
        "assumptions": [],
        "unresolved_questions": [],
        "proposed_examples": [],
        "approved": True,
        "responsible_owner": "confirmed: Finance",
    }
    with pytest.raises(DraftError, match="unknown key"):
        NormDraft.from_model_payload(payload)


def test_a_draft_is_never_usable_by_itself(approved):
    draft = NormDraft(parent_norm_hash=approved.fingerprint(), candidate_norm=None)
    assert draft.review_status == "pending_human_review"
    assert draft.usable is False
    assert not hasattr(draft, "activate") and not hasattr(draft, "apply")


# -------------------------------------------------------- A03 stale parent
def test_a_stale_parent_fingerprint_is_a_conflict(approved, log):
    draft = NormDraft(parent_norm_hash="0" * 64, candidate_norm=slower_lag(approved))
    with pytest.raises(DraftConflict, match="stale parent is a conflict"):
        preview_norm_draft(draft, approved, log)


def test_the_conflict_is_raised_before_the_candidate_is_even_checked(approved, log):
    draft = NormDraft(
        parent_norm_hash="0" * 64,
        candidate_norm={"constraints": [], "derived_attributes": [{"name": "x", "kind": "eval", "expr": "1"}]},
    )
    with pytest.raises(DraftConflict):
        preview_norm_draft(draft, approved, log)


def test_a_matching_parent_is_accepted(approved, log):
    draft = NormDraft(parent_norm_hash=approved.fingerprint(), candidate_norm=slower_lag(approved))
    preview = preview_norm_draft(draft, approved, log)
    assert preview.candidate_fingerprint and preview.candidate_fingerprint != approved.fingerprint()


# ------------------------------------------------ A04 the preview mutates nothing
def test_the_preview_leaves_the_approved_norm_and_its_fingerprint_untouched(approved, log):
    before_fingerprint = approved.fingerprint()
    before_payload = copy.deepcopy(approved.to_dict())
    draft = NormDraft(parent_norm_hash=before_fingerprint, candidate_norm=slower_lag(approved))
    preview_norm_draft(draft, approved, log)
    assert approved.fingerprint() == before_fingerprint
    assert approved.to_dict() == before_payload


def test_the_preview_leaves_the_log_its_columns_caches_and_attributes(approved, log):
    """The candidate *derives an attribute*, which is the only way a leak shows.

    A candidate that changes nothing but a threshold writes nothing to the log,
    so it could not detect a preview running on the caller's own object. This
    one adds a recipe: without the isolated copy, ``manual_touches`` would
    appear in the caller's case table and its recipe cache.
    """
    candidate = slower_lag(approved)
    candidate["derived_attributes"] = [{"name": "manual_touches", "kind": "count", "activities": ["Change Quantity"]}]
    before_columns = list(log.cases.columns)
    before_attributes = list(log.case_attributes)
    before_cache = dict(log._recipe_cache)
    before_events = log.events.copy(deep=True)
    draft = NormDraft(parent_norm_hash=approved.fingerprint(), candidate_norm=candidate)
    preview_norm_draft(draft, approved, log)
    assert "manual_touches" not in log.cases.columns, "the preview wrote a derived column into the caller's log"
    assert list(log.cases.columns) == before_columns
    assert list(log.case_attributes) == before_attributes
    assert log._recipe_cache == before_cache, "the preview left an entry in the caller's recipe cache"
    assert log.events.equals(before_events)


def test_the_draft_payload_itself_is_not_mutated(approved, log):
    candidate = slower_lag(approved)
    snapshot = copy.deepcopy(candidate)
    draft = NormDraft(parent_norm_hash=approved.fingerprint(), candidate_norm=candidate)
    preview_norm_draft(draft, approved, log)
    assert candidate == snapshot
    assert draft.candidate_norm == snapshot


def test_an_isolated_copy_does_not_share_mutable_state(log):
    clone = isolated_log(log)
    clone.add_case_attribute("only_on_the_clone", [1.0] * len(log))
    assert "only_on_the_clone" not in log.cases.columns
    assert "only_on_the_clone" not in log.case_attributes
    assert clone._count_cache is not log._count_cache


def test_a_preview_never_reports_itself_as_applied_or_activated(approved, log):
    draft = NormDraft(parent_norm_hash=approved.fingerprint(), candidate_norm=slower_lag(approved))
    preview = preview_norm_draft(draft, approved, log)
    assert preview.applied is False and preview.activated is False
    assert preview.to_dict()["applied"] is False and preview.to_dict()["activated"] is False


# ------------------------------------------------------------- the diff
def test_the_preview_reports_what_actually_changed(approved, log):
    draft = NormDraft(
        parent_norm_hash=approved.fingerprint(),
        candidate_norm=slower_lag(approved, delta=20.0),
        assumptions=("a slower clearing expectation is acceptable",),
    )
    preview = preview_norm_draft(draft, approved, log)
    changed = preview.assumption_diff["constraints_changed"]
    assert changed, "a changed lag threshold must appear in the assumption diff"
    assert preview.assumption_diff["declared"] == ["a slower clearing expectation is acceptable"]
    finance = preview.result_diff["per_view"]["Finance"]
    assert finance["mean_score_candidate"] > finance["mean_score_approved"], "a laxer norm scores higher"
    assert finance["n_units_changed"] > 0


def test_an_identical_candidate_changes_nothing(approved, log):
    draft = NormDraft(parent_norm_hash=approved.fingerprint(), candidate_norm=approved.to_dict())
    preview = preview_norm_draft(draft, approved, log)
    assert preview.assumption_diff["constraints_changed"] == {}
    assert preview.result_diff["per_view"]["Finance"]["n_units_changed"] == 0


def test_the_assumption_diff_names_added_and_removed_constraints(approved):
    payload = approved.to_dict()
    dropped = payload["constraints"][-1]["id"]
    payload["constraints"] = payload["constraints"][:-1]
    for view in payload["views"]:
        view.get("constraint_weights", {}).pop(dropped, None)
    smaller = wise.Norm.from_dict(payload)
    diff = assumption_diff(approved, smaller)
    assert diff["constraints_removed"] and not diff["constraints_added"]


# -------------------------------------------------------- structural limits
def test_a_candidate_that_is_not_an_object_is_refused():
    with pytest.raises(UnsafeDraft, match="must be an object"):
        validate_candidate(["constraints"])  # type: ignore[arg-type]


def test_an_unknown_top_level_key_is_refused(approved):
    payload = approved.to_dict()
    payload["run_this"] = "please"
    with pytest.raises(UnsafeDraft, match="unsupported top-level key"):
        validate_candidate(payload)


def test_an_unknown_constraint_key_is_refused(approved):
    payload = approved.to_dict()
    payload["constraints"][0]["callback"] = "os.system"
    with pytest.raises(UnsafeDraft, match="unrecognised key"):
        validate_candidate(payload)


def test_an_oversize_or_deeply_nested_candidate_is_refused(approved):
    payload = approved.to_dict()
    payload["description"] = "x" * 200_000
    with pytest.raises(UnsafeDraft, match="past the limit"):
        validate_candidate(payload)
    nested: dict = {"metadata": {}}
    cursor = nested["metadata"]
    for _ in range(30):
        cursor["deeper"] = {}
        cursor = cursor["deeper"]
    with pytest.raises(UnsafeDraft, match="nests deeper"):
        validate_candidate(nested, limits=DraftLimits(max_depth=5))


def test_too_many_constraints_are_refused(approved):
    payload = approved.to_dict()
    payload["constraints"] = payload["constraints"] * 100
    with pytest.raises(UnsafeDraft, match="exceed the limit"):
        validate_candidate(payload, limits=DraftLimits(max_constraints=10, max_bytes=1_000_000))


def test_an_unsupported_parameter_is_a_blocking_finding_not_an_exception(approved):
    payload = approved.to_dict()
    payload["constraints"][0]["params"] = {**payload["constraints"][0]["params"], "m": -3}
    findings = validate_candidate(payload)
    codes = {f.code for f in findings}
    assert "unsupported_parameter" in codes
    assert all(f.severity is Severity.BLOCKING for f in findings if f.code == "unsupported_parameter")


def test_a_candidate_the_loader_rejects_is_a_blocking_finding(approved):
    payload = approved.to_dict()
    payload["constraints"][0]["layer"] = "no-such-layer"
    findings = validate_candidate(payload)
    assert any(f.code == "loader_rejected" for f in findings)


def test_a_null_candidate_is_a_legitimate_draft(approved, log):
    draft = NormDraft(
        parent_norm_hash=approved.fingerprint(),
        candidate_norm=None,
        unresolved_questions=("What is the agreed clearing limit?",),
    )
    preview = preview_norm_draft(draft, approved, log)
    assert preview.candidate_fingerprint is None
    assert [f.code for f in preview.findings] == ["no_candidate"]
    assert preview.findings[0].provenance is Provenance.UNRESOLVED_PROPOSAL
    assert "What is the agreed clearing limit?" in preview.questions


def test_a_blocking_candidate_is_never_scored(approved, log):
    payload = approved.to_dict()
    payload["constraints"][0]["params"] = {**payload["constraints"][0]["params"], "m": -3}
    draft = NormDraft(parent_norm_hash=approved.fingerprint(), candidate_norm=payload)
    preview = preview_norm_draft(draft, approved, log)
    assert preview.blocking
    assert preview.result_diff == {}
