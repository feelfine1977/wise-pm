"""The bounded tool gateway and the access policy (items L02, L03).

Acceptance rows L02 (unknown tool or evidence id), L03 (unauthorised row,
column, document or run) and L04 (an unauthorised global baseline is not
leaked through an otherwise restricted explanation).

Each test here fails without the guard it names: remove the frozen approved
set and the first block passes a made-up tool through; remove the row mask
from ``compare_groups`` and the comparator block starts reporting the whole
company's mean to a principal restricted to one slice.
"""

from __future__ import annotations

import dataclasses
import json

import pytest
from _llm_fixtures import broad_policy, document_library, empty_policy, narrow_policy, packet_for, scored_run

import wise
from wise.errors import AccessDenied, BudgetExceeded, GatewayError
from wise.llm.gateway import APPROVED_TOOLS, DEFAULT_ENABLED_TOOLS, EvidenceHost, ToolGateway, check_string
from wise.llm.policy import (
    FULL_POPULATION_AGGREGATE,
    AccessPolicy,
    Principal,
    Scope,
    authorised_baseline,
    authorised_population,
)
from wise.llm.provider import BudgetLedger, CallBudget
from wise.llm.schemas import ToolCall


@pytest.fixture(scope="module")
def run():
    return scored_run()


@pytest.fixture(scope="module")
def packet(run):
    return packet_for(run)


def gateway_for(run, packet, policy, **kwargs):
    host = EvidenceHost(run, packet=packet, evidence=run.evidence, index=document_library())
    return ToolGateway(host, policy, **kwargs)


# ------------------------------------------------------- L02 the frozen set
def test_the_approved_set_is_exactly_the_six_names_the_handoff_lists():
    assert set(APPROVED_TOOLS) == {
        "get_run_summary",
        "get_evidence",
        "compare_groups",
        "explain_priority",
        "retrieve_approved_policy",
        "preview_norm_patch",
    }
    assert all(spec.read_only for spec in APPROVED_TOOLS.values())


def test_the_norm_preview_is_off_unless_it_is_turned_on():
    assert "preview_norm_patch" not in DEFAULT_ENABLED_TOOLS


@pytest.mark.parametrize(
    "name",
    ["run_sql", "read_file", "get_run_summary ", "GET_RUN_SUMMARY", "__class__", "os.system", ""],
)
def test_an_unapproved_tool_name_is_refused_before_anything_is_dispatched(run, packet, name):
    gateway = gateway_for(run, packet, broad_policy(run))
    with pytest.raises(GatewayError, match="unknown tool"):
        gateway.call(name, {})
    assert gateway.calls == [], "a refused name must not reach the audit trail as an execution"
    assert gateway.ledger.tool_calls == 0, "a refused name must not spend the tool budget"


def test_a_disabled_tool_is_refused_even_though_it_is_approved(run, packet):
    gateway = gateway_for(run, packet, broad_policy(run))
    with pytest.raises(GatewayError, match="unknown tool"):
        gateway.call("preview_norm_patch", {"draft_id": "d1"})


def test_a_tool_cannot_be_enabled_unless_it_is_approved(run, packet):
    with pytest.raises(GatewayError, match="unapproved tool"):
        gateway_for(run, packet, broad_policy(run), enabled_tools=("get_run_summary", "run_arbitrary_python"))


def test_the_registry_is_written_out_and_holds_only_approved_names(run, packet):
    gateway = gateway_for(run, packet, broad_policy(run))
    assert set(gateway._build_registry()) == set(APPROVED_TOOLS)


# ------------------------------------------------------ L02 argument schemas
@pytest.mark.parametrize(
    "arguments",
    [
        {"path": "/etc/passwd"},
        {"evaluation_ids": ["e1"], "extra": 1},
        {"evaluation_ids": "not-a-list"},
        {"evaluation_ids": [1, 2]},
        {"evaluation_ids": ["x"] * 40},
    ],
)
def test_malformed_or_undeclared_arguments_are_refused(run, packet, arguments):
    gateway = gateway_for(run, packet, broad_policy(run))
    with pytest.raises(GatewayError):
        gateway.call("get_evidence", arguments)


@pytest.mark.parametrize(
    "value",
    [
        "file:///etc/passwd",
        "http://attacker.invalid/x",
        "../../etc/passwd",
        "/absolute/path",
        "~/.ssh/id_rsa",
        "C:\\Windows\\System32",
    ],
)
def test_no_argument_can_carry_a_path_or_a_url(value):
    spec = APPROVED_TOOLS["retrieve_approved_policy"].argument("query")
    with pytest.raises(GatewayError, match="locator"):
        check_string(value, spec, where="t")


def test_an_ordinary_business_label_with_a_slash_still_passes():
    spec = APPROVED_TOOLS["retrieve_approved_policy"].argument("query")
    assert check_string("Vendor A/B payment terms", spec, where="t") == "Vendor A/B payment terms"


def test_control_characters_are_refused_in_arguments():
    spec = APPROVED_TOOLS["retrieve_approved_policy"].argument("query")
    with pytest.raises(GatewayError, match="control characters"):
        check_string("clearing\x00policy", spec, where="t")


# -------------------------------------------------------- L02 unknown ids
def test_an_unknown_evaluation_id_is_rejected_before_any_record_is_rendered(run, packet):
    gateway = gateway_for(run, packet, broad_policy(run))
    known = run.evidence.records[0].evaluation_id
    with pytest.raises(GatewayError, match="unknown evaluation id"):
        gateway.call("get_evidence", {"evaluation_ids": [known, "run-invented:c99:Z"]})


def test_a_known_evaluation_id_returns_exactly_that_record(run, packet):
    gateway = gateway_for(run, packet, broad_policy(run))
    known = run.evidence.records[0].evaluation_id
    outcome = gateway.call("get_evidence", {"evaluation_ids": [known]})
    assert outcome.ok and outcome.payload["n_returned"] == 1
    assert outcome.payload["records"][0]["evaluation_id"] == known


def test_an_explanation_of_another_slice_is_not_recomputed_on_request(run, packet):
    gateway = gateway_for(run, packet, broad_policy(run))
    with pytest.raises(GatewayError, match="not recomputed"):
        gateway.call("explain_priority", {"group": "A"})


# --------------------------------------------------------- L03 fail closed
def test_a_default_policy_permits_nothing(run):
    policy = empty_policy(run)
    for scope in Scope:
        assert not policy.allows(scope)
    assert list(policy.row_mask(run.cases)) == [False] * len(run.cases)


def test_every_tool_refuses_a_principal_without_its_scope(run, packet):
    gateway = gateway_for(run, packet, empty_policy(run))
    for name in gateway.tools:
        arguments = {
            "get_evidence": {"evaluation_ids": []},
            "compare_groups": {"by": ["company"], "view": "Finance"},
            "retrieve_approved_policy": {"query": "clearing"},
        }.get(name, {})
        with pytest.raises((AccessDenied, GatewayError)):
            gateway.call(name, arguments)


def test_a_run_the_principal_does_not_hold_is_refused(run, packet):
    policy = AccessPolicy(
        Principal("p"),
        scopes=frozenset({Scope.RUN_SUMMARY}),
        runs=frozenset({"run-someone-elses"}),
        views=frozenset(run.views),
        unrestricted_rows=True,
    )
    gateway = gateway_for(run, packet, policy)
    with pytest.raises(AccessDenied, match="not authorised for run"):
        gateway.call("get_run_summary", {})


# --------------------------------------------------------- L03 rows
def test_the_row_mask_narrows_before_any_aggregate_is_formed(run, packet):
    policy = narrow_policy(run, company="A")
    mask = policy.row_mask(run.cases)
    assert set(run.cases.loc[mask, "company"]) == {"A"}
    assert mask.sum() < len(run.cases)


def test_compare_groups_reports_only_the_authorised_population(run, packet):
    broad = gateway_for(run, packet, broad_policy(run)).call("compare_groups", {"by": ["company"], "view": "Finance"})
    narrow = gateway_for(run, packet, narrow_policy(run, company="A")).call(
        "compare_groups", {"by": ["company"], "view": "Finance"}
    )
    assert {row["company"] for row in broad.payload["rows"]} == {"A", "B"}
    assert {row["company"] for row in narrow.payload["rows"]} == {"A"}
    assert narrow.payload["comparator"]["population_size"] < broad.payload["comparator"]["population_size"]


def test_an_unauthorised_record_is_refused_rather_than_quietly_dropped(run, packet):
    policy = narrow_policy(run, company="A")
    other = next(r for r in run.evidence.records if run.cases.loc[r.unit_id, "company"] == "B")
    gateway = gateway_for(run, packet, policy)
    with pytest.raises(AccessDenied, match="not authorised for"):
        gateway.call("get_evidence", {"evaluation_ids": [other.evaluation_id]})


def test_a_row_filter_on_a_column_the_table_lacks_denies_rather_than_being_ignored(run):
    policy = AccessPolicy(Principal("p"), row_filters={"cost_centre": frozenset({"CC1"})})
    assert not policy.row_mask(run.cases).any()


# --------------------------------------------------------- L03 columns
def test_an_unauthorised_grouping_column_is_refused(run, packet):
    gateway = gateway_for(run, packet, narrow_policy(run))
    with pytest.raises(AccessDenied, match="columns"):
        gateway.call("compare_groups", {"by": ["vendor"], "view": "Finance"})


def test_visible_columns_returns_only_what_was_granted(run):
    policy = narrow_policy(run)
    assert policy.visible_columns(["company", "vendor", "flow_type"]) == ["company", "flow_type"]


def test_an_unauthorised_view_is_refused(run, packet):
    gateway = gateway_for(run, packet, narrow_policy(run))
    with pytest.raises(AccessDenied, match="view"):
        gateway.call("compare_groups", {"by": ["company"], "view": "Logistics"})


# ------------------------------------------- L04 the comparator cannot leak
def test_a_restricted_principal_gets_a_comparator_over_their_own_population(run):
    policy = narrow_policy(run, company="A")
    spec, change = authorised_baseline(run, policy, view="Finance")
    scored = authorised_population(run, policy, "Finance")
    assert change is None
    assert spec.population_size == int(scored.sum())
    assert spec.reference_score == pytest.approx(float(run.scores["Finance"][scored].mean()))
    assert spec.reference_score != pytest.approx(float(run.scores["Finance"].dropna().mean()))


def test_an_unentitled_full_population_baseline_is_recomputed_and_relabelled(run):
    whole = wise.explain.BaselineSpec.from_result(run, "Finance", baseline_id=FULL_POPULATION_AGGREGATE)
    policy = narrow_policy(run, company="A")
    spec, change = authorised_baseline(run, policy, view="Finance", requested=whole)
    assert change is not None
    assert change.requested_baseline_id == FULL_POPULATION_AGGREGATE
    assert spec.baseline_id != whole.baseline_id
    assert spec.reference_score != pytest.approx(whole.reference_score)
    assert "authorised" in spec.description
    assert "recomputed" in change.message()


def test_an_entitled_principal_may_be_shown_the_wider_aggregate_unchanged(run):
    whole = wise.explain.BaselineSpec.from_result(run, "Finance", baseline_id=FULL_POPULATION_AGGREGATE)
    policy = AccessPolicy(
        Principal("p"),
        scopes=frozenset({Scope.COMPARE_GROUPS}),
        runs=frozenset({run.manifest.run_id}),
        views=frozenset({"Finance"}),
        row_filters={"company": frozenset({"A"})},
        aggregate_entitlements=frozenset({FULL_POPULATION_AGGREGATE}),
    )
    spec, change = authorised_baseline(run, policy, view="Finance", requested=whole)
    assert change is None and spec is whole


def test_a_principal_with_no_scored_units_gets_a_refusal_not_a_wider_mean(run):
    policy = AccessPolicy(
        Principal("p"),
        scopes=frozenset({Scope.COMPARE_GROUPS}),
        runs=frozenset({run.manifest.run_id}),
        views=frozenset({"Finance"}),
        row_filters={"company": frozenset({"Z-does-not-exist"})},
    )
    with pytest.raises(AccessDenied, match="no scored units"):
        authorised_baseline(run, policy, view="Finance")


def test_the_comparison_payload_says_it_is_not_company_wide(run, packet):
    outcome = gateway_for(run, packet, narrow_policy(run)).call("compare_groups", {"by": ["company"], "view": "Finance"})
    assert "not a company-wide comparison" in outcome.payload["note"]


# ------------------------- L03/L04 the explanation obeys the same boundary
def _policy_over(run, *, company: str | None, principal: str) -> AccessPolicy:
    """Every scope, run, view and column — and rows restricted to one company."""
    return AccessPolicy(
        Principal(principal),
        scopes=frozenset(s.value for s in Scope),
        runs=frozenset({run.manifest.run_id}),
        views=frozenset(run.views),
        columns=frozenset(run.cases.columns),
        row_filters={} if company is None else {"company": frozenset({company})},
        label=principal,
    )


def test_an_explanation_of_a_slice_outside_the_authorised_population_is_refused(run, packet):
    """The packet describes company B; a principal holding only company A's rows
    is not entitled to it, however many scopes they hold."""
    policy = _policy_over(run, company="A", principal="reviewer-company-a")
    assert packet.group_label == "B"
    assert policy.row_mask(run.cases).any(), "the principal does own rows — just none of group B's"
    gateway = gateway_for(run, packet, policy)
    with pytest.raises(AccessDenied, match="authorised"):
        gateway.call("explain_priority", {})


def test_a_principal_with_no_authorised_rows_never_receives_the_population_mean(run, packet):
    """A row filter that matches nothing is an empty population, not the whole one.

    ``PRI-population_mean`` is the run-wide comparator. It must not appear in a
    payload, in the audit trail, or anywhere in the serialised gateway.
    """
    policy = _policy_over(run, company="Z-does-not-exist", principal="reviewer-no-rows")
    assert not policy.row_mask(run.cases).any()
    gateway = gateway_for(run, packet, policy)
    with pytest.raises(AccessDenied):
        gateway.call("explain_priority", {})

    results = gateway.run([ToolCall("explain_priority", {})])
    assert [r.ok for r in results] == [False]
    trail = json.dumps(gateway.to_dict(), default=str)
    assert "PRI-population_mean" not in trail
    assert "0.7098" not in trail, "the whole-population mean must not reach an unauthorised principal"


def test_an_explanation_whose_comparator_is_wider_than_the_population_is_relabelled(run, packet):
    """Authorised for every row of group B and no other: the slice's own facts are
    served, and the run-wide comparator is re-derived and declared changed."""
    policy = _policy_over(run, company="B", principal="reviewer-company-b")
    payload = gateway_for(run, packet, policy).call("explain_priority", {}).payload

    assert payload["group_label"] == "B"
    assert {f["fact_id"] for f in payload["facts"]} >= {"OBS-mean_score", "OBS-n_units"}
    assert not [f for f in payload["facts"] if f["fact_id"].startswith("PRI-")]

    change = payload["comparator"]["changed"]
    assert change is not None
    assert change["requested_baseline_id"] == packet.baseline.baseline_id
    assert payload["comparator"]["baseline_id"] != packet.baseline.baseline_id
    assert payload["comparator"]["reference_score"] != pytest.approx(packet.reference_score)
    assert "PRI-population_mean" in payload["withheld_facts"], "what was withheld is named, not silently dropped"
    assert "0.7098" not in json.dumps(payload, default=str), "the wider population's mean is still not disclosed"


def test_an_explanation_is_served_whole_to_a_principal_who_sees_the_whole_run(run, packet):
    """The fix must not narrow an unrestricted principal: same facts, same comparator."""
    payload = gateway_for(run, packet, broad_policy(run)).call("explain_priority", {}).payload
    assert {f["fact_id"] for f in payload["facts"]} == {f.fact_id for f in packet.facts}
    assert payload["comparator"]["baseline_id"] == packet.baseline.baseline_id
    assert payload["comparator"]["changed"] is None


# ------------------------------------------------------------- L06 budgets
def test_the_tool_call_budget_is_bounded(run, packet):
    ledger = BudgetLedger(CallBudget(max_tool_calls=2))
    gateway = gateway_for(run, packet, broad_policy(run), ledger=ledger)
    gateway.call("get_run_summary", {})
    gateway.call("get_run_summary", {})
    with pytest.raises(BudgetExceeded, match="tool call budget"):
        gateway.call("get_run_summary", {})


def test_a_result_past_its_declared_limit_is_refused_rather_than_returned(run, packet, monkeypatch):
    gateway = gateway_for(run, packet, broad_policy(run))
    assert gateway.call("get_run_summary", {}).ok, "the ordinary summary fits its declared limit"
    monkeypatch.setitem(
        APPROVED_TOOLS, "get_run_summary", dataclasses.replace(APPROVED_TOOLS["get_run_summary"], max_result_bytes=10)
    )
    with pytest.raises(GatewayError, match="bulk export"):
        gateway_for(run, packet, broad_policy(run)).call("get_run_summary", {})


def test_run_turns_every_refusal_into_a_typed_result_without_stopping(run, packet):
    gateway = gateway_for(run, packet, narrow_policy(run))
    results = gateway.run(
        [
            ToolCall("get_run_summary", {}),
            ToolCall("run_sql", {"q": "select 1"}),
            ToolCall("compare_groups", {"by": ["vendor"], "view": "Finance"}),
        ]
    )
    assert [r.ok for r in results] == [True, False, False]
    assert "GatewayError" in results[1].reason and "AccessDenied" in results[2].reason


# ------------------------------------------------------------- provenance
def test_the_run_summary_counts_only_authorised_units(run, packet):
    broad = gateway_for(run, packet, broad_policy(run)).call("get_run_summary", {}).payload
    narrow = gateway_for(run, packet, narrow_policy(run)).call("get_run_summary", {}).payload
    assert narrow["authorised_units"] < broad["authorised_units"]
    assert narrow["views"] == ["Finance"] and set(broad["views"]) == set(run.views)
    assert "not the run's totals" in narrow["population"]


def test_the_explanation_tool_returns_facts_and_performs_no_arithmetic(run, packet):
    payload = gateway_for(run, packet, broad_policy(run)).call("explain_priority", {}).payload
    ids = {f["fact_id"] for f in payload["facts"]}
    assert ids == {f.fact_id for f in packet.facts}
    assert "no arithmetic is performed" in payload["note"]


def test_the_gateway_records_an_audit_trail_that_serialises(run, packet):
    gateway = gateway_for(run, packet, broad_policy(run))
    gateway.call("get_run_summary", {})
    payload = json.loads(json.dumps(gateway.to_dict(), default=str))
    assert payload["tools"] == list(gateway.tools)
    assert payload["policy"]["principal"]["principal_id"] == "reviewer-broad"
    assert payload["ledger"]["tool_calls"] == 1
