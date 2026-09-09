"""Exercise the real contract's attrs assertion without patching runtime code."""

from __future__ import annotations

import ast
import runpy
from pathlib import Path

import pytest

import wise

CONTRACT = Path(__file__).with_name("test_baseline_contract.py")


@pytest.fixture(scope="module")
def check_contract_attrs():
    namespace = runpy.run_path(str(CONTRACT))
    tree = ast.parse(CONTRACT.read_text(encoding="utf-8"))
    test = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "test_backlog_columns_order_and_values")
    checks = [
        n
        for n in test.body
        if isinstance(n, ast.Assert) and isinstance(n.test, ast.Compare) and ast.unparse(n.test.left) == "b.attrs"
    ]
    assert len(checks) == 1
    # Execute the assertion itself, so a regression to exact equality or an
    # over-broad tolerance cannot be hidden by duplicating its expected value.
    code = compile(ast.Module(body=checks, type_ignores=[]), str(CONTRACT), "exec")

    def check(backlog):
        exec(code, namespace, {"b": backlog})

    return namespace, check


@pytest.mark.parametrize("delta, should_pass", [(-5e-13, True), (5e-13, True), (-2e-12, False), (2e-12, False)])
def test_baseline_numeric_cell_accepts_only_the_documented_tolerance(p2p_result, check_contract_attrs, delta, should_pass):
    namespace, check = check_contract_attrs
    backlog = wise.prioritize(p2p_result, "company", view="Finance", gamma=1.0)
    backlog.attrs["baseline"] = namespace["GLOBAL_MEAN_FINANCE"] + delta
    if should_pass:
        check(backlog)
    else:
        with pytest.raises(AssertionError):
            check(backlog)


def test_contract_metadata_still_requires_exact_equality(p2p_result, check_contract_attrs):
    _, check = check_contract_attrs
    backlog = wise.prioritize(p2p_result, "company", view="Finance", gamma=1.0)
    backlog.attrs["view"] = "different-view"
    with pytest.raises(AssertionError):
        check(backlog)
