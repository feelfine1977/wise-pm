# Baseline contract provenance and portability

The executable contract remains `tests/extensions/test_baseline_contract.py`
with the same 44 tests, public signatures, metadata keys, ordering, masks and
fingerprints. Revision 2 changes one expected numeric cell in
`test_backlog_columns_order_and_values`: the `baseline` attribute now uses
`pytest.approx(GLOBAL_MEAN_FINANCE, rel=TOL["rtol"], abs=TOL["atol"])`.
All other contract bytes are unchanged. This is a correction to the contract,
not a change to scoring or prioritization arithmetic.

| Contract | SHA-256 |
|---|---|
| Original v1 | `a47445b232fe3b3626e7bf8c788a918dfd813aba4d978164060f1b1a975dda5b` |
| Current v2 | `f66d8dd3827871462e21176a52ef9b959876d83d73ca54f7e57a0972c204e243` |

V1 is preserved in Git history at extension commit
`819d9849d7be7c96f79337390677e2138d56ee6b`. The historical validation records
retain the original digest and their original results. CI verifies the v2
file digest and requires classic reference commit
`df5db50b839cc124b489a269894f5a2bfe7dc634` in full history. LF checkout rules
prevent Windows line-ending conversion from changing these pinned bytes.

## Why one assertion changed

[The public CI run](https://github.com/feelfine1977/wise-pm/actions/runs/34407145639)
reported a Linux baseline of `0.7098333333333333` against the literal
`0.7098333333333334`: one ULP, about `1.11e-16`. The contract already documents
`rtol=0, atol=1e-12` for numerical values, and its other floating-point checks
passed. This dictionary cell had accidentally required exact equality.

Independent Linux x86-64 reproduction on Python 3.12.14, NumPy 2.5.3 and
pandas 3.0.5 compared a temporary archive of the unchanged classic reference
against pushed extension HEAD. OpenBLAS's Nehalem backend produced the literal
value on both; the Haswell backend produced the alternate value on both.
Finance score arrays were bit-identical between base and extension under
each backend. With Haswell, the original v1 contract failed the same assertion
on both sources: 43 passed, 1 failed. This establishes a portability defect
in the assertion rather than an extension arithmetic regression.

Backend selection was used only for diagnosis and regression reproduction.
CI does not force a backend, round results, modify equality, skip this test,
or relax any structural check. The shared-kernel bit-parity tests and all 180
historical prioritization configurations remain exact.

## Regression checks

`tests/extensions/test_baseline_contract_portability.py` executes the actual
contract attribute assertion against a copied backlog. Positive controls
accept sub-tolerance numeric differences. Negative controls change only the
baseline by `+2e-12` or `-2e-12` and require failure; a changed view must also
fail. These checks do not patch production functions or pytest equality.

```bash
python -m pytest -q tests/extensions/test_baseline_contract.py tests/extensions/test_baseline_contract_portability.py
```

The recorded-assistance harness also compares numeric markers within `1e-12`
with zero relative tolerance, including values rendered as text. Required
and forbidden markers use the same numerical rule; every previously detected
literal forbidden marker still fails. Access policies, refusals, and the
served negative control remain tested against the real gateway.
