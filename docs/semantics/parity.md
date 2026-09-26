# Parity and tolerance policy

| Class of computation | Comparison | Enforced by |
|---|---|---|
| Determinism of one implementation | byte-identical artifacts across two runs (parquet tables, manifest without run id and timestamp) | golden generator `--check`; artifacts round-trip tests |
| Unchanged classic arithmetic (violations, effective weights in both modes, scores, contributions, backlogs, drivers, agreement) | exact equality: identical values, NaN masks, row and column order, on the record environment (verified bit-identical on one macOS host across the Python 3.10 and 3.13 dependency sets and the floor pins; the CI runners have not yet executed the tier, and two products in the kernel go through BLAS) | `tests/golden` with `check_exact=True` (done, uncommitted at the time of writing); from M2 on, a table may differ only when a listed correction id is attached to it in the golden manifest, together with a per-constraint delta report in the CHANGELOG |
| Kernel refactors (vectorised effective weights, shared object-centric kernel, accelerated paths) | 0 ULP against the reference implementation on random and degenerate matrices; the reference path stays selectable until then | parity tests next to the kernel |
| Algebraic identities (layer decomposition, sum rule, scale freedom) | 1e-12 absolute | property tests |
| Semantic corrections (C2 lag default, C4/C25 missing-attribute scope, C15 calibration, C16 eval, C24 period zero-fill) | a positive spec test with the expected value cited to the paper section or decision record; a delta report on the golden dataset; the legacy profile keeps v2 documents exact | regression ledger rule in CONTRIBUTING |
| New numerical methods (sampling intervals, contrasts, sensitivity) | the method declares its own tolerance and design (Monte Carlo coverage within a stated band, seed-reproducible); never relaxes the rows above | the method's own tests |
| Environment-induced drift (a different BLAS, SIMD reduction order, a new numpy or pandas release) | not a semantic change and never a correction id: the exact gate runs on the record environment (the CI reference cell: ubuntu, Python 3.13, the lockfile); every other CI cell and the nightly pre-release job run the golden tier in report mode (`WISE_GOLDEN_MODE=report`: within 1e-9, largest difference logged); a drift on the record environment after a dependency update is re-baselined with the label `environment` in the CHANGELOG and the manifest's recorded versions | `ci.yml`, `nightly.yml`, golden manifest versions |

A strict expected failure that turns green shows that the old behaviour is
gone; it does not show that the new behaviour is right. Each closed defect
therefore replaces its expected-failure test with a positive assertion.
