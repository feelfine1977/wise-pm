# Contributing

Thanks for your interest in improving `wise`.

## Development setup

```bash
git clone https://github.com/feelfine1977/wise-pm.git   # or your fork
cd wise-pm
make setup        # uv sync --all-packages --group dev && pre-commit install
make test         # the fast suite: unit, regression, property, golden, contract, paper
make check        # ruff check + format, mypy, import-linter, pre-commit hooks
```

The workspace is managed by [uv](https://docs.astral.sh/uv/) with a committed
`uv.lock`; `make lock` refreshes it after a dependency change. The library
lives in `packages/wise-pm` (import name `wise`); shared tool configuration
is in the root `pyproject.toml`.

## Ground rules

- The paper's equations are the specification. A change to scoring or
  prioritisation semantics needs a test carrying `@pytest.mark.spec("<section>")`,
  an entry in `CHANGELOG.md` under *Unreleased*, and, when a public signature or
  output column changes, a refreshed contract snapshot (`make contract`).
- Write the failing test first. Confirmed defects are pinned by tests under
  `tests/regression/` marked `xfail(strict=True)`. Fixing one is not done when the
  marker comes off: replace it with a positive assertion of the intended
  behaviour citing the paper section or decision record, and attach the delta
  report on the golden dataset to the CHANGELOG entry (`docs/semantics/parity.md`).
- The golden pipeline snapshot (`tests/golden/`, regenerated with `make golden`)
  must not change unless the CHANGELOG explains why; review its diff.
- Keep the core dependency-light: `numpy` and `pandas` only. Integrations go
  behind optional extras (`pm4py`, `stats`, `io`) or into separate packages.
- Imports point downward through the layers listed in `[tool.importlinter]`;
  `make check` fails on a violation.
- Nothing in commits, pull requests or files states how the code was produced;
  commit messages describe the change.

## Test tiers

| Marker / folder | Purpose | Runs in |
|---|---|---|
| `tests/test_*.py` | unit tests of 0.1.0 | every push |
| `regression` (`tests/regression/`) | confirmed defects, strict xfail until fixed | every push |
| `property` (`tests/property/`) | Hypothesis invariants (`HYPOTHESIS_PROFILE=ci-long` nightly) | every push / nightly |
| `spec("golden")` (`tests/golden/`) | full pipeline on a committed synthetic log | every push |
| `spec("contract")` (`tests/contract/`) | public-API signature and output-column snapshot | every push |
| `paper` | worked examples of the paper | every push |
| `bpic19` | Section V reproduction; needs `WISE_BPIC19_CSV` | locally before a release |
| `bench` (`tests/perf/`) | timing thresholds; needs `WISE_RUN_BENCH=1` | nightly |

`scripts/compat_gate.sh` runs the downstream application suites against this
tree; run it before merging a change to a public symbol.

The coverage floor is 92 % (branch coverage; 92.5 % at the time of writing), so
new code needs tests before it lands; `uv run --frozen --no-sync pytest --cov --cov-branch --cov-report=term-missing:skip-covered`
shows the margin. Hypothesis example databases are not committed; the nightly
`ci-long` profile provides depth.

## Releasing

See `docs/PUBLISHING.md`. The tag, `wise.__version__`, `CITATION.cff` and the
first CHANGELOG heading must agree; `scripts/check_release.py v<version>`
checks that and the release workflow refuses otherwise.

## Licensing of prospective contributions

Read [COMMERCIAL_LICENSING.md](COMMERCIAL_LICENSING.md) before submitting.
The `wise-pm` distribution is MIT. Rights-controlled additions live in a
separate distribution under their own licence file. Identify pre-existing or
third-party material and retain its licences and attributions. Disclose any
employer, university or other approval required for the proposed grant; do
not assume that authorship alone establishes authority.

Submission under the repository policy does not transfer ownership or grant a
separate commercial exception. Before accepting work intended for a separately
licensed commercial offering, the maintainer must establish and document the
necessary permissions from the relevant rights holder(s), using a separate
agreement where needed. This contribution guide is not a CLA and makes no
claim that commercial relicensing rights have already been secured.
