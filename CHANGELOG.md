# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Repository layout: the library moved to `packages/wise-pm` inside a uv
  workspace with a committed `uv.lock`; the build backend is hatchling and
  `MANIFEST.in` is gone. Install from a checkout with
  `pip install ./packages/wise-pm` or `pip install "wise-pm @ git+https://github.com/feelfine1977/wise-pm.git@main#subdirectory=packages/wise-pm"`.
  The public API, the norm file format and every number are unchanged.
- The `wise-pm` distribution declares the `MIT` licence expression and ships
  only the MIT licence file; rights-controlled additions will live in a
  separate distribution.
- Development tooling: `make setup|test|check|golden|contract|bench|compat|build`,
  pre-commit hooks that run the locked tool versions, `ruff` with an extended
  rule set, `mypy --strict` (the 0.1.0 test modules are temporarily excluded
  and re-enter one by one as they are rewritten), `import-linter` enforcing
  the module layering, CI on Python 3.10–3.13 with floor-pin, macOS and
  Windows jobs plus wheel/sdist smoke tests, a nightly workflow for
  pre-release dependencies, benchmarks, the long Hypothesis profile and a
  dependency audit, and a tag-driven release workflow that refuses a tag whose
  version disagrees with the package, `CITATION.cff` or this changelog.
- New optional extra `io` (pyarrow) for parquet input.

### Added

- Test tiers: regression tests pinning the 16 behavioural defects of the
  2026-09-26 review (C1–C13, C15, C16, C18) as strict `xfail`s, the structural
  findings being tracked in the plan; Hypothesis property tests for the scoring,
  priority and JSON invariants; a golden pipeline snapshot on a committed
  synthetic 2 000-case purchase-to-pay log with planted hotspots; a public-API
  contract snapshot (names, signatures, dataclass fields, output columns);
  a benchmark on a synthetic 1.6 M-event log; `scripts/compat_gate.sh` running
  the downstream application suites against a source tree.

- `docs/decisions/0001-canonical-library-and-migration-map.md` and
  `docs/semantics/parity.md` record the ownership map, the recipe, calibration
  and missingness decisions, and the exact-parity policy for unchanged classic
  arithmetic (the golden tier compares parquet records exactly).
- Regression ledger: C24 (`compare_periods` zero-fills slices absent from one
  period), C25 (a negated applicability rule brings cases with a missing
  attribute into scope), C26 (`prioritize` accepts `baseline=NaN`), C27 (a
  negative `z` inflates `PI_lower` above `stable_PI`) and C28 (`min_cases` is
  truncated to an integer), all strict expected failures until fixed.
- `docs/decisions/known-defects.json`: every expected failure under
  `tests/regression` must cite an open ledger entry (checked at collection
  time); a release candidate with an open S1 entry is refused by
  `scripts/check_release.py`.
- The release workflow now runs the full CI gate for the release commit
  before building.

### Fixed

- `Norm.dump()` and `Norm.to_json(path=...)` end the written file with a newline.
- Three type annotations (`log.py`, `scoring.py`, `prioritization.py`) that
  the current pandas-stubs reject; no behavioural change.

## [0.1.0] — 2026-09-05

First release.

- Constraint catalogue: the threshold–saturation rule and the five constraint
  types of the method (presence, lag, balance, singularity, exclusion), plus
  precedence and metric constraints.
- `Norm`: layers, views with raw or two-stage weights, applicability rules,
  scoring mode, derived-attribute recipes, validation, JSON round-trip and
  fingerprinting.
- `EventLog`: pm4py column conventions, timestamp and lifecycle handling,
  observation window, vectorised case primitives, data-quality report.
- `score`: violation matrix with scope and evaluability masks,
  applicability-aware case scores in two modes, exact layer decomposition,
  per-constraint penalties, drill-down to worst cases and traces.
- Prioritisation: Priority Index with shrinkage, exposure weighting,
  conservative lower bound and fixed baseline; layer and constraint drivers;
  penalty mass; Pareto concentration; view agreement; hotspot typology;
  period comparison; shrinkage-constant estimate.
- Diagnostics: robust observation window, timestamp outliers,
  right-censoring, left truncation, within-case and cross-case event
  replication, gap retention, validation table.
- Command-line interface (`wise validate | describe | check | score`).
- Bundled running example of the paper and the BPIC'19 norm.
