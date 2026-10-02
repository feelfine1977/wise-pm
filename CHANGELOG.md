# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

The 0.2.0 line deepens the library's modules behind their current public
interface; classic numbers do not change. The package version is
`0.2.0.dev0`. `CITATION.cff` stays at the released version 0.1.0 and changes
at release time. The committed records that name the library version that
produced them (the golden manifest, the BPIC'19 oracle and the native
evaluation summary) keep `0.1.0`: they are provenance, the golden check
tolerates a version difference, and no golden value changes. The contract
snapshot writes the version as a placeholder and is unchanged.

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

- `wise.case_index.CaseIndex`, the immutable per-case index behind the event
  log: case codes, timestamps, validity, activity codes and case starts held
  in read-only arrays, with counts for an activity set or an event mask,
  first and last timestamps, the per-case aggregate of an event column, runs
  of equal timestamps and timestamp conversion. `EventLog` computes
  `count`, `first_ts`, `last_ts`, `first_after`, `activation_lags`,
  `count_scoped`, `count_before`, `count_after` and `total` through it; their
  results are unchanged, and building the log copies the event table no more
  often than before.
- `EventLog.case_index` (property) returns that index;
  `EventLog.to_timestamp(value)` converts a value to a timestamp comparable
  with the log's (naive values are localised to the log's time zone, aware
  values become naive UTC on a naive log, `None` stays `None`).

- Tests for `scripts/check_release.py`: a tag that agrees with the package,
  `CITATION.cff` and the changelog passes; a development version, the
  previous release tag and a tag without the `v` prefix are refused.

- Dataset evaluation tools for 15 configurations and 44 primary views: portable
  source catalogue, raw-data adapters, explicit domain configurations, native
  prepared-assessment norms, parity checks and an interactive notebook. Full
  logs and source-derived case outputs remain local and ignored. The core API,
  historical BPIC19 norm and golden records are unchanged.

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
  arithmetic (the golden tier compares parquet records exactly). The decision
  record also names the capability set, the contracts the application
  consumers wait for, the executed parity baseline, the packaging fallback and
  the disposition of the workbench analytics package between the MIT core and
  the noncommercial extension.
- Golden tier: the manifest records the environment that produced the record
  (system, machine, Python, numpy, pandas, BLAS, SIMD set); the golden test's
  default `auto` mode compares exactly on that host and within 1e-9 elsewhere,
  after CI run 10 showed last-bit drift on the ubuntu runner (seven frames,
  reduction order and fused multiply-add). `make_golden.py --check --mode
  report` gives CI a same-runner determinism proof and a drift report; the
  release check reproduces the goldens exactly on the maintainer's host.
- Regression ledger: C24 (`compare_periods` zero-fills slices absent from one
  period), C25 (a negated applicability rule brings cases with a missing
  attribute into scope), C26 (`prioritize` accepts `baseline=NaN`), C27 (a
  negative `z` inflates `PI_lower` above `stable_PI`) and C28 (`min_cases` is
  truncated to an integer) and the policy entry C29 (list-valued `where=`
  as a membership), all strict expected failures until fixed.
- `docs/decisions/known-defects.json`: every expected failure under
  `tests/regression` must cite an open ledger entry (checked at collection
  time); a release candidate with an open S1 entry is refused by
  `scripts/check_release.py`.
- The release workflow now runs the full CI gate for the release commit
  before building.

### Deprecated

- `EventLog._to_ts` emits a `DeprecationWarning` and delegates to
  `EventLog.to_timestamp`; it is removed in 1.0. The arrays `EventLog._codes`,
  `_ts`, `_valid` and `_starts` remain readable, but are now read-only views
  of the case index.

### Fixed

- `Norm.dump()` and `Norm.to_json(path=...)` end the written file with a newline.
- Three type annotations (`log.py`, `scoring.py`, `prioritization.py`) that
  the current pandas-stubs reject; no behavioural change.
- Fixed: C26 — `prioritize()` refuses `baseline=NaN` (and any non-finite baseline) with a `NormError` instead of returning null gaps and priorities. Delta on the golden dataset: none (the input was refused; every valid number is unchanged).
- Fixed: C27 — `prioritize()` refuses a negative or non-finite `z`; a negative multiplier turned `PI_lower` into an upper bound. Delta on the golden dataset: none.
- Fixed: C28 — `prioritize()` refuses a fractional, negative or non-finite `min_cases` instead of truncating it (an integral float such as `5.0` is still accepted). Delta on the golden dataset: none.
- Fixed: C9 — every public entry point raises a `WiseError` subclass for a bad argument: `EventLog` refuses an unknown `missing_timestamps` option, a reversed observation window and an unknown `exposure_agg` with `LogSchemaError` (the first two were bare `ValueError`s, the last an `AttributeError` from pandas); `Norm.dumps()`/`fingerprint()` serialise numpy arrays and pandas `Timedelta` metadata and raise `NormError` for anything else (was `TypeError`); `Norm.loads()`/`load()`/`from_json()` raise `NormError` for malformed JSON (was `json.JSONDecodeError`); `hotspot_table()` and `compare_periods()` refuse frames that are not backlogs with `NormError` (was `KeyError`); `timestamp_outliers()` accepts a half-open window `(None, end)` or `(start, None)` (was `TypeError`). Delta on the golden dataset: none.
- Fixed: C11 — `LogSchemaError` no longer inherits `KeyError`, and `NormError` and `NotScoredError` no longer inherit `ValueError`; an `except KeyError:` or `except ValueError:` in a caller cannot swallow a library error any more. Callers that relied on the builtin bases must catch the `wise` classes (or `WiseError`); the workbench already does. The public-API snapshot records the new bases. Delta on the golden dataset: none.

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
