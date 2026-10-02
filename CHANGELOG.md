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
- Derived-attribute recipes and the diagnostics compute through the public
  interface of the event log and its case index only; their results are
  unchanged. The record of which recipe produced a derived attribute, used to
  reuse an unchanged recipe, moved from the private `EventLog._recipe_cache`
  into `wise.derive`; reuse behaves as before. `wise.observation_window` is
  documented as an alias of `EventLog.observation_window`.
- `EventLog(exposure_agg=...)` and `LogSchema(exposure_agg=...)` take one of
  the names of `ExposureAggregation` (the pandas reductions that give one
  number per case: max, sum, first, last, min, mean, median, prod, std, var,
  sem, skew, count, nunique, size, any, all), checked even when no exposure
  column is set; `idxmin`/`idxmax` (which gave a row label as exposure) and
  callables are refused with a `LogSchemaError` that lists the names. An
  unknown `missing_timestamps` is refused with "must be one of ['raise',
  'drop', 'keep']". Delta on the golden dataset: none.
- `EventLog.validate()` reports `varying_within_case[<attribute>]` from the
  record of the parse step, for the case attributes of the schema. An
  attribute attached later with `add_case_attribute` or `derive` whose name is
  also an event column is no longer compared with that column. Delta on the
  golden dataset: none.
- `score()` no longer writes into the log it is given. The norm's derived
  attributes are computed for the run (`wise.derive.derived_log`) and
  returned with the result: `ScoreResult.cases` holds them, and
  `ScoreResult.log` is the log the norm was evaluated on, i.e. the scored log
  itself when nothing had to be derived, otherwise a new log that shares its
  events and case index and adds the derived attributes. Afterwards the scored log's
  case table, attribute list and recipe record are as they were. An attribute
  attached with `EventLog.derive` by the same recipe is still reused, and
  attributes computed by a run are remembered for that log until its case
  table changes, so scoring twice computes them once. A recipe named like a
  case attribute it did not produce is refused (see `Fixed: C3`).
  `EventLog.derive` and `EventLog.add_case_attribute` still change the log they
  are called on; their docstrings say so. An attribute that `add_case_attribute`
  replaces no longer counts as produced by a recipe.
- `EventLog.events`, `EventLog.cases` and `EventLog.case_attributes` are now
  properties that hand out a copy (for the two frames a shallow copy under
  copy-on-write, always on from pandas 3, so no data is copied until it is
  written to; a full copy on pandas 2 without copy-on-write, which would share
  the data otherwise); `EventLog.trace` hands out a copy the same way; `events` and
  `case_attributes` cannot be assigned, assigning `cases` is deprecated.
  Writing into a frame obtained from a log, adding a column to it or
  appending to the attribute list no longer changes the log's later answers
  (the event-log side of C7; the entry stays open for the norm objects).
  Code that added an event column through `log.events[...] = ...` or a case
  column through `log.cases[...] = ...` must build the column into the input
  frame, call `add_case_attribute`, or use `with_case_attributes`.
  `log.events = ...` raises `AttributeError`. `add_case_attribute` gives the
  log a new case table instead of writing into the old one, so a
  `copy.copy` of a log no longer shares added attributes with the original.
  Cost on the 1.6 M-event benchmark: within run-to-run noise (build 1.79 → 1.80 s,
  score 1.12 → 1.07 s, total 3.99 → 4.10 s, peak 0.3197 → 0.3198 GB; medians of three runs, pandas 3);
  on pandas 2.3.3 with full copies also within noise (score 2.03 and 2.66 s → 2.25 and 2.21 s in two
  runs each, peak 0.4678 → 0.4698 GB).
- Each constraint type is evaluated by its family, one module per type under
  `wise.families`, found through the registry of `wise.family`.
  `wise.scoring` and `Norm.check` no longer switch on the constraint type:
  scoring asks the family for the values, and the norm's check of activities,
  case attributes and event columns asks it what a constraint reads. Each
  family parses its own parameters from a norm document (`constraint_from_dict`
  delegates to it, with the same messages); `CONSTRAINT_TYPES` and
  `wise.constraints.TYPE_ALIASES` are the registry's tables, in the same order
  as before (alias order aside). `evaluate_constraint`, `violation_matrix` and
  `score` return exactly what they returned before (checked on 300 generated
  logs and norms and on the golden dataset, values, NaN masks and dtypes);
  `Norm.check`, `attributes` and `activities` give the same answers. The
  private `wise.scoring._evaluate`, `_evaluate_lag` and `_lag_missing` are
  gone; the Lag rule moved unchanged into `wise.families.lag`. Delta on the
  golden dataset: none.
- Lag and the `lag` recipe pair activations with responses by one rule,
  `wise.families.lag.pairs` (activation first, last or each; response
  first_after or first_overall; lags in the given unit; activation times in
  the log's time zone in every mode). A Lag case that has
  no value says why: `missing_activation` (no `a`, `missing_a="skip"`) or
  `missing_response` (no `b` follows, `missing_b="skip"`); 0.1.0's NaN cases
  of Lag were `not_evaluable`. A case scored by the age of an open activation
  (`missing_b="censor"`) has the reason `censored`, a reason with a value:
  `Evaluation` now requires a value exactly where the reason is one of
  `wise.family.VALUED` (`evaluated`, `censored`), and `Evaluation.of` applies
  a `why` pair with such a reason to cases with a value. Values, NaN masks and
  scores are unchanged (checked against 0.2.0.dev0 before this change on 150
  generated naive, time-zone-aware and partly untimed logs for every option
  of Lag and the recipe, and on the golden dataset). The recipe still reads
  an `activation` other than `"first"` as the last `a` and a `response` other
  than `"first_overall"` as `"first_after"`, as 0.1.0 did. Delta on the
  golden dataset: none.

- The where-clauses of derived-attribute recipes and the `where=` filters of
  `ScoreResult.worst_cases`, `constraint_drivers` and `penalty_mass` are parsed
  and evaluated by `wise.selection`, under the legacy missing-value policy.
  Every recipe clause 0.1.0 accepted (`{"column": c, op: v}` with the first of
  `in`, `not_in`, `regex`, `eq` taken and other keys ignored) selects the same
  events (a compiled `regex` pattern is used as given, as in 0.1.0), and a
  `where=` dict of scalars (equality, the case id under the name
  of the case index) or a boolean Series selects the same cases. Both now also
  take a rule of the selection grammar (`all`, `any`, `not`, the leaf
  operators, `has`, `lacks`); a recipe clause reads event columns and tests
  `has`/`lacks` on the event's case. A `where=` key named `all`, `any`, `not`,
  `attr`, `has` or `lacks` is read as a rule, not as an attribute. Refusals:
  a `where=` that is neither a mapping, a boolean Series nor `None` and an
  empty list value raise `NormError` (0.1.0 raised `AttributeError` or
  `ValueError`); a recipe clause with an invalid `regex` string raises
  `NormError` when it is evaluated (0.1.0 raised the string backend's error,
  `re.error` or `pyarrow.lib.ArrowInvalid`). A case
  whose `where=` comparison is unknown (`<NA>` in a nullable column) is not
  selected. Public signatures, output columns and the contract snapshot are
  unchanged. The private `wise.scoring._where_mask` is gone; `prioritization`
  and `scoring` share `wise.selection.select_cases`. In the import layers
  `wise.selection` moved below `wise.derive`. Delta on the golden dataset:
  none.
- `prioritize()` returns a `wise.Backlog` record instead of a DataFrame
  (return annotation `pd.DataFrame` -> `Backlog`). `Backlog.table` is,
  column for column and row for row, the frame 0.1.0 returned (the golden
  backlog and hotspot records are unchanged); `Backlog.spec` is a frozen
  `wise.BacklogSpec` with `by`, `view` (the resolved view, `None` for a
  per-case frame), `gamma`, `volume`, `baseline` (as requested, `None` for the
  global mean), `global_mean` (the realised `μ̄`), `min_cases`, `z`,
  `norm_fingerprint` (`None` for a per-case frame) and `score_col`. The
  parameters are no longer written into `DataFrame.attrs` (see `Fixed: C19`).
  The helpers take the record: `layer_drivers(data, by=<Backlog>)` and
  `view_agreement(result, by=<Backlog>)` use its slice keys and view (and,
  for `view_agreement`, its `gamma`, `volume`, `baseline`, `min_cases` and `z`);
  `constraint_drivers(result, view=<Backlog>)` and
  `penalty_mass(result, view=<Backlog>)` use its view and slice keys
  (`penalty_mass`'s `by` now defaults to `None` and is required unless a
  record is given); `pareto`, `concentration`, `hotspot_table` and
  `compare_periods` accept a record (a record is not re-checked for the
  backlog columns); `top_k_overlap` matches the slices of two records on
  their slice keys (records sliced by different keys are refused with
  `NormError`, and `as_index=False` backlogs are now matched by key instead
  of by row position). A record ranked on another norm than the result it is
  used with (different norm fingerprint), or a `view=` that is not the
  record's, is refused with `NormError`. Frames are accepted as before. The
  contract snapshot records the new return type, the widened parameter
  annotations and the two new classes; every output column is unchanged.
  `wise.backlog` sits directly below `wise.prioritization` in the import
  layers. Delta on the golden dataset: none.
- `compare_periods(previous, current, *, intended=())` refuses two backlog
  records whose specifications differ in slice keys (`by`), `view`, `gamma`,
  the requested `baseline` or `volume`, with a `NormError` that names each
  differing field with both values; a caller who means the difference names
  the fields in `intended` (unknown names are refused). Frames carry no
  specification and are compared as before; a record ranked with
  `as_index=False` is aligned on its slice keys instead of its row
  positions. The output gains the column
  `presence` (`"both"`, `"new"`, `"gone"`), and a slice ranked in one period
  only keeps missing `gap_change` and `PI_change` (see `Fixed: C24`).
- A case without a value for a slice key falls in the slice labelled
  `wise.MISSING_KEY` (`"(missing)"`, the label the workbench gateway already
  uses) in `prioritize`, `layer_drivers`, `penalty_mass`, `gap_retained` and
  `validation_table` instead of a NaN index entry; that slice ranks after
  every keyed slice on a tie, as NaN did. A numeric key column
  with missing values becomes `object`, a categorical one gains the label as
  its last category. A key column that has missing values and the value
  `"(missing)"` is refused with `NormError` (see `Fixed: C23`).
- `diagnostics.gap_retained` and `validation_table` compute through
  `prioritize`, the backlog record and `case_slices`; their results are
  unchanged. `wise.constraints._UNITS` is now the public, read-only
  `wise.constraints.TIME_UNITS`; `derive` and the Lag family use it. No module
  of the package imports an underscore name from another one (see
  `Fixed: C21`).

- Recipes are parsed when the norm is built (or, for `EventLog.derive` and
  `compute_recipe`, before the first one is computed, so a malformed recipe
  leaves the log as it was). A where-clause without an operator is refused
  then with `NormError`; a clause that also named an unknown event column
  raised `LogSchemaError` when computed. Every refusal names the recipe and
  its kind (`derived attribute 'x' (lag): ...`). An unknown event column or
  case attribute is still refused when the recipe is computed, as before.
  An attribute derived by a recipe is reused for an equivalent document
  (`"activities": "A"` and `["A"]`, an omitted option and its default).
  Every recipe computes exactly what it computed before: on the golden log
  and on 150 random logs (naive and Europe/Berlin, events without timestamp
  kept) every kind with every valid option, 21 745 results, are identical to
  0.1.0. Delta on the golden dataset: none.

### Added

- `wise.case_slices(data, by, view=None, score_col="score")`: the slice each
  case falls in, as `prioritize` forms it (missing keys labelled), and its
  score, one row per case indexed like the cases; `by` may be a backlog
  record, whose keys, view and score column are used; its frame can be
  ranked again, also when a numeric key mixes numbers and the label.
  `wise.MISSING_KEY`.
  The contract snapshot gains these two names.
- `wise.evaluate(constraint, log)` evaluates a constraint, or a
  `NormConstraint` with its applicability, on every case and returns a
  `wise.Evaluation`: `value` (`ν_c(σ)`, NaN without a value) and `reason` per
  case, `evaluated` if there is a value (`censored` where Lag scores an open
  activation by its age), `out_of_scope` outside the applicability, otherwise
  why the case is not evaluable (`missing_anchor`, `missing_activity`,
  `missing_attribute`, `missing_activation`, `missing_response`, or
  `not_evaluable` where the family names no reason) as a categorical Series;
  `in_scope` and `evaluated` give the masks. Scoring time on the 1.6 M-event
  benchmark is unchanged (score 0.96 → 0.95 s, total 4.10 → 3.97 s, peak
  0.3198 GB; medians of three runs).
  `evaluate_constraint` and `violation_matrix` are computed from it. The
  public-API snapshot gains these two names and nothing else.
- `wise.family`: the constraint-family interface (`Family` with `parse`,
  `requires` returning `Inputs` of activities, case attributes and event
  columns, and `evaluate`) and the registry (`register`, `unregister`,
  `family`, `family_of`, `families`). A new constraint type takes one module
  with its parameter class and its `Family`, and one `register` call; scoring,
  the norm document and the norm's check then know it.

- `wise.recipes`: a derived-attribute recipe is parsed into one frozen record
  per kind (`CountRecipe`, `CountEventsRecipe`, `NuniqueRecipe`, `AggRecipe`,
  `CvRecipe`, `LagRecipe`, `RatioRecipe`, `QuantileScaleRecipe`,
  `IndicatorTimesRecipe`, and the new kind `column`, `ColumnRecipe`, which
  exposes an existing case-table column as a number) by `parse_recipe`. A
  record states what it reads (`requires()` returns the `wise.family.Inputs`
  of activities, case attributes and event columns) and computes one value
  per case through the log's kernel (`compute(log)`); `to_dict()` gives its
  canonical document. `Norm.recipes` holds the records of a norm, in the
  order of `derived_attributes`, which keeps the document form unchanged (the
  public-API snapshot gains this property and nothing else). The norm's
  validation and `Norm.check` ask the records and no longer read recipe keys;
  `check` also reports the activities a recipe where-clause tests with
  `has`/`lacks`. `wise.derive.compute_recipe`, `derived_log`, `apply_recipes`
  and `EventLog.derive` accept records as well as documents.

- `wise.selection` grows the leaf operator `regex` (the value's text searched
  with the pattern, as in the recipe where-clauses of 0.1.0; an invalid
  pattern is refused with `NormError`), so applicability rules accept it too;
  `parse_recipe_where` (the recipe where-clause), `parse_where` (the `where=`
  filter: a scalar is equality, a list, tuple or set is membership) and
  `select_cases(cases, where, *, log=None)` (the plain case mask of a
  `where=` filter); `CaseTable(..., case_ids=True)` reads the case ids under
  the name of the case index; `RECIPE_WHERE_OPERATORS`.

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

- `EventLog.with_case_attributes(values)` returns a new log with the given
  case attributes attached; the log it is called on is not changed, and the
  two share their events and case index. `wise.derive.derived_log(log,
  recipes)` returns the log a norm is evaluated on, with the recipes'
  attributes added the same way.

- `wise.LogSchema`, `wise.parse` and `wise.ParsedLog` (module
  `wise.log_schema`): building a log is separate from querying it. A frozen
  `LogSchema` names the columns by role (case, activity, timestamp, case
  attributes, exposure, order, event id, lifecycle with its kept
  transitions, kept columns, timestamp format) and carries every policy as an
  enumerated value with a documented default: `missing_timestamps`
  (`MissingTimestamps`, `raise`), `duplicates` (`Duplicates`, `keep`),
  `time_zone` (`TimeZone`, `as_parsed`), `date_order` (`DateOrder`,
  `month_first`), `exposure_agg` (`ExposureAggregation`, `max`) and
  `case_attribute_values` (`CaseAttributeValues`, `first_non_null`).
  `LogSchema.digest` is a SHA-256 over the roles and policies; it changes
  when, and only when, one of them changes. `LogSchema.from_options(...)` maps
  the column and policy options of `EventLog` onto a schema. `parse(events,
  schema)` returns the sorted events, the case index and the case table.
  `EventLog(...)` is now a façade over both and builds the same log as before
  for every combination of its options. It takes an optional
  keyword-only `schema=` in place of those options and exposes the schema it
  was built with as `EventLog.schema`. `EventLog.from_csv`, `from_pm4py` and
  `from_xes` go through the same parse step.

- Two rules of the method's input became schema policies, documented with
  one table in `docs/semantics/log-policies.md`:
  `LogSchema(untimed_events=...)` (`wise.log_schema.UntimedEvents`, the last
  keyword of `LogSchema` and its last dataclass field) states what an event
  kept without a timestamp contributes to every primitive of the case index
  and the event log. `legacy` (default) is the 0.1.0 rule; `uniform` counts
  an untimed event wherever no time is compared. The two differ only in
  `EventLog.count_scoped` without `after` and `before`, and so in the
  `count` recipe. `CaseAttributeValues.STRICT` (`case_attribute_values=
  "strict"`) refuses a log in which a case attribute takes two non-null
  values within one case, naming each attribute and its number of cases.
  `ParsedLog.varying_case_attributes` (keyword-only, a read-only mapping)
  records, under the default, each case attribute that varies within a case
  with its number of such cases. Both new policies enter
  `LogSchema.digest`, so every schema digest differs from the one of the
  previous development state.

- `wise.selection`: one rule grammar for selecting rows. A rule (the short
  form or the rule form of applicability) is parsed once into a small tree
  that states the columns and activities it needs, and is evaluated over a
  table to in scope, out of scope or unknown (a nullable boolean Series). Two
  adapters: `CaseTable` (one row per case) and `EventTable` (one row per
  event; `has`/`lacks` test the event's case). The missing-value policy is an
  explicit argument: `"legacy"` reproduces 0.1.0 for every operator,
  `"three_valued"` makes a missing attribute unknown under every operator
  including negation, as decision 0001 prescribes. Applicability
  (`NormConstraint.applies_to`) now parses and evaluates through this module
  under the legacy policy; its masks, the norm file format and every number
  are unchanged. Nothing selects under the three-valued policy yet.

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
- Assigning `EventLog.cases` emits a `DeprecationWarning` (removed in 1.0;
  use `add_case_attribute` or `with_case_attributes`). The assigned table
  must have the log's case ids as its index, in order, or a `LogSchemaError`
  is raised. The log keeps a copy of it.
- Using the `Backlog` returned by `prioritize()` as a DataFrame (indexing,
  column assignment, `len`, iteration, `in`, and any DataFrame attribute such
  as `.loc`, `.index`, `.reset_index()` or `.attrs`) still works and emits a
  `DeprecationWarning` naming `Backlog.table` and `Backlog.spec`; removed in
  1.0. A frame obtained this way carries the 0.1.0 `attrs` (`view`, `gamma`,
  `baseline` = the realised global mean, `volume`, `by`), so code that filters
  the backlog and reads `attrs["baseline"]` keeps working; the record's own
  table carries none. `np.asarray(backlog)` gives the frame's values. Column
  assignment reaches the record's table; any other in-place change on this
  path (`backlog.loc[...] = ...`, `inplace=True`) acts on a copy and must be
  made on `backlog.table`. `isinstance(backlog, pd.DataFrame)` and pandas functions
  that take a frame (`pd.concat`, `pd.testing.assert_frame_equal`) need
  `backlog.table`.

### Fixed

- Fixed: C1 — `constraint_from_dict` (and so `Norm.from_dict`, `load` and
  `loads`) truncated a non-integer count before the constructor checked it, so
  a document with `m: 2.7` built `Presence(m=2)` and `k: 0.9` built
  `Singularity(k=0)`. The counts `m` and `k` of Sec. IV-B are integers: a
  document now refuses a non-integer `m` (Presence) or `k` (Singularity,
  Precedence) with the same `NormError` as the constructor ("m must be an
  integer >= 1, got 2.7"); an integral value such as `2.0` or `"2"` is still
  accepted as `2`. A boolean, a list or a non-finite count is refused with a
  `NormError` from the constructor instead of an escaping `TypeError` or
  `ValueError`. Delta on the golden dataset: none (its counts are integers;
  every table is unchanged).

- Fixed: C5 — recipe options were not validated: an unknown key (a typo
  such as `activitis`) was ignored, the `lag` recipe read any `activation`
  other than `"first"` (including `"each"`) as the last activation and any
  `response` other than `"first_overall"` as `"first_after"`, `agg` handed any
  string to pandas (`"describe"` failed deep inside it), and
  `allow_negative: "false"` meant true. Each recipe is now parsed strictly
  into its typed record (decision 0001, section 3) and the norm is refused
  with a `NormError` that names the recipe and the key: an unknown key (the
  allowed keys are listed), an unknown option value (`activation` is `first`
  or `last`, `response` `first_after` or `first_overall`, `unit` a time unit
  of `Lag`, `agg` one of the reductions of `ExposureAggregation`, `q` within
  `[0, 1]`) and a value of the wrong type (labels must be strings, numbers
  must be finite numbers and not text, flags booleans). Norms that relied
  on a lenient reading must be corrected. Delta on the golden dataset: none
  (its one recipe is valid; every table is unchanged).
- Fixed: C16 — an `eval` recipe executed the pandas expression of a norm
  file. Under schema version 2 an `eval` recipe is accepted only when its
  expression is a single case-table column name (`n_events`, or a
  back-quoted name) and is then the `column` recipe of that column (decision
  0001, section 4); any other expression is refused when the norm is loaded,
  with the recipe named. No expression is evaluated. The one shipped use,
  `total_events` (`n_events`) in the BPIC'19 norm, gives the same values,
  and both shipped norms load and round-trip byte-identically. Delta on the
  golden dataset: none (its norm has no `eval` recipe; every table is
  unchanged).
- Fixed: C29 — a list value in `where=` (`ScoreResult.worst_cases`,
  `constraint_drivers`, `penalty_mass`) is a membership test, as in the
  short form of the selection grammar (decision 0001: one owner for
  selection); 0.1.0 compared the column element-wise with the list and raised
  `ValueError` ("Lengths must match") unless the list had one entry per case.
  `{"company": ["A", "B"]}` now selects the cases of both companies. A scalar
  value still means equality. Delta on the golden dataset: none (the golden
  pipeline passes no `where=`; its one recipe where-clause, a `regex`, selects
  the same events, and every table is unchanged).
- `Norm.dump()` and `Norm.to_json(path=...)` end the written file with a newline.
- Three type annotations (`log.py`, `scoring.py`, `prioritization.py`) that
  the current pandas-stubs reject; no behavioural change.
- Fixed: C6 — the parse step refuses input that 0.1.0 built into a silently
  wrong or empty log, with a `LogSchemaError` that names the problem: an event
  frame without rows; a lifecycle column with null transitions (they never
  matched `keep_transitions` and were dropped; the error gives their number
  and the first row); a lifecycle filter that keeps no event (the transitions
  found are listed); and `missing_timestamps="drop"` on a log where every
  timestamp is missing. Consumers that build an empty `EventLog` (two tests
  of the workbench do) must stop doing so. Delta on the golden dataset: none
  (its log has rows, no lifecycle column and no missing timestamps; every
  table is unchanged).
- Fixed: C8 — a `count` recipe and a `Presence` constraint on the same
  activity can be made to agree on events kept without a timestamp: under
  `LogSchema(untimed_events="uniform")` `count_scoped` without a scope counts
  them as `count` does. The default `legacy` keeps the 0.1.0 rule (the recipe
  ignores them), so classic numbers do not move. Delta on the golden dataset:
  none under the default. Under `uniform` every golden table is unchanged as
  well (its norm has no `count` recipe); an unscoped `count_scoped` there
  would change for 1 to 34 cases per activity (126 events without a
  timestamp).
- Fixed: C17 — a case attribute that varies within a case is no longer
  resolved silently: the default `case_attribute_values="first_non_null"`
  still takes the first non-null value and records the attributes and their
  numbers of cases (`ParsedLog.varying_case_attributes`, reported by
  `EventLog.validate()`); `"strict"` refuses the log with the attributes
  named. Delta on the golden dataset: none (no attribute varies; the quality
  report is unchanged).
- Fixed: C22 — the rule for events without a timestamp is stated once, per
  primitive, in the schema (`UntimedEvents`) and in the table of
  `docs/semantics/log-policies.md`, which a test checks against the kernel
  row by row. Delta on the golden dataset: none.
- Fixed: C3 — `score()` left the derived attributes in the input log's case
  table and attribute list, and silently overwrote a user case attribute of
  the same name as a recipe. A score is a function of the log and the norm
  (Sec. IV-D): the log is now left as it was, and a recipe named like a case
  attribute that the same recipe did not produce (a user attribute, a built-in
  column such as `n_events`, or an attribute derived by a different recipe)
  is refused with a `NormError` naming the attribute. Delta on the golden
  dataset: none (its recipe `manual_touches` clashes with no case attribute;
  every table is unchanged).
- Fixed: C26 — `prioritize()` refuses `baseline=NaN` (and any non-finite baseline) with a `NormError` instead of returning null gaps and priorities. Delta on the golden dataset: none (the input was refused; every valid number is unchanged).
- Fixed: C27 — `prioritize()` refuses a negative or non-finite `z`; a negative multiplier turned `PI_lower` into an upper bound. Delta on the golden dataset: none.
- Fixed: C28 — `prioritize()` refuses a fractional, negative or non-finite `min_cases` instead of truncating it (an integral float such as `5.0` is still accepted). Delta on the golden dataset: none.
- Fixed: C9 — every public entry point raises a `WiseError` subclass for a bad argument: `EventLog` refuses an unknown `missing_timestamps` option, a reversed observation window and an unknown `exposure_agg` with `LogSchemaError` (the first two were bare `ValueError`s, the last an `AttributeError` from pandas); `Norm.dumps()`/`fingerprint()` serialise numpy arrays and pandas `Timedelta` metadata and raise `NormError` for anything else (was `TypeError`); `Norm.loads()`/`load()`/`from_json()` raise `NormError` for malformed JSON (was `json.JSONDecodeError`); `hotspot_table()` and `compare_periods()` refuse frames that are not backlogs with `NormError` (was `KeyError`); `timestamp_outliers()` accepts a half-open window `(None, end)` or `(start, None)` (was `TypeError`). Delta on the golden dataset: none.
- Fixed: C19 — the ranking parameters of a backlog rode in `DataFrame.attrs`,
  which pandas drops or merges in many operations. `prioritize()` now returns
  a record whose frozen specification is a field of its own; it survives
  copying and pickling and any operation on the table (decision record 0001,
  Consequences). Delta on the golden dataset: none (every table is unchanged;
  the committed backlog and hotspot parquet records still carry the 0.1.0
  attrs in their metadata until the next regeneration on the record host,
  and the golden comparison does not read attrs).
- Fixed: C24 — `compare_periods` filled a slice absent from one period with
  zeros, so a slice that disappeared read as an improvement by its whole gap
  and PI, a new slice as a deterioration, and either as no change when its PI
  was zero. A gap is defined for a ranked slice only (Sec. IV-E): such a slice
  is now reported as `new` or `gone` in `presence` with missing changes, and
  two records ranked with different specifications are refused unless the
  difference is `intended`. Delta on the golden dataset: none for its tables
  (`compare_periods` is not part of the golden pipeline). Splitting its cases
  at the median start into two periods (Finance, γ = 20, baseline 0.9): by
  vendor, all 60 slices occur in both periods and nothing changes; by company
  and vendor, 164 of 399 slices occur in one period only. Before, 67 of them
  reported their whole PI as a change and 97 a change of zero; now all 164
  have missing changes, and the sum of `PI_change` over the slices of both
  periods is 1.0017 where the zero-filled sum was 1.2304.
- Fixed: C23 — a missing slice key became a NaN index entry of the backlog,
  the drivers and the penalty mass, which `.loc`, joins and period comparisons
  cannot address reliably. Every scored case belongs to one slice (Sec. IV-E);
  the slice of cases without a key is now labelled `wise.MISSING_KEY`. Delta on
  the golden dataset: none (no case lacks `company` or `vendor`).
- Fixed: C21 — `diagnostics` imported the private `_frame` and `_keys` of
  `prioritization`, and `derive` and the Lag family the private `_UNITS` of
  `constraints`. They now use public names (decision record 0001, one owner
  per computation), and a test refuses any import of an underscore name from
  another module of the package. Delta on the golden dataset: none.
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
