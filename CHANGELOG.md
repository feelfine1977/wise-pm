# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

The 0.2.0 line deepens the library's modules behind their public interface.
Classic numbers do not change: a schema version 2 norm evaluates under the
0.1.0 semantics and every golden table is unchanged; the corrections of the
defect ledger apply to schema version 3 documents, to input that 0.1.0
accepted although it was malformed, and to outputs that were wrong (each
`Fixed: C<n>` below states its delta). The package version is `0.2.0.dev0`.
`CITATION.cff` stays at the released version 0.1.0 and changes at release
time. The committed records that name the library version that produced them
(the golden manifest, the BPIC'19 oracle and the native evaluation summary)
keep `0.1.0`: they are provenance, the golden check tolerates a version
difference, and no golden value changes. The contract snapshot writes the
version as a placeholder; every change to it is listed below. Callers of
0.1.0 start with *Migrating from 0.1.0* at the end of this section.

### Changed

- Tooling for the 0.2.0 layout: every module of the library is type-checked
  by `mypy --strict` without a relaxation (the overrides for the 0.1.0
  modules are gone; the 0.1.0 test modules keep theirs until they are
  rewritten), the import-linter layer contract places every module of the
  library with `wise.compat` on top, and a test keeps both true for new
  modules. The sdist ships `examples/`, which the packaged tests read; the
  packaged tests skip the checks that need the repository's changelog,
  documents or scripts.
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

- The norm document format moved out of `wise.norm` into `wise.norm_document`
  (`read`, `write`, `write_constraint`, `loads`, `load`, `dumps`, `dump`,
  `fingerprint`, `schema_version_of`, and one `Adapter` per schema version;
  `wise.norm_document.v2` reads and writes the 0.1.0 layout for schema
  versions 1 and 2). `wise.norm` holds the model only. `Norm.load`, `loads`,
  `dump`, `dumps`, `from_dict`, `to_dict`, `from_json`, `to_json`,
  `fingerprint` and `NormConstraint.to_dict` keep their signatures and
  delegate to it; the shipped norms (`examples/bpic19_norm.json`,
  `examples/running_p2p_norm.json`, the golden norm and the fifteen norms of
  `evaluation/norms/native`) round-trip byte for byte. The import-linter
  layers gain `wise.norm_document` above `wise.norm` (the delegating methods
  are the one declared upward import) and `wise._frozen` at the bottom.
- A norm carries the schema version of its document format as the frozen,
  keyword-only field `Norm.schema_version` (decision record 0001, section 4):
  the constructor defaults to `wise.SCHEMA_VERSION` (2) and accepts
  `wise.norm.SCHEMA_VERSIONS` (1 and 2, as plain integers); reading a document requires the
  `schema_version` key (a document without it is refused with "norm document
  has no 'schema_version'", where 0.1.0 assumed the current version); `replace`,
  `replace_constraint` and `map_constraints` keep it and `replace` refuses to
  change it; writing emits the carried value, so a version 1 document is
  written back as version 1 (0.1.0 rewrote it as 2). A `schema_version` that
  is a boolean, a float, `null`, zero or negative is refused with `NormError`
  (0.1.0 converted it with `int()`); an integer or a string of digits is read
  as before.
- A norm's structure is checked once, when it is built: each layer, view and
  constraint checks its own fields, and the norm checks only what needs the
  whole (ids, references, a non-zero view, scoring mode, schema version, recipe
  names). `Norm.validate()` repeats that same check; it no longer re-parses the
  applicability rules or re-checks the weights, and `score()` no longer calls
  it, since a norm cannot change after it is built (see `Fixed: C7`). The
  messages are unchanged.
- `Norm.layer_of` is a property that hands out a copy of the Series on every
  call (it was a cached Series that a caller could write into); the contract
  snapshot records the new kind and the new field `schema_version`, and
  nothing else. Scoring time on the golden log is unchanged (median of seven
  runs 44.3 → 43.4 ms); loading and fingerprinting the golden norm 0.62 →
  0.64 ms.
- `EventLog.count`, `first_ts` and `last_ts` hand out a copy of the Series the
  log remembers, as `EventLog.events` does (shallow under copy-on-write);
  writing into one no longer changes the log's later answers.

- Reading a norm document of schema version 1 or 2 emits one
  `wise.LegacySemanticsWarning` (a `FutureWarning`) per norm read, naming the
  norm: the norm evaluates under the 0.1.0 semantics until 2.0 (decision
  0001, section 4). Its numbers are unchanged. A norm built in Python, written,
  fingerprinted or scored does not warn. The golden, paper (Section V) and
  native evaluation tiers and the unit modules that read schema version 2
  documents filter the warning by name.
  `scripts/compat_gate.sh` passes `-W ignore::wise.LegacySemanticsWarning` to
  every downstream suite, because those suites treat warnings as errors.
  Callers that do the same must filter it by name, or migrate their norms to
  schema version 3.
- `wise.norm.SCHEMA_VERSIONS` is `(1, 2, 3)`; `wise.SCHEMA_VERSION`, the
  version a norm built in Python carries, stays 2. A schema version above 3
  is "newer than supported (3)", and the message for a missing
  `schema_version` lists `[1, 2, 3]`.
- Every refusal of a document's `schema_version` lists the supported
  versions: a value that is not an integer, is not known or is newer than
  supported ends in "; supported: [1, 2, 3]", and a missing one still says
  "give one of [1, 2, 3]" (see `Fixed: C14`).
- `wise.norm_document.v3.is_lag` (was the private `_is_lag`) tells whether a
  constraint entry of a document is a lag.
- `ScoreResult.in_scope` (and `Evaluation.in_scope`) is true where a
  constraint is known to apply: a case whose scope is unknown under the
  three-valued policy is not in scope. Under schema version 2 nothing is
  unknown, so the frames are unchanged.
- `ScoreResult(mode=...)` defaults to `None`, which becomes the norm's
  `scoring_mode` (see `Fixed: C13`); the contract snapshot records the new
  default and the annotation `str | None`.

### Added

- `wise.compat`, the compatibility module of decision record 0001 (section
  2), removed in 1.0. `wise.compat.PATHS` lists every 0.1.0 name whose
  location, type or behaviour changed and every private name the workbench
  consumers touch, each with its replacement and its status: `deprecated`
  (the 0.1.0 form works and emits one `DeprecationWarning` per use naming the
  replacement, returning what the replacement returns), `kept` (a private
  that works as in 0.1.0 because nothing public replaces it yet:
  `ScoreResult._weights`) or `changed` (a correction with no compatibility
  path). It serves `wise.compat.prioritize` (the 0.1.0 DataFrame with its
  `attrs`), `wise.compat.norm_from_dict` (a document without
  `schema_version` read as version 2, as 0.1.0 did) and
  `wise.compat._rule_attributes`, and re-exports `EventLog`, `ScoreResult`
  and `Backlog`, whose members serve the remaining paths. A test scans the
  Python sources of the workbench checkout (or, without one, the committed
  record of that scan) and fails on a name or private they use that neither
  works unchanged nor is listed; another fails on any 0.1.0 public name whose
  shape changed without an entry. The 0.1.0 quick start of the README and
  both example scripts run unchanged apart from deprecation warnings and
  print what 0.1.0 printed (the BPIC'19 script on the synthetic log written in
  the challenge's layout). `wise.backlog.legacy_frame(backlog)` gives the
  0.1.0 frame of a record for these paths.
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

- `EventLog.fingerprint()`: a SHA-256 over what the log consumed (the schema
  digest, every kept event column in event order, the case table including
  attributes attached later, and the explicit window). Changing one consumed
  value changes it; the order of the input rows, where the schema orders the
  events, and the order of the columns do not; a column dropped by
  `keep_columns` is not covered. It is computed once per log and kept until
  the case table or the window changes.
- `wise.Calibration` (module `wise.calibration`, with `CalibrationStatus` and
  `parse_calibration`): the frozen calibration record of a `quantile_scale`
  recipe (decision 0001, section 3). It holds the quantile, the status
  (`fitted` or `unfitted`) and, when fitted, the divisor, the fingerprint of
  the log it was fitted on, its size (the cases whose value entered the
  quantile) and its window (ISO 8601 start and end). A recipe document
  carries it under the new key `calibration`; `QuantileScaleRecipe` has the
  field `calibration` and the method `calibrate(log, *, log_fingerprint=None)`.
- `wise.fit_calibration(norm, log)` returns the norm with a fitted record on
  every `quantile_scale` recipe; the recipe documents keep their other keys.
  Scoring with a fitted record divides by its divisor whatever cases are
  scored. No shipped norm carries a fitted record.
- `ScoreResult.calibrations` (keyword-only field): per `quantile_scale` recipe
  computed in the run, the record it divided by. For a recipe without a fitted
  record (schema version 2 legacy profile) that is the record of the divisor
  realised on the scored cases, which states the scored log's fingerprint; on
  the log a record was fitted on, the two are equal.

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

- Schema version 3 of the norm document (`wise.norm_document.v3`, adapter
  `wise.norm_document.V3`), read and written beside schema version 2, with
  the corrected semantics of decision 0001, section 3. A v3 document names its
  missing-value policy in the required top-level field `"missing_values":
  "three_valued"`. A `lag` constraint states `delta` (with `width`) or
  `"unbounded": true` without `delta`; a lag with neither, or with `"delta":
  null`, is refused. A `quantile_scale` recipe carries its `calibration`
  record: an unfitted record is admitted, and scoring refuses it with a
  `NormError` that names the recipe and points to `wise.fit_calibration`. An
  `eval` recipe is refused, also for a norm built in Python with
  `schema_version=3`. `examples/running_p2p_norm_v3.json` is the v3 copy of
  the running example norm; the golden tier scores it beside the v2 original
  (delta report: `docs/semantics/schema-v3-delta.md`).
- The semantics profile (`wise.semantics`): `wise.Semantics` (name,
  `missing_values`, `fitted_calibration`), `LEGACY_SEMANTICS` (0.1.0, schema
  versions 1 and 2), `CORRECTED_SEMANTICS` (schema version 3),
  `semantics_of(schema_version)`, `wise.LegacySemanticsWarning`, and the
  property `Norm.semantics`. Scoring takes the profile from the norm and hands
  it to every evaluation as an argument; no module reads it from global state.
  `wise.evaluate` and `wise.evaluate_constraint` take a keyword-only
  `semantics` (default: the 0.1.0 profile).
  `NormConstraint.applies_to` takes a keyword-only `missing` policy (default
  `"legacy"`), and the new `NormConstraint.scope(cases, log=None, *,
  missing="legacy")` gives the nullable scope with `<NA>` for unknown. The
  reason `wise.family.SCOPE_UNKNOWN` (`"scope_unknown"`) marks a case whose
  applicability is unknown.

- `wise.migrate(source)`, the only operation that changes a norm's schema
  version (decision 0001, section 4). It takes a norm or its document of
  schema version 1 or 2 and returns a `wise.Migration`: the schema version 3
  `document` (read-only), the `norm` it describes and a
  `wise.MigrationReport` (`norm`, `source_version`, `target_version`,
  `rewrites`, `corrections`, `to_dict()`, `to_json()`, `from_dict()`). Each
  `wise.norm_document.Rewrite` states its `rule`, the JSON `path`, the
  `subject` (recipe, constraint or norm), the keys `before` and `after`, a
  `note`, the ledger `corrections` that can change a scored row, and a
  `recommendation` where something needs review. The rules: the
  missing-value policy is added (C4, C25); an `eval` over a bare case-table
  column becomes a `column` recipe (C16), and any other `eval` stops the
  migration with the recipe named; a `quantile_scale` without a record gets an
  unfitted calibration record and the recommendation to fit it (C15); a `lag`
  without `delta` gets `"delta": 0, "width": 0` written explicitly and the
  recommendation to review it (C2); `"delta": null` becomes `"unbounded":
  true`. The rest of the document is copied unchanged, so every number keeps
  its value and JSON form; the report of a document is always the same text.
  Migrating does not emit `LegacySemanticsWarning`; a schema version 3 source
  is refused. `wise.migrate`, `wise.Migration`, `wise.MigrationReport`,
  `wise.norm_document.migrate` and `wise.norm_document.load_document(path)`
  (the JSON document of a norm file, not yet read) are new public names. The
  shipped norms migrate; the delta report
  (`docs/semantics/schema-v3-delta.md`) lists their rewrites and shows that no
  row of their tables differs.
- `wise migrate NORM [--out PATH]` writes the migrated norm (by default
  `<NORM>_v3.json` beside it) and prints the migration report as JSON. It
  refuses to write over its input, also through another spelling of the path
  or a link.

### Deprecated

- `wise.compat` as a whole: every path it serves is removed in 1.0.
- `EventLog._to_ts` emits a `DeprecationWarning` and delegates to
  `EventLog.to_timestamp`; it is removed in 1.0. The arrays `EventLog._codes`,
  `_ts`, `_valid` and `_starts` remain readable as read-only views of the
  case index and emit a `DeprecationWarning` naming `EventLog.case_index`'s
  `case_codes`, `timestamps`, `valid` and `starts`; removed in 1.0.
- `wise.norm._rule_attributes(rule)` emits a `DeprecationWarning` naming
  `wise.selection.parse(rule).columns`; removed in 1.0.
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

- Fixed: C2 — a `Lag` without a time bound silently defaulted to `delta=0,
  width=0`, which makes any delay a full violation. Closed for schema version
  3 documents, as the ledger defines it (decision 0001, section 3): a v3 lag
  states `delta` or `"unbounded": true` and is refused otherwise, and
  `wise.migrate` writes the 0.1.0 bound `"delta": 0, "width": 0` into the
  document with a recommendation to review it, so the default is never
  silent. Schema version 2 documents and a `Lag` built in Python keep the
  0.1.0 default under the legacy profile (section 4). Delta on the golden
  dataset: none; the golden norm's lags state their bounds, and its migrated
  norm scores every golden table unchanged.
- Fixed: C14 — `schema_version` was accepted without migration semantics.
  A document whose `schema_version` is missing or unknown is refused with the
  supported versions listed, and only `wise.migrate` changes the version of a
  norm, always with a report of every rewrite (decision 0001, section 4).
  Delta on the golden dataset: none.
- Fixed: C4 — an `ne` or `not_in` applicability rule put a case whose
  attribute is missing in scope (`NaN != v` is true). Under schema version 3
  applicability is three-valued (decision 0001, section 3): such a case is
  unknown under every operator, is not in scope, has no value (reason
  `scope_unknown`) and is left out of the constraint's denominator. Schema
  version 2 documents keep the 0.1.0 scope under the legacy profile (section
  4). Delta on the golden dataset: none. The golden norm stays at schema
  version 2 and every table is unchanged. The v3 copy of the running example
  differs in no row (`docs/semantics/schema-v3-delta.md`).
- Fixed: C25 — a `not` rule brought a case with a missing attribute into
  scope. Under schema version 3, `not` keeps an unknown scope unknown, and
  `all`/`any` combine by Kleene logic (decision 0001, section 3); schema
  version 2 is unchanged. Delta on the golden dataset: none, as for C4
  (`docs/semantics/schema-v3-delta.md`).
- Fixed: C13 — `ScoreResult.mode` defaulted to `"flat"` while a norm defaults
  to `"layer_balanced"`, so a result built without a mode computed its
  effective weights in a mode the norm did not state. A result built without
  a mode now takes its norm's `scoring_mode` (Sec. IV-D). `score()` always
  passed the mode, so every scored number is unchanged. Delta on the golden
  dataset: none.

- Fixed: C7 — frozen norm objects handed out mutable containers: writing into
  `View.constraint_weights` or `layer_weights`, a `NormConstraint.applicability`
  rule, `Norm.derived_attributes`, `Norm.metadata`, the cached `Norm.layer_of`
  or the where-clause of a parsed recipe changed the norm (or, before a
  re-validation, its scores). A norm is fixed once built (Sec. IV-C): these
  values are now read-only `dict` and `list` subclasses
  (`wise._frozen.FrozenDict`, `FrozenList`) that raise `TypeError` on any
  in-place change but compare, serialise to JSON, pickle and pass
  `isinstance(x, dict)` / `isinstance(x, list)` as before; numpy arrays in
  metadata become read-only copies; inputs are copied on construction;
  `to_dict()` returns plain, independent containers. Code that edited a norm
  in place must build a new one (`replace`, `replace_constraint`, or a new
  `View`). The event-log side of the entry (`EventLog.events`, `cases`,
  `case_attributes`, `trace`) was fixed when scoring stopped writing into its
  input log; `count`, `first_ts` and `last_ts` complete it (see Changed).
  Delta on the golden dataset: none.
- Fixed: C12 — `Norm.fingerprint()` hashed the whole document, so a note in
  `metadata` or an edited description changed the provenance of every result.
  The fingerprint is now defined beside the format (`wise.norm_document`): the
  SHA-256 of the canonical JSON of the written document without `metadata` and
  without the `description` of the norm, its layers, views and constraints.
  Everything that changes what the norm measures (names and versions, the
  schema version, layer names, weights, parameters, applicability, recipes,
  scoring mode) still changes it. As a consequence `fingerprint()` no longer
  fails on metadata that JSON cannot hold (`dumps()` still refuses it, see C9).
  Declared delta of the fingerprints of the shipped norms:
  `examples/bpic19_norm.json` e17ca18ed3c1a30114694c29455b86def27480bb34000e797e1e06a82dc644b0
  → a284ebdb4a732e4029771220726cce2a3a4b9ed96e0daeec66d90e8b6b759fde;
  `examples/running_p2p_norm.json` and `wise.running_p2p_norm()`
  b76e8f39df8510658c4ffe45d07c360b97a834b1655ed70fddc4a758873ba995
  → 68e8462a1d19d8076dcb50376b4390720489116c92c8e24f72c41263a0f9006e;
  the golden norm `tests/data/golden/norm.json`
  d97dd1b4d93a0d328b841087063c16fcaa82ca064deca25409208113c6bffb1b
  → 5ffdbef31bf546d8d89d6fbb8f5f572e74084799149eb9976b4fb9b3e0d52f47.
  Delta on the golden dataset: no table changes; the golden manifest's
  `norm.fingerprint` is updated to the new value (that field only). The
  BPIC'19 hash oracle keeps the fingerprint it was recorded with, as
  provenance, like its library version; its digests are unchanged. Backlog
  records and score results made before this change carry the old
  fingerprints and are refused by `layer_drivers` and `constraint_drivers`
  against a result scored after it.
- Fixed: C20 — evaluation and validation logic existed twice. The evaluation
  side went with the constraint families (one module per type, found through
  the registry instead of type switches) and the one pairing rule that Lag
  and the `lag` recipe share (see Changed). The validation side is closed
  here: the norm checked its views' weights, its constraints' weights and
  their applicability rules a second time on construction and again before
  every scoring run; the structure is now checked once, in one place, and
  `Norm.validate()` calls that same check (see Changed). Delta on the golden
  dataset: none.
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
- Fixed: C15 — `quantile_scale` divided by a quantile of whatever cases were
  scored, so the same case got a different value in a different population.
  A recipe now carries a frozen calibration record (`wise.Calibration`,
  decision 0001, section 3); with a fitted record the divisor is the record's
  and a case's value does not depend on the other cases scored. Under the
  schema version 2 legacy profile a recipe without a record, or with an
  unfitted one, divides by the quantile of the scored cases as in 0.1.0, and
  the realised divisor is recorded in `ScoreResult.calibrations` (decision
  0001, section 4). Refusing an unfitted record belongs to schema version 3.
  A numeric `divisor` key, accepted by 0.1.0 as an unknown key and carried
  unused since, is refused with a pointer to `calibration`; `"divisor": null`
  is still read as no record. `QuantileScaleRecipe.divisor` is replaced by
  `QuantileScaleRecipe.calibration` and the canonical document writes
  `calibration` instead of `divisor`. The fitted BPIC'19 divisor is not
  committed. Delta on the golden dataset: none (its norm has no
  `quantile_scale` recipe; every table is unchanged); the shipped BPIC'19 norm
  has no record and scores as before.
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

### Migrating from 0.1.0

Every item below is listed with its replacement in `wise.compat.PATHS`.
Importing from `wise.compat` keeps a 0.1.0 caller running through 0.2.x with
deprecation warnings; the module and its paths are removed in 1.0.

- `wise.prioritize` returns a `Backlog` record: read `backlog.table` (the
  ranked frame) and `backlog.spec` (its parameters) instead of the frame and
  its `attrs`. Using the record as a DataFrame still works with a warning;
  `isinstance(x, pd.DataFrame)`, `pd.concat` and `assert_frame_equal` need
  `.table`. `wise.compat.prioritize` returns the 0.1.0 frame.
- A norm document must state `schema_version` (C14); `wise.compat.norm_from_dict`
  reads a document without it as version 2. A schema version 2 norm
  evaluates under the 0.1.0 semantics and emits one `LegacySemanticsWarning`
  per read until 2.0: filter it by name
  (`-W "ignore::wise.LegacySemanticsWarning"`, as `scripts/compat_gate.sh`
  does) or migrate the file with `wise migrate NORM` and, for a
  `quantile_scale` recipe, fit it with `wise.fit_calibration` before scoring.
  Norm fingerprints differ from those of 0.1.0 (C12).
- `EventLog.events`, `cases` and `case_attributes` hand out copies (C3): a
  write into them (`log.events[col] = ...`, `log.cases[col] = ...`) does not
  reach the log, and `events` and `case_attributes` cannot be assigned. Add an
  event column to the frame the log is built from; add case attributes with
  `add_case_attribute` or `with_case_attributes`. Assigning `log.cases` still
  works with a warning.
- `log._to_ts(value)` is `log.to_timestamp(value)`; `log._codes`, `_ts`,
  `_valid`, `_starts` are `log.case_index.case_codes`, `timestamps`, `valid`,
  `starts`. `wise.norm._rule_attributes(rule)` is
  `set(wise.selection.parse(rule).columns)`.
- `NormError`, `NotScoredError` and `LogSchemaError` no longer inherit
  `ValueError` or `KeyError` (C11): catch the `wise` classes or `WiseError`.
- A `ScoreResult` built by hand without `mode` takes its norm's
  `scoring_mode` (C13); pass `mode=` to keep a mode. Passing `_weights=` and
  reading `result._weights` work as in 0.1.0; `dataclasses.replace(result,
  log=None)` gives a result without its log.
- `compare_periods` adds the column `presence`, leaves the change of a slice
  absent from one period missing instead of comparing it with zero (C24), and
  refuses two records ranked with different specifications unless the
  difference is `intended`.
- Input that 0.1.0 accepted although it was malformed is refused with a
  `WiseError`: an empty event frame and null lifecycle transitions (C6),
  non-integer `m`/`k` (C1), unknown recipe keys and options (C5), an `eval`
  recipe other than one case-table column (C16), and invalid ranking
  parameters (C26–C28); scoring refuses a schema version 3 norm whose
  `quantile_scale` recipe has no fitted calibration (C15).

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
