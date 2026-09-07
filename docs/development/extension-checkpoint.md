# Extension checkpoint

Kept accurate across sessions. Sanitised summary only: no run payloads, no
event data, no dataset paths.

## Current position

| Field | Value |
|---|---|
| Branch | `feat/actionability-ocpm-local-llm` |
| Base commit | `df5db50b839cc124b489a269894f5a2bfe7dc634` |
| Scope | cycle L1: `foundation` (S0–S2). Cycle L2: stage **S4**. Cycle L3: stage **S3** (L01–L03) and the drafting half of **S5** (N01, N02). S6 is not started |
| Stages completed | **S0 — baseline and regression contract (E05)**, **S1 — evidence, observations and complete runs (E01, E02, E04)**, **S2 — shared baselines and exact explanations (E03)**, **S2-fix — the review's eight must-fix items**, **S4 — native object-centric assessment (O01–O05)**, **S3 — local read-only assistance (L01, L02, L03)** |
| Stage in progress | **S5, drafting half only**: N01 and N02 complete. A01–A03 (investigation records, queues, transitions, exports) are **out of scope for this library** by `library-cycles/SCOPE_BOUNDARY.md` — the application owns them |
| Next stage | **S6** (L04 held-out local explanation benchmark, M01 evaluation harness) — not started. A01–A03 are not planned here |
| Local commits on the branch | none yet; the owner commits at the stage boundary |
| Remote state | no push, no pull request, no tag, no release |

---

# Stage S3 + S5 (drafting half) — the provider, the gateway, retrieval and safe drafts (L01, L02, L03, N01, N02)

Cycle L3. The library gains its whole language-model surface, and it is
deliberately thin: a provider protocol with a fake and a read-only Ollama
implementation, a strict local transport, an explicit tool gateway, an access
policy applied before retrieval and before any aggregate, a lexical retrieval
baseline over approved sources, and the norm-draft envelope with its deep
validation and a preview that applies nothing. The application owns the
assistant itself.

## Settings honoured

`ALLOW_MODEL_DOWNLOADS=false` and `ALLOW_LIVE_LLM_TESTS=false`. Everything is
built and tested against `wise.llm.provider.FakeProvider`. No server was
started, contacted or configured; no model was pulled; no socket is opened in
any executed test. The one file that could contact a server,
`tests/extensions/test_live_ollama.py`, is skipped unless `WISE_OLLAMA_LIVE=1`
and `WISE_OLLAMA_MODEL` are both set, and it was not run.

## Changed paths in this stage

```text
src/wise/schema.py                  new — the supported catalogue, read off CONSTRAINT_TYPES
                                    and the recipe key tables, with a drift check
src/wise/llm/__init__.py            new — lazy, offline package surface
src/wise/llm/provider.py            new — LLMProvider protocol, budgets, FakeProvider
src/wise/llm/transport.py           new — strict loopback transport, injectable opener
src/wise/llm/ollama.py              new — read-only provider for an already-running server
src/wise/llm/policy.py              new — Principal, Scope, AccessPolicy, authorised_baseline
src/wise/llm/gateway.py             new — six approved tools, literal dispatch table
src/wise/llm/retrieval.py           new — approved documents, lexical index, review questions
src/wise/llm/schemas.py             new — the three JSON schemas and a strict reader
src/wise/llm/assistant.py           new — two bounded steps, faithful rendering
src/wise/llm/drafts.py              new — NormDraft, deep validation, preview, trade-offs
src/wise/errors.py                  eight new error classes, all deriving from WiseError
src/wise/derive.py                  REQUIRED_RECIPE_KEYS, OPTIONAL_RECIPE_KEYS and
                                    UNSAFE_RECIPE_KINDS lifted to module constants;
                                    validate_recipe reads the first. No behaviour change.
docs/security/local-assistant.md    new — threat model and what it does not promise
examples/local_review.py            new — the assisted path against a fake provider
examples/review_norm_draft.py       new — validate, preview, apply nothing
tests/extensions/_llm_fixtures.py   new — runs, policies, documents, a fake opener
tests/extensions/test_ollama_transport.py          new (45 tests)
tests/extensions/test_llm_gateway.py               new (49 tests)
tests/extensions/test_retrieval_scope.py           new (24 tests)
tests/extensions/test_prompt_injection_boundaries.py new (14 tests)
tests/extensions/test_local_assistant.py           new (38 tests)
tests/extensions/test_norm_drafts.py               new (38 tests)
tests/extensions/test_safe_draft_recipes.py        new (37 tests)
tests/extensions/test_draft_behaviour.py           new (21 tests)
tests/extensions/test_live_ollama.py               new (1 test, opt-in, skipped)
tests/fixtures/norm_drafts/                        new — the roadmap example, a worked draft
pyproject.toml                      [llm] extra, live_llm marker
MANIFEST.in, README.md, CHANGELOG.md               documentation of the opt-in surface
```

`src/wise/derive.py` is the only pre-existing source file this stage edits, and
the edit is additive: three module constants and one function body reading the
first of them instead of an inline copy.

## The two properties this stage rests on

**The report exists before the model does.** `LocalAssistant.review` renders
the deterministic explanation first. Every provider failure — absent, timed
out, oversize, invalid, budget-exhausted — returns that report byte-identical,
with a typed reason. A draft can reorder facts and add prose; it cannot supply
a number, and every fact and evidence identifier it names is checked against
the authorised packet before it is used.

**The loader is not the boundary.** `wise.derive.compute_recipe` reaches
`DataFrame.eval` for `kind="eval"`. `validate_candidate` refuses that kind —
and unknown kinds, unknown evaluators, unreviewed keys, unbounded regular
expressions and out-of-range shapes — *before* `Norm.from_dict` is called and
long before anything is derived. `test_safe_draft_recipes.py` proves the
ordering twice: once by making `DataFrame.eval` fatal, once by watching
`Norm.from_dict` and asserting it was never reached.

---

# Stage S4 (part 2) — the shared kernel, three native check families, accounting and the projection comparison (O03, O04, O05)

The second half of the native object-centric stage. One numerical kernel serves
both evaluators; three relational check families read preserved relations; every
additive quantity has a canonical identity and can only be allocated once; and
the difference between the native answer and a flattened one is a table of
numbers against a manually specified truth.

## Changed paths in this stage

```text
src/wise/_aggregation.py            new — the single scoring formula, extracted from score()
src/wise/scoring.py                 _effective_weights and the per-view loop delegate to it;
                                    the signature the stage-0 contract pins is unchanged
src/wise/oc/constraints.py          new — RelatedObjectCardinality, CrossObjectLag,
                                    RelationalBalance, CheckOutcome, the native catalogue
src/wise/oc/accounting.py           new — QuantityRecord, Allocation, allocate, AllocationReport
src/wise/oc/evaluation.py           new — ObjectNorm (wise-oc-norm/1), evaluate_units,
                                    score_units, OCScoreResult, object_backlog
src/wise/oc/__init__.py             exports the new names
src/wise/evaluation/__init__.py     new
src/wise/evaluation/ocel.py         new — the two projections, their reports, and the comparison
src/wise/errors.py                  additive: OCConstraintError, AccountingError
src/wise/evidence/models.py         additive: 2 ReasonCode and 6 QualificationCode members
tests/extensions/test_aggregation_parity.py            new
tests/extensions/test_oc_constraints.py                new
tests/extensions/test_oc_accounting.py                 new
tests/extensions/test_oc_case_parity.py                new
tests/extensions/test_oc_representation_comparison.py  new
tests/extensions/_oc_fixtures.py                       new — the awkward fixtures, in code
examples/oc_invoice_review.py       new — runnable, no data file, no optional package
README.md, CHANGELOG.md, docs/semantics/object_centric.md   updated
```

`src/wise/scoring.py` is the only pre-existing source file this stage touches.
`log.py`, `norm.py`, `constraints.py`, `prioritization.py`, `derive.py`,
`diagnostics.py`, `cli.py`, `datasets.py` and `src/wise/__init__.py` are not
edited; `wise.__all__` does not grow and `import wise` still loads neither
`wise.oc` nor `wise.evaluation`.

## Commands and real results

| # | Command | Result |
|---|---|---|
| 1 | `python -m pytest -q` (no data environment variables) | **672 passed, 8 skipped, 0 failed** (531/8 before this stage) |
| 2 | `python -m pytest -q -ra` with `WISE_BPIC19_CSV` | **675 passed, 5 skipped** |
| 3 | `python -m pytest -q -ra` with `WISE_BPIC19_CSV`, `WISE_OCEL2_JSON`, `WISE_OCEL2_SQLITE` | **679 passed, 1 skipped** (the one skip is "no parquet engine") |
| 4 | `python -m pytest -q tests/extensions/test_baseline_contract.py` | **44 passed**; `git diff --stat` on that file is empty |
| 5 | `python -m pytest -q tests/test_bpic19.py` with the challenge CSV | **3 passed** — Section V, Table XI and Table X / Figure 4 unmoved |
| 6 | kernel parity on BPIC'19, both modes, against the reviewed base's arithmetic | **max abs difference 0.0** over 251 734 cases x 29 constraints |
| 7 | `ruff check .` and `ruff format --check .` | all checks passed; 67 files formatted |
| 8 | `mypy` | no issues in **31** source files |
| 9 | `python -m pytest -q --cov=wise` | 672 passed; coverage **95.86 %** (floor 85 %); `_aggregation.py` **100 %** |
| 10 | `python examples/oc_invoice_review.py` | exit 0; also asserted as a test |

## What the projection comparison actually measured

Three invoices, five obligations, fifteen cells of manually specified truth
(`tests/extensions/test_oc_representation_comparison.py`):

| | false violations | missed obligations | unsupported claims | exact | duplicated amount | witness recall |
|---|---|---|---|---|---|---|
| native | 0 | 0 | 0 | 15 | 0.00 | 1.00 |
| naive projection | 6 | 2 | 4 | 7 | 200.00 | 0.00 |
| identity-preserving projection | 6 | 2 | 4 | 7 | 200.00 | 1.00 |

The two projections make the same five mistakes. Preserving event identity does
not restore a relation; what it restores is the ability to name a witness by
its source event id and to measure the duplication. The fixture is small and
simulated: a representational difference, not evidence of industrial benefit.

## Open items carried out of O03/O04/O05

* The native catalogue has three families and no more. Presence, exclusion and
  metric obligations over objects are not expressible; the case-parity subset
  is deliberately restricted to the two constraints both models can state.
* `ObjectNorm` has no applicability rule language: a check applies to the unit
  types it names, or to all of them. The case norm's attribute rules have no
  object counterpart yet.
* `OCScoreResult` carries `EvaluationRecord`s but no `RunManifest` and no
  `EvidencePacket`; `wise.evidence.capture_evidence` still takes a case
  `ScoreResult` only.
* `wise.explain` has not been extended to object units: `explain_priority`
  still requires a case `ScoreResult`.
* The comparison is run on a hand-built fixture. The published OCEL 2.0
  procure-to-pay log is read and validated (O01) but no benchmark on it is part
  of this stage.
* `build_units` performance is unchanged: O(anchors x path), no shared frontier.

---

# Stage S4 (part) — the object log, its adapters and bounded units (O01, O02)

The first half of the native object-centric stage: a versioned object log that
preserves the identities a flattened log has to choose between, adapters for
the interchanges that are actually supported, and typed bounded assessment
units. No check, no score, no accounting — those are O03–O05.

## Changed paths in this stage

```text
src/wise/oc/__init__.py   the package's own exports; NOT imported by `import wise`
src/wise/oc/model.py      OCEventLog (contract wise-oc/1), OCEvent, OCObject, E2O, O2O,
                          AttributeChange, SourceMetadata, ValidationReport/Issue,
                          AttributeReading, AttributePolicy, observed_precision
src/wise/oc/io.py         read/write OCEL 2.0 JSON, read OCEL 2.0 SQLite, native pandas
                          tables, interchange_support(), LossReport
src/wise/oc/units.py      UnitSpec, RolePath, PathStep, UnitScope, TraversalLimits,
                          AssessmentUnit, PathWitness, Hop, UnitTruncation, build_units,
                          context_report
src/wise/errors.py        additive: OCError, OCValidationError, OCInterchangeError, OCUnitError
src/wise/evidence/models.py  additive: seven object-centric QualificationCode members
tests/extensions/test_ocel_io.py            60 tests, 4 of them gated on the external log
tests/extensions/test_assessment_units.py   42 tests
tests/fixtures/ocel/p2p_mini.json           the pinned OCEL 2.0 JSON interchange
tests/fixtures/ocel/p2p_mini_defects.json   the same shape with the defects named
examples/oc_units_review.py                 runnable, no data file, no optional package
docs/semantics/object_centric.md            the reference for what this stage means
README.md, CHANGELOG.md                     new opt-in section; unreleased entries
```

Nothing else was touched. `src/wise/__init__.py` is **unedited**: `wise.oc` is
an explicit import, so `import wise` loads no new module and `wise.__all__` is
byte-identical to what S2-fix left.

## Commands and real results

| # | Command | Exit | Result |
|---|---|---|---|
| 1 | `python -m pytest -q` (no data env vars) | 0 | **531 passed, 8 skipped** (425/4 before: +106 passed — 56 + 42 new tests and 8 new doctests — and +4 dataset-gated skips) |
| 2 | `python -m pytest -q` with `WISE_BPIC19_CSV`, `WISE_OCEL2_JSON`, `WISE_OCEL2_SQLITE` | 0 | **538 passed, 1 skipped** (the one skip is `no parquet engine`) |
| 3 | `python -m pytest -q tests/extensions/test_baseline_contract.py` | 0 | **44 passed**, file unedited |
| 4 | `python -m pytest -q tests/test_bpic19.py` with the challenge CSV | 0 | **3 passed** — Section V, Table XI, Table X/Figure 4 all unmoved |
| 5 | `ruff check .` / `ruff format --check .` (0.16.6) | 0 | all checks passed; 54 files formatted |
| 6 | `mypy` (2.3.1) | 0 | 25 source files clean |
| 7 | `python -m pytest --cov=wise` | 0 | **95.65%** total; oc/model 94%, oc/io 93%, oc/units 99% (floor 85) |
| 8 | `python -m build && twine check --strict dist/*` | 0 | sdist + wheel, 2 PASSED; `wise/oc/*` in the wheel |
| 9 | installed-wheel check in a clean venv outside the source tree | 0 | reads a log, builds units; `sqlite3` still not imported |
| 10 | `python examples/{quickstart,evidence_review,explanation_review,oc_units_review}.py` | 0 | all four run; `running_p2p_norm.json` rewritten byte-identically |

## Acceptance rows executed in this stage

| Row | Executed as | Result |
|---|---|---|
| O01 supported-format round-trip | `test_the_pinned_document_round_trips_key_for_key`, `test_writing_and_reading_back_preserves_the_content`, `test_the_native_tables_round_trip_every_field`, `test_the_sqlite_serialisation_reads_back_to_the_same_log` | ids, types, qualifiers and histories preserved; unsupported fields reported, not carried |
| O02 equal times / different ids | `test_two_events_with_one_activity_and_one_timestamp_both_survive`, `test_two_events_with_equal_timestamps_are_both_in_scope_and_ordered_stably` | both survive; ordering is `(timestamp, event_id)` and is not an identity |
| O06 changing attribute | `test_an_attribute_is_read_at_the_declared_time_not_at_the_first_value`, `test_a_changing_owner_is_read_at_the_time_and_not_as_the_current_owner` | the value at the declared time; `latest_known` is opt-in and qualifies itself |
| O07 cycles and a large shared node | `test_a_cycle_terminates_and_the_anchor_is_never_bound_to_its_own_role`, `test_a_shared_vendor_is_cut_by_the_fan_out_limit_and_the_unit_says_so`, `test_the_visit_budget_bounds_the_whole_unit` | bounded typed traversal; every cut flagged, qualified and refused by `require_complete()` |

O03, O04, O05, O08, O09, O10 and O11 are **not** executed: they need the check
templates, the accounting and the projection comparison, which are the second
half of this stage.

## What the published OCEL 2.0 p2p log actually contains

Recorded because it is an argument for the contract, not against the data: the
published `ocel2-p2p` log (14 671 events, 9 543 objects) has **2 028**
object-to-object relations pointing at an `invoice receipt` object it never
declares, and **295** identical repeated attribute-history rows. Reading it
with the default policy therefore **fails**, by design; `on_dangling="drop"`
reads it as 18 374 O2O relations and 78 213 history rows and records both
repairs. The same defects are present in the SQLite serialisation, so they are
the dataset's, not an adapter's.

## Open items carried out of O01/O02

* No check, no score, no accounting: `AssessmentUnit` is a statement about
  what was looked at. `src/wise/_aggregation.py` has **not** been extracted
  yet, and `ScoreResult` is untouched.
* The OCEL 2.0 **SQLite** adapter is read-only and loses the original
  attribute names (the serialisation stores sanitised column identifiers with
  no mapping back). Asserted in the cross-adapter test rather than assumed.
* OCEL 2.0 **XML** and the CSV extracts in the same data directory are not
  supported and are not claimed.
* Timestamps are written back in ISO-8601 with an explicit `+00:00` offset;
  a source spelling times with `Z` round-trips to the same content
  fingerprint but not to the same bytes.
* `evaluation_truncated` still has one producer. O02 introduces
  `context_truncated` for a cut *context*, which is a different fact from a
  cut *evaluation*; the per-check evaluation budget arrives with O03.
* The external-data tests are gated on `WISE_OCEL2_JSON` and
  `WISE_OCEL2_SQLITE` and skip without them. A skip is not a reproduction;
  they were executed in this session against the local read-only copies.

---

# Stage S2-fix — the review's eight must-fix items

Closing the punch list of the L1 review. No new capability: eight places where
the library knew something the reader was not told.

## Changed paths in this stage

```text
src/wise/log.py              LogSnapshot.content_hashed_tables; the "events" digest covers
                             every event column, not only the three a witness prints
src/wise/scoring.py          a capturing score() takes a scope="events" snapshot
src/wise/evidence/models.py  from_dict on the packet and on every part it holds; the
                             `restored` flag and the `witness_access` block
src/wise/evidence/manifest.py  from_dict on the run record and its parts; only_fields
src/wise/evidence/capture.py   load_evidence(); per-table content_hashed; records_truncated
                             and evaluation_truncated qualifications; a bounded packet's
                             comparator is named captured-subset with its denominator
src/wise/explain/priority.py   record-scope qualifications lifted to group scope;
                             ConstraintComponent.share_lower_bound; a witness a bounded
                             capture did not keep is disclosed
src/wise/explain/render.py     the "share lower bound" column, in both renderings
tests/extensions/…            16 new tests across four files plus one doctest; the
                             stage-0 contract is unedited and still passes 44
```

## Commands and real results

| # | Command | Exit | Result |
|---|---|---|---|
| 1 | `python -m pytest -q` | 0 | **425 passed, 4 skipped** (408 before, plus 17) |
| 2 | `python -m pytest -q tests/extensions/test_baseline_contract.py` | 0 | **44 passed**, file unedited |
| 3 | `python -m pytest -q -ra` with `WISE_BPIC19_CSV` set | 0 | **428 passed, 1 skipped** — the reproduction that did not run in S2 |
| 4 | `ruff check .` / `ruff format --check .` | 0 | all checks passed; 47 files formatted |
| 5 | `mypy` | 0 | 21 source files clean |
| 6 | each fix reverted in turn, its test re-run | — | **15 of 15 fail without their fix** |

## Measured cost

`score()` without capture is untouched: it still takes a `"structure"`
snapshot (0.5 ms on the 1.6M-event BPIC'19 log). A capturing run now takes an
`"events"` snapshot, which hashes the whole event table: **1.08 s** on that log
against 0.11 s for the `"cases"` snapshot it used before, paid once per capture
and once per batch of lazy witness lookups. `freeze=True` avoids the repeat.

## Open items carried out of S2-fix

* The eleven next-cycle items of the review are untouched by design.
* `EvidencePacket.from_dict` does not restore the snapshot or the norm, and
  says so; re-attaching a live log to an exported packet is deliberately not
  offered.
* `evaluation_truncated` is set by exactly one path — a `max_records` budget
  that stops the traversal before every constraint is reached. A per-check
  evaluation budget (O02) would be the second.

---

# Stage S2 — shared baselines and exact explanations (E03)

## Changed paths in this stage

```text
src/wise/explain/__init__.py                    new — lightweight public exports
src/wise/explain/baseline.py                    new — BaselineSpec, compatibility, resolution
src/wise/explain/priority.py                    new — explain_priority, ExplanationPacket, facts
src/wise/explain/render.py                      new — deterministic text / Markdown / JSON, safe names
src/wise/prioritization.py                      keyword-only baseline_spec= on prioritize and layer_drivers
src/wise/cli.py                                 wise explain <packet>; --baseline-file / --explain-out on score
src/wise/evidence/models.py                     twelve explanation qualification codes; "group"/"explanation" scopes
src/wise/evidence/capture.py                    to_interchange(baseline=<BaselineSpec>) accepts a comparator object
src/wise/__init__.py                            new exports (appended; the base order is unchanged)
tests/extensions/test_baseline_spec.py          new — 34 tests
tests/extensions/test_priority_explanation.py   new — 47 tests
tests/extensions/test_deterministic_render.py   new — 35 tests
tests/conftest.py                               four appended fixtures for the awkward populations
examples/explanation_review.py                  new — runnable tour, no data needed
docs/semantics/explanation.md                   new — the contract, in prose
README.md, CHANGELOG.md                         new section / unreleased entry
```

`tests/extensions/test_baseline_contract.py` is **unchanged** and still passes
in full (44 tests). No expected value, benchmark tolerance, default, norm JSON
or serialisation was changed; `examples/running_p2p_norm.json` is byte-identical
after `examples/quickstart.py`. `src/wise/scoring.py`, `log.py`, `norm.py`,
`derive.py`, `constraints.py` and `diagnostics.py` were **not** touched in this
stage.

## Commands and real results

Environment: branch worktree, `.venv`, Python 3.13.9, editable `.[dev,stats]`,
NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1, pytest 9.1.1. Lint and type results are
reported for the versions CI pins (`ruff==0.12.0`, `mypy==1.17.1`, in a
throwaway environment) and for the newer versions in the worktree environment.

| # | Command | Result |
|---|---|---|
| 1 | `python -m pytest -q -ra` | **408 passed, 4 skipped, 0 failed** |
| 2 | `python -m pytest -q --cov=wise --cov-report=term-missing` | **408 passed, 4 skipped**; total coverage **95.52 %**, floor 85 % |
| 3 | `python -m pytest -q tests/extensions` | **310 passed, 0 failed, 0 skipped** |
| 4 | `python -m pytest -q tests/extensions/test_baseline_contract.py` | **44 passed** — the stage-0 contract, file unchanged |
| 5 | `python examples/quickstart.py` | exit 0; the norm file is rewritten byte-identically |
| 6 | `python examples/evidence_review.py` | exit 0 |
| 7 | `python examples/explanation_review.py` | exit 0 (also run as a test) |
| 8 | `ruff check .` (0.12.0 / 0.16.6) | All checks passed |
| 9 | `ruff format --check .` (0.12.0 / 0.16.6) | 44 / 47 files already formatted |
| 10 | `mypy` (1.17.1 / 2.3.1) | Success: no issues found in **21** source files |
| 11 | `python -m build` + `twine check --strict dist/*` | exit 0; **2 PASSED** |
| 12 | installed-wheel check in a clean venv outside the source tree | score, explain, render, write and load all work; the shipped schema is readable |
| 13 | CLI: default `score` versus the same call with all four sidecars | stdout **byte-identical**; sidecar messages only on stderr |
| 14 | base-commit equivalence of `prioritize` / `layer_drivers` | **180 configurations byte-identical** (68 on a bare score frame, 112 on `ScoreResult`s), all of them committed as tests; everything from `constraint_drivers` onward is byte-identical text |

Skips are unchanged in kind: three `tests/test_bpic19.py` tests without
`WISE_BPIC19_CSV` (dataset gate) and `tests/test_log.py::test_from_csv_parquet`
("no parquet engine", an environment gap).

**The BPIC gate was not executed in this stage.** The local read-only copy used
at S0 and S1 was not locatable in this session; the only large local CSV has a
different column schema and errors on load, which is a dataset problem, not a
code result. What stands instead: `src/wise/scoring.py` was not touched, and the
default `prioritize` / `layer_drivers` path is shown byte-identical to the
reviewed base commit over 132 configurations (row 14), which covers exactly the
two functions `tests/test_bpic19.py` exercises beyond scoring.

## Acceptance rows executed in this stage

Explanation section of the handoff's `ACCEPTANCE_TESTS.md`:

| Row | Status | Where |
|---|---|---|
| X01 current comparator | **passes** | `test_a_named_current_population_comparator_reproduces_the_legacy_output` (both modes x both views), `test_the_default_path_is_byte_identical_to_the_base_commit` |
| X02 historical comparator with a complete profile | **passes** | `test_a_historical_profile_makes_the_deltas_sum_to_the_same_gap`, `test_a_target_with_a_complete_profile_reconciles_too` |
| X03 scalar-only target | **passes** | `test_a_scalar_target_ranks_but_allocates_nothing`, `test_a_scalar_target_never_borrows_the_current_global_deltas` |
| X04 mismatched mode/view/norm/calibration/layer semantics | **passes** | `test_every_semantic_mismatch_is_named_and_refused` (eight parametrised mismatches), `test_a_changed_norm_needs_an_explicit_reviewed_mapping` |
| X05 positive gap with a negative layer offset | **passes** | `test_a_positive_gap_keeps_its_negative_layer_offsets` |
| X06 zero or negative net gap | **passes** | `test_a_non_positive_gap_clips_the_index_and_keeps_the_contrasts`, `test_an_exactly_zero_gap_is_also_clipped_without_a_division` |
| X07 exposure priority | **passes** | `test_exposure_scales_the_index_and_never_weights_the_mean`, `test_a_zero_exposure_slice_keeps_its_gap_and_loses_only_its_index` |
| X08 minimum support and null group keys | **passes** | `test_the_reference_is_computed_before_the_minimum_support_filter`, `test_a_slice_below_the_minimum_support_is_explained_but_not_ranked`, `test_null_group_keys_form_their_own_slice_and_are_matched_as_nulls` |
| X09 different scored populations between views | **passes** | `test_views_that_score_different_cases_are_disclosed`, `test_a_common_population_reports_no_difference` |

Also executed here: **A07** (a hostile business label cannot escape the chosen
output directory — `test_a_generated_filename_uses_safe_identifiers_only`,
`test_a_hostile_suffix_cannot_become_a_path`) and **L09** (a mandatory
qualification is retained by the renderer independently of any model —
`test_every_limitation_reaches_every_rendering`), both stated deterministically
and without any language model.

## First substantial checkpoint — reached

`python examples/explanation_review.py /tmp/wise-review` scores an ordinary
event log, writes the run record and the evidence packet, and produces a
coherent explanation of the exact priority comparison, with no language model
and nothing beyond the base dependencies. `wise explain <packet>` renders it
again. The end-to-end run is asserted by
`test_the_end_to_end_example_runs_and_writes_what_it_says`.

## Open items carried out of S2

* `layers=` restricts what the decomposition *shows*; a partial selection no
  longer reconciles to the gap, and the packet says so rather than pretending
  otherwise.
* Alternative views are always contrasted against **their own** current
  population, because a comparator declared for one view is not a comparator
  for another. A per-view comparator specification is a possible extension.
* `ExplanationPacket.from_dict` does not rebuild the live `RunManifest`; the
  serialised run record stays available under `packet.run`.
* The BPIC reproduction was not re-run in this session (see above).

---

# Stage S1 — evidence, observations and complete runs (E01, E02, E04)

## Changed paths in this stage

```text
src/wise/evidence/__init__.py                   new — lightweight public exports
src/wise/evidence/models.py                     new — records, witnesses, qualifications, packets
src/wise/evidence/manifest.py                   new — run records and canonical fingerprints
src/wise/evidence/capture.py                    new — result -> evidence, bounded witnesses, interchange
src/wise/evidence/calibration.py                new — explicit fit/apply of a fitted reference
src/wise/evidence/schemas/                      new — shipped interchange schema and its example
src/wise/scoring.py                             detailed evaluation path; evidence=/calibrations=; manifest/evidence fields
src/wise/log.py                                 options(), snapshot(), LogSnapshot; derive(calibrations=)
src/wise/derive.py                              _quantile_divisor shared; compute_recipe/apply_recipes take a calibration
src/wise/diagnostics.py                         typed_event_replication / typed_cross_case_replication / typed_right_censored
src/wise/errors.py                              EvidenceError, EvidenceUnavailableError, StaleEvidenceError
src/wise/cli.py                                 opt-in --manifest-out / --evidence-out / --evidence-frame-out / --evidence
src/wise/__init__.py                            new exports (appended; the base order is unchanged)
pyproject.toml, MANIFEST.in                     package the evidence schemas
tests/extensions/test_evidence_models.py        new — 29 tests
tests/extensions/test_run_manifest.py           new — 24 tests
tests/extensions/test_evidence_capture.py       new — 74 tests
tests/extensions/test_frozen_calibration.py     new — 13 tests
tests/extensions/test_qualification_contract.py new — 10 tests
examples/evidence_review.py                     new — runnable tour on the bundled example
docs/semantics/evidence.md                      new — the contract, in prose
README.md, CHANGELOG.md                         new section / unreleased entry
```

`tests/extensions/test_baseline_contract.py` is **unchanged** and still passes
in full (44 tests). No expected value, benchmark tolerance, default, norm JSON
or serialisation was changed; `examples/running_p2p_norm.json` is byte-identical
after `examples/quickstart.py`.

## Commands and real results

Environment: branch worktree, `.venv`, Python 3.13.9, editable `.[dev,stats]`,
NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1, pytest 9.1.1. Lint and type results are
reported for the versions CI pins (`ruff==0.12.0`, `mypy==1.17.1`, installed in
a throwaway environment) and for the newer versions in the worktree environment.

| # | Command | Result |
|---|---|---|
| 1 | `python -m pytest -q -ra` | **281 passed, 4 skipped, 0 failed** |
| 2 | `python -m pytest -q -ra` with `WISE_BPIC19_CSV` set | **284 passed, 1 skipped, 0 failed** |
| 3 | `python -m pytest -q --cov=wise --cov-report=term-missing` | **281 passed, 4 skipped**; total coverage **94.77 %**, floor 85 % reached |
| 4 | `python -m pytest -q tests/extensions` | **194 passed, 0 failed, 0 skipped** (44 of them the unchanged stage-0 contract, 150 new) |
| 5 | `python examples/quickstart.py` | exit 0; the norm file is rewritten byte-identically |
| 6 | `python examples/evidence_review.py` | exit 0 |
| 7 | `ruff check .` (0.12.0 / 0.16.6) | All checks passed |
| 8 | `ruff format --check .` (0.12.0 / 0.16.6) | 36 / 39 files already formatted |
| 9 | `mypy` (1.17.1 / 2.3.1) | Success: no issues found in **17** source files |
| 10 | `python -m build` + `twine check --strict dist/*` | exit 0; **2 PASSED** |
| 11 | installed-wheel check in a clean venv | imports; `wise.evidence.interchange_schema()` reads the shipped schema; capture and interchange work outside the source tree |
| 12 | CLI: default `score` versus the same call with all three sidecars | stdout **byte-identical**; sidecar messages only on stderr |

Skips, unchanged in kind from stage 0: three `tests/test_bpic19.py` tests
without `WISE_BPIC19_CSV` (dataset gate) and `tests/test_log.py::test_from_csv_parquet`
("no parquet engine" — an environment gap, since no parquet engine is part of
`[dev,stats]`).

**The BPIC gate was executed.** With `WISE_BPIC19_CSV` pointing at a local
read-only copy, the three tests pass in ~21 s against the modified scoring code:
the Section V scores and applicability density, the Table XI focus slices and
the Table X / Figure 4 figures all hold within their existing tolerances. No
value derived from that log is recorded here.

## Measured cost of the additions

On the challenge log (251,734 cases, 1,595,923 events, 29 constraints):

| Operation | Time |
|---|---|
| `score(log, norm)` — the default path | 1.73 s |
| the run manifest alone (structural input identity, no data hashing) | 2.0 ms = **0.12 %** of a score |
| `evaluate_detailed(log, norm)` — matrices plus primitives | 0.95 s |
| `capture_evidence(..., mode="full", units=<50 cases>)` | 2.35 s → 1,450 of 7,300,286 possible records |
| `packet.to_json()` for those 50 units | 0.21 s, 5.2 MiB |

On synthetic logs, summary capture costs about 40 µs and full capture about
120 µs per record. Capturing every check of a large log is therefore expensive
by construction — one typed object per check — so `units=`, `max_records=` and
`witness_limit=` bound it and `EvidencePacket.truncation` reports what was left
out. This is documented in `docs/semantics/evidence.md`.

## Acceptance rows executed in this stage

Foundation section of the handoff's `ACCEPTANCE_TESTS.md`:

| Row | Status | Where |
|---|---|---|
| F03 capture disabled / summary / witnesses | **passes** | `test_capture_does_not_move_a_single_number` (both capture modes × both scoring modes) |
| F04 all applicable weights zero / everything unavailable | **passes** | `test_an_unscored_unit_stays_unscored_in_the_evidence`, `test_a_check_that_was_not_evaluated_has_no_weight_and_no_penalty` |
| F05 `return_scope=True` | **passes** (re-asserted) | `test_return_scope_still_returns_exactly_two_results` |
| F07 runtime mode override | **passes** | `test_the_manifest_records_the_mode_actually_used_not_the_norm_default` |
| F08 skip / violate / censor | **passes** | `test_each_missing_response_policy_keeps_its_own_meaning`, `test_a_censored_lag_records_the_horizon_and_calls_itself_a_lower_bound` |
| F09 absence rule | **passes** | `test_an_absence_is_a_declared_search_never_an_invented_event` |
| F10 source id missing | **passes** | `test_without_an_event_id_column_references_are_explicitly_snapshot_local` and its counterpart with ids |
| F11 mutation after scoring | **passes** | `test_witnesses_are_refused_after_the_log_changed`, `test_a_frozen_snapshot_survives_a_later_change` |
| F12 frozen versus recomputed quantile | **passes** | `tests/extensions/test_frozen_calibration.py` |
| F13 JSON missing / non-finite data | **passes** | `test_packet_json_round_trips_with_explicit_nulls`, `test_json_refuses_a_non_finite_number_rather_than_writing_nan` |
| F14 truncated witness display | **passes** | `test_a_truncated_witness_display_keeps_the_exact_measurement_and_the_total` |
| F15 CLI default CSV | **passes** | `test_the_new_cli_sidecars_are_opt_in_and_leave_stdout_untouched`, plus the unchanged stage-0 contract test |
| F01, F02, F06 | **still pass**, unchanged | `tests/extensions/test_baseline_contract.py` |

`D02` (schemas available from an installed wheel) and `D03` (module collection
without live services) were executed as well: the wheel check above, and
`--doctest-modules` collecting the new modules on every run.

## Open items carried into S2

* E03 is the whole of S2: one `BaselineSpec` shared by `prioritize` and
  `layer_drivers`, signed layer deltas summing to the gap under a compatible
  profile, and an explicit refusal to invent a layer profile for a scalar
  target. `RunManifest.finalize(...)` already carries the comparator identity
  and value, so the explanation stage has somewhere to record it.
* `to_interchange` resolves only the current-population comparator. The
  historical and target comparators belong to E03's `BaselineSpec`.
* The reason codes `ambiguous_match`, `missing_source_identity` and
  `budget_truncated` are defined and validated but are not produced by the
  case-based evaluator, which never matches ambiguously and never truncates an
  evaluation. They exist for the object-centric evaluator of S4.
* Capturing every check of a very large log is expensive. If exhaustive capture
  ever becomes a workflow rather than a review-time drill-down, an array-backed
  evidence frame beside the typed records is the obvious next step.

## Next exact step (as recorded at the end of S1; done)

Stage S2 (`prompts/03_EXPLANATION.md`), item E03: `src/wise/explain/` with
`baseline.py`, `priority.py` and `render.py`; append the keyword-only
`baseline_spec=` to `prioritize()` and `layer_drivers()` while keeping the
scalar `baseline` argument and its positional compatibility; exact signed
attribution from the same effective penalties the selected scoring mode
produced. Gate: layer deltas sum to the signed gap under a compatible profile,
a scalar-only target yields absolute contributions and an explicit
unavailability flag, and negative offsets and clipping stay visible.
`tests/extensions/test_baseline_contract.py` must keep passing unchanged.

---

# Stage S0 — baseline and regression contract (E05)

## Changed paths in that stage

```text
tests/extensions/test_baseline_contract.py   new — executable regression contract
docs/development/extension-plan.md           new
docs/development/extension-checkpoint.md     new (this file)
docs/development/extension-status.json       new
.gitignore                                   extended (environments, output, run payloads, private data, model caches)
```

No file under `src/wise/` was modified in S0.

## Commands and real results

| # | Command | Result |
|---|---|---|
| 1 | `python -m pytest -q -ra` (unchanged base) | **84 passed, 4 skipped, 0 failed** |
| 2 | `python -m pytest -q -ra` (with the contract tests) | **128 passed, 4 skipped, 0 failed** |
| 3 | `python -m pytest -q -ra` with `WISE_BPIC19_CSV` set | **131 passed, 1 skipped, 0 failed** |
| 4 | `python -m pytest -q -ra tests/test_bpic19.py` with the dataset | **3 passed, 0 failed, 0 skipped** |
| 5 | `python -m pytest --cov=wise --cov-report=term-missing` | **128 passed, 4 skipped**; coverage **93.71 %** |
| 6 | `python examples/quickstart.py` | exit 0, empty stderr; norm file byte-identical |
| 7–9 | `ruff check` / `ruff format --check` / `mypy` (CI pins) | passed; 25 files formatted; 12 source files clean |
| 10–11 | `python -m build`, `twine check --strict dist/*` | exit 0; 2 PASSED |
| 12–14 | the same lint and type checks on the newer worktree versions, plus wheel inspection | passed |

**Nothing in the base was already failing.**

## Regression contract

`tests/extensions/test_baseline_contract.py` — 44 tests, all passing, unchanged
by S1. Contents and tolerance policy: [extension-plan.md](extension-plan.md),
section 3. Signatures and dataclass fields are pinned as *ordered prefixes*, so
S1's appended keyword-only parameters (`evidence=`, `calibrations=`) and its
appended defaulted fields (`manifest`, `evidence`) pass, while a rename,
reorder, removal or changed default fails.

## Captured identities

```text
Norm fingerprint, datasets.running_p2p_norm() and examples/running_p2p_norm.json
  b76e8f39df8510658c4ffe45d07c360b97a834b1655ed70fddc4a758873ba995
Norm fingerprint, examples/bpic19_norm.json
  e17ca18ed3c1a30114694c29455b86def27480bb34000e797e1e06a82dc644b0
SHA-256 of examples/running_p2p_norm.json as shipped
  7acda57c1c7e35fd4de92c5a3fd38e8bf0f2bbcc4e2e2761355e178e4c71b4bb
Norm schema version 2; SCORING_MODES ("layer_balanced", "flat");
DEFAULT_SCORING_MODE "layer_balanced"; ScoreResult.mode dataclass default "flat".
Running example: 5 cases, 21 events, 6 constraints, 5 layers, views
Finance/Logistics, scoring mode "flat", c4 out of scope for every case.
Backlog columns: n_cases, mean_score, volume, gap, PI, stable_mean, stable_gap,
stable_PI, [se, gap_lower, PI_lower,] global_mean.
```
