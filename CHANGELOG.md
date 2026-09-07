# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added — evidence, run records and fitted references (experimental, opt-in)

- `wise.evidence`: typed evaluation records, witnesses, qualifications and
  packets (`models`), immutable run records and canonical fingerprints
  (`manifest`), result-to-evidence export with bounded witness access
  (`capture`), and explicit fit/apply of a fitted reference (`calibration`).
  Standard library and the existing base dependencies only.
- `score(..., evidence="none" | "summary" | "full")` and
  `score(..., calibrations=[...])`, both keyword-only. The default is unchanged
  in output and in behaviour.
- `ScoreResult.manifest`, `ScoreResult.evidence` and
  `ScoreResult.evidence_frame(view=...)`, all optional and defaulted; every
  existing field and method is untouched.
- `wise.evaluate_detailed(log, norm)` returns the violation matrix, the scope
  matrix and the primitives behind them. `violation_matrix(..., return_scope=True)`
  still returns exactly two elements.
- `EventLog.options()` and `EventLog.snapshot(freeze=..., scope=...)`: the
  recorded preparation options and a witness interface that refuses stale
  inputs (`StaleEvidenceError`). Supplied `event_id_col` values are used as
  witness identities; without one, references are labelled snapshot-local.
- Typed diagnostics with explicit denominators: `typed_event_replication`,
  `typed_cross_case_replication`, `typed_right_censored`, and
  `coverage_report`. The existing numeric diagnostics are unchanged.
- `EventLog.derive(..., calibrations=...)` and
  `wise.derive.apply_recipes(..., calibrations=...)` apply a frozen fitted
  reference instead of refitting; `quantile_scale` still recomputes by default.
- Opt-in CLI sidecars `wise score --manifest-out --evidence-out
  --evidence-frame-out --evidence`; the default stdout CSV is unchanged.
- `examples/evidence_review.py`, `docs/semantics/evidence.md`, and the shipped
  interchange schema `wise/evidence/schemas/evidence_packet.schema.json`
  (`wise.evidence.interchange_schema()`).

### Added — one shared comparator and exact explanations (experimental, opt-in)

- `wise.explain`: the comparator specification (`baseline`), the exact signed
  attribution and the versioned explanation packet (`priority`), and
  deterministic text/Markdown/JSON rendering (`render`). Standard library and
  the existing base dependencies only.
- `BaselineSpec`: comparator identity and kind (current population, historical,
  target), reference score, optional complete layer profile, view and
  normalisation identity, assessment-unit type, norm and calibration
  references, aggregation conventions, provenance and an explicit
  `reviewed_mapping`. Validated on construction, including
  `reference_score == 1 - sum(reference_layer_penalties)` within a documented
  tolerance. Compatibility compares assessment semantics, never population
  identity; an unknown layer is never aligned with zero.
- `prioritize(..., baseline_spec=...)` and `layer_drivers(..., baseline_spec=...)`,
  both keyword-only and appended. Existing calls keep their signature, columns,
  ordering, values and attributes; passing both `baseline` and `baseline_spec`
  is an explicit error. The legacy `global_mean` column keeps its name and
  position and holds whatever reference was used — which need not be global.
- Under a scalar-only comparator, `layer_drivers` keeps the absolute layer
  means, nulls the target deltas and reports
  `reference_profile_available = False` with
  `unavailable_reason = "reference_profile_unavailable"`. The current
  population's profile is never substituted.
- `explain_priority(result, by, group, ...)` returns an `ExplanationPacket`
  with authoritative facts and fact ids, denominators, the resolved comparator,
  profile and priority components, constraint drivers inside the slice,
  per-view contrasts, bounded witnesses and limitations. Layer deltas are
  asserted to sum to the signed gap under a compatible full profile; negative
  offsets are retained; a non-positive gap clips the index while the unclipped
  contrasts stay visible; no share is formed by dividing by a zero net gap.
- `render_text`, `render_markdown`, `render_json`, `render_explanation`,
  `write_explanation` and `explanation_filename`: deterministic renderings with
  escaped labels and safe generated file names.
- `wise explain <packet>` renders a packet; `wise score --baseline-file` ranks
  against a reviewed comparator and `wise score --explain-out` writes the
  packet for the top-ranked slice. The default stdout CSV and its columns are
  unchanged.
- `wise.evidence.to_interchange(..., baseline=<BaselineSpec>)` now accepts a
  comparator object, so a historical or target comparator can be exported;
  previously only the current population could be resolved.
- `examples/explanation_review.py` and `docs/semantics/explanation.md`.

### Added — native object-centric data and bounded assessment units (experimental, opt-in)

- `wise.oc`, an explicit import: `import wise` does not load it, and importing
  it pulls in no optional package and no database driver. Standard library and
  the existing base dependencies only.
- `wise.oc.model`: the versioned `OCEventLog` contract (`wise-oc/1`) —
  canonical event and object identities, qualified event-to-object and
  object-to-object relations, object attribute histories, source/extraction
  metadata and observed timestamp precision, held as immutable indexed tables
  with bounded lookups. `OCEventLog.build` validates and repairs; the
  dataclass constructor re-checks every structural invariant.
- Validation with named outcomes (`ValidationReport`, `ValidationIssue`):
  identical repeated rows are deduplicated and *reported*; conflicting
  identities, two values for one attribute instant and mixed time zones raise;
  dangling relations raise unless `on_dangling="drop"` is asked for, and then
  the drop is recorded. Two distinct events sharing an activity and a
  timestamp always survive.
- `OCEventLog.value_at(object_id, attribute, at, policy=...)` reads a changing
  attribute at a declared instant and returns an `AttributeReading` with
  `found`, `effective_from` and `later_changes`. There is no first-non-null
  fallback; `AttributePolicy.LATEST_KNOWN` is opt-in and qualifies itself.
- `wise.oc.io`: adapters for the interchanges that are actually supported,
  declared in `interchange_support()` — OCEL 2.0 JSON (read and write, the
  pinned interchange), OCEL 2.0 SQLite (read only, with the sanitised
  attribute names it cannot recover reported as a loss), and six native pandas
  tables (`to_tables` / `from_tables`). Exporting refuses rather than loses
  unless `allow_loss=True`, and returns a `LossReport` either way.
- `wise.oc.units`: `UnitSpec`, `RolePath`, `PathStep`, `UnitScope`,
  `TraversalLimits` and `build_units` — typed, directed, qualified paths from
  an object anchor, with strict depth, fan-out, binding and visit limits. An
  `AssessmentUnit` carries its bindings, the subgraph witnesses that justify
  them, the events in scope, the declared evaluation time, and `complete`;
  a truncated context is flagged, qualified and refused by
  `require_complete()`. `context_report` summarises a set of units.
- Object-to-object validity intervals are stored only when the source supplies
  them and declares `relation_time_semantics="interval"`, and are then used on
  the half-open convention `[valid_from, valid_to)`. An atemporal source
  qualifies its units with `relation_validity_unknown` instead of inferring an
  interval.
- New qualification codes for the object path: `context_truncated`,
  `depth_limit_reached`, `fan_out_limit_reached`, `binding_limit_reached`,
  `attribute_not_set_at_time`, `attribute_read_outside_evaluation_time`,
  `relation_validity_unknown`.
- `docs/semantics/object_centric.md`, and the pinned interchange fixtures
  `tests/fixtures/ocel/p2p_mini.json` and `p2p_mini_defects.json`.

### Added — one numerical kernel, native relational checks and quantity accounting (experimental, opt-in)

- `wise._aggregation`, the single applicability-aware scoring formula,
  extracted from `score`: evaluated violations, raw weights, a
  `LayerAssignment`, row identities and an explicit mode in; scores, layer
  contributions and per-row effective weights out, with the same
  missing-value conventions. `wise.scoring.score` and `wise.oc.score_units`
  both use it, so a case score and an object score cannot drift apart.
  `scoring._effective_weights(M, w, norm, mode)` keeps its signature and
  delegates. On the BPI Challenge 2019 log the extracted kernel reproduces the
  reviewed base's arithmetic exactly (max absolute difference 0.0, both modes).
- `wise.oc.constraints`: three native relational check families in their own
  catalogue (`OBJECT_CHECKS`, `object_check_from_dict`), never reachable
  through `Norm.from_dict`.
  - `RelatedObjectCardinality` counts distinct related identities against a
    declared minimum and maximum. A shortfall needs a declared
    `completeness="assumed_complete"`; otherwise the reason is
    `unverified_absence` and no violation is produced. In a truncated role the
    count is a lower bound and a shortfall is `budget_truncated`.
  - `CrossObjectLag` selects activation and response events through object
    relations, with `match`, `clock`, `equal_time` and the two missingness
    policies as declared fields. A tie on the matching instant is a typed
    `ambiguous_match`; `on_ambiguous="tie_break"` resolves it and records that
    it did; there is no silent nearest-timestamp rule.
  - `RelationalBalance` compares two matched amount sets read at the unit's
    evaluation instant, through the accounting contract, so one amount cannot
    be consumed twice. Mixed currencies are `incompatible_units` unless a rate
    table and a target unit are declared.
- `wise.oc.accounting`: `QuantityRecord` with a canonical
  `object:attribute@instant` identity, `Allocation`, and `allocate`, which
  refuses negative shares, duplicate consumption within a group and
  over-allocation, and reports an incomplete allocation as a residual.
  `AllocationReport.coverage()`, `.overlap()` and `.combined()` separate the
  conserved allocated sum from the distinct evidence behind it and from what
  claiming the full value per group would double count.
- `wise.oc.evaluation`: `ObjectNorm` — a catalogue in its own versioned
  envelope (`wise-oc-norm/1`), reusing `Layer` and `View` unchanged, with its
  own validation path — plus `evaluate_units`, `score_units`, `unit_frame` and
  `object_backlog`. `OCScoreResult` is a separate type from `ScoreResult`: it
  carries `oc_log`, one `EvaluationRecord` per check per unit, and a
  `frame(view)` the existing prioritisation path reads. Ranking stays inside
  one unit type unless a `unit_type` is chosen or `allow_mixed=True` is passed.
- `wise.evaluation.ocel`: `naive_projection` and
  `identity_preserving_projection` with a `ProjectionReport` naming what each
  duplicated and could not carry, `measure` for runtime and peak allocation,
  and `Expectation` / `score_representation` / `compare_representations` for
  comparing both against a native evaluation on a manually specified truth.
- New reason codes `unverified_absence` and `incompatible_units` (both
  non-evaluable), and new qualification codes `ambiguous_match_resolved`,
  `count_is_a_lower_bound`, `unit_conversion_applied`, `allocation_incomplete`,
  `overlapping_groups_not_summed`, `heterogeneous_unit_types`.
- New errors `OCConstraintError` and `AccountingError`.
- `examples/oc_invoice_review.py`, runnable with no data file and no optional
  package, and the object-centric section of `docs/semantics/object_centric.md`.

### Added — bounded local assistance and safe norm drafts (experimental, opt-in)

- `wise.llm`, the library's whole language-model surface, and deliberately a
  thin one: the application owns the assistant, its prompts and its approval
  flow. `import wise` does not import it, and importing it needs no optional
  package, contacts nothing and opens no socket — the transport is
  `urllib.request` and the schema check is written out.
- `llm/provider.py`: the `LLMProvider` protocol, typed `ChatRequest` and
  `ProviderResult`, a `ProviderStatus` for every way a call can fail (absent,
  timed out, oversize, invalid, budget-exhausted, refused), `CallBudget` and
  `BudgetLedger`, and the `FakeProvider` every test in this repository runs
  against.
- `llm/transport.py`: a strict local transport — a loopback literal by default,
  validated when it is *configured*; a route allowlist; redirects refused; the
  final response host compared with the configured one; inherited `HTTP_PROXY`
  dropped unless explicitly trusted; bounded request and response bytes; a
  mandatory timeout and a bounded retry count. The opener is injectable, which
  is how the tests drive every path without a socket.
- `llm/ollama.py`: a read-only provider for a server that is **already
  running**. `/api/chat` with `stream=false` and the schema in `format`; no
  automatic pull, no automatic start, no remote fallback; an exact model
  allowlist, so a reply naming another model is `INVALID` rather than accepted.
  Records the server's own model identity, its digest where one is offered, the
  prompt and schema versions, the options and the status.
- `llm/policy.py`: `Principal`, `Scope` and `AccessPolicy` — supplied by the
  application, never by a response. It permits nothing by default; its
  `row_mask` runs before every aggregate; and `authorised_baseline` recomputes
  an unentitled wider-population comparator inside the authorised population
  and relabels it, returning a `ComparatorChange` that says so.
- `llm/gateway.py`: the six approved read-only tools (`get_run_summary`,
  `get_evidence`, `compare_groups`, `explain_priority`,
  `retrieve_approved_policy`, `preview_norm_patch`), frozen at import, each
  with a declared argument schema, a required scope and a result-size limit.
  Dispatch is a table written out literally: no `getattr` on a supplied name,
  no `eval`, no import chosen by a reply, no shell, no SQL, no path and no URL.
  `preview_norm_patch` is off unless enabled, and is validation-only.
- `llm/retrieval.py`: a local lexical baseline over `ApprovedDocument`s. Access
  tags, approval state and effective dates filter **before** anything is
  ranked; every `RetrievedSpan` keeps its document id, span offsets, digest,
  version, dates and tags; two approved sources that disagree produce a
  `ReviewQuestion` rather than a silent choice; vague language with no quantity
  produces a question rather than an invented number; and an event log cannot
  be indexed as a source at all.
- `llm/schemas.py`: the tool-selection, explanation-draft and norm-draft JSON
  schemas, sent to the server and re-checked locally, with a strict reader that
  refuses unknown keys and bounds every string and array.
- `llm/assistant.py`: two bounded steps around a report that is rendered
  *first*. A draft may reorder facts and add questions and hypotheses; it
  cannot supply a value, and `check_references` refuses valid JSON whose fact
  or evidence identifiers do not exist. `render_reviewed_report` prints the
  packet's own numbers, keeps hypotheses in a separate section marked
  unverified, counts the facts a draft omitted, and prints every mandatory
  qualification whether or not the draft mentioned one.
- `llm/drafts.py` and `wise.schema`: the `NormDraft` envelope (the roadmap
  contract exactly, through `to_contract_dict()`, with this library's findings
  kept outside it); catalogue introspection read off `CONSTRAINT_TYPES` and the
  recipe key tables rather than a second hand-written list; and a deep check
  that refuses `kind="eval"`, unknown kinds and evaluators, unreviewed keys,
  unbounded regular expressions and out-of-range resource shapes **before** any
  derivation — the norm loader is not the security boundary.
  `preview_norm_draft` scores a candidate on an isolated copy, diffs the
  assumptions and the results, checks proposed examples against the units they
  name, and can never apply or activate anything; a stale parent fingerprint is
  a `DraftConflict`.
- New errors, all deriving from `WiseError`: `LLMError`, `TransportError`,
  `BudgetExceeded`, `AccessDenied`, `GatewayError`, `DraftError`,
  `DraftConflict` and `UnsafeDraft`.
- `wise.derive` gains `REQUIRED_RECIPE_KEYS`, `OPTIONAL_RECIPE_KEYS` and
  `UNSAFE_RECIPE_KINDS` as module constants — one maintained table that
  `validate_recipe`, `wise.schema` and the untrusted path all read. Trusted
  `eval` recipes are unchanged and still work.
- An optional `[llm]` extra, `examples/local_review.py`,
  `examples/review_norm_draft.py`, `docs/security/local-assistant.md`, the
  `live_llm` pytest marker, and the opt-in `tests/extensions/test_live_ollama.py`
  (skipped unless `WISE_OLLAMA_LIVE=1`).

### Changed — evidence semantics, after review

- `EvidencePacket.from_dict` and `wise.load_evidence(path)` read an exported
  packet back. Every record is rebuilt under the invariants it was written
  under, so a censored check comes back censored and a file claiming a
  violation for an unevaluated check is refused on read. A restored packet
  carries neither the snapshot nor the norm, declares that (`packet.restored`
  and a `witness_access` block in the export), and refuses
  `packet.witnesses(...)` instead of returning an empty tuple.
- A capturing `score(..., evidence=...)` now takes a `scope="events"` snapshot,
  and an `"events"` snapshot hashes **every** column of the event table rather
  than the three a witness prints. A relabelled activity, a shifted timestamp
  and a changed amount now all invalidate lazy witness access, which previously
  returned an empty list for a summary packet.
- `LogSnapshot.content_hashed_tables` reports coverage per table, and
  `LogSnapshot.content_hashed` is `True` only when every table is covered — a
  `"cases"` snapshot is therefore no longer described as content-hashed.
  `manifest.preprocessing["content_hashed"]` is a map per table instead of a
  single `true`, and a packet whose fingerprint misses a table carries a
  run-scope `snapshot_not_content_hashed` qualification naming which.
- `max_records=` and `units=` now emit a run-scope `records_truncated`
  qualification with the captured and total counts, and a record budget that
  stops the traversal before every constraint is reached sets
  `Truncation.evaluation_truncated` and emits `evaluation_truncated` naming the
  constraints with no row. Both codes were declared and never used.
- `to_interchange(packet, view=...)` on a **bounded** packet exports a
  comparator named `captured-subset:<view>:<run>` with a qualification giving
  its denominator, instead of labelling a subset's mean `current_population`.
- `explain_priority` lifts the record-scope qualifications of the slice —
  `lower_bound`, `vacuous_satisfaction`, `both_totals_zero` — to group-scope
  limitations naming the code, the constraints and the counts, and
  `ConstraintComponent` gains `share_lower_bound` beside `share_evaluated`
  (`None` when no evidence packet was supplied, which is not zero). Both
  renderings show the new column.
- A unit that a bounded capture did not keep is no longer skipped silently:
  the packet carries a group-scope `records_truncated` limitation counting how
  many of the units selected for witnesses had no record.

### Errors

- New `EvidenceError`, `EvidenceUnavailableError` and `StaleEvidenceError`,
  all deriving from `WiseError`; new `BaselineError`, deriving from
  `NormError`.
- New `OCError` and its subclasses `OCValidationError`, `OCInterchangeError`
  and `OCUnitError`, deriving from `WiseError` and deliberately not from
  `EvidenceError`: an object log can be wrong long before any evidence is
  captured from it.

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
