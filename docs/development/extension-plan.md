# Extension plan — evidence, explanation and optional local assistance

Status: **development branch only.** Nothing in this document describes a
released capability, a published version or an evaluated research result. The
current released behaviour of `wise-pm` is unchanged.

## 1. Base and scope

| Item | Value |
|---|---|
| Reviewed base commit | `df5db50b839cc124b489a269894f5a2bfe7dc634` |
| Feature branch | `feat/actionability-ocpm-local-llm` |
| Requested scope | `IMPLEMENTATION_SCOPE=foundation` — stages S0, S1, S2 |
| Items in scope | E05 (S0), E01/E02/E04 (S1), E03 (S2) |
| Excluded by scope | local language assistance, object-centric data, investigation records, evaluation harness, research-gated work |
| Remote writes | disabled; no push, pull request, tag or release |
| Model downloads / live model tests | disabled |
| Private data export | disabled |

Python floor stays 3.10; base dependencies stay NumPy and pandas. New
capabilities must be importable without optional packages, network access or a
local server.

## 2. Backlog identities

The handoff supplies all twenty-four roadmap identities with their original
priorities, dependencies and acceptance conditions, together with the preserved
original roadmap and backlog. They are reproduced here as identities only; the
estimates attached to them in the original backlog are planning judgements, not
completion promises.

| Stage | Identities |
|---|---|
| S0 | E05 |
| S1 | E01, E02, E04 |
| S2 | E03 |
| S3 | L01, L02, L03 |
| S4 | O01, O02, O03, O04, O05 |
| S5 | A01, A02, A03, N01, N02 |
| S6 | L04, M01, V01, M02, M03, M04 |

Dependency picture for the foundation scope: E05 and E01 have no predecessors;
E02 and E04 depend on E01; E03 depends on E01. Outside the scope, N01 and M01
depend on E05, so this stage's contract is a gate for later work as well. V01
requires an actual field study and M02–M04 are research-gated; none of them can
be satisfied inside a coding session, and none is attempted here.

## 3. Stage 0 result — what the unchanged library actually does

All commands were run inside the branch worktree on Python 3.13.9 with
`.[dev,stats]` installed. Results are reproduced in
[extension-checkpoint.md](extension-checkpoint.md) with their real counts.
Summary: the base is **green**. Nothing was already failing, and the only skips
are an absent optional parquet engine and the BPIC dataset gate.

The regression contract is executable in
`tests/extensions/test_baseline_contract.py`. It pins:

* the public export list and its relative order, the module constants
  (`SCHEMA_VERSION = 2`, `SCORING_MODES`, `DEFAULT_SCORING_MODE`,
  `CONSTRAINT_TYPES`), and that importing `wise` needs no optional package;
* the signature of every public entry point, as an *ordered prefix*: recorded
  positional-or-keyword parameters must stay exactly as they are and recorded
  keyword-only parameters must keep their defaults, while a **new keyword-only
  parameter may be appended**;
* the dataclass field order of `ScoreResult`, `Norm`, `NormConstraint`, `Layer`
  and `View` as a prefix, so new fields can only be appended and must carry a
  default; and that a manually constructed `ScoreResult` still works;
* the canonical serialisation and SHA-256 fingerprints of the two shipped norms,
  and that a schema-2 round trip through `dumps`/`loads` and
  `to_dict`/`from_dict` reproduces the same fingerprint;
* the running example's frames: shape, index name, column order, dtypes and the
  exact NaN mask of the violation matrix, the boolean `in_scope` matrix, the
  scores in both modes, the layer contributions and the effective weights;
* that `violation_matrix(..., return_scope=True)` is exactly a two-item return;
* the prioritisation contract: column order (including where the `z` columns
  sit), index name, ranking order, the deterministic tie break, `DataFrame.attrs`,
  and the values of the running example's backlog, layer drivers, constraint
  drivers, penalty mass, worst cases, view agreement, hotspots and Pareto;
* the CLI's default stdout CSV header, and that stderr stays empty;
* the private weight kernel `_effective_weights` in both modes, including that
  "no applicable constraint" is NaN and not `0.0`.

**Tolerances.** Identities, orderings, masks, dtypes, column and index names and
fingerprints use strict equality. Floating-point values use `rtol=0,
atol=1e-12`, which is the documented allowance for an unchanged operation order
(the measured base residuals are around 1e-16). The existing benchmark
tolerances are untouched: `tests/test_bpic19.py` keeps `abs=1e-3`, `5e-4` and
`1.0`, and `tests/test_scoring.py` keeps `abs=5e-5`.

## 4. Semantic conflicts found in the base

These are properties of the current code, confirmed by reading it and by running
it. They are the problems the foundation scope exists to fix, and each is
recorded so that a later stage cannot quietly resolve one by changing a number.

1. **Two different comparisons in one backlog.** `prioritize` accepts a scalar
   `baseline` that replaces the global mean, while `layer_drivers` always
   contrasts a slice against the mean layer contribution of the *current* scored
   population and takes no baseline argument at all. Used together, the reported
   gap and the reported layer deltas answer different questions. This is E03.
2. **A scalar target has no layer profile.** Because a `baseline` is one number,
   there is no compatible reference decomposition for it. Any additive layer
   attribution against such a target would have to be invented. E03 must refuse
   it explicitly rather than fall back to the current population.
3. **Clipping hides the sign.** `prioritize` clips both `gap` and `stable_gap`
   at zero, so a slice above the baseline and a slice exactly at it are
   indistinguishable in `PI`, while `layer_drivers` keeps signed deltas. The
   signed contrast must remain visible next to the clipped index.
4. **Penalty mass is not a priority index.** `penalty_mass` sums `1 − S` with no
   baseline and no volume scale; `PI` is `volume · (baseline − mean)₊`. They must
   stay separately named and separately interpreted.
5. **Scored populations can differ between views.** `prioritize`,
   `layer_drivers` and `constraint_drivers` each drop cases that are unscored in
   the selected view. Two views can therefore aggregate different case sets, and
   nothing currently discloses that.
6. **`ScoreResult.log` is the live, mutable log.** `score` stores the same
   `EventLog` object and copies only `cases`. `EventLog.add_case_attribute` and
   `EventLog.derive` mutate that log in place, so after scoring `result.cases`
   and `result.log.cases` can disagree — confirmed by running it. Any lazy
   witness access added in S1 must either hold an immutable snapshot or verify an
   input fingerprint and raise a stale-evidence error.
7. **Scoring mutates its input by default.** `score(..., derive=True)` calls
   `log.derive(norm.derived_attributes, overwrite=False)`, which writes new
   columns into the caller's log and appends to `log.case_attributes`. A draft
   preview must work on an isolated copy.
8. **A fitted reference is recomputed, not frozen.** The `quantile_scale` recipe
   recomputes its quantile from whatever log it is applied to. Comparing two
   periods therefore silently changes the reference. S1 needs explicit
   capture-and-apply of a fitted reference (acceptance F12).
9. **`ScoreResult.mode` defaults to `"flat"` while `Norm.scoring_mode` defaults
   to `"layer_balanced"`.** A manually constructed result therefore claims the
   wrong mode. A run record must carry the mode actually used, not the dataclass
   default (acceptance F07).
10. **No event identity for witnesses.** `EventLog` accepts `event_id_col` but
    only `diagnostics.cross_case_replication` consults it; there is no accessor
    that returns event identities. S1 must expose either the supplied source
    identity or a clearly labelled snapshot-local row reference, never an
    invented event id (acceptance F10).

### State of these conflicts after stage S2

| # | Conflict | State |
|---|---|---|
| 1 | two comparisons in one backlog | **addressed**: one `BaselineSpec`, resolved once and used by `prioritize`, `layer_drivers` and `explain_priority`; passing both `baseline` and `baseline_spec` is an explicit error |
| 2 | a scalar target has no layer profile | **addressed**: `reference_profile_unavailable`; the deltas are null, the absolute layer means stay, and the current population's profile is never substituted |
| 3 | clipping hides the sign | **addressed**: `signed_gap` beside `gap`, unclipped contrasts beside clipped components, and a `non_positive_gap_clipped` limitation |
| 4 | `penalty_mass` is not `PI` | **preserved and separated**: `penalty_mass` is untouched, and the explanation packet reports `PI` and `stable_PI` with their own denominators and method strings |
| 5 | scored populations can differ between views | **addressed**: per-view contrasts each against their own current population, with a `different_scored_populations` disclosure, including views in which the slice has no scored unit at all |
| 6 | `ScoreResult.log` is live and mutable | **addressed**: `EventLog.snapshot()` fingerprints the log (or freezes the witness columns), and lazy witness access raises `StaleEvidenceError` |
| 7 | scoring mutates its input | **disclosed**: unchanged behaviour, recorded as a manifest note and an `input_mutated_by_derive` qualification. The isolated preview belongs to S5 |
| 8 | a fitted reference is recomputed | **addressed**: `fit_calibration` / `apply_calibration` / `score(calibrations=...)`; the default recalculation is unchanged |
| 9 | `ScoreResult.mode` misreports a manual result | **addressed**: `RunManifest.mode` and `mode_source` carry the mode actually used; the dataclass default is untouched |
| 10 | no event identity for witnesses | **addressed**: source ids when `event_id_col` is declared, otherwise `snapshot-local:row=` references labelled `SNAPSHOT_LOCAL` |

## 5. API additions

Stage S1's additions are **implemented**; the rest are still proposals. Every
addition is keyword-only or a new module, so the contract in section 3 keeps
passing unchanged.

```text
IMPLEMENTED in S1
wise.evidence                     lightweight subpackage (standard library + numpy/pandas)
  models.py      EvaluationRecord, WitnessRef, AbsenceSearch, Qualification, ReasonCode,
                 DiagnosticResult, CoverageReport, Truncation, EvidencePacket
  manifest.py    RunManifest (run id, actual mode, norm fingerprint, input identity, log
                 options, observation policy, calibrations, environment, config fingerprint),
                 fingerprint_events, fingerprint_result
  capture.py     capture_evidence, evidence_frame, coverage_report, to_interchange,
                 interchange_schema, bounded witness materialisation
  calibration.py CalibrationRecord, fit_calibration, apply_calibration

wise.score(log, norm, views=None, *, mode=None, derive=True,
           evidence="none", calibrations=None)      # two appended keyword-only options
wise.ScoreResult.manifest / .evidence               # appended defaulted fields
wise.ScoreResult.evidence_frame(view=..., ...)      # new accessor, existing fields kept
wise.evaluate_detailed(log, norm)                   # matrices plus the primitives behind them
wise.EventLog.options() / .snapshot(freeze=, scope=)
wise.EventLog.derive(..., calibrations=...)
wise.typed_event_replication / typed_cross_case_replication / typed_right_censored
wise.coverage_report(result)

IMPLEMENTED in S2
wise.explain                      lightweight subpackage (standard library + numpy/pandas)
  baseline.py    BaselineSpec, BaselineKind, BaselineError, AssessmentContext,
                 BaselineCompatibility, ResolvedBaseline, resolve_baseline,
                 norm_context, load_baseline
  priority.py    explain_priority, ExplanationPacket, Fact, FactKind, LayerComponent,
                 ConstraintComponent, PriorityComponents, ViewContrast, Denominator,
                 safe_identifier
  render.py      render_text / render_markdown / render_json / render (aliased
                 render_explanation), write_explanation, explanation_filename,
                 load_explanation, escape_markdown / escape_prose / escape_text

wise.prioritize(..., *, baseline_spec=None)         # keyword-only; scalar `baseline` kept
wise.layer_drivers(..., *, baseline_spec=None)      # keyword-only
wise.explain_priority(result, by, group=None, ...)  # exact signed explanation
CLI: wise explain <packet>; wise score --baseline-file --explain-out
```

The proposed `layer_drivers(..., reference_profile=...)` was **not**
implemented: a second way to state a profile is a second comparator, which is
the mistake the stage exists to prevent. The profile travels inside the
`BaselineSpec`.

Rules that follow from section 4 and are binding on the implementation:

* `score` keeps its default output, its default mode and its performance
  characteristics when `evidence="none"`.
* `violation_matrix(..., return_scope=True)` never grows a third element.
* A `baseline_spec` resolves once and is recorded; `prioritize` and
  `layer_drivers` reject an incompatible pairing rather than align it silently
  (implemented in S2; the rejection names every mismatch).
* Missing, unevaluable and out-of-scope stay three different states, and a JSON
  export writes an explicit null plus a reason, never a bare NaN.

## 6. Packaging and CI

Stage S1 added one packaging change: the interchange schema and its example are
shipped as package data (`[tool.setuptools.package-data] wise = ["py.typed",
"evidence/schemas/*.json"]` plus a `MANIFEST.in` line) and read through
`importlib.resources`, verified from an installed wheel outside the source tree.
No new pytest marker was needed. No packaging change was required by stage 0. The wheel already ships `py.typed`
and passes `twine check --strict`. Later stages that ship JSON schemas must add
them through explicit package data and verify them from an installed wheel, not
from the source checkout. The 85% coverage floor and the current lint and type
rules stay; new code is tested rather than excluded. New pytest markers must be
registered in `pyproject.toml` because `filterwarnings = ["error"]` turns an
unknown marker warning into a failure.

`.gitignore` was extended for local environments, build and test output, local
run payloads, private data and model caches. Small synthetic fixtures under
`tests/` and `examples/` match none of the added patterns and stay tracked.
