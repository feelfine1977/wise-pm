# 0001 — One canonical algorithm library and the migration map

Status: accepted 2026-09-27 (proposed the same day; the ten questions were answered by the maintainer one by one).

## Context

Three development states of this repository existed on 2026-09-26: `main`
(classic 0.1.0, `7293b5a`), the extension prototype (`08513bd`,
evidence, explanation, object-centric, sensitivity modules on
top of a modified core) and `next` (`a80f814`, the classic code relocated
into a uv workspace with a test safety net). The workbench repository holds
an analytics package (`wise-analytics`, 5 100 lines, numpy/pandas only) that
computes on `wise` objects, and knowledge templates that are norm files.

## Decision

1. **One owner per public computation.** The `wise-pm` distribution
   (import `wise`) owns every reusable WISE computation: input semantics,
   constraint evaluation, the single effective-weight and score kernel,
   prioritisation, drivers, diagnostics, selection and membership, and the
   deterministic analysis operations that today live in the workbench
   analytics package (quality and readiness, variants and paths, contrast
   point estimates, timing basics). Optional capabilities (evidence,
   explanation, object-centric assessment, and the resampling, scenario and
   subgroup analytics: uncertainty, what-if, subgroups) ship as `wise-pm-actionability`
   (import `wise_actionability`) from the same workspace, released in
   lock-step, and change no core number or membership when installed.
2. **Consumers and shims.** Consumers today: the workbench backend (pins
   `wise-pm >=0.1,<0.2`), `wise-analytics`, `process-knowledge` (norm
   templates, schema versions 1 and 2), the Studio prototype, and the paper
   reproduction script. Temporary shims: `wise.compat` (re-exports of every
   name and private the consumers touch today, with deprecation warnings;
   removed at 1.0) and a `wise_analytics` package that delegates to
   `wise.analysis` and `wise_actionability` slice by slice, running the
   workbench implementation for every function not yet ported; it ships from
   the workbench repository under that repository's licence (both shims
   removed at 1.0).
3. **Recipe, calibration and missingness decisions.**
   - `eval` recipes are removed from norm schema v3; the v2→v3 migration
     rewrites a bare case-table column (the only shipped use, `n_events`) to a
     typed `column` recipe and refuses any other expression with the recipe
     named.
   - `quantile_scale` carries a frozen calibration record (divisor, quantile,
     fitted-on log fingerprint, size, window, status fitted or unfitted);
     v3 admits `unfitted`, scoring refuses it, comparisons across different
     calibrations refuse without an explicit mapping.
   - Applicability is three-valued (in scope, out of scope, unknown); a missing
     attribute is unknown under every operator, including negations, and is
     excluded from denominators; the policy is a named v3 field.
   - `Lag` requires an explicit bound or `unbounded=True` in v3.
4. **Legacy semantics profile.** Documents with schema version 2 evaluate under
   the 0.1.0 semantics (bare-lag default, legacy missing-value scope,
   population quantile with the realised divisor recorded in the run
   specification, `eval` only for a single existing case-table column; any
   other expression is refused at load with the recipe named) with one
   `LegacySemanticsWarning` per loaded norm, so the paper's Section V
   reproduces bit-for-bit. `Norm` carries its schema version as a frozen
   field: the Python constructor defaults to the current version, `from_dict`
   requires the key, `replace*` preserve it, `to_dict` emits the carried value
   and only `migrate()` changes it (always with a migration report). The
   missing-timestamp rules of the event log are keyed on the log schema in the
   same way. v2 documents keep evaluating until 2.0; v3 documents use the
   corrected semantics.
5. **Parity policy.** See `docs/semantics/parity.md`: unchanged classic
   arithmetic is compared exactly on every CI environment; corrections and new
   methods declare their own acceptance.

## Capability names

`wise.capabilities()` returns a frozen set of strings. Core: `core`, `artifacts`,
`selection`, `analysis`. Extension: `evidence`, `explain`, `oc`, `uncertainty`,
`whatif`, `subgroups`. Reserved, unimplemented and documented as such:
`measurement`, `models`, `effects`.

## Disposition of the workbench analytics package (decided 2026-09-27)

The package is licensed PolyForm-NC today. The split below is the licence
line; the author relicenses under MIT only the parts that move into the core.

| Module today | Destination | Licence |
|---|---|---|
| `quality` (incl. drift and vocabulary checks), `readiness` | `wise.analysis.quality`, `wise.norm.Readiness` | MIT core |
| variants and paths (from `contrast`/selection helpers) | `wise.analysis.variants`, `wise.selection` | MIT core |
| `contrast` point estimates and denominators | `wise.analysis.contrast` on `ComparatorSpec` | MIT core |
| timing basics (`lead_time`, `elapsed`) | `wise.analysis.timing` | MIT core |
| what-if rescoring primitive | `wise.scoring.rescore` | MIT core |
| `uncertainty`, `bootstrap_backlog`, interval methods for contrast, sensitivity grids | `wise_actionability.uncertainty` on one `SamplingDesign` | PolyForm-NC extension |
| `whatif` scenarios, `headroom` | `wise_actionability.whatif` | PolyForm-NC extension |
| `subgroups` | `wise_actionability.subgroups` | PolyForm-NC extension |
| `provenance` | replaced by `AnalysisSpec` | MIT core |
| `synthetic` | merged into `wise.datasets.synthetic` | MIT core |
| `plain`, `demo` | stay in the workbench | unchanged |

The `wise_analytics` shim delegates to both distributions, runs the workbench
implementation for every function not yet ported, and raises a capability
error for a ported extension function when the extension is absent. The
workbench probes `wise.capabilities()` at startup and disables interval,
headroom, bootstrap, subgroup and scenario features when `uncertainty`,
`whatif` or `subgroups` is missing.

## Contracts the application consumers wait for

| Contract | Consumer need | Milestone |
|---|---|---|
| Comparison-kind vocabulary (`ComparatorSpec.kind`: fixed value, population mean, rest of selection, other cohort, other run, period; refusal with `ComparatorError`) | workbench hypothesis tests (WB-04) refuse unsupported kinds today with a 422 and map to these names later | names now, object in M4 |
| Interval method and confidence level carried per interval column on contrast results | WB-04 labels | M4b slice 2 (column slots and label fields on `ContrastResult`, empty in the core); M5 (values from `wise_actionability.uncertainty`) |
| Readiness with per-check measurement state (measured value including zero, unavailable with reason, inapplicable, insufficient support) and denominators | WB-05 | M3 `Readiness`, M4 slice 1 |
| `origin` (observed or scenario) and `baseline_run_id` on every result and manifest | WB-06 | M2 |
| Content fingerprint over consumed columns on an immutable log; memo lives on the frozen object | WB-E1 (stale identity-keyed cache) | M1 |
| `Membership` with a stable digest and count; unsupported scope raises `ScopeError`; stage kind with explicit original activity ids | FLOW-01, FLOW-06, WB-01 | M4 |
| Grouped directly-follows statistics by stage (count, distinct-case union, pooled quantiles) with aggregation kind and units | wise-flow collapsed stages | M4b |
| Witness and deviation tables keyed by original activity ids and pairs plus unit ids | FLOW-03 overlays | M2 seam, M5 |
| Release job writes version, wheel and sdist hashes and the norm schema version into a cross-product compatibility record | FLOW-05, R01 | M0.1 |

## Executed baseline: shim-compatibility floor and slice-1 oracle

Against the source tree of `a80f814` with `WISE_BPIC19_CSV` unset: workbench
analytics 86 passed, knowledge 74 passed and 3 environment skips (recorded by
the engineering audit of 2026-09-27 and by `scripts/compat_gate.sh`). Each
`wise.analysis` slice is accepted when the old suite passes through the shim
against the installed wheel and the slice's typed result equals the old
function's values on the shared fixtures. This run is the oracle for slice 1
(variants and paths, quality and readiness) and the compatibility floor for
the shim at every step; slices 2 and 3 (core) and 4 (extension) are accepted
against a capture taken after the workbench repairs WB-T1 and WB-T4, slice 4
with both wheels installed.

## Packaging fallback

Should the licence split be abandoned, the fallback recorded here is a single
`wise-pm` wheel with optional subpackages behind extras; nothing else in this
record changes. Either way no distribution may install a competing
implementation under the `wise` import name.

## Consequences

- The feature branch is mined module by module onto public seams; it is not
  merged. It is tagged as an archive by the maintainer.
- The workbench analytics package is ported in slices, deterministic
  operations into `wise.analysis` and resampling, scenario and subgroup
  operations into `wise_actionability`, each with installed-wheel parity
  tests; its test suite is the oracle.
- Every result carries a frozen analysis specification (identity of log,
  schema, norm, calibration, view, mode, selection, comparator and code);
  an extension result adds the `SamplingDesign` identity and seed in an
  extension block of the specification.

## Source identities

| Tree | Branch | Commit |
|---|---|---|
| classic | `main` | `7293b5a60076c39eb63c66159456dcba37952030` |
| extension | the extension prototype | `08513bd4879764c61dfb5098fafe022f5653d07a` |
| workspace | `next` | `a80f814` (M0) |
