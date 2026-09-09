# Concepts and interpretation

## Norm, applicability and missingness

A norm groups constraints into layers; a view states how their penalties are
weighted. A violation is between zero and one. An out-of-scope check, an
in-scope check that cannot be evaluated and an evaluated violation are different
states. `ScoreResult.in_scope` and `violations.notna()` distinguish the first
two; opt-in evidence adds a reason code. `NaN` is never a satisfied check.

No positively weighted evaluated check means no score. That unit is excluded
from the selected view's aggregates, so different views can compare different
populations. Evidence coverage counts describe evaluation and capture; they are
not confidence intervals or proof that the source log is complete.

`Lag` defaults to the first activation and first response at or after it, with
undefined endpoints treated as violations. Missingness, activation and response
policies can change that. The shipped paper/BPIC norms pin their own policies;
do not replace those historical settings with library defaults.

## Weighting and decomposition

`Norm.scoring_mode` defaults to `layer_balanced`. It normalises the evaluated
constraints within each layer and then the layers with applicable weight.
`flat` normalises raw constraint weights directly over the evaluated checks of
each unit. The running P2P norm uses `flat`; the BPIC19 norm uses
`layer_balanced`. These settings matter when only part of a layer is evaluated.

For a scored unit, layer contributions sum to `1 - score`. This is a penalty
decomposition. `penalty_mass` adds penalties across cases; it uses no comparator
and is a different quantity from Priority Index.

## Priority, exposure and comparators

Priority is `volume × max(reference_mean - slice_mean, 0)`. The default volume
is the number of scored cases and the reference is the current scored
population mean. `gamma` shrinks a slice mean toward that reference using its
scored support; `stable_PI` uses the resulting gap. Exposure changes the volume
multiplier, not the arithmetic means of the scores. Exposure is a declared
quantity, not automatically money, avoidable loss or causal benefit.

Classic scalar `baseline=` changes the priority reference, while classic
`layer_drivers` compares layer penalties with the current population. On the
extension, pass the same `BaselineSpec` to `prioritize`, `layer_drivers` and
`explain_priority` to align their comparison. Do not supply both scalar and
structured comparators.

Current-population, historical and target comparators answer different questions.
A current-population comparator moves with resampling; a fixed target or frozen
historical reference stays fixed. A scalar target has no reference layer profile:
the explanation reports `reference_profile_unavailable` instead of inventing an
additive attribution. Signed gaps and contributions remain visible even when
the positive-part priority clips them to zero.

## Evidence, units and reproducibility

Request case evidence during scoring with
`score(..., evidence="summary" | "full")` and read `result.evidence`. The lower-level
`capture_evidence` case path requires `details=` from the matching evaluation;
it refuses a plain result without those measurements. Native object capture uses
the records already held by `OCScoreResult`. A manifest records the norm, mode,
input identity and preparation choices. Lazy case witnesses require an unchanged snapshot;
frozen capture retains the selected witness material. Native object packets use
records already produced by evaluation and do not carry a case-log snapshot.

Use explicit `wise.oc` imports and a `UnitSpec` to define object assessment units.
An invoice count and an obligation count are different volumes. Prefer one unit
type per result for shared explanations and comparator helpers; see the
[mixed-result limitations](capabilities.md). Traversal budgets, evaluation
budgets and evidence capture limits restrict different things and must remain
visible in reports. Allocation shares conserve additive evidence within one
`allocate` call, not across independent calls.

Record the package version plus the exact commit and any local changes. The
development version identifies a development series, not every source revision.
The library does not read Git state on import; callers can attach a reviewed
commit with `manifest.with_commit(...)`. `score(..., derive=True)` can mutate
the input log by adding derived attributes. For fixed quantile references, use
the explicit calibration fit/apply APIs and retain the calibration identity.

## Assistance and sensitivity

`wise.llm` permits bounded read-only tools and reviewable drafts under an explicit
access policy. Schema and identifier validation do not establish that prose is
true. Approval, audit history and activating a norm belong to the application.

`wise.evaluation.llm` tests recorded boundary behavior, including an authorised
response control. `wise.evaluation.sensitivity` reports sampling, parameter and
construction variation separately. Dependence must be declared for resampling;
an unvaried source cannot support a stability claim. These diagnostics do not
establish live-model quality, stakeholder usefulness or business benefit.
