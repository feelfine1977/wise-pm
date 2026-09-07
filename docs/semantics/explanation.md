# One comparator, one explanation

*Status: experimental, opt-in, added on the extension branch. The defaults of
the library are unchanged: `prioritize(result, by)` and
`layer_drivers(result, by)` compute exactly what they always did, and
everything on this page is switched on by an explicit keyword argument.*

## The problem this fixes

`prioritize` has always accepted a scalar comparator:

```python
wise.prioritize(result, "vendor", view="Finance", baseline=0.95)   # rank against a target
```

`layer_drivers` has always contrasted against the **current scored
population**, and had no comparator argument at all:

```python
wise.layer_drivers(result, "vendor", view="Finance")   # deltas to this log's layer means
```

Put the two outputs side by side and the report describes two different
comparisons of one backlog: the ranking says how far each slice is from 0.95,
and the drivers say which layers are worse *than the rest of this log*. Neither
number is wrong; together, unlabelled, they are misleading.

`wise.explain` resolves the comparator **once** and uses it in both places.

## `BaselineSpec`

A comparator is an object, not a float. It carries

| Field | Why it is there |
|---|---|
| `baseline_id`, `kind` | who the comparator is: `current_population`, `historical` or `target` |
| `reference_score` | the number the gap is measured against |
| `reference_layer_penalties` | the complete layer profile, or `None` for a scalar-only comparator |
| `view`, `scoring_mode` | the view and the **normalisation** the reference was produced under |
| `unit_type` | the assessment unit the reference counts |
| `score_convention`, `aggregation` | that a score is `1 − Σ layer penalties`, and that a mean is unweighted over scored units |
| `norm_fingerprint`, `calibrations` | the configuration in force |
| `population_id`, `population_size`, `source_run_id`, `observed_at` | provenance — **never** compared |
| `reviewed_mapping` | an explicit human decision that authorises a comparison across a changed norm or calibration |
| `tolerance` | the numerical tolerance of the identity below, default `1e-9` |

Validation, on construction:

* the reference score is finite and inside `[0, 1]`;
* every layer penalty is finite and non-negative, and they sum to at most one;
* with a complete profile, `reference_score == 1 − Σ reference_layer_penalties`
  within `tolerance`. The identity is exact in real arithmetic; the tolerance
  absorbs floating-point summation order only. Given only the profile, the
  score is *derived* from it rather than guessed;
* a historical or target comparator must state its score. Only a
  current-population comparator is resolved from the run itself.

```python
from wise.explain import BaselineSpec

target   = BaselineSpec.target(0.95, baseline_id="board-2026")           # scalar only
frozen   = BaselineSpec.from_result(last_quarter, "Finance")             # score and profile
explicit = BaselineSpec.historical(None, {"lead_times": 0.10, ...})      # score derived
```

### Compatibility is about semantics, not about populations

A historical comparator comes from a *different* population; that is the point
of it, and population identity is never compared. What may not differ silently
is what the numbers mean. `check_compatible` reports three separate things:

* **reasons** — a real mismatch: a different view, a different normalisation
  (scoring mode), a different assessment-unit type, a different score
  convention or aggregation, a different layer partition, a changed norm
  fingerprint, a changed calibration set. Any of these refuses the comparison
  with `BaselineError`, naming every mismatch;
* **undeclared** — the specification says nothing about a field, so nothing was
  compared. It is recorded as a limitation, not assumed to match;
* **unverifiable** — the *assessment* cannot state a field (a bare score frame
  knows no scoring mode), so the specification's declaration could not be
  checked. Also a limitation, not a mismatch.

A changed norm or calibration set can be authorised, but only explicitly:
supply `reviewed_mapping=` and the comparison proceeds with a
`reviewed_mapping_applied` limitation attached. Nothing is aligned silently,
and **an unknown layer is never aligned with zero** — `profile_for` refuses a
layer the comparator does not mention and a layer the assessment does not
define.

## The two functions, changed together

```python
wise.prioritize(result, by, ..., *, baseline_spec=None)
wise.layer_drivers(result, by, ..., *, baseline_spec=None)
```

Both parameters are keyword-only and appended, so every existing call keeps its
signature, its columns, its ordering, its values and its attributes. Passing
both `baseline` and `baseline_spec` is an explicit error: two comparators for
one backlog is the mistake the argument exists to prevent.

* `prioritize` resolves the specification once, uses its score as `μ̄`, and
  records `baseline_id`, `comparator`, `comparator_resolved_from`,
  `explanation_kind` and the full `baseline_spec` in `DataFrame.attrs` — but
  **only** when a specification was supplied. A call without one keeps exactly
  the five attributes it always had.
* The `global_mean` **column keeps its name and position** for compatibility.
  With a comparator, the reference it holds need not be global; the attributes
  and the explanation packet say which comparator it is, and a
  `comparator_is_not_global` limitation says so in prose.
* `layer_drivers` with a compatible complete profile subtracts *that* profile.
  With a scalar-only comparator it keeps the absolute layer means, sets every
  `<layer>__delta` to null and `dominant_layer` to `None`, and reports
  `attrs["reference_profile_available"] = False` with
  `attrs["unavailable_reason"] = "reference_profile_unavailable"`. It does not
  substitute the current population's profile.
* The population mean is computed over every scored unit **before** the
  minimum-support filter, exactly as before.

## The exact contrast

For a slice `s` under view `p`, with the same scored mask, view, keys and
aggregation as the priority:

```text
delta_layer = mean_group_layer_penalty − reference_layer_penalty
signed_gap  = reference_score − mean_group_score
rho         = n_scored / (n_scored + gamma)
PI          = volume · max(signed_gap, 0)
stable_PI   = volume · rho · max(signed_gap, 0)
component   = volume · rho · delta_layer
```

Under a compatible complete profile the layer deltas sum to `signed_gap` and
the signed components sum to `stable_PI`. `explain_priority` **asserts** both
identities and raises rather than reporting a decomposition that does not
reconcile.

* **Negative offsets stay.** A slice can be better than the comparator in one
  layer and still carry priority. Showing only the positive layer terms no
  longer reconciles to the total, so all of them are shown.
* **Clipping is explained, not hidden.** For a non-positive gap the index and
  the clipped components are zero, while the unclipped contrasts remain
  visible with a `non_positive_gap_clipped` limitation.
* **No division by a zero net gap.** Contribution shares exist only when the
  stabilised index is strictly positive; otherwise they are `None`.
* **Support is counted in the declared unit.** `gamma` is in the same unit as
  `n_scored`, and both are named in the packet's denominators.
* **Exposure scales the index and nothing else.** The mean score is the
  unweighted mean over scored units; it is never silently exposure-weighted.
  A zero-exposure slice keeps its gap and loses only its index.

## `ExplanationPacket`

```python
packet = wise.explain_priority(result, "vendor", view="Finance", gamma=20,
                               evidence=result.evidence)
print(wise.render_explanation(packet, "markdown"))
```

The packet is versioned (`wise-explanation/1`) and keeps four kinds of claim
apart, because merging them is how a report starts saying more than it knows:

| `FactKind` | The question it answers |
|---|---|
| `observed_assessment` | what was observed in this slice |
| `relative_priority` | why it ranks where it does, against which comparator |
| `alternative_view` | how another approved view changes the reading |
| `evidence_qualification` | how much of the evidence was actually evaluated |

Every `Fact` carries an id, a value, a unit, a description, the name of a
`Denominator`, the method that produced it and at least one evidence
reference. Aggregate facts resolve to the run and to an `aggregate:` identity
(run, view, slice); unit-level references are evaluation ids of the run's own
evidence packet. Alongside the facts the packet carries the resolved
comparator, the layer components, the constraint drivers inside the slice, the
per-view contrasts, bounded witnesses and the limitations.

`explain_priority` needs a `ScoreResult`, not a score frame: an explanation
resolves every number to a run.

### What the packet always discloses

`reference_profile_unavailable`, `non_positive_gap_clipped`,
`comparator_is_not_global`, `comparator_semantics_undeclared`,
`different_scored_populations`, `below_minimum_support`, `null_group_key`,
`single_unit_group`, `unscored_unit`, `shrinkage_stabilised_not_stable`,
`exposure_scales_priority`, `reviewed_mapping_applied` and — always —
`priority_is_not_a_cause`.

With an evidence packet it also discloses **how** the checks under the slice
were answered. A record-scope qualification is a statement about one answer, and
an aggregate over those answers inherits it, so `lower_bound`,
`vacuous_satisfaction` and `both_totals_zero` are lifted to group-scope
limitations naming the code, the constraints and the counts — a slice whose
evaluations are three-quarters censored no longer reports a full evaluated share
and says nothing else. `ConstraintComponent.share_lower_bound` carries the same
fact per constraint, beside `share_evaluated`; it is `None`, not `0.0`, when no
evidence packet was supplied, because unknown is not "none of them".

A unit that a **bounded** capture did not keep is disclosed too: the packet
carries a group-scope `records_truncated` limitation counting how many of the
units selected for witnesses had no record, instead of quietly showing fewer
references.

Different scored populations deserve a word. A view may give zero weight to the
only constraint that applies to a unit, so two views of one log can score
different sets of units. The per-view rows then are not a common-population
comparison, and the packet says so; a view in which the slice has no scored
unit at all gets no row rather than a row of zeros.

## Rendering

`render_text`, `render_markdown` and `render_json` are pure functions of the
packet: the same packet renders to the same bytes. The renderer prints facts
and never computes one, which is the property that keeps a later language model
from introducing a number nobody asserted.

Two escapes, for two different kinds of text: `escape_markdown` fully escapes
untrusted **labels** (slice keys, activity names, ids), and `escape_prose`
neutralises only the structural characters in the library's own sentences, so a
description stays readable while it still cannot end a table cell, inject HTML
or become a link. Generated file names come from `explanation_filename`, built
from the explanation id and a sanitised label — a slice called
`../../etc/passwd` cannot escape the chosen directory.

The vocabulary is deliberate: *assessed penalty*, *reference contrast*,
*priority component*, *shrinkage-stabilised*. Not *root cause*, not *stable*,
not *money saved*.

## Command line

```bash
wise score norm.json log.csv --by vendor --view Finance \
    --baseline-file target.json --manifest-out run.json \
    --evidence-out evidence.json --explain-out explanation.json
wise explain explanation.json --format markdown --out report.md
```

`--baseline-file` reads a reviewed `BaselineSpec` as JSON — strictly, with
unknown keys refused, and nothing in it executed. It changes the reference the
backlog ranks against, which is what it is for, and changes no column. The
sidecars never touch the stdout CSV; their messages go to stderr.
`--explain-out` writes the packet for the top-ranked slice of the backlog just
computed. `wise explain` computes nothing: every number it prints is a fact of
the packet it reads.

## What this is not

A large layer component is where the assessed gap sits. It is not a
demonstrated cause of that gap, not an avoidable amount of money, and not a
recommendation. The shrinkage factor borrows strength from the comparator; it
is not a confidence statement. Coverage says how much was observed, not how
likely the result is to be right, and nothing here multiplies a score by it.
