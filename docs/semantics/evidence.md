# Evidence, run records and fitted references

*Status: experimental, opt-in, added on the extension branch. The defaults of
the library are unchanged: `score(log, norm)` computes exactly what it always
did, and everything on this page is switched on by an explicit argument.*

`ScoreResult.violations` answers *how much* a case violates a constraint. One
`NaN` in that matrix can mean three different things, and a review decision
needs to know which:

| Question | Where the answer lives |
|---|---|
| Is this expectation relevant to this unit at all? | `ScoreResult.in_scope`, `EvaluationRecord.in_scope` |
| Were the observations the check needs available? | `EvaluationRecord.evaluable`, `EvaluationRecord.reason_code` |
| What did the check return? | `EvaluationRecord.violation`, and the measurements behind it |

## Capturing evidence

```python
result = wise.score(log, norm, evidence="summary")   # or "full"
result.manifest                # the run record
result.evidence                # the packet
result.evidence_frame("Finance")   # long-format table, one row per check
```

`evidence="none"` is the default and keeps the historical code path. The three
modes produce **identical** scores, violations, masks and contributions; only
the captured metadata differs. That is a tested invariant, not a claim
(`tests/extensions/test_evidence_capture.py::test_capture_does_not_move_a_single_number`).

A result scored without capture says so:

```python
wise.score(log, norm).evidence_frame()
# EvidenceUnavailableError: this result carries no evidence: it was scored with
# evidence='none'. ... evidence cannot be reconstructed afterwards from a log
# that may since have changed.
```

That refusal is deliberate. `ScoreResult.log` is the *live* log, and both
`EventLog.derive` and `EventLog.add_case_attribute` change it in place. A table
rebuilt from it later is a description of the log as it is now, not evidence of
the score that was computed then.

## What one record contains

An `EvaluationRecord` is **one constraint on one declared assessment unit**:

```text
run_id, evaluation_id, unit_id, unit_type, constraint_id/version/type
in_scope, evaluable, reason_code, violation or None
measurements   – structured, each with its own unit
parameters     – the constraint's parameters as configured
policies       – the missingness and pairing policy in force
witnesses      – events, absences or attributes supporting it
qualifications – named limitations
```

Measurements stay structured. A lag keeps `activation_timestamp`,
`response_timestamp` and `lag` (in the constraint's own unit); a balance keeps
`total_x`, `total_y` (each labelled with the attribute it aggregates) and
`relative_mismatch`. There is no single unexplained `raw_value` in this
contract — see *Interchange* below for the one place where a single value is
unavoidable, and how it is labelled.

### Reason codes

Five reason codes carry a violation:

| Reason | Meaning |
|---|---|
| `observed` | the check ran on observed data |
| `satisfied_vacuously` | nothing could violate the rule here (no regulated activity occurred) |
| `policy_violation_missing_activation` | the activation is missing and the policy says `violate` |
| `policy_violation_missing_response` | the response is missing and the policy says `violate` |
| `open_observation_window` | a `censor` policy scored the time elapsed so far |

The rest mark a check that was **not** evaluated, and such a record has
`violation=None` — never `0.0`:

`out_of_scope`, `skipped_missing_activation`, `skipped_missing_response`,
`skipped_missing_anchor`, `missing_attribute`, `ambiguous_match`,
`missing_source_identity`, `budget_truncated`.

A missing endpoint under `Lag(missing_b="violate")` is therefore an *evaluated,
policy-based* violation, not an unevaluable check; the same endpoint under
`missing_b="skip"` is unevaluable; under `missing_b="censor"` it is evaluated
as a **lower bound** at the horizon, carries the `lower_bound` qualification,
keeps `lag = None` (no completed duration was observed) and records
`elapsed_at_horizon` instead. The resolved horizon is in
`manifest.observation.resolved_horizons`.

The last two codes are part of the contract but are not produced by the
case-based evaluator today: it never matches ambiguously and never truncates an
evaluation. They exist so that an evaluator that can do either has a way to say
so, and they are validated like the rest.

## Witnesses, and the absence of one

`evidence="full"` materialises bounded witnesses; `evidence="summary"`
materialises them on request through `packet.witnesses(evaluation_id)`.

An event witness carries a **source event id** when the log was built with
`EventLog(event_id_col=...)`, and otherwise a reference of the form
`snapshot-local:row=12` with `identity = snapshot_local_row`. The second kind
identifies a row of one snapshot and nothing else; the packet also carries a
run-level `no_source_event_identity` qualification so that the limitation is
visible without inspecting individual witnesses. Two events with the same
activity and timestamp stay two witnesses: equal values do not establish
identity.

Absence has no event id. A rule violated because nothing happened is supported
by an `AbsenceSearch`:

```python
witness.search.activities        # what was searched for
witness.search.window_start/_end # the unit's observed span
witness.search.n_events_searched # how many events were examined
witness.search.completeness      # assumed_complete | open | unknown
witness.search.filters           # lifecycle filter, dedupe, timestamp policy, window
```

`completeness` is `assumed_complete` only when no filter could have removed
events; a `dedupe=True`, a lifecycle filter or `missing_timestamps="drop"`
makes it `unknown`, and an open censoring window makes it `open`.

## Bounds, and the difference between two kinds of truncation

Capture builds one Python object per check: about 40 µs per record in
`"summary"` mode and 120 µs in `"full"` mode. On a large log, capture the units
under review rather than all of them:

```python
V, S, details = wise.evaluate_detailed(log, norm)
packet = wise.evidence.capture_evidence(result, details=details, units=worst_ids, mode="full")
```

`witness_limit`, `max_records` and `units` all bound the packet, and
`packet.truncation` reports what was left out. A truncated *display* keeps the
exact measurement and the total witness count; a truncated *evaluation* — a
traversal that stopped early — is a different field (`evaluation_truncated`),
because its number is incomplete rather than merely abbreviated.

Each bound is also a **named qualification**, not only a field:

* `witness_limit` cutting a record's witnesses → `witnesses_truncated` on that
  record, beside the total the measurement still counts;
* `units=` or `max_records=` → a run-scope `records_truncated` carrying the
  captured and total counts. The scores are complete; the evidence rows and the
  coverage counts cover the kept records only;
* `max_records=` stopping the traversal before every constraint was reached →
  a run-scope `evaluation_truncated` naming the constraints with no row at all.
  The coverage counts of such a packet are not the run's coverage.

A bounded packet also cannot resolve the run's current population. Exporting one
with `to_interchange(packet, view=...)` and no explicit comparator yields a
comparator named `captured-subset:<view>:<run>` — the mean of the units the
capture kept — with a qualification giving that denominator, never the run's
`current-population:` identity.

## Coverage is not confidence

`packet.coverage` (or `wise.coverage_report(result)` without capture) reports
units, checks, in-scope, evaluated, unevaluable, out-of-scope and scored counts
per view, and leaves a share **undefined** when its denominator is zero.

Coverage says how much was observed. It is not a probability that the result is
correct, and nothing in this library multiplies a score or a priority index by
it. Confidence intervals, model probabilities, domain validation and human
approval remain different fields.

The typed diagnostics follow the same rule. Each carries its numerator,
denominator, unit of counting, scope, threshold, policy and interpretation:

```python
wise.typed_event_replication(log)        # timestamp multiplicity inside a case
wise.typed_cross_case_replication(log, "purchasing_document")
wise.typed_right_censored(log, "Clear Invoice", window="60D")
```

`typed_cross_case_replication` states which key established identity: with an
`event_id_col` the policy is `cross_case_source_event_id_match`; without one it
is `cross_case_activity_timestamp_key_match`, and the result carries a
qualification saying that it is an upper bound on real replication. The
underlying numeric helpers are unchanged and still available.

## The run record

`result.manifest` is a `RunManifest`. A norm fingerprint identifies a
configuration, not a run:

```python
m = result.manifest
m.mode, m.mode_source        # the mode actually used, and whether the call overrode the norm
m.views, m.norm_fingerprint
m.input                      # shapes, columns, event-id column, snapshot fingerprint
m.preprocessing              # every EventLog option, including the tie-order policy
m.observation                # window source, resolved horizons, policy id
m.derived_recipes, m.calibrations
m.environment                # library, Python, NumPy, pandas, platform
m.config_fingerprint()       # deterministic over the configuration
m.run_id, m.created_at       # per execution, and excluded from the fingerprint
```

Three deliberate omissions:

* **Git is never read.** `environment.git_commit` is `None` unless a caller
  supplies it with `manifest.with_commit(...)`; an installed wheel has no
  repository to ask.
* **The data is never hashed automatically.** `wise.evidence.fingerprint_events(log)`
  returns `(digest, canonicalisation)` and `manifest.with_data_digest(...)`
  records both, because a digest without its canonicalisation means nothing.
  The canonicalisation preserves stored row order on purpose: with no
  `order_col`, timestamp ties are broken by input row order, so sorting the data
  to obtain an order-insensitive hash would erase a real difference.
* **The input identity is recorded at the level the run actually verified,
  table by table.** A run with capture content-hashes the case table *and* the
  event table (`snapshot_scope: "events"`), because a `Balance` or a `Metric`
  reads an amount and a changed amount is a changed input. A run without capture
  records the structural identity only (`snapshot_scope: "structure"`), rather
  than paying for a hash on every score call.
  `preprocessing.content_hashed` is a **map per table**
  (`{"cases": true, "events": true}`), never a single `true`: "content hashed"
  without naming the table would claim more than the run verified. Whenever some
  table is not covered, the packet carries a run-scope
  `snapshot_not_content_hashed` qualification naming exactly which.

A score run knows nothing about a backlog. `manifest.finalize(grouping=...,
view=..., gamma=...)` returns a **new** manifest at stage `"backlog"` with the
priority configuration; only that one reports `is_complete_backlog_run`.

`manifest.to_json()` and `packet.to_json()` write UTF-8 with explicit nulls,
normalised ISO-8601 timestamps and `allow_nan=False`. A non-finite number or an
unsupported type raises `EvidenceError` instead of being stringified.

## Reading a packet back

Export is only half of "evaluated and unevaluated checks remain distinguishable
after export **and** reload":

```python
path.write_text(result.evidence.to_json(), encoding="utf-8")
packet = wise.load_evidence(path)                      # or EvidencePacket.from_dict(json.loads(...))
```

Every record is rebuilt through the same invariants it was written under, so a
file claiming a violation for a check it also calls unevaluated is refused on
read. A censored record comes back with `open_observation_window`, its
`lower_bound` qualification and its `lower_bound: true` measurement; an
out-of-scope check comes back out of scope with a null violation.

What does **not** come back is the log. A restored packet carries neither the
snapshot nor the norm, says so (`packet.restored`, and a `witness_access` block
in the export), and refuses `packet.witnesses(...)` with
`EvidenceUnavailableError` rather than returning an empty tuple. Witnesses
captured with `evidence="full"` travel inside the records and are still there.

## Stale evidence

Lazy witness access is protected by the snapshot the packet was captured from:

```python
packet.witnesses(evaluation_id)      # fine
log.add_case_attribute("x", values)  # the input changed
packet.witnesses(evaluation_id)      # StaleEvidenceError
```

`EventLog.snapshot(scope=...)` chooses how much is verified — `"structure"`
(shapes, columns, options), `"cases"` (also the case table, which is what
`derive` and `add_case_attribute` change) or `"events"` (also **every column**
of the event table, which is what capture uses, so a relabelled activity, a
shifted timestamp and a changed amount are all refused).
`snapshot.content_hashed_tables` says which tables a snapshot covers and
`snapshot.content_hashed` is `True` only when every one of them is.
`EventLog.snapshot(freeze=True)` copies the witness columns instead,
so the packet stays usable whatever happens to the log afterwards:

```python
V, S, details = wise.evaluate_detailed(log, norm)
packet = wise.evidence.capture_evidence(result, details=details, snapshot=log.snapshot(freeze=True))
```

The guarantee is paid for at each check. On the 1.6M-event BPIC'19 log an
`"events"` snapshot costs about 1.1 s against 0.11 s for `"cases"`, and
`check_fresh()` recomputes it on every batch of lazy witness lookups. A run
without capture is untouched (it takes a `"structure"` snapshot, well under a
millisecond); a long-lived reader that materialises witnesses repeatedly should
capture with `freeze=True`, which copies the witness columns once and never
re-hashes.

Note that `score(derive=True)` — the default — computes the norm's derived
attributes on the caller's log and therefore modifies it. That is the
historical behaviour; the manifest records it as a note and the packet as an
`input_mutated_by_derive` qualification.

## Fitted references

`derive.py`'s `quantile_scale` recipe divides an attribute by a quantile of the
log it is applied to. That default is unchanged. The explicit alternative
freezes the reference:

```python
from wise.evidence import fit_calibration
calibration = fit_calibration(january_log, recipe)   # divisor fitted once
result = wise.score(february_log, norm, calibrations=[calibration])
```

The record carries the recipe, its fingerprint, the quantile, the fitted
divisor, the fitting population and a `calibration_id`. Applying it never
refits. A changed recipe is a different identity and is refused rather than
silently reused; a re-fit on another population produces a different record.
`score` records the applied calibrations in the manifest, so a frozen run and a
recomputed run have different configuration fingerprints.

## Interchange

`wise.evidence.to_interchange(packet, view=...)` exports the shape of the
roadmap's proposed `EvidencePacket` contract, whose schema ships with the
package (`wise.evidence.interchange_schema()`) and is checked in the tests.

That shape compresses each check to a single `raw_value` and `raw_unit`. This
library does not: the export selects the primary measurement (the duration of a
lag, the relative mismatch of a balance) and the native record keeps all of
them. `commit` is exported as `"unknown"` when nobody supplied one, and
`data_hash` falls back to the snapshot fingerprint, labelled
`snapshot-fingerprint:…`. It is an interchange view, not a new supported WISE
serialisation format; `packet.to_json()` is the native one.

## Command line

The run record and the evidence packet are opt-in sidecars. The default stdout
CSV of `wise score` is byte-for-byte what it always was, and every progress
message goes to stderr:

```bash
wise score norm.json log.csv --case case --activity activity --timestamp time \
    --by company --view Finance \
    --manifest-out run.json --evidence-out evidence.json --evidence-frame-out evidence.csv
```

`examples/evidence_review.py` runs the whole of this page on the bundled
example and needs no data.
