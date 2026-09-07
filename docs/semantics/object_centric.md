# Object-centric data and bounded assessment units

*Status: experimental, opt-in, added on the extension branch. Nothing on this
page is reachable from `import wise`; `wise.oc` is an explicit import, and
importing it pulls in no optional package and no database driver. Every
existing number, default and signature is untouched.*

A case log answers questions about a case. A procure-to-pay reality does not
have one: an invoice belongs to a purchase order that has a hundred items, is
matched against two goods receipts and is settled by a payment that also
settles three other invoices. Flattening that into cases has to pick one
identity and duplicate or discard the rest. `wise.oc` keeps the identities.

This page documents what stages **O01** (the log and its adapters) and **O02**
(typed bounded units) actually provide. The relational checks, the quantity
accounting and the projection comparison are later stages and are not here.

## The log

```python
from wise import oc

log = oc.read_ocel2_json("p2p.json")
log.summary()["n_events"]
log.objects_of("event:52")                   # which objects, in which role
log.related("inv1", qualifier="belongs to")  # directed, qualified
log.history("inv1", "amount")                # the whole attribute history
```

`OCEventLog` is a versioned (`wise-oc/1`), immutable, indexed set of tables:

```text
events             canonical event id, activity, timestamp, attributes
objects            canonical object id, object type
e2o                event id, object id, qualifier          (the role played)
o2o                source id, target id, qualifier, optional validity interval
attribute_history  object id, attribute, effective timestamp, value
source             interchange, spec, reader, digest, precision, semantics
validation         every repair, drop and refusal, by name and count
```

No graph database, no query language: bounded lookups on prebuilt indices.

### Four rules the type enforces

**Identity is never manufactured.** Two rows claiming one event id with
different content raise `OCValidationError`; the later one does not win, and
they are not merged. Symmetrically, two *different* events that share an
activity and a timestamp are two events — `events_of` returns both, ordered by
`(timestamp, event_id)` so the order is stable without the timestamp becoming
an identity.

**Identical rows may be deduplicated, with a report.** A relation table that
repeats one triple loses the repeat and gains a `ValidationIssue` of severity
`repaired`. `deduplicate=False` turns even that into an error for a caller who
needs the source to be exact.

**A link that points nowhere is not silently dropped.** Dangling relations
raise under the default `on_dangling="error"`. `on_dangling="drop"` removes
them and records the admission in the log's own validation report and in
`source.options` — the log carries the fact that it is not the whole file.

**An attribute is read at a declared instant.**

```python
reading = log.value_at("inv1", "owner", "2024-01-10T00:00:00Z")
reading.value            # the value in force then
reading.found            # False when the history has none in force yet
reading.effective_from   # which change supplied it
reading.later_changes    # how many changes came after — a hint, never a value
```

There is no "first non-null" policy and there never will be. Before the first
change the reading is `found=False` with an `attribute_not_set_at_time`
qualification: missing is not empty, and it is not the next value either.
`AttributePolicy.LATEST_KNOWN` exists for the reviewer who genuinely wants
today's owner, and it attaches
`attribute_read_outside_evaluation_time` when that value comes from after the
instant being assessed.

### Observed precision

`source.precision` says what the timestamps in the data can support — the
coarsest grid every timestamp sits on, measured against the UTC epoch. The
published procure-to-pay log reads as `minute`. That is a fact about the data,
not a claim by the source system, and it is what says whether a check that
argues about seconds is answerable at all.

## The interchanges, and what each can carry

`oc.interchange_support()` is the machine-readable declaration. Three entries,
and each states what it *cannot* carry:

| Interchange | Read | Write | Not carried |
|---|---|---|---|
| `ocel2-json` (OCEL 2.0 JSON) | yes | yes | relation validity intervals; source metadata |
| `ocel2-sqlite` (OCEL 2.0 SQLite) | yes | **no** | original attribute names; declared OCEL types |
| `wise-oc-tables` (six pandas tables) | yes | yes | the validation report and the source metadata |

The pinned interchange is **OCEL 2.0 JSON**. `tests/fixtures/ocel/p2p_mini.json`
is the pin: a complete hand-written document that the writer reproduces key
for key and list order for list order. Nothing is claimed for OCEL 1.0, for
XML, for a later revision, or for a CSV extract that happens to be readable —
reading generic JSON is not support for a standard.

The **SQLite** serialisation is read-only for a stated reason: its per-type
tables name attributes by sanitised column identifier — `Invoice Receipt
(MSEG-WEAHR)` becomes `InvoiceReceiptMSEGWEAHR` — and carry no mapping back.
The reader records that loss in the validation report rather than pretending
the names survived. On the published log the two adapters agree exactly on
events, objects, both relation tables and every attribute timestamp, and
differ only in those names; the test asserts both halves of that.

The **native tables** round-trip and the **OCEL export** are deliberately
different acceptance tests. One shows nothing is lost inside the library; the
other shows what leaves it conforms to the standard.

Export refuses rather than loses:

```python
document, report = oc.ocel2_json_document(log)     # raises if anything would be dropped
document, report = oc.ocel2_json_document(log, allow_loss=True)
report.lossless, [i.message for i in report.issues]
```

Values are preserved exactly as the source wrote them. OCEL declares an
attribute's type; the reader keeps the declaration in
`object_type_attributes` and does **not** apply it, because a silent coercion
is an unrecorded edit of the data.

### What the published p2p log actually contains

Reading `ocel2-p2p.json` with the default policy **fails**, and that is the
point: 2 028 of its 20 402 object-to-object relations point at an
`invoice receipt` object the file never declares, and 295 of its 78 508
attribute history rows are identical repeats. With `on_dangling="drop"` the log
reads as 14 671 events, 9 543 objects, 35 927 E2O and 18 374 O2O relations,
and says so in its report. A reader that silently accepted the file would be
computing on 2 028 relations to objects that do not exist.

## Relation time

OCEL 2.0 object-to-object relations are **atemporal**: the standard records
that an invoice belongs to a purchase order, not the interval during which it
did. `source.relation_time_semantics` therefore stays `"atemporal"` for such a
source, an interval on a relation from such a source is refused, and any unit
built at an evaluation instant carries `relation_validity_unknown`.

A source that genuinely supplies intervals declares
`relation_time_semantics="interval"`, and then the intervals are *used* —
traversal follows only relations in force at the evaluation instant, on the
half-open convention `[valid_from, valid_to)`. Using a recorded fact is not
inference; inventing one is.

## Assessment units

An object-centric check needs to be told what it is about. `UnitSpec` says it:

```python
spec = oc.UnitSpec(
    unit_type="invoice_review",
    anchor_type="invoice",
    roles=(
        oc.RolePath("order", (oc.PathStep("belongs to", target_type="purchase_order"),)),
        oc.RolePath("items", (
            oc.PathStep("belongs to", target_type="purchase_order"),
            oc.PathStep("has item", target_type="item"),
        )),
    ),
    scope=oc.UnitScope(roles=("order",)),
    limits=oc.TraversalLimits(max_depth=3, max_fan_out=25, max_bindings=100, max_visited=1000),
)
units = oc.build_units(log, spec, at="2024-06-30T00:00:00Z")
```

A path is an ordered tuple of typed, directed, qualified steps — not a query
language, and there is no wildcard hop. `direction` is not symmetric: an
invoice belonging to an order and an order owning an invoice are different
statements, and a traversal that mixes them is how a bounded context becomes
the whole business. A step may be `transitive` (follow this same relation while
it keeps reaching the same type), and even then `max_depth` bounds it.

**A name that does not occur in the log is refused.** `build_units` checks the
anchor type, every step's type and qualifier, and every scope activity against
the log, and raises unless `strict=False`. A path with a mistyped qualifier
selects nothing, and "selected nothing" is indistinguishable from "there is
nothing" once the result is a number.

### The bounds, and what happens at each

| Limit | What it bounds | On reaching it |
|---|---|---|
| `max_fan_out` | neighbours one object contributes at one step | role flagged `fan_out_limited` |
| `max_bindings` | objects one role may hold | role flagged `binding_limited` |
| `max_depth` | hops a transitive step may run | role flagged `depth_limited` |
| `max_visited` | objects one unit may touch in total | unit flagged `visit_limited` |

The unit is still returned — a bounded partial context is often exactly what a
reviewer wants — but:

```python
unit.complete                # False
unit.truncation.reasons      # ('max_fan_out reached for role(s) [...]',)
unit.truncation.seen, unit.truncation.kept
[q.code.value for q in unit.qualifications]
# ['context_truncated', 'fan_out_limit_reached', 'relation_validity_unknown']
unit.require_complete()      # OCUnitError, rather than a flag someone forgets
```

Cycles terminate: each object is visited once per role, and the anchor is
never bound to its own role — coming back to it is a cycle, not a related
object. `oc.context_report(units)` gives the number a reviewer needs before
reading any aggregate: a backlog over 900 complete and 100 truncated contexts
is not a backlog over 1 000 contexts.

### Witnesses

Every binding carries the subgraph that justifies it:

```python
unit.witnesses_for("items")[0].describe()
# 'inv1 -[belongs to]-> po1 po1 -[has item]-> it1'
```

### Events in scope

`UnitScope` says whose events count: the anchor's, plus those of the named
roles, optionally narrowed by relation qualifier, activity and time window. It
is a different thing from `wise.evidence.manifest.ObservationScope`, which is
the *run's* window over a case log.

## One numerical kernel

`src/wise/_aggregation.py` holds the single applicability-aware scoring
formula. `wise.scoring.score` calls it for cases and `wise.oc.score_units`
calls it for units; there is no second implementation to drift. It is pure
array code — evaluated violations, raw weights, a `LayerAssignment`, row
identities and an explicit mode in; scores, layer contributions and per-row
effective weights out — with the same missing-value conventions as before:
`NaN` means *not evaluated*, never *satisfied*, and a row with no positively
weighted applicable check is unscored rather than zero.

`scoring._effective_weights(M, w, norm, mode)` keeps its historical signature
and delegates. On the BPI Challenge 2019 log (251 734 cases, 29 constraints)
the extracted kernel reproduces the reviewed base's arithmetic with a maximum
absolute difference of `0.0` in both modes. The kernel is as layout-sensitive
as the code it replaces — row sums of a C- and an F-ordered copy of one matrix
can differ in the last bit — and `score` hands it the same F-ordered array it
always did.

## The three native check families

They live in `wise.oc.constraints`, in their own catalogue
(`OBJECT_CHECKS`). They are **not** members of `wise.constraints` and cannot be
loaded through `Norm.from_dict`: a check whose parameters name roles and
qualifiers is not a schema-2 case constraint.

### RelatedObjectCardinality

Counts the **distinct object identities** a role binds, against a declared
minimum, maximum or both, graded over `width`.

* A shortfall against `minimum` is an *absence claim*. Unless `completeness`
  is `assumed_complete`, the check is **not evaluable**: the reason is
  `unverified_absence` and no violation is produced. Declaring completeness
  turns the shortfall into a violation and attaches the assumption to it.
* An excess over `maximum` needs no such assumption — the objects that prove
  it are bound.
* Inside a **truncated** role the count is a lower bound
  (`count_is_a_lower_bound`). A shortfall there is `budget_truncated` and
  carries no number; an excess still stands.

### CrossObjectLag

The time from an activation event to its matching response, where the two are
selected through *different objects of the unit*. Five policies are declared
fields, because each of them silently changes the answer: `activation` /
`response` selectors (role, activities, first or last occurrence), `match`
(`first_after` or `first_overall`), `clock`, `equal_time` (`counts` or
`excluded` — what a response bearing the activation's own timestamp means) and
`missing_activation` / `missing_response` (`violate`, `skip`, and for the
response `censor` against a `horizon`).

When two candidate events tie on the matching timestamp the result is a typed
`ambiguous_match` with both events as witnesses and no violation. There is no
silent nearest-timestamp rule. `on_ambiguous="tie_break"` resolves by the
smaller event id **and says so** (`ambiguous_match_resolved`);
`on_ambiguous="error"` raises.

### RelationalBalance

Two matched sets of amounts, read at the unit's evaluation instant, turned
into canonical `QuantityRecord`s and allocated through `wise.oc.accounting`.
That is what enforces "no amount consumed twice": a configuration that binds
one object into both sides is refused, not quietly counted twice. Amounts in
different currencies are `incompatible_units` unless
`unit_policy="declared_rates"` supplies a rate table and a target unit, and a
conversion is then marked `unit_conversion_applied`. Nothing is netted or
converted implicitly.

## Quantity accounting

`wise.oc.accounting` gives every additive value a canonical identity —
`object:attribute@effective_instant`, so a changed amount is a *different*
record rather than a correction — and makes every use of it an `Allocation`:
a named group taking a declared, non-negative share.

`allocate` enforces one rule: the shares of one record, over all groups, never
exceed one. Negative shares, one group consuming one record twice, and
over-allocation are refused. An **incomplete** allocation is not an error: its
residual is reported per record and in total (`allocation_incomplete`), and
`require_complete=True` turns it into a refusal for a caller who cannot use a
partial answer.

The contract is about additive quantities, not obligations. Three checks
resting on one payment are three checks. `AllocationReport.coverage()` reports
how much *distinct* evidence each group rests on, `overlap()` names the records
two groups share, and `combined()` returns three numbers side by side: the
conserved `allocated` sum of declared shares, the `evidence_value` of the
distinct records once, and `sum_if_each_group_claimed_all` — with
`overlapping_groups_not_summed` attached whenever the last two differ.

## The typed object result

`score_units(log, catalogue, units)` returns an `OCScoreResult`: a **different
type** from `ScoreResult`, whose `log` attribute promises a case-based
`EventLog` and keeps promising it. The object result carries its log as
`oc_log`, one `EvaluationRecord` per check per unit (out-of-scope pairs
included, recorded as out of scope rather than omitted), the run and content
fingerprints, and `frame(view)` — the wide per-unit table the existing
prioritisation path already reads.

Ranking stays inside one unit type by default. A result holding more than one
refuses to produce a single table unless a `unit_type` is chosen or
`allow_mixed=True` is passed: a count of invoices and a count of item
obligations are not one interchangeable volume. `object_backlog` checks the
typed grouping and the exposure column before handing anything to
`wise.prioritize`.

## Native versus projected

`wise.evaluation.ocel` implements the two honest projections and measures what
each costs. `naive_projection` copies a shared event into every case it
touches and drops its identity; `identity_preserving_projection` copies the
rows too but keeps the source event id and flags the copies. Both report what
they duplicated and what they could not carry.

On the fixture in `tests/extensions/test_oc_representation_comparison.py` —
three invoices, five obligations, fifteen cells of manually specified truth:

| | false violations | missed obligations | unsupported claims | exact | duplicated amount | witness recall |
|---|---|---|---|---|---|---|
| native | 0 | 0 | 0 | 15 | 0.00 | 1.00 |
| naive projection | 6 | 2 | 4 | 7 | 200.00 | 0.00 |
| identity-preserving projection | 6 | 2 | 4 | 7 | 200.00 | 1.00 |

The two projections make **the same** five mistakes: preserving event identity
does not restore a relation. What it does restore is the ability to name a
witness by its source event id and to detect the duplication. The fixture is
small and simulated: this is a representational difference, not evidence of
industrial benefit.

## What this stage does not do

* No discovery, no prediction, no causal claim.
* No object check reachable through `Norm.from_dict`, and no `OCEventLog`
  inside `ScoreResult.log`.
* No fourth check family, and no process-query language: the three templates
  above are the whole catalogue.
* No performance work on `build_units`, which is still O(anchors x path) with
  no shared frontier between anchors.
