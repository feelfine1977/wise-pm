# wise

`wise` implements **WISE (Weighted Insights for Evaluating Efficiency)**, a
norm-based, slice-first method for prioritising process deviations from
event logs:

> U. Jessen, D. Fahland, F. Zerbato. *WISE: Actionable Norm-Based Scoring for
> Process Mining.* Eindhoven University of Technology.

The method works in three phases:

1. **Norm.** Expected execution is written as a compact catalogue of
   machine-checkable constraints, grouped into business-facing *layers* and
   read through stakeholder *views* (weight vectors over the same constraints).
2. **Scoring.** Every case receives a bounded violation `ν_c(σ) ∈ [0, 1]` per
   applicable constraint and a view-specific score `S^(p)(σ) ∈ [0, 1]` whose
   penalty decomposes exactly into layer contributions.
3. **Prioritisation.** Cases are aggregated into ownership-aligned slices and
   ranked by the Priority Index `PI = n_s · (μ̄ − μ_s)_+`, optionally stabilised
   for small slices and weighted by business exposure, with drill-down to
   layers, constraints and cases.

The package depends only on `numpy` and `pandas`. Scoring is vectorised: the
BPI Challenge 2019 log (1.6 million events, 251,734 cases) loads in a few
seconds and scores in about two.

## Installation

```bash
pip install wise-pm
```

The distribution is called `wise-pm` because the name `wise` on PyPI belongs
to an unrelated package from 2013; the import name is `wise`. Extras:
`wise-pm[pm4py]` for XES import, `wise-pm[stats]` for Spearman and Kendall
correlations in `view_agreement`, and `wise-pm[llm]` for applications that
want a richer HTTP client or a full JSON-Schema implementation beside the
library — `wise.llm` itself needs neither, and the base install can already
drive a local model and validate a draft.

From a checkout:

```bash
pip install -e ".[dev]"
pytest
```

Python 3.10 or later is required.

## Quick start

```python
import pandas as pd
import wise

# Event log: one row per event. Column names default to the pm4py conventions
# (case:concept:name, concept:name, time:timestamp); pass others explicitly.
events = pd.read_csv("log.csv")
log = wise.EventLog(
    events,
    case_col="case",
    activity_col="activity",
    timestamp_col="time",
    case_attributes=["flow_type", "company", "spend_area", "vendor"],
    exposure_col="net_worth",  # optional: exposure-weighted priorities
)

# Norm N = (C, Λ, lay) with views; built in code or loaded from a JSON file
norm = wise.Norm(
    constraints=[
        wise.NormConstraint("c1", "completeness", wise.Presence("Record Invoice Receipt")),
        wise.NormConstraint(
            "c2", "lead_times", wise.Lag("Record Goods Receipt", "Record Invoice Receipt", delta=10, width=20, unit="D")
        ),
        wise.NormConstraint(
            "c3",
            "match",
            wise.Balance("amount", "Record Invoice Receipt", "amount", "Record Goods Receipt", tau=0.05, width=0.20),
        ),
        wise.NormConstraint("c5", "handling", wise.Singularity("Record Goods Receipt", k=2, K=3)),
        wise.NormConstraint(
            "c6", "exceptions", wise.Exclusion("Cancel Invoice Receipt"), applicability={"flow_type": ["DF1", "DF2"]}
        ),
    ],
    layers=[wise.Layer(name) for name in ["completeness", "lead_times", "match", "handling", "exceptions"]],
    views=[
        wise.View("Finance", constraint_weights={"c1": 0.20, "c2": 0.45, "c3": 0.20, "c5": 0.05, "c6": 0.10}),
        wise.View(
            "Logistics",
            layer_weights={"completeness": 0.25, "lead_times": 0.15, "match": 0.05, "handling": 0.45, "exceptions": 0.10},
        ),
    ],
)
norm.dump("norm_v1.json")  # versioned artefact; wise.Norm.load(...) reads it back

# Score
result = wise.score(log, norm)
result.violations  # cases × constraints, ν_c(σ); NaN = not applicable
result.scores  # cases × views, S^(p)(σ); NaN = unscored
result.contributions["Finance"]  # cases × layers, Δ_λ (sums to 1 − S)

# Prioritise and explain
backlog = wise.prioritize(result, by=["company", "spend_area"], view="Finance", gamma=20)
wise.layer_drivers(result, by=["company", "spend_area"], view="Finance")
wise.constraint_drivers(result, "Finance", {"company": "A", "spend_area": "Packaging"})
wise.hotspot_table(backlog, drivers=wise.layer_drivers(result, ["company", "spend_area"], view="Finance"))
wise.concentration(backlog)  # share of slices carrying 80 % / 95 % of the priority mass
wise.view_agreement(result, by=["company", "spend_area"], k=20)
result.worst_cases("Finance", where={"company": "A"})
```

`python examples/quickstart.py` runs the method on the paper's running
purchase-to-pay example (Tables V and VI); `examples/bpic19_evaluation.py`
runs the BPI Challenge 2019 evaluation (Section V) on the challenge CSV with
the norm in `examples/bpic19_norm.json`. That norm file records the settings
of the evaluation (lags not applicable when an endpoint is missing and
measured to the first response in the case), which differ from the
library defaults described below.

## Concepts and API

| Concept (paper Section IV) | Symbol | API |
|---|---|---|
| Event log, cases, attributes, exposure | `L, Σ, att_β(σ), exp(σ)` | `EventLog` (`.cases`, `.count`, `.first_ts`, `.first_after`, `.total`) |
| Threshold–saturation rule | `sat(z; ϑ, W)` | `sat` |
| Presence `(pres, a, m)` | `1 − min(cnt/m, 1)` | `Presence(activity, m=1)` |
| Lag `(lag, a, b, δ, Δ)` | `sat(t_b − t_a; δ, Δ)` | `Lag(a, b, delta, width, unit)` |
| Balance `(bal, α_x, A_x, α_y, A_y, τ, Γ)` | `sat(d; τ, Γ)` | `Balance(attr_x, activities_x, attr_y, activities_y, tau, width)` |
| Singularity `(sing, a, k, K)` | `min(max(0, cnt − k)/K, 1)` | `Singularity(activity, k, K)` |
| Exclusion `(excl, a)` | `1[cnt > 0]` | `Exclusion(activity)` |
| Norm `N = (C, Λ, lay)` | | `Norm(constraints, layers, views)`, `NormConstraint(id, layer, constraint, weight, applicability)` |
| Applicability `C_app(σ)` | | `applicability={"attr": [values]}` or rule form (`all`/`any`/`not`, `has`/`lacks`) |
| View, raw weights `w_c^(p)` | | `View(name, constraint_weights=…)` |
| Two-stage weights `w_c = a_λ · b̂_c` | | `View(name, layer_weights=…)` with `NormConstraint.weight` as `b_c` |
| Case score `S^(p)(σ)` | | `score(log, norm)` → `ScoreResult.scores` |
| Layer contributions `Δ_λ^(p)(σ)` | | `ScoreResult.contributions[view]`, `ScoreResult.penalties(view)` |
| Slice mean, global mean, PI, stabilised PI, exposure | | `prioritize(result, by, view, gamma, volume)` |
| Drill-down | | `layer_drivers`, `constraint_drivers`, `penalty_mass`, `ScoreResult.worst_cases`, `ScoreResult.trace` |
| Backlog concentration, view agreement, hotspot typology | | `pareto`, `concentration`, `top_k_overlap`, `view_agreement`, `hotspot_table` |
| Validation diagnostics (Section V) | | `right_censored`, `left_truncated`, `event_replication`, `cross_case_replication`, `gap_retained`, `validation_table` |
| Evidence, run records, fitted references (opt-in, experimental) | | `score(..., evidence="summary")`, `ScoreResult.manifest`, `ScoreResult.evidence_frame`, `wise.evidence` |

Two constraint types extend the catalogue: `Precedence(a, b)` requires that
no `b` occurs before the first `a`, and `Metric(attribute, threshold, width)`
applies the saturation rule to a numeric case attribute. Case attributes can
be computed from declarative recipes stored in the norm
(`derived_attributes`, see `wise.derive`), so that engineered signals such as
manual-touch counts are part of the versioned artefact.

### Conventions

- Slice-level results are DataFrames indexed by the slice keys; use
  `as_index=False` for the keys as columns. The parameters of a backlog are
  stored in `DataFrame.attrs`.
- `NaN` in `ScoreResult.violations` means the constraint was not evaluated
  for that case: out of scope (`ScoreResult.in_scope`) or not evaluable
  (e.g. a lag endpoint skipped by `Lag(missing_b="skip")`, a missing metric
  attribute). A case with no evaluated, positively weighted constraint has a
  `NaN` score and is excluded from every aggregate.
- Activity parameters accept one label or a list of labels; a list is one
  merged activity.
- `Lag` measures from the first `a` to the first `b` at or after it; undefined
  endpoints are full violations. `missing_a`, `missing_b`, `activation` and
  `response` change these rules per constraint.
- Scoring uses the norm's `scoring_mode`: `"layer_balanced"` (default)
  averages the applicable constraints within each layer and then the
  applicable layers with the layer weights; `"flat"` renormalises the raw
  constraint weights over the applicable constraints of each case. The two
  coincide when every constraint of a layer applies.
- `EventLog` raises on null case ids, negative exposure and, by default, on
  missing timestamps; `EventLog.validate()` summarises data-quality signals
  and `Norm.check(log)` verifies that the norm's activities and attributes
  exist in the log.

## Evidence and run records (experimental)

Opt-in and off by default: `score(log, norm)` behaves exactly as before. With
capture switched on, every number resolves to the observation behind it and
every check that could not be evaluated carries a reason.

```python
result = wise.score(log, norm, evidence="summary")  # identical scores, plus evidence
result.manifest.mode  # the mode actually used, not the norm default
result.evidence_frame("Finance")  # one row per constraint x case
result.evidence.coverage  # in scope / evaluated / scored / excluded
```

* An evaluation record separates **in scope**, **evaluable** and **violated**,
  and keeps raw measurements with their units — a lag keeps both endpoints and
  the duration, a balance both totals and the relative mismatch.
* A check that was not evaluated has no violation, never a zero.
* A rule violated because nothing happened is supported by a declared absence
  search (unit, window, filters, completeness assumption), never by an invented
  event id. Without `event_id_col` an event witness is explicitly a
  snapshot-local row reference.
* `RunManifest` records the mode actually used, the views, the input identity,
  the preparation options and tie-order policy, the observation policy and
  resolved horizons, the derived recipes and frozen calibrations, and the
  library and environment versions. It reads no Git state and hashes no data
  unless asked (`wise.evidence.fingerprint_events`).
* Witness access is guarded: if the log changed after capture, materialising a
  witness raises `StaleEvidenceError` instead of answering from the new data.
  A capturing run fingerprints the case table **and every column of the event
  table**, so a relabelled activity, a shifted timestamp and a changed amount
  are all refused; `manifest.preprocessing["content_hashed"]` says which tables
  are covered, table by table.
* Evidence survives a round trip: `wise.load_evidence(path)` (or
  `EvidencePacket.from_dict`) reads an exported packet back under the same
  invariants it was written under, and a restored packet declares that it no
  longer has the log rather than answering witness queries with an empty list.
* Every bound is a named qualification, not just a field: `witnesses_truncated`,
  `records_truncated` and `evaluation_truncated`, each with its counts.
* `wise.evidence.fit_calibration` freezes a fitted reference so that
  `quantile_scale` is not silently recomputed on later data; the historical
  recalculation stays the default.
* Coverage is not confidence, and no score is multiplied by it.

`docs/semantics/evidence.md` is the reference; `python examples/evidence_review.py`
is a runnable tour on the bundled example.

## One comparator, one explanation (experimental)

Opt-in and off by default. `prioritize` has always accepted a scalar
comparator while `layer_drivers` always contrasted against the current
population; together, unlabelled, they described two different comparisons of
one backlog. `wise.explain` resolves the comparator once and uses it in both.

```python
from wise.explain import BaselineSpec

frozen = BaselineSpec.from_result(last_quarter, "Finance")  # score and layer profile
backlog = wise.prioritize(result, "vendor", view="Finance", gamma=20, baseline_spec=frozen)
packet = wise.explain_priority(result, "vendor", view="Finance", gamma=20, baseline_spec=frozen)
print(wise.render_explanation(packet, "markdown"))
```

* `BaselineSpec` carries the comparator's identity, its reference score, an
  optional **complete** layer profile, and the view, normalisation, unit type,
  norm and calibration references it is valid under. It validates
  `reference_score == 1 - sum(layer penalties)` within a documented tolerance.
* Compatibility is about assessment semantics, not population identity: a
  historical comparator is *expected* to come from another population, while a
  different view, normalisation or layer partition is refused by name. An
  unknown layer is never aligned with zero, and a changed norm needs an
  explicit reviewed mapping.
* Layer deltas sum to the same signed score gap the priority used; the signed
  components sum to the stabilised index. Negative offsets are kept, clipping
  is reported rather than hidden, and no share is formed by dividing by a zero
  net gap.
* A **scalar-only** target still ranks, but the layer attribution is reported
  as `reference_profile_unavailable` instead of being invented from the current
  population.
* `ExplanationPacket` keeps four kinds of claim apart — what was observed, why
  it has this priority, how another approved view reads, and what the evidence
  covers — with a denominator for every number and limitations that survive
  whoever writes the prose. Rendering is deterministic and escapes untrusted
  labels.
* `prioritize(..., baseline_spec=...)` and `layer_drivers(..., baseline_spec=...)`
  are keyword-only additions; the historical calls, columns, ordering, values
  and attributes are unchanged, and `global_mean` keeps its name while holding
  whatever reference was used.

```bash
wise score norm.json log.csv --by vendor --view Finance \
    --baseline-file target.json --explain-out explanation.json
wise explain explanation.json --format markdown --out report.md
```

`docs/semantics/explanation.md` is the reference;
`python examples/explanation_review.py` is a runnable tour on the bundled
example.

## Object-centric data and bounded units (experimental)

Opt-in and off by default. `wise.oc` is an **explicit import**: `import wise`
does not load it, and importing it needs no optional package and no database
driver. It keeps the identities a flattened log has to pick between.

```python
from wise import oc

log = oc.read_ocel2_json("p2p.json", on_dangling="drop")  # OCEL 2.0 JSON
log.validation.to_dict()  # every repair, by name
log.value_at("inv1", "owner", "2024-01-10T00:00:00Z")  # at that instant, not the first value

spec = oc.UnitSpec(
    unit_type="invoice_review",
    anchor_type="invoice",
    roles=(oc.RolePath("order", (oc.PathStep("belongs to", target_type="purchase_order"),)),),
    limits=oc.TraversalLimits(max_fan_out=25, max_bindings=100),
)
units = oc.build_units(log, spec, at="2024-06-30T00:00:00Z")
units[0].role("order"), units[0].complete
```

* `OCEventLog` is a versioned (`wise-oc/1`), immutable, indexed set of tables:
  event and object identities, qualified event-to-object and object-to-object
  relations, object attribute histories, and the source, extraction metadata
  and observed timestamp precision behind them. No graph database.
* Identical repeated rows are deduplicated **and reported**; two rows claiming
  one identity with different content raise; dangling relations raise unless
  dropping them is explicitly requested, and then the drop is recorded. Two
  different events that share an activity and a timestamp both survive.
* An attribute is read at a declared evaluation time. There is no first
  non-null fallback: before the first change the reading says `found=False`
  with a named qualification, because missing is not empty and not the value
  that arrives later.
* `oc.interchange_support()` declares what is supported and what each format
  cannot carry: OCEL 2.0 JSON read **and** write, OCEL 2.0 SQLite read only
  (its sanitised attribute names cannot be recovered), and the native pandas
  tables. Export refuses rather than silently loses.
* A `UnitSpec` declares an anchor type, named roles as typed/directed/qualified
  paths, an observation scope and strict depth, fan-out, binding and visit
  limits. There is no process-query language and no wildcard hop.
* A truncated context is never presented as a complete one: `unit.complete` is
  `False`, `unit.truncation` says which limit bit, a `context_truncated`
  qualification travels with the unit, and `unit.require_complete()` raises.

## Native relational checks and quantity accounting (experimental)

Three check families evaluate over the relations themselves, score through the
*same* numerical kernel `wise.score` uses (`wise._aggregation`), and produce a
separate typed result.

```python
from wise import oc
from wise.norm import Layer, View

catalogue = oc.ObjectNorm(
    constraints=(
        # a missing order is unknown, not absent, unless completeness is declared
        oc.ObjectConstraint("match", "matching", oc.RelatedObjectCardinality(role="order", minimum=1)),
        oc.ObjectConstraint(
            "three_way",
            "matching",
            oc.RelationalBalance(
                left=oc.AmountSelector(attribute="amount", unit="EUR"),
                right=oc.AmountSelector(role="receipts", attribute="amount", unit="EUR"),
                tolerance=0.05,
                width=0.20,
            ),
        ),
        oc.ObjectConstraint(
            "paid_late",
            "lead_times",
            oc.CrossObjectLag(
                activation=oc.EventSelector(activities=("Record Invoice Receipt",)),
                response=oc.EventSelector(role="payment", activities=("Execute Payment",)),
                delta=10,
                width=20,
            ),
        ),
    ),
    layers=(Layer("matching"), Layer("lead_times")),
    views=(View("Finance", constraint_weights={"match": 0.3, "three_way": 0.4, "paid_late": 0.3}),),
)
result = oc.score_units(log, catalogue, units, spec=spec)
result.frame("Finance")  # score + contrib__<layer>, ready for prioritize
oc.object_backlog(result, by="anchor_type", view="Finance")
```

* **`RelatedObjectCardinality`** counts distinct related identities against a
  declared minimum and maximum. A shortfall is only a violation under a
  declared completeness assumption — otherwise the reason is
  `unverified_absence` and no number is produced. Inside a truncated context a
  count is a lower bound and an absence is refused outright.
* **`CrossObjectLag`** selects activation and response through object
  relations. Activation, matching rule, clock, equal-time interpretation and
  missingness are all declared fields. Two responses sharing the matching
  instant give a typed `ambiguous_match`, never a silent nearest-timestamp
  choice; a tie-break has to be asked for, and says that it was applied.
* **`RelationalBalance`** compares two matched amount sets through
  `wise.oc.accounting`, so one source amount cannot be consumed twice.
  Currencies are not converted without a declared rate table.
* **`wise.oc.accounting`** gives additive values canonical identities and
  allocates them in shares that never exceed one. Incomplete allocation is a
  reported residual, not an error; negative shares, duplicate consumption and
  over-allocation are refused. `combined()` shows the conserved sum, the
  distinct evidence behind it, and what claiming the full value per group would
  double count.
* **`OCScoreResult`** is deliberately not a `ScoreResult`: that class's `log`
  promises a case-based `EventLog`. Ranking stays inside one unit type unless
  you say otherwise.

```python
from wise.evaluation.ocel import naive_projection, identity_preserving_projection

events, report = naive_projection(log, case_type="invoice", amount_attribute="amount")
report.duplicated_amount, report.lost_o2o, report.keeps_event_identity
```

On the small fixture with a manually specified truth, both projections invent
six violations, miss two obligations and count one shared payment twice; the
native evaluation matches all fifteen cells and says "unknown" where the data
is silent. Preserving event identity does not restore a relation — it restores
the ability to name a witness and to measure the duplication.

`docs/semantics/object_centric.md` is the reference and
`examples/oc_invoice_review.py` runs the whole thing end to end.

## Bounded local assistance and safe norm drafts (experimental)

`wise.llm` is the library's whole language-model surface, and it is
deliberately thin: the application owns the assistant, its prompts, its audit
trail and its approval flow. `import wise` does not import it; importing it
needs no optional package, contacts no network and opens no socket.

```python
import wise
from wise.llm import AccessPolicy, EvidenceHost, LocalAssistant, Principal, Scope, ToolGateway
from wise.llm.ollama import OllamaConfig, OllamaProvider

result = wise.score(log, norm, evidence="full")
packet = wise.explain_priority(result, "company", "B", view="Finance", gamma=1.0)

# the application authenticates the principal and states what they may see;
# an empty policy permits nothing, and no reply can widen one
policy = AccessPolicy(
    Principal("reviewer-7", authenticated_by="sso"),
    scopes=frozenset({Scope.EXPLANATION, Scope.COMPARE_GROUPS}),
    runs=frozenset({result.manifest.run_id}),
    views=frozenset({"Finance"}),
    columns=frozenset({"company"}),
    row_filters={"company": frozenset({"A"})},
)
gateway = ToolGateway(EvidenceHost(result, packet=packet), policy)
provider = OllamaProvider(OllamaConfig(model="llama3.1:8b", allowed_models=("llama3.1:8b",)))

review = LocalAssistant(provider, gateway).review(packet)
print(review.report)  # the deterministic report, whether or not a model answered
review.status  # 'drafted' or 'deterministic_only'
review.reasons  # exactly what did not happen, if anything
```

What a model can do here, exhaustively: ask for one of six named read-only
tools, and propose an order for facts that already exist plus questions and
hypotheses marked unverified. What it cannot do, structurally rather than by
instruction:

- **compute or change a number.** The report prints `packet.facts`; a draft
  supplies order and prose. Every fact and evidence identifier it names is
  checked against the authorised packet, so valid JSON with invented
  references is refused.
- **drop a qualification.** Every `packet.limitations` entry is printed
  whether or not the draft mentioned one.
- **reach data the policy does not allow.** `AccessPolicy` permits nothing by
  default and its row mask runs *before* any aggregate. A wider-population
  comparator the principal is not entitled to is recomputed inside the
  authorised population and relabelled, with a `ComparatorChange` saying so.
- **name a seventh tool, a path or a URL.** `APPROVED_TOOLS` is frozen at
  import; dispatch is a table written out literally; every string argument is
  screened for schemes, control characters and traversal.
- **leave the machine.** The transport takes a loopback literal by default,
  validated when it is configured; it posts only to declared routes, refuses
  redirects, compares the final host, and drops an inherited proxy.
- **fall back to a cloud.** There is no second endpoint in the code. A missing
  server or model is `ProviderStatus.UNAVAILABLE`; nothing is pulled or
  started.
- **approve a rule.** A `NormDraft` is always `pending_human_review`; a
  candidate with an `eval` recipe is refused before anything is computed; a
  stale parent fingerprint is a `DraftConflict`; and `preview_norm_draft`
  scores on an isolated copy and can never apply or activate.

Retrieval over approved documents keeps document identity, span offsets,
digest, version, effective dates and access tags, filters on all of them
before ranking, and turns a disagreement between two approved sources into a
review question instead of a choice. Vague language — "promptly" — becomes a
clarification question, never an invented limit.

```python
from wise.llm import ApprovedDocument, LexicalIndex

index = LexicalIndex(
    [ApprovedDocument("pol-1", "Clearing", text, access_tags=("finance",), topic="invoice_clearing", version="3")]
)
found = index.search("invoice clearing", policy=policy)
found.spans[0].citation()  # 'pol-1 v3 [0:69] (approved, digest 96c848310b77)'
found.questions  # a contradiction between approved sources is a question
```

`docs/security/local-assistant.md` is the threat model, including what
loopback binding does *not* promise and which server-side controls remain the
deployment's own. `examples/local_review.py` and
`examples/review_norm_draft.py` run the whole path against a fake provider,
with no server and no model.

## Norm file

```json
{
  "schema_version": 2,
  "name": "Running P2P example", "version": "1", "scoring_mode": "layer_balanced",
  "layers": [
    {"id": "completeness", "name": "Core completeness"},
    {"id": "lead_times", "name": "Working-capital lead times"}
  ],
  "views": [
    {"name": "Finance", "constraint_weights": {"c1": 0.20, "c2": 0.45}},
    {"name": "Logistics", "layer_weights": {"completeness": 0.25, "lead_times": 0.15}}
  ],
  "derived_attributes": [
    {"name": "manual_touch_count", "kind": "count_events", "where": {"column": "org:resource", "regex": "^user"}}
  ],
  "constraints": [
    {"id": "c1", "layer": "completeness", "type": "presence",
     "params": {"activity": "Record Invoice Receipt", "m": 1},
     "weight": 1.0, "applicability": {}, "description": "Require an invoice receipt"},
    {"id": "c2", "layer": "lead_times", "type": "lag",
     "params": {"a": "Record Goods Receipt", "b": "Record Invoice Receipt", "delta": 10, "width": 20, "unit": "D"},
     "weight": 1.0, "applicability": {"flow_type": ["DF1", "DF2"]},
     "description": "GR → INV within 10 days"}
  ]
}
```

Type names may use the short forms `pres`, `lag`, `bal`, `sing`, `excl`,
`order`. Unknown keys are rejected. `Norm.fingerprint()` returns a hash of the
canonical form, recorded on every `ScoreResult` for provenance.

## Command line

```bash
wise validate norm.json
wise describe norm.json
wise check norm.json log.csv --case case --activity activity --timestamp time --attr flow_type
wise score norm.json log.csv --case case --activity activity --timestamp time \
     --attr company --attr spend_area --by company --by spend_area --view Finance --gamma 20 --out backlog.csv
```

`wise score` also accepts the opt-in sidecars `--manifest-out run.json`,
`--evidence-out evidence.json` and `--evidence-frame-out evidence.csv`. They do
not change the stdout CSV or its columns.

## Layout

```
src/wise/
  constraints.py     sat() and the constraint catalogue
  norm.py            Norm, Layer, View, applicability, JSON I/O
  log.py             EventLog and vectorised case primitives
  derive.py          declarative recipes for derived case attributes
  scoring.py         violation matrix, case scores, layer contributions
  prioritization.py  Priority Index, drivers, agreement, hotspots
  diagnostics.py     observation window, censoring, replication, validation table
  schema.py          the supported catalogue, read off the implementation
  evidence/          opt-in evidence records, run manifests, fitted calibrations
  explain/           opt-in comparator specification, exact explanations, rendering
  oc/                opt-in object-centric log, OCEL adapters, bounded units
  llm/               opt-in provider protocol, strict local transport, tool gateway,
                     access policy, lexical retrieval, norm-draft envelope
  datasets.py        the running example of the paper
  cli.py             command-line interface
tests/               unit tests, including the tables of the paper
tests/fixtures/ocel/ the pinned OCEL 2.0 JSON interchange fixtures
tests/fixtures/norm_drafts/  draft envelopes, including the roadmap contract example
examples/            quickstart, running-example norm, BPIC'19 norm and evaluation,
                     evidence review, explanation review, local review, norm-draft review
docs/semantics/      evidence and run records; comparators and explanations;
                     object-centric data and bounded units
docs/security/       the local assistant's threat model and its limits
docs/PUBLISHING.md   release procedure
```

## Citing

See `CITATION.cff`. Please cite the paper when you use the method.

## License

MIT.
