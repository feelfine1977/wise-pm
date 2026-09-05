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
correlations in `view_agreement`.

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
    case_col="case", activity_col="activity", timestamp_col="time",
    case_attributes=["flow_type", "company", "spend_area", "vendor"],
    exposure_col="net_worth",          # optional: exposure-weighted priorities
)

# Norm N = (C, Λ, lay) with views; built in code or loaded from a JSON file
norm = wise.Norm(
    constraints=[
        wise.NormConstraint("c1", "completeness", wise.Presence("Record Invoice Receipt")),
        wise.NormConstraint("c2", "lead_times", wise.Lag("Record Goods Receipt", "Record Invoice Receipt", delta=10, width=20, unit="D")),
        wise.NormConstraint("c3", "match", wise.Balance("amount", "Record Invoice Receipt", "amount", "Record Goods Receipt", tau=0.05, width=0.20)),
        wise.NormConstraint("c5", "handling", wise.Singularity("Record Goods Receipt", k=2, K=3)),
        wise.NormConstraint("c6", "exceptions", wise.Exclusion("Cancel Invoice Receipt"),
                            applicability={"flow_type": ["DF1", "DF2"]}),
    ],
    layers=[wise.Layer(name) for name in ["completeness", "lead_times", "match", "handling", "exceptions"]],
    views=[
        wise.View("Finance", constraint_weights={"c1": .20, "c2": .45, "c3": .20, "c5": .05, "c6": .10}),
        wise.View("Logistics", layer_weights={"completeness": .25, "lead_times": .15, "match": .05, "handling": .45, "exceptions": .10}),
    ],
)
norm.dump("norm_v1.json")                 # versioned artefact; wise.Norm.load(...) reads it back

# Score
result = wise.score(log, norm)
result.violations                         # cases × constraints, ν_c(σ); NaN = not applicable
result.scores                             # cases × views, S^(p)(σ); NaN = unscored
result.contributions["Finance"]           # cases × layers, Δ_λ (sums to 1 − S)

# Prioritise and explain
backlog = wise.prioritize(result, by=["company", "spend_area"], view="Finance", gamma=20)
wise.layer_drivers(result, by=["company", "spend_area"], view="Finance")
wise.constraint_drivers(result, "Finance", {"company": "A", "spend_area": "Packaging"})
wise.hotspot_table(backlog, drivers=wise.layer_drivers(result, ["company", "spend_area"], view="Finance"))
wise.concentration(backlog)               # share of slices carrying 80 % / 95 % of the priority mass
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
  datasets.py        the running example of the paper
  cli.py             command-line interface
tests/               unit tests, including the tables of the paper
examples/            quickstart, running-example norm, BPIC'19 norm and evaluation
docs/PUBLISHING.md   release procedure
```

## Citing

See `CITATION.cff`. Please cite the paper when you use the method.

## License

MIT.
