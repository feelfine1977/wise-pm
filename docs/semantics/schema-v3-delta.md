# Delta report: schema version 3 against schema version 2

Decision 0001 lets a norm document choose its semantics through its schema
version. Schema version 2 documents evaluate under the 0.1.0 semantics, so
their numbers are unchanged; reading one emits a `LegacySemanticsWarning`.
Schema version 3 documents evaluate under the corrected semantics. This report
lists, per correction, what changes between a schema version 2 norm and its
schema version 3 copy, and the rows that change on the committed data.

## Corrections and the rows they change

| Correction | Ledger | What differs under schema version 3 | Rows that can change |
|---|---|---|---|
| Three-valued applicability | C4, C25 | A case whose applicability rule reads a missing attribute is unknown under every operator, including `ne`, `not_in`, `isna` and `not`. It is not in scope, has no value (reason `scope_unknown`) and is left out of the constraint's denominator. | Only cases with a missing attribute under a rule that put them in scope in 0.1.0: `ne`, `not_in`, `isna`, a negated rule, or an `any` that one of these satisfied. Under `in`, `eq`, `gt`, `gte`, `lt`, `lte`, `notna` and `regex`, 0.1.0 already left such a case out of scope, so its value and score stay the same. |
| Explicit lag bound | C2 (closed for v3 documents) | A `lag` states `delta` or `"unbounded": true`. A lag without either is refused when the document is read; the migration writes the bound 0.1.0 read it with, `"delta": 0, "width": 0`, and recommends reviewing it. | None: such a document cannot be read, and a migrated lag keeps the 0.1.0 bound until it is reviewed. |
| Fitted calibration | C15 | A `quantile_scale` recipe carries its calibration record. Scoring refuses an unfitted one. | None: such a norm is refused by scoring, so nothing is scored with it. A fitted record gives the values of C15. |
| No `eval` recipe | C16 | An `eval` recipe is refused. | None: such a norm cannot be built. |

## The running example: `examples/running_p2p_norm_v3.json`

The schema version 3 copy of `examples/running_p2p_norm.json` differs from it
only in `"schema_version": 3` and `"missing_values": "three_valued"`. The golden
tier (`tests/golden/test_schema_v3_golden.py`) scores both norms on the paper's
running example (`wise.running_p2p_log()`, five cases) and on the golden
dataset (`tests/data/p2p_synthetic_2k.parquet`, 2 000 cases), in both scoring
modes. It compares the violations, scope, scores and layer contributions
exactly. A differing row must be a case with an unknown scope, and the set of
differing rows must equal the declared delta below.

| Dataset | Mode | Rows that differ | Correction |
|---|---|---|---|
| running example | flat | none | — |
| running example | layer_balanced | none | — |
| golden dataset | flat | none | — |
| golden dataset | layer_balanced | none | — |

No row differs, for two reasons. First, the only applicability rule of the
norm is `c4: {"flow_type": ["DF1"]}`, a membership rule; under it, 0.1.0
already left a case without `flow_type` out of scope. Neither dataset has a
case without `flow_type` in any case. Second, the norm's only lag (`c2`)
states its bound, and the norm has no recipes. The scores of paper Table VI
are reproduced by the copy as well.

The fingerprints differ, because the schema version and the missing-value
policy are part of what a norm measures:
`running_p2p_norm.json` 68e8462a1d19d8076dcb50376b4390720489116c92c8e24f72c41263a0f9006e,
`running_p2p_norm_v3.json` f675d41633acf29d420cb7cca7e17c0ce154af43c7c78571c019cffeb2c5aed5.

## The golden pipeline snapshot

The golden norm (`tests/data/golden/norm.json`), the BPIC'19 norm and every
other shipped norm stay at schema version 2. They evaluate under the 0.1.0
semantics, and every golden table is unchanged.

## Migrated norms

`wise.migrate` rewrites a schema
version 1 or 2 document to schema version 3 and reports every rewrite. Each
entry of the report names the ledger entries whose correction can change a
row of the migrated norm:

| Rule | Rewrite | Corrections named |
|---|---|---|
| `missing_values` | `"missing_values": "three_valued"` is added | C4, C25 |
| `eval_column` | an `eval` over a bare case-table column becomes `{"kind": "column", "attribute": ...}`; any other `eval` stops the migration with the recipe named | C16 |
| `calibration` | a `quantile_scale` without a record gets `{"status": "unfitted", "quantile": q}`, with the recommendation to fit it before scoring | C15 |
| `lag_bound` | a `lag` without `delta` gets `"delta": 0` (and `"width": 0` where it has none), with the recommendation to review the bound | C2 |
| `lag_unbounded` | `"delta": null` is written `"unbounded": true` | none |

Everything else in the document is copied as it was, numbers included.
The shipped norms migrate as follows (`tests/golden/test_schema_v3_golden.py`,
`tests/test_migration.py`, `tests/test_bpic19.py`):

| Norm | Rewrites | Rows that differ from the schema version 2 record |
|---|---|---|
| `examples/running_p2p_norm.json` | `missing_values`; the result is byte-identical to `examples/running_p2p_norm_v3.json` | none (see above) |
| `tests/data/golden/norm.json` | `missing_values` | none, in both modes on the golden dataset: no rule of the norm reads a missing attribute |
| `examples/bpic19_norm.json` | `missing_values`, `eval_column` (`total_events`), `calibration` (`exposure_norm`) | none once the record is fitted on the scored log: the column recipe reads what the `eval` read, and the fitted divisor is the quantile 0.1.0 took. Checked on a BPIC'19-shaped log in the fast suite; against the recorded Section V oracle when `WISE_BPIC19_CSV` is set. Fitted on another log, `exposure_norm` and the two `metric` constraints that read it change (C15). |
