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
| Explicit lag bound | (C2, closed for v3 documents by the migration) | A `lag` states `delta` or `"unbounded": true`. A lag without either is refused when the document is read. | None: such a document cannot be read, so nothing is scored with it. |
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
