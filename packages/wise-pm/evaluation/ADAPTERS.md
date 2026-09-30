# Evaluation adapter index

The linked configurations define the recorded concerns, thresholds, widths and view weights used by each adapter. These are frozen illustrative assessment choices, not validated organisational policies. Source origins, identifiers, local hashes, representation relationships and available licence information are recorded per dataset in [data_catalog.json](data_catalog.json).

## Shared interpretation

- A ramp uses `clip((value - threshold) / width, 0, 1)`; binary and fractional rules are identified separately. Scores combine bounded severities, not monetary consequences.
- Scope and evidence are separate: an out-of-scope criterion is excluded; an in-scope criterion without usable evidence remains unavailable. Missing evidence is not a zero penalty. Coverage concerns available in-scope weight, not completeness of the whole source or criterion catalogue.
- Primary runs use flat aggregation, each view's raw criterion weights, case-count exposure, the current scored population as reference and `gamma = 0`. A layer multiplier applies to each criterion in that layer; it does not prescribe the layer's total weight. Declared sensitivity runs vary these choices separately.
- Group labels describe recorded contexts. They do not establish responsibility, causation, avoidable effort or savings. Population-relative priority measures excess bounded burden, and signed explanations retain negative offsets.
- [NATIVE_REPLAY.md](NATIVE_REPLAY.md) defines the execution boundary: native WISE scores prepared severities and scope through `Metric` norms; the adapters retain ownership of raw-event operationalisations and evidence witnesses.

## BPIC 2013: incident and problem histories

[Configuration](adapters/bpic2013/illustrative_norm.json) · [Adapter](adapters/bpic2013/bpic2013_transfer.py) · catalogue ID `bpic2013`.

- **Unit and rules:** each XES trace, analysed separately for incidents, open problems and closed problems. C01 counts support-team changes; C02 counts compressed A–B–A returns; C03 counts repeated substatus observations; C04 records Wait–User; C05 measures first-to-last observed span; C06 records active status after Completed.
- **Scope and evidence:** C04/C06 apply only to incidents; other criteria apply to all traces. Sequence rules reject invalid timestamps and ambiguous tied sequence values. Missing team/status rows are not removed to join the surrounding sequence. Open-problem XES has no team field, so routing evidence remains unavailable.
- **Contexts and views:** stable product is primary, with organisational-line summaries separately; missing/multiple context remains explicit. Views are Balanced, Routing and Timing.
- **Source limits:** open/closed exports overlap and are not pooled; their names are not adjudicated terminal states. Observed span is not SLA or resolution time, and later activity does not establish failed resolution. Auxiliary CSV fields do not silently enrich the primary XES.

## BPIC 2020: domestic declarations

[Configuration](adapters/bpic2020/norm_frozen.json) · [Adapter](adapters/bpic2020/run_bpic2020.py) · catalogue ID `bpic2020`.

- **Unit and rules:** declaration trace. D01/D02 measure repeated submissions and rejection observations; D03–D05 measure first submission→first final approval→first payment request→first handling intervals. D06 records request without handling in the available history; D07 counts multiple handling observations.
- **Scope and evidence:** each interval requires its starting activity for scope; missing endpoints or invalid/negative first-to-first intervals are unavailable. D06 requires a request. All other count rules apply to all declarations. Any invalid occurrence of an endpoint family invalidates that duration; later occurrences do not repair it.
- **Contexts and views:** recorded amount bands, including a separate zero band; submission quarter is secondary. Balanced, ClaimantService and PaymentReview use declaration counts, not amount exposure.
- **Source limits:** domestic declarations do not require preapproval; approval roles may be merged. No separate mandatory-role failures are inferred. Missing handling is not proof of nonpayment; multiple handling records are not proof of duplicate payment.

## OCEL purchase orders and ICPM purchase-order items

[Configuration](adapters/ocel_icpm/norms.json) · [Adapter](adapters/ocel_icpm/run_transfer.py) · catalogue IDs `ocel_p2p`, `icpm_hackathon`.

- **OCEL unit and rules:** purchase-order object, using direct event–object links only. O01 measures first creation→first approval; O02 compares first goods receipt with first approval; O03 counts distinct execution events per directly linked payment object, taking the maximum; O04 measures first creation→first direct payment.
- **OCEL evidence and context:** all four criteria are in scope; missing prerequisites remain unavailable. Each event is counted once per directly related PO. Initial vendor metadata defines primary groups; purchasing group is alternative context. Views are Balanced, Control and Timeliness.
- **OCEL limits:** the source is simulated. Shared payment events can affect multiple PO assessments; PO exposure is not unique payment/action exposure. Dangling object–object links are audited but not traversed. Initial attributes are not reconstructed historical vendor truth; full criterion coverage does not certify graph integrity.
- **ICPM unit and rules:** union of master and activity document+item keys, retaining activity-only items with unknown master context. I01 compares snapshot confirmed/scheduled dates; I02 compares first goods-receipt date with snapshot schedule; I03/I04 count recorded delivery-date/quantity changes; I05 is the bounded positive snapshot confirmed-quantity shortfall fraction.
- **ICPM scope and evidence:** known deletion flag `L` excludes I01/I02, but not I05. Missing dates, absent receipt evidence or unusable quantities stay unavailable. I03/I04 require activity evidence; no matching changes then means observed zero. I05 requires finite positive ordered and finite nonnegative confirmed quantities.
- **ICPM contexts and limits:** vendor ID×country is primary, SKU secondary; views are Balanced, Delivery and Changes. Source-row identity retains exact repeats, with collapse sensitivity. Snapshot commitments can differ from historical ones; first receipt is not completed fulfilment. I05 is not loss or unfulfilled-goods valuation; a separate nondeleted-cohort diagnostic checks deletion context.

## BPIC 2012 and 2017: loan applications

[Configuration](adapters/bpic2012_2017/illustrative_norm.json) · [Adapter](adapters/bpic2012_2017/bpic2012_2017_transfer.py) · catalogue IDs `bpic2012`, `bpic2017`.

- **Unit and rules:** application trace. C01 measures observed span; C02 counts repeated `W_` work starts per activity; C03 counts distinct resources on those starts; C04 measures offer multiplicity. BPIC 2017 also includes C05, repeated recorded `A_Incomplete` observations.
- **Scope and evidence:** C04 requires an offer activity; other criteria apply to all applications. Span needs valid timestamps and at least two events. Work-start rules retain lifecycle distinctions; missing resource evidence makes breadth unavailable, whereas no work starts gives observed zero. No lifecycle duration pairing or human-worker classification is inferred.
- **Offer semantics:** 2012 counts qualifying `O_CREATED` records, not distinct offer IDs. In 2017, `O_Create Offer` event identifiers are reconciled with `O_Created` offer identifiers; inconsistent offer evidence is unavailable.
- **Contexts and views:** 2012 uses recorded amount bands, with month secondary; 2017 uses application type×loan goal, with amount bands secondary. Views are Balanced, Timeliness and Interaction.
- **Source limits:** the cleaned 2017 CSV is a checked representation of the XES, not another replication. The two years are separate applications of the norm; similar fields do not establish calibrated cross-year score comparability or processing costs.

## Incident, AMR and procurement CSV transfers

[Configuration](adapters/kaggle_transfers/norms.json) · [Adapter](adapters/kaggle_transfers/run_kaggle_transfers.py) · catalogue IDs `incident_template`, `uci_incidents`, `amr_orders`, `p2p_csv`.

- **Shared policy:** all configured criteria are in scope. Missing prerequisites, conflicting required constants and invalid/negative intervals remain unavailable. Timestamps use the supplied naive local clock without business-calendar adjustment. Rows are retained in primary runs; exact-row collapse is a sensitivity.
- **Incident template:** M01 first Ticket created→last Ticket closed; M02 customer-reopen count; M03 shortfall below satisfaction 4 on a constant valid 1–5 integer scale. Group by constant issue type×report channel; views Balanced, Timing, Experience. Source identity is recorded, but real/synthetic status and licence remain unverified.
- **UCI incidents:** U01 constant resolved/opened endpoint interval; U02/U03 maximum observed cumulative reassignment/reopen counters, never sums across snapshots. Group by earliest recorded category with deterministic tie-breaking; views Balanced, Timing, Routing. Category can change, so that label is not responsible ownership. The catalogue links the original ServiceNow-derived UCI record and its licence.
- **AMR orders:** A01 first `CAPTURO`→last `TERMINO`; A02 assignment observations beyond one; A03 recorded centre/data/status changes. Group by constant service type; views Balanced, Timing, Changes. Provenance and domain meanings are unverified; invalid temporal order is unavailable, and repeated rows can affect results.
- **Procurement CSVs:** P01 first requisition start→last invoice-payment completion; P02 amendments; P03 supplier-dispute observations. Group by constant country; views Balanced, Timing, Changes. The two pooled files have disjoint case IDs; the third is an exact subset, not extra evidence. Phase mapping is descriptive, not required order; absent payment gives unknown duration, and missing currency prevents amount exposure.

## Sepsis histories and hospital workbook fixture

[Configuration](adapters/healthcare_transfers/norm_frozen.json) · [Adapter](adapters/healthcare_transfers/run_healthcare.py) · catalogue IDs `sepsis`, `hospital_workbook`.

- **Sepsis rules:** S01 counts additional nonmeasurement activity observations, excluding CRP, Leucocytes and LacticAcid; S02 measures first-to-last history; S03 measures first registration→first release; S04 records Return ER. S03 is in scope only when registration is observed; other criteria apply to all traces. Missing release or invalid endpoint evidence remains unavailable.
- **Contexts and views:** groups use distinct recorded release-label sets, including no recorded release. Views are Balanced, Patterns and Time. Shifted timestamps support within-case intervals, not calendar comparisons between cases.
- **Workbook:** three-case technical fixture with unknown provenance. W01 counts additional activity observations; W02 measures recorded span; all cases share one group. Views are Balanced and Time. This is not an independent clinical evaluation.
- **Interpretation:** these rules describe recorded pathways, not clinical appropriateness, risk, quality or provider fault. No laboratory values or demographic attributes enter the norm; release-code clinical meanings are not assumed.

## BPIC 2019: purchasing items

[Configuration](oracle/evaluation/illustrative_norm.json) · [Adapter](oracle/evaluation/bpic19_pipeline.py) · catalogue ID `bpic2019`.

- **Unit and rules:** purchasing item. C01–C03 record absent invoice receipt, invoice clearing and goods/service receipt under declared flow-specific scope; C04 measures first invoice receipt→first clearing; C05 tests first invoice before first goods/service receipt; C06 records cancellation; C07 counts price/quantity changes; C08 counts repeated invoice receipt; C09/C10 count recorded human touches/distinct human identifiers.
- **Scope and evidence:** invoice-related C01/C02/C04/C08 exclude Consignment; C03 applies to DF1/DF2 and C05 to DF1. C06/C07/C09/C10 apply to all items. Unusable resources make C09/C10 unavailable, not out of scope. Missing/implausible endpoints or negative lag make C04 unavailable; presence/absence concerns refer only to available history.
- **Contexts and views:** company×spend area; Balanced, Completion and Operations use explicit per-criterion layer multipliers. Document-level bootstrap preserves dependence between items sharing records; it does not deduplicate exposure.
- **Source limits:** this is the ten-criterion configuration, distinct from the historical 29-criterion setup. First-to-first lag is not invoice response pairing, user identifiers are not labour costs, and observed absence is not certified noncompletion. All weighted concerns are compensatory; there is no mandatory-control veto.
