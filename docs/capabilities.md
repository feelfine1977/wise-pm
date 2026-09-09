# Capabilities and current limitations

This describes the development source, not a released or certified deployment.
The classic case workflow remains the default. Extension modules require
explicit imports/calls; `wise.oc`, `wise.llm` and `wise.evaluation` are not loaded
by `import wise`.

| Capability | State | Boundary |
|---|---|---|
| Classic scoring and prioritisation | Available | Pinned case signatures, masks, fingerprints and arithmetic |
| Evidence, manifests, calibration | Experimental | Case capture and native object records; bounded evidence |
| Comparator explanations | Experimental | Deterministic facts, signed contrasts and explicit unavailable profiles |
| Object-centric assessment | Experimental | OCEL 2.0 JSON read/write, SQLite read; bounded units and three check families |
| Local assistance and norm drafts | Experimental | Read-only gateway, explicit access policy, lexical retrieval and isolated previews |
| Recorded boundary evaluation | Implemented, experimental | Legacy S6 L04/M01 library work; nine offline scenarios and a gated model scenario |
| Rank sensitivity | Implemented, experimental | Sampling, parameter, construction and combined reports |
| Investigation records and application runtime selection | Application-owned | Legacy A01–A03; no application integration here |
| Live-model quality benchmark and field study | Not established | Offline boundary checks do not satisfy a model evaluation or V01 |
| Research-gated methods | Not implemented | Legacy M02–M04; no new research scope implied |

Machine-readable status is in
[extension-status.json](development/extension-status.json). Earlier validation
counts and pinned baseline observations remain in the
[historical checkpoint](development/extension-checkpoint.md) and
[foundation design](development/extension-plan.md). Their stage descriptions
refer to those earlier runs.

## Remaining hardening and API consistency gaps

- **Mixed object results.** `object_backlog(..., unit_type=...)` and native
  evidence capture can select a unit type. Shared comparator/explanation helpers
  still call `frame(view)` without consistently forwarding that selection, so
  `unit_type=` alone does not make all mixed-result paths work. Use a result
  scored for one type when building shared comparators or explanations. Explicit
  pooling carries a `heterogeneous_unit_types` attribute; it does not make the
  pooled volume meaningful. Some public type annotations still name only
  `ScoreResult` despite runtime support for a homogeneous `OCScoreResult`.
- **Native capture options and qualifications.** The general `capture_evidence`
  dispatcher does not apply case-only `mode`, `snapshot` or `witness_limit` options
  to native records. Use `capture_object_evidence` for its explicit supported
  signature. Native records have no lazy snapshot for additional witnesses.
  Record-limit and evaluation-limit qualifications still overlap in wording and
  flags; a capture can say the run is complete even when an evaluation budget
  also cut it. Inspect the result's `budget` alongside packet truncation. A
  bounded packet is not proof of a complete assessment.
- **Metadata locators.** Caller-supplied source labels and document locations
  are provenance metadata, not a filesystem or URL access policy. They are not
  a general-purpose sanitizer for secrets or unsafe locators. Review them before
  export; the library does not dereference them as an authority to read files.
- **Transport.** Routes are restricted to `/api/chat` and `/api/embed`; callers
  may narrow that set. Loopback validation is syntactic, including the accepted
  `localhost` name, rather than a guarantee about DNS resolution, proxies, server
  logging or host egress. Reviewed non-loopback hosts require explicit opt-in.
  Transport configuration does not secure or configure the model server.
- **Projection counters.** The native-versus-projected fixture counters are
  comparison diagnostics against declared truth. Error categories may overlap;
  do not sum them into a mutually exclusive error total or interpret them as
  general industrial accuracy. Event identity can recover witness provenance
  while leaving relation semantics lost by projection.
- **Accounting.** Conservation holds within one allocation report. Independent
  reports can reuse the same evidence. Net overlapping positive and negative
  amounts can cancel in an aggregate discrepancy; inspect record overlap as
  well as total-value qualifications.
- **Sensitivity comparator context.** `sampling_sensitivity` currently uses a
  supplied `BaselineSpec`'s kind and scalar value without enforcing its view,
  norm, scoring-mode or unit-type compatibility. Supply a comparator known to
  match the assessed result; do not treat this path as a compatibility check.
  A target declaring a different view/unit/norm is currently accepted.
- **Sensitivity.** Resampling calls the ranking function for each replicate.
  Large-run performance is not established here. Dependence and comparability
  of constructions are caller declarations; rank stability under supplied
  variants is not causal validity or a calibrated probability of benefit.

Large mathematical modules remain intact. Closing these gaps should use focused
contracts and regression cases, preserving classic numerical behavior. No
application integration, model download, live dataset run or research extension
is required to use the offline capabilities above.
