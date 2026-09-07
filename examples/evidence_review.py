"""Evidence review: score the running example and read why each number is what it is.

Run it from the repository root::

    python examples/evidence_review.py

It uses the bundled purchase-to-pay example, so it needs no data and no
optional dependency. Nothing is written unless you pass an output directory::

    python examples/evidence_review.py /tmp/wise-run
"""

from __future__ import annotations

import sys
from pathlib import Path

import wise
from wise.evidence import fingerprint_events, fit_calibration, to_interchange

log = wise.datasets.running_p2p_log()
norm = wise.datasets.running_p2p_norm()

# 1. Score with evidence capture. The scores are identical with and without it.
plain = wise.score(log, norm)
result = wise.score(log, norm, evidence="full")
assert result.scores.equals(plain.scores), "capture never moves a number"

print("Scores (Finance):")
print(result.scores["Finance"].round(4).to_string(), "\n")

# 2. The run record: the mode actually used, the views, how the input was identified.
manifest = result.manifest
print("Run record")
print(f"  run id              {manifest.run_id}")
print(f"  configuration       {manifest.config_fingerprint()[:16]}…")
print(f"  mode                {manifest.mode} ({manifest.mode_source}; norm default {manifest.norm_default_mode})")
print(f"  views               {list(manifest.views)}")
print(f"  norm fingerprint    {manifest.norm_fingerprint[:16]}…")
print(f"  observation policy  {manifest.observation.policy_id}")
print(f"  tie order           {manifest.preprocessing['tie_order_policy']}")
print(f"  source event ids    {'declared' if manifest.input.source_identity_available else 'not declared'}")
print(f"  git commit          {manifest.environment.git_commit or 'unknown (never read from the repository)'}")

# A data digest costs a pass over the events, so it is an explicit request.
digest, canonicalisation = fingerprint_events(log)
manifest = manifest.with_data_digest(digest, algorithm="sha256", canonicalisation=canonicalisation)
print(f"  data digest         {digest[:16]}… ({canonicalisation.split(',')[0]})\n")

# 3. One evidence row per constraint and case, with the raw measurements.
frame = result.evidence_frame("Finance")
columns = ["constraint_id", "in_scope", "evaluable", "reason_code", "violation", "effective_weight", "penalty"]
print("Evidence for case E (no invoice was ever recorded):")
print(result.evidence_frame("Finance", unit_id="E")[columns].to_string(index=False), "\n")

# 4. Why is c1 violated for E? Because of a *declared absence*, not an event.
packet = result.evidence
record = packet.record(f"{packet.run_id}:c1:E")
(witness,) = record.witnesses
search = witness.search
print("Witness for c1 on case E:")
print(f"  kind          {witness.kind.value} (source identity: {witness.identity.value})")
print(f"  searched for  {list(search.activities)}")
print(f"  window        {search.window_start} .. {search.window_end}")
print(f"  events seen   {search.n_events_searched}")
print(f"  completeness  {search.completeness.value}")
print(f"  filters       {search.filters}\n")

# 5. A lag keeps both endpoints and the duration, never one unexplained number.
lag = packet.record(f"{packet.run_id}:c2:A")
print("Measurements behind c2 on case A:")
for measurement in lag.measurements:
    bound = " (lower bound)" if measurement.lower_bound else ""
    print(f"  {measurement.name:<22} {measurement.value!s:<22} {measurement.unit}{bound}")
print(f"  -> violation {lag.violation} under {lag.policies}\n")

# 6. Coverage, and what it is not.
coverage = packet.coverage
print("Coverage")
print(f"  units {coverage.n_units}, checks {coverage.n_checks}")
print(f"  in scope {coverage.n_in_scope}, evaluated {coverage.n_evaluated}, unevaluable {coverage.n_unevaluable}")
print(f"  scored per view {coverage.n_scored}")
print(f"  {coverage.interpretation}\n")

print("Typed diagnostics")
for diagnostic in (wise.typed_event_replication(log), wise.typed_right_censored(log, "Clear Invoice", window="10D")):
    value = "undefined (zero denominator)" if diagnostic.value is None else f"{diagnostic.value:.2f}"
    print(f"  {diagnostic.name}: {diagnostic.numerator:g}/{diagnostic.denominator:g} {diagnostic.unit_of_counting} = {value}")
    print(f"    policy {diagnostic.policy}; {diagnostic.interpretation}")
print()

# 7. A fitted reference, frozen once and applied without refitting.
log.add_case_attribute("touches", [1.0, 2.0, 3.0, 4.0, 20.0])
calibration = fit_calibration(log, {"name": "touch_index", "kind": "quantile_scale", "attribute": "touches", "q": 0.95})
print(
    f"Calibration {calibration.calibration_id}: divisor {calibration.divisor:g} "
    f"fitted on {calibration.fitting_population['n_units']} cases\n"
)

# 8. Qualifications travel with the packet.
print("Run qualifications")
for qualification in packet.qualifications:
    print(f"  [{qualification.code.value}] {qualification.message}")

# 9. Optional export: the native packet and the roadmap's interchange shape.
if len(sys.argv) > 1:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(manifest.to_json(), encoding="utf-8")
    (out / "evidence.json").write_text(packet.to_json(), encoding="utf-8")
    interchange = to_interchange(packet, view="Finance")
    import json

    (out / "evidence_packet.interchange.json").write_text(json.dumps(interchange, indent=2), encoding="utf-8")
    print(f"\nwrote run.json, evidence.json and evidence_packet.interchange.json to {out}")
