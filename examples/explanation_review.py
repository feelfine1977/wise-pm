"""Explain a priority exactly: one comparator, one decomposition, no model.

Run it from the repository root::

    python examples/explanation_review.py

It uses the bundled purchase-to-pay example, so it needs no data and no
optional dependency. Pass an output directory to write the run record, the
evidence packet and the explanation next to each other::

    python examples/explanation_review.py /tmp/wise-review

This is the first checkpoint of the extension: score an ordinary event log,
export the run and its evidence, and obtain a coherent explanation of the
exact priority comparison — without any language model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import wise
from wise.explain import BaselineSpec, explain_priority, render_markdown, render_text, write_explanation

log = wise.datasets.running_p2p_log()
norm = wise.datasets.running_p2p_norm()

# 1. Score once, with evidence, and rank. The comparator is the default one:
#    the current scored population. It is now a named object rather than an
#    implicit convention.
result = wise.score(log, norm, evidence="full")
backlog = wise.prioritize(result, "company", view="Finance", gamma=1.0)
print("Backlog (Finance, gamma = 1)")
print(backlog[["n_cases", "mean_score", "gap", "PI", "stable_PI", "global_mean"]].round(6).to_string(), "\n")

# 2. Explain the top slice. The packet calls prioritize and layer_drivers with
#    the same comparator, view, keys and scored mask, so it cannot drift from
#    the ranking above.
packet = explain_priority(result, "company", view="Finance", gamma=1.0, evidence=result.evidence)
print(render_text(packet))

# 3. The identity that makes the decomposition exact.
deltas = sum(layer.delta for layer in packet.layers)
print(f"sum of layer deltas   {deltas:+.12f}")
print(f"signed score gap      {packet.priority.signed_gap:+.12f}")
print(f"sum of components     {sum(layer.component for layer in packet.layers):+.12f}")
print(f"stabilised index      {packet.priority.stable_priority_index:+.12f}\n")

# 4. The same slice against a frozen historical comparator: last period's run,
#    over a different population (three of the five cases). A different
#    population is expected; different assessment *semantics* would not be.
events = wise.datasets.running_p2p_events()
last_period = wise.EventLog(
    events[events["case"].isin(["A", "B", "C"])].copy(),
    case_col="case",
    activity_col="activity",
    timestamp_col="time",
    case_attributes=["flow_type", "company", "vendor"],
)
previous = wise.score(last_period, norm)
frozen = BaselineSpec.from_result(previous, "Finance", baseline_id="2025-Q4", population_id="2025-Q4")
print(f"Last period scored {frozen.population_size} cases; this period scores {int(result.scores['Finance'].notna().sum())}.")
against_history = explain_priority(result, "company", "B", view="Finance", gamma=1.0, baseline_spec=frozen)
print(f"Against {frozen.baseline_id}: reference {against_history.reference_score:.6f}, ")
print(f"  signed gap {against_history.priority.signed_gap:+.6f}, stable PI {against_history.priority.stable_priority_index:.6f}")
print(f"  explanation kind: {against_history.explanation_kind}\n")

# 5. And against a target that states only a score. The priority still works;
#    the layer attribution is explicitly unavailable rather than invented.
target = BaselineSpec.target(0.95, baseline_id="board-2026", view="Finance", scoring_mode=result.mode)
against_target = explain_priority(result, "company", "B", view="Finance", gamma=1.0, baseline_spec=target)
print(f"Against {target.baseline_id}: reference {against_target.reference_score:.2f}, ")
print(f"  stable PI {against_target.priority.stable_priority_index:.6f}, kind {against_target.explanation_kind}")
for layer in against_target.layers:
    print(f"  {layer.layer:<14} observed {layer.group_penalty:.6f}   delta {layer.delta if layer.delta is not None else 'null'}")
print()
for qualification in against_target.limitations:
    if qualification.code.value == "reference_profile_unavailable":
        print(f"  -> {qualification.message}\n")

# 6. Optional: write the run record, the evidence and the explanation.
if len(sys.argv) > 1:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(packet.manifest.to_json(), encoding="utf-8")
    (out / "evidence.json").write_text(result.evidence.to_json(), encoding="utf-8")
    written = write_explanation(packet, out, formats=("json", "md"))
    (out / "explanation.md").write_text(render_markdown(packet), encoding="utf-8")
    print("wrote:")
    for path in (out / "run.json", out / "evidence.json", *written, out / "explanation.md"):
        print(f"  {path}")
    print("\nRender the packet again at any time with:")
    print(f"  wise explain {written[0]} --format markdown")
