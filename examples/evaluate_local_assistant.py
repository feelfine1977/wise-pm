"""Score the local-assistance boundary against recorded material — offline.

Run it from the repository root::

    python examples/evaluate_local_assistant.py

It reads the recorded scenarios in ``tests/fixtures/llm_tasks`` and puts each
one through the real gateway, the real access policy, the real draft reader and
the real evidence packet. It needs no data, no optional dependency, no server
and no model, and nothing in this file opens a socket.

What it shows, in order:

1. the per-scenario report — what was asked, what happened, and whether that is
   the behaviour the library declares;
2. the scenario that would need a model, skipped with its environment variable
   named, because a benchmark that quietly reports a model it never ran is a
   fabricated result;
3. that the harness can fail: one declared outcome is deliberately replaced
   with the wrong one, and the run turns red;
4. the sensitivity beside it — a rank is only as good as its stability, and the
   resampling states the dependence it is over.

What this measures and what it does not: the boundary's behaviour on recorded
material, not the quality, latency or usefulness of any model. A green run says
the refusals hold; it says nothing about whether a reviewer is better off.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import wise
from wise.evaluation.llm import BoundaryMaterial, Outcome, load_tasks, run_tasks
from wise.evaluation.sensitivity import combined_stability, parameter_sensitivity, sampling_sensitivity

TASKS = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "llm_tasks"

# 1. the harness ------------------------------------------------------------
tasks = load_tasks(TASKS)
material = BoundaryMaterial.running_example()
report = run_tasks(tasks, material=material)
print(report.render())

# 2. what it could not run --------------------------------------------------
print("=" * 100)
print("2. Scenarios that did not run, and why.")
print("=" * 100)
for outcome in report.skipped():
    print(f"   {outcome.task_id}: {outcome.skip_reason}")
if not report.skipped():  # pragma: no cover - only with a caller-supplied provider
    print("   (none)")
print()

# 3. the harness can fail ---------------------------------------------------
broken = [
    dataclasses.replace(task, expected_outcome=Outcome.SERVED.value) if task.task_id.startswith("01_") else task for task in tasks
]
red = run_tasks(broken, material=material)
print("=" * 100)
print("3. One declared outcome replaced with the wrong one. The run turns red.")
print("=" * 100)
for failure in red.failures():
    print(f"   {failure.task_id}: declared {failure.expected!r}, observed {failure.observed!r}")
    print(f"      {failure.happened}")
print(f"   ok before: {report.ok}    ok after: {red.ok}")
print()

# 4. rank sensitivity beside it ---------------------------------------------
result = wise.score(wise.running_p2p_log(), wise.running_p2p_norm())
sampling = sampling_sensitivity(result, "company", view="Finance", dependence="vendor", B=300, seed=7, gamma=1.0)
parameters = parameter_sensitivity(
    result, "company", view="Finance", settings=[{"gamma": 0.0}, {"gamma": 1.0}, {"gamma": 20.0}, {"baseline": 0.9}]
)
print("=" * 100)
print("4. Does the ranking hold? Rank stability, with the dependence stated.")
print("=" * 100)
print(f"   {sampling.dependence.statement()}")
print(sampling.table()[["n_units", "point_rank", "rank_low", "rank_high", "present_share", "verdict"]].to_string())
print()
combined = combined_stability([sampling, parameters])
print(f"   combined verdicts: {combined.table()['verdict'].to_dict()}")
for qualification in combined.qualifications:
    print(f"   [{qualification.code.value}] {qualification.message}")
print()
print("   does not cover:")
for line in sampling.does_not_cover:
    print(f"     - {line}")
