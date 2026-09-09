"""Reproducible evaluation helpers. Nothing here is imported by ``import wise``.

Three modules, each measuring something the rest of the library only asserts:

:mod:`wise.evaluation.ocel`
    projects an object-centric log into the case model in two documented ways
    and scores what each projection costs against a manually specified truth,
    so that "the native evaluation is different" is a table of numbers.
:mod:`wise.evaluation.llm`
    an offline harness for the local-assistance boundary. Recorded material
    plus the outcome the library declares for it; the runner reports, per
    scenario, what was asked, what happened and whether that is the declared
    behaviour. It scores the boundary, not a model, and a scenario that would
    need a model is skipped with its environment variable named.
:mod:`wise.evaluation.sensitivity`
    rank sensitivity under dependency-aware resampling, under the method's own
    settings, and under alternative constructions — reported separately,
    because they are separate claims, and with the dependence stated.

Import them explicitly::

    from wise.evaluation.llm import load_tasks, run_tasks
    from wise.evaluation.sensitivity import sampling_sensitivity

None of the three is a claim about stakeholder usability, causal validity or
business benefit. A green harness says the refusals hold; a narrow rank
interval says the order is stable against what was varied.
"""

from __future__ import annotations

__all__: list[str] = []
