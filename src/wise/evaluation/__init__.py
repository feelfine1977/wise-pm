"""Reproducible evaluation helpers. Nothing here is imported by ``import wise``.

Currently one module: :mod:`wise.evaluation.ocel`, which projects an
object-centric log into the case model in two documented ways and scores what
each projection costs against a manually specified truth. It exists so that the
claim "the native evaluation is different" is a table of numbers rather than an
assertion.

The sensitivity and task-evaluation helpers the roadmap places in this package
belong to a later stage and are deliberately absent: an empty public module
that returns nothing is worse than no module.
"""

from __future__ import annotations

__all__: list[str] = []
