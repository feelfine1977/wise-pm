"""One shared comparator, one exact explanation (opt-in, no new dependency).

:func:`wise.prioritize` has always accepted a scalar comparator while
:func:`wise.layer_drivers` always contrasted against the current population.
Used together they described two different comparisons of one backlog. This
package resolves the comparator **once**:

* :mod:`~wise.explain.baseline` — :class:`BaselineSpec`, the comparator's
  identity, reference score, optional complete layer profile and the
  assessment semantics under which it is valid;
* :mod:`~wise.explain.priority` — :func:`explain_priority` and the versioned
  :class:`ExplanationPacket`: authoritative facts with identities, profile and
  priority components, denominators, witnesses and limitations;
* :mod:`~wise.explain.render` — deterministic text, Markdown and JSON, with
  escaped labels and safe generated file names.

Usage::

    result = wise.score(log, norm, evidence="summary")
    packet = wise.explain_priority(result, "vendor", view="Finance", gamma=20)
    print(wise.render_explanation(packet))

Importing this package pulls in nothing beyond NumPy, pandas and the standard
library, and contacts no network.
"""

from __future__ import annotations

from .baseline import (
    AGGREGATION,
    BASELINE_SCHEMA_VERSION,
    DEFAULT_TOLERANCE,
    SCORE_CONVENTION,
    AssessmentContext,
    BaselineCompatibility,
    BaselineError,
    BaselineKind,
    BaselineSpec,
    ResolvedBaseline,
    load_baseline,
    norm_context,
    resolve_baseline,
)
from .priority import (
    EXPLANATION_SCHEMA_VERSION,
    ConstraintComponent,
    Denominator,
    ExplanationPacket,
    Fact,
    FactKind,
    LayerComponent,
    PriorityComponents,
    ViewContrast,
    explain_priority,
    safe_identifier,
)
from .render import (
    FORMATS,
    escape_markdown,
    escape_prose,
    escape_text,
    explanation_filename,
    load_explanation,
    render_json,
    render_markdown,
    render_text,
    write_explanation,
)
from .render import render as render_explanation

__all__ = [
    "AGGREGATION",
    "BASELINE_SCHEMA_VERSION",
    "DEFAULT_TOLERANCE",
    "EXPLANATION_SCHEMA_VERSION",
    "FORMATS",
    "SCORE_CONVENTION",
    "AssessmentContext",
    "BaselineCompatibility",
    "BaselineError",
    "BaselineKind",
    "BaselineSpec",
    "ConstraintComponent",
    "Denominator",
    "ExplanationPacket",
    "Fact",
    "FactKind",
    "LayerComponent",
    "PriorityComponents",
    "ResolvedBaseline",
    "ViewContrast",
    "escape_markdown",
    "escape_prose",
    "escape_text",
    "explain_priority",
    "explanation_filename",
    "load_baseline",
    "load_explanation",
    "norm_context",
    "render_explanation",
    "render_json",
    "render_markdown",
    "render_text",
    "resolve_baseline",
    "safe_identifier",
    "write_explanation",
]
