"""Deterministic rendering of an explanation packet.

The renderer owns the *presentation*; the packet owns the *numbers*. Every
figure printed here is looked up as a :class:`~wise.explain.priority.Fact`, so
a report cannot contain a number that no fact asserts — which is the property
that keeps a later language model from quietly introducing one. It also owns
the vocabulary: "assessed penalty", "reference contrast", "priority
component", never "root cause" or "money saved".

Three renderings, one packet:

``render_text``
    a plain report for a terminal or a log;
``render_markdown``
    the same facts as tables, with every label escaped so a slice called
    ``|--|`` cannot break out of a cell or inject a link;
``render_json``
    the packet itself, byte-stable for diffing two runs.

All three are pure functions of the packet: the same packet renders to the
same bytes, on any machine, in any order of calls.

Generated file names come from :func:`explanation_filename`, which builds them
from safe identifiers — the explanation id and a sanitised group label — never
from a raw business label and never from anything that could contain a path.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import EvidenceError
from .priority import ExplanationPacket, safe_identifier

if TYPE_CHECKING:  # pragma: no cover
    from .priority import Fact

#: Renderings :func:`render` understands.
FORMATS = ("text", "markdown", "json")

#: File suffix to rendering, for :func:`write_explanation`.
_SUFFIX_FORMAT = {"txt": "text", "text": "text", "md": "markdown", "markdown": "markdown", "json": "json"}

_KIND_TITLE = {
    "observed_assessment": "What was observed",
    "relative_priority": "Why it has this priority",
    "alternative_view": "Under another approved view",
    "evidence_qualification": "What the evidence covers",
}

#: Full escape set, applied to **untrusted labels** — slice keys, activity
#: names, unit ids, view and layer names, ids.
_MD_LABEL_ESCAPE = "\\`*_{}[]()#+-.!|<>&~"

#: Structural escape set, applied to the library's own prose inside a table
#: cell. It neutralises everything that could break the cell, inject HTML or
#: form a link, and leaves ordinary punctuation readable.
_MD_CELL_ESCAPE = "\\`|<>&[]"


def escape_markdown(text: Any) -> str:
    """Escape a raw label for a Markdown table cell.

    Control characters, including the newline and the pipe that would end a
    cell, are removed rather than escaped, and every Markdown metacharacter is
    backslash-escaped, so a business label can neither restructure the table
    nor become a link.

    >>> escape_markdown("A|B")
    'A\\\\|B'
    >>> escape_markdown("[click](http://example.invalid)")
    '\\\\[click\\\\]\\\\(http://example\\\\.invalid\\\\)'
    >>> escape_markdown("line\\nbreak")
    'line break'
    """
    return "".join("\\" + ch if ch in _MD_LABEL_ESCAPE else ch for ch in escape_text(text))


def escape_prose(text: Any) -> str:
    """Make the library's own sentence safe inside a Markdown cell.

    Only the structural characters are escaped, so a description stays
    readable while it still cannot end the cell, inject HTML or become a link.

    >>> escape_prose("volume is 'exposure': it scales the gap (not the mean).")
    "volume is 'exposure': it scales the gap (not the mean)."
    >>> escape_prose("a | b <img> [x]")
    'a \\\\| b \\\\<img\\\\> \\\\[x\\\\]'
    """
    return "".join("\\" + ch if ch in _MD_CELL_ESCAPE else ch for ch in escape_text(text))


def escape_text(text: Any) -> str:
    """Flatten a raw label for a plain-text line: no control characters.

    >>> escape_text("vendor\\nV1")
    'vendor V1'
    """
    return "".join(" " if ch in "\r\n\t" else ch for ch in str(text) if ch.isprintable() or ch in "\r\n\t")


def explanation_filename(packet: ExplanationPacket, *, suffix: str = "md", directory: str | Path | None = None) -> Path:
    """A safe file name for a rendered explanation.

    Built from the explanation id and a sanitised group label, so a slice
    called ``../../etc/passwd`` cannot escape ``directory``. The suffix is
    restricted to a short alphanumeric extension.

    >>> from wise.explain.priority import ExplanationPacket
    >>> import wise
    >>> result = wise.score(wise.datasets.running_p2p_log(), wise.datasets.running_p2p_norm())
    >>> packet = wise.explain_priority(result, "company", "B", view="Finance")
    >>> explanation_filename(packet, suffix="md").name.endswith("-B.md")
    True
    """
    clean_suffix = "".join(ch for ch in str(suffix).lower() if ch.isalnum())[:8] or "txt"
    label = safe_identifier(packet.group_label, max_length=32)
    name = f"{safe_identifier(packet.explanation_id, max_length=32)}-{label}.{clean_suffix}"
    base = Path(directory) if directory is not None else Path()
    return base / name


# ------------------------------------------------------------------ helpers
def _fmt(value: Any, *, digits: int = 6) -> str:
    """Format a fact value. ``None`` is printed as an explicit ``null``."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return escape_text(value)


def _fact_line(fact: Fact) -> str:
    unit = "" if fact.unit in ("", "position") else f" {fact.unit}"
    return f"  {fact.fact_id:<34} {_fmt(fact.value):>14}{unit}   {escape_text(fact.description)}"


def _fmt_md(value: Any, *, digits: int = 6) -> str:
    """Format a fact value for Markdown: numbers plain, raw strings escaped."""
    return escape_markdown(value) if isinstance(value, str) else _fmt(value, digits=digits)


def _facts_by_kind(packet: ExplanationPacket, kind: str) -> tuple[Fact, ...]:
    return tuple(f for f in packet.facts if f.kind.value == kind)


def _comparator_sentence(packet: ExplanationPacket) -> str:
    spec = packet.baseline
    what = {
        "current_population": "the current scored population's own mean",
        "historical": "a frozen comparator observed on an earlier population",
        "target": "an explicit target",
    }[spec.kind.value]
    return f"Comparator {spec.baseline_id!r} ({spec.kind.value}): {what}, reference score {_fmt(packet.reference_score)}. " + (
        "It carries a complete layer profile, so the layer contrasts below add up to the score gap."
        if packet.additive_attribution
        else "It carries no layer profile, so no additive layer attribution against it is determined."
    )


# ----------------------------------------------------------------- renderers
def render_text(packet: ExplanationPacket) -> str:
    """A deterministic plain-text report.

    >>> import wise
    >>> result = wise.score(wise.datasets.running_p2p_log(), wise.datasets.running_p2p_norm())
    >>> packet = wise.explain_priority(result, "company", "B", view="Finance", gamma=1.0)
    >>> print(render_text(packet).splitlines()[0])
    Priority explanation - company B under view 'Finance'
    """
    p = packet.priority
    lines: list[str] = [
        f"Priority explanation - {escape_text(' '.join(packet.grouping))} {escape_text(packet.group_label)} "
        f"under view {p.view!r}",
        "=" * 100,
        f"explanation  {packet.explanation_id}",
        f"run          {packet.run_id}",
        f"unit type    {packet.unit_type}",
        f"assessment   {packet.explanation_kind}",
        "",
        _comparator_sentence(packet),
        "",
    ]
    for kind, title in _KIND_TITLE.items():
        facts = _facts_by_kind(packet, kind)
        if not facts:
            continue
        lines.append(title)
        lines.append("-" * len(title))
        lines.extend(_fact_line(f) for f in facts)
        lines.append("")

    lines.append("Layer decomposition")
    lines.append("-" * 19)
    header = f"  {'layer':<20}{'slice':>12}{'reference':>12}{'delta':>12}{'component':>14}{'clipped':>14}"
    lines.append(header)
    for layer in packet.layers:
        lines.append(
            f"  {escape_text(layer.layer):<20}{_fmt(layer.group_penalty):>12}{_fmt(layer.reference_penalty):>12}"
            f"{_fmt(layer.delta):>12}{_fmt(layer.component):>14}{_fmt(layer.clipped_component):>14}"
        )
    lines.append("")

    if packet.constraints:
        lines.append("Where the penalty sits")
        lines.append("-" * 22)
        lines.append(
            f"  {'constraint':<14}{'layer':<16}{'mean penalty':>14}{'share violated':>16}"
            f"{'share evaluated':>17}{'share bounded':>15}"
        )
        for constraint in packet.constraints:
            lines.append(
                f"  {escape_text(constraint.constraint_id):<14}{escape_text(constraint.layer):<16}"
                f"{_fmt(constraint.mean_penalty):>14}{_fmt(constraint.share_violated):>16}"
                f"{_fmt(constraint.share_evaluated):>17}{_fmt(constraint.share_lower_bound):>15}"
            )
        lines.append("")

    if packet.witnesses:
        lines.append("Witnesses")
        lines.append("-" * 9)
        lines.extend(f"  {line}" for line in _witness_lines(packet.witnesses))
        lines.append("")

    lines.append("Denominators")
    lines.append("-" * 12)
    for denominator in packet.denominators:
        lines.append(
            f"  {denominator.name:<22}{_fmt(denominator.count):>12} {denominator.unit_of_counting:<10} "
            f"{escape_text(denominator.description)}"
        )
    lines.append("")

    lines.append("Limitations")
    lines.append("-" * 11)
    for qualification in packet.limitations:
        lines.append(f"  [{qualification.code.value}] ({qualification.scope}) {escape_text(qualification.message)}")
    return "\n".join(lines) + "\n"


def render_markdown(packet: ExplanationPacket) -> str:
    """The same facts as escaped Markdown tables.

    >>> import wise
    >>> result = wise.score(wise.datasets.running_p2p_log(), wise.datasets.running_p2p_norm())
    >>> packet = wise.explain_priority(result, "company", "B", view="Finance")
    >>> render_markdown(packet) == render_markdown(packet)
    True
    """
    p = packet.priority
    out: list[str] = [
        f"# Priority explanation — {escape_markdown(packet.group_label)} under view `{escape_markdown(p.view)}`",
        "",
        f"- **Explanation** `{escape_markdown(packet.explanation_id)}`",
        f"- **Run** `{escape_markdown(packet.run_id)}`",
        f"- **Grouping** {', '.join('`' + escape_markdown(k) + '`' for k in packet.grouping)}",
        f"- **Assessment unit** {escape_markdown(packet.unit_type)}",
        f"- **Explanation kind** `{escape_markdown(packet.explanation_kind)}`",
        "",
        escape_prose(_comparator_sentence(packet)),
        "",
    ]
    for kind, title in _KIND_TITLE.items():
        facts = _facts_by_kind(packet, kind)
        if not facts:
            continue
        out.extend(
            [
                f"## {title}",
                "",
                "| Fact | Value | Unit | Of | Description |",
                "|---|---:|---|---|---|",
            ]
        )
        for fact in facts:
            out.append(
                f"| `{escape_markdown(fact.fact_id)}` | {_fmt_md(fact.value)} | "
                f"{escape_markdown(fact.unit)} | {escape_markdown(fact.denominator or '')} | "
                f"{escape_prose(fact.description)} |"
            )
        out.append("")

    out.extend(
        [
            "## Layer decomposition",
            "",
            "| Layer | Slice penalty | Reference penalty | Delta | Signed component | Clipped component |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for layer in packet.layers:
        out.append(
            f"| {escape_markdown(layer.layer)} | {_fmt(layer.group_penalty)} | {_fmt(layer.reference_penalty)} | "
            f"{_fmt(layer.delta)} | {_fmt(layer.component)} | {_fmt(layer.clipped_component)} |"
        )
    out.append("")

    if packet.constraints:
        out.extend(
            [
                "## Where the penalty sits",
                "",
                "| Constraint | Layer | Type | Mean penalty | Share violated | Share evaluated | Share lower bound |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for constraint in packet.constraints:
            out.append(
                f"| `{escape_markdown(constraint.constraint_id)}` | {escape_markdown(constraint.layer)} | "
                f"{escape_markdown(constraint.constraint_type)} | {_fmt(constraint.mean_penalty)} | "
                f"{_fmt(constraint.share_violated)} | {_fmt(constraint.share_evaluated)} | "
                f"{_fmt(constraint.share_lower_bound)} |"
            )
        out.append("")

    if packet.view_contrasts:
        out.extend(
            [
                "## The same slice under each approved view",
                "",
                "| View | Comparator | Mean score | Signed gap | Stabilised index | Rank | Scored population differs |",
                "|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for contrast in packet.view_contrasts:
            out.append(
                f"| {escape_markdown(contrast.view)} | `{escape_markdown(contrast.comparator)}` | "
                f"{_fmt(contrast.mean_score)} | {_fmt(contrast.signed_gap)} | {_fmt(contrast.stable_priority_index)} | "
                f"{_fmt(contrast.rank)} | {_fmt(contrast.scored_population_differs)} |"
            )
        out.append("")

    if packet.witnesses:
        out.extend(["## Witnesses", ""])
        out.extend(f"- {escape_prose(line)}" for line in _witness_lines(packet.witnesses))
        out.append("")

    out.extend(["## Denominators", "", "| Name | Count | Unit | Meaning |", "|---|---:|---|---|"])
    for denominator in packet.denominators:
        out.append(
            f"| `{escape_markdown(denominator.name)}` | {_fmt(denominator.count)} | "
            f"{escape_markdown(denominator.unit_of_counting)} | {escape_prose(denominator.description)} |"
        )
    out.append("")

    out.extend(["## Limitations", ""])
    for qualification in packet.limitations:
        out.append(
            f"- **{escape_markdown(qualification.code.value)}** ({escape_markdown(qualification.scope)}) — "
            f"{escape_prose(qualification.message)}"
        )
    out.append("")
    return "\n".join(out)


def render_json(packet: ExplanationPacket, *, indent: int | None = 2) -> str:
    """The packet as JSON — the interchange form, not a summary."""
    return packet.to_json(indent=indent)


def render(packet: ExplanationPacket, fmt: str = "text") -> str:
    """Render a packet in one of :data:`FORMATS`."""
    if fmt not in FORMATS:
        raise EvidenceError(f"format must be one of {FORMATS}, got {fmt!r}")
    if fmt == "json":
        return render_json(packet)
    return render_markdown(packet) if fmt == "markdown" else render_text(packet)


def _witness_lines(witnesses: Iterable[Any]) -> list[str]:
    lines: list[str] = []
    for witness in witnesses:
        if witness.kind.value == "absence":
            search = witness.search
            window = f"{search.window_start} .. {search.window_end}" if search is not None else "unknown window"
            activities = ", ".join(search.activities) if search is not None else ""
            completeness = search.completeness.value if search is not None else "unknown"
            lines.append(
                f"{witness.role} on {witness.unit_id}: nothing matching [{escape_text(activities)}] was found in "
                f"{window} ({completeness}); an absence has no event identity"
            )
        else:
            identity = witness.event_id if witness.event_id is not None else witness.reference
            lines.append(
                f"{witness.role} on {witness.unit_id}: {escape_text(witness.activity)} at {witness.timestamp} "
                f"[{witness.identity.value} {escape_text(identity)}]"
            )
    return lines


# --------------------------------------------------------------------- input
def load_explanation(path: str | Path) -> ExplanationPacket:
    """Read a packet written by :meth:`ExplanationPacket.to_json`.

    Parsed strictly and never executed: an explanation file is a record, and a
    reader must not be able to make a new number by editing prose around it.
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"{path}: not a valid explanation packet ({exc})") from exc
    if not isinstance(data, dict):
        raise EvidenceError(f"{path}: an explanation packet must be a JSON object")
    return ExplanationPacket.from_dict(data)


def write_explanation(
    packet: ExplanationPacket,
    directory: str | Path,
    *,
    formats: Sequence[str] = ("json", "md"),
) -> list[Path]:
    """Write one packet into ``directory`` under safe generated names.

    Returns the paths written, in the order of ``formats``. The names come
    from :func:`explanation_filename`, so an untrusted slice label cannot
    determine where the file lands.
    """
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        if fmt not in _SUFFIX_FORMAT:
            raise EvidenceError(f"unknown output format {fmt!r}; known: {sorted(_SUFFIX_FORMAT)}")
        target = explanation_filename(packet, suffix=fmt, directory=base)
        target.write_text(render(packet, _SUFFIX_FORMAT[fmt]), encoding="utf-8")
        written.append(target)
    return written
