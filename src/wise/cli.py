"""Command-line interface: ``wise validate | describe | check | score | explain``.

``wise score`` writes the same slice backlog to stdout as it always has. The
run record, the evidence packet and the explanation packet are **opt-in
sidecars**: they are written only when ``--manifest-out``, ``--evidence-out``,
``--evidence-frame-out`` or ``--explain-out`` name a file, they never change
the stdout CSV or its columns, and their progress messages go to stderr.

``--baseline-file`` reads a reviewed comparator (a
:class:`~wise.explain.BaselineSpec` as JSON) and ranks against it instead of
the current population's mean. It changes the reference the backlog uses — that
is what it is for — and it changes no column.

``wise explain <packet>`` renders an explanation packet written by
``--explain-out`` (or by :meth:`~wise.explain.ExplanationPacket.to_json`) as
text, Markdown or JSON. It computes nothing: every number in the output is a
fact of the packet.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

import pandas as pd

from ._version import __version__
from .errors import WiseError
from .explain import explain_priority, load_baseline, load_explanation, render_explanation
from .explain.render import FORMATS
from .log import ACTIVITY_COL, CASE_COL, TIMESTAMP_COL, EventLog
from .norm import Norm
from .prioritization import prioritize
from .scoring import score


def _read_log(args: argparse.Namespace) -> EventLog:
    read_kwargs = {"encoding": args.encoding} if args.encoding else {}
    return EventLog.from_csv(
        args.log,
        read_kwargs=read_kwargs,
        case_col=args.case,
        activity_col=args.activity,
        timestamp_col=args.timestamp,
        case_attributes=args.attr,
        exposure_col=args.exposure,
        missing_timestamps=args.missing_timestamps,
    )


def _add_log_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("log", help="event log (.csv or .parquet)")
    p.add_argument("--case", default=CASE_COL, help=f"case id column (default {CASE_COL!r})")
    p.add_argument("--activity", default=ACTIVITY_COL, help=f"activity column (default {ACTIVITY_COL!r})")
    p.add_argument("--timestamp", default=TIMESTAMP_COL, help=f"timestamp column (default {TIMESTAMP_COL!r})")
    p.add_argument("--attr", action="append", default=[], help="case attribute column (repeatable)")
    p.add_argument("--exposure", default=None, help="exposure column")
    p.add_argument("--encoding", default=None, help="CSV encoding")
    p.add_argument("--missing-timestamps", default="raise", choices=["raise", "drop", "keep"])


def _needs_evidence(args: argparse.Namespace) -> bool:
    return bool(args.evidence_out or args.evidence_frame_out)


def _write_sidecars(args: argparse.Namespace, result: Any, *, view: str, backlog: pd.DataFrame, spec: Any = None) -> None:
    """Write the opt-in run, evidence and explanation sidecars. Never touches stdout."""
    if args.manifest_out:
        manifest = result.manifest.finalize(
            grouping=args.by,
            view=view,
            volume=args.volume,
            comparator=f"{spec.kind.value}:{spec.baseline_id}" if spec is not None else "current_population_mean",
            comparator_value=backlog.attrs.get("baseline"),
            min_cases=args.min_cases,
            gamma=args.gamma,
        )
        with open(args.manifest_out, "w", encoding="utf-8") as fh:
            fh.write(manifest.to_json())
        print(f"wrote the run record to {args.manifest_out}", file=sys.stderr)
    if args.evidence_out:
        with open(args.evidence_out, "w", encoding="utf-8") as fh:
            fh.write(result.evidence.to_json())
        print(f"wrote {len(result.evidence)} evidence rows to {args.evidence_out}", file=sys.stderr)
    if args.evidence_frame_out:
        result.evidence_frame(view).to_csv(args.evidence_frame_out)
        print(f"wrote {len(result.evidence)} evidence rows to {args.evidence_frame_out}", file=sys.stderr)
    if args.explain_out:
        packet = explain_priority(
            result,
            args.by,
            view=view,
            gamma=args.gamma,
            volume=args.volume,
            min_cases=args.min_cases,
            baseline_spec=spec,
            evidence=result.evidence,
        )
        with open(args.explain_out, "w", encoding="utf-8") as fh:
            fh.write(packet.to_json())
        print(
            f"wrote the explanation of slice {packet.group_label!r} ({len(packet.facts)} facts) to {args.explain_out}",
            file=sys.stderr,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wise", description="WISE: norm-based, slice-first prioritisation")
    parser.add_argument("--version", action="version", version=f"wise {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="validate a norm file (structure only)")
    v.add_argument("norm")

    d = sub.add_parser("describe", help="print the constraint catalogue and weights of a norm")
    d.add_argument("norm")

    c = sub.add_parser("check", help="check a norm against a log (activities and attributes present, data quality)")
    c.add_argument("norm")
    _add_log_args(c)

    s = sub.add_parser("score", help="score a log and write a slice backlog")
    s.add_argument("norm")
    _add_log_args(s)
    s.add_argument("--by", action="append", required=True, help="slice key column (repeatable)")
    s.add_argument("--view", default=None, help="view name (default: first view)")
    s.add_argument("--gamma", type=float, default=0.0, help="shrinkage constant")
    s.add_argument("--volume", default="cases", help="'cases', 'exposure' or a column")
    s.add_argument("--min-cases", type=int, default=1)
    s.add_argument("--mode", default=None, choices=["flat", "layer_balanced"], help="override the norm's scoring mode")
    s.add_argument("--out", default="-", help="output CSV path ('-' = stdout)")
    s.add_argument("--cases-out", default=None, help="optional CSV with per-case scores")
    s.add_argument("--manifest-out", default=None, help="optional JSON run record (mode, views, input identity, options)")
    s.add_argument("--evidence-out", default=None, help="optional JSON evidence packet (implies evidence capture)")
    s.add_argument(
        "--evidence",
        default=None,
        choices=["summary", "full"],
        help="capture evidence: 'summary' (measurements and reasons) or 'full' (also bounded witnesses)",
    )
    s.add_argument("--evidence-frame-out", default=None, help="optional CSV of the long-format evidence rows")
    s.add_argument(
        "--baseline-file",
        default=None,
        help="optional JSON BaselineSpec: rank against a reviewed historical or target comparator",
    )
    s.add_argument(
        "--explain-out",
        default=None,
        help="optional JSON explanation packet for the top-ranked slice of this backlog",
    )

    e = sub.add_parser("explain", help="render an explanation packet (no computation, no model)")
    e.add_argument("packet", help="explanation packet written by 'wise score --explain-out'")
    e.add_argument("--format", default="text", choices=list(FORMATS), help="rendering (default text)")
    e.add_argument("--out", default="-", help="output path ('-' = stdout)")

    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        return _run(args)
    except WiseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    if args.cmd == "validate":
        norm = Norm.load(args.norm)
        print(f"OK: {norm}")
        return 0
    if args.cmd == "describe":
        norm = Norm.load(args.norm)
        with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 60):
            print(norm)
            print(norm.describe()[["layer", "type", "weight", "applicability", "description"]])
            print("\nRaw weights per view\n", norm.weight_table())
        return 0
    if args.cmd == "check":
        norm = Norm.load(args.norm)
        log = _read_log(args)
        if norm.derived_attributes:
            log.derive(norm.derived_attributes)
        issues = norm.check(log)
        print(log)
        print(log.validate().to_string())
        if issues:
            print("\nIssues:")
            for i in issues:
                print(" -", i)
            return 1
        print("\nNo issues found.")
        return 0
    if args.cmd == "score":
        norm = Norm.load(args.norm)
        log = _read_log(args)
        wants_evidence = args.evidence or (("full" if args.evidence_out else "summary") if _needs_evidence(args) else "none")
        result = score(log, norm, views=[args.view] if args.view else None, mode=args.mode, evidence=wants_evidence)
        view = args.view or result.views[0]
        spec = load_baseline(args.baseline_file) if args.baseline_file else None
        backlog = prioritize(
            result,
            args.by,
            view=view,
            gamma=args.gamma,
            volume=args.volume,
            min_cases=args.min_cases,
            baseline_spec=spec,
        )
        text = backlog.reset_index().to_csv(index=False)
        if args.out == "-":
            sys.stdout.write(text)
        else:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text)
            print(f"wrote {len(backlog)} slices to {args.out}", file=sys.stderr)
        if args.cases_out:
            result.frame().to_csv(args.cases_out)
            print(f"wrote {len(result.scores)} cases to {args.cases_out}", file=sys.stderr)
        _write_sidecars(args, result, view=view, backlog=backlog, spec=spec)
        return 0
    if args.cmd == "explain":
        text = render_explanation(load_explanation(args.packet), args.format)
        if args.out == "-":
            sys.stdout.write(text if text.endswith("\n") else text + "\n")
        else:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text)
            print(f"wrote the explanation to {args.out}", file=sys.stderr)
        return 0
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
