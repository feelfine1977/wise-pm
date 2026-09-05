"""Command-line interface: ``wise validate | describe | check | score``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import pandas as pd

from ._version import __version__
from .errors import WiseError
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

    args = parser.parse_args(argv)
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
        result = score(log, norm, views=[args.view] if args.view else None, mode=args.mode)
        view = args.view or result.views[0]
        backlog = prioritize(result, args.by, view=view, gamma=args.gamma, volume=args.volume, min_cases=args.min_cases)
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
        return 0
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
