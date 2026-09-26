"""Record content hashes of the BPIC'19 reproduction under the current library (maintainer-run).

Usage: uv run python packages/wise-pm/scripts/record_bpic19_oracle.py /path/to/BPI_Challenge_2019.csv

Writes tests/data/bpic19_oracle.json with SHA-256 digests of the violation matrix, the scores, the
contributions per view and the Table XI backlog, plus the library and dependency versions. The gated
test tests/test_bpic19.py compares these digests first and the paper's printed values second, so a
drift smaller than the printed precision cannot pass unnoticed. The CSV is read only and never copied.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import wise

PKG = Path(__file__).resolve().parents[1]
ORACLE = PKG / "tests" / "data" / "bpic19_oracle.json"
BACKLOG_COLUMNS = ["n_cases", "mean_score", "gap", "PI", "stable_gap", "stable_PI"]


def digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(np.ascontiguousarray(frame.to_numpy(dtype=float)).tobytes()).hexdigest()


def digests(result: wise.ScoreResult) -> dict[str, str]:
    backlog = wise.prioritize(result, ["case Company", "case Spend area text"], view="Automation", gamma=20)
    return {
        "violations": digest(result.violations),
        "scores": digest(result.scores),
        **{f"contributions_{v}": digest(result.contributions[v]) for v in result.views},
        "backlog_automation_gamma20": digest(backlog[BACKLOG_COLUMNS]),
    }


def compute(csv: str) -> dict[str, object]:
    sys.path.insert(0, str(PKG / "examples"))
    from bpic19_evaluation import load  # example module, imported on demand

    log = load(csv)
    norm = wise.Norm.load(PKG / "examples" / "bpic19_norm.json")
    result = wise.score(log, norm)
    return {
        "wise": wise.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "norm_fingerprint": norm.fingerprint(),
        "n_cases": len(result.scores),
        "digests": digests(result),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    oracle = compute(argv[1])
    ORACLE.write_text(json.dumps(oracle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {ORACLE} ({oracle['n_cases']} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
