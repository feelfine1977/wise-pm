#!/usr/bin/env python3
"""Bounded parameter/evidence checks for the ICPM transfer; no source edits."""

import sys

sys.dont_write_bytecode = True
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import run_transfer as r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Dataset root containing the ICPM Hackathon directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON file for the four check results",
    )
    parser.add_argument(
        "--norm",
        type=Path,
        default=Path(__file__).with_name("norms.json"),
        help="Frozen norm JSON used by run_transfer.py",
    )
    args = parser.parse_args()
    root = args.data_root / "ICPM Hackathon"
    cfg = json.loads(args.norm.read_text())["icpm"]
    m = r.readcsv(root / "ICPM Data - Purchase Orders.csv")
    a = r.readcsv(root / "ICPM Data - Purchase Order Activities.csv")
    _f, v, A, _reason, raw, _events = r.icpm_assess(m, a, cfg)
    checks = []
    for field, bad in [
        ("Quantity confirmed", "inf"),
        ("Quantity confirmed", "-1"),
        ("Order Quantity", "0"),
    ]:
        changed = m.copy()
        changed.loc[0, field] = bad
        _ff, vv, _aa, rr, _, _ = r.icpm_assess(changed, a, cfg)
        case = m.loc[0, "Purchasing Document"] + "|" + m.loc[0, "Purchasing Document Item No"]
        assert np.isnan(vv.loc[case, "I05"])
        assert rr.loc[case, "I05"] == "missing_prerequisite"
        checks.append({"test": f"{field}={bad} remains unavailable", "passed": True})
    changed = copy.deepcopy(cfg)
    for c in changed["criteria"]:
        if c["id"] == "I01":
            c["threshold"] = 10
            c["width"] = 60
    _ff, vv, _aa, _rr, _, _ = r.icpm_assess(m, a, changed)
    expected = ((raw.I01 - 10) / 60).clip(0, 1).where(A.I01)
    assert np.allclose(vv.I01, expected, equal_nan=True)
    assert ((vv.I01 - v.I01).abs() > 1e-12).any()
    checks.append(
        {
            "test": "I01 norm threshold/width changes computed case assessments",
            "passed": True,
        }
    )
    report = {
        "status": "passed",
        "check_count": len(checks),
        "checks": checks,
        "claim": "Targeted parameter and unavailable-value checks, not domain validation",
        "data_root": str(args.data_root.resolve()),
        "norm_sha256": r.sha(args.norm),
        "scoring_script_sha256": r.sha(Path(r.__file__)),
        "check_script_sha256": r.sha(Path(__file__)),
    }
    r.dump(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": len(checks),
                "output": str(args.output.resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
