"""Rebuild raw assessments and compare them with the installed wise library."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from data_manager import DEFAULT_CATALOG, DEFAULT_DATA_ROOT, legacy_layout, load_catalogue, sha256

HERE = Path(__file__).resolve().parent
FAMILIES = ["bpic2013", "bpic2020", "ocel_icpm", "kaggle_transfers", "healthcare_transfers", "bpic2012_2017", "bpic2019"]


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_step(name: str, script: Path, arguments: list[str], output: Path, steps: list[dict]) -> None:
    if not script.is_file():
        raise FileNotFoundError(f"Missing evaluation entrypoint: {script.name}")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    started = time.monotonic()
    log_dir = output / "logs"
    log_dir.mkdir(exist_ok=True)
    with (log_dir / f"{name}.txt").open("w") as stream:
        result = subprocess.run(
            [sys.executable, str(script), *arguments],
            cwd=HERE,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=1800,
        )
    steps.append(
        {
            "name": name,
            "script": script.relative_to(HERE).as_posix(),
            "script_sha256": sha256(script),
            "returncode": result.returncode,
            "elapsed_seconds": time.monotonic() - started,
            "log": f"logs/{name}.txt",
        }
    )
    if result.returncode:
        raise RuntimeError(f"Evaluation step failed: {name}; inspect logs/{name}.txt")


def raw_steps(layout: Path, raw: Path, output: Path, steps: list[dict]) -> None:
    adapters, oracle = HERE / "adapters", HERE / "oracle"
    definitions = [
        ("bpic2013", "bpic2013_transfer.py", ["--input-dir", str(layout / "BPI Challenge 2013")]),
        (
            "bpic2020",
            "run_bpic2020.py",
            [
                "--input",
                str(layout / "BPI Challenge 2020_ Domestic Declarations_1_all"),
                "--config",
                str(adapters / "bpic2020/norm_frozen.json"),
            ],
        ),
        ("ocel_icpm", "run_transfer.py", ["--data-root", str(layout), "--norm", str(adapters / "ocel_icpm/norms.json")]),
        (
            "kaggle_transfers",
            "run_kaggle_transfers.py",
            ["--data-root", str(layout), "--norm", str(adapters / "kaggle_transfers/norms.json")],
        ),
        (
            "healthcare_transfers",
            "run_healthcare.py",
            ["--data-root", str(layout), "--norm", str(adapters / "healthcare_transfers/norm_frozen.json")],
        ),
        ("bpic2012_2017", "bpic2012_2017_transfer.py", ["--data-root", str(layout)]),
    ]
    for family, filename, source_args in definitions:
        target = raw / family
        run_step(
            family,
            adapters / family / filename,
            [*source_args, "--project-root", str(oracle), "--output", str(target)],
            output,
            steps,
        )
        if family == "bpic2013":
            run_step(
                "bpic2013_contract",
                adapters / family / "verify_contract.py",
                ["--results", str(target), "--input-dir", str(layout / "BPI Challenge 2013")],
                output,
                steps,
            )
        elif family == "ocel_icpm":
            run_step(
                "ocel_icpm_contract",
                adapters / family / "verify_transfer_contract.py",
                [
                    "--data-root",
                    str(layout),
                    "--norm",
                    str(adapters / family / "norms.json"),
                    "--output",
                    str(target / "contract_checks.json"),
                ],
                output,
                steps,
            )
    run_step(
        "bpic2019",
        oracle / "evaluation/bpic19_pipeline.py",
        ["--input", str(layout / "BPIC 2019/BPI_Challenge_2019.csv"), "--output", str(raw / "bpic2019")],
        output,
        steps,
    )


def run(data_root: Path, output: Path, prepared_root: Path | None = None, catalog: Path = DEFAULT_CATALOG) -> dict:
    data_root, output = data_root.resolve(), output.resolve()
    if output == data_root or output.is_relative_to(data_root) or data_root.is_relative_to(output):
        raise ValueError("Run output and raw data must be separate directory trees")
    if prepared_root is not None:
        prepared_root = prepared_root.resolve()
        if output == prepared_root or output.is_relative_to(prepared_root) or prepared_root.is_relative_to(output):
            raise ValueError("Run output and prepared input must be separate directory trees")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Run output must be empty; existing results are preserved")
    output.mkdir(parents=True, exist_ok=True)
    steps: list[dict] = []
    started = time.monotonic()
    oracle_file = HERE / "oracle/code/wise_reference.py"
    oracle_hash = sha256(oracle_file)
    manifest = {
        "status": "running",
        "mode": "prepared" if prepared_root else "raw",
        "source_policy": "catalogue-verified local bytes; no download or pickle loading",
        "catalog_sha256": sha256(catalog),
        "oracle_sha256": oracle_hash,
        "raw_replayed": prepared_root is None,
        "families": FAMILIES,
        "python_version": sys.version.split()[0],
        "steps": steps,
    }
    save(output / "run_manifest.json", manifest)
    try:
        if prepared_root is None:
            catalogue = load_catalogue(catalog)
            raw = output / "raw"
            raw.mkdir()
            with legacy_layout(data_root, catalogue) as layout:
                raw_steps(layout, raw, output, steps)
            manifest["source_preservation_checked"] = True
        else:
            raw = prepared_root
            if not raw.is_dir():
                raise FileNotFoundError("Prepared root is not a directory")
            manifest["source_preservation_checked"] = False
            manifest["prepared_mode_limit"] = "Reuses prepared assessments; no raw-source replay in this invocation"
        run_step(
            "native_parity",
            HERE / "run_native_parity.py",
            [
                "--evaluation-root",
                str(raw),
                "--adapter-root",
                str(HERE / "adapters"),
                "--output",
                str(output / "native"),
                "--norm-output",
                str(output / "native_norms"),
                "--oracle",
                str(oracle_file),
            ],
            output,
            steps,
        )
        if sha256(oracle_file) != oracle_hash:
            raise RuntimeError("Independent reference changed during execution")
        manifest.update(status="passed", oracle_preserved=True)
    except Exception as exc:
        manifest.update(status="failed", failure_type=type(exc).__name__)
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["steps"] = steps
        save(output / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()
    result = run(args.data_root, args.output, args.prepared_root, args.catalog)
    print(json.dumps({key: result[key] for key in ["status", "mode", "raw_replayed", "elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
