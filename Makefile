# Developer loop. Every target goes through uv and the committed lockfile.
UV ?= uv
RUN := $(UV) run --frozen --no-sync

.PHONY: help setup lock test test-all check golden contract bench compat build clean

help:             ## list the targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed -E 's/:[^#]*##/ -/'

setup:            ## create/update the environment and install pre-commit hooks
	$(UV) sync --all-packages --group dev
	$(RUN) pre-commit install

lock:             ## refresh uv.lock after editing dependencies
	$(UV) lock

test:             ## fast suite (no benchmarks, no BPIC'19)
	$(RUN) pytest -m "not bench and not bpic19" -q

test-all:         ## everything that can run locally, including the long Hypothesis profile
	HYPOTHESIS_PROFILE=ci-long WISE_RUN_BENCH=1 $(RUN) pytest -m "not bpic19" -q

check:            ## what CI's lint job runs
	$(RUN) pre-commit run --all-files
	$(RUN) mypy
	$(RUN) lint-imports

golden:           ## regenerate the golden pipeline snapshot (review the diff before committing)
	$(RUN) python packages/wise-pm/scripts/make_golden.py --write

contract:         ## refresh the public-API snapshot (record the change in CHANGELOG first)
	$(RUN) python packages/wise-pm/scripts/snapshot_api.py --write

bench:            ## benchmark on the synthetic 1.6 M-event log
	$(RUN) python packages/wise-pm/benchmarks/bench_synthetic.py

compat:           ## workbench, analytics and knowledge suites against this tree
	scripts/compat_gate.sh

build:            ## wheels and sdists for every package, checked with twine
	rm -rf dist && $(UV) build --all-packages && $(RUN) twine check --strict dist/*

clean:            ## remove build, cache and smoke directories
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache .hypothesis .compat-results smoke sdist-check
