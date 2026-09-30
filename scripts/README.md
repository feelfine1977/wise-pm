# scripts/

Maintainer tooling that is not part of the `wise` package. Nothing here is
installed with the distribution; every script prints its usage with `--help`.

| Script | Purpose | Writes to the checkout |
|---|---|---|
| `compat_gate.sh` | Runs the wise-workbench suites against a `wise` source tree (the external compatibility gate). `make compat`. | No: JUnit files and logs go under `--out` |
| `packages/wise-pm/scripts/make_golden.py` | Regenerates the golden pipeline snapshot under `tests/`. `make golden`. | Writes only with `--write` (comparison is `--check`) |
| `packages/wise-pm/scripts/snapshot_api.py` | Regenerates the public-API contract snapshot `tests/contract/api_snapshot.json`. `make contract`. | Writes only with `--write` (comparison is `--check`) |
| `check_release.py` | Verifies that a release tag, the package versions, `CITATION.cff` and `CHANGELOG.md` agree. | No |
| `changelog_section.py` | Prints the `CHANGELOG.md` section of one version, for release notes. | No |

`make_golden.py` and `snapshot_api.py` touch the committed files only when
`--write` is passed; review the diff before committing a regenerated file.
Their `--help` describes the other modes.

## compat_gate.sh

The M0 safety net's last item (`06-plan.md` §1, M0.7): the workbench backend,
`wise-analytics` and `process-knowledge` suites, plus the BPIC'19 reference
reproduction through the application, run against the working tree of this
repository. It replaces the by-hand procedure recorded in
`wise-workbench-notes/library-cycles/*/EXTERNAL_GATE.md`; `make compat` calls
it with the defaults, and the future `compat.yml` workflow will run the same
steps against a built wheel.

### What it runs

For each suite, in this order, the script changes into the suite directory and
runs the suite's **own interpreter** with `PYTHONPATH` pointing at the library
source. `PYTHONPATH` precedes site-packages, so the editable `wise-pm` install in
each workbench environment is shadowed for the duration of the run; nothing is
installed, upgraded or edited in either tree.

| Suite id | Directory (under the workbench) | Interpreter | Command |
|---|---|---|---|
| `backend` | `apps/backend` | `apps/backend/.venv/bin/python` | `python -m pytest -q` |
| `analytics` | `packages/wise-analytics` | `packages/wise-analytics/.venv/bin/python` | `python -m pytest -q` |
| `knowledge` | `packages/process-knowledge` | `packages/process-knowledge/.venv/bin/python` | `python -m pytest -q` |
| `reference` | `apps/backend` | `apps/backend/.venv/bin/python` | `WISE_BPIC19_CSV=… python -m pytest -q tests/golden -k bpic` |

Every command also gets `-p no:cacheprovider --junitxml=<out>/<suite>.xml`, and
the environment carries `PYTHONDONTWRITEBYTECODE=1` and a
`HYPOTHESIS_STORAGE_DIRECTORY` under the results directory, so the gate leaves
no `.pytest_cache`, `__pycache__` or `.hypothesis` behind in the workbench.

Before running a suite the script imports `wise` with that suite's interpreter
and checks that it resolves to `<LIB_SRC>/wise/__init__.py`. The probe runs
from the suite directory with `PYTHONSAFEPATH=1`, so the directory you invoke
the script from is never on its `sys.path` (a stray `wise/` next to you cannot
be mistaken for the library). A mismatch marks the suite `FAIL` without running
it, because its numbers would not be about the tree under test. The check is a
guard on the interpreter's resolution order, not an observation of pytest's
own `sys.path`; for these three suites pytest inserts nothing that shadows
`PYTHONPATH`.

`WISE_BPIC19_CSV` is exported **only** for the `reference` suite and is
explicitly unset for the other three. This is how the notes obtained their
numbers: the backend's own BPIC'19 golden tests skip in the `backend` run and
are exercised once, in `reference`. Without a CSV the reference suite still
runs, all of its tests skip, and the summary reports it as *not exercised*
rather than passed. When a CSV **was** supplied and every reference test still
skipped, the suite is `FAIL` and the skip reasons recorded in its JUnit file
are echoed (`skip reason: …`): an explicitly requested reproduction that did
not happen must not turn the gate green. The CSV path is made absolute before
the suite changes into `apps/backend`, so a relative path works. The challenge
CSV is never committed. Supply its location explicitly with `--bpic19-csv`
or `WISE_BPIC19_CSV`; the path must identify the intended input version.

### Usage

```bash
scripts/compat_gate.sh [options] [LIB_SRC] [WORKBENCH_DIR] [-- <extra pytest args>]
```

Defaults: `LIB_SRC` is `packages/wise-pm/src/`, `WORKBENCH_DIR` is the sibling
checkout `../wise-workbench`, results go to `.compat-results/` in this
repository. `--help` prints the resolved defaults. Every path argument
(`LIB_SRC`, `WORKBENCH_DIR`, `--out`, `--bpic19-csv`, `--python`) may be
relative; it is resolved against the directory you run the script from before
any suite starts, because each suite runs after `cd` into its own directory.

```bash
scripts/compat_gate.sh                                  # three suites; reference not exercised
scripts/compat_gate.sh --bpic19-csv ~/data/BPI_Challenge_2019.csv   # full gate (reference adds ~3 min)
WISE_BPIC19_CSV=~/data/BPI_Challenge_2019.csv scripts/compat_gate.sh # same, via the environment
scripts/compat_gate.sh --only backend,analytics          # a subset (repeatable, comma separated)
scripts/compat_gate.sh --skip reference
scripts/compat_gate.sh --list                            # suites, directories, interpreter presence
scripts/compat_gate.sh -v -- -x                          # stream pytest output; stop each suite at first failure
scripts/compat_gate.sh --python ~/some/.venv/bin/python  # one interpreter for every suite (CI-style single env)
scripts/compat_gate.sh /path/to/other/wise/src /path/to/wise-workbench --out /tmp/gate
```

The script runs under the macOS system bash (3.2) and needs nothing beyond
`git` (optional, for the revision line) and the suites' interpreters.

### Output

One line per suite, then a verdict. This is the first run, against the tree
identical to release 0.1.0 (workbench at `a9dca0b`), with the reference CSV:

```
compat gate
  library source : /…/wise-next/src  (wise 0.1.0, 7293b5a-dirty)
  workbench      : /…/wise-workbench  (a9dca0b)
  results        : /…/wise-next/.compat-results
  WISE_BPIC19_CSV: /…/BPI_Challenge_2019.csv (reference suite only)

backend    FAIL  passed=255   failed=1    skipped=8    pytest exit 1 (tests failed)
    =========================== short test summary info ============================
    FAILED tests/distribution/test_wheels.py::test_installed_release - subprocess...
    1 failed, 255 passed, 8 skipped in 71.90s (0:01:11)
analytics  PASS  passed=86    failed=0    skipped=0
knowledge  PASS  passed=74    failed=0    skipped=3
reference  PASS  passed=4     failed=0    skipped=0

compat gate: FAIL  (3 passed, 1 failed, 0 not exercised; failing: backend)
```

The one failure above is environmental, not a library regression:
`test_installed_release` builds the workbench's own release wheels and needs
`build`, `wheel` and `setuptools`, which are in the backend's `dev` extra but
absent from the current `apps/backend/.venv`; it fails identically with the
released library and without `PYTHONPATH`. Reinstalling the backend's dev
extra in that environment (`apps/backend/.venv/bin/python -m pip install -e
'.[dev]'` inside the workbench) makes the gate green. The `reference` line is
the paper's Table XI reproduced through the application; `knowledge`'s three
skips are opt-in local datasets.

On a failing suite the `short test summary info` section of its log is echoed
to stderr. The results directory holds `<suite>.xml` (JUnit, from pytest),
`<suite>.log` (full pytest output) and `summary.txt` (the lines above without
the excerpt; no wall-clock, so two runs on the same trees compare byte for
byte). `.compat-results/` and `.hypothesis/` are listed in `.gitignore` as
part of M0; if your checkout predates that change, add `.compat-results/`
yourself (the script never edits `.gitignore`).

### Exit status

| Code | Meaning |
|---|---|
| 0 | every selected suite passed; a reference suite that skipped for lack of a CSV is reported as not exercised, not as a failure |
| 1 | at least one suite failed, `wise` did not resolve to `LIB_SRC` in a suite's interpreter, or the reference suite skipped every test although a CSV was supplied |
| 2 | usage error (unknown option or suite, missing directory, CSV path that does not exist) |
| 3 | a suite directory, its interpreter or its `pytest` module is missing; the message names the exact path, and no suite runs |

### Environments the script expects

The workbench's `tools/install.sh --dev` creates `apps/backend/.venv` with the
backend, both packages and the dev extras. The two package suites are run from
their own `.venv` directories (`python -m venv .venv && .venv/bin/pip install -e
'.[dev]'` inside `packages/wise-analytics` and `packages/process-knowledge`,
plus `wise-pm`). When only the backend environment exists, `--python
apps/backend/.venv/bin/python` (run from the workbench root; the script makes
the path absolute before changing into the suite directories) runs all four
suites with it, and the `running …` progress line names that interpreter
instead of the suite's `.venv`.
