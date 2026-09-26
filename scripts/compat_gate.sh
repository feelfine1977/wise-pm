#!/usr/bin/env bash
# compat_gate.sh - run the wise-workbench suites against a wise source tree.
#
# Reproduces the by-hand external gate from
# wise-workbench-notes/library-cycles/*/EXTERNAL_GATE.md: each suite runs with
# its own interpreter, PYTHONPATH points at the library source under test, and
# nothing is installed or modified. One JUnit file per suite lands in --out.
#
# Works with the macOS system bash (3.2): no associative arrays, no mapfile.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
DEFAULT_LIB_SRC="$REPO_ROOT/packages/wise-pm/src"
DEFAULT_WORKBENCH="$(cd "$REPO_ROOT/.." && pwd -P)/wise-workbench"
DEFAULT_OUT="$REPO_ROOT/.compat-results"

# Suite table: parallel arrays, same index. Dirs are relative to the workbench.
SUITE_IDS=(backend analytics knowledge reference)
SUITE_DIRS=(apps/backend packages/wise-analytics packages/process-knowledge apps/backend)
SUITE_PYTEST_ARGS=("" "" "" "tests/golden -k bpic")
SUITE_DESCS=(
  "workbench backend: API, services, unit, golden (running example, presets)"
  "wise-analytics package"
  "process-knowledge packs (schemas, loaders, templates, matching)"
  "reference reproduction (paper Table XI through the application); needs WISE_BPIC19_CSV"
)

usage() {
  cat <<EOF
Usage: scripts/compat_gate.sh [options] [LIB_SRC] [WORKBENCH_DIR] [-- <extra pytest args>]

Run the workbench backend, wise-analytics and process-knowledge suites and the
BPIC'19 reference reproduction against the wise source tree LIB_SRC, each with
the suite's own interpreter (<suite dir>/.venv/bin/python) and
PYTHONPATH=LIB_SRC. Nothing is installed or modified in either tree.

Positional arguments
  LIB_SRC        library source directory containing wise/   (default: $DEFAULT_LIB_SRC)
  WORKBENCH_DIR  wise-workbench checkout                     (default: $DEFAULT_WORKBENCH)

Options
  --out DIR          results directory for <suite>.xml, <suite>.log, summary.txt
                     (default: $DEFAULT_OUT)
  --only SUITES      run only these suites (comma separated, repeatable)
  --skip SUITES      skip these suites (comma separated, repeatable)
  --bpic19-csv PATH  BPI Challenge 2019 CSV for the reference suite
                     (default: \$WISE_BPIC19_CSV; without it the suite skips and counts as not exercised)
  --python PATH      use this interpreter for every suite instead of each suite's .venv
  --list             list the suites, their directories and interpreters, then exit
  -v, --verbose      stream pytest output to the terminal as well as to the log
  -h, --help         this text
  --                 everything after it is appended to every pytest command

Relative paths (LIB_SRC, WORKBENCH_DIR, --out, --bpic19-csv, --python) are resolved
from the current directory before any suite runs; each suite itself runs after
cd into its own directory.

Suites: ${SUITE_IDS[*]}

Exit status
  0  every selected suite passed (a reference suite that skipped for lack of a CSV counts as not exercised)
  1  at least one suite failed, wise did not resolve to LIB_SRC in a suite's interpreter, or the
     reference suite skipped every test although a CSV was supplied
  2  usage error
  3  a suite directory, its interpreter or its pytest module is missing
EOF
}

die() {
  printf 'compat_gate: %s\n' "$*" >&2
  exit 2
}

abspath_dir() {
  (cd "$1" 2>/dev/null && pwd -P)
}

abspath_file() {
  # $1 = an existing file; prints it with its directory resolved physically
  local d
  d="$(cd "$(dirname "$1")" 2>/dev/null && pwd -P)" || return 1
  printf '%s/%s' "$d" "$(basename "$1")"
}

suite_index() {
  local i
  for i in "${!SUITE_IDS[@]}"; do
    if [ "${SUITE_IDS[$i]}" = "$1" ]; then
      printf '%s' "$i"
      return 0
    fi
  done
  return 1
}

validate_suite_list() {
  local item
  for item in $1; do
    suite_index "$item" >/dev/null || die "unknown suite '$item' (valid: ${SUITE_IDS[*]})"
  done
}

in_list() {
  local needle="$1" item
  shift
  for item in "$@"; do
    [ "$item" = "$needle" ] && return 0
  done
  return 1
}

git_describe() {
  local rev dirty=""
  rev="$(git -C "$1" rev-parse --short HEAD 2>/dev/null)" || {
    printf 'not a git checkout'
    return 0
  }
  if [ -n "$(git -C "$1" status --porcelain --untracked-files=no 2>/dev/null | head -n 1)" ]; then
    dirty="-dirty"
  fi
  printf '%s%s' "$rev" "$dirty"
}

LIB_SRC=""
WORKBENCH=""
OUT="$DEFAULT_OUT"
ONLY=""
SKIP=""
BPIC19_CSV="${WISE_BPIC19_CSV:-}"
PYTHON_OVERRIDE=""
LIST=0
VERBOSE=0
EXTRA_PYTEST_ARGS=()
POSITIONAL=()

while [ $# -gt 0 ]; do
  case "$1" in
    --out) [ $# -ge 2 ] || die "--out needs a directory"; OUT="$2"; shift 2 ;;
    --out=*) OUT="${1#--out=}"; shift ;;
    --only) [ $# -ge 2 ] || die "--only needs a suite list"; ONLY="$ONLY ${2//,/ }"; shift 2 ;;
    --only=*) ONLY="$ONLY ${1#--only=}"; ONLY="${ONLY//,/ }"; shift ;;
    --skip) [ $# -ge 2 ] || die "--skip needs a suite list"; SKIP="$SKIP ${2//,/ }"; shift 2 ;;
    --skip=*) SKIP="$SKIP ${1#--skip=}"; SKIP="${SKIP//,/ }"; shift ;;
    --bpic19-csv) [ $# -ge 2 ] || die "--bpic19-csv needs a path"; BPIC19_CSV="$2"; shift 2 ;;
    --bpic19-csv=*) BPIC19_CSV="${1#--bpic19-csv=}"; shift ;;
    --python) [ $# -ge 2 ] || die "--python needs a path"; PYTHON_OVERRIDE="$2"; shift 2 ;;
    --python=*) PYTHON_OVERRIDE="${1#--python=}"; shift ;;
    --list) LIST=1; shift ;;
    -v|--verbose) VERBOSE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_PYTEST_ARGS=("$@"); break ;;
    -*) die "unknown option '$1' (see --help)" ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

case "${#POSITIONAL[@]}" in
  0) ;;
  1) LIB_SRC="${POSITIONAL[0]}" ;;
  2) LIB_SRC="${POSITIONAL[0]}"; WORKBENCH="${POSITIONAL[1]}" ;;
  *) die "at most two positional arguments (LIB_SRC WORKBENCH_DIR), got ${#POSITIONAL[@]}" ;;
esac
LIB_SRC="${LIB_SRC:-$DEFAULT_LIB_SRC}"
WORKBENCH="${WORKBENCH:-$DEFAULT_WORKBENCH}"

validate_suite_list "$ONLY"
validate_suite_list "$SKIP"

# A relative --python is resolved once, here, from the caller's directory: every
# suite runs after cd into its own directory, where the relative path would not exist.
if [ -n "$PYTHON_OVERRIDE" ] && [ -f "$PYTHON_OVERRIDE" ] && [ -x "$PYTHON_OVERRIDE" ]; then
  PYTHON_OVERRIDE="$(abspath_file "$PYTHON_OVERRIDE")"
fi

# --- --list needs the workbench only -----------------------------------------
WORKBENCH_ABS="$(abspath_dir "$WORKBENCH")" || WORKBENCH_ABS=""

interpreter_for() {
  # $1 = suite dir (absolute)
  if [ -n "$PYTHON_OVERRIDE" ]; then
    printf '%s' "$PYTHON_OVERRIDE"
  else
    printf '%s/.venv/bin/python' "$1"
  fi
}

if [ "$LIST" = 1 ]; then
  printf 'workbench: %s%s\n\n' "$WORKBENCH" "$([ -n "$WORKBENCH_ABS" ] || printf ' (not found)')"
  printf '%-10s %-28s %-44s %s\n' "suite" "directory" "interpreter" "status"
  for i in "${!SUITE_IDS[@]}"; do
    dir="${WORKBENCH_ABS:-$WORKBENCH}/${SUITE_DIRS[$i]}"
    py="$(interpreter_for "$dir")"
    if [ ! -d "$dir" ]; then
      status="directory missing"
    elif [ ! -x "$py" ]; then
      status="interpreter missing"
    else
      status="present ($("$py" --version 2>&1))"
    fi
    shown="$py"
    if [ -n "$WORKBENCH_ABS" ]; then
      shown="${py#"$WORKBENCH_ABS"/}"     # relative to the workbench when it lies inside it
    fi
    printf '%-10s %-28s %-44s %s\n' "${SUITE_IDS[$i]}" "${SUITE_DIRS[$i]}" "$shown" "$status"
    printf '%-10s %s\n' "" "${SUITE_DESCS[$i]}"
    [ -n "${SUITE_PYTEST_ARGS[$i]}" ] && printf '%-10s pytest args: %s\n' "" "${SUITE_PYTEST_ARGS[$i]}"
  done
  exit 0
fi

# --- validate inputs ----------------------------------------------------------
LIB_SRC_ABS="$(abspath_dir "$LIB_SRC")" || die "library source directory not found: $LIB_SRC"
[ -f "$LIB_SRC_ABS/wise/__init__.py" ] || die "no wise/__init__.py under $LIB_SRC_ABS (pass the src directory, not the repository root)"
[ -n "$WORKBENCH_ABS" ] || die "workbench directory not found: $WORKBENCH"
if [ -n "$BPIC19_CSV" ]; then
  [ -f "$BPIC19_CSV" ] || die "BPI Challenge 2019 CSV not found: $BPIC19_CSV"
  BPIC19_CSV="$(abspath_file "$BPIC19_CSV")"   # the reference suite resolves it from its own directory
fi
if [ -n "$PYTHON_OVERRIDE" ] && { [ ! -f "$PYTHON_OVERRIDE" ] || [ ! -x "$PYTHON_OVERRIDE" ]; }; then
  printf 'missing: --python interpreter not found or not executable: %s\n' "$PYTHON_OVERRIDE" >&2
  exit 3
fi
mkdir -p "$OUT"
OUT_ABS="$(abspath_dir "$OUT")"
EXPECTED_WISE="$LIB_SRC_ABS/wise/__init__.py"

export PYTHONPATH="$LIB_SRC_ABS${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1                                # leave no __pycache__ in either tree
export HYPOTHESIS_STORAGE_DIRECTORY="${HYPOTHESIS_STORAGE_DIRECTORY:-$OUT_ABS/hypothesis}"

# --- which suites run ---------------------------------------------------------
SELECTED=()
for i in "${!SUITE_IDS[@]}"; do
  id="${SUITE_IDS[$i]}"
  if [ -n "$ONLY" ] && ! in_list "$id" $ONLY; then continue; fi
  if [ -n "$SKIP" ] && in_list "$id" $SKIP; then continue; fi
  SELECTED+=("$i")
done
[ "${#SELECTED[@]}" -gt 0 ] || die "no suite selected (--only/--skip left nothing)"

# --- preflight: every selected suite's directory, interpreter and pytest -----
MISSING=0
for i in "${SELECTED[@]}"; do
  id="${SUITE_IDS[$i]}"
  dir="$WORKBENCH_ABS/${SUITE_DIRS[$i]}"
  py="$(interpreter_for "$dir")"
  if [ ! -d "$dir" ]; then
    printf 'missing: suite %-9s directory not found: %s\n' "$id" "$dir" >&2
    MISSING=1
  elif [ ! -x "$py" ]; then
    printf 'missing: suite %-9s interpreter not found: %s\n' "$id" "$py" >&2
    MISSING=1
  elif ! "$py" -c 'import pytest' >/dev/null 2>&1; then
    printf 'missing: suite %-9s pytest is not importable from: %s\n' "$id" "$py" >&2
    MISSING=1
  fi
done
if [ "$MISSING" = 1 ]; then
  printf 'compat_gate: aborting before any suite ran; create the missing environments (workbench: tools/install.sh --dev for the backend, then a .venv with -e ".[dev]" in each package directory) or pass --python.\n' >&2
  exit 3
fi

# --- header ---------------------------------------------------------------------
# Probes run from the suite directory with PYTHONSAFEPATH=1 (3.11+): the caller's
# cwd is never on sys.path, so a stray wise/ next to the user cannot be picked up.
probe_wise() {
  # $1 = suite dir, $2 = interpreter, $3 = python expression to print
  (cd "$1" && PYTHONSAFEPATH=1 "$2" -c "import os, wise; print($3)")
}

FIRST_DIR="$WORKBENCH_ABS/${SUITE_DIRS[${SELECTED[0]}]}"
LIB_VERSION="$(probe_wise "$FIRST_DIR" "$(interpreter_for "$FIRST_DIR")" 'wise.__version__' 2>/dev/null || printf '?')"
SUMMARY="$OUT_ABS/summary.txt"
{
  printf 'compat gate\n'
  printf '  library source : %s  (wise %s, %s)\n' "$LIB_SRC_ABS" "$LIB_VERSION" "$(git_describe "$LIB_SRC_ABS")"
  printf '  workbench      : %s  (%s)\n' "$WORKBENCH_ABS" "$(git_describe "$WORKBENCH_ABS")"
  printf '  results        : %s\n' "$OUT_ABS"
  if [ -n "$BPIC19_CSV" ]; then
    printf '  WISE_BPIC19_CSV: %s (reference suite only)\n' "$BPIC19_CSV"
  else
    printf '  WISE_BPIC19_CSV: unset (the reference suite will skip)\n'
  fi
  [ "${#EXTRA_PYTEST_ARGS[@]}" -gt 0 ] && printf '  extra pytest   : %s\n' "${EXTRA_PYTEST_ARGS[*]}"
  printf '\n'
} | tee "$SUMMARY"

# --- run ------------------------------------------------------------------------
# Parses pytest's JUnit file and prints "passed failed skipped".
read_junit() {
  "$1" - "$2" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
tests = failures = errors = skipped = 0
for suite in suites:
    tests += int(suite.get("tests", 0))
    failures += int(suite.get("failures", 0))
    errors += int(suite.get("errors", 0))
    skipped += int(suite.get("skipped", 0))
print(tests - failures - errors - skipped, failures + errors, skipped)
PY
}

# Prints the distinct skip reasons recorded in a JUnit file, one per line (at most five).
junit_skip_reasons() {
  "$1" - "$2" <<'PY'
import sys
import xml.etree.ElementTree as ET

reasons: list[str] = []
for node in ET.parse(sys.argv[1]).getroot().iter("skipped"):
    lines = (node.get("message") or node.text or "").strip().splitlines()
    reason = lines[0] if lines else "(no reason recorded)"
    if reason not in reasons:
        reasons.append(reason)
print("\n".join(reasons[:5]))
PY
}

FAILED_SUITES=""
N_PASS=0
N_FAIL=0
N_SKIP=0

for i in "${SELECTED[@]}"; do
  id="${SUITE_IDS[$i]}"
  dir="$WORKBENCH_ABS/${SUITE_DIRS[$i]}"
  py="$(interpreter_for "$dir")"
  xml="$OUT_ABS/$id.xml"
  log="$OUT_ABS/$id.log"
  rm -f "$xml" "$log"

  # The numbers are meaningless unless this interpreter imports wise from LIB_SRC.
  resolved="$(probe_wise "$dir" "$py" 'os.path.realpath(wise.__file__)' 2>>"$log" || true)"
  if [ "$resolved" != "$EXPECTED_WISE" ]; then
    line="$(printf '%-10s FAIL     wise resolved to %s, expected %s' "$id" "${resolved:-<import failed, see $id.log>}" "$EXPECTED_WISE")"
    printf '%s\n' "$line" | tee -a "$SUMMARY"
    FAILED_SUITES="$FAILED_SUITES $id"
    N_FAIL=$((N_FAIL + 1))
    continue
  fi

  shown="${SUITE_DIRS[$i]}/.venv"
  if [ -n "$PYTHON_OVERRIDE" ]; then
    shown="$py"
  fi
  printf 'running %-9s %s%s\n' "$id" "$shown" "${SUITE_PYTEST_ARGS[$i]:+ (${SUITE_PYTEST_ARGS[$i]})}" >&2
  set +e
  (
    cd "$dir"
    if [ "$id" = "reference" ] && [ -n "$BPIC19_CSV" ]; then
      export WISE_BPIC19_CSV="$BPIC19_CSV"
    else
      unset WISE_BPIC19_CSV        # the notes ran the three suites without it; only the reference gets the CSV
    fi
    # shellcheck disable=SC2086  # SUITE_PYTEST_ARGS is a deliberate word list
    if [ "$VERBOSE" = 1 ]; then
      "$py" -m pytest -q -p no:cacheprovider --junitxml="$xml" ${SUITE_PYTEST_ARGS[$i]} \
        ${EXTRA_PYTEST_ARGS[@]+"${EXTRA_PYTEST_ARGS[@]}"} 2>&1 | tee "$log"
      exit "${PIPESTATUS[0]}"
    fi
    "$py" -m pytest -q -p no:cacheprovider --junitxml="$xml" ${SUITE_PYTEST_ARGS[$i]} \
      ${EXTRA_PYTEST_ARGS[@]+"${EXTRA_PYTEST_ARGS[@]}"} >"$log" 2>&1
  )
  rc=$?
  set -e

  passed=0; failed=0; skipped=0
  if [ -f "$xml" ]; then
    read -r passed failed skipped <<<"$(read_junit "$py" "$xml")"
  fi

  note=""
  skip_reasons=""
  if [ "$rc" -eq 0 ] && [ "$passed" -eq 0 ] && [ "$failed" -eq 0 ] && [ "$skipped" -gt 0 ] \
      && [ "$id" = "reference" ] && [ -n "$BPIC19_CSV" ]; then
    # The CSV was asked for explicitly; a reproduction that silently skipped must not turn the gate green.
    status="FAIL"
    note="CSV supplied but every test skipped; reasons below"
    skip_reasons="$(junit_skip_reasons "$py" "$xml")"
    FAILED_SUITES="$FAILED_SUITES $id"
    N_FAIL=$((N_FAIL + 1))
  elif [ "$rc" -eq 0 ] && [ "$passed" -eq 0 ] && [ "$failed" -eq 0 ]; then
    status="SKIP"
    note="nothing exercised"
    if [ "$id" = "reference" ] && [ -z "$BPIC19_CSV" ]; then
      note="WISE_BPIC19_CSV unset; reproduction not exercised"
    fi
    if [ "$skipped" -gt 0 ]; then
      skip_reasons="$(junit_skip_reasons "$py" "$xml")"
    fi
    N_SKIP=$((N_SKIP + 1))
  elif [ "$rc" -eq 0 ]; then
    status="PASS"
    N_PASS=$((N_PASS + 1))
  else
    status="FAIL"
    case "$rc" in
      1) note="pytest exit 1 (tests failed)" ;;
      2) note="pytest exit 2 (interrupted)" ;;
      3) note="pytest exit 3 (internal error)" ;;
      4) note="pytest exit 4 (usage error)" ;;
      5) note="pytest exit 5 (no tests collected)" ;;
      *) note="pytest exit $rc" ;;
    esac
    [ -f "$xml" ] || note="$note; no JUnit file written, see $id.log"
    FAILED_SUITES="$FAILED_SUITES $id"
    N_FAIL=$((N_FAIL + 1))
  fi

  line="$(printf '%-10s %-5s passed=%-5s failed=%-4s skipped=%-4s %s' "$id" "$status" "$passed" "$failed" "$skipped" "$note" | sed 's/[[:space:]]*$//')"
  printf '%s\n' "$line" | tee -a "$SUMMARY"
  if [ -n "$skip_reasons" ]; then
    printf '%s\n' "$skip_reasons" | sed 's/^/    skip reason: /' >&2
  elif [ "$status" = "FAIL" ] && [ -f "$log" ]; then
    excerpt="$(sed -n '/short test summary info/,$p' "$log" | head -n 40)"
    [ -n "$excerpt" ] || excerpt="$(tail -n 30 "$log")"
    printf '%s\n' "$excerpt" | sed 's/^/    /' >&2
  fi
done

printf '\n' | tee -a "$SUMMARY"
if [ -n "$FAILED_SUITES" ]; then
  printf 'compat gate: FAIL  (%d passed, %d failed, %d not exercised; failing:%s)\n' "$N_PASS" "$N_FAIL" "$N_SKIP" "$FAILED_SUITES" | tee -a "$SUMMARY"
  printf 'details: %s/<suite>.log and <suite>.xml\n' "$OUT_ABS" >&2
  exit 1
fi
printf 'compat gate: PASS  (%d passed, %d not exercised)\n' "$N_PASS" "$N_SKIP" | tee -a "$SUMMARY"
