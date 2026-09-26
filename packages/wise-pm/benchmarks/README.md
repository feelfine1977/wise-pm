# Benchmarks

End-to-end timing of the classic `wise` pipeline on a synthetic
purchase-to-pay log, plus the performance gate that runs it in CI when asked
to. The script is outside `testpaths`; `tests/perf/test_benchmarks.py` is collected but skipped unless `WISE_RUN_BENCH=1`.

## What is measured

`bench_synthetic.py` generates, from `numpy.random.default_rng(0)`, a log of
**250 000 cases / 1 600 000 events**:

- integer case ids; 7 activities from the running example vocabulary
  (`Create Purchase Order Item`, `Record Goods Receipt`, `Record Invoice
  Receipt`, `Clear Invoice`, `Cancel Invoice Receipt`, `Change Price`,
  `Vendor creates invoice`); every case has a PO/GR/INV/CLR skeleton and the
  remaining 600 000 events are drawn with weights favouring fragmented
  receipts, cancellations and price changes;
- case attributes `company` (20 values), `vendor` (2 000 values) and
  `flow_type` (`DF1`/`DF2`, constant per case);
- `amount` uniform in [10, 10 000], timestamps `2018-01-01 + U(0, 400)` days,
  an event `resource` column (`user_NN` / `batch_NN`) for the
  `count_events` recipe;
- rows shuffled, so the build has to sort.

The norm is the running example (Table V, six constraints in five layers)
extended to **10 constraints and 4 views**, scoring mode `layer_balanced`:
a `Lag` with `activation="each"` (`c7`), a `Precedence` (`c8`), an
`Exclusion` scoped to events after the goods receipt (`c9`) and a `Metric` on
the derived attribute `manual_touches` (`count_events` with a regex
where-clause, `c10`). Views `Finance` and `Logistics` are given by constraint
weights, `Procurement` and `Audit` by layer weights.

Steps timed with `time.perf_counter`, one cold pass each:

| key | what runs |
|---|---|
| `build` | `EventLog(...)` with `case_attributes=["company", "flow_type", "vendor"]`, `exposure_col="amount"`, `exposure_agg="max"` |
| `score` | `score(log, norm)` for all four views, including the derived attribute |
| `prioritize` | `prioritize(result, ["company", "vendor"], view="Finance", gamma=20)` |
| `drivers` | `layer_drivers(...)` on the same slice keys and `constraint_drivers(...)` for the top slice |
| `agreement` | `view_agreement(result, ["company", "vendor"], k=20, gamma=20)` |
| `diagnostics` | `right_censored(log, "Clear Invoice", window="60D", opened_by=INV)` and `event_replication(log)` |
| `validation_table` | `validation_table(result, "Finance", ["company", "vendor"], censored=..., replication=..., gamma=20)` |

The report also carries `total`, `peak_gb` (tracemalloc peak over one
pipeline pass, see below), `peak_rss_gb` (`ru_maxrss` of the whole process),
`input_gb` (size of the input frame), the sizes, `scored_share`,
`decomposition_error` (`result.check_decomposition()`), `platform` and
`versions{python, numpy, pandas, wise}`.

### How memory is measured

Timing and memory come from two separate passes. With tracemalloc active,
the pyarrow-backed string columns of pandas 3 make the build step about five
times slower (7.4 s instead of 1.5 s on the baseline machine), so the timed
pass runs without tracing and a second, untimed pass under tracemalloc gives
`peak_gb`. tracemalloc counts allocations made through the Python allocator
(numpy arrays, pandas blocks, Python objects); pyarrow string buffers and
the input frame, which exists before tracing starts, are not in `peak_gb`.
`peak_rss_gb` is the process-wide high-water mark and includes both passes,
the input and the interpreter itself, so it is an upper bound rather than
the pipeline's footprint.

## How to run

```bash
# full size, JSON report on stdout (about 10 s on a laptop)
uv run --frozen --no-sync python packages/wise-pm/benchmarks/bench_synthetic.py

# 5 % of the size (12 500 cases / 80 000 events) as a quick check
uv run --frozen --no-sync python packages/wise-pm/benchmarks/bench_synthetic.py --scale 0.05

# keep the report, skip the memory pass
uv run --frozen --no-sync python packages/wise-pm/benchmarks/bench_synthetic.py --json-out bench.json --no-tracemalloc
```

`--scale F` multiplies both the case and the event count by `F`
(`1` is the full size). `--seed` changes the generator seed (default 0).
Run on an otherwise idle machine and expect roughly 10–20 % run-to-run
variation of the step times on a laptop.

The performance gate lives in `tests/perf/test_benchmarks.py`. It is marked
`bench`, skipped by default, and when `WISE_RUN_BENCH=1` is set it runs the
script at `--scale 1` in a subprocess and asserts:

| budget | value |
|---|---|
| `build` | < 3.0 s |
| `score` | < 3.0 s |
| `prioritize` | < 0.5 s |
| `peak_gb` | < 1.5 GB |

```bash
WISE_RUN_BENCH=1 uv run --frozen --no-sync pytest -q packages/wise-pm/tests/perf
```

The architecture target for the rewrite (`05-testing-and-ci.md`) is
tighter: build < 3 s, score (10 × 4) < 2 s, peak RSS < 1 GB on the CI
runner. Tighten the budgets above once the runner is known.

## Baseline 0.1.0

Reference numbers for release 0.1.0 (code identical to branch `next` at its
start), measured 2026-09-26 on an Apple silicon macOS laptop, Python 3.13.9,
numpy 2.5.3, pandas 3.0.6, one cold pass at full size:

| step | seconds |
|---|---|
| build | 1.5 |
| score (cold) | 1.4 |
| prioritize | 0.05 |
| agreement | 0.13 |
| diagnostics | 0.03 |
| validation_table | 0.12 |
| peak (tracemalloc) | 0.46 GB |

First run of `bench_synthetic.py` as committed, same day and machine
(`--scale 1`, report keys as printed):

| step | seconds |
|---|---|
| build | 1.82 |
| score | 0.57 |
| prioritize | 0.035 |
| drivers | 0.088 |
| agreement | 0.13 |
| diagnostics | 0.031 |
| validation_table | 0.31 |
| total | 2.98 |
| peak_gb (tracemalloc) | 0.32 GB |
| peak_rss_gb (process) | 1.37 GB |
| input_gb | 0.16 GB |

At `--scale 0.05` the same run gives build 0.047 s, score 0.030 s,
prioritize 0.007 s, validation_table 0.086 s, peak 0.018 GB.

When the numbers move, say why in the pull request (data layer change,
dependency major, machine) and add a row here rather than editing the
baseline.
