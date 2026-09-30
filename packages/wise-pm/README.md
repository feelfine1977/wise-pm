# wise-pm

WISE (Weighted Insights for Evaluating Efficiency) is a Python library for
norm-based, slice-first prioritisation of process deviations from event logs.
A norm states expected execution as constraints; views weight those constraints;
WISE scores cases and ranks slices such as companies, vendors or flow types.
Layer and constraint contributions help explain where the penalties arise.

This is the MIT case-based core. Rights-controlled additions will ship as a
separate distribution (`wise-pm-actionability`, import `wise_actionability`);
see [COMMERCIAL_LICENSING.md](https://github.com/feelfine1977/wise-pm/blob/main/COMMERCIAL_LICENSING.md).

## Install and run

Requires Python 3.10+; base dependencies are NumPy and pandas. The library
lives in the `packages/wise-pm` directory of the repository, which is a
[uv](https://docs.astral.sh/uv/) workspace. A pinned classic source install is:

```bash
git clone --branch v0.1.0 https://github.com/feelfine1977/wise-pm.git
cd wise-pm
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install .
```

From the current `main` (workspace layout) install the package directory:

```bash
python -m pip install "wise-pm @ git+https://github.com/feelfine1977/wise-pm.git@main#subdirectory=packages/wise-pm"
```

Optional extras: `.[pm4py]` for XES import, `.[stats]` for SciPy-backed
Spearman/Kendall correlations in `view_agreement`, and `.[io]` for parquet
input through pyarrow. For development, clone the repository and run
`make setup` (see [CONTRIBUTING.md](https://github.com/feelfine1977/wise-pm/blob/main/CONTRIBUTING.md)).

The included synthetic example needs no dataset download:

```python
import wise

log = wise.running_p2p_log()
norm = wise.running_p2p_norm()
result = wise.score(log, norm)
backlog = wise.prioritize(result, by="company", view="Finance", gamma=1.0)
print(backlog)
result.check_decomposition()
print(wise.constraint_drivers(result, "Finance", {"company": "B"}))
```

`python packages/wise-pm/examples/quickstart.py` reproduces the paper's running
example and writes `examples/running_p2p_norm.json` next to itself. For your own data, pass a pandas
DataFrame to `wise.EventLog`, declaring case/activity/timestamp columns and
case attributes. Build a `wise.Norm` or load JSON with `wise.Norm.load(...)`.
Use `norm.describe()` and `norm.check(log)` to inspect and validate it.

## Norm JSON

A two-constraint example, loadable with `wise.Norm.loads(...)`:

```json
{
  "schema_version": 2,
  "name": "Running P2P example", "version": "1", "scoring_mode": "layer_balanced",
  "layers": [
    {"id": "completeness", "name": "Core completeness"},
    {"id": "lead_times", "name": "Working-capital lead times"}
  ],
  "views": [
    {"name": "Finance", "constraint_weights": {"c1": 0.20, "c2": 0.45}},
    {"name": "Logistics", "layer_weights": {"completeness": 0.25, "lead_times": 0.15}}
  ],
  "derived_attributes": [
    {"name": "manual_touch_count", "kind": "count_events", "where": {"column": "org:resource", "regex": "^user"}}
  ],
  "constraints": [
    {"id": "c1", "layer": "completeness", "type": "presence",
     "params": {"activity": "Record Invoice Receipt", "m": 1},
     "weight": 1.0, "applicability": {}, "description": "Require an invoice receipt"},
    {"id": "c2", "layer": "lead_times", "type": "lag",
     "params": {"a": "Record Goods Receipt", "b": "Record Invoice Receipt", "delta": 10, "width": 20, "unit": "D"},
     "weight": 1.0, "applicability": {"flow_type": ["DF1", "DF2"]},
     "description": "GR → INV within 10 days"}
  ]
}
```

## Concepts and conventions

- **Constraints and applicability.** Presence, Lag, Balance, Singularity and
  Exclusion implement the main catalogue; Precedence and Metric extend it.
  Applicability selects which cases a constraint concerns. `violations` contains
  bounded values in `[0, 1]`; `NaN` means out of scope or not evaluable, never
  satisfied. A case without a positively weighted evaluated constraint has a
  `NaN` score and is excluded from aggregates.
- **Views and weights.** The norm defaults to `layer_balanced`: renormalise
  evaluated constraints within each layer, then the layers carrying weight.
  `flat` renormalises raw constraint weights over each case's evaluated checks.
  The shipped running example uses `flat`; the BPIC19 norm uses
  `layer_balanced`. `result.contributions[view]` sums to `1 - score`.
- **Priority and exposure.** `PI = volume × max(baseline - slice_mean, 0)`;
  the default volume is scored case count and the default comparator is the
  current scored population mean. `gamma` shrinks small-slice means toward that
  comparator. Exposure changes the volume multiplier; score means remain
  unweighted. Priority is distinct from `penalty_mass`, the sum of case penalties.
- **Comparators and drivers.** A scalar `baseline=` changes the priority
  reference. Classic `layer_drivers` always contrasts with the current population
  layer profile, so it does not decompose a scalar target's gap. Use matching
  interpretations when presenting these outputs together.
- **Missingness and observation.** Lag defaults to first activation and first
  response at or after it; undefined endpoints are full violations unless the
  constraint's missingness policy says otherwise. The BPIC19 example norm uses
  different declared policies. Diagnostics flag censoring and replication;
  they cannot establish that an extract is complete.

Slice tables are indexed by grouping keys (`as_index=False` returns columns).
Parameters are recorded in `DataFrame.attrs`; preserve them when exporting.
Norm schema version 2 supports JSON round trips and `norm.fingerprint()`.
`score(..., derive=True)` can add norm-derived attributes to its input log.

## Command line and validation

```bash
wise validate packages/wise-pm/examples/running_p2p_norm.json
wise describe packages/wise-pm/examples/running_p2p_norm.json
wise score norm.json log.csv --case case --activity activity --timestamp time \
  --attr company --by company --view Finance --gamma 20 --out backlog.csv
```

`make test` runs the unit, regression, property, golden, contract and paper
tests after `make setup`. BPIC19 tests require an explicitly supplied
`WISE_BPIC19_CSV`; optional dependency tests may skip. Historical benchmark
settings live in `packages/wise-pm/examples/bpic19_norm.json` and the evaluation script.

WISE ranks measured deviations under the chosen norm, weights and comparator.
It does not discover a correct norm, prove causes, estimate guaranteed savings
or optimise an action roadmap. Applications own data preparation, norm approval,
workflow and deployment. This classic runtime uses case-based event logs.

This distribution is [MIT](https://github.com/feelfine1977/wise-pm/blob/main/packages/wise-pm/LICENSE) licensed. Prospective rights-controlled
additions live in a separate distribution under
[PolyForm Noncommercial 1.0.0](https://github.com/feelfine1977/wise-pm/blob/main/LICENSE); see
[prospective scope and commercial permission](https://github.com/feelfine1977/wise-pm/blob/main/COMMERCIAL_LICENSING.md).
Please cite the method using [CITATION.cff](https://github.com/feelfine1977/wise-pm/blob/main/CITATION.cff). See
[CHANGELOG.md](https://github.com/feelfine1977/wise-pm/blob/main/CHANGELOG.md) and the [publishing guide](https://github.com/feelfine1977/wise-pm/blob/main/docs/PUBLISHING.md)
for release history and procedure.

For additional raw-data adapters, prepared-assessment norms and native parity
checks, see the repository [dataset evaluation package](evaluation/README.md).
