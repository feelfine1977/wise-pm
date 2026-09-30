# Dataset evaluations

This package evaluates declared process-assessment configurations on procurement, incident-management, loan-application and healthcare records. It includes 15 configurations and 44 primary views. One configuration is a three-case workbook fixture; archive copies, cleaned exports, overlapping populations and subsets are not independent replications.

The evaluation has two explicit stages:

1. Dataset adapters interpret raw records and emit bounded criterion severities, scope masks, evidence states, groups and source-linked witnesses. Their configuration files define the domain rules and illustrative thresholds.
2. The installed `wise` library evaluates the prepared matrices through genuine serialized `wise.Norm` files, `wise.score`, `wise.prioritize` and the checked explanation APIs. Native Metric constraints preserve the externally evaluated severities. A separate numerical oracle checks the arithmetic.

The carrier log used in the second stage has one assessment record per case. It does not reconstruct raw process events. Agreement at this stage establishes scoring and aggregation parity after domain adaptation, not equivalence of every raw-event constraint implementation.

## Run

The evaluation tools require Python 3.11 or later. The library itself retains its existing Python support and public API.

From the repository root, use an environment with the library and evaluation dependencies:

```sh
python -m pip install -e './packages/wise-pm[io,stats]'
python -m pip install -r packages/wise-pm/evaluation/requirements.txt
```

Import the exact supplied inputs into the local catalogue layout:

```sh
python packages/wise-pm/evaluation/data_manager.py import --source-root /path/to/Datasets
python packages/wise-pm/evaluation/data_manager.py verify
```

Run the raw adapters, their checks and native parity into a new output folder:

```sh
python packages/wise-pm/evaluation/run.py --output packages/wise-pm/evaluation/results/new-run
```

The `--data-root` option accepts an alternative directory already organized according to the catalogue. `WISE_EVALUATION_DATA_ROOT` provides the equivalent default. Source filenames from the supplied collection are preserved in catalogue metadata; destination paths are normalized.

To reuse a completed prepared-data run:

```sh
python packages/wise-pm/evaluation/run.py \
  --prepared-root packages/wise-pm/evaluation/data/prepared \
  --output packages/wise-pm/evaluation/results/native-replay
```

Use [the notebook](notebooks/Dataset_Evaluation.ipynb) for interactive inspection and execution. Its committed copy has no saved local outputs. Full runs require the documented input files; the tools do not silently download substitute data.

## Data and norms

[The catalogue](data_catalog.json) identifies 13 source families, their origins and terms, and 52 required raw/audit files totaling 1,071,618,043 bytes. Files are checked against exact SHA-256 values. A different hash requires an explicit new input/configuration identity.

| Material | Location | Version-control policy |
|---|---|---|
| Raw logs, archives and source documentation | `data/raw/` | Local and ignored |
| Complete prepared assessments, case data and witnesses | `data/prepared/` | Local and ignored |
| Adapter definitions and domain configurations | `adapters/` | Tracked |
| Loadable prepared-assessment norms | `norms/native/` | Tracked |
| Compact aggregate expectations | `expected/` | Tracked |
| Constructed contract tests | `tests/` | Tracked |
| Complete raw/native execution outputs | `results/` | Local and ignored |

The local data are available for reproduction without adding raw archives or case records to a commit. Dataset rights remain separate from the software licence. Unknown provenance is stated explicitly. The source files are not packaged in the wheel or source distribution. Existing historical BPIC19 examples and their norm remain unchanged; the ten-criterion assessment here is a separate configuration.

The [adapter index](ADAPTERS.md) links the domain configurations and explains their units, grouping and missingness. Adapter JSON files describe feature extraction, threshold ramps, scope and missingness. Native JSON files are loadable with `wise.Norm.load` and consume the prepared severity/scope attributes. They are related artifacts, not interchangeable representations of every raw-event operator. Edits to either require a new evaluation run.

## Applications and boundaries

| Configuration | Cases | Main interpretation issue |
|---|---:|---|
| BPIC19 procurement | 251,734 | PO-item exposure, not unique actions or avoidable cost |
| BPIC13 incidents | 7,554 | Recorded routing patterns and product context |
| BPIC13 open problems | 819 | Team evidence omitted from XES but retained in CSV |
| BPIC13 closed problems | 1,487 | Different field mapping; overlap with open export |
| BPIC20 Domestic | 10,500 | Journey scope differs; amount bands are not owners |
| Simulated OCEL procurement | 1,598 | Direct PO projection and shared payment events |
| ICPM procurement | 1,412 | Missing master context and snapshot commitments |
| BPIC12 applications | 13,087 | Lifecycle counts and missing resource records |
| BPIC17 applications | 31,509 | Offer identities and application segments |
| UCI incidents | 24,918 | Cumulative counters must not be summed over snapshots |
| Incident-management template | 31,588 | Production provenance remains unverified |
| AMR orders | 1,792 | Negative endpoint intervals and duplicate sensitivity |
| P2P examples | 380 | Declared disjoint union; redundant subset excluded |
| Sepsis pathways | 1,050 | Nonclinical observations; shifted dates and absent releases |
| Hospital workbook | 3 | Input-handling fixture only |

The two BPIC13 problem exports share 465 identifiers. The cleaned BPIC17 CSV is checked as a representation of its XES. P2P file 3 is a subset of the main file. These populations are not pooled into one performance estimate.

Notable evidence limitations remain visible in passing runs. The simulated OCEL source has 2,028 dangling object-object links; the direct event-to-PO projection does not traverse them. ICPM retains 32 activity-only cases. AMR has 87 negative endpoint intervals under the stated parsing and 59 duplicate rows; collapsing those duplicates changes the leading group. Sepsis has 268 pathways without a recorded release; zero observed concern in those records does not establish complete observation or clinical quality.

All norms are explicit illustrative choices. Penalties across different norms are not calibrated for comparison. Priorities and signed components explain the declared assessment, not causal responsibility, financial loss, achievable savings or approved interventions.

## Verification

The raw adapters run numerical reconstruction checks and source/constructed witnesses. Native parity compares complete case identities and scope, case scores, effective weights, components, coverage, group counts and priorities. It also exercises flat and layer-balanced scoring and gamma values 0 and 20. Preserve the scored-case mask: an unscored native case has a missing score even where component tables use zeros.

For the recorded configurations, the native/reference tolerance is `1e-8`; the measured maximum is retained in `expected/native_summary.json`. This is distinct from exact source hashes and byte-level reproduction within a pinned environment. Expected aggregates are reference evidence, not a substitute for executing the library.

Run constructed checks without the external logs:

```sh
python -m pytest packages/wise-pm/evaluation/tests -q
```

The main library's existing unit, regression, property, contract and golden tests remain separate. The evaluation does not modify their historical norms, expected results, public API or implementation.
