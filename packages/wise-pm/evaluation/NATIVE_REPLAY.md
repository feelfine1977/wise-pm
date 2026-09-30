# Native WISE replay of prepared assessments

This evaluation bridge runs the installed WISE library against the frozen case-by-criterion assessments from the dataset adapters. It uses the public `wise.Metric`, `wise.Norm`, `wise.EventLog`, `wise.score` and `wise.prioritize` APIs. It contains no second scoring implementation and does not install or replace the library.

The executed run covers **15 configurations and 44 primary views**. This includes the ten-criterion BPIC 2019 evaluation, all transfer applications and the three-case hospital workbook fixture. All primary scores and group priorities agree with the stored results within a maximum absolute residual of **2.05×10⁻¹¹**. Additional differential checks cover both flat and layer-balanced scoring and gamma values 0 and 20. The workbook and overlapping BPIC 2013 exports are retained as labelled applications, not counted as independent empirical replications.

## What this verifies

Each bounded severity becomes a numeric case attribute consumed by `Metric(threshold=0, width=1)`. Its explicit scope flag becomes the `NormConstraint.applicability` condition. Missing severity inside scope remains unevaluable. Finite severity outside scope is rejected before native evaluation. Each view uses its declared raw **criterion** weights; layer weights are not silently reinterpreted as target layer totals.

The bridge serializes each genuine `Norm`, reloads it with `Norm.load`, calls native scoring, and compares:

- Exact scope and severity missingness.
- Case scores and weighted evidence coverage against stored adapter outputs.
- Group support, means, volume, gaps and priorities against stored outputs.
- Stored effective weights and criterion contributions where the adapters exported them.
- Native layer decomposition for every primary view.
- Native `wise.layer_drivers` deltas multiplied by native group support, preserving negative offsets. Their row sum is clipped only after summation and checked against primary priority; individual signed layer values are also compared with the explicit oracle when supplied.
- With the explicit differential oracle, effective weights, criterion contributions, scores and priorities in both scoring modes and at gamma 0/20.

Native `ScoreResult.effective_weights` and `penalties` expose zeros for unscored cases; the oracle exposes missing rows. The differential check masks native unscored rows using the native score mask before comparing, without changing a native score or including an unscored case in priority calculations. Coverage is an evaluation diagnostic calculated from the public scope/evaluability matrices and public norm weight vector, because it is not a native `ScoreResult` field.

The carrier `EventLog` has exactly one labelled **prepared assessment** record per case and a fixed placeholder timestamp. This is a transport for case attributes. It is not a reconstruction of the source event sequence, and `result.trace()` on that carrier does not return the original process trace. Original evidence remains in the adapter's source-linked witnesses and raw inputs.

Consequently, this run establishes native scoring/prioritisation parity for the supplied assessments. It does **not** establish equivalence between every raw adapter rule and native `Lag`, `Presence`, `Balance` or other event operators. The identity `Metric` norms preserve prepared severities; the separately versioned adapter configurations own their raw-event meaning and thresholds.

## Reproduce

The evaluation requires **Python 3.11 or newer**. Install the local `wise-pm` package and the evaluation dependencies NumPy, pandas and pyarrow. No private import or source patch is required. Either use an installed package or set `PYTHONPATH` to the intended checkout's `packages/wise-pm/src` directory.

```sh
python run_native_parity.py \
  --evaluation-root /path/to/evaluation \
  --adapter-root /path/to/evaluation/adapters \
  --oracle /path/to/evaluation/oracle/code/wise_reference.py \
  --output /path/to/evaluation/results/native-replay \
  --norm-output /path/to/evaluation/norms/native
```

Prepared input resolution accepts `<root>/<family>/results`, `<root>/<family>`, `<root>/data/prepared/<family>/results` and `<root>/data/prepared/<family>`. Configurations are read from executed result files where available; the loan-adapter fallback is `--adapter-root/bpic2012_2017/illustrative_norm.json`. There are no machine-specific defaults. `--dataset` can be repeated for a selected subset. The default requires all 15 configurations and fails visibly if one is absent.

`--oracle` is optional. Omitting it still checks the native execution against frozen stored results and native decomposition, but skips additional differential mode/shrinkage checks. A supplied oracle is an explicit evaluation dependency, not an implementation imported by the public WISE package. Keep it in the evaluation oracle directory only.

The result manifest records native library module hashes, bridge code hashes, hashes of the consumed prepared files, norm fingerprints and output hashes. It records relative source paths, without machine-specific absolute paths. Case-application totals are not unique-person or independent-dataset totals.

## Loading one exported native norm

Each dataset output directory contains `assessment_features.parquet` and `native_norm.json`. To replay a prepared case table directly:

```python
import pandas as pd
import wise
from native_assessments import carrier_log

cases = pd.read_parquet("assessment_features.parquet")
norm = wise.Norm.load("native_norm.json")
result = wise.score(carrier_log(cases), norm, derive=False)
backlog = wise.prioritize(result, by="assessment_group", view="Balanced")
```

The optional `--norm-output` also writes one portable JSON per configuration for a tracked norm catalogue. Keep raw data, prepared features and generated result tables in the evaluation's local data/results directories according to their source terms. The bridge does not need original event data to replay scoring, but adapter reproduction and evidence inspection do.

## Tests and executed evidence

Run:

```sh
python -m pytest tests/test_native_assessments.py
```

Thirteen contract test cases passed. They cover scope versus unevaluable evidence, a wholly unscored case, exact norm serialization, a partial-layer case where the two scoring modes differ, zero criterion weights, misalignment, invalid severity and negative signed layer offsets. Parity checks require unique, exactly matching row/column identities before alignment; extra expected cases cannot disappear during reindexing. Native group keys retain their types, and file-identifier string conversion rejects collisions such as missing NaN versus literal `"nan"`. These are behavioral checks of the native transport and comparison boundary.

The full result manifest reports 15 passed configurations, 44 primary views and 88 differential mode/view combinations. Each differential combination additionally checks both priority shrinkage settings. Per-configuration `verification.json` files retain individual residuals rather than only a blanket pass flag. The largest residual occurs in BPIC 2019 group-level arithmetic; it is well below the declared absolute comparison tolerance `1e-8`.
