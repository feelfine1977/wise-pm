# wise-pm — actionability development branch

WISE (Weighted Insights for Evaluating Efficiency) scores event-log cases
against an explicit norm and ranks slices by their deviation from a comparator.
This branch adds experimental evidence, explanations, object-centric assessment,
local assistance boundaries and evaluation helpers. It is a method library;
project workflows, norm approval and roadmap management belong to applications.

**Classic remains the default.** Ordinary `wise.score(log, norm)` and
`wise.prioritize(...)` calls keep the classic case workflow. Evidence capture is
opt-in; object-centric, assistance and evaluation modules use explicit imports.
Installing this branch does not activate a model or an application integration.

## Install

Requires Python 3.10+ and NumPy/pandas. The distribution is `wise-pm`, the import
is `wise`, and this source tree builds as **`0.1.1.dev0`**. It is an unreleased
development identity, distinct from classic `0.1.0`; it does not identify an
individual commit or local diff. Record those separately for reproducible runs.

Use a separate environment for the extension: classic and extension share the
same distribution and import names and cannot coexist in one interpreter.

```bash
git clone --branch feat/actionability-ocpm-local-llm https://github.com/feelfine1977/wise-pm.git wise-pm-actionability
cd wise-pm-actionability
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,stats]"
python -m pytest -q -ra
```

Use `python -m pip install .` for a non-editable install without developer tools.
`[pm4py]` adds XES import; `[stats]` adds SciPy for Spearman/Kendall correlations.
`wise.llm` uses standard-library HTTP transport and validation: **no `llm` extra
or model installation is required**. The clone above retrieves committed work;
uncommitted local changes are present only when building from that checkout.
Full Git history is required to run the source parity tests. They fail if the
pinned classic reference is absent; source archives alone cannot prove parity.

For classic use, install the `v0.1.0` tag in its own environment. Do not substitute
a moving feature branch for a pinned classic dependency. See the
[classic README](https://github.com/feelfine1977/wise-pm/blob/main/README.md).

## Start with the synthetic example

```python
import wise

log = wise.running_p2p_log()
norm = wise.running_p2p_norm()
result = wise.score(log, norm)
backlog = wise.prioritize(result, by="company", view="Finance", gamma=1.0)
print(backlog)
result.check_decomposition()
```

No external dataset is needed. For your own data, construct `wise.EventLog`
from a pandas DataFrame, declaring the case, activity and timestamp columns,
case attributes and optional exposure column. Load a norm with
`wise.Norm.load("norm.json")`; inspect it with `norm.describe()` and validate
its activities/attributes with `norm.check(log)`.

Evidence and a shared comparator are explicit additions:

```python
from wise.explain import BaselineSpec, explain_priority, render_explanation

result = wise.score(log, norm, evidence="full")
baseline = BaselineSpec.current_population(baseline_id="current")
packet = explain_priority(
    result,
    "company",
    "B",
    view="Finance",
    gamma=1.0,
    baseline_spec=baseline,
    evidence=result.evidence,
)
print(render_explanation(packet, fmt="text"))
```

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

## Capabilities and concepts

| Surface | What it provides | Status |
|---|---|---|
| `wise` | Norms, bounded violations, scores, priorities, drivers and diagnostics | Classic workflow |
| `wise.evidence` | Run manifests, reasoned records, bounded witnesses, fitted calibration | Experimental |
| `wise.explain` | Shared comparator specification and deterministic signed decomposition | Experimental |
| `wise.oc` | OCEL adapters, bounded assessment units, three object-check families, allocation accounting | Experimental |
| `wise.llm` | Read-only tool gateway, access policy, local transport, approved-document retrieval and norm drafts | Experimental |
| `wise.evaluation.ocel` | Synthetic native-versus-projection comparisons | Experimental |
| `wise.evaluation.llm` | Recorded refusal and permitted-response scenarios | Implemented, offline |
| `wise.evaluation.sensitivity` | Sampling, parameter and construction rank variation | Implemented, experimental |

A norm groups constraints into layers; views supply their weights. `NaN` means
not evaluated or unscored, never satisfied. A score's penalty decomposes into
layer contributions. Priority compares a slice mean with an explicit reference
and scales the positive gap by volume; it is not a savings estimate. Read the
[concepts guide](docs/concepts.md) for applicability, weighting, exposure,
comparator choice and decomposition.

The former S6 library work is implemented; a live-model quality benchmark,
field study and research-gated methods are not established by it. The
[capability matrix and current limitations](docs/capabilities.md) describe the
remaining API gaps. Earlier baseline measurements remain in the
[historical validation record](docs/development/extension-checkpoint.md).

## Offline examples and verification

```bash
python examples/quickstart.py
python examples/evidence_review.py
python examples/explanation_review.py
python examples/oc_invoice_review.py
python examples/local_review.py
python examples/review_norm_draft.py
python examples/evaluate_local_assistant.py
```

The assistance examples use a fake provider. The evaluation example reports nine
recorded scenarios and one explicitly skipped model scenario, then deliberately
changes an expected outcome to demonstrate failure detection. It also shows rank
sensitivity on the synthetic example. It does not measure a model's quality.
Examples may write local output files; `quickstart.py` writes the shipped norm.

Keep `WISE_BPIC19_CSV`, `WISE_OCEL2_JSON`, `WISE_OCEL2_SQLITE`,
`WISE_OLLAMA_LIVE` and `WISE_OLLAMA_MODEL` unset for offline source tests.
CI fetches full history, checks the [pinned baseline contract](docs/development/baseline-contract.md), runs the offline
suite and recorded scenarios, and checks lint, types and wheel metadata.
See [CONTRIBUTING.md](CONTRIBUTING.md) for development commands.

## Limits and references

Coverage is not confidence; absent events require a stated observation policy.
Object assessment uses bounded contexts and can be incomplete. Allocation
conservation holds within one report. Local transport is not server security,
and a schema-valid draft still requires substantive review. No causal benefit,
optimal roadmap, live-model usefulness or production deployment is certified.

Read the [evidence semantics](docs/semantics/evidence.md),
[explanation semantics](docs/semantics/explanation.md),
[object-centric semantics](docs/semantics/object_centric.md) and
[assistance boundary](docs/security/local-assistant.md).
Prospective rights-controlled additions use [PolyForm Noncommercial 1.0.0](LICENSE).
Pre-policy material, including the publicly offered actionability baseline,
remains available under [MIT](LICENSE-MIT). See
[prospective scope and commercial permission](COMMERCIAL_LICENSING.md)
and [CITATION.cff](CITATION.cff).
