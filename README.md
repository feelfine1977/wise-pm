# wise-pm

WISE (Weighted Insights for Evaluating Efficiency) is a Python method library
for norm-based, slice-first prioritisation of process deviations from event
logs (Jessen, Fahland, Zerbato: *WISE: Actionable Norm-Based Scoring for
Process Mining*).

This repository is a workspace with one released distribution today:

| Distribution | Import | What it holds | Licence |
|---|---|---|---|
| [`packages/wise-pm`](packages/wise-pm/README.md) | `wise` | norms, bounded violations, case scores, priorities, drivers, diagnostics, CLI | MIT |

The library README under `packages/wise-pm/` is the user documentation and
the PyPI page. `CHANGELOG.md` and `CITATION.cff` are shared by the workspace;
the examples live in `packages/wise-pm/examples/`.

## Working on the code

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/feelfine1977/wise-pm.git && cd wise-pm
make setup      # uv sync --all-packages --group dev; pre-commit install
make test       # fast suite
make check      # ruff, format, mypy, import-linter, hooks
```

`make help` lists the targets; the full developer loop, test tiers and
release procedure are described in [CONTRIBUTING.md](CONTRIBUTING.md) and
[docs/PUBLISHING.md](docs/PUBLISHING.md).
