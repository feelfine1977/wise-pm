# Contributing

Thanks for your interest in improving `wise`.

## Development setup

```bash
git clone https://github.com/feelfine1977/wise-pm.git   # or your fork
cd wise-lib
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install          # optional: runs ruff on every commit
pytest
```

## Ground rules

- The paper's equations are the specification. A change to scoring or
  prioritisation semantics needs a test that cites the paper section it
  implements and an entry in `CHANGELOG.md`.
- Keep the core dependency-light: `numpy` and `pandas` only. Integrations
  (pm4py, plotting, UIs) go behind optional extras or into separate packages.
- Run `ruff check . && ruff format . && mypy src/wise && pytest` before opening
  a pull request.
- Update `CHANGELOG.md` under *Unreleased*.

## Releasing

See `docs/PUBLISHING.md`.
