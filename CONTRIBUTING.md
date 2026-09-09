# Contributing

Thanks for your interest in improving `wise`.

## Development setup

```bash
git clone https://github.com/feelfine1977/wise-pm.git   # or your fork
cd wise-pm
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install          # optional: runs ruff on every commit
pytest
```

For extension work, select `feat/actionability-ocpm-local-llm` in a separate
checkout and environment. Classic remains the default runtime. Use a full
Git clone: parity tests require base commit
`df5db50b839cc124b489a269894f5a2bfe7dc634` and fail if it is unavailable.
Keep `tests/extensions/test_baseline_contract.py` byte-for-byte unchanged
(SHA-256 `a47445b232fe3b3626e7bf8c788a918dfd813aba4d978164060f1b1a975dda5b`).

Ordinary tests use synthetic fixtures and fake providers. Leave
`WISE_BPIC19_CSV`, `WISE_OCEL2_JSON`, `WISE_OCEL2_SQLITE`,
`WISE_OLLAMA_LIVE` and `WISE_OLLAMA_MODEL` unset for the offline suite.
Run `python examples/evaluate_local_assistant.py` for the recorded boundary
scenarios and a small sensitivity example; no model is required. See the
[capabilities and limits](docs/capabilities.md) before extending the API.

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
