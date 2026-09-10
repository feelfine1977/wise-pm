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
The 44-test baseline contract is pinned at v2, SHA-256
`f66d8dd3827871462e21176a52ef9b959876d83d73ca54f7e57a0972c204e243`.
Its one numeric-cell correction applies the existing `rtol=0, atol=1e-12`
policy; production arithmetic and all structural assertions are unchanged.
The original v1 digest is
`a47445b232fe3b3626e7bf8c788a918dfd813aba4d978164060f1b1a975dda5b`.
See [contract provenance and portability](docs/development/baseline-contract.md)
before changing this file; CI verifies the current bytes.

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

## Licensing of prospective contributions

Read [COMMERCIAL_LICENSING.md](COMMERCIAL_LICENSING.md) before submitting.
Rights-controlled additions from adoption of that policy use PolyForm
Noncommercial 1.0.0; pre-existing MIT material keeps its original permissions.
Identify pre-existing or third-party material and retain its licences and
attributions. Disclose any employer, university or other approval required for
the proposed grant; do not assume that authorship alone establishes authority.

Submission under the repository policy does not transfer ownership or grant a
separate commercial exception. Before accepting work intended for a separately
licensed commercial offering, the maintainer must establish and document the
necessary permissions from the relevant rights holder(s), using a separate
agreement where needed. This contribution guide is not a CLA and makes no
claim that commercial relicensing rights have already been secured.
