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
