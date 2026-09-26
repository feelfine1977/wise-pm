# Publishing `wise` — PyPI, conda-forge, Zenodo, and friends

This is the release runbook. It assumes the repository lives on GitHub and
that you have a PyPI account with two-factor authentication enabled
(mandatory on PyPI since 2024).

## 0. Names

| what | value | note |
|---|---|---|
| distribution name (what people `pip install`) | `wise-pm` | `wise` on PyPI is taken by an unrelated package |
| import name (what people `import`) | `wise` | may differ from the distribution name |
| GitHub repository | `github.com/feelfine1977/wise-pm` | referenced from `pyproject.toml`, `CITATION.cff` and `CONTRIBUTING.md` |

Check availability at any time:

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://pypi.org/pypi/wise-pm/json   # 404 = free
```

## 1. Build and check locally (every release)

```bash
make build                           # uv build --all-packages && twine check --strict dist/*
```

`uv build --all-packages` builds every distribution of the workspace
(`packages/wise-pm` today) into `dist/`.

Install the wheel into a *fresh* virtual environment and run the quickstart,
so you catch files missing from the wheel or an undeclared dependency:

```bash
uv venv /tmp/wise-check && uv pip install --python /tmp/wise-check/bin/python dist/wise_pm-*.whl
/tmp/wise-check/bin/python -c "import wise; print(wise.__version__)"
/tmp/wise-check/bin/python packages/wise-pm/examples/quickstart.py
```

The CI `build` job does the same on Linux, macOS and Windows; on Linux it
additionally installs the sdist and runs the packaged tests.

## 2. First upload: TestPyPI, then PyPI

TestPyPI is a sandbox with separate accounts; use it once to see the
project page rendered.

```bash
uv run twine upload --repository testpypi dist/*
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ wise-pm
```

For the very first real upload you need an API token (PyPI → Account
settings → API tokens → scope "Entire account", because the project does not
exist yet). Put it in `~/.pypirc` or pass it interactively:

```bash
uv run twine upload dist/*  # username: __token__ ; password: pypi-...
```

After the project exists, delete that token and switch to trusted publishing
(next section) so no secret is ever stored.

## 3. Automated releases with trusted publishing (recommended)

`.github/workflows/release.yml` builds and uploads when a tag `v*` is pushed.
One-time setup on PyPI:

1. Open <https://pypi.org/manage/project/wise-pm/settings/publishing/>.
2. Add a GitHub publisher: owner = your GitHub user/org, repository = the
   repo name, workflow name = `release.yml`, environment name = `pypi`.
3. In the GitHub repository: Settings → Environments → create `pypi`
   (optionally require a reviewer, which gives you a manual approval step).

Release procedure from then on:

```bash
# 1. bump the version in packages/wise-pm/src/wise/_version.py (the single source) and CITATION.cff
# 2. move CHANGELOG "Unreleased" items under the new version with today's date
# 3. check locally that everything agrees (the release workflow runs the same script):
uv run python scripts/check_release.py v0.2.0
git commit -am "Release 0.2.0"
git tag -a v0.2.0 -m "wise-pm 0.2.0"
git push && git push origin v0.2.0
# 4. .github/workflows/release.yml verifies tag == versions == CHANGELOG, runs the
#    suite, builds, publishes through trusted publishing (environment "pypi"), and
#    creates the GitHub release from the CHANGELOG section. Watch the Actions tab.
```

Versioning: semantic versioning. While the API is settling, stay on
`0.x` and treat minor bumps as potentially breaking; document every change in
`CHANGELOG.md`. The norm file format carries its own `schema_version`
independent of the package version.

## 4. Install paths users get

```bash
pip install wise-pm                      # PyPI
pip install "wise-pm[pm4py]"             # with the pm4py extra (XES import, pm4py objects)
pip install "wise-pm[stats]"             # SciPy, for Spearman / Kendall view agreement
uv add wise-pm                           # uv / pyproject-based projects
pip install "wise-pm @ git+https://github.com/feelfine1977/wise-pm.git@main#subdirectory=packages/wise-pm"   # straight from GitHub, no PyPI needed
```

`pipx install wise-pm` gives the `wise` command-line tool in an isolated environment.

## 5. conda-forge (optional, after PyPI)

conda-forge builds from the PyPI source distribution.

```bash
pip install grayskull
grayskull pypi wise-pm                   # writes wise-pm/meta.yaml
```

Fork <https://github.com/conda-forge/staged-recipes>, add the generated
recipe under `recipes/wise-pm/`, open a pull request. After the review a
feedstock is created and future PyPI releases are picked up automatically by
a bot. Pure-Python packages with numpy/pandas dependencies are usually
accepted within days.

## 6. Citable software: Zenodo DOI

1. Log in at <https://zenodo.org> with GitHub, enable the repository under
   "GitHub" in your account.
2. Publish a GitHub Release; Zenodo archives it and mints a DOI (a
   concept DOI for all versions plus one per version).
3. Put the concept DOI into `CITATION.cff` (`doi:` field) and the README
   badge. GitHub shows a "Cite this repository" button from `CITATION.cff`.

## 7. Documentation site (optional)

`mkdocs` with `mkdocstrings` renders the docstrings as an API reference:

```bash
uv sync --group docs
uv run mkdocs new . && uv run mkdocs serve
```

Host on Read the Docs (free for open source, builds on every push) or GitHub
Pages (`mkdocs gh-deploy`).

## 8. Pre-release checklist

- [ ] `make test` and `make check` green locally; CI green on the matrix (Linux 3.10–3.13 incl. floor pins, macOS, Windows)
- [ ] `make build` clean; wheel installs in a fresh venv (the CI build job does this on three OSes)
- [ ] `scripts/compat_gate.sh` green against the workbench, and the BPIC'19 reproduction run locally
- [ ] version bumped; `CHANGELOG.md` updated; `CITATION.cff` version updated
- [ ] README renders on PyPI (check on TestPyPI the first time)
- [ ] the sdist holds `src`, `scripts`, `benchmarks`, `tests` (with its golden data), README and LICENSE (`tar tzf dist/*.tar.gz`)
- [ ] `uv run python scripts/check_release.py vX.Y.Z` prints the version (tag, package, CITATION.cff and CHANGELOG agree)
