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
python -m pip install --upgrade build twine
rm -rf dist/
python -m build                      # creates dist/wise_pm-X.Y.Z.tar.gz and .whl
twine check dist/*                   # validates metadata and README rendering
```

Install the wheel into a *fresh* virtual environment and run the quickstart,
so you catch files missing from the wheel or an undeclared dependency:

```bash
python -m venv /tmp/wise-check && /tmp/wise-check/bin/pip install dist/*.whl
/tmp/wise-check/bin/python -c "import wise; print(wise.__version__)"
/tmp/wise-check/bin/python examples/quickstart.py
```

## 2. First upload: TestPyPI, then PyPI

TestPyPI is a sandbox with separate accounts; use it once to see the
project page rendered.

```bash
twine upload --repository testpypi dist/*
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ wise-pm
```

For the very first real upload you need an API token (PyPI → Account
settings → API tokens → scope "Entire account", because the project does not
exist yet). Put it in `~/.pypirc` or pass it interactively:

```bash
twine upload dist/*         # username: __token__ ; password: pypi-...
```

After the project exists, delete that token and switch to trusted publishing
(next section) so no secret is ever stored.

## 3. Automated releases with trusted publishing (recommended)

`.github/workflows/publish.yml` builds and uploads on every GitHub Release.
One-time setup on PyPI:

1. Open <https://pypi.org/manage/project/wise-pm/settings/publishing/>.
2. Add a GitHub publisher: owner = your GitHub user/org, repository = the
   repo name, workflow name = `publish.yml`, environment name = `pypi`.
3. In the GitHub repository: Settings → Environments → create `pypi`
   (optionally require a reviewer, which gives you a manual approval step).

Release procedure from then on:

```bash
# 1. bump the version in src/wise/_version.py (the single source) and CITATION.cff
# 2. move CHANGELOG "Unreleased" items under the new version with today's date
git commit -am "Release 0.2.0"
git tag -a v0.2.0 -m "wise 0.2.0"
git push && git push --tags
# 3. GitHub → Releases → "Draft a new release" → choose tag v0.2.0 → publish
#    The workflow builds, then uploads to PyPI. Watch the Actions tab.
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
pip install git+https://github.com/feelfine1977/wise-pm.git@main   # straight from GitHub, no PyPI needed
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
pip install mkdocs mkdocs-material mkdocstrings[python]
mkdocs new . && mkdocs serve
```

Host on Read the Docs (free for open source, builds on every push) or GitHub
Pages (`mkdocs gh-deploy`).

## 8. Pre-release checklist

- [ ] `pytest` green on the CI matrix (Linux/macOS/Windows × 3.10–3.13)
- [ ] `ruff check . && ruff format --check . && mypy src/wise` clean
- [ ] `python -m build && twine check dist/*` clean; wheel installs in a fresh venv
- [ ] version bumped; `CHANGELOG.md` updated; `CITATION.cff` version updated
- [ ] README renders on PyPI (check on TestPyPI the first time)
- [ ] no data files or notebooks in the sdist (`tar tzf dist/*.tar.gz`)
- [ ] tag `vX.Y.Z` matches the package version
