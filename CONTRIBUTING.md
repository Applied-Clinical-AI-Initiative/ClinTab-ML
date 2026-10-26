# Contributing to ClinTAB-ML

Thanks for considering a contribution. This project follows the
[Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating.

## Dev setup

```bash
git clone https://github.com/Applied-Clinical-AI-Initiative/ClinTab-ML.git
cd ClinTab-ML
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

`requirements-dev.txt` installs everything in `requirements.txt` plus
`pytest`. See [`docs/install.md`](docs/install.md) for more detail,
including running the web app or the CLI once dependencies are installed.

## Running the tests

```bash
pytest tests/ -v
```

The suite covers everything in `clintab/` except `plots.py`, plus the HTTP
routes (`tests/test_routes.py`). Any change to those modules should come with a
test covering the new behavior, and any bug fix should come with a test that
would have caught it. `clintab/plots.py`, `clintab_cli.py` and the front end
(`static/`) aren't under test, and a PR adding coverage there is welcome.

CI (`.github/workflows/tests.yml`) runs this same suite on every push and
pull request against `main`, across Python 3.10 and 3.12.

## Code style

There's no configured linter or formatter in this repo yet. Match the
existing style in the file you're editing: minimal comments (only where the
*why* isn't obvious from the code), no unused imports, and keep Flask/HTTP
concerns in `routes.py` / `app.py` out of `clintab/`. Those modules are
pure Python (pandas/numpy/sklearn in, plain dicts/DataFrames out) so they
stay importable and testable without a running server.

## Branch and PR conventions

- One logical change per branch and per PR. Don't bundle unrelated fixes
  together.
- Branch off an up-to-date `main`:
  ```bash
  git fetch origin
  git checkout main
  git merge origin/main
  git checkout -b <short-description>
  ```
- Open an issue describing the bug or feature before starting work on
  anything non-trivial, and reference it in your commits and PR
  (`Closes #123`) so it closes automatically on merge.
- Fill in the PR template: summary, linked issue, and the tests/docs/CI
  checklist.
- Keep commits scoped and messages descriptive; squash-merge is fine for
  keeping `main`'s history clean.

## Reporting bugs / requesting features

Use the issue templates under **New Issue** on GitHub, bug report or
feature request, whichever fits. Include reproduction steps for bugs, and
the problem you're trying to solve (not just the solution) for feature
requests.
