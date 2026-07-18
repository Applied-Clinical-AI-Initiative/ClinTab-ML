# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). No versioned
release has been tagged yet; everything below is grouped under Unreleased.

## [Unreleased]

### Added
- Restructured the flat script layout into a proper `clintab/` package
  (`ml.py`, `epi.py`, `stats.py`, `spline.py`, `plots.py`, `store.py`),
  imported by `app.py`, `routes.py`, and `clintab_cli.py`.
- Automated test suite (`tests/`) covering `clintab.stats`, `clintab.epi`,
  `clintab.ml`, and `clintab.spline` — 27 tests, run via `pytest`.
- `requirements-dev.txt` for test/dev dependencies.
- `Data/README.md` documenting the synthetic `sample_clinical.csv` dataset.
- `docs/install.md`, `docs/usage.md`, `docs/api.md`.
- `.github/ISSUE_TEMPLATE/bug_report.md` and `feature_request.md`,
  `.github/PULL_REQUEST_TEMPLATE.md`, and `.github/workflows/tests.yml`
  (runs the test suite on every push/PR to `main`, Python 3.10 and 3.12).
- Statement of Need and Contributing & Support sections in the README.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`.
- ORCID identifiers for all three authors in `CITATION.cff`.


### Fixed
- `clintab_cli.py`'s `test` command crashed with `KeyError: None` when
  evaluating a raw (non-dict) pickled pipeline with no recorded outcome
  column; it now skips such models with a clear message instead.
- `/api/epi/hl` checked the model before checking for an active session,
  and had no error handling around a bad/missing model name; session check
  now runs first, and the model lookup is wrapped in try/except.
- A comment in `epi.two_by_two` incorrectly said the chi-square test ran on
  the "original" table; it actually runs on the Haldane-Anscombe-corrected
  table, and the comment now says so.
- `clintab/store.py` resolved `runtime/` and `models/` relative to its own
  file location. After the package restructure moved it into `clintab/`,
  this silently redirected generated session/model data into `clintab/`
  instead of the repo root; fixed to resolve relative to the repo root
  regardless of where `store.py` lives.
- `/runtime/` and `/models/` were never excluded from git; added to
  `.gitignore`.
- CI failed on every push with `ModuleNotFoundError: No module named 'clintab'`,
  since the bare `pytest` command doesn't add the repo root to `sys.path` the
  way `python -m pytest` does. Added `pytest.ini` with `pythonpath = .` so
  `clintab` is importable regardless of how pytest is invoked.

[Unreleased]: https://github.com/Applied-Clinical-AI-Initiative/ClinTab-ML/commits/main
