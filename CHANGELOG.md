# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Methods guardrails, the statistics reviewers ask for, and teaching tools.
Statistical outputs added here are tested against reference values from R (glm,
pROC, survival, cmprsk, epitools, pmsampsize) and the dcurves package.

### Fixed
- Missing value fills chosen at Confirm were computed on the whole dataset
  before the split. They are now learned on the training partition only and the
  same values are applied to validation, test and the full cleaned dataset.
- Columns with 30 to 50% missing defaulted to "fill with 0" (an NSQIP albumin
  column would have become 0 g/dL for about 44% of patients). The default is now
  training-fitted imputation up to 50% and removal above; zero fill is still
  available for counts where missing means none.
- The app recommended SMOTE whenever the minority class was under 30%. It now
  reports the imbalance and explains that resampling usually worsens
  calibration without improving discrimination (van den Goorbergh et al.,
  JAMIA 2022); SMOTE stays opt-in.
- TRIPOD+AI report: a random internal test split no longer counts as external
  validation (new internal validation item), Hosmer-Lemeshow alone no longer
  counts as calibration, sample size is only marked found when a calculation was
  run, not resampling is no longer penalised, and the missing data item checks
  whether the data actually had missing values.
- 2x2 table: risks, risk difference, NNT and the chi-square test use the
  observed counts; the 0.5 correction is applied only to the odds ratio and risk
  ratio when a cell is zero.
- Cox models accept text covariates (one-hot with a reported reference level)
  instead of failing; log-rank works for any number of groups.
- "LogisticRegression" and "RidgeLogistic" were the same model.
  LogisticRegression is now an unpenalized multivariable model; the catalogue
  no longer triggers scikit-learn deprecation warnings (SVM probabilities now
  come from CalibratedClassifierCV).
- Methods text now covers missing data handling, preprocessing, test results,
  the seed and software versions, and names models by type and outcome.

### Added
- Calibration slope, calibration-in-the-large, O:E ratio and ICI with 95% CIs,
  a flexible (lowess) calibration curve, DeLong CI for AUROC and bootstrap CIs
  for AUPRC and Brier score.
- Decision curve analysis (net benefit against treat all and treat none).
- Decision thresholds chosen on the validation set (Youden, target sensitivity
  or specificity, outcome rate, or fixed) and saved with the model; optional
  logistic recalibration on the validation set.
- Adjusted odds ratios with 95% CIs, reference levels and events per parameter
  from the unpenalized logistic model.
- Minimum sample size for prediction models (Riley et al., BMJ 2020) for binary
  and continuous outcomes, with a check against the loaded data.
- Competing risks: Aalen-Johansen cumulative incidence with CIs, Gray's test,
  and the naive 1 minus Kaplan-Meier curve for comparison.
- Kaplan-Meier risk tables, median survival with CI and survival at chosen
  times; proportional hazards tests for Cox models (matching R cox.zph).
- NNT and NNH with 95% CIs (Altman), Fisher's exact test.
- Cohort builder: include or exclude rows by code lists (CPT, ICD-10, wildcards
  and ranges, across several columns), values or ranges, with an inclusion flow
  diagram and counts recorded in the log.
- Leakage demonstration for teaching: trains the same model correctly and in
  common wrong ways on the user's data and compares each with an untouched
  lockbox.
- Test set lock: models not tested at first use need a logged reason to be
  tested on the internal test partition.
- External validation on an uploaded CSV is logged and reported as such.
- Tamper-evident analysis log (hash chain) with the data file's SHA-256, the
  seed and library versions, and a log check in the Reports tab.
- Missing indicator option at Confirm.
- CLI: `samplesize`, `teach`, `cohort` and `epi cif` subcommands, threshold
  and recalibration options for `train`.

## [1.0.0] - 2026-10-26

First tagged release.

### Added
- Restructured the flat script layout into a proper `clintab/` package
  (`ml.py`, `epi.py`, `stats.py`, `spline.py`, `plots.py`, `store.py`),
  imported by `app.py`, `routes.py`, and `clintab_cli.py`.
- Analysis log (`clintab/analysis_log.py`): an append-only JSON Lines record of
  every upload, split, model, spline and epidemiology calculation in a session.
- Methods text (`clintab/methods_text.py`) and a TRIPOD+AI coverage report
  (`clintab/tripod_report.py`), both built from the analysis log, with a new
  Reports tab in the web app.
- Dataset management in the sidebar (switch, delete, clear all), bring-your-own
  train/validation split, cancel training, drag-to-select for model lists,
  feature-outcome correlations on the Model Testing tab, and a CV folds box for
  k-fold hyperparameter search on the training set.
- Hover definitions for statistical terms in the web app.
- Desktop app for clinicians: `clintab_desktop.py` starts ClinTAB-ML on this
  computer, opens it in the browser and shows a small control window. The
  `packaging/` folder builds a Windows installer (per user, no admin rights), a
  macOS disk image and a Linux tarball with PyInstaller, and
  `.github/workflows/release.yml` builds, self-tests and attaches them to each
  GitHub release.
- `clintab.__version__`, a `/api/version` endpoint, and the version in the
  sidebar footer.
- Redesigned toast and warning dialogs (#44).
- `paper/`: the JOSS paper and `paper/analysis/`, the scripts behind the
  manuscript (validation experiments 1 to 4). No NSQIP data and no results
  derived from it are in the repository; `.gitignore` and a CI check block them.
- Automated test suite (`tests/`) covering the analysis modules, the analysis
  log, methods text, TRIPOD+AI report and the HTTP routes, run via `pytest`.
- `requirements-dev.txt`, `Data/README.md` for the synthetic `sample_clinical.csv`,
  `docs/install.md`, `docs/usage.md`, `docs/api.md`.
- `.github/ISSUE_TEMPLATE/bug_report.md` and `feature_request.md`,
  `.github/PULL_REQUEST_TEMPLATE.md`, and `.github/workflows/tests.yml`
  (runs the test suite on every push/PR to `main`, Python 3.10 and 3.12).
- Statement of Need and Contributing & Support sections in the README.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff` with ORCID identifiers
  for all three authors.

### Changed
- `./run.sh prod` and the documented gunicorn commands bind `127.0.0.1` instead
  of `0.0.0.0`. Set `HOST` to change it.
- The session cookie secret is `CLINTAB_SECRET` if set, otherwise a random key
  stored in `runtime/.secret_key`, instead of a fixed default.
- PapaParse is bundled in `static/vendor/` instead of loaded from a CDN, so the
  page works offline.
- Every `/api` error is now JSON with a matching status code.
- `DEBUG` is read as a flag (`1`, `true`, `yes`, `on`). `DEBUG=0` used to turn
  debug mode on.
- Python 3.10 or later is now stated as the minimum, matching what CI tests.
- `CLINTAB_HOME` moves all generated data (sessions, models, exports) out of
  the repository folder. The desktop app uses the user's local app data folder.
- `.gitignore` rewritten for this project. It now also blocks real cohorts placed
  in `Data/`, model pickles, the paper's working folder `paper/v2/`, local
  documents at the repository root and AI coding assistant files.

### Fixed
- The spline plot, `fit_rcs()` output and API reported fitted probabilities on a
  curve labelled log-odds. The curve and its 95% band are now on the log-odds
  scale. Spline figures made with earlier versions need regenerating.
- The spline design matrix was rank deficient (patsy's `cr()` basis already spans
  the intercept, and two more constant columns were added). The basis is now
  centred, which gives the same AIC and the same curve to within 1e-13 of an
  independent fit, with identifiable coefficients.
- "Clear all data" left the upload, type and split panels of the old dataset on
  screen (#44).
- A continuous outcome such as length of stay could be classed as multiclass
  because only the first 200 rows were read (#43).
- A missing outcome value was read as an extra "nan" class, which broke metrics
  for binary outcomes with missing values. Missing outcomes are now dropped.
- Validation and test labels were coded independently of the training labels, so
  a set holding only one class had its labels flipped. They now use the classes
  the model was trained with.
- The outcome type was guessed from the first 200 training rows, so a rare binary
  outcome could be treated as multiclass. It now uses the whole column, and a
  text outcome with many levels is multiclass instead of continuous.
- One-vs-rest ROC and PR curves were misaligned with the probability columns when
  a test set was missing a class.
- Methods text said hyperparameters were tuned for runs that used the defaults,
  and ignored a custom train/validation split. It also wrote "A SVM".
- A stratified split with no stratify column ran as a random split but was
  logged as stratified. It is now an error, and a fallback to a random split is
  only reported when stratification was actually requested.
- Confounding variables had no effect. They are now always kept as features and
  recorded in the model metadata and the analysis log.
- A long outcome name cut the timestamp off the saved model name, so a later run
  overwrote the earlier model.
- `/api/test`, `/api/model/features`, `/api/predict-single` and `/api/epi/hl`
  accepted model names with path characters, and crashed on a missing name,
  non-form requests, or an upload source with no file. One failing model no longer
  fails the whole test request.
- A CSV that failed to parse left a half-created session behind.
- NNT/NNH was computed from a risk difference already rounded to 4 decimals.
  Negative or non-finite 2x2 counts are rejected.
- Two elements on the page shared the id `confirmMsg`, so the delete confirmation
  dialogs showed no message.
- Column names, filenames and category labels were inserted into the page as
  HTML. A header such as `<img onerror=...>` ran as script, and a quote in a
  column name broke the type table. They are now escaped.
- Feature-importance CSV downloads broke on names containing commas or
  apostrophes.
- The `.term` hover definitions never displayed (no CSS).
- The "Applied Clinical AI Initiative" footer said "Institute".
- A half-written last line in `analysis_log.jsonl` made the whole log unreadable.
- `clintab_cli.py`: `train --test` found its models by filename prefix, and
  `epi hl` crashed on a raw pickled pipeline.
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

### Security
- Loading a model unpickles it, which runs code in the file. This is now stated
  in the README, the server no longer listens on all interfaces by default, and
  model names are sanitised before they are turned into paths.

[1.0.0]: https://github.com/Applied-Clinical-AI-Initiative/ClinTab-ML/releases/tag/v1.0.0
