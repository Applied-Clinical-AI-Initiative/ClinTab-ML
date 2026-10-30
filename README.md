# ClinTAB-ML

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23697356.svg)](https://doi.org/10.5281/zenodo.23697356)

A local application for building and checking clinical prediction models on
tabular data: a registry extract (ACS NSQIP, TQIP, a joint replacement
registry), an administrative dataset, or a single-centre hospital cohort saved
as a CSV. It runs on the researcher's own computer, needs no programming, and
builds in the methods conventions that prediction model studies are most often
criticised for missing.

One HTML page talks to one Flask backend. JavaScript only handles the UI and
`fetch` calls. **PapaParse** previews the CSV instantly in the browser; the raw
file still goes to Flask, which does *all* real parsing, statistics and ML with
pandas / scikit-learn / statsmodels / lifelines. Every chart is rendered
**server-side with matplotlib** and returned as a PNG (with PDF / CSV export).

---

## Statement of need

Reviews of clinical prediction models keep finding the same problems. In
orthopaedic surgery, for example, a review of 433 machine learning prognostic
studies found that 36% reported calibration, and among those published in the
leading orthopaedic journals the median study reported 45% of TRIPOD+AI items
(Kuijten et al., J Clin Epidemiol 2026). Common errors in the
analysis itself inflate reported performance: oversampling before the data are
split puts synthetic copies of patients into the test set, and tuning on the
partition used for reporting rewards chance. On 292,373 ACS NSQIP arthroplasty
patients, SMOTE applied before splitting inflated the reported AUROC of tree
models by up to 0.49, and the inflation was still about 0.4 at 100,000 patients.

General tools leave these decisions to the analyst. In a notebook, a visual
workflow tool or a script written by a chat assistant, a leaky pipeline and a
correct one are equally easy to produce and look the same in the output. Cloud
tools and chat assistants also usually can't be used with patient data or with
registry files held under a data use agreement.

ClinTAB-ML makes the decisions the literature agrees on part of the software and
records every step it takes:

- **Leakage safe by construction.** Missing value fills are learned on the
  training partition only, resampling (if chosen at all) happens inside the
  training fold, hyperparameters are tuned on the validation set, decision
  thresholds are chosen on validation data, and the test partition is locked
  after its first use.
- **The outputs reviewers ask for.** Discrimination and calibration with 95%
  CIs (calibration slope, calibration-in-the-large, O:E ratio, ICI and a
  flexible calibration curve), decision curve analysis, optional recalibration,
  adjusted odds ratios from an unpenalized logistic model, Riley minimum sample
  size, Kaplan-Meier with risk tables, Cox models with proportional hazards
  tests, competing-risk cumulative incidence with Gray's test, and the usual 2x2
  calculations with CIs.
- **Any tabular study.** A cohort builder includes or excludes rows by code
  lists (CPT, ICD-10, wildcards and ranges, across several columns), values or
  ranges, and records the inclusion flow with counts.
- **A record a reviewer can check.** A tamper-evident analysis log (with the
  data file's hash, the seed and every library version) from which the tool
  writes a methods paragraph and a TRIPOD+AI coverage table.
- **Teaching.** A leakage demonstration trains the same model correctly and in
  several common wrong ways on the user's own data, and compares what each would
  report with an untouched lockbox.
- **Local.** Nothing leaves the computer; the page makes no network requests.

Statistical outputs are tested against reference values from R (glm, pROC,
survival, cmprsk, epitools, pmsampsize) and the dcurves package. The restricted
cubic spline is the exception (see Known limitations).

---

## For clinicians: the desktop app

You don't need Python or a terminal.

1. Go to the [latest release](https://github.com/Applied-Clinical-AI-Initiative/ClinTab-ML/releases/latest)
   and download the file for your computer:
   - **Windows:** `ClinTAB-ML-<version>-windows-setup.exe`. Run it and follow the steps.
     It installs for your account only, so it doesn't need admin rights.
   - **Mac (Apple silicon, macOS 12 or later):** `ClinTAB-ML-<version>-macos-arm64.dmg`.
     Open it and drag ClinTAB-ML into Applications.
2. Open ClinTAB-ML from the Start menu, the desktop shortcut or Applications. A small
   ClinTAB-ML window appears straight away and says when it's ready (the very first start
   can take a minute), then ClinTAB-ML opens in your web browser. Keep that window open
   while you work and close it when you're done.
3. To try it out, click **Load demo data** on the Data Upload tab (a made-up hip and
   knee replacement cohort of 10,000 patients) and follow the tabs from top to bottom.

Everything runs on your own computer. Uploaded data and trained models are kept in a
local folder that the window's "Open data folder" button opens. The first time you
open it, Windows or macOS may warn that the app is from an unknown publisher; see
[packaging/README.md](packaging/README.md) for what to click, and check with your
hospital IT if software installs are restricted.

---

## Quick start (terminal)

```bash
cd ClinTab-ML
./run.sh                 # creates .venv, installs deps, serves http://127.0.0.1:5000
```

Then open **http://127.0.0.1:5000** in a browser. A ready-made synthetic cohort
is at `Data/sample_clinical.csv` to try the whole flow.

Manual / server options:

```bash
pip install -r requirements.txt
python app.py                       # local dev  (PORT, HOST, DEBUG env vars)
./run.sh prod                       # gunicorn on 127.0.0.1:5000 (HOST / PORT env vars)
# or directly:
gunicorn -w 1 -k gthread --threads 8 --timeout 600 -b 127.0.0.1:5000 "app:create_app()"
```

> Use **one worker** with threads. The training **Server-Sent-Events** stream
> and the Flask session must stay on the same process; threads handle
> concurrency fine for a research tool.

### Runs entirely on your machine
This is a **local** tool. `python app.py` and `./run.sh` (including `prod`) bind
`127.0.0.1`, so it is only reachable from your own computer. All processing,
model files and plots stay in local folders and nothing is uploaded anywhere.
The in-browser CSV preview uses a copy of PapaParse bundled in `static/vendor/`,
so the page makes no network requests at all. You do **not** need to host it on
a server. If you do set `HOST=0.0.0.0` for others on your network, read the
Security section below first.

---

## Command-line interface (no browser)

`clintab_cli.py` gives the full feature set from the terminal, built on the same
pure modules as the web app:

```bash
python clintab_cli.py demo --out demo.csv            # the made-up hip and knee cohort
python clintab_cli.py summarize --csv Data/sample_clinical.csv
python clintab_cli.py models                       # list the 15 models
python clintab_cli.py train --csv Data/sample_clinical.csv --outcome mortality_30d \
    --exclude surv_time death_event rare_lab \
    --models LogisticRegression RandomForest --scoring roc --stratify --smote --test
python clintab_cli.py test  --models <saved_name> --csv newdata.csv --out roc.png
python clintab_cli.py spline --csv Data/sample_clinical.csv \
    --predictor age --outcome mortality_30d --knots 4 --out spline.png
python clintab_cli.py epi or  --a 20 --b 80 --c 10 --d 90
python clintab_cli.py epi km  --csv Data/sample_clinical.csv --time surv_time --event death_event --group sex
python clintab_cli.py epi cox --csv Data/sample_clinical.csv --time surv_time --event death_event --cov age diabetes
python clintab_cli.py epi hl  --csv Data/sample_clinical.csv --model <saved_name>
```

Models trained from the CLI save to the same `models/` folder and are visible in
the web app (and vice-versa). Run any subcommand with `-h` for its options.

---

## Architecture

The HTTP layer (`app.py`, `routes.py`) is kept apart from the analysis code in
the `clintab/` package, which has nothing to do with HTTP and never imports
Flask. That keeps the analysis modules importable and testable without a server:

| File | Responsibility | Knows about Flask? |
|------|----------------|--------------------|
| `app.py` | Creates the Flask app and starts the server. Nothing else. | yes |
| `routes.py` | **All** HTTP endpoints + the SSE stream. Parses requests, calls the modules below, returns JSON / streams / downloads. No stats or sklearn logic. | yes |
| `clintab/ml.py` | All sklearn: the 15 models, preprocessing, validation-set grid search, SMOTE, metrics, feature importance, single prediction. | **no** |
| `clintab/validation.py` | Calibration slope, calibration-in-the-large, O:E and ICI, flexible calibration curve, DeLong and bootstrap CIs, decision curves, threshold choice, logistic recalibration. | **no** |
| `clintab/stats.py` | Column-type detection, descriptive summaries, missing value handling fitted on the training partition. | **no** |
| `clintab/cohort.py` | Cohort builder: code lists, values and ranges across columns, with the inclusion flow. | **no** |
| `clintab/spline.py` | Restricted cubic splines (patsy + statsmodels). | **no** |
| `clintab/epi.py` | Clinical epi: OR, RR, NNT/NNH with CIs, Kaplan-Meier with risk tables, Cox with proportional hazards tests, competing-risk cumulative incidence with Gray's test, Hosmer-Lemeshow. | **no** |
| `clintab/samplesize.py` | Minimum sample size for prediction models (Riley et al.), binary and continuous outcomes. | **no** |
| `clintab/teaching.py` | Leakage demonstration: the same model trained correctly and in common wrong ways, compared with an untouched lockbox. | **no** |
| `clintab/plots.py` | matplotlib → base64 PNG / PDF rendering. | **no** |
| `clintab/store.py` | Filesystem layout + session metadata (paths, JSON). | **no** |
| `clintab/analysis_log.py` | Append-only, hash-chained JSON Lines log of every action in a session. | **no** |
| `clintab/methods_text.py` | Template-based methods paragraph built from the log (no language model). | **no** |
| `clintab/tripod_report.py` | TRIPOD+AI reporting coverage built from the log. | **no** |
| `static/index.html` + `static/app.js` | The single-page UI (PapaParse is bundled in `static/vendor/`). | n/a |
| `clintab_desktop.py` + `packaging/` | Desktop launcher and the installer builds, see `packaging/README.md`. | yes |
| `paper/` | JOSS paper and the scripts behind the manuscript (no NSQIP data or results), see `paper/analysis/README.md`. | n/a |

Everything in `clintab/` is pure and importable on its own.

### Where data is saved (auto-created on first use)

```
runtime/
  sessions/<session_id>/
    raw.csv      raw upload         full.csv  cleaned full dataset
    train.csv val.csv test.csv      meta.json column types + split config
    analysis_log.jsonl              every action taken in the session
  exports/       feature-importance CSVs, etc.
models/
  <ModelType>_<outcome>_<timestamp>.pkl        trained model
  <ModelType>_<outcome>_<timestamp>.meta.json  metrics, params, features
```

The active session id is stored in the Flask session cookie ("locked into the
Flask session"); everything heavy lives on disk so it survives restarts.

---

## The seven sections (plus a teaching tab)

1. **Data Upload**: drop a CSV (PapaParse preview), or click **Load demo data** for a
   made-up hip and knee replacement cohort (`clintab/demo.py`: CPT codes, missing
   albumin, 30-day readmission, length of stay, and revision with death as a
   competing event), enough to try every tab. Flask auto-types each
   column: *binary* = 2 unique values, *categorical* = strings or <10 unique
   integers, *continuous* = numeric >10 unique values, *date* = parses as a
   date. Any column with missing values gets a handling choice, defaulted by how
   much is missing (**impute** up to 50 %, **remove** above 50 %); you can also
   choose **impute + missing indicator**, **leave missing** (the model pipeline
   imputes inside the training data), or **fill 0** (never a default, only for
   counts where a blank means none). Columns with 30 % or more missing get a note
   to consider whether missingness is informative. Fill values are learned from
   the training partition only, after the split, and then applied to validation,
   test and external data. Per-row radio buttons override any type. Set the split
   (default **70/15/15**, random or stratified) and an optional stratify column.
   If its minority class is under 30 % the app shows an imbalance note: resampling
   such as SMOTE usually worsens calibration without improving discrimination
   (van den Goorbergh et al., JAMIA 2022), so SMOTE stays off unless you tick it.
   **Build cohort (optional)** before confirming: include or exclude rows by
   code lists across one or more columns (CPT, ICD-10 with or without dots,
   wildcards such as `M16*`, ranges such as `27130-27132`), by values or by
   ranges, with a preview of the inclusion flow and a flow diagram.
   **Confirm** applies the cohort, splits, handles missing data, and saves the
   partitions, locking the config into the session.

2. **Data Summary**: auto-populates after confirm. Continuous vars: N, %
   missing, mean, median, SD, IQR, min, max. Binary/categorical: N, % missing,
   count & % per category. Top card: total N, % complete cases, date range.
   Tables export to CSV.

3. **Model Training**: pick outcome + exclusion/confounding variables; pick
   from the **15 models** (10 classifiers for binary/categorical outcomes, 5
   regressors for continuous; the task is auto-detected from the outcome).
   Confounding variables are always kept as features, even if also ticked under
   exclusion, and are recorded in the model metadata and the analysis log.
   Optionally edit the hyperparameter grid per model. Grid search is scored on
   the **validation set only** (then refit on train alone, so the reported
   validation metrics are leakage-free). Choose the tuning metric (ROC default,
   AUPRC, F1, F2, recall, …), and optionally score the search with k-fold CV on the
   training set (**CV folds**) instead of the single validation set. You can also
   upload your own train and validation CSVs. **SSE streams live progress.** Binary metrics:
   AUROC, AUPRC, sensitivity, specificity, PPV, NPV, F1, Brier. Continuous: MAE,
   RMSE, R². Every binary metric comes with a 95 % CI, plus calibration slope,
   calibration-in-the-large, O:E ratio and ICI. For binary outcomes the decision
   threshold is chosen on the validation set (Youden by default, or a target
   sensitivity or specificity, the outcome rate, or a fixed value) and saved with
   the model; optional logistic recalibration on the validation set. The plain
   **LogisticRegression** model is unpenalized and also gives a table of adjusted
   odds ratios with 95 % CIs, reference levels and events per parameter. If the
   chosen outcome had missing values filled at Confirm, training refuses and asks
   you to leave that column missing instead. Feature importance is shown as a PNG
   and exported as CSV. Models save as pickles to `models/`.

4. **Model Testing**: pick saved pickles (or upload one) and a held-out test
   set or new CSV. Get a metrics table plus matplotlib **ROC, PR, confusion,
   calibration** (binary), **one-vs-rest ROC / PR per class** (multiclass), or
   **residuals, predicted-vs-actual** (regression). Multiple binary models
   **overlay** on one ROC / PR chart. Each model uses the threshold saved at
   training unless you type another. Binary models get a flexible (lowess)
   calibration curve with a histogram of predicted risks and a plain-language
   calibration summary, and all binary models share a **decision curve** (net
   benefit against treat all and treat none). The internal test partition is
   locked after its first use; testing a new model on it asks for a reason, which
   is logged. Testing on an uploaded CSV is logged as external validation.
   Download PNG / PDF / CSV. **Single
   prediction:** pick a model, fill the auto-built input form (or upload a
   model `.pkl` + a `.json` row) → predicted outcome with probability and a
   plain-language explanation.

5. **Spline**: pick a continuous predictor + binary outcome. Flask fits a
   restricted cubic spline (patsy `cr()` + statsmodels logistic) and plots
   log-odds (Y) vs predictor (X) with knots as vertical lines and a 95 % CI.
   `cr()` adds boundary knots at the minimum and maximum of the predictor on top
   of the interior knots, so the curve is not identical to `rms::rcs()`.

6. **Clinical Epi**: OR / RR / NNT / NNH from a 2×2 table (with CIs, the risk
   difference CI, NNT CI after Altman, χ² and Fisher's exact test; the
   Haldane-Anscombe correction is applied to OR and RR only when a cell is zero);
   Kaplan-Meier with an at-risk table, median survival with CI, survival at
   chosen times and a log-rank test for any number of groups; Cox models with
   categorical covariates, a chosen reference level and a proportional hazards
   test (matching R `cox.zph`); **competing risks** (Aalen-Johansen cumulative
   incidence with CIs and Gray's test, shown against the naive 1 minus
   Kaplan-Meier curve, for example revision with death as a competing event);
   **sample size** for developing a prediction model (Riley et al., BMJ 2020),
   with a check against the loaded data; and Hosmer-Lemeshow from a saved
   binary model.

7. **Reports**: a methods paragraph and a TRIPOD+AI coverage table, both built
   from the session's analysis log. Every sentence comes from a fixed template, so
   the same log always gives the same text. The TRIPOD+AI table only says what the
   log can show was run, it does not replace the published checklist. Each log
   entry stores the SHA-256 hash of the line before it, and the Reports tab checks
   that chain, so an edited or deleted entry shows up.

**Leakage demo (teaching):** pick a binary outcome and a model, and the app
trains it the right way and in common wrong ways (SMOTE before the split,
preprocessing before the split, settings picked on the test set, scoring on
the training patients, SMOTE inside the training fold) on your own data. Each
version is scored on a lockbox of patients none of them touched, so you can see
how much each mistake inflates the AUROC it would report, and what it does to
calibration. Random forest shows the effect most clearly.

### The 15 models
*Classifiers (binary/categorical):* Logistic (unpenalized), Ridge-Logistic,
Lasso-Logistic, ElasticNet-Logistic, KNN, Decision Tree, Random Forest, Gradient
Boosting, SVM (probabilities from sigmoid calibration), Naïve Bayes.
*Regressors (continuous):* Linear, Ridge, Lasso, ElasticNet, Random-Forest
Regressor.

---

## API reference (all under `/api`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/upload` | upload CSV → detected column types + preview |
| POST | `/confirm` | apply types/missing handling, split, save, lock session |
| GET  | `/summary` | descriptive-statistics payload |
| GET  | `/session` · `/sessions` | current session metadata, list of datasets |
| POST · DELETE | `/sessions/<id>/activate` · `/sessions/<id>` · `/sessions` | switch to, delete, or clear datasets |
| GET  | `/train/columns` | outcome candidates, columns, default grids |
| POST | `/train/upload-split` | replace the train/validation split with your own CSVs |
| POST | `/train` | store training config |
| GET  | `/train/stream` | **SSE** training progress + per-model results |
| GET  | `/models` · POST `/model/upload` · GET `/model/features` · DELETE `/models/<name>` | model registry |
| POST | `/test` | evaluate model(s) → metrics + plot PNGs (overlay ROC/PR) |
| POST | `/predict-single` | single-row prediction (form or model+json upload) |
| POST | `/spline` | fit restricted cubic spline → coords + PNG |
| POST | `/epi/or` · `/epi/nnt` · `/epi/km` · `/epi/cox` · `/epi/hl` | clinical epi |
| POST | `/render/pdf` | render any chart spec as a downloadable PDF |
| GET  | `/report/methods` · `/report/tripod` | methods text and TRIPOD+AI coverage from the analysis log |
| GET  | `/report/verify-log` | check the analysis log's hash chain |

---

## Status

### Done & verified end-to-end (compute + live HTTP)
- [x] Upload → auto column typing (binary/categorical/continuous/date) →
      missing-data handling (impute / zero / include / remove) → overrides.
- [x] Confirm: random **and** stratified 70/15/15 split, class imbalance note, saved
      partitions, session lock.
- [x] Data summary card + continuous & categorical tables, CSV export.
- [x] Training: all **15** models, validation-set grid search (leakage-free
      refit), selectable tuning metric incl. F2, optional SMOTE on train only,
      SSE progress, binary + regression metric tables, feature-importance
      PNG + CSV, pickles saved with metadata sidecars.
- [x] Testing: multi-model evaluation, ROC/PR overlay, confusion/calibration/
      residual/predicted-vs-actual PNGs, single prediction (form + upload).
- [x] Spline: restricted cubic spline with knots + CI.
- [x] Clinical epi: OR/RR/NNT/NNH, Kaplan-Meier + log-rank, Cox PH,
      Hosmer-Lemeshow + calibration deciles.
- [x] Analysis log, methods text and TRIPOD+AI coverage report.
- [x] PNG / PDF / CSV export. Terminal launch via `run.sh`; gunicorn for server.

### Known limitations / next steps (good "continue-it-yourself" list)
- **Confounding variables** are kept as ordinary features and tagged in the model
  metadata and the analysis log. The unpenalized LogisticRegression model gives
  adjusted odds ratios for every feature, confounders included; there is no
  separate exposure-focused model yet.
- **Internal validation** is a single random train/validation/test split.
  Bootstrap optimism correction on all the data (preferred by Collins et al.,
  BMJ 2024) is not built yet, and the TRIPOD+AI table says so.
- **Not yet available:** multiple imputation, Fine-Gray regression (cumulative
  incidence and Gray's test are), propensity score methods, MCID calculations,
  and sample size for time-to-event outcomes.
- **Adjusted odds ratios** are fitted on the original training rows. If SMOTE is
  on, the LogisticRegression pipeline itself is fitted on resampled rows, so its
  coefficients differ from the table.
- **The CLI** runs the same analysis functions but does not write the analysis
  log, so CLI runs don't appear in the methods text or TRIPOD+AI table.
- **SMOTE timing:** the toggle at *Confirm* records a preference, but SMOTE is
  applied at *training* time to the training fold only (it needs the chosen
  outcome, which is picked later). Both the Confirm and Training screens expose
  the toggle. This is intentional and noted so it is not mistaken for a bug.
- **Preprocessing** one-hot-encodes categorical columns (including integer-coded
  ones tagged categorical at upload) and median-imputes/scales numerics inside
  each model pipeline.
- **Grid search** defaults to a single train/val split (`PredefinedSplit`),
  per the "validation set only" requirement. On small datasets that can make
  hyperparameter selection noisy; pass `cv_folds` (`--cv-folds` on the CLI)
  (the **CV folds** box in the UI) to use k-fold CV on the training set instead. Either way, the validation
  set is never touched by the search itself, only by the metrics reported
  afterward.
- **Spline CI** is the GLM linear-predictor (log-odds) confidence band.
- **Test set lock:** the first time the held-out test partition is used, it is
  locked to the models tested then. Re-testing those models (for example at a new
  threshold) is fine; testing any other model on it (Model Testing or
  Hosmer-Lemeshow) asks for a reason, which is written to the analysis log and the
  methods text. The lock survives a re-split, because the old test rows have
  already been seen. External CSVs are never locked.
- **Log integrity** is tamper evident, not tamper proof: editing, inserting or
  deleting an entry before the last one breaks the hash chain, but rewriting the
  whole chain or the last entry is not detected.
- **Security.** There is no authentication or user isolation, and loading a model
  (`/api/model/upload`, `/api/predict-single`) unpickles it, which runs code
  contained in the file. Only load model files you trust, keep the server on
  `127.0.0.1`, and put an authenticating reverse proxy in front before sharing it.
  Session cookies are signed with `CLINTAB_SECRET` if set, otherwise with a random
  key stored in `runtime/.secret_key`.

### Legacy CLI (removed)
The original `src/` pipeline (`class_tqip.py` / `select_features.py`) only
covered the old classification path and has been **replaced** by
`clintab_cli.py`, which exposes the full feature set on top of the same shared
modules as the web app. Recover the old files from git history if ever needed.

---

## Contributing & support

Bug reports and feature requests are welcome via [GitHub
Issues](https://github.com/Applied-Clinical-AI-Initiative/ClinTab-ML/issues).
Use the bug report or feature request template, whichever fits. See
`CONTRIBUTING.md` for dev setup, how to run the test suite, and branch/PR
conventions before opening a pull request. This project follows the
`CODE_OF_CONDUCT.md` in this repo; please report any violations as described
there.

For questions about using the app rather than a bug or a code change, open a
GitHub Discussion or Issue rather than emailing maintainers directly, so the
answer is visible to future users with the same question.

---

## Citing

If you use ClinTAB-ML, please cite the version you used. Every release is archived on
Zenodo, and https://doi.org/10.5281/zenodo.23697356 always points to the newest one,
with links to each version and its own DOI. For example:

> Ravindran D, Ravindran K, Abbas A. ClinTAB-ML: a clinical machine-learning web app
> for tabular registry data. Version 1.0.0. Zenodo; 2026. doi:10.5281/zenodo.23697358

GitHub's "Cite this repository" button gives the same details from `CITATION.cff`.

---

## Requirements
Python 3.10 or later (tested on 3.10 and 3.12). See `requirements.txt` (Flask, pandas, numpy, scikit-learn,
imbalanced-learn, matplotlib, scipy, statsmodels, patsy, lifelines, joblib,
gunicorn).
