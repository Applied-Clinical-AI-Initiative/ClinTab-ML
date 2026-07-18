# ClinTAB-ML

A single-page **clinical machine-learning web app** for tabular registry data
(TQIP, NSQIP, and any CSV cohort). It is the nexus for an ML study: upload a
CSV, profile it, train and compare models, test them with full diagnostic
plots, fit splines, and run standard clinical-epidemiology calculators — all
from one page, all processed in Python.

One HTML page talks to one Flask backend. JavaScript only handles the UI and
`fetch` calls. **PapaParse** previews the CSV instantly in the browser; the raw
file still goes to Flask, which does *all* real parsing, statistics and ML with
pandas / scikit-learn. Every chart is rendered **server-side with matplotlib**
and returned as a PNG (with PDF / CSV export) — no SVG, no canvas, no Chart.js.

---

## Statement of need

Clinical researchers working with registry data (TQIP, NSQIP, single-center
cohorts) typically stitch together an ML study from several disconnected
tools: a notebook for data profiling, hand-written scikit-learn code for
model comparison, a separate online calculator for odds ratios and NNT, and
another package entirely for Kaplan-Meier or Cox survival analysis. Each
hand-off is a place for the leakage-free validation and correction
conventions clinical work depends on (e.g. scoring hyperparameter search on
a held-out validation set rather than the training fold, or applying a
Haldane-Anscombe correction to a 2x2 table with a zero cell) to quietly get
skipped. Several of those disconnected tools (online calculators, cloud
AutoML platforms) also mean patient-derived data leaving the researcher's
machine and landing on a third party's server, which is a real problem for
registry data that is rarely fully de-identified.

ClinTAB-ML exists to remove those hand-offs. It is a single local tool that
takes a research team from a raw CSV through column profiling, model
training and tuning across 15 algorithms, full diagnostic evaluation
(ROC/PR/calibration/Hosmer-Lemeshow), restricted cubic splines, and the
standard clinical-epidemiology calculators, all with the conventions
clinical prediction-model work expects built in by default rather than
left to whoever wrote that day's notebook, and all running on the
researcher's own machine so the data never has to leave it (see "Runs
entirely on your machine" below). It targets researchers who want to move
from cohort to trained, evaluated model without switching tools, sending
data off-machine, or re-deriving the same statistical corrections each
time, and who would otherwise be choosing between general-purpose,
cloud-hosted AutoML tools (which don't know what a Hosmer-Lemeshow test is,
and don't keep the data local) and writing the whole pipeline by hand.

---

## Quick start (terminal)

```bash
cd ClinTAB-ML
./run.sh                 # creates .venv, installs deps, serves http://127.0.0.1:5000
```

Then open **http://127.0.0.1:5000** in a browser. A ready-made synthetic cohort
is at `Data/sample_clinical.csv` to try the whole flow.

Manual / server options:

```bash
pip install -r requirements.txt
python app.py                       # local dev  (PORT, HOST, DEBUG env vars)
./run.sh prod                       # gunicorn on 0.0.0.0:5000 (server deploy)
# or directly:
gunicorn -w 1 -k gthread --threads 8 --timeout 600 -b 0.0.0.0:5000 "app:create_app()"
```

> Use **one worker** with threads. The training **Server-Sent-Events** stream
> and the Flask session must stay on the same process; threads handle
> concurrency fine for a research tool.

### Runs entirely on your machine
This is a **local** tool — `python app.py` / `./run.sh` binds `127.0.0.1`, so it
is only reachable from your own computer. All processing, model files and plots
stay in local folders; nothing is uploaded anywhere. You do **not** need to host
it on a server. The gunicorn / `0.0.0.0` instructions above are optional, for the
case where you later want others on your network to reach it. (One online
dependency: the in-browser CSV preview loads PapaParse from a CDN the first
time; the actual data processing is fully offline.)

---

## Command-line interface (no browser)

`clintab_cli.py` gives the full feature set from the terminal, built on the same
pure modules as the web app:

```bash
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

## Architecture — why the files are split this way

You could start with just `app.py` + `index.html` to prove the upload works.
As it grows the HTTP layer gets long, so the ML is pulled out (it has nothing to
do with HTTP); models start saving everywhere, so a `models/` folder is created;
static files get their own place. The result:

| File | Responsibility | Knows about Flask? |
|------|----------------|--------------------|
| `app.py` | Creates the Flask app and starts the server. Nothing else. | yes |
| `routes.py` | **All** HTTP endpoints + the SSE stream. Parses requests, calls the modules below, returns JSON / streams / downloads. No stats or sklearn logic. | yes |
| `ml.py` | All sklearn: the 15 models, preprocessing, validation-set grid search, SMOTE, metrics, feature importance, single prediction. | **no** |
| `stats.py` | Column-type detection + descriptive summaries. | **no** |
| `spline.py` | Restricted cubic splines (patsy + statsmodels). | **no** |
| `epi.py` | Clinical epi: OR, RR, NNT/NNH, Kaplan–Meier, Cox, Hosmer–Lemeshow (lifelines/scipy). | **no** |
| `plots.py` | matplotlib → base64 PNG / PDF rendering. *(Pulled out of routes so HTTP and drawing stay separate — a small addition to the original file list.)* | **no** |
| `store.py` | Filesystem layout + session metadata (paths, JSON). *(Infra helper, also an addition.)* | **no** |
| `static/index.html` + `static/app.js` | The single-page UI. | — |

`ml.py`, `stats.py`, `spline.py`, `epi.py`, `plots.py`, `store.py` are all pure
and importable on their own — easy to test without a server.

### Where data is saved (auto-created on first use)

```
runtime/
  sessions/<session_id>/
    raw.csv      raw upload         full.csv  cleaned full dataset
    train.csv val.csv test.csv      meta.json column types + split config
  exports/       feature-importance CSVs, etc.
models/
  <ModelType>_<outcome>_<timestamp>.pkl        trained model
  <ModelType>_<outcome>_<timestamp>.meta.json  metrics, params, features
```

The active session id is stored in the Flask session cookie ("locked into the
Flask session"); everything heavy lives on disk so it survives restarts.

---

## The six sections

1. **Data Upload** — drop a CSV (PapaParse preview). Flask auto-types each
   column: *binary* = 2 unique values, *categorical* = strings or <10 unique
   integers, *continuous* = numeric >10 unique values, *date* = parses as a
   date. Columns >50 % missing are flagged → choose **include / fill 0 /
   remove**. Per-row radio buttons override any type. Set the split (default
   **70/15/15**, random or stratified), pick a stratify/SMOTE column; if its
   minority class <30 % a SMOTE toggle appears. **Confirm** cleans, splits, and
   saves the partitions, locking the config into the session.

2. **Data Summary** — auto-populates after confirm. Continuous vars: N, %
   missing, mean, median, SD, IQR, min, max. Binary/categorical: N, % missing,
   count & % per category. Top card: total N, % complete cases, date range.
   Tables export to CSV.

3. **Model Training** — pick outcome + exclusion/confounding variables; pick
   from the **15 models** (10 classifiers for binary/categorical outcomes, 5
   regressors for continuous — the task is auto-detected from the outcome).
   Optionally edit the hyperparameter grid per model. Grid search is scored on
   the **validation set only** (then refit on train alone, so the reported
   validation metrics are leakage-free). Choose the tuning metric (ROC default,
   AUPRC, F1, F2, recall, …). **SSE streams live progress.** Binary metrics:
   AUROC, AUPRC, sensitivity, specificity, PPV, NPV, F1, Brier. Continuous: MAE,
   RMSE, R². Feature importance is shown as a PNG and exported as CSV. Models
   save as pickles to `models/`.

4. **Model Testing** — pick saved pickles (or upload one) and a held-out test
   set or new CSV. Get a metrics table plus matplotlib **ROC, PR, confusion,
   calibration** (binary), **one-vs-rest ROC / PR per class** (multiclass), or
   **residuals, predicted-vs-actual** (regression). Multiple binary models
   **overlay** on one ROC / PR chart. Download PNG / CSV. **Single
   prediction:** pick a model, fill the auto-built input form (or upload a
   model `.pkl` + a `.json` row) → predicted outcome with probability and a
   plain-language explanation.

5. **Spline** — pick a continuous predictor + binary outcome. Flask fits a
   restricted cubic spline (patsy `cr()` + statsmodels logistic) and plots
   log-odds (Y) vs predictor (X) with knots as vertical lines and a 95 % CI.

6. **Clinical Epi** — OR / RR / NNT / NNH from a 2×2 table (with CIs, χ², and a
   Haldane–Anscombe correction for zero cells); Kaplan–Meier (with log-rank) and
   Cox proportional hazards via **lifelines**; Hosmer–Lemeshow goodness-of-fit
   with a calibration-by-decile chart from a saved binary model.

### The 15 models
*Classifiers (binary/categorical):* Logistic, Ridge-Logistic, Lasso-Logistic,
ElasticNet-Logistic, KNN, Decision Tree, Random Forest, Gradient Boosting, SVM,
Naïve Bayes.
*Regressors (continuous):* Linear, Ridge, Lasso, ElasticNet, Random-Forest
Regressor.

---

## API reference (all under `/api`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/upload` | upload CSV → detected column types + preview |
| POST | `/confirm` | apply types/missing handling, split, save, lock session |
| GET  | `/summary` | descriptive-statistics payload |
| GET  | `/session` | current session metadata |
| GET  | `/train/columns` | outcome candidates, columns, default grids |
| POST | `/train` | store training config |
| GET  | `/train/stream` | **SSE** training progress + per-model results |
| GET  | `/models` · POST `/model/upload` · GET `/model/features` | model registry |
| POST | `/test` | evaluate model(s) → metrics + plot PNGs (overlay ROC/PR) |
| POST | `/predict-single` | single-row prediction (form or model+json upload) |
| POST | `/spline` | fit restricted cubic spline → coords + PNG |
| POST | `/epi/or` · `/epi/nnt` · `/epi/km` · `/epi/cox` · `/epi/hl` | clinical epi |
| POST | `/render/pdf` | render any chart spec as a downloadable PDF |

---

## Status

### Done & verified end-to-end (compute + live HTTP)
- [x] Project structure exactly as specified (+ `plots.py`, `store.py` helpers).
- [x] Upload → auto column typing (binary/categorical/continuous/date) →
      >50 %-missing flagging with include/zero/remove → overrides.
- [x] Confirm: random **and** stratified 70/15/15 split, SMOTE hint, saved
      partitions, session lock.
- [x] Data summary card + continuous & categorical tables, CSV export.
- [x] Training: all **15** models, validation-set grid search (leakage-free
      refit), selectable tuning metric incl. F2, optional SMOTE on train only,
      SSE progress, binary + regression metric tables, feature-importance
      PNG + CSV, pickles saved with metadata sidecars.
- [x] Testing: multi-model evaluation, ROC/PR overlay, confusion/calibration/
      residual/predicted-vs-actual PNGs, single prediction (form + upload).
- [x] Spline: restricted cubic spline with knots + CI.
- [x] Clinical epi: OR/RR/NNT/NNH, Kaplan–Meier + log-rank, Cox PH, Hosmer–
      Lemeshow + calibration deciles.
- [x] PNG / PDF / CSV export. Terminal launch via `run.sh`; gunicorn for server.

### Known limitations / next steps (good "continue-it-yourself" list)
- **Confounding variables** are currently kept as ordinary features and tagged;
  they are not yet given special statistical treatment (e.g. forced-in adjusted
  models). The UI captures them and they ride along in training.
- **SMOTE timing:** the toggle at *Confirm* records a preference, but SMOTE is
  applied at *training* time to the training fold only (it needs the chosen
  outcome, which is picked later). Both the Confirm and Training screens expose
  the toggle. This is intentional and noted so it is not mistaken for a bug.
- **Preprocessing** one-hot-encodes `object` columns and median-imputes/scales
  numerics inside each model pipeline. Integer-coded categoricals are left
  numeric; if you want them one-hot encoded, cast them to string first or extend
  `ml.build_preprocessor` to read `meta['coltypes']`.
- **Grid search** uses a single train/val split (`PredefinedSplit`), per the
  "validation set only" requirement — not k-fold CV.
- **Spline CI** is the GLM linear-predictor (log-odds) confidence band.
- Authentication / multi-user isolation is **not** implemented — this is a
  local / trusted-server research tool. Add a reverse proxy + auth before
  exposing it publicly, and set `CLINTAB_SECRET` in the environment.

### Legacy CLI (removed)
The original `src/` pipeline (`class_tqip.py` / `select_features.py`) only
covered the old classification path and has been **replaced** by
`clintab_cli.py`, which exposes the full feature set on top of the same shared
modules as the web app. Recover the old files from git history if ever needed.

---

## Contributing & support

Bug reports and feature requests are welcome via [GitHub
Issues](https://github.com/Applied-Clinical-AI-Initiative/ClinTab-ML/issues) —
use the bug report or feature request template, whichever fits. See
`CONTRIBUTING.md` for dev setup, how to run the test suite, and branch/PR
conventions before opening a pull request. This project follows the
`CODE_OF_CONDUCT.md` in this repo; please report any violations as described
there.

For questions about using the app rather than a bug or a code change, open a
GitHub Discussion or Issue rather than emailing maintainers directly, so the
answer is visible to future users with the same question.

---

## Requirements
Python 3.9+. See `requirements.txt` (Flask, pandas, numpy, scikit-learn,
imbalanced-learn, matplotlib, scipy, statsmodels, patsy, lifelines, joblib,
gunicorn).
