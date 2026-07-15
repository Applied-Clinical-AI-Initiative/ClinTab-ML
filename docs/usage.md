# Usage

This walks through one realistic end-to-end example — profiling a cohort,
training and evaluating a model, and running a clinical-epi calculation —
using both the CLI and the web app. All commands below use the synthetic
cohort at `Data/sample_clinical.csv` (see [`Data/README.md`](../Data/README.md)
for what's in it).

## 1. Profile the dataset

```bash
python clintab_cli.py summarize --csv Data/sample_clinical.csv
```

```
Data/sample_clinical.csv: 600 rows x 11 cols

Detected column types:
  name            type         n_unique  pct_missing  high_missing
  ----------------------------------------------------------------
  age             continuous   360       0.0          False
  bmi             continuous   196       0.0          False
  sex             binary       2         0.0          False
  asa_class       categorical  4         0.0          False
  length_of_stay  continuous   14        0.0          False
  diabetes        binary       2         0.0          False
  mortality_30d   binary       2         0.0          False
  surv_time       continuous   320       0.0          False
  death_event     binary       2         0.0          False
  admit_date      date         414       0.0          False
  rare_lab        continuous   254       57.67        True

Card: {'total_n': 600, 'n_columns': 11, 'complete_cases': 254,
       'pct_complete_cases': 42.33, 'date_range': {...}}
```

Column types (binary / categorical / continuous / date) and missingness are
detected automatically — this is the same logic the web app's Data Upload
step uses to pre-fill its column-type form.

## 2. Train a model

```bash
python clintab_cli.py train --csv Data/sample_clinical.csv \
    --outcome mortality_30d --exclude surv_time death_event \
    --models LogisticRegression --scoring roc --no-grid
```

```
Training 1 model(s) | task=binary | features=7 | split 420/90/90 | scoring=roc | smote=False

  … LogisticRegression
    saved -> models/LogisticRegression_mortality_30d_<timestamp>.pkl  (top feature: num__age)

Validation metrics:
  model               AUROC   AUPRC   Sensitivity  Specificity  PPV     NPV     F1      Brier  Threshold  N
  ----------------------------------------------------------------------------------------------------------
  LogisticRegression  0.7233  0.5905  0.2727       0.9853       0.8571  0.8072  0.4138  0.151  0.5        90
```

`--exclude` drops columns that shouldn't be used as predictors (here,
`surv_time`/`death_event` are the survival-analysis pair, not features for
the mortality classifier). `--no-grid` skips hyperparameter tuning for a
quick run; drop it to grid-search each model's parameter grid on the
validation set. The trained pipeline is pickled to `models/`, alongside a
`.meta.json` sidecar with its metrics and parameters.

Run `python clintab_cli.py models` to see all 15 available models (10
classifiers, 5 regressors), and `python clintab_cli.py train -h` for every
option (`--split`, `--stratify`, `--smote`, `--seed`, `--test`, ...).

## 3. Evaluate it on held-out data

```bash
python clintab_cli.py test --models LogisticRegression_mortality_30d_<timestamp> \
    --csv Data/sample_clinical.csv --out roc.png
```

This reports the same metrics table on the CSV you point it at, and (with
`--out`) saves ROC/PR overlay plots as PNGs.

## 4. A clinical-epi calculation

```bash
python clintab_cli.py epi or --a 20 --b 80 --c 10 --d 90
```

```json
{
  "odds_ratio": 2.25,  "or_ci": [0.9943, 5.0915],
  "risk_ratio": 2.0,   "rr_ci": [0.9866, 4.0545],
  "abs_risk_diff": 0.1, "nnt": 10.0, "nnt_type": "NNH (harm)",
  "chi2": 3.1765, "p_value": 0.0747059
}
For every 10.0 patients exposed, one additional event occurs (number needed to harm).
```

`a`/`b`/`c`/`d` are the four cells of a 2x2 exposure/outcome table (see the
convention documented at the top of `clintab/epi.py`). The other epi
subcommands (`km`, `cox`, `hl`) work the same way against a CSV — see
`python clintab_cli.py epi -h`.

## Using the web app instead

```bash
./run.sh
```

Open `http://127.0.0.1:5000`, upload `Data/sample_clinical.csv`, and the
same six steps (Data Upload → Data Summary → Model Training → Model Testing
→ Spline → Clinical Epi) are available from one page, with live progress and
inline charts instead of terminal output. Models trained from the CLI and
the web app share the same `models/` folder, so either one can evaluate
what the other trained. See the main [README](../README.md) for a full
description of each section.

## Full API reference

For the underlying HTTP endpoints and Python function signatures, see
[`docs/api.md`](api.md).
