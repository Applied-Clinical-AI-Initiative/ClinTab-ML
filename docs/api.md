# API reference

Two ways to use ClinTAB-ML programmatically: the HTTP API behind the web
app, or the pure Python functions in `clintab/` (which have no Flask
dependency and can be imported directly, as `clintab_cli.py` does).

## HTTP API

All endpoints are under `/api` (see `routes.py`). Requests/responses are
JSON unless noted. Most endpoints require an active upload session — call
`/api/upload` first, which sets a session cookie.

### Data upload & session

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| POST | `/api/upload` | multipart `file` (CSV) | `session_id`, detected `columns`, row/col counts, a 20-row `preview` |
| POST | `/api/confirm` | `types`, `missing`, `ratios`, `method`, `stratify_col`, `smote`, `seed` | split sizes (`n_train`/`n_val`/`n_test`), SMOTE hint |
| GET | `/api/summary` | — | descriptive-statistics payload (continuous/categorical tables + summary card) |
| GET | `/api/session` | — | current session metadata, or `{"active": false}` |

### Model training

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| GET | `/api/train/columns` | — | outcome candidates, columns, default hyperparameter grids per model |
| POST | `/api/train` | `outcome`, `exclude`, `confounders`, `models`, `grids`, `scoring`, `grid_search`, `smote`, `threshold` | stores the config; training itself runs on the stream below |
| GET | `/api/train/stream` | — | `text/event-stream` (SSE) — `start` → `progress`/`model_done`/`model_error` per model → `complete` |

### Model testing & prediction

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| GET | `/api/models` | — | list of saved models + their metadata |
| POST | `/api/model/upload` | multipart `file` (`.pkl`) | registers an externally-trained pipeline |
| GET | `/api/model/features` | `?name=` | feature list, outcome, task, classes for a saved model |
| POST | `/api/test` | `models` (names), `source` (`test`/`upload`), `threshold` | metrics + plot coordinates per model, ROC/PR overlay PNGs |
| POST | `/api/predict-single` | `model`, `row` (JSON) — or `model_file`/`json_file` uploads | prediction, probability (binary), plain-language explanation |

### Spline & clinical epi

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| POST | `/api/spline` | `predictor`, `outcome`, `n_knots` | log-odds curve, 95% CI, knot positions, PNG |
| POST | `/api/epi/or` | `a`, `b`, `c`, `d` | OR, RR, CIs, chi-square, NNT/NNH |
| POST | `/api/epi/nnt` | `a`, `b`, `c`, `d` | NNT/NNH breakdown with plain-language interpretation |
| POST | `/api/epi/km` | `time`, `event`, `group` (optional) | Kaplan-Meier curve(s), log-rank test, PNG |
| POST | `/api/epi/cox` | `time`, `event`, `covariates` | hazard ratios + CIs per covariate |
| POST | `/api/epi/hl` | `model` (saved name) | Hosmer-Lemeshow statistic, calibration deciles, PNG |

### Export

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| POST | `/api/render/pdf` | `kind` (chart type), `data` | downloadable PDF of the requested chart |

## Python library (`clintab/`)

Every module here is pure — pandas/numpy/sklearn in, plain dicts/DataFrames
out, no Flask. `routes.py` and `clintab_cli.py` both call straight into
these.

### `clintab.stats`

| Function | Takes | Returns |
|----------|-------|---------|
| `detect_column_types(df)` | a DataFrame | list of `{name, type, n_unique, pct_missing, high_missing, numeric, sample}` per column |
| `summarize(df, coltypes)` | a DataFrame + `{col: type}` | `{card, continuous, categorical}` summary payload |
| `apply_missing_handling(df, decisions)` | a DataFrame + `{col: 'include'\|'zero'\|'remove'}` | the cleaned DataFrame |
| `minority_fraction(series)` | a pandas Series | smallest class proportion, or `None` |

### `clintab.ml`

| Function | Takes | Returns |
|----------|-------|---------|
| `determine_task(series)` | outcome column | `"binary"`, `"multiclass"`, or `"continuous"` |
| `model_catalogue(task)` | task string | dict of model name → `{model, params}` (15 models total) |
| `train_one_model(name, df_train, df_val, outcome, feat_cols, task, ...)` | training/validation frames + config | `(fitted_pipeline, info_dict)` |
| `evaluate(pipe, df, outcome, feat_cols, task, threshold=0.5)` | a fitted pipeline + a DataFrame | `(metrics_dict, plot_coords_dict)` |
| `feature_importance(pipe, df_val, outcome, feat_cols, task)` | a fitted pipeline | DataFrame of `feature`/`importance`, sorted descending |
| `predict_single(pipe, row_dict, feat_cols, task, classes=None)` | a fitted pipeline + one row as a dict | prediction, probability (binary), explanation string |

### `clintab.epi`

| Function | Takes | Returns |
|----------|-------|---------|
| `two_by_two(a, b, c, d, conf=0.95)` | 2x2 table cell counts | odds ratio, risk ratio, CIs, chi-square, NNT/NNH |
| `nnt_nnh(a, b, c, d)` | 2x2 table cell counts | NNT/NNH value + plain-language interpretation |
| `kaplan_meier(df, time_col, event_col, group_col=None)` | a DataFrame | survival curve(s) + log-rank test if grouped |
| `cox_ph(df, time_col, event_col, covariates)` | a DataFrame + covariate list | hazard ratios, CIs, concordance index |
| `hosmer_lemeshow(y_true, y_prob, g=10)` | true labels + predicted probabilities | HL statistic, p-value, per-decile calibration table |

### `clintab.spline`

| Function | Takes | Returns |
|----------|-------|---------|
| `fit_rcs(df, predictor, outcome, n_knots=4)` | a DataFrame, predictor/outcome column names | log-odds curve, 95% CI, knot positions, AIC/deviance |

### `clintab.plots`

Matplotlib rendering only — one function per chart type (`roc_overlay`,
`pr_overlay`, `confusion_plot`, `calibration_plot`, `residual_plot`,
`pred_vs_actual_plot`, `importance_plot`, `spline_plot`, `km_plot`,
`calibration_hl_plot`), each returning a `matplotlib.figure.Figure`.
`fig_to_png(fig)` and `fig_to_pdf_bytes(fig)` encode a figure for the HTTP
responses above.

### `clintab.store`

Filesystem layout and session metadata helpers (`ensure_dirs`,
`session_dir`, `save_meta`/`load_meta`, `model_path`, `list_models`). See
the module docstring for the on-disk directory layout.
