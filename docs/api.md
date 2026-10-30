# API reference

Two ways to use ClinTAB-ML programmatically: the HTTP API behind the web
app, or the pure Python functions in `clintab/` (which have no Flask
dependency and can be imported directly, as `clintab_cli.py` does).

## HTTP API

All endpoints are under `/api` (see `routes.py`). Requests/responses are
JSON unless noted. Most endpoints require an active upload session, so call
`/api/upload` first, which sets a session cookie. Every JSON response is
strict JSON: NaN and infinity come back as `null`.

### Data upload & session

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| GET | `/api/version` | none | `{app, version}` |
| POST | `/api/upload` | multipart `file` (CSV) | `session_id`, detected `columns`, row/col counts, a 20-row `preview` |
| POST | `/api/cohort/preview` | `filters` (list, as `cohort.apply_filters` takes) | `{flow, lines, n_start, n_final, png}` for the raw upload; nothing is saved. A bad filter list is a 400 `{error, problems: [...]}` |
| POST | `/api/confirm` | `types`, `missing`, `ratios`, `method`, `stratify_col`, `smote`, `seed`, optional `cohort` (filters) | `{ok, n_train, n_val, n_test, method, smote_hint, missing_pct, fill_values, indicators, cohort}` |
| GET | `/api/summary` | none | descriptive-statistics payload (continuous/categorical tables + summary card) |
| GET | `/api/session` | none | current session metadata, or `{"active": false}` |
| GET | `/api/sessions` | none | every dataset on this machine, and which one is active |
| POST · DELETE | `/api/sessions/<id>/activate` · `/api/sessions/<id>` · `/api/sessions` | none | switch to a dataset, delete one, or delete all |

`/api/confirm` in order: reads the upload, applies the `cohort` filters (if
any), measures missingness on the cohort (`missing_pct`), drops excluded and
removed columns, splits, then learns fill values on the training partition
only (`fill_values`, `{column: value}`) and applies them, with any missing
indicator columns (`indicators`, `{column: indicator column}`), to every
partition. `missing` maps a column to `include` (leave missing, the model
pipeline imputes inside training), `zero`, `impute`, `impute_indicator` or
`remove`. `smote_hint` is `{column, minority_fraction, imbalanced}` for the
stratify column, or `null`. `cohort` is `{flow, lines, n_start, n_final}`, or
`null` when no filters were given; a cohort that leaves no rows is a 400. The
filters, flow and counts are saved in the session metadata (`meta.cohort`)
and logged as `cohort_filter` just before the `confirm` entry.

### Model training

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| GET | `/api/train/columns` | none | outcome candidates, columns, default hyperparameter grids per model |
| POST | `/api/train/upload-split` | multipart `train_file`, `val_file` | replaces the train/validation partitions with your own CSVs and re-detects column types |
| POST | `/api/train` | `outcome`, `exclude`, `confounders`, `models`, `grids`, `scoring`, `grid_search`, `cv_folds`, `smote`, `threshold_method`, `threshold_target`, `recalibrate` (`threshold` still accepted) | stores the config; training itself runs on the stream below |
| GET | `/api/train/stream` | none | `text/event-stream` (SSE): `start` → `progress`/`model_done`/`model_error` per model → `complete` |

Threshold settings (binary outcomes): `threshold_method` is one of
`youden` (default), `sensitivity`, `specificity`, `prevalence` or `fixed`.
`threshold_target` is required for `fixed` (between 0 and 1), `sensitivity`
and `specificity` (above 0, at most 1); for `fixed` an older client's
`threshold` is used when `threshold_target` is missing. The threshold is
chosen on the validation rows that have an outcome. `recalibrate: true`
first refits the model's risks with logistic recalibration (intercept and
slope on the logit scale) on the same validation rows.

`/api/train` refuses (400) an outcome whose missing values were filled in at
confirm (`impute` or `impute_indicator`); set that column to "Include as is"
and confirm again, rows with a missing outcome are then dropped at training.

Each `model_done` event carries `model`, `saved_as`, `metrics` (with a
nested `CI` dict for binary outcomes), `best_params`, `coords`,
`importance`, `importance_png`, `pct`, plus `threshold`,
`threshold_method`, `threshold_target`, `recalibrated`, `recalibration`
(`{a, b}`, only when recalibrated) and `notes` (plain text, e.g. that
validation calibration is optimistic after recalibration). A
`LogisticRegression` model on a binary outcome also carries `odds_ratios`
(see `ml.adjusted_odds_ratios`), logged as `adjusted_or`. The threshold
fields are saved in the model file and its `.meta.json`.

### Model testing & prediction

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| GET | `/api/models` | none | list of saved models + their metadata |
| DELETE | `/api/models/<name>` | none | deletes a saved model |
| POST | `/api/model/upload` | multipart `file` (`.pkl`) | registers an externally-trained pipeline |
| GET | `/api/model/features` | `?name=` | `{name, features, derived_features, outcome, task, classes}`; `derived_features` are missing indicator columns the app builds itself, so a form should not ask for them |
| GET | `/api/model/<name>/odds-ratios` | none | the adjusted odds ratio table saved with a `LogisticRegression` model (the `ml.adjusted_odds_ratios` dict plus `name`, `model`, `outcome`); 404 for any other model |
| POST | `/api/test` | `models` (names), `source` (`test`/`upload`), `threshold` (optional), `unlock_reason` (optional), multipart `file` for `upload` | metrics + plot coordinates per model, ROC/PR overlay PNGs, decision curves |
| POST | `/api/predict-single` | `model`, `row` (JSON), or `model_file`/`json_file` uploads | prediction, probability (binary), plain-language explanation |

`/api/test` thresholds: with no `threshold` (missing or blank) each model
uses the threshold saved with it; a `threshold` overrides it for every
model. Each model entry has `name`, `metrics`, `coords`, `threshold`,
`threshold_method` (`"saved"` or `"user"`), `recalibrated`, and for binary
models `confusion_png`, `calibration_png` (flexible lowess calibration curve
with grouped points, a risk histogram and a box with slope, CITL, O:E and
ICI) and `calibration_stats`. The response adds `roc_png`, `pr_png`,
`dca` (`{thresholds, treat_all, models: [{name, net_benefit}]}`) and
`dca_png` when any model is binary, plus `source` and, for the internal
test partition, `test_lock`.

Test-set lock: the internal test partition is locked to the models of its
first use. Testing a model outside the lock on it returns 409
`{error, locked: true, new_models, locked_models}` unless the request has a
non-empty `unlock_reason`; the reason is logged (`test_unlock`) and the
models join the lock. `/api/epi/hl` is under the same lock. External CSVs
(`source=upload`) are never locked; they get the model's training-fitted
fills and indicators and are logged with filename, SHA-256 and n.

### Spline & clinical epi

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| POST | `/api/spline` | `predictor`, `outcome`, `n_knots` | log-odds curve, 95% CI, knot positions, PNG |
| POST | `/api/epi/or` | `a`, `b`, `c`, `d` | `epi.two_by_two`: OR, RR, CIs, risk difference with `rd_ci`, `nnt` with `nnt_ci`, chi-square and Fisher p-values, notes |
| POST | `/api/epi/nnt` | `a`, `b`, `c`, `d` | `epi.nnt_nnh`: NNT/NNH with `nnt_ci`, `rd_ci`, `value_rounded_up` and a plain-language interpretation |
| POST | `/api/epi/km` | `time`, `event`, `group` (optional), `time_points` (optional list) | `epi.kaplan_meier`: curves, log-rank test (k groups), `risk_table`, PNG with a number at risk table |
| POST | `/api/epi/cox` | `time`, `event`, `covariates`, `categorical` (list), `reference` (`{col: level}`), `ph_transform` (`rank` default, `km`, `identity`, `log`) | `epi.cox_ph`: hazard ratios + CIs per term, `ph_test` (per term and global), `ph_notes` |
| POST | `/api/epi/cif` | `time`, `event`, `event_of_interest` (default 1), `group` (optional), `time_points` (optional) | `epi.cumulative_incidence`: Aalen-Johansen CIF per group, naive 1 minus KM, `gray_test`, `risk_table`, `notes`, PNG |
| POST | `/api/epi/hl` | `model` (saved name), `unlock_reason` (optional) | Hosmer-Lemeshow statistic, calibration deciles, PNG |

KM, Cox and CIF use the cleaned full dataset (`full.csv`). CIF event coding:
0 = censored, `event_of_interest` = the event, any other code = a competing
event.

### Sample size

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| POST | `/api/samplesize` | see below | `{result, compare}` |
| GET | `/api/samplesize/defaults` | `?outcome=X&exclude=a,b` | `{prevalence, n_parameters, n, events}` from the training partition (binary outcome only, else 400) |

`/api/samplesize` body: `outcome_type` `binary` (default) or `continuous`.
Binary: `prevalence`, `n_parameters`, exactly one of `cstatistic`,
`csrsquared`, `nagrsquared`, optional `shrinkage`, `delta`, `margin`.
Continuous: `rsquared`, `n_parameters`, `intercept`, `sd`, optional
`shrinkage`, `delta`, `mmoe`. `result` is `samplesize.binary_sample_size`
or `samplesize.continuous_sample_size`. Optional `compare: {outcome,
exclude}` checks the session's training partition with
`samplesize.compare_with_data` (n = rows with an outcome, events = rows in
the positive class, parameters counted as for training; a continuous outcome
uses n for events); `compare` is `null` without it. Works without a session
(then `compare` must be absent); with a session it is logged as
`sample_size`. Bad input is a 400.

### Teaching mode

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| POST | `/api/teaching` | `outcome`, `exclude` (list), `model` (`logistic`, `random_forest`, `gradient_boosting`), `n_seeds` (1 to 5, default 3), `max_rows` (default 20000) | `{ok: true}`; stores the config. 400 for an outcome that is not binary or an unknown column |
| GET | `/api/teaching/stream` | none | SSE: `{"event": "progress", "message"}` events, then `{"event": "complete", "result", "png"}` or `{"event": "error", "error"}` |

The stream runs `teaching.leakage_demo` on the cleaned full dataset with the
same features training would use (all columns but the outcome, excluded and
date columns), seeds 1 to `n_seeds`. It is logged as `leakage_demo`, which the
methods text and TRIPOD report ignore.

### Reports

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| GET | `/api/report/methods` | none | `{text}`, a methods paragraph built from the session's analysis log |
| GET | `/api/report/tripod` | none | TRIPOD+AI coverage: `{disclaimer, items, summary}` |
| GET | `/api/report/verify-log` | none | `{ok, n_entries, first_bad_line, legacy}`: checks the analysis log's hash chain (each entry stores the hash of the line before it) |

### Errors

Every `/api` error is JSON, `{"error": "..."}`, with a matching status code.

### Export

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| POST | `/api/render/pdf` | `kind` (chart type), `data`, optional `name` | downloadable PDF of the requested chart |

Chart kinds and their `data`:

| kind | data |
|------|------|
| `roc`, `pr` | `{series}` |
| `confusion` | `{confusion}` |
| `calibration` | `{prob_pred, prob_true}` |
| `calibration_flexible` | `{p_grid, observed, binned_pred, binned_obs, risk_histogram, calibration_stats, name}`, or a binary model's test `coords` plus `calibration_stats` |
| `residuals` | `{pred, resid}` |
| `pred_vs_actual` | `{actual, pred}` |
| `importance` | `{names, values}` |
| `spline` | `{x, logodds, lower, upper, knots, predictor}` |
| `km` | `{curves, risk_table}` (`risk_table` optional) |
| `cif` | `{curves, naive, risk_table}` (`naive`, `risk_table` optional) |
| `dca` | the `/api/test` `dca` object, `{thresholds, treat_all, models}` |
| `hl` | `{mean_pred, observed_rate}` |
| `teaching` | `{result}` (a `leakage_demo` result) |
| `flow` | `{flow, n_start, title}` |

## Python library (`clintab/`)

Every module here is pure: pandas/numpy/sklearn in, plain dicts/DataFrames
out, no Flask. `routes.py` and `clintab_cli.py` both call straight into
these.

### `clintab.stats`

| Function | Takes | Returns |
|----------|-------|---------|
| `detect_column_types(df)` | a DataFrame | list of `{name, type, n_unique, pct_missing, high_missing, numeric, sample}` per column |
| `summarize(df, coltypes)` | a DataFrame + `{col: type}` | `{card, continuous, categorical}` summary payload |
| `missing_percentages(df)` | a DataFrame | `{col: % missing}` for every column with a missing value |
| `fit_missing_handling(train_df, decisions, coltypes=None)` | the training partition + `{col: 'include'\|'zero'\|'impute'\|'impute_indicator'\|'remove'}` | `{fills, indicators}` learned on the training rows only |
| `apply_fitted_missing_handling(df, decisions, fitted)` | any partition + the decisions + `fit_missing_handling` output | the handled DataFrame (what the app uses) |
| `apply_missing_handling(df, decisions)` | a DataFrame + `{col: 'include'\|'zero'\|'impute'\|'remove'}` | the cleaned DataFrame, with fill values from the frame passed in. Kept for backward compatibility; on data that will be split it leaks test rows into the fills |
| `minority_fraction(series)` · `class_balance(series)` | a pandas Series | smallest class proportion, or `{minority_fraction, imbalanced}` |

### `clintab.ml`

| Function | Takes | Returns |
|----------|-------|---------|
| `determine_task(series)` | outcome column | `"binary"`, `"multiclass"`, or `"continuous"` |
| `model_catalogue(task)` | task string | dict of model name → `{model, params}` (15 models total; `LogisticRegression` is unpenalized, `SVM` is wrapped in `CalibratedClassifierCV`) |
| `train_one_model(name, df_train, df_val, outcome, feat_cols, task, ...)` | training/validation frames + config | `(fitted_pipeline, info_dict)` |
| `evaluate(pipe, df, outcome, feat_cols, task, threshold=0.5, classes=None)` | a fitted pipeline + a DataFrame (`classes`: the `info["classes"]` the model was trained with) | `(metrics_dict, plot_coords_dict)`; binary metrics add `CalibrationSlope`, `CalibrationInTheLarge`, `OE_ratio`, `ICI` and a nested `CI`, coords add `calibration_flexible`, `calibration_stats`, `dca`, `risk_histogram` |
| `count_parameters(df, feat_cols, coltypes=None)` | a DataFrame + features | candidate predictor parameters, one-hot with a reference level dropped |
| `adjusted_odds_ratios(df_train, outcome, feat_cols, coltypes=None, reference=None)` | training data + binary outcome | `{rows: [{term, or, ci_low, ci_high, p, ...}], n, events, n_parameters, epp, reference_levels, imputed, n_missing, outcome_level, converged, warning}` |
| `feature_importance(pipe, df_val, outcome, feat_cols, task, classes=None)` | a fitted pipeline | DataFrame of `feature`/`importance`, sorted descending |
| `predict_single(pipe, row_dict, feat_cols, task, classes=None)` | a fitted pipeline + one row as a dict | prediction, probability (binary), explanation string |

### `clintab.validation`

| Function | Takes | Returns |
|----------|-------|---------|
| `calibration_stats(y, p)` | 0/1 outcome + predicted risks | `{slope, slope_ci, citl, citl_ci, oe_ratio, oe_ci, ici, e50, e90, n, events, reason}` |
| `flexible_calibration_curve(y, p)` | as above | `{p_grid, observed, method, frac}` (lowess) |
| `auroc_ci(y, p)` | as above | `(auc, lo, hi)`, DeLong |
| `bootstrap_ci(y, p, stat, n_boot=500, seed=42)` | as above + a callable or `"auprc"`/`"brier"` | `(lo, hi)` |
| `decision_curve(y, p, thresholds=None)` | as above | `{threshold, net_benefit, treat_all, treat_none}` |
| `choose_threshold(y, p, method="youden", target=None)` | validation outcome + risks | `{threshold, method, target, sensitivity, specificity}` |
| `recalibrate(pipe, X_val, y_val)` | a fitted binary model + validation rows | a `LogisticRecalibrator` (drop-in for the pipeline, `a` and `b` attributes) |

### `clintab.epi`

| Function | Takes | Returns |
|----------|-------|---------|
| `two_by_two(a, b, c, d, conf=0.95)` | 2x2 table cell counts | odds ratio, risk ratio, CIs, risk difference with CI, NNT with CI, chi-square and Fisher p-values |
| `nnt_nnh(a, b, c, d)` | 2x2 table cell counts | NNT/NNH value with CI + plain-language interpretation |
| `kaplan_meier(df, time_col, event_col, group_col=None, time_points=None)` | a DataFrame | survival curve(s), log-rank test across k groups, `risk_table` |
| `cox_ph(df, time_col, event_col, covariates, categorical=None, reference=None, ph_transform="rank")` | a DataFrame + covariate list | hazard ratios, CIs, concordance index, proportional hazards test |
| `cumulative_incidence(df, time_col, event_col, event_of_interest=1, group_col=None, time_points=None)` | a DataFrame | Aalen-Johansen CIF per group, naive 1 minus KM, Gray's test, `risk_table` |
| `hosmer_lemeshow(y_true, y_prob, g=10)` | true labels + predicted probabilities | HL statistic, p-value, per-decile calibration table |

### `clintab.cohort`

| Function | Takes | Returns |
|----------|-------|---------|
| `parse_code_list(text)` | pasted codes (`'27130-27132, M16*'`) | list of normalised codes |
| `apply_filters(df, filters)` | a DataFrame + list of `{column or columns, op, values, label}` | `(cohort_df, flow)`; raises `CohortFilterError` (with `.problems`) |
| `summarize_flow(flow, n_start=None)` | a flow | plain text lines for a methods paragraph or a flow diagram |

### `clintab.samplesize`

| Function | Takes | Returns |
|----------|-------|---------|
| `binary_sample_size(prevalence, n_parameters, csrsquared=None, cstatistic=None, nagrsquared=None, shrinkage=0.9, delta=0.05, margin=0.05)` | anticipated values | Riley et al. criteria, `min_n`, `min_events`, `epp`, notes (same numbers as R pmsampsize) |
| `continuous_sample_size(rsquared, n_parameters, intercept, sd, shrinkage=0.9, delta=0.05, mmoe=1.1)` | anticipated values | the four continuous criteria, `min_n` |
| `compare_with_data(result, n, events, n_parameters)` | a result + what the data has | `{adequate, n, events, epp, shortfall, notes, ...}` |

### `clintab.teaching`

| Function | Takes | Returns |
|----------|-------|---------|
| `leakage_demo(df, outcome, feat_cols, coltypes=None, model="logistic", seeds=(1, 2, 3), max_rows=20000, progress=None)` | a dataset with a binary outcome | per arm, reported vs lockbox AUROC, optimism and calibration (mean, min, max over seeds) |

### `clintab.spline`

| Function | Takes | Returns |
|----------|-------|---------|
| `fit_rcs(df, predictor, outcome, n_knots=4)` | a DataFrame, predictor/outcome column names | log-odds curve, 95% CI, knot positions, AIC/deviance |

### `clintab.plots`

Matplotlib rendering only, with one function per chart type (`roc_overlay`,
`pr_overlay`, `confusion_plot`, `calibration_plot`,
`flexible_calibration_plot`, `dca_plot`, `residual_plot`,
`pred_vs_actual_plot`, `importance_plot`, `spline_plot`, `km_plot` (with an
optional `risk_table`), `cif_plot`, `calibration_hl_plot`, `teaching_plot`,
`flow_plot`), each returning a `matplotlib.figure.Figure`.
`fig_to_png(fig)` and `fig_to_pdf_bytes(fig)` encode a figure for the HTTP
responses above.

### `clintab.store`

Filesystem layout and session metadata helpers (`ensure_dirs`,
`session_dir`, `save_meta`/`load_meta`, `model_path`, `list_models`). See
the module docstring for the on-disk directory layout.
