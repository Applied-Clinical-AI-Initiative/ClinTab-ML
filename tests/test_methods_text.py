import pytest

from clintab import store, analysis_log, methods_text


@pytest.fixture
def session_id(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "SESSIONS", str(tmp_path))
    sid = "testsession"
    store.session_dir(sid, create=True)
    return sid


def test_generate_methods_text_empty_session_returns_empty_string(session_id):
    assert methods_text.generate_methods_text(session_id) == ""


def test_generate_methods_text_covers_upload_and_confirm(session_id):
    analysis_log.log_action(session_id, "upload", inputs={"filename": "sample_clinical.csv"},
                            outputs={"n_rows": 600, "n_cols": 11})
    analysis_log.log_action(session_id, "confirm",
                            inputs={"method": "stratified", "ratios": [0.7, 0.15, 0.15],
                                    "stratify_col": "mortality_30d", "smote_pref": True},
                            outputs={"n_train": 420, "n_val": 90, "n_test": 90})

    text = methods_text.generate_methods_text(session_id)
    assert "600 rows and 11 columns" in text
    assert "sample_clinical.csv" in text
    assert "n=420" in text and "n=90" in text
    assert "split stratified on mortality_30d" in text
    assert "stratified split stratified" not in text
    # a smote preference at confirm is not an action; smote is reported where
    # a model was actually trained with it
    assert "SMOTE" not in text


def test_generate_methods_text_train_model_with_cv_folds(session_id):
    analysis_log.log_action(session_id, "train_model",
                            inputs={"model": "LogisticRegression", "outcome": "mortality_30d",
                                    "scoring": "roc", "smote": False, "cv_folds": 5},
                            outputs={"saved_as": "x", "metrics": {"AUROC": 0.81, "F1": 0.5}})

    text = methods_text.generate_methods_text(session_id)
    assert "LogisticRegression model was trained to predict mortality_30d" in text
    assert "5-fold cross-validation" in text
    assert "area under the receiver operating characteristic curve (AUROC)" in text
    assert "(validation AUROC=0.81)" in text  # f1 isn't in the metric order list, so it's omitted


def test_generate_methods_text_train_model_without_cv_folds_mentions_val_set(session_id):
    analysis_log.log_action(session_id, "train_model",
                            inputs={"model": "RandomForest", "outcome": "y",
                                    "scoring": "roc", "smote": False, "cv_folds": None},
                            outputs={"saved_as": "x", "metrics": {}})

    text = methods_text.generate_methods_text(session_id)
    assert "held-out validation set" in text
    assert "fold" not in text


def test_generate_methods_text_covers_spline_km_cox_hl(session_id):
    analysis_log.log_action(session_id, "fit_spline",
                            inputs={"predictor": "age", "outcome": "y", "n_knots": 4},
                            outputs={"aic": 123.4, "n": 500})
    analysis_log.log_action(session_id, "epi_km",
                            inputs={"time": "surv_time", "event": "death_event", "group": "sex"},
                            outputs={"n_curves": 2, "logrank": {"p_value": 0.03}})
    analysis_log.log_action(session_id, "epi_cox",
                            inputs={"time": "surv_time", "event": "death_event",
                                    "covariates": ["age", "diabetes"]},
                            outputs={"concordance": 0.71, "n": 500})
    analysis_log.log_action(session_id, "epi_hl",
                            inputs={"model": "LogisticRegression_y_ts"},
                            outputs={"hl_statistic": 4.2, "p_value": 0.65})

    text = methods_text.generate_methods_text(session_id)
    assert "restricted cubic spline with 4 knots" in text
    assert "Kaplan-Meier" in text and "stratified by sex" in text and "p=0.03" in text
    assert "Cox proportional-hazards regression" in text and "age, diabetes" in text
    assert "Hosmer-Lemeshow" in text and "p=0.65" in text


def test_generate_methods_text_includes_validation_metrics_parenthetical(session_id):
    analysis_log.log_action(session_id, "train_model",
                            inputs={"model": "GradientBoosting", "outcome": "mortality_30d",
                                    "scoring": "roc", "smote": True, "cv_folds": None},
                            outputs={"saved_as": "x",
                                     "metrics": {"AUROC": 0.6801, "AUPRC": 0.4344}})
    text = methods_text.generate_methods_text(session_id)
    assert "(validation AUROC=0.6801, AUPRC=0.4344)" in text


def test_generate_methods_text_groups_two_models_under_same_protocol(session_id):
    analysis_log.log_action(session_id, "train_model",
                            inputs={"model": "LogisticRegression", "outcome": "mortality_30d",
                                    "scoring": "roc", "smote": True, "cv_folds": None},
                            outputs={"saved_as": "x", "metrics": {"AUROC": 0.69}})
    analysis_log.log_action(session_id, "train_model",
                            inputs={"model": "RandomForest", "outcome": "mortality_30d",
                                    "scoring": "roc", "smote": True, "cv_folds": None},
                            outputs={"saved_as": "y", "metrics": {"AUROC": 0.66}})

    text = methods_text.generate_methods_text(session_id)
    assert "A RandomForest (validation AUROC=0.66) was trained under the same protocol." in text
    assert ", and" not in text  # exactly two models: "x and y", no serial comma


def test_generate_methods_text_groups_three_models_with_plain_and(session_id):
    for model in ["LogisticRegression", "RandomForest", "GradientBoosting"]:
        analysis_log.log_action(session_id, "train_model",
                                inputs={"model": model, "outcome": "mortality_30d",
                                        "scoring": "roc", "smote": True, "cv_folds": None},
                                outputs={"saved_as": model, "metrics": {"AUROC": 0.7}})

    text = methods_text.generate_methods_text(session_id)
    assert ("A RandomForest (validation AUROC=0.7) and a GradientBoosting "
            "(validation AUROC=0.7) were trained under the same protocol." in text)


def test_generate_methods_text_groups_four_models_with_oxford_comma(session_id):
    for model in ["LogisticRegression", "RandomForest", "GradientBoosting", "KNN"]:
        analysis_log.log_action(session_id, "train_model",
                                inputs={"model": model, "outcome": "mortality_30d",
                                        "scoring": "roc", "smote": True, "cv_folds": None},
                                outputs={"saved_as": model, "metrics": {"AUROC": 0.7}})

    text = methods_text.generate_methods_text(session_id)
    assert ("A RandomForest (validation AUROC=0.7), a GradientBoosting "
            "(validation AUROC=0.7), and a KNN (validation AUROC=0.7) were "
            "trained under the same protocol." in text)


def test_generate_methods_text_skips_unrecognized_actions(session_id):
    analysis_log.log_action(session_id, "something_new", inputs={"x": 1})
    analysis_log.log_action(session_id, "test_model", inputs={"model": "x"}, outputs={"metrics": {}})
    assert methods_text.generate_methods_text(session_id) == ""


def test_generate_methods_text_is_deterministic(session_id):
    analysis_log.log_action(session_id, "upload", inputs={"filename": "a.csv"},
                            outputs={"n_rows": 10, "n_cols": 2})
    first = methods_text.generate_methods_text(session_id)
    second = methods_text.generate_methods_text(session_id)
    assert first == second


def test_train_model_without_tuning_does_not_claim_hyperparameter_tuning(session_id):
    analysis_log.log_action(session_id, "train_model",
                            inputs={"model": "LogisticRegression", "outcome": "y",
                                    "scoring": "roc", "smote": False, "cv_folds": None,
                                    "tuned": False},
                            outputs={"metrics": {"AUROC": 0.7}})
    text = methods_text.generate_methods_text(session_id)
    assert "default hyperparameters" in text
    assert "tuned" not in text


def test_article_reads_acronyms_letter_by_letter():
    assert methods_text._article("SVM") == "An"
    assert methods_text._article("KNN") == "A"
    assert methods_text._article("ElasticNetLogistic") == "An"
    assert methods_text._article("RandomForest") == "A"


def test_custom_split_is_reported_until_the_dataset_is_confirmed_again(session_id):
    analysis_log.log_action(session_id, "confirm", inputs={"method": "random"},
                            outputs={"n_train": 70, "n_val": 15, "n_test": 15})
    analysis_log.log_action(session_id, "upload_custom_split",
                            outputs={"n_train": 80, "n_val": 20})
    text = methods_text.generate_methods_text(session_id)
    assert "replaced by user-supplied files" in text
    assert "training (n=80)" in text

    analysis_log.log_action(session_id, "confirm", inputs={"method": "random"},
                            outputs={"n_train": 70, "n_val": 15, "n_test": 15})
    assert "user-supplied" not in methods_text.generate_methods_text(session_id)


# ---------------------------------------------------------------------------
# provenance, missing data, evaluation, other analyses
# ---------------------------------------------------------------------------
SHA = "3fa9c0e1b2d4" + "0" * 52
SAVED = "LogisticRegression_mortality_30d_20261030-012610"
ENV = {"python": "3.12.3", "clintab": "1.0.0", "numpy": "2.5.4", "pandas": "2.3.3",
       "scikit-learn": "1.9.1", "imbalanced-learn": "0.14.2", "statsmodels": "0.15.0",
       "lifelines": "0.30.3", "scipy": "1.18.1", "platform": "Linux-x86_64"}


def _full_log(sid):
    log = analysis_log.log_action
    log(sid, "upload", inputs={"filename": "nsqip.csv", "sha256": SHA},
        outputs={"n_rows": 600, "n_cols": 11, "environment": ENV})
    log(sid, "cohort_filter", inputs={"filters": []},
        outputs={"flow": [{"step": 1, "label": "age 18 or older", "n_before": 620,
                           "n_after": 600, "n_removed": 20}]})
    log(sid, "sample_size",
        inputs={"prevalence": 0.1, "n_parameters": 9, "cstatistic": 0.75, "shrinkage": 0.9},
        outputs={"min_n": 900, "min_events": 90, "epp": 10, "criteria": []})
    log(sid, "confirm",
        inputs={"method": "stratified", "ratios": [0.7, 0.15, 0.15], "seed": 7,
                "stratify_col": "mortality_30d", "smote_pref": False,
                "missing": {"bmi": "impute", "albumin": "impute_indicator",
                            "rare_lab": "remove", "prior_ops": "zero"},
                "missing_pct": {"bmi": 2.5, "albumin": 44.0, "rare_lab": 61.2,
                                "prior_ops": 10.0},
                "excluded": ["mrn"], "smote_hint": None},
        outputs={"n_rows": 600, "n_train": 420, "n_val": 90, "n_test": 90,
                 "fill_values": {"bmi": 27.1, "albumin": 3.9},
                 "indicators": {"albumin": "albumin_missing"}, "removed": ["rare_lab"]})
    log(sid, "train_model",
        inputs={"model": "LogisticRegression", "outcome": "mortality_30d", "task": "binary",
                "scoring": "roc", "smote": False, "tuned": True, "cv_folds": None},
        outputs={"saved_as": SAVED, "metrics": {"AUROC": 0.8}, "n_train": 420,
                 "events": 63, "positive_class": "1", "n_parameters": 9})
    log(sid, "test_model", inputs={"model": SAVED, "source": "internal_test", "n": 90},
        outputs={"metrics": {"AUROC": 0.78, "Brier": 0.11, "CalibrationSlope": 0.91,
                             "CalibrationInTheLarge": 0.02, "Sensitivity": 0.6,
                             "Specificity": 0.8, "Threshold": 0.15, "N": 90,
                             "CI": {"AUROC": [0.7, 0.85], "CalibrationSlope": [0.6, 1.2]}},
                 "dca": True, "threshold_method": "Youden index on the validation set"})
    log(sid, "test_unlock", inputs={"reason": "reviewer asked \u2014 add RF",
                                    "models": ["RandomForest_mortality_30d_20261030-013000"]})
    log(sid, "test_model", inputs={"model": SAVED, "source": "external",
                                   "filename": "site_b.csv", "sha256": "ab" * 32, "n": 250},
        outputs={"metrics": {"AUROC": 0.72, "N": 250}})
    log(sid, "adjusted_or", inputs={"outcome": "mortality_30d", "features": ["age", "asa_class"]},
        outputs={"n": 600, "events": 90, "n_parameters": 4})
    log(sid, "epi_cif", inputs={"time": "surv_time", "event": "status", "event_of_interest": 1,
                                "group": "sex"}, outputs={"n": 600, "n_curves": 2})
    log(sid, "epi_or", inputs={"a": 20, "b": 80, "c": 10, "d": 90},
        outputs={"odds_ratio": 2.25, "or_ci": [0.98, 5.17], "risk_ratio": 2.0,
                 "rr_ci": [0.98, 4.08], "corrected": False})
    log(sid, "epi_nnt", inputs={"a": 20, "b": 80, "c": 10, "d": 90},
        outputs={"measure": "NNH", "value": 10.0})
    log(sid, "leakage_demo", inputs={"outcome": "mortality_30d", "model": "LogisticRegression"},
        outputs={"arms": [{"name": "leaky"}]})


def test_full_log_covers_every_section(session_id):
    _full_log(session_id)
    text = methods_text.generate_methods_text(session_id)
    # data source with checksum prefix
    assert "nsqip.csv (600 rows and 11 columns; SHA-256 prefix 3fa9c0e1b2d4)" in text
    # cohort steps
    assert "Starting from 620 records" in text and "age 18 or older (20 excluded, 600 remaining)" in text
    # sample size
    assert "Riley et al." in text and "At least 900 participants and 90 events" in text
    # missing data by decision, training-only fills
    assert "Of the 4 columns with missing values (2.5% to 61.2% missing)" in text
    assert "1 (bmi) was imputed with the median" in text
    assert "1 (albumin) was imputed the same way with an added binary missing-value indicator" in text
    assert "1 (prior_ops) was filled with zero" in text
    assert "1 (rare_lab) was removed" in text
    assert "estimated on the training partition only" in text
    # exclusions, split, seed
    assert "One column (mrn) was excluded from the analysis" in text
    assert "reviewer asked, add RF" in text
    assert "(70%/15%/15%) using a split stratified on mortality_30d, with random seed 7" in text
    # training data and events per parameter
    assert "63 (15.0%) had mortality_30d = 1" in text and "7.0 events per parameter" in text
    # internal test with cis and calibration, models by type not file id
    assert "internal test partition (n=90)" in text
    assert "The LogisticRegression model for mortality_30d had an AUROC of 0.78 (95% CI 0.7 to 0.85)" in text
    assert "a calibration slope of 0.91 (95% CI 0.6 to 1.2)" in text
    assert "a calibration-in-the-large of 0.02" in text
    assert "threshold of 0.15 (chosen by Youden index on the validation set)" in text
    assert "20261030" not in text
    # test reuse stated plainly with the reason
    assert "internal test partition was reused" in text and "reviewer asked" in text
    assert "the RandomForest model for mortality_30d" in text
    # external validation
    assert "evaluated on external data from site_b.csv (n=250; SHA-256 prefix abababababab)" in text
    # decision curves, adjusted or, competing risks, or/nnt
    assert "Decision curve analysis was performed" in text
    assert "adjusted odds ratios (n=600, 90 events, 4 parameters)" in text
    assert "cumulative incidence of status = 1" in text and "competing risks" in text
    assert "the odds ratio was 2.25 (95% CI 0.98 to 5.17)" in text
    assert "the number needed to harm was 10" in text
    # software, last
    assert text.endswith("Analyses were performed with ClinTAB-ML 1.0.0 in Python 3.12.3 "
                         "(scikit-learn 1.9.1, imbalanced-learn 0.14.2, statsmodels 0.15.0, "
                         "lifelines 0.30.3, numpy 2.5.4, pandas 2.3.3, scipy 1.18.1) on "
                         "Linux-x86_64.")


def test_leakage_demo_is_never_described(session_id):
    _full_log(session_id)
    text = methods_text.generate_methods_text(session_id).lower()
    assert "leak" not in text
    analysis_log.log_action("other", "leakage_demo", inputs={"outcome": "y"}, outputs={"arms": []})
    assert methods_text.generate_methods_text("other") == ""


def test_methods_text_has_no_long_dashes_even_from_user_text(session_id):
    _full_log(session_id)
    text = methods_text.generate_methods_text(session_id)
    assert "\u2014" not in text and "\u2013" not in text


def test_full_log_text_is_deterministic(session_id):
    _full_log(session_id)
    assert (methods_text.generate_methods_text(session_id)
            == methods_text.generate_methods_text(session_id))


def test_legacy_confirm_imputation_is_reported_as_before_the_split(session_id):
    analysis_log.log_action(session_id, "confirm",
                            inputs={"method": "random", "missing": {"bmi": "impute",
                                                                    "x": "impute"}},
                            outputs={"n_train": 70, "n_val": 15, "n_test": 15})
    text = methods_text.generate_methods_text(session_id)
    assert "computed on the full dataset before it was split" in text
    assert "training partition only" not in text


def test_no_missing_values_is_stated(session_id):
    analysis_log.log_action(session_id, "confirm",
                            inputs={"method": "random", "missing": {}, "missing_pct": {}},
                            outputs={"n_train": 70, "n_val": 15, "n_test": 15,
                                     "fill_values": {}})
    assert "There were no missing values" in methods_text.generate_methods_text(session_id)


def test_single_missing_column_sentence(session_id):
    analysis_log.log_action(session_id, "confirm",
                            inputs={"method": "random", "missing": {"albumin": "impute"},
                                    "missing_pct": {"albumin": 44.0}},
                            outputs={"n_train": 70, "n_val": 15, "n_test": 15,
                                     "fill_values": {"albumin": 3.9}})
    text = methods_text.generate_methods_text(session_id)
    assert "One column with missing values, albumin (44% missing), was imputed" in text


def test_indicator_without_plain_imputation_does_not_say_the_same_way(session_id):
    analysis_log.log_action(session_id, "confirm",
                            inputs={"method": "random",
                                    "missing": {"albumin": "impute_indicator", "lab": "remove"},
                                    "missing_pct": {"albumin": 42.0, "lab": 60.0}},
                            outputs={"n_train": 70, "n_val": 15, "n_test": 15,
                                     "fill_values": {"albumin": 3.6}})
    text = methods_text.generate_methods_text(session_id)
    assert "the same way" not in text
    assert ("1 (albumin) was imputed with the median (continuous) or most frequent value "
            "(categorical), with an added binary missing-value indicator") in text


def test_smote_training_carries_the_calibration_caveat(session_id):
    analysis_log.log_action(session_id, "train_model",
                            inputs={"model": "RandomForest", "outcome": "y", "scoring": "roc",
                                    "smote": True, "cv_folds": None},
                            outputs={"metrics": {}})
    text = methods_text.generate_methods_text(session_id)
    assert "SMOTE applied to the training data only" in text
    assert "at risk of miscalibration" in text


def test_legacy_test_entry_is_named_from_the_training_log(session_id):
    analysis_log.log_action(session_id, "train_model",
                            inputs={"model": "GradientBoosting", "outcome": "readmit",
                                    "scoring": "roc", "smote": False, "cv_folds": None},
                            outputs={"saved_as": "GradientBoosting_readmit_20261030-010101",
                                     "metrics": {}})
    analysis_log.log_action(session_id, "test_model",
                            inputs={"model": "GradientBoosting_readmit_20261030-010101",
                                    "source": "test"},
                            outputs={"metrics": {"AUROC": 0.7}})
    text = methods_text.generate_methods_text(session_id)
    assert "The GradientBoosting model for readmit had an AUROC of 0.7" in text
    assert "20261030" not in text


def test_retesting_reports_only_the_latest_result(session_id):
    for auroc in (0.7, 0.71):
        analysis_log.log_action(session_id, "test_model",
                                inputs={"model": SAVED, "source": "internal_test"},
                                outputs={"metrics": {"AUROC": auroc}})
    text = methods_text.generate_methods_text(session_id)
    assert "0.71" in text and "AUROC of 0.7." not in text
