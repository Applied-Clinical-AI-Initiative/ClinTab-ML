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
    assert "stratified split stratified on mortality_30d" in text
    assert "SMOTE was applied" in text


def test_generate_methods_text_train_model_with_cv_folds(session_id):
    analysis_log.log_action(session_id, "train_model",
                            inputs={"model": "LogisticRegression", "outcome": "mortality_30d",
                                    "scoring": "roc", "smote": False, "cv_folds": 5},
                            outputs={"saved_as": "x", "metrics": {"AUROC": 0.81, "F1": 0.5}})

    text = methods_text.generate_methods_text(session_id)
    assert "LogisticRegression was trained to predict mortality_30d" in text
    assert "5-fold cross-validation" in text
    assert "AUROC=0.81" in text


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
    assert "Cox proportional hazards" in text and "age, diabetes" in text
    assert "Hosmer-Lemeshow" in text and "p=0.65" in text


def test_generate_methods_text_skips_unrecognized_actions(session_id):
    analysis_log.log_action(session_id, "test_model", inputs={"model": "x"}, outputs={"metrics": {}})
    assert methods_text.generate_methods_text(session_id) == ""


def test_generate_methods_text_is_deterministic(session_id):
    analysis_log.log_action(session_id, "upload", inputs={"filename": "a.csv"},
                            outputs={"n_rows": 10, "n_cols": 2})
    first = methods_text.generate_methods_text(session_id)
    second = methods_text.generate_methods_text(session_id)
    assert first == second
