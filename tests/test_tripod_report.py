import pytest

from clintab import store, analysis_log, tripod_report


@pytest.fixture
def session_id(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "SESSIONS", str(tmp_path))
    sid = "testsession"
    store.session_dir(sid, create=True)
    return sid


def _item(report, item_id):
    return next(i for i in report["items"] if i["id"] == item_id)


def test_report_has_disclaimer_and_all_checklist_items(session_id):
    report = tripod_report.generate_tripod_report(session_id)
    assert report["disclaimer"]
    assert len(report["items"]) == report["summary"]["total"]
    assert report["summary"]["total"] > 0


def test_empty_session_has_no_found_items(session_id):
    report = tripod_report.generate_tripod_report(session_id)
    assert report["summary"]["found"] == 0
    assert _item(report, "data_source")["status"] == "not_found"
    assert _item(report, "sample_size")["status"] == "not_found"


def test_narrative_items_are_always_manual_review(session_id):
    analysis_log.log_action(session_id, "upload", inputs={"filename": "x.csv"},
                            outputs={"n_rows": 10, "n_cols": 2})
    report = tripod_report.generate_tripod_report(session_id)
    assert _item(report, "limitations")["status"] == "manual_review"
    assert _item(report, "title_identifies_study_type")["status"] == "manual_review"


def test_upload_and_confirm_mark_data_source_and_sample_size_found(session_id):
    analysis_log.log_action(session_id, "upload", inputs={"filename": "x.csv"},
                            outputs={"n_rows": 600, "n_cols": 11})
    analysis_log.log_action(session_id, "confirm",
                            inputs={"method": "random", "ratios": [0.7, 0.15, 0.15],
                                    "stratify_col": None, "smote_pref": False},
                            outputs={"n_train": 420, "n_val": 90, "n_test": 90})

    report = tripod_report.generate_tripod_report(session_id)
    assert _item(report, "data_source")["status"] == "found"
    assert _item(report, "sample_size")["status"] == "found"


def test_train_model_marks_outcome_and_model_building_found(session_id):
    analysis_log.log_action(session_id, "train_model",
                            inputs={"model": "LogisticRegression", "outcome": "y",
                                    "scoring": "roc", "smote": False, "cv_folds": None},
                            outputs={"saved_as": "x", "metrics": {"AUROC": 0.8}})

    report = tripod_report.generate_tripod_report(session_id)
    assert _item(report, "outcome_definition")["status"] == "found"
    assert _item(report, "model_building_strategy")["status"] == "found"
    assert _item(report, "discrimination_reported")["status"] == "found"
    # no test_model logged, no epi_hl logged
    assert _item(report, "external_validation")["status"] == "not_found"
    assert _item(report, "calibration_reported")["status"] == "not_found"


def test_discrimination_not_applicable_when_nothing_trained(session_id):
    report = tripod_report.generate_tripod_report(session_id)
    assert _item(report, "discrimination_reported")["status"] == "not_applicable"
    assert _item(report, "calibration_reported")["status"] == "not_applicable"
    assert _item(report, "external_validation")["status"] == "not_applicable"


def test_test_model_marks_external_validation_found(session_id):
    analysis_log.log_action(session_id, "train_model",
                            inputs={"model": "LogisticRegression", "outcome": "y",
                                    "scoring": "roc", "smote": False, "cv_folds": None},
                            outputs={"saved_as": "x", "metrics": {}})
    analysis_log.log_action(session_id, "test_model", inputs={"model": "x", "source": "test"},
                            outputs={"metrics": {"AUROC": 0.75}})

    report = tripod_report.generate_tripod_report(session_id)
    assert _item(report, "external_validation")["status"] == "found"


def test_epi_hl_marks_calibration_found(session_id):
    analysis_log.log_action(session_id, "train_model",
                            inputs={"model": "LogisticRegression", "outcome": "y",
                                    "scoring": "roc", "smote": False, "cv_folds": None},
                            outputs={"saved_as": "x", "metrics": {}})
    analysis_log.log_action(session_id, "epi_hl", inputs={"model": "x"},
                            outputs={"hl_statistic": 4.2, "p_value": 0.6})

    report = tripod_report.generate_tripod_report(session_id)
    assert _item(report, "calibration_reported")["status"] == "found"


def test_epi_km_marks_survival_analysis_found_others_not_applicable(session_id):
    analysis_log.log_action(session_id, "epi_km",
                            inputs={"time": "t", "event": "e", "group": None},
                            outputs={"n_curves": 1, "logrank": None})

    report = tripod_report.generate_tripod_report(session_id)
    assert _item(report, "survival_analysis")["status"] == "found"
    assert _item(report, "nonlinearity_explored")["status"] == "not_applicable"


def test_fit_spline_marks_nonlinearity_found(session_id):
    analysis_log.log_action(session_id, "fit_spline",
                            inputs={"predictor": "age", "outcome": "y", "n_knots": 4},
                            outputs={"aic": 100.0, "n": 500})

    report = tripod_report.generate_tripod_report(session_id)
    assert _item(report, "nonlinearity_explored")["status"] == "found"


def test_summary_counts_match_item_statuses(session_id):
    analysis_log.log_action(session_id, "upload", inputs={"filename": "x.csv"},
                            outputs={"n_rows": 10, "n_cols": 2})
    report = tripod_report.generate_tripod_report(session_id)
    s = report["summary"]
    recomputed = {
        "found": sum(1 for i in report["items"] if i["status"] == "found"),
        "not_found": sum(1 for i in report["items"] if i["status"] == "not_found"),
        "not_applicable": sum(1 for i in report["items"] if i["status"] == "not_applicable"),
        "manual_review": sum(1 for i in report["items"] if i["status"] == "manual_review"),
    }
    assert s["found"] == recomputed["found"]
    assert s["not_found"] == recomputed["not_found"]
    assert s["not_applicable"] == recomputed["not_applicable"]
    assert s["manual_review"] == recomputed["manual_review"]
    assert s["found"] + s["not_found"] + s["not_applicable"] + s["manual_review"] == s["total"]
