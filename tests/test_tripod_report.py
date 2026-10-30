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


def _report(sid):
    return tripod_report.generate_tripod_report(sid)


def _train(sid, model="LogisticRegression", smote=False, events=60, n_train=420,
           n_parameters=9, task="binary", metrics=None):
    outputs = {"saved_as": f"{model}_y_20261030-010101",
               "metrics": metrics if metrics is not None else {"AUROC": 0.8}}
    if events is not None:
        outputs["events"] = events
    if n_train is not None:
        outputs["n_train"] = n_train
    if n_parameters is not None:
        outputs["n_parameters"] = n_parameters
    inputs = {"model": model, "outcome": "y", "scoring": "roc", "smote": smote,
              "cv_folds": None}
    if task:
        inputs["task"] = task
    analysis_log.log_action(sid, "train_model", inputs=inputs, outputs=outputs)


def _test(sid, source="internal_test", metrics=None, **outputs):
    inputs = {"model": "LogisticRegression_y_20261030-010101", "source": source}
    if source == "external":
        inputs.update({"filename": "other_site.csv", "sha256": "ab" * 32, "n": 200})
    analysis_log.log_action(sid, "test_model", inputs=inputs,
                            outputs={"metrics": metrics or {"AUROC": 0.75}, **outputs})


def _confirm(sid, missing=None, missing_pct=None, excluded=None, legacy=False):
    inputs = {"method": "random", "ratios": [0.7, 0.15, 0.15], "seed": 42,
              "stratify_col": None, "smote_pref": False, "missing": missing or {},
              "excluded": excluded or [], "smote_hint": None}
    if not legacy:
        inputs["missing_pct"] = missing_pct or {}
    analysis_log.log_action(sid, "confirm", inputs=inputs,
                            outputs={"n_train": 420, "n_val": 90, "n_test": 90})


CAL = {"AUROC": 0.75, "CalibrationSlope": 0.92, "CalibrationInTheLarge": 0.03}


def test_report_has_disclaimer_and_all_checklist_items(session_id):
    report = _report(session_id)
    assert report["disclaimer"]
    assert len(report["items"]) == report["summary"]["total"]
    assert report["summary"]["total"] > 0
    assert all("note" in i for i in report["items"])


def test_item_ids_are_unique_and_include_the_new_items(session_id):
    ids = [i["id"] for i in _report(session_id)["items"]]
    assert len(ids) == len(set(ids))
    for new in ("internal_validation", "external_validation", "clinical_utility",
                "model_specification", "sample_size", "calibration_reported"):
        assert new in ids


def test_empty_session_has_no_found_items(session_id):
    report = _report(session_id)
    assert report["summary"]["found"] == 0
    assert _item(report, "data_source")["status"] == "not_found"
    assert _item(report, "sample_size")["status"] == "manual_review"


def test_narrative_items_are_always_manual_review(session_id):
    analysis_log.log_action(session_id, "upload", inputs={"filename": "x.csv"},
                            outputs={"n_rows": 10, "n_cols": 2})
    report = _report(session_id)
    assert _item(report, "limitations")["status"] == "manual_review"
    assert _item(report, "title_identifies_study_type")["status"] == "manual_review"


def test_upload_marks_data_source_found_but_a_split_is_not_a_sample_size(session_id):
    analysis_log.log_action(session_id, "upload", inputs={"filename": "x.csv"},
                            outputs={"n_rows": 600, "n_cols": 11})
    _confirm(session_id)
    report = _report(session_id)
    assert _item(report, "data_source")["status"] == "found"
    assert _item(report, "sample_size")["status"] == "manual_review"


def test_sample_size_found_only_with_a_logged_calculation(session_id):
    analysis_log.log_action(session_id, "sample_size",
                            inputs={"prevalence": 0.1, "n_parameters": 12,
                                    "cstatistic": 0.75, "shrinkage": 0.9},
                            outputs={"min_n": 1200, "min_events": 120, "epp": 10,
                                     "criteria": []})
    item = _item(_report(session_id), "sample_size")
    assert item["status"] == "found"
    assert "1200" in item["note"]


def test_sample_size_note_gives_events_per_parameter(session_id):
    _train(session_id, events=60, n_parameters=12)
    item = _item(_report(session_id), "sample_size")
    assert item["status"] == "manual_review"
    assert "5.0 events per parameter" in item["note"]


def test_train_model_marks_outcome_and_model_building_found(session_id):
    _train(session_id)
    report = _report(session_id)
    assert _item(report, "outcome_definition")["status"] == "found"
    assert _item(report, "model_building_strategy")["status"] == "found"
    assert _item(report, "discrimination_reported")["status"] == "found"
    # no test_model logged, no calibration logged
    assert _item(report, "external_validation")["status"] == "not_found"
    assert _item(report, "internal_validation")["status"] == "not_found"
    assert _item(report, "calibration_reported")["status"] == "not_found"


def test_discrimination_not_applicable_when_nothing_trained(session_id):
    report = _report(session_id)
    for item_id in ("discrimination_reported", "calibration_reported", "external_validation",
                    "internal_validation", "clinical_utility", "class_imbalance"):
        assert _item(report, item_id)["status"] == "not_applicable"


def test_internal_test_is_internal_validation_never_external(session_id):
    _train(session_id)
    _test(session_id, source="internal_test")
    report = _report(session_id)
    assert _item(report, "internal_validation")["status"] == "found"
    assert "Bootstrap" in _item(report, "internal_validation")["note"]
    ext = _item(report, "external_validation")
    assert ext["status"] == "not_found"
    assert "internal validation, not external" in ext["note"]


def test_legacy_test_source_counts_as_internal(session_id):
    _train(session_id)
    _test(session_id, source="test")
    report = _report(session_id)
    assert _item(report, "internal_validation")["status"] == "found"
    assert _item(report, "external_validation")["status"] == "not_found"


def test_external_source_marks_external_validation_found(session_id):
    _train(session_id)
    _test(session_id, source="external")
    item = _item(_report(session_id), "external_validation")
    assert item["status"] == "found"
    assert "other_site.csv" in item["note"]


def test_test_unlock_is_flagged_in_the_internal_validation_note(session_id):
    _train(session_id)
    _test(session_id)
    analysis_log.log_action(session_id, "test_unlock",
                            inputs={"reason": "added a model", "models": ["RF_y_1"]})
    note = _item(_report(session_id), "internal_validation")["note"]
    assert "reused" in note and "optimistic" in note


def test_calibration_found_with_slope_and_calibration_in_the_large(session_id):
    _train(session_id)
    _test(session_id, metrics=CAL)
    assert _item(_report(session_id), "calibration_reported")["status"] == "found"


def test_calibration_slope_alone_is_not_enough(session_id):
    _train(session_id)
    _test(session_id, metrics={"AUROC": 0.7, "CalibrationSlope": 0.9})
    assert _item(_report(session_id), "calibration_reported")["status"] == "not_found"


def test_hosmer_lemeshow_alone_is_manual_review_with_a_note(session_id):
    _train(session_id)
    analysis_log.log_action(session_id, "epi_hl", inputs={"model": "x"},
                            outputs={"hl_statistic": 4.2, "p_value": 0.6})
    item = _item(_report(session_id), "calibration_reported")
    assert item["status"] == "manual_review"
    assert "Van Calster" in item["note"]


def test_epi_km_marks_survival_analysis_found_others_not_applicable(session_id):
    analysis_log.log_action(session_id, "epi_km",
                            inputs={"time": "t", "event": "e", "group": None},
                            outputs={"n_curves": 1, "logrank": None})
    report = _report(session_id)
    assert _item(report, "survival_analysis")["status"] == "found"
    assert _item(report, "nonlinearity_explored")["status"] == "not_applicable"


def test_competing_risks_count_as_survival_analysis(session_id):
    analysis_log.log_action(session_id, "epi_cif",
                            inputs={"time": "t", "event": "e", "event_of_interest": 1,
                                    "group": None}, outputs={"n": 100, "n_curves": 1})
    assert _item(_report(session_id), "survival_analysis")["status"] == "found"


def test_fit_spline_marks_nonlinearity_found(session_id):
    analysis_log.log_action(session_id, "fit_spline",
                            inputs={"predictor": "age", "outcome": "y", "n_knots": 4},
                            outputs={"aic": 100.0, "n": 500})
    assert _item(_report(session_id), "nonlinearity_explored")["status"] == "found"


# missing data ---------------------------------------------------------------
def test_missing_data_not_applicable_when_no_column_was_missing(session_id):
    _confirm(session_id, missing_pct={})
    assert _item(_report(session_id), "missing_data_handling")["status"] == "not_applicable"


def test_missing_data_found_when_every_missing_column_has_a_decision(session_id):
    _confirm(session_id, missing={"rare_lab": "remove", "bmi": "impute"},
             missing_pct={"rare_lab": 60.0, "bmi": 5.0})
    assert _item(_report(session_id), "missing_data_handling")["status"] == "found"


def test_missing_data_excluded_columns_need_no_decision(session_id):
    _confirm(session_id, missing={"bmi": "impute"}, missing_pct={"bmi": 5.0, "mrn": 1.0},
             excluded=["mrn"])
    assert _item(_report(session_id), "missing_data_handling")["status"] == "found"


def test_missing_data_not_found_when_a_missing_column_has_no_decision(session_id):
    # the old bug: an empty decision dict read as "nothing was missing"
    _confirm(session_id, missing={}, missing_pct={"albumin": 44.0})
    item = _item(_report(session_id), "missing_data_handling")
    assert item["status"] == "not_found"
    assert "albumin" in item["note"]


def test_missing_data_legacy_log_is_manual_review(session_id):
    _confirm(session_id, missing={"rare_lab": "remove"}, legacy=True)
    assert _item(_report(session_id), "missing_data_handling")["status"] == "manual_review"


# class imbalance ---------------------------------------------------------------
def test_class_imbalance_found_without_resampling_when_prevalence_is_known(session_id):
    _train(session_id, smote=False, events=40, n_train=420)
    item = _item(_report(session_id), "class_imbalance")
    assert item["status"] == "found"
    assert "40/420" in item["note"]


def test_class_imbalance_not_found_when_smote_used_without_calibration(session_id):
    _train(session_id, smote=True)
    _test(session_id, metrics={"AUROC": 0.7})
    assert _item(_report(session_id), "class_imbalance")["status"] == "not_found"


def test_class_imbalance_found_when_smote_used_and_calibration_measured(session_id):
    _train(session_id, smote=True)
    _test(session_id, metrics=CAL)
    assert _item(_report(session_id), "class_imbalance")["status"] == "found"


def test_class_imbalance_not_applicable_for_a_continuous_outcome(session_id):
    _train(session_id, model="Ridge", events=None, task="continuous",
           metrics={"R2": 0.3})
    assert _item(_report(session_id), "class_imbalance")["status"] == "not_applicable"


def test_class_imbalance_legacy_log_without_prevalence_is_manual_review(session_id):
    _train(session_id, events=None, n_train=None, n_parameters=None, task=None,
           metrics={})
    assert _item(_report(session_id), "class_imbalance")["status"] == "manual_review"


# clinical utility, model specification, eligibility ---------------------------
def test_clinical_utility_found_with_a_decision_curve(session_id):
    _train(session_id)
    _test(session_id)
    assert _item(_report(session_id), "clinical_utility")["status"] == "not_found"
    _test(session_id, dca=True)
    assert _item(_report(session_id), "clinical_utility")["status"] == "found"


def test_model_specification_found_with_adjusted_odds_ratios(session_id):
    _train(session_id)
    assert _item(_report(session_id), "model_specification")["status"] == "manual_review"
    analysis_log.log_action(session_id, "adjusted_or",
                            inputs={"outcome": "y", "features": ["age", "sex"]},
                            outputs={"n": 500, "events": 60, "n_parameters": 2})
    assert _item(_report(session_id), "model_specification")["status"] == "found"


def test_cohort_filter_marks_eligibility_found(session_id):
    assert _item(_report(session_id), "eligibility_criteria")["status"] == "manual_review"
    analysis_log.log_action(session_id, "cohort_filter", inputs={"filters": []},
                            outputs={"flow": [{"step": 1, "label": "age >= 18",
                                               "n_before": 600, "n_after": 550,
                                               "n_removed": 50}]})
    assert _item(_report(session_id), "eligibility_criteria")["status"] == "found"


def test_leakage_demo_is_ignored(session_id):
    analysis_log.log_action(session_id, "leakage_demo",
                            inputs={"outcome": "y", "model": "LogisticRegression"},
                            outputs={"arms": [{"metrics": {"AUROC": 0.99,
                                                           "CalibrationSlope": 1,
                                                           "CalibrationInTheLarge": 0}}]})
    report = _report(session_id)
    assert report["summary"]["found"] == 0
    assert _item(report, "discrimination_reported")["status"] == "not_applicable"


def test_summary_counts_match_item_statuses(session_id):
    analysis_log.log_action(session_id, "upload", inputs={"filename": "x.csv"},
                            outputs={"n_rows": 10, "n_cols": 2})
    report = _report(session_id)
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


def test_report_text_has_no_long_dashes(session_id):
    _train(session_id, smote=True)
    _test(session_id)
    analysis_log.log_action(session_id, "epi_hl", inputs={"model": "x"},
                            outputs={"hl_statistic": 4.2, "p_value": 0.6})
    report = _report(session_id)
    blob = report["disclaimer"] + "".join(
        (i["description"] or "") + (i["note"] or "") for i in report["items"])
    assert "\u2014" not in blob and "\u2013" not in blob
