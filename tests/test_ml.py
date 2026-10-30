import numpy as np
import pandas as pd
import pytest

from clintab import ml


def _binary_frame(n=120, seed=0):
    rng = np.random.RandomState(seed)
    age = rng.normal(60, 10, size=n)
    sex = rng.choice(["M", "F"], size=n)
    # outcome correlated with age so models have something real to learn
    logit = (age - 60) / 10
    p = 1 / (1 + np.exp(-logit))
    outcome = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame({"age": age, "sex": sex, "outcome": outcome})


def test_determine_task():
    assert ml.determine_task(pd.Series([0, 1, 1, 0, 1])) == "binary"
    assert ml.determine_task(pd.Series(range(50))) == "continuous"
    assert ml.determine_task(pd.Series([1, 2, 3, 1, 2, 3, 1])) == "multiclass"


def test_feature_columns_excludes_outcome_and_excluded():
    df = pd.DataFrame({"a": [1], "b": [2], "outcome": [0], "admit_date": ["x"]})
    feats = ml.feature_columns(df, "outcome", ["admit_date"])
    assert feats == ["a", "b"]


def test_build_preprocessor_splits_numeric_and_categorical():
    df = _binary_frame()
    pre, num, cat = ml.build_preprocessor(df, ["age", "sex"])
    assert num == ["age"]
    assert cat == ["sex"]


def test_build_preprocessor_without_coltypes_treats_int_column_as_numeric():
    df = pd.DataFrame({"asa_class": [1, 2, 3, 2, 1], "age": [50, 60, 70, 65, 55]})
    pre, num, cat = ml.build_preprocessor(df, ["asa_class", "age"])
    assert set(num) == {"asa_class", "age"}
    assert cat == []


def test_build_preprocessor_with_coltypes_one_hot_encodes_integer_categorical():
    df = pd.DataFrame({"asa_class": [1, 2, 3, 2, 1], "age": [50, 60, 70, 65, 55]})
    coltypes = {"asa_class": "categorical", "age": "continuous"}
    pre, num, cat = ml.build_preprocessor(df, ["asa_class", "age"], coltypes=coltypes)
    assert num == ["age"]
    assert cat == ["asa_class"]


def test_encode_outcome_binary_and_continuous():
    y_bin, classes = ml.encode_outcome(pd.Series(["yes", "no", "yes"]), "binary")
    assert classes == ["no", "yes"]
    assert list(y_bin) == [1, 0, 1]

    y_cont, classes_cont = ml.encode_outcome(pd.Series([1.0, 2.0, 3.0]), "continuous")
    assert classes_cont is None
    assert list(y_cont) == [1.0, 2.0, 3.0]


def test_make_scoring_defaults():
    assert ml.make_scoring("roc", "binary") == "roc_auc"
    assert ml.make_scoring("r2", "continuous") == "r2"
    assert ml.make_scoring("unknown_metric", "binary") == "roc_auc"


def test_train_one_model_and_evaluate_binary():
    df = _binary_frame(n=160)
    train, val = df.iloc[:120], df.iloc[120:]
    feat_cols = ["age", "sex"]

    pipe, info = ml.train_one_model(
        "LogisticRegression", train, val, "outcome", feat_cols, "binary",
        do_grid_search=False)

    assert info["task"] == "binary"
    assert info["n_train"] == 120

    metrics, coords = ml.evaluate(pipe, val, "outcome", feat_cols, "binary")
    assert 0.0 <= metrics["AUROC"] <= 1.0
    assert {"roc", "pr", "confusion", "calibration"} <= set(coords.keys())


def test_train_one_model_grid_search_default_uses_val_split_not_cv():
    df = _binary_frame(n=160)
    train, val = df.iloc[:120], df.iloc[120:]
    feat_cols = ["age", "sex"]
    grid = {"clf__C": [0.1, 1.0]}

    pipe, info = ml.train_one_model(
        "RidgeLogistic", train, val, "outcome", feat_cols, "binary",
        param_grid=grid, do_grid_search=True)

    assert info["cv_folds"] is None
    assert info["best_params"] != "default"
    metrics, _ = ml.evaluate(pipe, val, "outcome", feat_cols, "binary")
    assert 0.0 <= metrics["AUROC"] <= 1.0


def test_train_one_model_grid_search_with_cv_folds_uses_kfold_on_train_only():
    df = _binary_frame(n=160)
    train, val = df.iloc[:120], df.iloc[120:]
    feat_cols = ["age", "sex"]
    grid = {"clf__C": [0.1, 1.0]}

    pipe, info = ml.train_one_model(
        "RidgeLogistic", train, val, "outcome", feat_cols, "binary",
        param_grid=grid, do_grid_search=True, cv_folds=3)

    assert info["cv_folds"] == 3
    assert info["best_params"] != "default"
    metrics, _ = ml.evaluate(pipe, val, "outcome", feat_cols, "binary")
    assert 0.0 <= metrics["AUROC"] <= 1.0


def test_binary_metrics_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    metrics, _ = ml.binary_metrics(y_true, y_score)
    assert metrics["AUROC"] == 1.0
    assert metrics["Sensitivity"] == 1.0
    assert metrics["Specificity"] == 1.0


def test_regression_metrics_basic():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    metrics, coords = ml.regression_metrics(y_true, y_pred)
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["R2"] == 1.0


def test_multiclass_metrics_without_proba_has_no_curve_coords():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2])
    metrics, coords = ml.multiclass_metrics(y_true, y_pred)
    assert metrics["Accuracy"] == pytest.approx(5 / 6, abs=1e-4)
    assert coords == {}


def test_multiclass_metrics_with_proba_returns_one_vs_rest_roc_pr():
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    # perfectly confident, correct probabilities for 3 classes
    y_proba = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
    ], dtype=float)

    metrics, coords = ml.multiclass_metrics(y_true, y_pred, y_proba)
    assert set(coords.keys()) == {"roc", "pr"}
    assert len(coords["roc"]) == 3
    assert len(coords["pr"]) == 3
    for series in coords["roc"]:
        assert series["auc"] == 1.0
    for series in coords["pr"]:
        assert series["ap"] == 1.0


def test_evaluate_multiclass_returns_roc_pr_coords():
    rng = np.random.RandomState(0)
    n = 180
    age = rng.normal(60, 10, size=n)
    grp = rng.choice([0, 1, 2], size=n)
    # outcome mostly matches group, with some noise
    outcome = np.where(rng.uniform(size=n) < 0.85, grp, rng.choice([0, 1, 2], size=n))
    df = pd.DataFrame({"age": age, "grp": grp, "outcome": outcome})
    train, val = df.iloc[:120], df.iloc[120:]
    feat_cols = ["age", "grp"]

    pipe, info = ml.train_one_model(
        "RandomForest", train, val, "outcome", feat_cols, "multiclass",
        do_grid_search=False)
    assert info["task"] == "multiclass"

    metrics, coords = ml.evaluate(pipe, val, "outcome", feat_cols, "multiclass")
    assert "MacroF1" in metrics
    assert set(coords.keys()) == {"roc", "pr"}
    assert len(coords["roc"]) == len(coords["pr"])


def test_feature_outcome_correlations_binary_ranks_by_abs_r():
    rng = np.random.RandomState(0)
    n = 200
    strong = rng.normal(0, 1, n)
    outcome = (strong + rng.normal(0, 0.1, n) > 0).astype(int)  # tightly tied to `strong`
    noise = rng.normal(0, 1, n)  # unrelated to outcome
    df = pd.DataFrame({"strong": strong, "noise": noise, "outcome": outcome})

    out = ml.feature_outcome_correlations(df, "outcome", ["strong", "noise"], "binary")
    assert list(out["feature"]) == ["strong", "noise"]
    assert abs(out.loc[0, "pearson_r"]) > abs(out.loc[1, "pearson_r"])
    assert -1.0 <= out.loc[0, "pearson_r"] <= 1.0


def test_feature_outcome_correlations_continuous_outcome():
    rng = np.random.RandomState(0)
    n = 100
    x = rng.normal(0, 1, n)
    y = 2 * x + rng.normal(0, 0.05, n)
    df = pd.DataFrame({"x": x, "y": y})

    out = ml.feature_outcome_correlations(df, "y", ["x"], "continuous")
    assert out.loc[0, "pearson_r"] > 0.95  # strong positive linear relationship


def test_feature_outcome_correlations_skips_non_numeric_and_constant_columns():
    df = pd.DataFrame({
        "cat": ["a", "b", "a", "b"],
        "constant": [1, 1, 1, 1],
        "age": [50, 60, 70, 80],
        "outcome": [0, 1, 0, 1],
    })
    out = ml.feature_outcome_correlations(df, "outcome", ["cat", "constant", "age"], "binary")
    assert list(out["feature"]) == ["age"]


def test_feature_outcome_correlations_no_usable_features_returns_empty():
    df = pd.DataFrame({"cat": ["a", "b"], "outcome": [0, 1]})
    out = ml.feature_outcome_correlations(df, "outcome", ["cat"], "binary")
    assert out.empty
    assert list(out.columns) == ["feature", "pearson_r"]


def test_predict_single_binary_returns_probability():
    df = _binary_frame(n=160)
    train, val = df.iloc[:120], df.iloc[120:]
    feat_cols = ["age", "sex"]
    pipe, info = ml.train_one_model(
        "LogisticRegression", train, val, "outcome", feat_cols, "binary",
        do_grid_search=False)

    row = {"age": 75, "sex": "F"}
    result = ml.predict_single(pipe, row, feat_cols, "binary", info["classes"])
    assert "probability" in result
    assert 0.0 <= result["probability"] <= 1.0


def test_determine_task_many_text_levels_is_multiclass_not_continuous():
    assert ml.determine_task(pd.Series(list("abcdefghijklmn"))) == "multiclass"


def test_encode_outcome_keeps_missing_outcome_as_nan_not_a_class():
    y, classes = ml.encode_outcome(pd.Series([0, 1, np.nan, 1, 0, 1.0]), "binary")
    assert classes == ["0", "1"]
    assert np.isnan(y[2])
    assert list(y[~np.isnan(y)]) == [0, 1, 1, 0, 1]


def test_encode_outcome_with_training_classes_on_a_one_class_subset():
    y, _ = ml.encode_outcome(pd.Series(["yes", "yes"]), "binary", classes=["no", "yes"])
    assert list(y) == [1, 1]          # not [0, 0], wich a fresh encoder would give


def test_encode_outcome_float_and_int_labels_are_the_same_class():
    y, _ = ml.encode_outcome(pd.Series([0, 1, 1]), "binary", classes=["0.0", "1.0"])
    assert list(y) == [0, 1, 1]


def test_encode_outcome_unseen_label_becomes_nan():
    y, _ = ml.encode_outcome(pd.Series(["a", "z"]), "multiclass", classes=["a", "b"])
    assert y[0] == 0 and np.isnan(y[1])


def test_train_and_evaluate_with_missing_outcome_rows():
    df = _binary_frame(n=160)
    df["outcome"] = df["outcome"].astype(float)
    df.loc[[3, 10, 50, 130, 140], "outcome"] = np.nan
    train, val = df.iloc[:120], df.iloc[120:]

    pipe, info = ml.train_one_model(
        "LogisticRegression", train, val, "outcome", ["age", "sex"], "binary",
        do_grid_search=False)
    assert info["classes"] == ["0", "1"]
    assert info["n_train"] == 120 - 3
    metrics, _ = ml.evaluate(pipe, val, "outcome", ["age", "sex"], "binary",
                             classes=info["classes"])
    assert metrics["N"] == 40 - 2
    assert 0.0 <= metrics["AUROC"] <= 1.0


def test_train_info_records_whether_hyperparameters_were_tuned():
    df = _binary_frame(n=160)
    train, val = df.iloc[:120], df.iloc[120:]
    _, tuned = ml.train_one_model("RidgeLogistic", train, val, "outcome",
                                  ["age", "sex"], "binary",
                                  param_grid={"clf__C": [0.1, 1.0]}, do_grid_search=True)
    _, untuned = ml.train_one_model("LogisticRegression", train, val, "outcome",
                                    ["age", "sex"], "binary", do_grid_search=False)
    assert tuned["tuned"] is True
    assert untuned["tuned"] is False


def test_multiclass_metrics_when_a_class_is_absent_from_y_true():
    # the model knows 3 classes, this slice of data only contains classes 0 and 2
    y_true = np.array([0, 2, 0, 2])
    y_pred = np.array([0, 2, 0, 2])
    y_proba = np.array([[0.9, 0.05, 0.05], [0.05, 0.05, 0.9],
                        [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
    _, coords = ml.multiclass_metrics(y_true, y_pred, y_proba)
    assert [s["name"] for s in coords["roc"]] == ["class 0", "class 2"]
    assert all(s["auc"] == 1.0 for s in coords["roc"])


def test_encode_outcome_keeps_01_and_1_apart_but_folds_1_0_into_1():
    y, classes = ml.encode_outcome(pd.Series(["01", "1", "2"]), "multiclass")
    assert classes == ["01", "1", "2"]
    y, classes = ml.encode_outcome(pd.Series([1.0, 2.0, 1.0]), "binary")
    assert classes == ["1", "2"]


# ---------------------------------------------------------------------------
# model catalogue: unpenalized reference model, no deprecation warnings
# ---------------------------------------------------------------------------
def _aor_fixture():
    # arithmetic only (no libm calls) so the data are identical everywhere.
    # the R golden values below were computed on exactly this frame.
    rng = np.random.RandomState(7)
    n = 300
    age = np.round(rng.uniform(40, 90, n), 1)
    bmi = np.round(rng.uniform(18, 40, n), 1)
    sex = np.where(rng.uniform(size=n) < 0.45, "F", "M").astype(object)
    asa = rng.randint(1, 5, n).astype(float)
    smoker = (rng.uniform(size=n) < 0.3).astype(int)
    risk = (0.02 + 0.004 * (age - 40) + 0.01 * (bmi - 18) + 0.08 * (sex == "M")
            + 0.05 * (asa - 1) + 0.1 * smoker)
    outcome = (rng.uniform(size=n) < risk).astype(int)
    age[[3, 50, 120]] = np.nan
    sex[[7, 99]] = np.nan
    asa[[11, 200]] = np.nan
    return pd.DataFrame({"age": age, "bmi": bmi, "sex": sex, "asa": asa,
                         "smoker": smoker, "outcome": outcome})


_AOR_COLTYPES = {"age": "continuous", "bmi": "continuous", "sex": "binary",
                 "asa": "categorical", "smoker": "binary"}
_AOR_FEATS = ["age", "bmi", "sex", "asa", "smoker"]

# R 4.3.3, base glm on the fixture after median (age) / mode (sex = M, asa = 1,
# the first of the tied levels 1 and 2) imputation:
#   glm(outcome ~ age + bmi + sex + factor(asa) + factor(smoker), binomial)
#   exp(coef), exp(confint.default), summary()$coefficients[, 4]
_R_GLM = {
    "age": (1.02091668802, 1.00348662276, 1.03864950487, 0.0184674914987),
    "bmi": (1.04740660403, 1.0083042484, 1.08802536131, 0.0170335780631),
    "sex=M": (1.74295046821, 1.05705819028, 2.87389697422, 0.0294476891122),
    "asa=2": (1.38029898711, 0.683998226041, 2.78542432025, 0.368266880012),
    "asa=3": (1.29146753853, 0.636658967451, 2.61975168552, 0.478463043669),
    "asa=4": (1.79816497619, 0.891931383285, 3.62516370901, 0.100950658806),
    "smoker=1": (1.76248679009, 1.05569191301, 2.94248695759, 0.0302183529396),
}


def test_catalogue_still_has_15_models():
    assert len(ml.model_catalogue("binary")) + len(ml.model_catalogue("continuous")) == 15
    assert set(ml.model_catalogue("binary")) == set(ml.model_catalogue("multiclass"))


def test_logistic_regression_is_unpenalized_with_no_grid():
    cfg = ml.model_catalogue("binary")["LogisticRegression"]
    assert cfg["params"] == {}
    clf = cfg["model"]
    if ml._LOGREG_L1_RATIO_API:
        assert clf.C == np.inf
    else:
        assert clf.penalty is None
    assert ml.model_catalogue("binary")["RidgeLogistic"]["params"] == {"clf__C": [0.01, 0.1, 1, 10]}


def test_unpenalized_logistic_regression_matches_r_glm():
    # per-unit ORs recovered from the scaled sklearn coefficients. agreement
    # is about 1e-7 on scikit-learn 1.9 and about 1e-5 on 1.3 (lbfgs stops
    # earlier there); any penalty would be off by far more than 1e-4.
    df = _aor_fixture()
    pipe, info = ml.train_one_model("LogisticRegression", df, df, "outcome", _AOR_FEATS,
                                    "binary", coltypes=_AOR_COLTYPES)
    pre = pipe.named_steps["pre"]
    names = list(pre.get_feature_names_out())
    scale = pre.named_transformers_["num"].named_steps["scale"].scale_
    coef = pipe.named_steps["clf"].coef_.ravel()
    or_age = np.exp(coef[names.index("num__age")] / scale[0])
    or_bmi = np.exp(coef[names.index("num__bmi")] / scale[1])
    or_sex = np.exp(coef[names.index("cat__sex_M")] - coef[names.index("cat__sex_F")])
    assert or_age == pytest.approx(_R_GLM["age"][0], rel=1e-4)
    assert or_bmi == pytest.approx(_R_GLM["bmi"][0], rel=1e-4)
    assert or_sex == pytest.approx(_R_GLM["sex=M"][0], rel=1e-4)


def test_logistic_regression_ignores_a_penalty_grid():
    df = _binary_frame(n=160)
    train, val = df.iloc[:120], df.iloc[120:]
    pipe, info = ml.train_one_model("LogisticRegression", train, val, "outcome",
                                    ["age", "sex"], "binary",
                                    param_grid={"clf__C": [0.01, 0.1]}, do_grid_search=True)
    assert info["tuned"] is False and info["best_params"] == "default"
    if ml._LOGREG_L1_RATIO_API:
        assert pipe.named_steps["clf"].C == np.inf


def _deprecations(caught):
    return [str(w.message) for w in caught
            if issubclass(w.category, (FutureWarning, DeprecationWarning))
            and "deprecat" in str(w.message).lower()]


def test_every_catalogue_model_fits_without_deprecation_warnings():
    import warnings
    from sklearn.model_selection import ParameterGrid

    df = _binary_frame(n=160)
    df["grp"] = np.arange(len(df)) % 3
    df["y_cont"] = df["age"] * 0.3 + np.arange(len(df)) % 7
    train, val = df.iloc[:120], df.iloc[120:]
    for task, outcome in (("binary", "outcome"), ("multiclass", "grp"),
                          ("continuous", "y_cont")):
        for name, cfg in ml.model_catalogue(task).items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pipe, _ = ml.train_one_model(name, train, val, outcome, ["age", "sex"],
                                             task, do_grid_search=False)
                # every grid value of the logistic family and the svm, in process
                # (grid search runs in worker processes, where warnings are lost)
                if "Logistic" in name or name == "SVM":
                    for combo in ParameterGrid(cfg["params"] or [{}]):
                        pipe.set_params(**combo).fit(train[["age", "sex"]],
                                                     ml.encode_outcome(train[outcome], task)[0])
            assert _deprecations(caught) == [], (task, name, _deprecations(caught))


def test_svm_grid_accepts_old_style_parameter_names():
    df = _binary_frame(n=160)
    train, val = df.iloc[:120], df.iloc[120:]
    _, info = ml.train_one_model("SVM", train, val, "outcome", ["age", "sex"], "binary",
                                 param_grid={"clf__C": [0.5, 2.0]}, do_grid_search=True)
    assert set(info["best_params"]) == {"clf__estimator__C"}
    pipe, _ = ml.train_one_model("SVM", train, val, "outcome", ["age", "sex"], "binary",
                                 do_grid_search=False)
    proba = pipe.predict_proba(val[["age", "sex"]])
    assert proba.shape == (40, 2)


# ---------------------------------------------------------------------------
# binary_metrics: backward compatible keys + calibration, CIs, flexible
# calibration and decision curve
# ---------------------------------------------------------------------------
_OLD_METRIC_KEYS = ["AUROC", "AUPRC", "Sensitivity", "Specificity", "PPV", "NPV",
                    "F1", "Brier", "Threshold", "N"]


def _scores(n=400, seed=3):
    rng = np.random.RandomState(seed)
    x = rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(-1 + x)))).astype(int)
    p = 1 / (1 + np.exp(-(-0.8 + 1.3 * x)))
    return y, p


def test_binary_metrics_keeps_old_keys_and_coordinates():
    y, p = _scores()
    metrics, coords = ml.binary_metrics(y, p, threshold=0.3)
    for k in _OLD_METRIC_KEYS:
        assert k in metrics
    assert metrics["Threshold"] == 0.3 and metrics["N"] == 400
    assert set(coords["roc"]) == {"fpr", "tpr", "auc"}
    assert set(coords["pr"]) == {"precision", "recall", "ap"}
    assert set(coords["confusion"]) == {"tn", "fp", "fn", "tp"}
    assert set(coords["calibration"]) == {"prob_true", "prob_pred"}
    # old positional call still works
    m2, _ = ml.binary_metrics(y, p, 0.3)
    assert m2["Sensitivity"] == metrics["Sensitivity"]


def test_binary_metrics_adds_calibration_and_intervals():
    from clintab import validation as V
    y, p = _scores()
    metrics, coords = ml.binary_metrics(y, p)
    cal = V.calibration_stats(y, p)
    assert metrics["CalibrationSlope"] == round(cal["slope"], 4)
    assert metrics["CalibrationInTheLarge"] == round(cal["citl"], 4)
    assert metrics["OE_ratio"] == round(cal["oe_ratio"], 4)
    assert metrics["ICI"] == round(cal["ici"], 4)
    ci = metrics["CI"]
    assert set(ci) == {"AUROC", "AUPRC", "Brier", "CalibrationSlope",
                       "CalibrationInTheLarge", "OE_ratio"}
    for k, (lo, hi) in ci.items():
        assert lo <= hi, k
    assert ci["AUROC"][0] <= metrics["AUROC"] <= ci["AUROC"][1]
    assert ci["AUPRC"][0] <= metrics["AUPRC"] <= ci["AUPRC"][1]
    assert ci["Brier"][0] <= metrics["Brier"] <= ci["Brier"][1]
    _, lo, hi = V.auroc_ci(y, p)
    assert ci["AUROC"] == [round(lo, 4), round(hi, 4)]

    flex = coords["calibration_flexible"]
    assert flex["method"] == "lowess" and len(flex["p_grid"]) == len(flex["observed"]) == 100
    dca = coords["dca"]
    assert set(dca) == {"threshold", "net_benefit", "treat_all", "treat_none"}
    assert len(dca["threshold"]) == 99
    assert coords["calibration_stats"]["e90"] is not None
    hist = coords["risk_histogram"]
    assert len(hist["edges"]) == 41 and len(hist["counts"]) == 40
    assert sum(hist["counts"]) == 400
    assert hist["edges"][-1] == pytest.approx(p.max(), abs=1e-6)


def test_binary_metrics_is_deterministic_and_ci_can_be_switched_off():
    y, p = _scores()
    a, _ = ml.binary_metrics(y, p)
    b, _ = ml.binary_metrics(y, p)
    assert a == b
    m, coords = ml.binary_metrics(y, p, with_ci=False)
    assert "CI" not in m
    assert "CalibrationSlope" in m and "dca" in coords


def test_binary_metrics_one_class_gives_none_not_a_crash():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, coords = ml.binary_metrics(np.zeros(20, dtype=int), np.linspace(0.05, 0.5, 20))
    assert m["CalibrationSlope"] is None and m["CalibrationInTheLarge"] is None
    assert m["CI"]["AUROC"] == [None, None]
    assert coords["calibration_stats"]["reason"]


def test_evaluate_passes_with_ci_through():
    df = _binary_frame(n=160)
    train, val = df.iloc[:120], df.iloc[120:]
    pipe, _ = ml.train_one_model("LogisticRegression", train, val, "outcome",
                                 ["age", "sex"], "binary", do_grid_search=False)
    m, _ = ml.evaluate(pipe, val, "outcome", ["age", "sex"], "binary")
    assert "CI" in m
    m, _ = ml.evaluate(pipe, val, "outcome", ["age", "sex"], "binary", with_ci=False)
    assert "CI" not in m


# ---------------------------------------------------------------------------
# adjusted odds ratios + parameter counts
# ---------------------------------------------------------------------------
def test_adjusted_odds_ratios_match_r_glm():
    r = ml.adjusted_odds_ratios(_aor_fixture(), "outcome", _AOR_FEATS, _AOR_COLTYPES)
    assert [row["term"] for row in r["rows"]] == list(_R_GLM)
    for row in r["rows"]:
        odds, lo, hi, p = _R_GLM[row["term"]]
        assert row["or"] == pytest.approx(odds, rel=1e-6), row["term"]
        assert row["ci_low"] == pytest.approx(lo, rel=1e-6), row["term"]
        assert row["ci_high"] == pytest.approx(hi, rel=1e-6), row["term"]
        assert row["p"] == pytest.approx(p, rel=1e-6), row["term"]
    assert r["n"] == 300 and r["events"] == 111
    assert r["n_parameters"] == 7
    assert r["epp"] == round(111 / 7, 2)
    assert r["reference_levels"] == {"sex": "F", "asa": "1", "smoker": "0"}
    assert r["imputed"] == {"age": 63.9, "sex": "M", "asa": "1"}
    assert r["n_missing"] == {"age": 3, "sex": 2, "asa": 2}
    assert r["converged"] is True and r["warning"] is None
    assert r["outcome_level"] == "1"


def test_adjusted_odds_ratios_reference_override():
    r = ml.adjusted_odds_ratios(_aor_fixture(), "outcome", _AOR_FEATS, _AOR_COLTYPES,
                                reference={"sex": "M"})
    row = next(x for x in r["rows"] if x["variable"] == "sex")
    assert row["term"] == "sex=F" and row["level"] == "F"
    assert row["or"] == pytest.approx(1 / _R_GLM["sex=M"][0], rel=1e-6)
    assert r["reference_levels"]["sex"] == "M"


def test_count_parameters():
    df = _aor_fixture()
    assert ml.count_parameters(df, _AOR_FEATS, _AOR_COLTYPES) == 7
    # without coltypes asa and smoker are numeric, sex is text -> 1 + 1 + 1 + 1 + 1
    assert ml.count_parameters(df, _AOR_FEATS) == 5
    df["empty"] = np.nan
    assert ml.count_parameters(df, ["empty"]) == 0


def test_adjusted_odds_ratios_flags_separation_and_collinearity_without_crashing():
    import warnings
    df = _aor_fixture()
    df["grp"] = np.where(np.arange(len(df)) < 20, "rare", "common")
    df.loc[:19, "outcome"] = 0                    # no events in the rare group
    df["const"] = 5.0
    df["age_twice"] = df["age"] * 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")             # nothing leaks out as a warning
        r = ml.adjusted_odds_ratios(df, "outcome", ["age", "grp", "const", "age_twice"])
    assert [row["term"] for row in r["rows"]] == ["age", "grp=rare"]
    assert "separation" in r["warning"] and "grp=rare" in r["warning"]
    assert "const" in r["warning"] and "age_twice" in r["warning"]
    assert r["n_parameters"] == 4


def test_adjusted_odds_ratios_complete_separation():
    d = pd.DataFrame({"x": np.arange(40.0), "outcome": (np.arange(40) >= 20).astype(int)})
    r = ml.adjusted_odds_ratios(d, "outcome", ["x"])
    assert "separation" in r["warning"] and "x" in r["warning"]
    assert len(r["rows"]) == 1


def test_adjusted_odds_ratios_text_outcome_imputation_and_bad_outcomes():
    d = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0, 5.0, 2.0, 6.0, 7.0],
                      "o": ["no", "yes", "no", "yes", "no", None, "yes", "no"]})
    r = ml.adjusted_odds_ratios(d, "o", ["x"])
    assert r["n"] == 7 and r["events"] == 3        # missing outcome row dropped
    assert r["outcome_level"] == "yes"
    assert r["imputed"] == {"x": 4.5}              # median of the rows kept
    with pytest.raises(ValueError):
        ml.adjusted_odds_ratios(pd.DataFrame({"x": [1, 2, 3], "o": [0, 1, 2]}), "o", ["x"])
    with pytest.raises(ValueError):
        ml.adjusted_odds_ratios(pd.DataFrame({"x": [1, 2, 3], "o": [1, 1, 1]}), "o", ["x"])

