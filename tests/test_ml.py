import numpy as np
import pandas as pd

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
    assert set(coords.keys()) == {"roc", "pr", "confusion", "calibration"}


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
