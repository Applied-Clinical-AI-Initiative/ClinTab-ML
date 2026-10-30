import hashlib
import io
import json
import os

import numpy as np
import pandas as pd
import pytest

from clintab import analysis_log, store


@pytest.fixture
def client(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(store, "RUNTIME", str(runtime))
    monkeypatch.setattr(store, "SESSIONS", str(runtime / "sessions"))
    monkeypatch.setattr(store, "EXPORTS", str(runtime / "exports"))
    monkeypatch.setattr(store, "MODELS", str(tmp_path / "models"))
    from app import create_app
    return create_app().test_client()


def _csv(df):
    return (io.BytesIO(df.to_csv(index=False).encode()), "cohort.csv")


def _cohort(n=300, seed=0, rare_positives_late=False):
    rng = np.random.RandomState(seed)
    age = rng.normal(60, 10, n)
    outcome = (rng.uniform(size=n) < 1 / (1 + np.exp(-(age - 65) / 5))).astype(int)
    if rare_positives_late:
        outcome[:250] = 0
        outcome[250:] = rng.randint(0, 2, n - 250)
    return pd.DataFrame({"age": age, "sex": rng.choice(["M", "F"], n), "outcome": outcome})


def _confirm(client, df, **body):
    r = client.post("/api/upload", data={"file": _csv(df)})
    assert r.status_code == 200
    types = {c["name"]: c["type"] for c in r.get_json()["columns"]}
    payload = {"types": types, "missing": {}, "ratios": [70, 15, 15], "method": "random"}
    payload.update(body)
    return client.post("/api/confirm", json=payload)


def test_errors_come_back_as_json_not_html(client):
    r = client.get("/api/nonexistent")
    assert r.status_code == 404
    assert "error" in r.get_json()


def test_unparseable_upload_does_not_leave_a_session_behind(client):
    r = client.post("/api/upload", data={"file": (io.BytesIO(b""), "empty.csv")})
    assert r.status_code == 400
    assert os.listdir(store.SESSIONS) == []


def test_upload_rejects_non_csv(client):
    r = client.post("/api/upload", data={"file": (io.BytesIO(b"a,b\n1,2\n"), "x.txt")})
    assert r.status_code == 400


def test_confirm_rejects_a_zero_share_split(client):
    r = _confirm(client, _cohort(), ratios=[80, 20, 0])
    assert r.status_code == 400


def test_confirm_stratified_without_a_column_is_an_error_not_a_silent_random_split(client):
    r = _confirm(client, _cohort(), method="stratified", stratify_col=None)
    assert r.status_code == 400


def test_confirm_stratified_logs_the_method_that_was_really_used(client):
    r = _confirm(client, _cohort(), method="stratified", stratify_col="outcome")
    assert r.status_code == 200
    assert r.get_json()["method"] == "stratified"
    r = _confirm(client, _cohort(), method="random")
    assert r.get_json()["method"] == "random"


def test_train_config_sees_a_rare_class_that_is_not_in_the_first_rows(client):
    # every outcome in the first 250 rows is 0, so a head(200) look says "one class"
    assert _confirm(client, _cohort(n=400, rare_positives_late=True)).status_code == 200
    r = client.post("/api/train", json={"outcome": "outcome", "models": ["LogisticRegression"]})
    assert r.status_code == 200
    assert r.get_json()["task"] == "binary"


def test_train_config_rejects_unknown_outcome_and_model(client):
    _confirm(client, _cohort())
    assert client.post("/api/train", json={"outcome": "nope", "models": []}).status_code == 400
    r = client.post("/api/train", json={"outcome": "outcome", "models": ["Nonsense"]})
    assert r.status_code == 400


def _train(client, **cfg):
    body = {"outcome": "outcome", "models": ["LogisticRegression"], "grid_search": False}
    body.update(cfg)
    assert client.post("/api/train", json=body).status_code == 200
    stream = client.get("/api/train/stream").get_data(as_text=True)
    return [json.loads(chunk[6:]) for chunk in stream.split("\n\n") if chunk.startswith("data: ")]


def test_train_then_test_round_trip(client):
    _confirm(client, _cohort())
    events = _train(client)
    done = [e for e in events if e["event"] == "model_done"]
    assert len(done) == 1
    name = done[0]["saved_as"]

    r = client.post("/api/test", data={"models": [name], "source": "test"})
    assert r.status_code == 200
    assert 0 <= r.get_json()["models"][0]["metrics"]["AUROC"] <= 1


def test_confounders_are_forced_in_even_if_also_excluded(client):
    _confirm(client, _cohort())
    events = _train(client, exclude=["age", "sex"], confounders=["age"])
    name = [e for e in events if e["event"] == "model_done"][0]["saved_as"]
    meta = json.load(open(store.model_meta_path(name)))
    assert "age" in meta["features"] and "sex" not in meta["features"]
    assert meta["confounders"] == ["age"]


def test_untuned_training_is_logged_as_untuned(client):
    _confirm(client, _cohort())
    _train(client, grid_search=False)
    text = client.get("/api/report/methods").get_json()["text"]
    assert "default hyperparameters" in text


def test_long_outcome_names_do_not_overwrite_earlier_models(client):
    df = _cohort().rename(columns={"outcome": "o" * 90})
    _confirm(client, df)
    first = _train(client, outcome="o" * 90)
    names = {e["saved_as"] for e in first if e["event"] == "model_done"}
    assert all(n.split("_")[-1].count("-") == 1 for n in names)     # timestamp survived


def test_test_endpoint_refuses_path_traversal_model_names(client):
    _confirm(client, _cohort())
    r = client.post("/api/test", data={"models": ["../../etc/passwd"], "source": "test"})
    assert r.status_code == 200
    assert "error" in r.get_json()["models"][0]


def test_test_endpoint_upload_source_needs_a_file(client):
    _confirm(client, _cohort())
    r = client.post("/api/test", data={"models": ["x"], "source": "upload"})
    assert r.status_code == 400


def test_model_features_without_a_name_is_a_404_not_a_crash(client):
    assert client.get("/api/model/features").status_code == 404


def test_model_upload_only_accepts_pickle_files(client):
    r = client.post("/api/model/upload", data={"file": (io.BytesIO(b"x"), "model.txt")})
    assert r.status_code == 400


def test_epi_or_rejects_negative_counts_with_a_400(client):
    r = client.post("/api/epi/or", json={"a": -1, "b": 2, "c": 3, "d": 4})
    assert r.status_code == 400


def test_render_pdf_unknown_chart_kind_is_a_400(client):
    r = client.post("/api/render/pdf", json={"kind": "nope", "data": {}})
    assert r.status_code == 400


def test_session_cookie_secret_is_not_the_old_hardcoded_default(client):
    assert client.application.secret_key != "clintab-dev-secret-change-me"
    assert os.path.exists(os.path.join(store.RUNTIME, ".secret_key"))


def test_train_columns_reports_the_task_the_server_will_use(client):
    # a numeric column with more than 10 levels is continuous even if it was overridden to categorical
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"age": rng.normal(60, 10, 200), "score": rng.randint(0, 40, 200),
                       "outcome": rng.randint(0, 2, 200)})
    r = client.post("/api/upload", data={"file": _csv(df)})
    types = {c["name"]: c["type"] for c in r.get_json()["columns"]}
    types["score"] = "categorical"
    assert client.post("/api/confirm", json={"types": types, "missing": {}, "ratios": [70, 15, 15],
                                             "method": "random"}).status_code == 200
    outs = {o["name"]: o for o in client.get("/api/train/columns").get_json()["outcomes"]}
    assert outs["outcome"]["task"] == "binary"
    assert outs["score"]["task"] == "continuous"
    assert outs["age"]["task"] == "continuous"


# ---------------------------------------------------------------------------
# missing data: fill values learned on the training partition only
# ---------------------------------------------------------------------------
def _sid(client):
    return client.get("/api/sessions").get_json()["active"]


def _log(client):
    return analysis_log.read_log(_sid(client))


def _missing_cohort(n=300, seed=1):
    df = _cohort(n=n, seed=seed)
    rng = np.random.RandomState(seed)
    df["alb"] = rng.normal(3.5, 0.6, n)
    df.loc[rng.uniform(size=n) < 0.44, "alb"] = np.nan        # nsqip-like albumin
    df["mrn"] = np.arange(n)
    return df


def test_confirm_fills_from_the_training_partition_not_the_whole_column(client):
    df = _missing_cohort()
    r = client.post("/api/upload", data={"file": _csv(df)})
    types = {c["name"]: c["type"] for c in r.get_json()["columns"]}
    types["mrn"] = "continuous"           # kept so train rows can be traced back
    r = client.post("/api/confirm", json={"types": types, "missing": {"alb": "impute"},
                                          "ratios": [70, 15, 15], "method": "random"})
    assert r.status_code == 200
    fill = r.get_json()["fill_values"]["alb"]
    sid = _sid(client)
    train = pd.read_csv(store.session_path(sid, "train.csv"))
    raw_train = df.set_index("mrn").loc[train["mrn"], "alb"]
    assert fill == pytest.approx(raw_train.median())
    assert fill != pytest.approx(df["alb"].median())           # not the pooled median
    for part in ("train", "val", "test", "full"):
        p = pd.read_csv(store.session_path(sid, f"{part}.csv"))
        assert p["alb"].isna().sum() == 0
    test = pd.read_csv(store.session_path(sid, "test.csv"))
    filled = df.set_index("mrn").loc[test["mrn"], "alb"].isna().values
    assert np.allclose(test["alb"][filled], fill)


def test_confirm_impute_indicator_adds_a_binary_column_everywhere(client):
    r = _confirm(client, _missing_cohort(), missing={"alb": "impute_indicator"})
    assert r.status_code == 200
    assert r.get_json()["indicators"] == {"alb": "alb_missing"}
    sid = _sid(client)
    for part in ("train", "val", "test", "full"):
        p = pd.read_csv(store.session_path(sid, f"{part}.csv"))
        assert set(p["alb_missing"].unique()) <= {0, 1}
        assert p["alb"].isna().sum() == 0
    assert store.load_meta(sid)["coltypes"]["alb_missing"] == "binary"


def test_confirm_include_leaves_missing_values_for_the_model_pipeline(client):
    assert _confirm(client, _missing_cohort(), missing={"alb": "include"}).status_code == 200
    train = pd.read_csv(store.session_path(_sid(client), "train.csv"))
    assert train["alb"].isna().sum() > 0


def test_confirm_rejects_an_unknown_missing_decision(client):
    r = _confirm(client, _missing_cohort(), missing={"alb": "mean_of_everything"})
    assert r.status_code == 400


def test_confirm_logs_missingness_before_handling_and_the_fill_values(client):
    df = _missing_cohort()
    r = client.post("/api/upload", data={"file": _csv(df)})
    types = {c["name"]: c["type"] for c in r.get_json()["columns"]}
    types["mrn"] = "exclude"
    client.post("/api/confirm", json={"types": types, "missing": {"alb": "impute"},
                                      "ratios": [70, 15, 15], "method": "random", "seed": 11})
    entry = [e for e in _log(client) if e["action"] == "confirm"][-1]
    assert entry["inputs"]["missing_pct"] == {"alb": round(100 * df["alb"].isna().mean(), 2)}
    assert entry["inputs"]["missing"] == {"alb": "impute"}
    assert entry["inputs"]["seed"] == 11
    assert entry["inputs"]["excluded"] == ["mrn"]
    assert entry["inputs"]["method"] == "random"
    assert "alb" in entry["outputs"]["fill_values"]


def test_confirm_imbalance_note_replaces_the_smote_suggestion(client):
    df = _cohort(n=400)
    df["outcome"] = (np.arange(400) % 10 == 0).astype(int)          # 10% positive
    r = _confirm(client, df, method="stratified", stratify_col="outcome")
    hint = r.get_json()["smote_hint"]
    assert hint["imbalanced"] is True and hint["minority_fraction"] == 0.1
    assert "suggest_smote" not in hint


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def test_upload_logs_file_hash_and_environment(client):
    df = _cohort()
    raw = df.to_csv(index=False).encode()
    client.post("/api/upload", data={"file": (io.BytesIO(raw), "cohort.csv")})
    entry = _log(client)[0]
    assert entry["action"] == "upload"
    assert entry["inputs"]["sha256"] == hashlib.sha256(raw).hexdigest()
    env = entry["outputs"]["environment"]
    assert env["scikit-learn"] and env["python"] and "platform" in env


def test_train_model_logs_events_n_train_and_parameters(client):
    _confirm(client, _cohort())
    _train(client)
    entry = [e for e in _log(client) if e["action"] == "train_model"][-1]
    sid = _sid(client)
    train = pd.read_csv(store.session_path(sid, "train.csv"))
    assert entry["outputs"]["events"] == int(train["outcome"].sum())
    assert entry["outputs"]["n_train"] == len(train)
    assert entry["outputs"]["n_parameters"] == 2                # age + sex (reference dropped)
    assert entry["inputs"]["task"] == "binary"


def test_epi_calculators_are_logged_when_a_session_is_active(client):
    assert client.post("/api/epi/or", json={"a": 20, "b": 80, "c": 10, "d": 90}).status_code == 200
    _confirm(client, _cohort())
    client.post("/api/epi/or", json={"a": 20, "b": 80, "c": 10, "d": 90})
    client.post("/api/epi/nnt", json={"a": 20, "b": 80, "c": 10, "d": 90})
    by_action = {e["action"]: e for e in _log(client)}
    assert by_action["epi_or"]["inputs"] == {"a": 20, "b": 80, "c": 10, "d": 90}
    assert by_action["epi_or"]["outputs"]["odds_ratio"] == 2.25
    assert by_action["epi_nnt"]["outputs"]["measure"] == "NNH"


def test_verify_log_endpoint_detects_tampering(client):
    _confirm(client, _cohort())
    _train(client)
    r = client.get("/api/report/verify-log").get_json()
    # upload, confirm, train_model, adjusted_or (logistic regression)
    assert r["ok"] is True and r["n_entries"] == 4 and r["legacy"] is False
    path = analysis_log.log_path(_sid(client))
    with open(path) as f:
        lines = f.read().splitlines()
    lines[0] = lines[0].replace("cohort.csv", "other.csv")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    r = client.get("/api/report/verify-log").get_json()
    assert r["ok"] is False and r["first_bad_line"] == 2


def test_verify_log_needs_a_session(client):
    assert client.get("/api/report/verify-log").status_code == 400


# ---------------------------------------------------------------------------
# test-set lock + external validation logging
# ---------------------------------------------------------------------------
def _two_models(client):
    _confirm(client, _cohort())
    events = _train(client, models=["LogisticRegression", "DecisionTree"])
    return [e["saved_as"] for e in events if e["event"] == "model_done"]


def test_first_internal_test_locks_the_partition_to_those_models(client):
    lr, dt = _two_models(client)
    r = client.post("/api/test", data={"models": [lr], "source": "test"})
    assert r.status_code == 200
    assert r.get_json()["test_lock"]["models"] == [lr]
    meta = store.load_meta(_sid(client))
    assert meta["test_lock"]["models"] == [lr] and meta["test_lock"]["locked_at"]
    entry = [e for e in _log(client) if e["action"] == "test_model"][-1]
    assert entry["inputs"]["source"] == "internal_test"
    assert entry["inputs"]["model_type"] == "LogisticRegression"
    assert entry["inputs"]["outcome"] == "outcome"


def test_retesting_a_locked_model_at_a_new_threshold_is_allowed(client):
    lr, dt = _two_models(client)
    client.post("/api/test", data={"models": [lr], "source": "test"})
    r = client.post("/api/test", data={"models": [lr], "source": "test", "threshold": "0.2"})
    assert r.status_code == 200


def test_testing_a_new_model_on_a_locked_partition_needs_a_reason(client):
    lr, dt = _two_models(client)
    client.post("/api/test", data={"models": [lr], "source": "test"})
    r = client.post("/api/test", data={"models": [lr, dt], "source": "test"})
    assert r.status_code == 409
    body = r.get_json()
    assert body["locked"] is True and body["error"] and body["new_models"] == [dt]
    r = client.post("/api/test", data={"models": [dt], "source": "test", "unlock_reason": "   "})
    assert r.status_code == 409


def test_unlock_reason_proceeds_logs_and_extends_the_lock(client):
    lr, dt = _two_models(client)
    client.post("/api/test", data={"models": [lr], "source": "test"})
    r = client.post("/api/test", data={"models": [dt], "source": "test",
                                       "unlock_reason": "reviewer asked for a tree"})
    assert r.status_code == 200
    unlock = [e for e in _log(client) if e["action"] == "test_unlock"]
    assert len(unlock) == 1
    assert unlock[0]["inputs"] == {"reason": "reviewer asked for a tree", "models": [dt]}
    assert store.load_meta(_sid(client))["test_lock"]["models"] == [lr, dt]
    # now both are locked in, so no reason is needed again
    assert client.post("/api/test", data={"models": [lr, dt], "source": "test"}).status_code == 200
    text = client.get("/api/report/methods").get_json()["text"]
    assert "reused" in text and "reviewer asked for a tree" in text


def test_hosmer_lemeshow_on_the_test_set_is_under_the_same_lock(client):
    lr, dt = _two_models(client)
    client.post("/api/test", data={"models": [lr], "source": "test"})
    assert client.post("/api/epi/hl", json={"model": lr}).status_code == 200
    r = client.post("/api/epi/hl", json={"model": dt})
    assert r.status_code == 409 and r.get_json()["locked"] is True
    assert client.post("/api/epi/hl", json={"model": dt, "unlock_reason": "why not"}).status_code == 200


def test_external_test_is_logged_and_never_locked(client):
    lr, dt = _two_models(client)
    client.post("/api/test", data={"models": [lr], "source": "test"})
    ext = _cohort(n=120, seed=5)
    raw = ext.to_csv(index=False).encode()
    r = client.post("/api/test", data={"models": [dt], "source": "upload",
                                       "file": (io.BytesIO(raw), "site_b.csv")})
    assert r.status_code == 200
    entry = [e for e in _log(client) if e["action"] == "test_model"][-1]
    assert entry["inputs"]["source"] == "external"
    assert entry["inputs"]["filename"] == "site_b.csv"
    assert entry["inputs"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert entry["inputs"]["n"] == 120
    assert store.load_meta(_sid(client))["test_lock"]["models"] == [lr]
    report = client.get("/api/report/tripod").get_json()
    status = {i["id"]: i["status"] for i in report["items"]}
    assert status["external_validation"] == "found"
    assert status["internal_validation"] == "found"


def test_internal_test_alone_is_not_external_validation(client):
    lr, _ = _two_models(client)
    client.post("/api/test", data={"models": [lr], "source": "test"})
    report = client.get("/api/report/tripod").get_json()
    status = {i["id"]: i["status"] for i in report["items"]}
    assert status["external_validation"] == "not_found"
    assert status["internal_validation"] == "found"
    assert status["sample_size"] == "manual_review"


def test_external_data_gets_the_models_training_fitted_missing_handling(client):
    df = _missing_cohort()
    _confirm(client, df.drop(columns=["mrn"]), missing={"alb": "impute_indicator"})
    events = _train(client)
    name = [e for e in events if e["event"] == "model_done"][0]["saved_as"]
    # an external file has no alb_missing column and still has blanks in alb
    ext = _missing_cohort(n=150, seed=8).drop(columns=["mrn"])
    r = client.post("/api/test", data={"models": [name], "source": "upload",
                                       "file": _csv(ext)})
    assert "metrics" in r.get_json()["models"][0], r.get_json()
    assert [e for e in _log(client) if e["action"] == "test_model"][-1]["inputs"]["source"] == "external"
    feats = client.get(f"/api/model/features?name={name}").get_json()
    assert feats["derived_features"] == ["alb_missing"]
    # a blank albumin in a single prediction sets the indicator, not a form field
    row = {"age": 70, "sex": "M"}
    r = client.post("/api/predict-single", json={"model": name, "row": row})
    assert r.status_code == 200 and 0 <= r.get_json()["probability"] <= 1


def test_relocking_survives_a_new_split(client):
    lr, dt = _two_models(client)
    client.post("/api/test", data={"models": [lr], "source": "test"})
    # re-confirming must not reset the lock, the old test rows have been seen
    r = client.post("/api/confirm", json={"types": {"age": "continuous", "sex": "binary",
                                                    "outcome": "binary"},
                                          "missing": {}, "ratios": [70, 15, 15],
                                          "method": "random", "seed": 3})
    assert r.status_code == 200
    assert client.post("/api/test", data={"models": [dt], "source": "test"}).status_code == 409


# ---------------------------------------------------------------------------
# strict json helpers
# ---------------------------------------------------------------------------
def _strict(text):
    """json.loads that fails on NaN / Infinity, as a browser's JSON.parse does."""
    def bad(token):
        raise ValueError(f"non-standard JSON token {token}")
    return json.loads(text, parse_constant=bad)


def _events(stream):
    return [_strict(chunk[6:]) for chunk in stream.split("\n\n") if chunk.startswith("data: ")]


def _done(events):
    return [e for e in events if e["event"] == "model_done"]


# ---------------------------------------------------------------------------
# cohort
# ---------------------------------------------------------------------------
AGE_FILTER = [{"column": "age", "op": "gte", "values": [55]}]


def _lab_cohort(n=300, seed=2):
    """lab is missing only for patients under 55, so a cohort of 55 and over
    has no missing lab values at all."""
    df = _cohort(n=n, seed=seed)
    df["lab"] = np.where(df["age"] < 55, np.nan, 1.0 + df["age"] / 100)
    return df


def test_cohort_preview_returns_flow_lines_and_a_diagram(client):
    df = _lab_cohort()
    client.post("/api/upload", data={"file": _csv(df)})
    r = client.post("/api/cohort/preview", json={"filters": AGE_FILTER})
    assert r.status_code == 200
    body = r.get_json()
    keep = int((df["age"] >= 55).sum())
    assert body["n_start"] == len(df) and body["n_final"] == keep
    assert body["flow"][0]["n_removed"] == len(df) - keep
    assert body["lines"][0] == f"Records in file: {len(df):,}"
    assert body["lines"][-1] == f"Final cohort: {keep:,}"
    assert body["png"].startswith("data:image/png;base64,")
    # nothing was saved: still only the upload in the log
    assert [e["action"] for e in _log(client)] == ["upload"]


def test_cohort_preview_reports_every_problem_with_a_400(client):
    client.post("/api/upload", data={"file": _csv(_cohort())})
    r = client.post("/api/cohort/preview", json={"filters": [
        {"column": "nope", "op": "gte", "values": [1]},
        {"column": "age", "op": "sideways", "values": [1]}]})
    assert r.status_code == 400
    body = r.get_json()
    assert body["error"] and len(body["problems"]) == 2


def test_cohort_preview_needs_a_session(client):
    assert client.post("/api/cohort/preview", json={"filters": AGE_FILTER}).status_code == 400


def test_confirm_applies_the_cohort_before_missingness_fills_and_the_split(client):
    df = _lab_cohort()
    r = _confirm(client, df, missing={"lab": "impute"}, cohort=AGE_FILTER)
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    keep = int((df["age"] >= 55).sum())
    # every patient under 55 is gone, so lab has no missing values left
    assert "lab" not in body["missing_pct"]
    assert body["n_train"] + body["n_val"] + body["n_test"] == keep
    assert body["cohort"]["n_start"] == len(df) and body["cohort"]["n_final"] == keep
    assert body["cohort"]["lines"][-1] == f"Final cohort: {keep:,}"
    sid = _sid(client)
    full = pd.read_csv(store.session_path(sid, "full.csv"))
    assert len(full) == keep and full["age"].min() >= 55
    meta = store.load_meta(sid)
    assert meta["cohort"]["filters"] == AGE_FILTER and meta["cohort"]["n_final"] == keep
    actions = [e["action"] for e in _log(client)]
    assert actions[-2:] == ["cohort_filter", "confirm"]
    entry = [e for e in _log(client) if e["action"] == "cohort_filter"][0]
    assert entry["inputs"] == {"filters": AGE_FILTER}
    assert entry["outputs"]["n_start"] == len(df) and entry["outputs"]["n_final"] == keep
    text = client.get("/api/report/methods").get_json()["text"]
    assert f"Starting from {len(df)} records" in text
    status = {i["id"]: i["status"] for i in client.get("/api/report/tripod").get_json()["items"]}
    assert status["eligibility_criteria"] == "found"


def test_confirm_without_a_cohort_reports_null_and_retires_an_earlier_cohort(client):
    df = _lab_cohort()
    _confirm(client, df, missing={"lab": "impute"}, cohort=AGE_FILTER)
    types = {"age": "continuous", "sex": "binary", "outcome": "binary", "lab": "continuous"}
    r = client.post("/api/confirm", json={"types": types, "missing": {"lab": "impute"},
                                          "ratios": [70, 15, 15], "method": "random"})
    assert r.status_code == 200
    assert r.get_json()["cohort"] is None
    assert "lab" in r.get_json()["missing_pct"]
    assert store.load_meta(_sid(client))["cohort"] is None
    text = client.get("/api/report/methods").get_json()["text"]
    assert "cohort was defined" not in text
    status = {i["id"]: i["status"] for i in client.get("/api/report/tripod").get_json()["items"]}
    assert status["eligibility_criteria"] == "manual_review"


def test_confirm_cohort_that_leaves_no_rows_is_a_400(client):
    r = _confirm(client, _cohort(), cohort=[{"column": "age", "op": "gte", "values": [500]}])
    assert r.status_code == 400
    assert "No rows" in r.get_json()["error"]
    assert "confirm" not in [e["action"] for e in _log(client)]


def test_confirm_bad_cohort_filter_is_a_400_with_problems(client):
    r = _confirm(client, _cohort(), cohort=[{"column": "nope", "op": "eq", "values": [1]}])
    assert r.status_code == 400
    assert r.get_json()["problems"]


# ---------------------------------------------------------------------------
# training: threshold, recalibration, outcome guard, adjusted odds ratios
# ---------------------------------------------------------------------------
def test_train_rejects_bad_threshold_settings(client):
    _confirm(client, _cohort())
    base = {"outcome": "outcome", "models": ["LogisticRegression"]}
    assert client.post("/api/train", json={**base, "threshold_method": "vibes"}).status_code == 400
    r = client.post("/api/train", json={**base, "threshold_method": "sensitivity"})
    assert r.status_code == 400 and "target" in r.get_json()["error"]
    r = client.post("/api/train", json={**base, "threshold_method": "fixed",
                                        "threshold_target": 1.5})
    assert r.status_code == 400
    r = client.post("/api/train", json={**base, "threshold_method": "specificity",
                                        "threshold_target": "abc"})
    assert r.status_code == 400
    r = client.post("/api/train", json={**base, "threshold_method": "specificity",
                                        "threshold_target": 0.9})
    assert r.status_code == 200
    cfg = store.load_meta(_sid(client))["train_config"]
    assert cfg["threshold_method"] == "specificity" and cfg["threshold_target"] == 0.9
    assert cfg["recalibrate"] is False


def test_youden_is_the_default_threshold_and_is_saved_with_the_model(client):
    _confirm(client, _cohort())
    done = _done(_train(client))[0]
    assert done["threshold_method"] == "youden" and done["threshold_target"] is None
    assert done["recalibrated"] is False and "recalibration" not in done
    assert done["metrics"]["Threshold"] == done["threshold"]
    assert any("chosen on the validation set" in n for n in done["notes"])
    import joblib
    mdl = joblib.load(store.model_path(done["saved_as"]))
    assert mdl["threshold"] == done["threshold"] and mdl["threshold_method"] == "youden"
    meta = json.load(open(store.model_meta_path(done["saved_as"])))
    assert meta["threshold"] == done["threshold"] and meta["recalibrated"] is False
    entry = [e for e in _log(client) if e["action"] == "train_model"][-1]
    assert entry["outputs"]["threshold"] == done["threshold"]
    assert entry["outputs"]["threshold_method"] == "youden"
    text = client.get("/api/report/methods").get_json()["text"]
    assert "maximising the Youden index" in text


def test_fixed_threshold_is_used_as_given(client):
    _confirm(client, _cohort())
    done = _done(_train(client, threshold_method="fixed", threshold_target=0.3))[0]
    assert done["threshold"] == 0.3 and done["metrics"]["Threshold"] == 0.3
    # older clients send only threshold, with fixed that is the target
    done = _done(_train(client, threshold_method="fixed", threshold=0.4))[0]
    assert done["threshold"] == 0.4


def test_recalibration_is_fitted_on_validation_and_saved(client):
    _confirm(client, _cohort())
    done = _done(_train(client, models=["LogisticRegression", "RandomForest"],
                        recalibrate=True, threshold_method="sensitivity",
                        threshold_target=0.8))
    assert len(done) == 2
    for d in done:
        assert d["recalibrated"] is True
        assert set(d["recalibration"]) == {"a", "b"}
        assert any("optimistic" in n for n in d["notes"])
        # recalibrated on these rows, so validation calibration is ideal
        assert d["metrics"]["CalibrationSlope"] == pytest.approx(1.0, abs=1e-3)
        assert d["metrics"]["Sensitivity"] >= 0.8 - 1e-9
        meta = json.load(open(store.model_meta_path(d["saved_as"])))
        assert meta["recalibration"] == d["recalibration"]
    entry = [e for e in _log(client) if e["action"] == "train_model"][-1]
    assert entry["outputs"]["recalibrated"] is True and "recalibration" in entry["outputs"]
    # the recalibrated model still tests and predicts
    names = [d["saved_as"] for d in done]
    r = client.post("/api/test", data={"models": names, "source": "test"})
    assert r.status_code == 200
    assert all(m["recalibrated"] is True for m in r.get_json()["models"])
    r = client.post("/api/predict-single", json={"model": names[0], "row": {"age": 70, "sex": "M"}})
    assert r.status_code == 200 and 0 <= r.get_json()["probability"] <= 1
    text = client.get("/api/report/methods").get_json()["text"]
    assert "logistic recalibration" in text and "after recalibration" in text


def _cohort_missing_outcome(n=300, seed=4):
    df = _cohort(n=n, seed=seed)
    df["outcome"] = df["outcome"].astype(float)
    df.loc[np.arange(n) % 15 == 0, "outcome"] = np.nan
    return df


@pytest.mark.parametrize("decision", ["impute", "impute_indicator"])
def test_training_refuses_an_outcome_that_confirm_filled_in(client, decision):
    _confirm(client, _cohort_missing_outcome(), missing={"outcome": decision})
    r = client.post("/api/train", json={"outcome": "outcome", "models": ["LogisticRegression"]})
    assert r.status_code == 400
    assert "Include as is" in r.get_json()["error"]


def test_training_accepts_an_outcome_left_missing_and_drops_those_rows(client):
    _confirm(client, _cohort_missing_outcome(), missing={"outcome": "include"})
    sid = _sid(client)
    train = pd.read_csv(store.session_path(sid, "train.csv"))
    assert train["outcome"].isna().sum() > 0
    events = _train(client)
    done = _done(events)
    assert done, events
    entry = [e for e in _log(client) if e["action"] == "train_model"][-1]
    assert entry["outputs"]["n_train"] == int(train["outcome"].notna().sum())


def test_logistic_regression_reports_adjusted_odds_ratios(client):
    _confirm(client, _cohort())
    events = _train(client, models=["LogisticRegression", "DecisionTree"])
    lr, dt = _done(events)
    odds = lr["odds_ratios"]
    assert {r["term"] for r in odds["rows"]} == {"age", "sex=M"}
    assert odds["n_parameters"] == 2 and odds["reference_levels"] == {"sex": "F"}
    assert "odds_ratios" not in dt
    entry = [e for e in _log(client) if e["action"] == "adjusted_or"][-1]
    assert entry["inputs"] == {"outcome": "outcome", "features": ["age", "sex"]}
    assert set(entry["outputs"]) == {"n", "events", "n_parameters", "epp"}
    r = client.get(f"/api/model/{lr['saved_as']}/odds-ratios")
    assert r.status_code == 200
    assert r.get_json()["rows"] == odds["rows"] and r.get_json()["name"] == lr["saved_as"]
    assert client.get(f"/api/model/{dt['saved_as']}/odds-ratios").status_code == 404
    assert client.get("/api/model/nothing_here/odds-ratios").status_code == 404


def test_training_stream_is_strict_json(client):
    _confirm(client, _cohort())
    assert client.post("/api/train", json={"outcome": "outcome", "grid_search": False,
                                           "models": ["LogisticRegression"]}).status_code == 200
    events = _events(client.get("/api/train/stream").get_data(as_text=True))
    assert _done(events) and events[-1]["event"] == "complete"


# ---------------------------------------------------------------------------
# testing: saved threshold, flexible calibration, decision curves
# ---------------------------------------------------------------------------
def test_test_uses_each_models_saved_threshold_unless_one_is_given(client):
    _confirm(client, _cohort())
    done = _done(_train(client, models=["LogisticRegression", "DecisionTree"]))
    saved = {d["saved_as"]: d["threshold"] for d in done}
    names = list(saved)
    r = client.post("/api/test", data={"models": names, "source": "test"})
    for m in r.get_json()["models"]:
        assert m["threshold"] == saved[m["name"]] and m["threshold_method"] == "saved"
        assert m["metrics"]["Threshold"] == saved[m["name"]]
    entry = [e for e in _log(client) if e["action"] == "test_model"][-1]
    assert entry["outputs"]["threshold_method"] == "youden"
    assert entry["outputs"]["threshold_source"] == "saved"
    # a blank threshold counts as not given
    r = client.post("/api/test", data={"models": names, "source": "test", "threshold": ""})
    assert [m["threshold_method"] for m in r.get_json()["models"]] == ["saved", "saved"]
    r = client.post("/api/test", json={"models": names, "source": "test", "threshold": 0.25})
    for m in r.get_json()["models"]:
        assert m["threshold"] == 0.25 and m["threshold_method"] == "user"
    entry = [e for e in _log(client) if e["action"] == "test_model"][-1]
    assert entry["outputs"]["threshold_method"] == "user"
    assert client.post("/api/test", json={"models": names, "threshold": "x"}).status_code == 400


def test_binary_test_has_flexible_calibration_and_a_decision_curve(client):
    lr, dt = _two_models(client)
    r = client.post("/api/test", data={"models": [lr, dt], "source": "test"})
    assert r.status_code == 200
    body = _strict(r.get_data(as_text=True))
    for m in body["models"]:
        assert m["calibration_png"].startswith("data:image/png;base64,")
        assert {"slope", "citl", "oe_ratio", "ici"} <= set(m["calibration_stats"])
        assert isinstance(m["metrics"]["CI"], dict)
    dca = body["dca"]
    assert len(dca["thresholds"]) == len(dca["treat_all"]) == 99
    assert [m["name"] for m in dca["models"]] == [lr, dt]
    assert all(len(m["net_benefit"]) == 99 for m in dca["models"])
    assert body["dca_png"].startswith("data:image/png;base64,")
    entry = [e for e in _log(client) if e["action"] == "test_model"][-1]
    assert entry["outputs"]["dca"] is True and entry["outputs"]["recalibrated"] is False
    status = {i["id"]: i["status"] for i in client.get("/api/report/tripod").get_json()["items"]}
    assert status["clinical_utility"] == "found"


# ---------------------------------------------------------------------------
# epi: km, cox, competing risks
# ---------------------------------------------------------------------------
def _surv_cohort(n=300, seed=6):
    df = _cohort(n=n, seed=seed)
    rng = np.random.RandomState(seed)
    df["months"] = np.round(rng.exponential(24, n), 1) + 0.1
    df["status"] = rng.choice([0, 1, 2], n, p=[0.5, 0.3, 0.2])      # 2 = competing death
    df["asa"] = rng.choice([1, 2, 3], n)
    return df


def test_km_takes_time_points_and_draws_a_risk_table(client):
    df = _surv_cohort()
    df["died"] = (df["status"] > 0).astype(int)
    _confirm(client, df)
    r = client.post("/api/epi/km", json={"time": "months", "event": "died", "group": "sex",
                                         "time_points": [0, 6, 12]})
    assert r.status_code == 200
    body = r.get_json()
    assert body["risk_table"]["time_points"] == [0, 6, 12]
    assert len(body["risk_table"]["groups"]) == 2 and body["png"]
    entry = [e for e in _log(client) if e["action"] == "epi_km"][-1]
    assert entry["inputs"]["time_points"] == [0, 6, 12]
    r = client.post("/api/epi/km", json={"time": "months", "event": "died",
                                         "time_points": ["soon"]})
    assert r.status_code == 400


def test_cox_takes_categorical_covariates_and_logs_the_ph_test(client):
    df = _surv_cohort()
    df["died"] = (df["status"] > 0).astype(int)
    _confirm(client, df)
    r = client.post("/api/epi/cox", json={"time": "months", "event": "died",
                                          "covariates": ["age", "asa"], "categorical": ["asa"],
                                          "reference": {"asa": 2}})
    assert r.status_code == 200
    body = r.get_json()
    assert [row["covariate"] for row in body["rows"]] == ["age", "asa: 1 vs 2", "asa: 3 vs 2"]
    assert body["ph_test"]["transform"] == "rank"
    entry = [e for e in _log(client) if e["action"] == "epi_cox"][-1]
    assert entry["inputs"]["categorical"] == ["asa"]
    assert entry["outputs"]["ph_test"]["global"]["p_value"] == body["ph_test"]["global"]["p_value"]
    text = client.get("/api/report/methods").get_json()["text"]
    assert "Schoenfeld" in text
    r = client.post("/api/epi/cox", json={"time": "months", "event": "died",
                                          "covariates": ["age"], "ph_transform": "wobbly"})
    assert r.status_code == 400


def test_cumulative_incidence_with_competing_risks(client):
    _confirm(client, _surv_cohort())
    r = client.post("/api/epi/cif", json={"time": "months", "event": "status",
                                          "event_of_interest": 1, "group": "sex",
                                          "time_points": [0, 12, 24]})
    assert r.status_code == 200
    body = _strict(r.get_data(as_text=True))
    assert [c["name"] for c in body["curves"]] == ["F", "M"]
    assert body["competing_codes"] == [2]
    assert body["gray_test"]["df"] == 1 and body["png"].startswith("data:image/png")
    assert body["risk_table"]["time_points"] == [0, 12, 24]
    entry = [e for e in _log(client) if e["action"] == "epi_cif"][-1]
    assert entry["inputs"] == {"time": "months", "event": "status", "event_of_interest": 1,
                               "group": "sex"}
    assert entry["outputs"]["n"] == 300 and entry["outputs"]["n_curves"] == 2
    assert entry["outputs"]["gray_p"] == body["gray_test"]["p_value"]
    # event_of_interest defaults to 1
    r = client.post("/api/epi/cif", json={"time": "months", "event": "status"})
    assert r.status_code == 200 and r.get_json()["gray_test"] is None
    text = client.get("/api/report/methods").get_json()["text"]
    assert "competing risks" in text


def test_cumulative_incidence_errors_are_400s(client):
    assert client.post("/api/epi/cif", json={"time": "months", "event": "status"}).status_code == 400
    _confirm(client, _surv_cohort())
    assert client.post("/api/epi/cif", json={"event": "status"}).status_code == 400
    r = client.post("/api/epi/cif", json={"time": "months", "event": "status",
                                          "event_of_interest": 7})
    assert r.status_code == 400 and "code 7" in r.get_json()["error"]
    r = client.post("/api/epi/cif", json={"time": "nope", "event": "status"})
    assert r.status_code == 400
    r = client.post("/api/epi/cif", json={"time": "months", "event": "status",
                                          "event_of_interest": "one"})
    assert r.status_code == 400


def test_epi_or_and_nnt_pass_through_the_nnt_interval(client):
    r = client.post("/api/epi/or", json={"a": 20, "b": 80, "c": 10, "d": 90})
    assert {"nnt_ci", "rd_ci", "fisher_p_value"} <= set(r.get_json())
    r = client.post("/api/epi/nnt", json={"a": 20, "b": 80, "c": 10, "d": 90})
    assert "nnt_ci" in r.get_json() and "value_rounded_up" in r.get_json()


# ---------------------------------------------------------------------------
# sample size
# ---------------------------------------------------------------------------
def test_sample_size_works_without_a_session_and_matches_pmsampsize(client):
    r = client.post("/api/samplesize", json={"prevalence": 0.2, "n_parameters": 10,
                                             "cstatistic": 0.75})
    assert r.status_code == 200
    body = r.get_json()
    assert body["result"]["min_n"] == 666 and body["result"]["min_events"] == 134
    assert body["compare"] is None


def test_sample_size_with_a_session_compares_and_is_logged(client):
    _confirm(client, _cohort())
    sid = _sid(client)
    train = pd.read_csv(store.session_path(sid, "train.csv"))
    r = client.post("/api/samplesize", json={"prevalence": 0.3, "n_parameters": 2,
                                             "csrsquared": 0.1, "shrinkage": 0.9,
                                             "compare": {"outcome": "outcome", "exclude": []}})
    assert r.status_code == 200
    cmp = r.get_json()["compare"]
    assert cmp["n"] == len(train) and cmp["events"] == int(train["outcome"].sum())
    entry = [e for e in _log(client) if e["action"] == "sample_size"][-1]
    assert entry["inputs"]["prevalence"] == 0.3 and entry["inputs"]["csrsquared"] == 0.1
    assert entry["inputs"]["n_parameters"] == 2 and entry["inputs"]["shrinkage"] == 0.9
    assert {"min_n", "min_events", "epp", "criteria"} <= set(entry["outputs"])
    status = {i["id"]: i["status"] for i in client.get("/api/report/tripod").get_json()["items"]}
    assert status["sample_size"] == "found"
    text = client.get("/api/report/methods").get_json()["text"]
    assert "Riley et al." in text


def test_sample_size_continuous_outcome(client):
    r = client.post("/api/samplesize", json={"outcome_type": "continuous", "rsquared": 0.2,
                                             "n_parameters": 5, "intercept": 4, "sd": 2})
    assert r.status_code == 200
    assert r.get_json()["result"]["outcome"] == "continuous"
    assert r.get_json()["result"]["min_n"] > 0


@pytest.mark.parametrize("body", [
    {"outcome_type": "ordinal", "prevalence": 0.2, "n_parameters": 5, "cstatistic": 0.7},
    {"n_parameters": 5, "cstatistic": 0.7},
    {"prevalence": 0.2, "cstatistic": 0.7},
    {"prevalence": 0.2, "n_parameters": 5},
    {"prevalence": 0.2, "n_parameters": 5, "cstatistic": 0.7, "csrsquared": 0.1},
    {"prevalence": 1.5, "n_parameters": 5, "cstatistic": 0.7},
    {"prevalence": "lots", "n_parameters": 5, "cstatistic": 0.7},
    {"prevalence": 0.2, "n_parameters": 5, "cstatistic": 0.7, "compare": {"outcome": "x"}},
    {"outcome_type": "continuous", "rsquared": 0.2, "n_parameters": 5},
])
def test_sample_size_bad_input_is_a_400(client, body):
    r = client.post("/api/samplesize", json=body)
    assert r.status_code == 400 and r.get_json()["error"]


def test_sample_size_compare_needs_a_known_matching_outcome(client):
    df = _cohort()
    df["los"] = np.random.RandomState(0).normal(5, 2, len(df))
    _confirm(client, df)
    base = {"prevalence": 0.3, "n_parameters": 2, "cstatistic": 0.7}
    r = client.post("/api/samplesize", json={**base, "compare": {"outcome": "nope"}})
    assert r.status_code == 400
    r = client.post("/api/samplesize", json={**base, "compare": {"outcome": "los"}})
    assert r.status_code == 400


def test_sample_size_defaults_come_from_the_training_partition(client):
    assert client.get("/api/samplesize/defaults?outcome=outcome").status_code == 400
    df = _cohort()
    df["los"] = np.random.RandomState(0).normal(5, 2, len(df))
    _confirm(client, df)
    train = pd.read_csv(store.session_path(_sid(client), "train.csv"))
    r = client.get("/api/samplesize/defaults?outcome=outcome&exclude=los")
    assert r.status_code == 200
    body = r.get_json()
    assert body["n"] == len(train) and body["events"] == int(train["outcome"].sum())
    assert body["n_parameters"] == 2                        # age + sex, los excluded
    assert body["prevalence"] == pytest.approx(body["events"] / body["n"], abs=1e-4)
    r = client.get("/api/samplesize/defaults?outcome=outcome")
    assert r.get_json()["n_parameters"] == 3                # los counts too
    assert client.get("/api/samplesize/defaults?outcome=los").status_code == 400
    assert client.get("/api/samplesize/defaults?outcome=nope").status_code == 400


# ---------------------------------------------------------------------------
# teaching mode
# ---------------------------------------------------------------------------
def test_teaching_config_is_validated(client):
    df = _cohort(n=400)
    df["los"] = np.random.RandomState(0).normal(5, 2, len(df))
    assert client.post("/api/teaching", json={"outcome": "outcome"}).status_code == 400
    _confirm(client, df)
    ok = {"outcome": "outcome", "exclude": ["los"], "model": "logistic"}
    assert client.post("/api/teaching", json={**ok, "outcome": "los"}).status_code == 400
    assert client.post("/api/teaching", json={**ok, "outcome": "nope"}).status_code == 400
    assert client.post("/api/teaching", json={**ok, "exclude": ["nope"]}).status_code == 400
    assert client.post("/api/teaching", json={**ok, "model": "xgboost"}).status_code == 400
    assert client.post("/api/teaching", json={**ok, "n_seeds": 6}).status_code == 400
    assert client.post("/api/teaching", json={**ok, "n_seeds": 0}).status_code == 400
    assert client.post("/api/teaching", json={**ok, "max_rows": "many"}).status_code == 400
    r = client.post("/api/teaching", json=ok)
    assert r.status_code == 200 and r.get_json() == {"ok": True}
    cfg = store.load_meta(_sid(client))["teaching_config"]
    assert cfg == {"outcome": "outcome", "exclude": ["los"], "model": "logistic",
                   "seeds": [1, 2, 3], "max_rows": 20000}


def test_teaching_stream_needs_a_config(client):
    _confirm(client, _cohort())
    assert client.get("/api/teaching/stream").status_code == 400


def test_teaching_stream_runs_the_leakage_demo_end_to_end(client):
    _confirm(client, _cohort(n=400))
    r = client.post("/api/teaching", json={"outcome": "outcome", "model": "logistic",
                                           "n_seeds": 1, "max_rows": 350})
    assert r.status_code == 200
    events = _events(client.get("/api/teaching/stream").get_data(as_text=True))
    kinds = [e["event"] for e in events]
    assert kinds[0] == "progress" and kinds[-1] == "complete", events[-1]
    assert all(e["message"] for e in events if e["event"] == "progress")
    result = events[-1]["result"]
    assert result["seeds"] == [1] and result["n_used"] == 350 and result["subsampled"]
    assert [a["arm"] for a in result["arms"]][0] == "correct"
    assert events[-1]["png"].startswith("data:image/png;base64,")
    entry = [e for e in _log(client) if e["action"] == "leakage_demo"][-1]
    assert entry["inputs"] == {"outcome": "outcome", "model": "logistic", "seeds": [1],
                               "max_rows": 350}
    arms = entry["outputs"]["arms"]
    assert set(arms[0]) == {"arm", "reported_auroc", "lockbox_auroc", "optimism"}
    assert arms[0]["optimism"] == result["arms"][0]["optimism"]["mean"]
    # a teaching demo is not part of the study
    assert "leakage" not in client.get("/api/report/methods").get_json()["text"].lower()


def test_teaching_stream_reports_a_demo_error_as_an_event(client):
    df = _cohort(n=200)
    df["outcome"] = (np.arange(200) % 20 == 0).astype(int)       # 10 events, too few
    _confirm(client, df)
    assert client.post("/api/teaching", json={"outcome": "outcome", "n_seeds": 1}).status_code == 200
    events = _events(client.get("/api/teaching/stream").get_data(as_text=True))
    assert events[-1]["event"] == "error" and "at least" in events[-1]["error"]
    assert "leakage_demo" not in [e["action"] for e in _log(client)]


# ---------------------------------------------------------------------------
# pdf export of the new charts
# ---------------------------------------------------------------------------
_PDF_KINDS = {
    "km": {"curves": [{"name": "A", "timeline": [0, 1, 2], "survival": [1, 0.8, 0.5],
                       "lower": [1, 0.7, 0.4], "upper": [1, 0.9, 0.6]}],
           "risk_table": {"time_points": [0, 1, 2], "groups": [{"name": "A", "at_risk": [10, 8, 5]}]}},
    "cif": {"curves": [{"name": "A", "timeline": [0, 1, 2], "cif": [0, 0.1, 0.3]}],
            "naive": [{"name": "A", "timeline": [0, 1, 2], "risk": [0, 0.12, 0.35]}]},
    "dca": {"thresholds": [0.1, 0.2, 0.3], "treat_all": [0.2, 0.1, None],
            "models": [{"name": "m", "net_benefit": [0.22, 0.15, 0.05]}]},
    "calibration_flexible": {"p_grid": [0.1, 0.5], "observed": [0.12, 0.45],
                             "risk_histogram": {"edges": [0, 0.5, 1], "counts": [3, 4]},
                             "calibration_stats": {"slope": 1.0, "slope_ci": [0.8, 1.2]}},
    "teaching": {"result": {"arms": [{"arm": "correct", "label": "Done correctly",
                                      "reported_auroc": {"mean": 0.7, "min": 0.68, "max": 0.72},
                                      "lockbox_auroc": {"mean": 0.69, "min": 0.66, "max": 0.71},
                                      "optimism": {"mean": 0.01},
                                      "calibration_slope": {"mean": 1.0, "min": 0.9, "max": 1.1}}],
                            "seeds": [1], "model_label": "logistic regression"}},
    "flow": {"flow": [{"step": 1, "label": "Adults", "kind": "exclude", "n_before": 10,
                       "n_after": 8, "n_removed": 2}], "n_start": 10},
}


@pytest.mark.parametrize("kind", sorted(_PDF_KINDS))
def test_render_pdf_handles_the_new_chart_kinds(client, kind):
    r = client.post("/api/render/pdf", json={"kind": kind, "data": _PDF_KINDS[kind]})
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.mimetype == "application/pdf" and r.data[:4] == b"%PDF"


def test_render_pdf_calibration_flexible_also_takes_test_coords(client):
    coords = {"calibration_flexible": {"p_grid": [0.1, 0.5], "observed": [0.12, 0.45]},
              "calibration": {"prob_pred": [0.1, 0.4], "prob_true": [0.1, 0.5]},
              "risk_histogram": {"edges": [0, 0.5, 1], "counts": [3, 4]}}
    r = client.post("/api/render/pdf", json={"kind": "calibration_flexible", "data": coords})
    assert r.status_code == 200
