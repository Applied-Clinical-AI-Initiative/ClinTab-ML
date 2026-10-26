import io
import json
import os

import numpy as np
import pandas as pd
import pytest

from clintab import store


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
