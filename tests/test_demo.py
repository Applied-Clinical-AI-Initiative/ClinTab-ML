"""tests for the made-up demo dataset and how it reaches the app and the cli."""
import io

import pandas as pd
import pytest

from clintab import demo, store


@pytest.fixture
def client(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(store, "RUNTIME", str(runtime))
    monkeypatch.setattr(store, "SESSIONS", str(runtime / "sessions"))
    monkeypatch.setattr(store, "EXPORTS", str(runtime / "exports"))
    monkeypatch.setattr(store, "MODELS", str(tmp_path / "models"))
    from app import create_app
    return create_app().test_client()


def test_demo_dataset_is_deterministic():
    a = demo.make_demo_dataset(n=500, seed=7)
    b = demo.make_demo_dataset(n=500, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(demo.make_demo_dataset(n=500, seed=8))


def test_demo_dataset_has_what_every_tab_needs():
    df = demo.make_demo_dataset()
    assert len(df) == 10000
    assert set(df["cpt_code"]) == {27130, 27447, 27125, 27446}
    assert set(df["revision_status"]) == {0, 1, 2}
    assert set(df["readmission_30d"]) == {0, 1}
    assert 0.05 < df["readmission_30d"].mean() < 0.20
    assert 0.35 < df["albumin"].isna().mean() < 0.45
    assert df["years_followup"].min() > 0
    assert (df["revised"] == (df["revision_status"] == 1)).all()


def test_demo_dataset_rejects_tiny_n():
    with pytest.raises(ValueError):
        demo.make_demo_dataset(n=50)


def test_demo_csv_route_serves_the_dataset(client):
    r = client.get("/api/demo.csv")
    assert r.status_code == 200
    assert r.mimetype == "text/csv"
    assert demo.DEMO_FILENAME in r.headers["Content-Disposition"]
    df = pd.read_csv(io.BytesIO(r.data))
    assert df.shape == (10000, 16)


def test_demo_csv_uploads_like_any_file(client):
    data = client.get("/api/demo.csv").data
    r = client.post("/api/upload", data={"file": (io.BytesIO(data), demo.DEMO_FILENAME)},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["n_rows"] == 10000 and body["n_cols"] == 16


def test_cli_demo_writes_the_csv(tmp_path):
    import clintab_cli
    out = tmp_path / "demo.csv"
    clintab_cli.main(["demo", "--out", str(out), "--n", "300"])
    assert pd.read_csv(out).shape == (300, 16)
