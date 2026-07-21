import pytest

from clintab import store
from clintab import analysis_log


@pytest.fixture
def session_id(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "SESSIONS", str(tmp_path))
    sid = "testsession"
    store.session_dir(sid, create=True)
    return sid


def test_log_action_appends_structured_entry(session_id):
    entry = analysis_log.log_action(
        session_id, "train_model",
        inputs={"model": "LogisticRegression", "outcome": "mortality_30d"},
        outputs={"auroc": 0.81})

    assert entry["action"] == "train_model"
    assert entry["inputs"] == {"model": "LogisticRegression", "outcome": "mortality_30d"}
    assert entry["outputs"] == {"auroc": 0.81}
    assert "timestamp" in entry


def test_log_action_defaults_inputs_and_outputs_to_empty_dict(session_id):
    entry = analysis_log.log_action(session_id, "epi_km")
    assert entry["inputs"] == {}
    assert entry["outputs"] == {}


def test_read_log_returns_entries_in_order(session_id):
    analysis_log.log_action(session_id, "fit_spline", inputs={"predictor": "age"})
    analysis_log.log_action(session_id, "epi_or", inputs={"a": 20, "b": 80, "c": 10, "d": 90})

    entries = analysis_log.read_log(session_id)
    assert len(entries) == 2
    assert entries[0]["action"] == "fit_spline"
    assert entries[1]["action"] == "epi_or"


def test_read_log_empty_session_returns_empty_list(session_id):
    assert analysis_log.read_log(session_id) == []


def test_log_action_creates_session_dir_if_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "SESSIONS", str(tmp_path))
    # note: no store.session_dir(..., create=True) call first
    analysis_log.log_action("brand-new-session", "upload", inputs={"filename": "x.csv"})
    entries = analysis_log.read_log("brand-new-session")
    assert len(entries) == 1
    assert entries[0]["action"] == "upload"
