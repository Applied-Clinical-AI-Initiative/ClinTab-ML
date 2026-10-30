import hashlib
import json

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
    # note: no store.session_dir() call with create first
    analysis_log.log_action("brand-new-session", "upload", inputs={"filename": "x.csv"})
    entries = analysis_log.read_log("brand-new-session")
    assert len(entries) == 1
    assert entries[0]["action"] == "upload"


def test_read_log_skips_a_half_written_line(session_id):
    analysis_log.log_action(session_id, "upload")
    with open(analysis_log.log_path(session_id), "a") as f:
        f.write('{"timestamp": 1, "action": "tra')       # crash mid-write
    assert [e["action"] for e in analysis_log.read_log(session_id)] == ["upload"]


# ---------------------------------------------------------------------------
# hash chain + provenance
# ---------------------------------------------------------------------------
def _lines(sid):
    with open(analysis_log.log_path(sid)) as f:
        return f.read().splitlines()


def test_each_entry_chains_to_the_previous_raw_line(session_id):
    first = analysis_log.log_action(session_id, "upload")
    second = analysis_log.log_action(session_id, "confirm")
    lines = _lines(session_id)
    assert first["prev_hash"] == ""
    assert second["prev_hash"] == hashlib.sha256(lines[0].encode()).hexdigest()
    assert json.loads(lines[1])["prev_hash"] == second["prev_hash"]


def test_verify_log_ok_on_an_untouched_log(session_id):
    for a in ("upload", "confirm", "train_model"):
        analysis_log.log_action(session_id, a, inputs={"x": 1})
    assert analysis_log.verify_log(session_id) == {
        "ok": True, "n_entries": 3, "first_bad_line": None, "legacy": False}


def test_verify_log_empty_session_is_ok(session_id):
    assert analysis_log.verify_log(session_id)["ok"] is True
    assert analysis_log.verify_log(session_id)["n_entries"] == 0


def test_verify_log_catches_an_edited_entry(session_id):
    for a in ("upload", "confirm", "train_model", "test_model"):
        analysis_log.log_action(session_id, a, outputs={"metrics": {"AUROC": 0.7}})
    lines = _lines(session_id)
    lines[1] = lines[1].replace("0.7", "0.9")           # quietly improve a result
    with open(analysis_log.log_path(session_id), "w") as f:
        f.write("\n".join(lines) + "\n")
    res = analysis_log.verify_log(session_id)
    assert res["ok"] is False and res["first_bad_line"] == 3


def test_verify_log_catches_a_deleted_entry(session_id):
    for a in ("upload", "confirm", "test_model", "train_model"):
        analysis_log.log_action(session_id, a)
    lines = _lines(session_id)
    del lines[2]                                        # drop an embarrassing test result
    with open(analysis_log.log_path(session_id), "w") as f:
        f.write("\n".join(lines) + "\n")
    res = analysis_log.verify_log(session_id)
    assert res["ok"] is False and res["first_bad_line"] == 3


def test_verify_log_catches_an_inserted_unchained_entry(session_id):
    analysis_log.log_action(session_id, "upload")
    analysis_log.log_action(session_id, "confirm")
    lines = _lines(session_id)
    lines.insert(1, json.dumps({"timestamp": 1, "action": "sample_size", "inputs": {},
                                "outputs": {}}))
    with open(analysis_log.log_path(session_id), "w") as f:
        f.write("\n".join(lines) + "\n")
    res = analysis_log.verify_log(session_id)
    assert res["ok"] is False and res["first_bad_line"] == 2


def test_legacy_log_without_prev_hash_still_reads_and_verifies_as_legacy(session_id):
    with open(analysis_log.log_path(session_id), "w") as f:
        f.write(json.dumps({"timestamp": 1, "action": "upload", "inputs": {}, "outputs": {}}) + "\n")
        f.write(json.dumps({"timestamp": 2, "action": "confirm", "inputs": {}, "outputs": {}}) + "\n")
    assert [e["action"] for e in analysis_log.read_log(session_id)] == ["upload", "confirm"]
    res = analysis_log.verify_log(session_id)
    assert res == {"ok": True, "n_entries": 2, "first_bad_line": None, "legacy": True}
    # new entries chain onto the old tail
    analysis_log.log_action(session_id, "train_model")
    res = analysis_log.verify_log(session_id)
    assert res["ok"] is True and res["legacy"] is True and res["n_entries"] == 3


def test_entry_after_a_half_written_line_lands_on_its_own_line(session_id):
    analysis_log.log_action(session_id, "upload")
    with open(analysis_log.log_path(session_id), "a") as f:
        f.write('{"timestamp": 1, "action": "tra')
    analysis_log.log_action(session_id, "confirm")
    assert [e["action"] for e in analysis_log.read_log(session_id)] == ["upload", "confirm"]
    res = analysis_log.verify_log(session_id)
    assert res["ok"] is False and res["first_bad_line"] == 2      # the damage is reported


def test_environment_lists_python_clintab_and_core_packages():
    env = analysis_log.environment()
    for key in ("python", "clintab", "numpy", "pandas", "scikit-learn", "imbalanced-learn",
                "statsmodels", "lifelines", "scipy", "platform"):
        assert key in env
    assert env["python"] and env["clintab"] and env["scikit-learn"]
