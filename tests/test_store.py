import pytest

from clintab import store


def test_safe_name_strips_path_characters():
    assert store.safe_name("../../etc/passwd") == ".._.._etc_passwd"
    assert "/" not in store.safe_name("a/b\\c d")


def test_model_base_name_keeps_the_timestamp_on_long_outcome_names():
    ts = "20261026-101010"
    name = store.model_base_name("GradientBoosting", "x" * 200, ts)
    assert name.endswith("_" + ts)
    # two runs of the same model must not collapse onto one file name
    assert name != store.model_base_name("GradientBoosting", "x" * 200, "20261026-101011")


def test_valid_session_id_rejects_path_tricks():
    assert store.valid_session_id("a1b2c3d4e5f6")
    for bad in ("", "..", "../x", "a/b", None, "x" * 80):
        assert not store.valid_session_id(bad)


def test_session_dir_refuses_an_invalid_id(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "SESSIONS", str(tmp_path))
    with pytest.raises(ValueError):
        store.session_dir("../escape", create=True)
    assert store.session_exists("../escape") is False


def test_model_path_stays_inside_the_models_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "MODELS", str(tmp_path))
    p = store.model_path("../../outside")
    assert p.startswith(str(tmp_path))


def test_session_listing_ignores_stray_files_like_ds_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "SESSIONS", str(tmp_path / "sessions"))
    monkeypatch.setattr(store, "RUNTIME", str(tmp_path))
    monkeypatch.setattr(store, "EXPORTS", str(tmp_path / "exports"))
    monkeypatch.setattr(store, "MODELS", str(tmp_path / "models"))
    store.ensure_dirs()
    store.session_dir("abc123", create=True)
    store.save_meta("abc123", {"filename": "x.csv"})
    (tmp_path / "sessions" / ".DS_Store").write_text("junk")

    assert [s["session_id"] for s in store.list_sessions()] == ["abc123"]
    store.delete_all_sessions()
    assert store.list_sessions() == []
