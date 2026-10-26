import os
import subprocess
import sys

import clintab_desktop


def test_self_test_runs_one_analysis_end_to_end():
    # the same check the release workflow runs inside each packaged build
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "clintab_desktop.py", "--self-test"], cwd=repo,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "self test passed" in r.stdout


def test_user_data_dir_is_local_app_data_not_documents(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\x\AppData\Local")
    assert clintab_desktop.user_data_dir().endswith(os.path.join("Local", "ClinTAB-ML"))
    monkeypatch.setattr(sys, "platform", "darwin")
    assert "Application Support" in clintab_desktop.user_data_dir()


def test_python_style_launch_is_only_refused_in_a_packaged_build(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ClinTAB-ML", "-c", "print(1)"])
    monkeypatch.setattr(clintab_desktop, "FROZEN", False)
    clintab_desktop._refuse_python_style_launch()          # no exit from source
    monkeypatch.setattr(clintab_desktop, "FROZEN", True)
    try:
        clintab_desktop._refuse_python_style_launch()
        raise AssertionError("should have exited")
    except SystemExit as e:
        assert e.code == 2
