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


FAKE_TK = r'''
import sys, time, types

class _Widget:
    def __init__(self, *a, **k): self.opts = dict(k)
    def grid(self, *a, **k): return self
    def configure(self, **k): self.opts.update(k)

class _Root:
    def __init__(self): self.queue, self.alive, self.now = [], True, 0.0
    def title(self, *a): pass
    def resizable(self, *a): pass
    def iconphoto(self, *a): pass
    def protocol(self, *a): pass
    def lift(self): pass
    def destroy(self): self.alive = False
    def after(self, ms, fn): self.queue.append((self.now + ms / 1000, fn))
    def mainloop(self):
        while self.alive and self.queue:
            self.queue.sort(key=lambda x: x[0])
            due, fn = self.queue.pop(0)
            time.sleep(max(0.0, due - self.now)); self.now = due
            fn()

tk = types.ModuleType("tkinter")
tk.Tk = _Root
tk.PhotoImage = lambda **k: (_ for _ in ()).throw(RuntimeError("no images in the fake"))
ttk = types.ModuleType("tkinter.ttk")
ttk.Frame = ttk.Label = ttk.Button = _Widget
tk.ttk = ttk
sys.modules["tkinter"], sys.modules["tkinter.ttk"] = tk, ttk

sys.argv = ["clintab_desktop.py", "--gui-smoke"]
import clintab_desktop
sys.exit(clintab_desktop.main(["--gui-smoke"]))
'''


def test_window_opens_first_then_reports_the_running_server():
    # the window is built before the heavy imports and only switches to
    # "running" once the server answers. a stand in tk drives the same code
    # path the real window uses; ci runs the real one on macos and windows.
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-c", FAKE_TK], cwd=repo, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "gui smoke passed" in r.stdout
