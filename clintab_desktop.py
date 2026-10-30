#!/usr/bin/env python3
"""clintab_desktop.py
Desktop launcher. Starts ClinTAB-ML on this computer only (127.0.0.1), opens
it in the default browser and shows a small window to reopen it, open the
data folder, or quit. This is what the packaged app (packaging/clintab.spec)
runs when someone double-clicks it.

    python clintab_desktop.py               window + browser
    python clintab_desktop.py --headless    no window, stop with ctrl+c
    python clintab_desktop.py --self-test   start, run one analysis end to end, exit 0 or 1
    python clintab_desktop.py --gui-smoke   open the real window, wait for the server, close, exit 0 or 1

The window opens before anything heavy is imported, so a slow first start
(macOS checks every bundled library the first time) still shows something
straight away. The packaged app keeps sessions, uploads, models and its log
in the user's local app data folder (not Documents, which hospitals often
sync to the cloud). Running from source keeps them in the repo, like app.py.
CLINTAB_HOME overrides both.
"""
import argparse
import faulthandler
import json
import multiprocessing
import os
import platform
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import webbrowser

APP_NAME = "ClinTAB-ML"
DEFAULT_PORT = 8765           # stable, so a bookmark keeps working. 5000 clashes with airplay on macs
FROZEN = getattr(sys, "frozen", False)
HERE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))


def resource(*parts):
    return os.path.join(HERE, *parts)


def user_data_dir():
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, APP_NAME)


def _setup_environment(throwaway):
    """Pick the data folder before clintab.store is imported (it reads
    CLINTAB_HOME at import time). throwaway: a temp folder, for the tests."""
    if throwaway:
        home = tempfile.mkdtemp(prefix="clintab-selftest-")
    elif os.environ.get("CLINTAB_HOME"):
        home = os.environ["CLINTAB_HOME"]
    elif FROZEN:
        home = user_data_dir()
    else:
        home = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(home, exist_ok=True)
    os.environ["CLINTAB_HOME"] = home
    # matplotlib wants a writable cache folder, the app bundle isn't one
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(home, "runtime", "matplotlib"))
    return home


class _Tee:
    def __init__(self, log, other):
        self.log, self.other = log, other

    def write(self, text):
        self.log.write(text)
        try:
            if self.other:
                self.other.write(text)
        except Exception:
            pass
        return len(text)

    def flush(self):
        self.log.flush()
        try:
            if self.other:
                self.other.flush()
        except Exception:
            pass

    def isatty(self):
        return False


def _start_log(home):
    """runtime/clintab-desktop.log in the data folder. a packaged app has no
    terminal, so its output goes here, and so does the traceback of a hard
    crash. started from a terminal, output still shows there too."""
    log_dir = os.path.join(home, "runtime")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "clintab-desktop.log")
    log = open(path, "a", buffering=1, encoding="utf-8")
    if FROZEN:
        # everything goes to the log, and also wherever output went before
        # (a terminal, a ci log, or nowhere when opened from finder)
        sys.stdout, sys.stderr = _Tee(log, sys.stdout), _Tee(log, sys.stderr)
    else:
        sys.stdout = sys.stdout or log            # windowed builds can have none at all
        sys.stderr = sys.stderr or log
    faulthandler.enable(file=log, all_threads=True)
    log.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} start | python {platform.python_version()} | "
              f"{platform.platform()} | frozen={FROZEN} | argv={sys.argv[1:]}\n")

    def to_log(exc_type, exc, tb):
        traceback.print_exception(exc_type, exc, tb, file=log)
    sys.excepthook = to_log
    threading.excepthook = lambda a: to_log(a.exc_type, a.exc_value, a.exc_traceback)
    return path


def _alert(title, message):
    """last resort, when even the window can't open."""
    try:
        if sys.platform == "darwin":
            q = lambda s: json.dumps(s, ensure_ascii=False)     # applescript string literal
            subprocess.run(["osascript", "-e", f"display alert {q(title)} message {q(message)} as critical"],
                           timeout=300)
        elif sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)
        else:
            print(f"{title}: {message}", file=sys.stderr)
    except Exception:
        traceback.print_exc()


def _use_threads_for_joblib():
    """scikit-learn's grid search runs its jobs through joblib, which by default
    starts worker processes by relaunching the python interpreter. in a packaged
    app the "interpreter" is this program, so that relaunch would open a second
    copy instead of a worker. threads avoid it and still use every core."""
    from joblib.parallel import ThreadingBackend, register_parallel_backend
    register_parallel_backend("threading", ThreadingBackend, make_default=True)


def _refuse_python_style_launch():
    """if anything still starts this packaged app like `python -c ...`, stop
    instead of opening another window."""
    if FROZEN and ("-c" in sys.argv[1:] or "-m" in sys.argv[1:]):
        sys.stderr and sys.stderr.write("ClinTAB-ML is not a python interpreter.\n")
        sys.exit(2)


def _instance_file(store):
    return os.path.join(store.RUNTIME, "desktop_instance.json")


def _running_instance(store):
    """URL of a ClinTAB-ML this user already has open, or None."""
    try:
        with open(_instance_file(store)) as f:
            url = json.load(f)["url"]
        with urllib.request.urlopen(url + "api/version", timeout=1.5) as r:
            if json.load(r).get("app") == APP_NAME:
                return url
    except Exception:
        pass
    return None


def _write_instance(store, url):
    try:
        with open(_instance_file(store), "w") as f:
            json.dump({"url": url, "pid": os.getpid()}, f)
    except OSError:
        traceback.print_exc()


def _remove_instance(store):
    try:
        os.remove(_instance_file(store))
    except OSError:
        pass


def _make_server(app, port, any_port=False):
    from werkzeug.serving import make_server
    for p in ([port] if port else [0] if any_port else [DEFAULT_PORT, 0]):
        try:
            return make_server("127.0.0.1", p, app, threaded=True)
        except OSError:
            continue
    raise RuntimeError(f"could not start the server on port {port or DEFAULT_PORT}")


def _start_server(port, any_port=False):
    """the slow part: imports scikit-learn and the rest, then serves in a
    background thread. returns (server, url). any_port: the tests use a free
    port so they never clash with a copy the user has open on 8765."""
    if FROZEN:
        _use_threads_for_joblib()
    import logging
    from app import create_app
    logging.getLogger("werkzeug").setLevel(logging.WARNING)   # one line per request is just noise here
    server = _make_server(create_app(), port, any_port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/"


def _answers(url):
    try:
        with urllib.request.urlopen(url + "api/version", timeout=5) as r:
            return json.load(r).get("app") == APP_NAME
    except Exception:
        return False


def open_folder(path):
    if sys.platform == "win32":
        os.startfile(path)                          # noqa: S606 (windows only)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class _Window:
    """the small control window. it is built before the server starts, so it
    can say "starting" while the heavy imports run in the background."""

    def __init__(self, version, home, log_path):
        import tkinter as tk
        from tkinter import ttk

        self.home, self.log_path, self.url = home, log_path, None
        self.root = root = tk.Tk()
        root.title(f"{APP_NAME} {version}")
        root.resizable(False, False)
        try:
            self._icon = tk.PhotoImage(file=resource("static", "assets", "acai-icon.png"))
            root.iconphoto(True, self._icon)
        except Exception:
            pass

        frame = ttk.Frame(root, padding=20)
        frame.grid()
        self.heading = ttk.Label(frame, text=f"Starting {APP_NAME}...", font=("TkDefaultFont", 13, "bold"))
        self.heading.grid(row=0, column=0, columnspan=3, sticky="w")
        self.body = ttk.Label(frame, wraplength=380, justify="left", text=(
            "The first start can take a minute while your computer checks the app. "
            "Later starts are quicker."))
        self.body.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 10))
        self.link = ttk.Label(frame, text="", foreground="#2c6fb3")
        self.link.grid(row=2, column=0, columnspan=3, sticky="w")
        ttk.Label(frame, wraplength=380, justify="left", foreground="#6b7785",
                  text=f"Data folder: {home}").grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 14))
        self.open_button = ttk.Button(frame, text=f"Open {APP_NAME}", state="disabled",
                                      command=lambda: self.url and webbrowser.open(self.url))
        self.open_button.grid(row=4, column=0, padx=(0, 8))
        ttk.Button(frame, text="Open data folder", command=lambda: open_folder(home)).grid(row=4, column=1, padx=(0, 8))
        ttk.Button(frame, text="Quit", command=root.destroy).grid(row=4, column=2)
        root.protocol("WM_DELETE_WINDOW", root.destroy)
        root.lift()

    def ready(self, url):
        self.url = url
        self.heading.configure(text=f"{APP_NAME} is running")
        self.body.configure(text=(
            "It opens in your web browser but runs only on this computer, so your data "
            "never leaves it. Keep this window open while you work. Closing it stops "
            f"{APP_NAME}."))
        self.link.configure(text=url)
        self.open_button.configure(state="normal")

    def failed(self, message):
        self.heading.configure(text=f"{APP_NAME} could not start")
        self.body.configure(text=f"{message}\n\nThe details are in {self.log_path}. "
                                 "Please send that file when you report the problem.")


def _run_with_window(args, home, log_path, store, version):
    try:
        win = _Window(version, home, log_path)
    except Exception as e:                         # no display, or python built without tk
        print(f"  no window ({type(e).__name__}: {e})", flush=True)
        traceback.print_exc()
        if args.gui_smoke:
            return 1
        if FROZEN:
            _alert(f"{APP_NAME} is opening without its window",
                   f"It will open in your browser. To stop it, quit {APP_NAME} from the Dock "
                   f"or the taskbar. Details: {log_path}")
        return _run_headless(args, home, store, version)

    state, started = {}, time.monotonic()

    def boot():
        try:
            state["server"], state["url"] = _start_server(args.port, any_port=args.gui_smoke)
        except Exception as e:
            traceback.print_exc()
            state["error"] = f"{type(e).__name__}: {e}"

    def poll():
        if "error" in state:
            win.failed(state["error"])
            if args.gui_smoke:
                state["exit"] = 1
                win.root.after(500, win.root.destroy)
            return
        if "url" in state:
            url = state["url"]
            win.ready(url)
            print(f"\n  {APP_NAME} {version} running at {url}\n  data folder: {home}\n", flush=True)
            if args.gui_smoke:
                state["exit"] = 0 if _answers(url) else 1
                win.root.after(1500, win.root.destroy)
            else:
                _write_instance(store, url)
                if not args.no_browser:
                    webbrowser.open(url)
            return
        if args.gui_smoke and time.monotonic() - started > 600:
            print("  gui smoke: server did not start within 10 minutes", flush=True)
            state["exit"] = 1
            win.root.destroy()
            return
        win.root.after(200, poll)

    def heartbeat():                               # lets python handle signals while tk waits
        win.root.after(500, heartbeat)

    threading.Thread(target=boot, daemon=True).start()
    win.root.after(200, poll)
    win.root.after(500, heartbeat)
    try:
        win.root.mainloop()
    finally:
        if "server" in state:
            state["server"].shutdown()
        if not args.gui_smoke:
            _remove_instance(store)
    if args.gui_smoke:
        print("gui smoke passed" if state.get("exit") == 0 else "gui smoke FAILED", flush=True)
    return state.get("exit", 0)


def _run_headless(args, home, store, version):
    server, url = _start_server(args.port)
    _write_instance(store, url)
    try:
        print(f"\n  {APP_NAME} {version} running at {url}\n  data folder: {home}\n", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0
    finally:
        server.shutdown()
        _remove_instance(store)


def _self_test(url):
    """Drive one analysis through the http api, the same calls the page makes.
    Proves a packaged build has everything it needs (sklearn, lifelines,
    statsmodels, matplotlib, smote, the static files)."""
    import http.cookiejar
    import uuid

    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def call(path, body=None, ctype="application/json", method=None):
        data = json.dumps(body).encode() if ctype == "application/json" and body is not None else body
        req = urllib.request.Request(url + path.lstrip("/"), data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", ctype)
        with opener.open(req, timeout=600) as r:
            raw = r.read().decode()
        return raw if path.endswith("stream") or path == "/" else json.loads(raw)

    def multipart(fields, files):
        boundary = uuid.uuid4().hex
        out = b""
        for k, v in fields:
            out += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        for k, name, content in files:
            out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{name}"\r\n'
                    f"Content-Type: text/csv\r\n\r\n").encode() + content + b"\r\n"
        return out + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"

    def step(name, ok):
        print(("  ok    " if ok else "  FAIL  ") + name, flush=True)
        if not ok:
            raise SystemExit(1)

    try:
        step("page loads", "ClinTAB-ML" in call("/"))
        with open(resource("Data", "sample_clinical.csv"), "rb") as f:
            body, ctype = multipart([], [("file", "sample_clinical.csv", f.read())])
        up = call("/api/upload", body, ctype)
        step("upload and column typing", up.get("n_rows") == 600)
        types = {c["name"]: c["type"] for c in up["columns"]}
        conf = call("/api/confirm", {"types": types, "missing": {"rare_lab": "remove"},
                                     "ratios": [70, 15, 15], "method": "stratified",
                                     "stratify_col": "mortality_30d", "smote": True})
        step("stratified split", conf.get("method") == "stratified")
        step("data summary", "card" in call("/api/summary"))
        cfg = call("/api/train", {"outcome": "mortality_30d", "models": ["LogisticRegression", "RandomForest"],
                                  "exclude": ["surv_time", "death_event", "admit_date"],
                                  "grid_search": True, "smote": True, "scoring": "roc"})
        step("training config", cfg.get("task") == "binary")
        events = [json.loads(c[6:]) for c in call("/api/train/stream").split("\n\n") if c.startswith("data: ")]
        done = [e["saved_as"] for e in events if e["event"] == "model_done"]
        errors = [e for e in events if e["event"] == "model_error"]
        step(f"grid search, smote and training ({len(done)} models, {len(errors)} errors)", len(done) == 2 and not errors)
        body, ctype = multipart([("models", done[0]), ("models", done[1]), ("source", "test")], [])
        res = call("/api/test", body, ctype)
        step("test set evaluation and plots", all("metrics" in m for m in res["models"]) and "roc_png" in res)
        step("restricted cubic spline", "logodds" in call("/api/spline", {"predictor": "age", "outcome": "mortality_30d"}))
        step("kaplan-meier", "curves" in call("/api/epi/km", {"time": "surv_time", "event": "death_event", "group": "sex"}))
        step("cox model", "rows" in call("/api/epi/cox", {"time": "surv_time", "event": "death_event", "covariates": ["age", "diabetes"]}))
        step("2x2 table", "odds_ratio" in call("/api/epi/or", {"a": 20, "b": 80, "c": 10, "d": 90}))
        step("hosmer-lemeshow", "hl_statistic" in call("/api/epi/hl", {"model": done[0]}))
        step("methods text", "LogisticRegression" in call("/api/report/methods")["text"])
        step("tripod+ai report", call("/api/report/tripod")["summary"]["total"] > 0)
    except SystemExit:
        raise
    except Exception as e:
        print(f"  FAIL  {type(e).__name__}: {e}", flush=True)
        return 1
    print("self test passed", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog=APP_NAME, description="Start ClinTAB-ML on this computer.")
    parser.add_argument("--headless", action="store_true", help="no window, stop with ctrl+c")
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    parser.add_argument("--port", type=int, default=0, help=f"port to use (default {DEFAULT_PORT}, or any free one)")
    parser.add_argument("--self-test", action="store_true", help="run one analysis end to end and exit")
    parser.add_argument("--gui-smoke", action="store_true", help="open the window, wait for the server, close and exit")
    # macos passes a -psn_ argument to apps opened from finder
    args, _unknown = parser.parse_known_args(argv)
    if args.gui_smoke:
        args.no_browser = True

    home = _setup_environment(args.self_test or args.gui_smoke)
    log_path = _start_log(home)
    from clintab import __version__, store         # light, after clintab_home is set

    if args.self_test:
        server, url = _start_server(args.port, any_port=True)
        try:
            return _self_test(url)
        finally:
            server.shutdown()

    if not args.gui_smoke:
        url = _running_instance(store)
        if url:                                    # already open, just bring it up again
            webbrowser.open(url)
            return 0

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # so the clean up still runs
    try:
        if args.headless:
            return _run_headless(args, home, store, __version__)
        return _run_with_window(args, home, log_path, store, __version__)
    except SystemExit:
        raise
    except Exception as e:                         # anything unexpected: say so instead of vanishing
        traceback.print_exc()
        if FROZEN:
            _alert(f"{APP_NAME} could not start", f"{type(e).__name__}: {e}\n\nDetails: {log_path}")
        return 1


if __name__ == "__main__":
    multiprocessing.freeze_support()               # standard guard for packaged builds
    _refuse_python_style_launch()
    sys.exit(main())
