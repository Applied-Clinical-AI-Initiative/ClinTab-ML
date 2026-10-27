#!/usr/bin/env python3
"""clintab_desktop.py
Desktop launcher. Starts ClinTAB-ML on this computer only (127.0.0.1), opens
it in the default browser and shows a small window to reopen it, open the
data folder, or quit. This is what the packaged app (packaging/clintab.spec)
runs when someone double-clicks it.

    python clintab_desktop.py               window + browser
    python clintab_desktop.py --headless    no window, stop with ctrl+c
    python clintab_desktop.py --self-test   start, run one analysis end to end, exit 0 or 1

The packaged app keeps sessions, uploads and models in the user's local app
data folder (not Documents, which hospitals often sync to the cloud). Running
from source keeps them in the repo, like app.py. CLINTAB_HOME overrides both.
"""
import argparse
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
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


def _setup_environment(self_test):
    """Pick the data folder before clintab.store is imported (it reads
    CLINTAB_HOME at import time)."""
    if self_test:
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

    # a windowed build has no console, so stdout/stderr are none and the first
    # log line from flask would crash it. send them to a log file instead.
    if sys.stdout is None or sys.stderr is None:
        log_dir = os.path.join(home, "runtime")
        os.makedirs(log_dir, exist_ok=True)
        log = open(os.path.join(log_dir, "clintab-desktop.log"), "a", buffering=1, encoding="utf-8")
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log
    return home


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


def _make_server(app, port):
    from werkzeug.serving import make_server
    for p in ([port] if port else [DEFAULT_PORT, 0]):
        try:
            return make_server("127.0.0.1", p, app, threaded=True)
        except OSError:
            continue
    raise SystemExit(f"Could not start the server on port {port}.")


def open_folder(path):
    if sys.platform == "win32":
        os.startfile(path)                          # noqa: S606 (windows only)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def _window(url, home, version):
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title(f"{APP_NAME} {version}")
    root.resizable(False, False)
    try:
        icon = tk.PhotoImage(file=resource("static", "assets", "acai-icon.png"))
        root.iconphoto(True, icon)
    except Exception:
        pass

    frame = ttk.Frame(root, padding=20)
    frame.grid()
    ttk.Label(frame, text=f"{APP_NAME} is running", font=("TkDefaultFont", 13, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w")
    ttk.Label(frame, wraplength=380, justify="left", text=(
        "It opens in your web browser but runs only on this computer, so your data "
        "never leaves it. Keep this window open while you work. Closing it stops "
        f"{APP_NAME}.")).grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 10))
    ttk.Label(frame, text=url, foreground="#2c6fb3").grid(row=2, column=0, columnspan=3, sticky="w")
    ttk.Label(frame, wraplength=380, justify="left", foreground="#6b7785",
              text=f"Data folder: {home}").grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 14))
    ttk.Button(frame, text=f"Open {APP_NAME}", command=lambda: webbrowser.open(url)).grid(row=4, column=0, padx=(0, 8))
    ttk.Button(frame, text="Open data folder", command=lambda: open_folder(home)).grid(row=4, column=1, padx=(0, 8))
    ttk.Button(frame, text="Quit", command=root.destroy).grid(row=4, column=2)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


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
    # macos passes a -psn_ argument to apps opened from finder
    args, _unknown = parser.parse_known_args(argv)

    home = _setup_environment(args.self_test)
    if FROZEN:
        _use_threads_for_joblib()
    from clintab import __version__, store         # after clintab_home is set
    from app import create_app
    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)   # one line per request is just noise here

    if not args.self_test:
        url = _running_instance(store)
        if url:                                    # already open, just bring it up again
            webbrowser.open(url)
            return 0

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # so the finally below still runs
    server = _make_server(create_app(), args.port)
    url = f"http://127.0.0.1:{server.server_port}/"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    if not args.self_test:
        with open(_instance_file(store), "w") as f:
            json.dump({"url": url, "pid": os.getpid()}, f)

    try:
        if args.self_test:
            return _self_test(url)
        print(f"\n  {APP_NAME} {__version__} running at {url}\n  data folder: {home}\n", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        if not args.headless:
            try:
                _window(url, home, __version__)
                return 0
            except Exception as e:                 # no display, or python built without tk
                print(f"  (no window: {e}). press ctrl+c to stop.", flush=True)
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0
    finally:
        server.shutdown()
        if not args.self_test:
            try:
                os.remove(_instance_file(store))
            except OSError:
                pass


if __name__ == "__main__":
    multiprocessing.freeze_support()               # standard guard for packaged builds
    _refuse_python_style_launch()
    sys.exit(main())
