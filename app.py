"""app.py
Creates the Flask app and starts the server. Nothing else lives here, all
endpoints are in routes.py and all real work is in clintab/ (ml.py / stats.py
/ spline.py / epi.py / plots.py).

Run locally:        python app.py
Run on a server:    gunicorn -w 1 -k gthread --threads 8 -b 127.0.0.1:5000 "app:create_app()"
                    (use 1 worker so SSE training streams + the Flask session
                     stay on the same process, threads handle concurrency.
                     Keep it on localhost or behind an authenticating proxy,
                     there is no login and /api/model/upload loads pickles.)
"""
import os
import secrets
import sys

from flask import Flask, jsonify, request
from flask.json.provider import DefaultJSONProvider
from werkzeug.exceptions import HTTPException

from clintab import store
from routes import bp, _sanitize


class SafeJSONProvider(DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        return super().dumps(_sanitize(obj), **kwargs)


def _secret_key():
    """CLINTAB_SECRET if set, otherwise a random key kept in runtime/ so
    session cookies survive restarts but aren't signed with a known default."""
    key = os.environ.get("CLINTAB_SECRET")
    if key:
        return key
    path = os.path.join(store.RUNTIME, ".secret_key")
    try:
        # o_excl: if several workers start at once, only one gets to write
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_hex(32))
    except FileExistsError:
        pass
    with open(path) as f:
        return f.read().strip()


def _env_flag(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def create_app():
    store.ensure_dirs()
    # sys._meipass is where a pyinstaller build unpacks static/
    here = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    app = Flask(__name__, static_folder=os.path.join(here, "static"), static_url_path="/static")
    app.json = SafeJSONProvider(app)
    app.secret_key = _secret_key()
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024   # 512 mb uploads
    app.register_blueprint(bp)

    # the front end always does r.json(), so api errors have to be json too
    @app.errorhandler(Exception)
    def _api_error(e):
        if isinstance(e, HTTPException):
            code, msg = e.code, e.description
        else:
            app.logger.exception("unhandled error in %s", request.path)
            code, msg = 500, f"Server error: {e}"
        if request.path.startswith("/api/"):
            return jsonify({"error": msg}), code
        return e if isinstance(e, HTTPException) else (msg, code)

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    app = create_app()
    print(f"\n  ClinTAB-ML running at http://{host}:{port}\n")
    # threaded, so the sse training stream doesn't block other requests
    app.run(host=host, port=port, debug=_env_flag("DEBUG"), threaded=True)
