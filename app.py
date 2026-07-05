"""app.py
Creates the Flask app and starts the server. Nothing else lives here -- all
endpoints are in routes.py and all real work is in ml.py / stats.py / spline.py
/ epi.py / plots.py.

Run locally:        python app.py
Run on a server:    gunicorn -w 1 -k gthread --threads 8 -b 0.0.0.0:5000 "app:create_app()"
                    (use 1 worker so SSE training streams + the Flask session
                     stay on the same process; threads handle concurrency.)
"""
import math
import os

import numpy as np
from flask import Flask
from flask.json.provider import DefaultJSONProvider

import store
from routes import bp


def _sanitize(o):
    """Recursively make a payload strict-JSON safe: NaN/Infinity -> null, and
    numpy scalars/arrays -> plain Python. Browsers' JSON.parse rejects the NaN
    and Infinity tokens that Python's json emits by default, which otherwise
    surfaces as 'server returned an unexpected response' on the client."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, np.floating):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return _sanitize(o.tolist())
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o


class SafeJSONProvider(DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        return super().dumps(_sanitize(obj), **kwargs)


def create_app():
    store.ensure_dirs()
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.json = SafeJSONProvider(app)
    app.secret_key = os.environ.get("CLINTAB_SECRET", "clintab-dev-secret-change-me")
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024   # 512 MB uploads
    app.register_blueprint(bp)
    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    app = create_app()
    print(f"\n  ClinTAB-ML-Foundry running at http://{host}:{port}\n")
    # threaded=True so the SSE training stream doesn't block other requests
    app.run(host=host, port=port, debug=bool(os.environ.get("DEBUG")), threaded=True)
