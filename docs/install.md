# Installation

## Requirements

- Python 3.10 or later (tested on 3.10 and 3.12)
- pip

All Python dependencies are pinned to minimum versions in
[`requirements.txt`](../requirements.txt): Flask, pandas, numpy,
scikit-learn, imbalanced-learn, matplotlib, scipy, statsmodels, patsy,
lifelines, joblib, and gunicorn.

## Desktop app (no Python needed)

Clinicians should use the installer from the
[latest release](https://github.com/Applied-Clinical-AI-Initiative/ClinTab-ML/releases/latest)
instead of anything below. See "For clinicians" in the main README, and
[`packaging/README.md`](../packaging/README.md) for how the installers are built.

To run the desktop launcher from source (opens the browser and the control window):

```bash
python clintab_desktop.py
```

## Quick install (recommended)

From the repo root:

```bash
./run.sh
```

This creates a virtual environment at `.venv`, installs everything from
`requirements.txt`, and starts the web app at `http://127.0.0.1:5000`. On
later runs it reuses the existing `.venv`, so startup is fast.

## Manual install

If you'd rather manage the virtual environment yourself:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in a browser.

## Installing for development / running the tests

The test suite needs `pytest` on top of the runtime dependencies:

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## Running the CLI instead of the web app

No separate install step. The CLI (`clintab_cli.py`) uses the same
dependencies as the web app:

```bash
python clintab_cli.py summarize --csv Data/sample_clinical.csv
```

## Production / server deployment

```bash
./run.sh prod
# or directly:
gunicorn -w 1 -k gthread --threads 8 --timeout 600 -b 127.0.0.1:5000 "app:create_app()"
```

Use exactly **one** gunicorn worker with threads. The training progress
stream (Server-Sent Events) and the Flask session need to stay on the same
process; threads handle concurrent requests fine for a research tool. Both
commands bind `127.0.0.1` (change it with `HOST`). The app has no login and
loads uploaded model pickles, so only expose it beyond your own machine behind
an authenticating reverse proxy, and set `CLINTAB_SECRET` (see Known limitations
in the main README).

## Troubleshooting

- **`ModuleNotFoundError: No module named 'lifelines'`** (or any other
  package): the environment `python`/`pip` is resolving to isn't the one you
  installed `requirements.txt` into. Confirm `which python3` and `which pip`
  point into `.venv`, or re-run `pip install -r requirements.txt`.
- **Port already in use.** Set a different port: `PORT=5050 python app.py`.
- **CSV upload fails to parse.** The app reads uploads with
  `pandas.read_csv`; make sure the file is a standard comma-separated CSV
  with a header row.
