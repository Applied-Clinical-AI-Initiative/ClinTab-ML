#!/usr/bin/env bash
# Launch ClinTAB-ML-Foundry from the terminal.
#   ./run.sh            -> local dev server at http://127.0.0.1:5000
#   ./run.sh prod       -> gunicorn (production, binds 0.0.0.0:5000)
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment (.venv)…"
  python3 -m venv .venv
  ./.venv/bin/pip install --quiet --upgrade pip
  ./.venv/bin/pip install --quiet -r requirements.txt
fi
source .venv/bin/activate

if [ "$1" == "prod" ]; then
  exec gunicorn -w 1 -k gthread --threads 8 --timeout 600 -b 0.0.0.0:${PORT:-5000} "app:create_app()"
else
  exec python app.py
fi
