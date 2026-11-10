#!/usr/bin/env bash
# One command to open the analyst app.
#
#   ./start_app.sh          start the app (sets everything up the first time)
#   ./start_app.sh --demo   same, and load 1,300 reports from the dataset so
#                           every page has something to show
#
# First run takes 5-10 minutes (Python packages, the severity model, the
# web pages). Later runs start in seconds. Then open http://127.0.0.1:8000
# Stop with Ctrl-C.
#
# Needs: Python 3.10-3.12, Node.js 18+, and the dataset at
# data/LOCAL_ONLY_student_facing_candidate.parquet (or SEVERITY_DATA_PATH).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
DEMO=false
[[ "${1:-}" == "--demo" ]] && DEMO=true

step() { printf '\n==> %s\n' "$1"; }

# 1. Python environment, shared by the models and the app
if [[ ! -x .venv/bin/python ]]; then
  PY=""
  for candidate in python3.11 python3.12 python3.10; do
    if command -v "$candidate" >/dev/null; then PY="$candidate"; break; fi
  done
  if [[ -z "$PY" ]]; then
    echo "Install Python 3.11 (python.org or 'brew install python@3.11') and run this again." >&2
    exit 1
  fi
  step "Creating the Python environment with $PY (first run only)"
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install --quiet -r requirements.txt -r analyst_app/requirements.txt
fi

# 2. The severity model (trained from the dataset, never committed)
if [[ ! -f severity_model/model.joblib ]]; then
  DATA="${SEVERITY_DATA_PATH:-data/LOCAL_ONLY_student_facing_candidate.parquet}"
  if [[ ! -f "$DATA" ]]; then
    echo "Put the dataset at $DATA (from the team's shared capstone files) and run this again." >&2
    exit 1
  fi
  step "Training the severity model (first run only, a few minutes)"
  .venv/bin/python severity_model.py
fi

# 3. The text model behind similar reports and themes (no report data is sent)
if ! HF_HUB_OFFLINE=1 .venv/bin/python -c \
  "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')" \
  >/dev/null 2>&1; then
  step "Downloading the MiniLM text model (first run only)"
  .venv/bin/python -c \
    "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')"
fi

# 4. The web pages
if [[ ! -f analyst_app/frontend/dist/index.html ]]; then
  if ! command -v npm >/dev/null; then
    echo "Install Node.js 18 or newer (nodejs.org) and run this again." >&2
    exit 1
  fi
  step "Building the web pages (first run only)"
  (cd analyst_app/frontend && npm ci --silent && npm run build --silent)
fi

# 5. Optional: the event-type model from the 11-buckets repo
if [[ ! -f ../11-buckets/outputs/baseline/model_seed0.pt ]]; then
  echo "Note: event types will show as unavailable. To turn them on, clone 11-buckets"
  echo "next to this folder and train it (see README, 'Open the app')."
fi

export MIDAS_PYTHON="$ROOT/.venv/bin/python"
if $DEMO; then
  export MIDAS_DATABASE="local_state/demo.sqlite3"
  step "Starting the app and loading demo reports"
  analyst_app/run_app.sh &
  SERVER=$!
  trap 'kill $SERVER 2>/dev/null' INT TERM EXIT
  .venv/bin/python analyst_app/load_demo.py
  echo "Open http://127.0.0.1:8000  (Ctrl-C to stop)"
  wait $SERVER
else
  export MIDAS_DATABASE="${MIDAS_DATABASE:-local_state/midas.sqlite3}"
  step "Starting the app: open http://127.0.0.1:8000  (Ctrl-C to stop)"
  exec analyst_app/run_app.sh
fi
