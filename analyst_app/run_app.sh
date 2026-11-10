#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PYTHON="${MIDAS_PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
umask 077
if [[ ! -x "$PYTHON" ]]; then
  echo 'Create .venv and install requirements.txt, or set MIDAS_PYTHON to an installed interpreter.' >&2
  exit 1
fi
if [[ ! -f "$ROOT/frontend/dist/index.html" ]]; then
  echo 'Build frontend first: cd frontend && npm ci && npm run build' >&2
  exit 1
fi
exec "$PYTHON" -m uvicorn backend.app:create_app --factory \
  --host 127.0.0.1 --port 8000 --no-access-log --log-level critical
