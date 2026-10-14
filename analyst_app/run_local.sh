#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ -x /opt/pyvenv/bin/python ]]; then
  PY=/opt/pyvenv/bin/python
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
else
  PY="$(command -v python3)"
fi
export PYTHONPATH="$ROOT/src"
"$PY" "$ROOT/src/cli.py" \
  --input "$ROOT/data/synthetic_sample.md" \
  --output "$ROOT/output" \
  --query "A medication was nearly administered at the wrong dose after a barcode scanner problem." \
  "$@"
