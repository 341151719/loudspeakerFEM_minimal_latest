#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$ROOT/.venv/bin/python" -m pip install -e "$ROOT[test]"
echo "Environment ready. Run: source '$ROOT/activate.sh'"
