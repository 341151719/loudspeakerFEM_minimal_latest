#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -d "$ROOT/.venv" ]]; then
  echo "Missing .venv. Run: $ROOT/setup.sh" >&2
  return 2 2>/dev/null || exit 2
fi
source "$ROOT/.venv/bin/activate"
export PYTHONPATH="$ROOT/src:$ROOT/best_model:${PYTHONPATH:-}"
