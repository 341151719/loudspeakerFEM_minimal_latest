#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python cli.py solve --freq 50 --drive current --current 1 --render --outdir runs/solve_50Hz
