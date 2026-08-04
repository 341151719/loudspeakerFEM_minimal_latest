#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python cli.py magnetics --outdir runs/magnetics
python cli.py sweep --freqs diagnostic --drive current --current 1 --save-each --render --outdir runs/diagnostic
