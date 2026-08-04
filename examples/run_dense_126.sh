#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python cli.py sweep --freqs comsol_126 --jobs 0 --blas-threads 1 --drive current --current 1 --save-each --render --outdir runs/comsol126
