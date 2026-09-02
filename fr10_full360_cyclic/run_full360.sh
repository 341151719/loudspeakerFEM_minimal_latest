#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
# Set these if using the supplied offline wheel/runtime environment.
: "${PYTHON:=python3}"
$PYTHON cyclic_full360_solver.py --freq 90 500 1000 2000
$PYTHON cyclic_full360_solver.py --freq 2000 --diagnostic-phases 1 2 3
