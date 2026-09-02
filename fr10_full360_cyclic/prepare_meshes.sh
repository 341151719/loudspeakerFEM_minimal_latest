#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
: "${PYTHON:=python3}"
$PYTHON generate_structural_meshes.py
$PYTHON gen_periodic_quarter_ac.py
