#!/usr/bin/env bash
set -euo pipefail

if (($# == 0)); then
    echo "usage: $0 COMMAND [ARG ...]" >&2
    exit 2
fi

total_cpus=$(nproc)
cpu_limit=$((total_cpus * 80 / 100))
((cpu_limit >= 1)) || cpu_limit=1
last_cpu=$((cpu_limit - 1))

mem_total_kib=$(awk '/MemTotal:/ {print $2}' /proc/meminfo)
mem_limit_kib=$((mem_total_kib * 80 / 100))
ulimit -v "$mem_limit_kib"

export OMP_NUM_THREADS="$cpu_limit"
export OPENBLAS_NUM_THREADS="$cpu_limit"
export MKL_NUM_THREADS="$cpu_limit"
export NUMEXPR_NUM_THREADS="$cpu_limit"
export VECLIB_MAXIMUM_THREADS="$cpu_limit"
export GMSH_NUM_THREADS="$cpu_limit"

echo "resource limit: ${cpu_limit}/${total_cpus} logical CPUs, $((mem_limit_kib / 1024)) MiB virtual memory" >&2
exec nice -n 10 taskset --cpu-list "0-${last_cpu}" "$@"
