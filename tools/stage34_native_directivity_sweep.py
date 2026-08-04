#!/usr/bin/env python3
"""Resource-limited native complex-directivity sweep without field dumps."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "best_model"))

from cli import parse_freqs
from coupled_solver import build_best_model, load_config, solve_frequency

_HIGH = None
_LOW = None
_CROSSOVER = 3000.0
_LIMITER = None
_NRA_ENABLED = True


def _init(high_config: str, low_config: str | None, crossover: float, threads: int, nra_enabled: bool) -> None:
    global _HIGH, _LOW, _CROSSOVER, _LIMITER, _NRA_ENABLED
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(max(1, int(threads)))
    from threadpoolctl import threadpool_limits
    _LIMITER = threadpool_limits(limits=max(1, int(threads)))
    magnetics = ROOT / "inputs/comsol_reference/magnetostatic_converged_55iter.vtu"
    _HIGH = build_best_model(ROOT, config_path=high_config, magnetostatic_vtu=magnetics)
    _LOW = None if low_config is None else build_best_model(ROOT, config_path=low_config, magnetostatic_vtu=magnetics)
    _CROSSOVER = float(crossover)
    _NRA_ENABLED = bool(nra_enabled)


def _domain_mean_abs(model, pressure_base: np.ndarray, domain: int) -> float:
    mesh = model.mesh
    mask = mesh.tri_domains == int(domain)
    tri = mesh.triangles[mask]
    local = np.asarray([[model.acoustic_model.acoustic_node_map[int(g)] for g in t] for t in tri], int)
    xy = mesh.points_rz_m[tri]
    center = xy.mean(axis=1)
    area = 0.5 * np.abs(
        (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
        - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1])
    )
    weight = 2 * np.pi * np.maximum(center[:, 0], 1e-12) * area
    value = np.abs(pressure_base[local].mean(axis=1))
    return float(np.sum(weight * value) / max(np.sum(weight), 1e-300))


def _one(freq: float) -> dict:
    model = _LOW if _LOW is not None and float(freq) <= _CROSSOVER else _HIGH
    s = solve_frequency(model, float(freq), drive="voltage", voltage_V_peak=3.55, nra_enabled=_NRA_ENABLED)
    return {
        "freq_Hz": float(freq),
        "angle_deg": s.directivity_angles_deg,
        "pressure": s.directivity_pressure_Pa_peak,
        "axis_SPL_dB_RMS": float(s.axis_SPL_dB),
        "current": complex(s.current_A_peak),
        "motional_impedance": complex(s.motional_impedance_ohm),
        "total_impedance": complex(s.total_impedance_ohm),
        "domain8_mean_abs_pressure_Pa": _domain_mean_abs(model, s.pressure_base, 8),
        "domain22_mean_abs_pressure_Pa": _domain_mean_abs(model, s.pressure_base, 22),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/best_model.json")
    ap.add_argument("--freqs", default="comsol_126")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--blas-threads", type=int, default=1)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--without-nra", action="store_true")
    args = ap.parse_args()
    cfg = load_config(ROOT, args.config)
    freqs = parse_freqs(args.freqs, cfg)
    hybrid = cfg.get("acoustics", {}).get("hybrid_sweep", {})
    low = hybrid.get("low_frequency_config") if hybrid.get("enabled", False) else None
    crossover = float(hybrid.get("crossover_Hz", -np.inf))
    jobs = max(1, min(int(args.jobs), len(freqs), 8))
    start = time.perf_counter()
    results = []
    with ProcessPoolExecutor(
        max_workers=jobs,
        initializer=_init,
        initargs=(args.config, low, crossover, args.blas_threads, not args.without_nra),
    ) as pool:
        futures = {pool.submit(_one, float(f)): float(f) for f in freqs}
        for i, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            print(f"[{i}/{len(freqs)}] {futures[future]:g} Hz", flush=True)
    results.sort(key=lambda x: x["freq_Hz"])
    directivity = []
    sweep = []
    for row in results:
        p = row["pressure"]
        a = row["angle_deg"]
        i0 = int(np.argmin(np.abs(a)))
        relative = 20 * np.log10(np.maximum(np.abs(p) / max(abs(p[i0]), 1e-300), 1e-300))
        directivity.extend({
            "freq_Hz": row["freq_Hz"], "angle_deg": float(angle),
            "p_real_Pa": float(value.real), "p_imag_Pa": float(value.imag),
            "relative_dB": float(rel),
        } for angle, value, rel in zip(a, p, relative))
        sweep.append({
            "freq_Hz": row["freq_Hz"], "axis_SPL_dB_RMS": row["axis_SPL_dB_RMS"],
            "current_real_A": row["current"].real, "current_imag_A": row["current"].imag,
            "Zmot_real_ohm": row["motional_impedance"].real, "Zmot_imag_ohm": row["motional_impedance"].imag,
            "Ztotal_real_ohm": row["total_impedance"].real, "Ztotal_imag_ohm": row["total_impedance"].imag,
            "domain8_mean_abs_pressure_Pa": row["domain8_mean_abs_pressure_Pa"],
            "domain22_mean_abs_pressure_Pa": row["domain22_mean_abs_pressure_Pa"],
        })
    args.outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(directivity).to_csv(args.outdir / "native_directivity_complex.csv", index=False)
    pd.DataFrame(sweep).to_csv(args.outdir / "native_sweep_metrics.csv", index=False)
    metadata = {
        "frequency_count": len(freqs), "jobs": jobs, "blas_threads": int(args.blas_threads),
        "elapsed_seconds": time.perf_counter() - start, "config": args.config,
        "hybrid_low_config": low, "hybrid_crossover_Hz": crossover if low else None,
        "comsol_runtime_data_consumed": False,
        "nra_enabled": not args.without_nra,
    }
    (args.outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
