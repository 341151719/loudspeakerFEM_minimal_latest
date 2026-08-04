#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "best_model"))

from loudspeaker_axisym_fem.axisym_magnetics import load_tagged_meshio
from native_blocked_coil import NativeBlockedCoil


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate the repaired native blocked impedance")
    ap.add_argument("--reference", required=True, help="COMSOL blocked-impedance CSV")
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--tangent-factor",
        type=float,
        help="override tangent anisotropy factor; 1.0 is the uncalibrated physics model",
    )
    args = ap.parse_args()

    cfg = json.loads((ROOT / "configs/best_model.json").read_text(encoding="utf-8"))
    bcfg = dict(cfg["blocked_coil"])
    if args.tangent_factor is not None:
        bcfg["tangent_anisotropy_factor"] = float(args.tangent_factor)
    mesh = load_tagged_meshio(ROOT / bcfg["field_mesh"])
    solver = NativeBlockedCoil.from_vtu(
        mesh, ROOT / bcfg["magnetostatic_vtu"], bcfg
    )

    ref = pd.read_csv(args.reference)
    f = ref["freq_Hz"].to_numpy(float)
    z_ref = (
        ref["Z_blocked_real_ohm"].to_numpy(float)
        + 1j * ref["Z_blocked_imag_ohm"].to_numpy(float)
    )
    t0 = time.perf_counter()
    z = np.asarray([solver.solve(float(fi)).raw_impedance_ohm for fi in f])
    elapsed = time.perf_counter() - t0
    err = z - z_ref

    out = pd.DataFrame(
        {
            "freq_Hz": f,
            "native_R_ohm": z.real,
            "native_X_ohm": z.imag,
            "native_L_H": z.imag / (2.0 * math.pi * f),
            "comsol_R_ohm": z_ref.real,
            "comsol_X_ohm": z_ref.imag,
            "error_R_ohm": err.real,
            "error_X_ohm": err.imag,
            "relative_complex_error_percent": 100.0
            * np.abs(err)
            / np.maximum(np.abs(z_ref), 1e-300),
        }
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "blocked_impedance_comparison.csv", index=False)

    metrics = {
        "n_frequencies": int(len(f)),
        "tangent_anisotropy_factor": float(
            bcfg.get("tangent_anisotropy_factor", 1.0)
        ),
        "COMSOL_fitted": bool(
            abs(float(bcfg.get("tangent_anisotropy_factor", 1.0)) - 1.0) > 1e-12
        ),
        "complex_NRMSE_percent": float(
            100.0
            * np.sqrt(np.mean(np.abs(err) ** 2))
            / np.sqrt(np.mean(np.abs(z_ref) ** 2))
        ),
        "R_RMSE_ohm": float(np.sqrt(np.mean(err.real**2))),
        "X_RMSE_ohm": float(np.sqrt(np.mean(err.imag**2))),
        "amplitude_RMSE_dB": float(
            np.sqrt(np.mean((20.0 * np.log10(np.abs(z) / np.abs(z_ref))) ** 2))
        ),
        "phase_RMSE_deg": float(
            np.sqrt(np.mean(np.angle(z / z_ref, deg=True) ** 2))
        ),
        "max_relative_complex_error_percent": float(
            np.max(100.0 * np.abs(err) / np.maximum(np.abs(z_ref), 1e-300))
        ),
        "solve_seconds_serial": float(elapsed),
        "seconds_per_frequency": float(elapsed / len(f)),
        "runtime_COMSOL_required": False,
        "runtime_COMSOL_table_required": False,
    }
    (outdir / "blocked_impedance_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
