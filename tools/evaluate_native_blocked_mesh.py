#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from best_model.native_blocked_coil import NativeBlockedCoil
from loudspeaker_axisym_fem.axisym_magnetics import load_tagged_meshio


def main() -> int:
    ap = argparse.ArgumentParser(description="Quality-gated native blocked-mesh check")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--magnetostatic-vtu", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs/best_model.json"))
    ap.add_argument("--frequencies", default="50,900,2000,5000,8000")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))["blocked_coil"]
    config = dict(config)
    config["subgrid_closure_enabled"] = False
    mesh = load_tagged_meshio(args.mesh)
    model = NativeBlockedCoil.from_vtu(mesh, args.magnetostatic_vtu, config)

    t0 = time.perf_counter()
    model._ensure_assembled()
    assembly_seconds = time.perf_counter() - t0

    ref = pd.read_csv(args.reference)
    if args.frequencies.strip().lower() == "all":
        requested = [float(x) for x in ref["freq_Hz"].to_numpy()]
    else:
        requested = [float(x) for x in args.frequencies.split(",") if x.strip()]
    rows = []
    for f in requested:
        hit = ref.iloc[int(np.argmin(np.abs(ref["freq_Hz"].to_numpy() - f)))]
        if not np.isclose(float(hit.freq_Hz), f, rtol=0.0, atol=1e-9):
            raise ValueError(f"reference has no exact frequency {f:g} Hz")
        ts = time.perf_counter()
        point = model.solve(f)
        solve_seconds = time.perf_counter() - ts
        z = point.raw_impedance_ohm
        zr = complex(hit.Z_blocked_real_ohm, hit.Z_blocked_imag_ohm)
        rows.append(
            {
                "freq_Hz": f,
                "R_native_ohm": z.real,
                "X_native_ohm": z.imag,
                "R_COMSOL_ohm": zr.real,
                "X_COMSOL_ohm": zr.imag,
                "abs_error_ohm": abs(z - zr),
                "relative_error_percent": 100.0 * abs(z - zr) / abs(zr),
                "solve_seconds": solve_seconds,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "blocked_key_frequencies.csv", index=False)
    error = (frame.R_native_ohm - frame.R_COMSOL_ohm).to_numpy() + 1j * (
        frame.X_native_ohm - frame.X_COMSOL_ohm
    ).to_numpy()
    zref = frame.R_COMSOL_ohm.to_numpy() + 1j * frame.X_COMSOL_ohm.to_numpy()
    metrics = {
        "mesh": str(args.mesh),
        "n_nodes": mesh.n_nodes,
        "n_triangles": mesh.n_triangles,
        "assembly_seconds": assembly_seconds,
        "total_solve_seconds": float(frame.solve_seconds.sum()),
        "mean_solve_seconds": float(frame.solve_seconds.mean()),
        "complex_NRMSE_percent": float(
            100.0 * np.sqrt(np.mean(abs(error) ** 2)) / np.sqrt(np.mean(abs(zref) ** 2))
        ),
        "R_RMSE_ohm": float(np.sqrt(np.mean(error.real ** 2))),
        "X_RMSE_ohm": float(np.sqrt(np.mean(error.imag ** 2))),
        "max_relative_error_percent": float(frame.relative_error_percent.max()),
        "n_frequencies": len(requested),
        "frequencies_Hz": requested,
    }
    (out / "blocked_key_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
