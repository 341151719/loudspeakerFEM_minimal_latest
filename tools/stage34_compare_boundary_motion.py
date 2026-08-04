#!/usr/bin/env python3
"""Localize high-frequency directivity error on radiating structure boundaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "best_model")]

from coupled_solver import build_best_model
from comparison import complex_metrics
from p2_axisym_solid import P2BoundarySampler


def energy_proxy(frame: pd.DataFrame, value: np.ndarray) -> float:
    x = frame[["r_m", "z_m"]].to_numpy(float)
    # The exported points on each speaker boundary are monotonic in radius.
    order = np.lexsort((x[:, 1], x[:, 0]))
    x = x[order]
    v = np.asarray(value)[order]
    ds = np.linalg.norm(np.diff(x, axis=0), axis=1)
    if not len(ds):
        return 0.0
    integrand = 2 * np.pi * np.maximum(x[:, 0], 1e-12) * np.abs(v) ** 2
    return float(np.sum(0.5 * (integrand[:-1] + integrand[1:]) * ds))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage34_structure_refined2.json")
    ap.add_argument("--native-checkpoints", type=Path, required=True)
    ap.add_argument("--comsol-motion", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    model = build_best_model(
        ROOT,
        config_path=args.config,
        magnetostatic_vtu=ROOT / "inputs/comsol_reference/magnetostatic_converged_55iter.vtu",
    )
    ref = pd.read_csv(args.comsol_motion)
    ref = ref[ref.node_id.astype(str) != "ERROR"].copy()
    for name in ("freq_Hz", "boundary_id", "r_m", "z_m", "nr", "nz"):
        ref[name] = pd.to_numeric(ref[name])
    rows = []
    points = []
    for freq, fframe in ref.groupby("freq_Hz"):
        npz = args.native_checkpoints / f"{freq:g}Hz" / f"solution_{freq:g}Hz.npz"
        if not npz.exists():
            continue
        u = np.load(npz)["solid_displacement"]
        for boundary, frame in fframe.groupby("boundary_id"):
            frame = frame.reset_index(drop=True).rename(columns={"nr": "normal_r", "nz": "normal_z"})
            sampler = P2BoundarySampler(model.solid, frame)
            _, _, native = sampler.sample(u)
            # COMSOL numerical Eval returns the displacement display unit (mm).
            comsol = (frame.u_n_real.to_numpy(float) + 1j * frame.u_n_imag.to_numpy(float)) * 1e-3
            raw = np.linalg.norm(native - comsol) / max(np.linalg.norm(comsol), 1e-300)
            m = complex_metrics(native, comsol)
            ne = energy_proxy(frame, native)
            ce = energy_proxy(frame, comsol)
            rows.append({
                "freq_Hz": float(freq), "boundary_id": int(boundary),
                "raw_complex_NRMSE_pct": 100 * float(raw),
                "shape_NRMSE_after_diagnostic_scale_pct": 100 * m["normalized_residual_after_scale"],
                "diagnostic_scale_abs": m["best_scale_abs"],
                "diagnostic_scale_phase_deg": m["best_scale_phase_deg"],
                "native_motion_energy_proxy": ne, "COMSOL_motion_energy_proxy": ce,
                "motion_energy_ratio_native_over_COMSOL": ne / max(ce, 1e-300),
                "max_projection_distance_m": float(np.max(sampler.dist)),
            })
            for i in range(len(frame)):
                points.append({
                    "freq_Hz": float(freq), "boundary_id": int(boundary), "node_id": i + 1,
                    "r_m": frame.r_m.iloc[i], "z_m": frame.z_m.iloc[i],
                    "native_un_real_m": native[i].real, "native_un_imag_m": native[i].imag,
                    "COMSOL_un_real_m": comsol[i].real, "COMSOL_un_imag_m": comsol[i].imag,
                })
    table = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.outdir / "boundary_motion_metrics.csv", index=False)
    pd.DataFrame(points).to_csv(args.outdir / "boundary_motion_points.csv", index=False)
    summary = {}
    for boundary, group in table.groupby("boundary_id"):
        summary[str(int(boundary))] = {
            "frequency_count": int(len(group)),
            "shape_NRMSE_pct_mean": float(group.shape_NRMSE_after_diagnostic_scale_pct.mean()),
            "shape_NRMSE_pct_max": float(group.shape_NRMSE_after_diagnostic_scale_pct.max()),
            "energy_ratio_geometric_mean": float(np.exp(np.mean(np.log(np.maximum(group.motion_energy_ratio_native_over_COMSOL, 1e-300))))),
            "energy_ratio_min": float(group.motion_energy_ratio_native_over_COMSOL.min()),
            "energy_ratio_max": float(group.motion_energy_ratio_native_over_COMSOL.max()),
        }
    (args.outdir / "boundary_motion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
