#!/usr/bin/env python3
"""Stage35 data-driven layer substitution for the high-frequency chain.

COMSOL data are diagnostic inputs to this tool only.  They are never imported by
the production solver.  The tool answers which replacement (structure motion,
acoustic propagation, Boundary93 p/q, or HK) produces the large error drop.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "best_model"), str(ROOT / "src")]

from coupled_solver import build_best_model, _exterior_field
from loudspeaker_axisym_fem.exterior_field import hk_pressure_from_samples
from loudspeaker_axisym_fem.stage4F_hk_refinement import boundary93_hk_samples_recovered


def _complex(frame: pd.DataFrame, re: str, im: str) -> np.ndarray:
    return frame[re].to_numpy(float) + 1j * frame[im].to_numpy(float)


def _metrics(pred: np.ndarray, ref: np.ndarray, angles: np.ndarray) -> dict:
    pred = np.asarray(pred, complex)
    ref = np.asarray(ref, complex)
    angles = np.asarray(angles, float)
    i0 = int(np.argmin(np.abs(angles)))
    pred_rel = 20 * np.log10(np.maximum(np.abs(pred) / max(abs(pred[i0]), 1e-300), 1e-300))
    ref_rel = 20 * np.log10(np.maximum(np.abs(ref) / max(abs(ref[i0]), 1e-300), 1e-300))
    err = pred_rel - ref_rel
    main = ref_rel >= -20.0
    solid_angle_weight = np.maximum(np.sin(np.deg2rad(np.abs(angles))), 1e-6)
    field_weight = solid_angle_weight * np.abs(ref) ** 2
    field_weight /= max(float(np.sum(field_weight)), 1e-300)
    scale = np.vdot(pred, ref) / max(np.vdot(pred, pred).real, 1e-300)
    corr = abs(np.vdot(pred, ref)) / max(np.linalg.norm(pred) * np.linalg.norm(ref), 1e-300)
    amp_axis = 20 * np.log10(max(abs(pred[i0]), 1e-300) / max(abs(ref[i0]), 1e-300))
    phase_axis = np.angle(pred[i0] / (ref[i0] if abs(ref[i0]) else 1), deg=True)
    out = {
        "relative_RMSE_dB": float(np.sqrt(np.mean(err**2))),
        "relative_max_abs_error_dB": float(np.max(np.abs(err))),
        "main_ge_minus20dB_RMSE_dB": float(np.sqrt(np.mean(err[main] ** 2))),
        "field_energy_weighted_relative_RMSE_dB": float(np.sqrt(np.sum(field_weight * err**2))),
        "shape_NRMSE_pct": float(100 * np.linalg.norm(scale * pred - ref) / max(np.linalg.norm(ref), 1e-300)),
        "raw_complex_NRMSE_pct": float(100 * np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1e-300)),
        "solid_angle_field_weighted_raw_complex_NRMSE_pct": float(
            100 * np.sqrt(np.sum(field_weight * np.abs(pred - ref) ** 2 / np.maximum(np.abs(ref) ** 2, 1e-300)))
        ),
        "complex_correlation": float(corr),
        "diagnostic_scale_abs": float(abs(scale)),
        "diagnostic_scale_phase_deg": float(np.angle(scale, deg=True)),
        "axis_amplitude_error_dB": float(amp_axis),
        "axis_phase_error_deg": float(phase_axis),
    }
    for threshold in (-10, -20, -30):
        mask = ref_rel >= threshold
        out[f"ge_{threshold}dB_RMSE_dB"] = float(np.sqrt(np.mean(err[mask] ** 2))) if np.any(mask) else float("nan")

    def beamwidth(rel: np.ndarray) -> float:
        positive = np.flatnonzero(angles >= 0)
        a = angles[positive]; y = rel[positive]
        below = np.flatnonzero(y <= -6.0)
        if not len(below): return 180.0
        j = int(below[0])
        if j == 0: return 0.0
        den = y[j] - y[j - 1]
        if abs(den) < 1e-300: return float(2 * a[j])
        crossing = a[j - 1] + (-6.0 - y[j - 1]) * (a[j] - a[j - 1]) / den
        return float(2 * crossing)
    out["beamwidth_6dB_pred_deg"] = beamwidth(pred_rel)
    out["beamwidth_6dB_ref_deg"] = beamwidth(ref_rel)
    out["beamwidth_6dB_error_deg"] = out["beamwidth_6dB_pred_deg"] - out["beamwidth_6dB_ref_deg"]
    return out


def _reference_directivity(path: Path, freq: float) -> tuple[np.ndarray, np.ndarray]:
    d = pd.read_csv(path)
    d = d[np.isclose(d.freq_Hz.astype(float), freq)].sort_values("theta_deg")
    if not len(d):
        raise ValueError(f"frequency {freq:g} Hz missing from {path}")
    return d.theta_deg.to_numpy(float), _complex(d, "pext_real_Pa", "pext_imag_Pa")


def _collapse_interp_coordinate(x: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x = np.asarray(x, float)[order]
    values = np.asarray(values, complex)[order]
    # Eval may return coincident element-end points.  Average these before interpolation.
    key = np.round(x, 12)
    unique, inverse = np.unique(key, return_inverse=True)
    total = np.zeros(len(unique), complex)
    count = np.zeros(len(unique), int)
    np.add.at(total, inverse, values)
    np.add.at(count, inverse, 1)
    return unique, total / np.maximum(count, 1)


def _interp_complex(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    xp, fp = _collapse_interp_coordinate(xp, fp)
    return np.interp(x, xp, fp.real) + 1j * np.interp(x, xp, fp.imag)


def _native_samples(model, pressure_mixed: np.ndarray):
    ext = model.config["exterior"]
    if hasattr(model.acoustic_operator, "boundary_samples"):
        return model.acoustic_operator.boundary_samples(
            pressure_mixed,
            boundary_id=int(ext.get("boundary_id", 93)),
            intorder=4,
            force_radial_normals=bool(ext.get("force_radial_normals", True)),
        )[0]
    return boundary93_hk_samples_recovered(
        model.acoustic_model, model.acoustic_operator.base_pressure(pressure_mixed),
        boundary_id=int(ext.get("boundary_id", 93)),
        recovery_method="ppr" if str(ext.get("recovery_method", "ppr")).startswith("ppr") else "zz",
        force_radial_normals=bool(ext.get("force_radial_normals", True)),
    )[0]


def _comsol_boundary93_on_native_samples(path: Path, freq: float, samples):
    d = pd.read_csv(path)
    d = d[np.isclose(d.freq_Hz.astype(float), freq)].copy()
    if not len(d):
        raise ValueError(f"frequency {freq:g} Hz missing from {path}")
    # theta is measured from +z.  COMSOL's Boundary93 normal points inward;
    # the native HK surface normal points from physical air toward the PML.
    theta_ref = np.arctan2(d.r_m.to_numpy(float), d.z_m.to_numpy(float))
    theta_native = np.arctan2(samples[0], samples[1])
    p = _interp_complex(theta_native, theta_ref, _complex(d, "p_real", "p_imag"))
    ppr_cols = ("dpdn_ppr_real", "dpdn_ppr_imag") if "dpdn_ppr_real" in d else ("q_ppr_real", "q_ppr_imag")
    plain_cols = ("dpdn_plain_real", "dpdn_plain_imag") if "dpdn_plain_real" in d else ("q_plain_real", "q_plain_imag")
    q_ppr_outward = _interp_complex(theta_native, theta_ref, -_complex(d, *ppr_cols))
    q_plain_outward = _interp_complex(theta_native, theta_ref, -_complex(d, *plain_cols))
    return p, q_ppr_outward, q_plain_outward


def _hk(model, freq: float, samples, p: np.ndarray, q: np.ndarray, angles: np.ndarray) -> np.ndarray:
    rs, zs, nr, nz, ds, _, _ = samples
    theta = np.deg2rad(angles)
    radius = float(model.config["exterior"]["observation_radius_m"])
    return hk_pressure_from_samples(
        freq, model.config["air"]["c0_m_s"], rs, zs, nr, nz, ds, p, q,
        obs_r=np.abs(np.sin(theta)) * radius,
        obs_z=np.cos(theta) * radius,
        nphi=int(model.config["exterior"]["azimuth_quadrature_points"]),
        mirror=bool(model.config["exterior"]["mirror_sound_hard_plane"]), sign=-1,
    )


def _motion_column_pair(frame: pd.DataFrame) -> tuple[str, str, str, str, float]:
    if {"u_r_SI_real", "u_r_SI_imag", "u_z_SI_real", "u_z_SI_imag"}.issubset(frame.columns):
        return "u_r_SI_real", "u_r_SI_imag", "u_z_SI_real", "u_z_SI_imag", 1.0
    return "u_r_real", "u_r_imag", "u_z_real", "u_z_imag", 1e-3


def _replace_motion(model, native_u: np.ndarray, motion: pd.DataFrame, tags: set[int]) -> tuple[np.ndarray, dict]:
    out = np.asarray(native_u, complex).copy()
    accum = np.zeros(len(model.solid.points_rz_m), complex)
    accum_z = np.zeros(len(model.solid.points_rz_m), complex)
    count = np.zeros(len(model.solid.points_rz_m), int)
    cr, ci, czr, czi, unit = _motion_column_pair(motion)
    max_distance = 0.0
    for tag in sorted(tags):
        ref = motion[motion.boundary_id.astype(int) == tag]
        if not len(ref):
            continue
        xy = ref[["r_m", "z_m"]].to_numpy(float)
        ur = _complex(ref, cr, ci) * unit
        uz = _complex(ref, czr, czi) * unit
        nodes = set()
        for a, b, edge_tag in model.solid.boundary_edges:
            if int(edge_tag) != tag:
                continue
            nodes.update((int(a), int(b), int(model.solid.edge_mid_nodes[tuple(sorted((int(a), int(b))))])))
        if not nodes:
            continue
        nodes = np.asarray(sorted(nodes), int)
        target = model.solid.points_rz_m[nodes]
        tree = cKDTree(xy)
        dist, idx = tree.query(target, k=min(2, len(xy)))
        if np.ndim(idx) == 1:
            val_r, val_z = ur[idx], uz[idx]
            max_distance = max(max_distance, float(np.max(dist)))
        else:
            weight = 1.0 / np.maximum(dist, 1e-14)
            weight /= weight.sum(axis=1, keepdims=True)
            val_r = np.sum(weight * ur[idx], axis=1)
            val_z = np.sum(weight * uz[idx], axis=1)
            max_distance = max(max_distance, float(np.max(dist[:, 0])))
        accum[nodes] += val_r
        accum_z[nodes] += val_z
        count[nodes] += 1
    used = np.flatnonzero(count)
    out[2 * used] = accum[used] / count[used]
    out[2 * used + 1] = accum_z[used] / count[used]
    return out, {"replaced_nodes": int(len(used)), "max_nearest_distance_m": max_distance, "tags": sorted(tags)}


def _acoustic_from_motion(model, freq: float, u: np.ndarray) -> np.ndarray:
    omega = 2 * math.pi * freq
    A, _ = model.acoustic_operator.matrix(freq, nra_enabled=True)
    G = model.G
    if G.shape[1] < model.acoustic_operator.n2:
        from scipy.sparse import hstack, csr_matrix
        G = hstack([G, csr_matrix((G.shape[0], model.acoustic_operator.n2 - G.shape[1]))], format="csr")
    rhs = model.config["air"]["rho0_kg_m3"] * omega**2 * (G.T @ u)
    return splu(A.tocsc()).solve(rhs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage34_structure_refined2.json")
    ap.add_argument("--frequency", type=float, required=True)
    ap.add_argument("--native-npz", type=Path, required=True)
    ap.add_argument("--comsol-boundary93", type=Path, required=True)
    ap.add_argument("--comsol-directivity", type=Path, required=True)
    ap.add_argument("--comsol-motion", type=Path)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    model = build_best_model(
        ROOT, config_path=args.config,
        magnetostatic_vtu=ROOT / "inputs/comsol_reference/magnetostatic_converged_55iter.vtu",
        build_blocked_coil=False,
    )
    z = np.load(args.native_npz)
    freq = float(args.frequency)
    angles, ref = _reference_directivity(args.comsol_directivity, freq)
    native_u = z["solid_displacement"]
    native_pressure = z["pressure_mixed"]
    native_samples = _native_samples(model, native_pressure)
    cp, cq, cq_plain = _comsol_boundary93_on_native_samples(args.comsol_boundary93, freq, native_samples)
    np_, nq = native_samples[5], native_samples[6]

    variants: dict[str, np.ndarray] = {
        "L0_native_full_chain": z["directivity_pressure_Pa_peak"],
        "L3_COMSOL_B93_p_native_q_to_native_HK": _hk(model, freq, native_samples, cp, nq, angles),
        "L3_native_B93_p_COMSOL_q_to_native_HK": _hk(model, freq, native_samples, np_, cq, angles),
        "L4_COMSOL_B93_pq_to_native_HK": _hk(model, freq, native_samples, cp, cq, angles),
        "L3_native_B93_p_COMSOL_plain_q_to_native_HK": _hk(model, freq, native_samples, np_, cq_plain, angles),
        "L4_COMSOL_B93_p_plain_q_to_native_HK": _hk(model, freq, native_samples, cp, cq_plain, angles),
    }
    notes = {}
    diagnostic_fields = {}

    if args.comsol_motion is not None:
        motion = pd.read_csv(args.comsol_motion)
        motion = motion[np.isclose(motion.freq_Hz.astype(float), freq)].copy()
        interface = set(map(int, model.G_info["interface_boundaries"]))
        groups = {
            "L1_COMSOL_all_ASB_motion_to_native_acoustic_HK": interface,
            "L1_COMSOL_cone46_47_motion_to_native_acoustic_HK": {46, 47},
            "L1_COMSOL_dustcap91_92_motion_to_native_acoustic_HK": {91, 92},
        }
        for name, tags in groups.items():
            u, info = _replace_motion(model, native_u, motion, tags)
            pressure = _acoustic_from_motion(model, freq, u)
            a, p, _, _, _ = _exterior_field(model, freq, pressure)
            variants[name] = np.interp(angles, a, p.real) + 1j * np.interp(angles, a, p.imag)
            notes[name] = info
            # The all-motion pressure also permits the next acoustic/B93 split.
            if tags == interface:
                diagnostic_fields["COMSOL_all_ASB_motion_native_acoustic_pressure_mixed"] = pressure
                s = _native_samples(model, pressure)
                variants["L2_COMSOL_motion_native_acoustic_B93p_COMSOLq_to_HK"] = _hk(
                    model, freq, s, s[5], _comsol_boundary93_on_native_samples(args.comsol_boundary93, freq, s)[1], angles
                )

    rows = []
    curves = []
    for name, pred in variants.items():
        m = _metrics(pred, ref, angles)
        rows.append({"variant": name, **m})
        i0 = int(np.argmin(np.abs(angles)))
        rel = 20 * np.log10(np.maximum(np.abs(pred) / max(abs(pred[i0]), 1e-300), 1e-300))
        for angle, value, rv in zip(angles, pred, rel):
            curves.append({"variant": name, "freq_Hz": freq, "theta_deg": angle,
                           "p_real_Pa": value.real, "p_imag_Pa": value.imag, "relative_dB": rv})
    table = pd.DataFrame(rows).sort_values("main_ge_minus20dB_RMSE_dB")
    baseline = table.loc[table.variant == "L0_native_full_chain"].iloc[0]
    for metric in ("main_ge_minus20dB_RMSE_dB", "field_energy_weighted_relative_RMSE_dB",
                   "shape_NRMSE_pct", "raw_complex_NRMSE_pct"):
        table[f"{metric}_reduction_vs_native_pct"] = 100 * (1 - table[metric] / max(float(baseline[metric]), 1e-300))
    args.outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.outdir / "layer_substitution_metrics.csv", index=False)
    pd.DataFrame(curves).to_csv(args.outdir / "layer_substitution_curves.csv", index=False)
    if diagnostic_fields:
        np.savez_compressed(args.outdir / "diagnostic_fields.npz", **diagnostic_fields)
    result = {
        "frequency_Hz": freq,
        "benchmark_policy": "COMSOL diagnostic substitution only; production runtime consumes none of these data",
        "ranked_by_main_field_RMSE": table.to_dict(orient="records"),
        "motion_projection": notes,
    }
    (args.outdir / "layer_substitution_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
