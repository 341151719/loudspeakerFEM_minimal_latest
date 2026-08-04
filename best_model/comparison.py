from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from loudspeaker_axisym_fem.stage4F_hk_refinement import boundary93_hk_samples_recovered
from p2_axisym_solid import P2BoundarySampler


def cplx(df: pd.DataFrame, real: str, imag: str) -> np.ndarray:
    return df[real].to_numpy(float) + 1j * df[imag].to_numpy(float)


def complex_metrics(pred: np.ndarray, ref: np.ndarray, prefix: str = "") -> dict:
    pred = np.asarray(pred, complex)
    ref = np.asarray(ref, complex)
    mask = np.isfinite(pred.real) & np.isfinite(pred.imag) & np.isfinite(ref.real) & np.isfinite(ref.imag)
    pred = pred[mask]; ref = ref[mask]
    if len(pred) == 0:
        return {prefix + "n": 0}
    scale = np.vdot(pred, ref) / max(np.vdot(pred, pred).real, 1e-300)
    aligned = pred * scale
    corr = abs(np.vdot(pred, ref)) / max(np.linalg.norm(pred) * np.linalg.norm(ref), 1e-300)
    amp_err = 20 * np.log10(np.maximum(np.abs(pred), 1e-300) / np.maximum(np.abs(ref), 1e-300))
    phase_err = np.angle(aligned / np.where(np.abs(ref) > 0, ref, 1), deg=True)
    return {
        prefix + "n": int(len(pred)),
        prefix + "complex_corr": float(corr),
        prefix + "normalized_residual_after_scale": float(np.linalg.norm(aligned - ref) / max(np.linalg.norm(ref), 1e-300)),
        prefix + "best_scale_abs": float(abs(scale)),
        prefix + "best_scale_phase_deg": float(np.angle(scale, deg=True)),
        prefix + "amplitude_RMSE_dB": float(np.sqrt(np.mean(amp_err**2))),
        prefix + "amplitude_MAE_dB": float(np.mean(np.abs(amp_err))),
        prefix + "phase_RMSE_deg_after_scale": float(np.sqrt(np.mean(phase_err**2))),
    }


def _interpolate_complex(freq, source_freq, values):
    x = np.log(np.asarray(source_freq, float))
    xf = np.log(np.asarray(freq, float))
    values = np.asarray(values, complex)
    return np.interp(xf, x, values.real) + 1j * np.interp(xf, x, values.imag)


def compare_impedance_sweep(our_csv: str | Path, req5_raw: str | Path, outdir: str | Path) -> dict:
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    ours = pd.read_csv(our_csv)
    ref = pd.read_csv(Path(req5_raw) / "layer03_impedance_power_decomposition.csv")
    if {"Ztotal_real_ohm", "Ztotal_imag_ohm"}.issubset(ours.columns):
        zo = cplx(ours, "Ztotal_real_ohm", "Ztotal_imag_ohm")
    else:
        raise KeyError("our sweep CSV needs Ztotal_real_ohm/Ztotal_imag_ohm")
    zr = _interpolate_complex(ours.freq_Hz, ref.freq_Hz, cplx(ref, "Z_total_real_ohm", "Z_total_imag_ohm"))
    table = pd.DataFrame({
        "freq_Hz": ours.freq_Hz,
        "our_Z_real_ohm": zo.real,
        "our_Z_imag_ohm": zo.imag,
        "COMSOL_Z_real_ohm": zr.real,
        "COMSOL_Z_imag_ohm": zr.imag,
        "our_Z_abs_ohm": np.abs(zo),
        "COMSOL_Z_abs_ohm": np.abs(zr),
        "Z_abs_error_ohm": np.abs(zo) - np.abs(zr),
        "Z_complex_error_abs_ohm": np.abs(zo - zr),
    })
    table.to_csv(out / "impedance_comparison.csv", index=False)
    metrics = {
        "abs_RMSE_ohm": float(np.sqrt(np.mean(table.Z_abs_error_ohm**2))),
        "abs_max_error_ohm": float(np.max(np.abs(table.Z_abs_error_ohm))),
        "complex_RMSE_ohm": float(np.sqrt(np.mean(table.Z_complex_error_abs_ohm**2))),
    }
    (out / "impedance_comparison_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.semilogx(table.freq_Hz, table.COMSOL_Z_abs_ohm, label="COMSOL")
        ax.semilogx(table.freq_Hz, table.our_Z_abs_ohm, "--", label="reproduction")
        ax.grid(True, which="both", alpha=0.3); ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("|Z| [ohm]"); ax.legend(); fig.tight_layout(); fig.savefig(out / "impedance_comparison.png", dpi=180); plt.close(fig)
    except Exception:
        pass
    return metrics


def compare_single_solution(model, solution, req5_raw: str | Path, outdir: str | Path, req6_raw: str | Path | None = None) -> dict:
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    raw = Path(req5_raw)
    f = float(solution.freq_Hz)
    summary = {"freq_Hz": f}

    # L5 boundary motion.
    l5 = pd.read_csv(raw / "layer05_solid_asb_boundary_fields.csv")
    gf = l5[np.isclose(l5.freq_Hz.astype(float), f)].sort_values(["boundary_id", "node_id"]).reset_index(drop=True)
    if len(gf):
        sampler = P2BoundarySampler(model.solid, gf)
        ur, uz, un = sampler.sample(solution.solid_displacement)
        ref_un = cplx(gf, "u_n_real", "u_n_imag") * 1e-3
        m = complex_metrics(un, ref_un, "L5_un_")
        summary.update(m)
        pt = gf[["boundary_id", "node_id", "s_arc_m", "r_m", "z_m"]].copy()
        pt["our_un_real_m"] = un.real; pt["our_un_imag_m"] = un.imag
        pt["COMSOL_un_real_m"] = ref_un.real; pt["COMSOL_un_imag_m"] = ref_un.imag
        pt.to_csv(out / f"L5_boundary_motion_{f:g}Hz.csv", index=False)

    # L8 Boundary93 source field using nearest recovered quadrature samples.
    l8 = pd.read_csv(raw / "layer08_boundary93_source_points.csv")
    b = l8[np.isclose(l8.freq_Hz.astype(float), f)].sort_values("node_id").reset_index(drop=True)
    if len(b):
        ext_cfg = model.config.get("exterior", {})
        if hasattr(model.acoustic_operator, "boundary_samples"):
            samples, _ = model.acoustic_operator.boundary_samples(
                solution.pressure_mixed,
                boundary_id=int(ext_cfg.get("boundary_id", 93)),
                intorder=4,
                force_radial_normals=bool(ext_cfg.get("force_radial_normals", True)),
            )
        else:
            samples, _ = boundary93_hk_samples_recovered(
                model.acoustic_model, solution.pressure_base,
                recovery_method="ppr" if str(ext_cfg.get("recovery_method", "ppr")).startswith("ppr") else "zz",
                force_radial_normals=bool(ext_cfg.get("force_radial_normals", True)),
            )
        rs, zs, nr, nz, _, ps, qs = samples
        # Interpolate along the meridian arc; nearest quadrature-point matching can
        # introduce a millimetre-scale coordinate mismatch on the coarse mesh.
        radius = float(model.config.get("pml", {}).get("inner_radius_m", np.mean(np.hypot(rs, zs))))
        ss = np.arctan2(zs, rs) * radius
        order = np.argsort(ss); ss = ss[order]; ps = ps[order]; qs = qs[order]; nr = nr[order]; nz = nz[order]
        sb = b["s_arc_m"].to_numpy(float)
        pp = np.interp(sb, ss, ps.real) + 1j * np.interp(sb, ss, ps.imag)
        qq = np.interp(sb, ss, qs.real) + 1j * np.interp(sb, ss, qs.imag)
        pref = cplx(b, "p_real_Pa", "p_imag_Pa")
        summary.update(complex_metrics(pp, pref, "L8_p_"))
        q6_path = Path(req6_raw) / "req6_boundary93_gradient_recovery_audit.csv" if req6_raw else None
        if q6_path is not None and q6_path.exists():
            q6 = pd.read_csv(q6_path)
            qrow = q6[np.isclose(q6.solved_freq_Hz.astype(float), f)].sort_values("node_id")
            if len(qrow):
                qref = cplx(qrow, "dpdn_ppr_real", "dpdn_ppr_imag")
                # COMSOL Boundary93 normal is opposite the physical->PML normal used by Python.
                qref = -qref
                if len(qref) == len(qq):
                    summary.update(complex_metrics(qq, qref, "L8_q_PPR_"))
                    summary["L8_q_reference_status"] = "DIRECT_REQ6_PPR"
                else:
                    summary["L8_q_reference_status"] = "REQ6_POINT_COUNT_MISMATCH"
            else:
                summary["L8_q_reference_status"] = "REQ6_FREQUENCY_MISSING"
        else:
            summary["L8_q_reference_status"] = "BLOCKED_REQUIRES_REQ6_PPR_GRADIENT"
            summary["L8_q_note"] = "REQ5 dp_dn is the ordinary gradient and must not be used as a reference for COMSOL EFC UsePPR=1."

    # L9 directivity.
    l9 = pd.read_csv(raw / "layer09_farfield_directivity_matrix.csv")
    d = l9[np.isclose(l9.freq_Hz.astype(float), f)].sort_values("theta_deg").reset_index(drop=True)
    if len(d):
        our = np.interp(d.theta_deg, solution.directivity_angles_deg, solution.directivity_relative_dB)
        
        if "relative_dB" in d.columns:
            ref = d.relative_dB.to_numpy(float)
        elif "COMSOL_relative_dB" in d.columns:
            ref = d.COMSOL_relative_dB.to_numpy(float)
        elif "SPL_relative_to_0deg_dB" in d.columns:
            ref = d.SPL_relative_to_0deg_dB.to_numpy(float)
        else:
            raise KeyError("layer09 needs a relative directivity column")
        err = our - ref
        summary.update({
            "L9_relative_RMSE_dB": float(np.sqrt(np.mean(err**2))),
            "L9_relative_max_abs_error_dB": float(np.max(np.abs(err))),
        })
        # Absolute complex pext comparison, which exposes normalization and phasor errors.
        if {"pext_real_Pa", "pext_imag_Pa"}.issubset(d.columns):
            pref_complex = cplx(d, "pext_real_Pa", "pext_imag_Pa")
            pcur = np.interp(d.theta_deg, solution.directivity_angles_deg, solution.directivity_pressure_Pa_peak.real) + 1j * np.interp(d.theta_deg, solution.directivity_angles_deg, solution.directivity_pressure_Pa_peak.imag)
            amp_err = 20 * np.log10(np.maximum(np.abs(pcur), 1e-300) / np.maximum(np.abs(pref_complex), 1e-300))
            phase_err = np.angle(pcur / np.where(np.abs(pref_complex) > 0, pref_complex, 1), deg=True)
            i0 = int(np.argmin(np.abs(d.theta_deg.to_numpy(float))))
            rel_ref = 20 * np.log10(np.maximum(np.abs(pref_complex) / max(abs(pref_complex[i0]), 1e-300), 1e-300))
            main = rel_ref >= -20.0
            summary.update({
                "L9_complex_NRMSE": float(np.linalg.norm(pcur - pref_complex) / max(np.linalg.norm(pref_complex), 1e-300)),
                "L9_amplitude_RMSE_dB": float(np.sqrt(np.mean(amp_err**2))),
                "L9_phase_RMSE_deg": float(np.sqrt(np.mean(phase_err**2))),
                "L9_phase_main_ge_minus20dB_RMSE_deg": float(np.sqrt(np.mean(phase_err[main]**2))),
                "L9_axis_amplitude_error_dB": float(amp_err[i0]),
                "L9_axis_phase_error_deg": float(phase_err[i0]),
            })
        pd.DataFrame({"theta_deg": d.theta_deg, "our_relative_dB": our, "COMSOL_relative_dB": ref, "error_dB": err}).to_csv(out / f"L9_directivity_{f:g}Hz.csv", index=False)

    (out / f"comparison_summary_{f:g}Hz.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
