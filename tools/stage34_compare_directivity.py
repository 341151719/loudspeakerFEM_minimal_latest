#!/usr/bin/env python3
"""Compare native and COMSOL complex directivity without fitting the solver.

The optional best complex scale is reported only as a shape diagnostic.  It is
never written to configuration and never applied to native results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _complex(df: pd.DataFrame) -> np.ndarray:
    pairs = [
        ("p_real_Pa", "p_imag_Pa"),
        ("pext_real_Pa", "pext_imag_Pa"),
    ]
    for re, im in pairs:
        if re in df and im in df:
            return df[re].to_numpy(float) + 1j * df[im].to_numpy(float)
    raise KeyError("complex pressure columns not found")


def _angle_col(df: pd.DataFrame) -> str:
    for name in ("theta_deg", "angle_deg"):
        if name in df:
            return name
    raise KeyError("angle column not found")


def _freq_col(df: pd.DataFrame) -> str:
    for name in ("freq_Hz", "f_Hz"):
        if name in df:
            return name
    raise KeyError("frequency column not found")


def load_native(checkpoint_root: Path) -> pd.DataFrame:
    compact = checkpoint_root / "native_directivity_complex.csv"
    if compact.exists():
        return pd.read_csv(compact)
    frames = []
    for path in sorted(checkpoint_root.glob("*Hz/directivity_*Hz.csv")):
        frame = pd.read_csv(path)
        if "freq_Hz" not in frame:
            name = path.parent.name.removesuffix("Hz")
            frame.insert(0, "freq_Hz", float(name))
        frames.append(frame)
    if not frames:
        # Also support one solve directory rather than a checkpoint tree.
        for path in sorted(checkpoint_root.glob("directivity_*Hz.csv")):
            frame = pd.read_csv(path)
            if "freq_Hz" not in frame:
                name = path.stem.removeprefix("directivity_").removesuffix("Hz")
                frame.insert(0, "freq_Hz", float(name))
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no native directivity CSV below {checkpoint_root}")
    return pd.concat(frames, ignore_index=True)


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _interp_complex(angle_source, pressure_source, angle_target):
    order = np.argsort(angle_source)
    a = np.asarray(angle_source)[order]
    p = np.asarray(pressure_source)[order]
    return np.interp(angle_target, a, p.real) + 1j * np.interp(angle_target, a, p.imag)


def metrics(a_deg: np.ndarray, test: np.ndarray, ref: np.ndarray) -> dict:
    i0 = int(np.argmin(np.abs(a_deg)))
    rt = 20 * np.log10(np.maximum(np.abs(test) / max(abs(test[i0]), 1e-300), 1e-300))
    rr = 20 * np.log10(np.maximum(np.abs(ref) / max(abs(ref[i0]), 1e-300), 1e-300))
    relerr = rt - rr
    scale = np.vdot(test, ref) / max(float(np.vdot(test, test).real), 1e-300)
    raw = np.linalg.norm(test - ref) / max(np.linalg.norm(ref), 1e-300)
    shaped = np.linalg.norm(scale * test - ref) / max(np.linalg.norm(ref), 1e-300)
    amp_axis = 20 * np.log10(max(abs(test[i0]), 1e-300) / max(abs(ref[i0]), 1e-300))
    phase_axis = np.angle(test[i0] / ref[i0], deg=True)
    main = rr >= -20.0
    worst = int(np.argmax(np.abs(relerr)))
    out = {
        "complex_NRMSE_pct": 100 * float(raw),
        "shape_NRMSE_after_diagnostic_complex_scale_pct": 100 * float(shaped),
        "diagnostic_scale_abs": float(abs(scale)),
        "diagnostic_scale_phase_deg": float(np.angle(scale, deg=True)),
        "relative_RMSE_dB": float(np.sqrt(np.mean(relerr**2))),
        "relative_main_ge_minus20dB_RMSE_dB": float(np.sqrt(np.mean(relerr[main] ** 2))),
        "relative_max_abs_error_dB": float(np.max(np.abs(relerr))),
        "worst_angle_deg": float(a_deg[worst]),
        "worst_angle_error_dB": float(relerr[worst]),
        "axis_amplitude_error_dB": float(amp_axis),
        "axis_phase_error_deg": float(phase_axis),
    }
    aa = np.abs(a_deg)
    for lo, hi in ((0, 30), (30, 60), (60, 91)):
        mask = (aa >= lo) & (aa < hi)
        out[f"relative_RMSE_absangle_{lo}_{hi}_dB"] = float(np.sqrt(np.mean(relerr[mask] ** 2)))
    return out


def compare(test_df: pd.DataFrame, ref_df: pd.DataFrame, label: str) -> pd.DataFrame:
    tf, rf = _freq_col(test_df), _freq_col(ref_df)
    ta, ra = _angle_col(test_df), _angle_col(ref_df)
    common = sorted(set(np.round(test_df[tf], 8)) & set(np.round(ref_df[rf], 8)))
    rows = []
    for freq in common:
        t = test_df[np.isclose(test_df[tf], freq)].sort_values(ta)
        r = ref_df[np.isclose(ref_df[rf], freq)].sort_values(ra)
        angles = r[ra].to_numpy(float)
        pt = _interp_complex(t[ta].to_numpy(float), _complex(t), angles)
        pr = _complex(r)
        row = {"comparison": label, "freq_Hz": float(freq), **metrics(angles, pt, pr)}
        rows.append(row)
    return pd.DataFrame(rows)


def summary(table: pd.DataFrame) -> dict:
    result = {}
    for label, group in table.groupby("comparison"):
        result[label] = {
            "frequency_count": int(len(group)),
            "relative_RMSE_dB_mean": float(group.relative_RMSE_dB.mean()),
            "relative_RMSE_dB_max": float(group.relative_RMSE_dB.max()),
            "relative_RMSE_worst_freq_Hz": float(group.loc[group.relative_RMSE_dB.idxmax(), "freq_Hz"]),
            "shape_NRMSE_pct_mean": float(group.shape_NRMSE_after_diagnostic_complex_scale_pct.mean()),
            "shape_NRMSE_pct_max": float(group.shape_NRMSE_after_diagnostic_complex_scale_pct.max()),
            "raw_complex_NRMSE_pct_mean": float(group.complex_NRMSE_pct.mean()),
            "axis_amplitude_error_dB_RMSE": float(np.sqrt(np.mean(group.axis_amplitude_error_dB**2))),
            "axis_phase_error_deg_RMSE": float(np.sqrt(np.mean(group.axis_phase_error_deg**2))),
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", type=Path, required=True, help="native checkpoint root")
    ap.add_argument("--comsol", type=Path, required=True, help="COMSOL directivity_complex.csv")
    ap.add_argument("--native-refined", type=Path)
    ap.add_argument("--comsol-original", type=Path)
    ap.add_argument("--comsol-acoustic-only", type=Path)
    ap.add_argument("--comsol-structure-only", type=Path)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    native = load_native(args.native)
    comsol = load_table(args.comsol)
    tables = [compare(native, comsol, "native_baseline_vs_comsol_refined")]
    if args.native_refined:
        nref = load_native(args.native_refined)
        tables.append(compare(nref, comsol, "native_refined_vs_comsol_refined"))
        tables.append(compare(nref, native, "native_refined_vs_native_baseline"))
    if args.comsol_original:
        corig = load_table(args.comsol_original)
        tables.append(compare(corig, comsol, "comsol_original_vs_comsol_refined"))
    if args.comsol_acoustic_only:
        cac = load_table(args.comsol_acoustic_only)
        tables.append(compare(cac, comsol, "comsol_acoustic_only_vs_comsol_refined"))
    if args.comsol_structure_only:
        cst = load_table(args.comsol_structure_only)
        tables.append(compare(cst, comsol, "comsol_structure_only_vs_comsol_refined"))
    table = pd.concat(tables, ignore_index=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.outdir / "directivity_metrics_by_frequency.csv", index=False)
    data = summary(table)
    (args.outdir / "directivity_summary.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
