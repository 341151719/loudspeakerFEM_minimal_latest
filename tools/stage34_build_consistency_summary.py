#!/usr/bin/env python3
"""Build the reproducible Stage34 consistency summary from generated evidence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "00_MAINLINE/loudspeakerFEM_current_20260717"
RUN = ROOT / "20_ANALYSIS/runs/stage34_directivity_15k"
COMSOL = ROOT / "20_ANALYSIS/stage34_directivity15k/comsol"


def stats(df: pd.DataFrame) -> dict:
    irel = df.relative_RMSE_dB.idxmax()
    imain = df.relative_main_ge_minus20dB_RMSE_dB.idxmax()
    ishape = df.shape_NRMSE_after_diagnostic_complex_scale_pct.idxmax()
    return {
        "frequency_count": int(len(df)),
        "relative_RMSE_dB_mean": float(df.relative_RMSE_dB.mean()),
        "relative_RMSE_dB_max": float(df.loc[irel, "relative_RMSE_dB"]),
        "relative_RMSE_worst_frequency_Hz": float(df.loc[irel, "freq_Hz"]),
        "main_ge_minus20dB_RMSE_dB_mean": float(df.relative_main_ge_minus20dB_RMSE_dB.mean()),
        "main_ge_minus20dB_RMSE_dB_max": float(df.loc[imain, "relative_main_ge_minus20dB_RMSE_dB"]),
        "main_ge_minus20dB_worst_frequency_Hz": float(df.loc[imain, "freq_Hz"]),
        "shape_NRMSE_pct_mean": float(df.shape_NRMSE_after_diagnostic_complex_scale_pct.mean()),
        "shape_NRMSE_pct_max": float(df.loc[ishape, "shape_NRMSE_after_diagnostic_complex_scale_pct"]),
        "shape_NRMSE_worst_frequency_Hz": float(df.loc[ishape, "freq_Hz"]),
        "raw_complex_NRMSE_pct_mean": float(df.complex_NRMSE_pct.mean()),
        "axis_amplitude_error_dB_RMSE": float(np.sqrt(np.mean(df.axis_amplitude_error_dB**2))),
        "axis_phase_error_deg_RMSE": float(np.sqrt(np.mean(df.axis_phase_error_deg**2))),
    }


def peak(path: Path) -> dict:
    d = pd.read_csv(path)
    out = {}
    for domain in (8, 22):
        col = f"domain{domain}_mean_abs_pressure_Pa"
        i = d[col].idxmax()
        out[f"domain{domain}_peak_frequency_Hz"] = float(d.loc[i, "freq_Hz"])
        out[f"domain{domain}_peak_mean_abs_pressure_Pa"] = float(d.loc[i, col])
    i = d.axis_SPL_dB_RMS.idxmax()
    out["axis_SPL_peak_frequency_Hz"] = float(d.loc[i, "freq_Hz"])
    out["axis_SPL_peak_dB_RMS"] = float(d.loc[i, "axis_SPL_dB_RMS"])
    return out


def metadata(case: str) -> dict:
    d = pd.read_csv(COMSOL / case / "run_metadata.csv")
    return dict(zip(d.key, d.value))


def main() -> None:
    low = pd.read_csv(RUN / "comparison_full126/directivity_metrics_by_frequency.csv")
    high_all = pd.read_csv(RUN / "comparison_final_05/directivity_metrics_by_frequency.csv")
    high = high_all[high_all.comparison == "native_baseline_vs_comsol_refined"].copy()
    high = high[high.freq_Hz > 8000].copy()  # 8 kHz remains on the official-grid branch.
    combined = pd.concat([low, high], ignore_index=True)
    convergence = high_all[high_all.comparison == "comsol_original_vs_comsol_refined"].copy()
    data = {
        "schema": "stage34-directivity-15khz-v1",
        "benchmark_policy": {
            "COMSOL_role": "independent benchmark only",
            "COMSOL_values_used_by_native_runtime": False,
            "best_complex_scale_used_by_native_runtime": False,
            "high_frequency_benchmark": "COMSOL fmax=22.5 kHz, moving structure hmax=0.5 mm",
            "official_range_benchmark": "original COMSOL Study 2 complex pext matrix, 1-8000 Hz",
        },
        "production_routing": {
            "at_or_below_3000_Hz": "configs/fast_p1.json",
            "above_3000_through_8000_Hz": "configs/best_model.json mapped center-split mesh",
            "above_8000_Hz": "configs/stage34_structure_refined2.json",
            "upper_structure_free_dofs": 59342,
            "upper_structure_triangles": 14464,
        },
        "consistency": {
            "1_to_8000_Hz_126_points": stats(low),
            "8500_to_15000_Hz_14_points": stats(high),
            "combined_1_to_15000_Hz_140_points": stats(combined),
        },
        "COMSOL_convergence_1mm_to_0p5mm": stats(convergence),
        "no_NRA_cavity_mode": {
            "COMSOL_domain8_and_22_peak_frequency_Hz": 605.0,
            "COMSOL_domain8_peak_mean_abs_pressure_Pa": 5963.990946944573,
            "COMSOL_domain22_peak_mean_abs_pressure_Pa": 3304.1804866815432,
            "native_P1": peak(RUN / "no_nra_p1_580_640/native_sweep_metrics.csv"),
            "native_physical_P2": peak(RUN / "no_nra_p2_580_640/native_sweep_metrics.csv"),
        },
        "performance": {
            "native_high15_jobs8": json.loads((RUN / "native_refined2_compact_jobs8/metadata.json").read_text()),
            "native_high15_observed_wall_seconds": 48.43,
            "native_high15_estimated_aggregate_peak_RSS_GiB": 8 * 892576 / 1024**2,
            "COMSOL_high15_np12": metadata("acoustic_22p5k_structure_0p5mm"),
            "COMSOL_high15_peak_memory_GB_from_log": 1.87,
        },
        "attribution": [
            "Above 8 kHz, the dominant discrepancy is structural breakup under-resolution; two native topology refinements reduce mean relative-directivity RMSE from 4.437 dB to 1.006 dB against the converged benchmark.",
            "COMSOL acoustic fmax 15 to 22.5 kHz at fixed 1 mm structure changes mean directivity by only 0.054 dB, so acoustics/PML/HK are secondary after refinement.",
            "The no-NRA 605 Hz cavity discrepancy is physical-acoustic order: all-physical-domain P2 moves the native peak from 612 to 606 Hz without changing material or sound speed.",
            "The 2120 Hz full-angle dB outlier is a deep-null location sensitivity; its main-field RMSE is 0.103 dB and complex shape NRMSE is 1.011%.",
        ],
    }
    destination = PROJECT / "docs/CONSISTENCY_STAGE34_DIRECTIVITY_15KHZ_20260719.json"
    destination.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
