#!/usr/bin/env python3
"""Compare the 0.5x/1x/2x cone density and stiffness SUM RULES scans."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_SCAN = ROOT / "runs" / "sum_rules_parameter_scan"
OUT = ROOT / "sum_rules" / "results" / "parameter_scan"
ARCHIVED_SCAN = OUT / "raw"
BANDS_HZ = [(1.0, 20.0), (20.0, 200.0), (200.0, 2000.0), (2000.0, 15000.0)]


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def integrate_between(frame: pd.DataFrame, column: str, lo: float, hi: float, power: int = 0) -> float:
    frame = frame.sort_values("freq_Hz")
    f = frame["freq_Hz"].to_numpy(float)
    omega = frame["omega_rad_s"].to_numpy(float)
    y = frame[column].to_numpy(float)
    selected = (f > lo) & (f < hi)
    fq = np.r_[lo, f[selected], hi]
    wq = 2.0 * np.pi * fq
    yq = np.interp(wq, omega, y)
    return 2.0 / np.pi * trapz(yq / wq**power, wq)


def spectral_quantile_hz(frame: pd.DataFrame, quantile: float) -> float:
    frame = frame.sort_values("omega_rad_s")
    omega = frame["omega_rad_s"].to_numpy(float)
    resistance = frame["Zmot_real_ohm"].to_numpy(float)
    increments = 0.5 * (resistance[:-1] + resistance[1:]) * np.diff(omega)
    cumulative = np.r_[0.0, np.cumsum(increments)]
    target = quantile * cumulative[-1]
    return float(np.interp(target, cumulative, frame["freq_Hz"].to_numpy(float)))


def load_cases() -> dict[str, tuple[pd.DataFrame, dict, dict]]:
    scan = RUN_SCAN if RUN_SCAN.exists() else ARCHIVED_SCAN
    baseline_path = ROOT / "sum_rules" / "results" / "main_321"
    adaptive_path = ROOT / "sum_rules" / "results" / "adaptive_387"
    base = pd.read_csv(adaptive_path / "sum_rule_frequency_audit_adaptive.csv")
    base_summary = json.loads((baseline_path / "sum_rule_summary.json").read_text(encoding="utf-8"))
    base_refinement = json.loads((adaptive_path / "summary.json").read_text(encoding="utf-8"))
    cases = {"baseline_1p0": (base, base_summary, base_refinement)}
    for name in ("cone_density_0p5", "cone_density_2p0", "cone_stiffness_0p5", "cone_stiffness_2p0"):
        path = scan / name
        cases[name] = (
            pd.read_csv(path / "merged" / "sum_rule_frequency_audit_adaptive.csv"),
            json.loads((path / "sum_rule_summary.json").read_text(encoding="utf-8")),
            json.loads((path / "merged" / "summary.json").read_text(encoding="utf-8")),
        )
    return cases


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    baseline_m0 = integrate_between(cases["baseline_1p0"][0], "Zmot_real_ohm", 1.0, 15000.0)
    rows = []
    bands = []
    for name, (frame, summary, refinement) in cases.items():
        endpoint = summary["endpoints"]
        m0 = integrate_between(frame, "Zmot_real_ohm", 1.0, 15000.0)
        m2 = integrate_between(frame, "Zmot_real_ohm", 1.0, 15000.0, power=2)
        useful = integrate_between(frame, "useful_efficiency_budget_ohm", 1.0, 15000.0)
        internal = integrate_between(frame, "internal_loss_penalty_budget_ohm", 1.0, 15000.0)
        rows.append({
            "case": name,
            "Ainf_ohm_per_s": endpoint["A_inf_matrix_ohm_per_s"],
            "A0_ohm_s": endpoint["A0_structure_only_ohm_s"],
            "M0_1_15k_ohm_per_s": m0,
            "M0_relative_to_baseline": m0 / baseline_m0,
            "M0_fraction_of_Ainf": m0 / endpoint["A_inf_matrix_ohm_per_s"],
            "M2_1_15k_ohm_s": m2,
            "M2_fraction_of_A0": m2 / endpoint["A0_structure_only_ohm_s"],
            "useful_M0_ohm_per_s": useful,
            "internal_M0_ohm_per_s": internal,
            "useful_fraction_of_M0": useful / m0,
            "M0_q25_Hz": spectral_quantile_hz(frame, 0.25),
            "M0_q50_Hz": spectral_quantile_hz(frame, 0.50),
            "M0_q75_Hz": spectral_quantile_hz(frame, 0.75),
            "minimum_Re_Zmot_ohm": summary["checks"]["minimum_Re_Zmot_ohm"],
            "minimum_Rinternal_ohm": summary["checks"]["minimum_Rinternal_far_ohm"],
            "passivity_pass": summary["checks"]["passivity_on_sampled_real_axis"],
            "internal_loss_pass": summary["checks"]["nonnegative_internal_loss_on_sampled_real_axis"],
            "base_grid_points": refinement["base_frequencies"],
            "adaptive_points": refinement["adaptive_frequencies"],
            "adaptive_M0_relative_correction": refinement["adaptive_minus_base_moment_0_relative"],
            "posterior_M0_indicator_fraction": refinement["posterior_indicator_fraction_of_moment_0"],
        })
        for lo, hi in BANDS_HZ:
            value = integrate_between(frame, "Zmot_real_ohm", lo, hi)
            bands.append({"case": name, "band_lo_Hz": lo, "band_hi_Hz": hi,
                          "M0_ohm_per_s": value, "fraction_of_case_M0": value / m0})

    summary_frame = pd.DataFrame(rows)
    band_frame = pd.DataFrame(bands)
    summary_frame.to_csv(OUT / "parameter_scan_summary.csv", index=False)
    band_frame.to_csv(OUT / "local_band_M0.csv", index=False)
    cases["baseline_1p0"][0].to_csv(OUT / "baseline_387_frequency_audit.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    for name, (frame, _, _) in cases.items():
        frame = frame.sort_values("freq_Hz")
        axes[0].semilogx(frame["freq_Hz"], frame["Zmot_real_ohm"], label=name)
        axes[1].semilogx(frame["freq_Hz"], frame["cumulative_moment_0"] / frame["cumulative_moment_0"].iloc[-1], label=name)
    axes[0].set_ylabel("Re Zmot [ohm]")
    axes[1].set_ylabel("Normalized cumulative M0")
    axes[1].set_xlabel("Frequency [Hz]")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "parameter_scan_spectra.png", dpi=180)
    plt.close(fig)

    payload = {
        "frequency_grid": "1 Hz to 15 kHz; baseline 321+66 points, variants 81 plus targeted adaptive points",
        "varied_domain": 21,
        "bands_Hz": BANDS_HZ,
        "cases": summary_frame.to_dict("records"),
    }
    (OUT / "parameter_scan_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary_frame.to_string(index=False))
    print("\nLocal bands:\n", band_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
