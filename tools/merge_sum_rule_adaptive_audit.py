#!/usr/bin/env python3
"""Merge an adaptive sum-rule refinement into the complete base frequency grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validate_sum_rules_fem import integrate_moments


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="runs/sum_rules_highest_accuracy/sum_rule_frequency_audit.csv")
    parser.add_argument("--adaptive", default="runs/sum_rules_adaptive_round1/sum_rule_frequency_audit.csv")
    parser.add_argument("--selection", default="runs/sum_rules_adaptive_round1/frequencies.json")
    parser.add_argument("--outdir", default="runs/sum_rules_adaptive_merged")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = pd.read_csv(root / args.base)
    adaptive = pd.read_csv(root / args.adaptive)
    selection = json.loads((root / args.selection).read_text(encoding="utf-8"))
    combined = pd.concat([base, adaptive], ignore_index=True).sort_values("freq_Hz")
    combined = combined.drop_duplicates("freq_Hz", keep="last").reset_index(drop=True)
    combined, moments = integrate_moments(combined)
    _, base_moments = integrate_moments(base)

    selected_old_abs = 0.0
    selected_new_abs = 0.0
    selected_net_change = 0.0
    for interval in selection["selected_intervals"]:
        lo, hi = interval["left_Hz"], interval["right_Hz"]
        old = base[(base.freq_Hz >= lo * (1 - 1e-12)) & (base.freq_Hz <= hi * (1 + 1e-12))].sort_values("omega_rad_s")
        new = combined[(combined.freq_Hz >= lo * (1 - 1e-12)) & (combined.freq_Hz <= hi * (1 + 1e-12))].sort_values("omega_rad_s")
        old_area = 2 / np.pi * np.trapezoid(old.Zmot_real_ohm, old.omega_rad_s)
        new_area = 2 / np.pi * np.trapezoid(new.Zmot_real_ohm, new.omega_rad_s)
        selected_old_abs += float(interval["absolute_error_indicator"])
        selected_new_abs += abs(new_area - old_area)
        selected_net_change += new_area - old_area
    unselected_old_abs = selection["total_absolute_error_indicator_ohm_per_s"] - selected_old_abs
    posterior_indicator = selected_new_abs + unselected_old_abs
    result = {
        "base_frequencies": int(len(base)),
        "adaptive_frequencies": int(len(adaptive)),
        "combined_frequencies": int(len(combined)),
        "base_moments": base_moments,
        "adaptive_moments": moments,
        "adaptive_minus_base_moment_0_ohm_per_s": moments["window_moment_0_ohm_per_s"] - base_moments["window_moment_0_ohm_per_s"],
        "adaptive_minus_base_moment_0_relative": (moments["window_moment_0_ohm_per_s"] - base_moments["window_moment_0_ohm_per_s"]) / moments["window_moment_0_ohm_per_s"],
        "posterior_absolute_local_error_indicator_ohm_per_s": posterior_indicator,
        "posterior_indicator_fraction_of_moment_0": posterior_indicator / moments["window_moment_0_ohm_per_s"],
        "selected_interval_net_change_check_ohm_per_s": selected_net_change,
    }
    out = root / args.outdir
    out.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out / "sum_rule_frequency_audit_adaptive.csv", index=False)
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.loglog(base.freq_Hz, base.Zmot_real_ohm, color="0.65", lw=1, label="321-point base")
    ax.scatter(adaptive.freq_Hz, adaptive.Zmot_real_ohm, s=12, color="tab:red", label="66 adaptive points")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Re Zmot [ohm]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "adaptive_sampling.png", dpi=180)
    plt.close(fig)

    report = f"""# Sum rules 自适应频率积分

- 基础嵌套网格：{len(base)} 点；自适应新增：{len(adaptive)} 点；合并：{len(combined)} 点。
- 321 点 M0：{base_moments['window_moment_0_ohm_per_s']:.9g} ohm/s。
- 合并自适应 M0：{moments['window_moment_0_ohm_per_s']:.9g} ohm/s，相对变化 {100*result['adaptive_minus_base_moment_0_relative']:.4g}%。
- 合并自适应 M2：{moments['window_moment_2_ohm_s']:.9g} ohm s。
- 后验绝对局部误差指标：{posterior_indicator:.9g} ohm/s，占 M0 的 {100*result['posterior_indicator_fraction_of_moment_0']:.4g}%。

该后验指标是相邻两层复合梯形积分的局部差之和，用于数值收敛判读，不是严格数学误差上界。
"""
    (out / "VALIDATION_REPORT_CN.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
