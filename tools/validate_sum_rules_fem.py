#!/usr/bin/env python3
"""Fixed-profile FEM audit of the motional-impedance sum rules.

This is deliberately separate from the production hybrid sweep.  A Herglotz
sum rule concerns one transfer function, so changing meshes/configurations in
the middle of a frequency sweep would invalidate the mathematical test.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "best_model"))

from coupled_solver import build_best_model, solve_frequency  # noqa: E402
from loudspeaker_axisym_fem.exterior_field import (  # noqa: E402
    halfspace_power_from_directivity,
    intensity_power_from_samples,
)
from loudspeaker_axisym_fem.stage4F_hk_refinement import boundary93_hk_samples_recovered  # noqa: E402
from p2_axisym_solid import complex_stiffness  # noqa: E402

_WORKER_MODEL = None
_WORKER_NRA = True
_WORKER_LIMITER = None


def worker_init(config_path: str, magnetic_path: str, nra_enabled: bool, blas_threads: int) -> None:
    global _WORKER_MODEL, _WORKER_NRA, _WORKER_LIMITER
    from threadpoolctl import threadpool_limits
    _WORKER_LIMITER = threadpool_limits(limits=int(blas_threads))
    _WORKER_MODEL = build_best_model(ROOT, config_path=config_path, magnetostatic_vtu=magnetic_path)
    _WORKER_NRA = bool(nra_enabled)


def worker_solve(frequency: float) -> dict[str, float]:
    start = time.perf_counter()
    solution = solve_frequency(
        _WORKER_MODEL, frequency, drive="current", current_A_peak=1.0 + 0j,
        nra_enabled=_WORKER_NRA,
    )
    return solution_row(_WORKER_MODEL, solution, time.perf_counter() - start)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_frequencies(spec: str) -> np.ndarray:
    if spec.startswith("log:"):
        _, lo, hi, count = spec.split(":")
        return np.geomspace(float(lo), float(hi), int(count))
    path = Path(spec)
    if path.is_file():
        frame = pd.read_csv(path)
        column = next(c for c in ("freq_Hz", "f_Hz", "frequency_Hz") if c in frame)
        return np.asarray(frame[column], float)
    return np.asarray([float(item) for item in spec.split(",")], float)


def endpoint_budgets(model) -> dict[str, float]:
    free = np.asarray(model.solid.free_dofs, int)
    g = np.asarray(model.lorentz_per_A[free], complex)
    mass = model.solid.M[free][:, free].tocsc()
    stiffness = model.solid.K_real[free][:, free].tocsc()
    a_inf = float(np.real(np.vdot(g, spsolve(mass, g))))
    a0_structure = float(np.real(np.vdot(g, spsolve(stiffness, g))))
    return {
        "A_inf_matrix_ohm_per_s": a_inf,
        "A0_structure_only_ohm_s": a0_structure,
        "effective_inertial_mass_for_axial_BL_kg": (
            float(model.lorentz_info["axial_BL_N_per_A"]) ** 2 / a_inf
        ),
        "warning": (
            "A0_structure_only excludes the zero-frequency sealed-air constraint of the pressure formulation; "
            "use the numerical low-frequency limit for the coupled-system endpoint."
        ),
    }


def boundary_flux_power(model, solution) -> float:
    ext = model.config["exterior"]
    if hasattr(model.acoustic_operator, "boundary_samples"):
        samples, _ = model.acoustic_operator.boundary_samples(
            solution.pressure_mixed,
            boundary_id=int(ext.get("boundary_id", 93)),
            intorder=4,
            force_radial_normals=bool(ext.get("force_radial_normals", True)),
        )
    else:
        samples, _ = boundary93_hk_samples_recovered(
            model.acoustic_model,
            solution.pressure_base,
            recovery_method="ppr" if str(ext.get("recovery_method", "ppr")).startswith("ppr") else "zz",
            force_radial_normals=bool(ext.get("force_radial_normals", True)),
        )
    return intensity_power_from_samples(
        solution.freq_Hz,
        model.config["air"]["rho0_kg_m3"],
        samples[0], samples[4], samples[5], samples[6],
    )


def solution_row(model, solution, elapsed_s: float) -> dict[str, float]:
    angles = np.asarray(solution.directivity_angles_deg, float)
    keep = angles >= -1e-12
    theta = np.deg2rad(angles[keep])
    pressure = np.asarray(solution.directivity_pressure_Pa_peak)[keep]
    order = np.argsort(theta)
    p_far = halfspace_power_from_directivity(
        pressure[order], theta[order],
        model.config["air"]["rho0_kg_m3"],
        model.config["air"]["c0_m_s"],
        model.config["exterior"]["observation_radius_m"],
    )
    p_flux = boundary_flux_power(model, solution)
    current2 = abs(solution.current_A_peak) ** 2
    r_ac_far = 2.0 * p_far / max(current2, 1e-300)
    r_ac_flux = 2.0 * p_flux / max(current2, 1e-300)
    r_mot = float(solution.motional_impedance_ohm.real)
    r_coil = float(solution.blocked_impedance_ohm.real)
    r_internal = r_mot - r_ac_far
    total_active = r_coil + r_mot
    eta = r_ac_far / total_active if total_active > 0 else float("nan")
    odds = eta / (1.0 - eta) if 0 <= eta < 1 else float("nan")
    odds_residual = r_mot - (r_coil * odds + r_internal / (1.0 - eta)) if 0 <= eta < 1 else float("nan")
    omega = 2.0 * math.pi * solution.freq_Hz
    return {
        "freq_Hz": solution.freq_Hz,
        "omega_rad_s": omega,
        "Zmot_real_ohm": r_mot,
        "Zmot_imag_ohm": solution.motional_impedance_ohm.imag,
        "Zblocked_real_ohm": r_coil,
        "Zblocked_imag_ohm": solution.blocked_impedance_ohm.imag,
        "radiated_power_far_W": p_far,
        "radiated_power_boundary_flux_W": p_flux,
        "Rac_far_ohm": r_ac_far,
        "Rac_boundary_flux_ohm": r_ac_flux,
        "Rinternal_far_ohm": r_internal,
        "eta_far": eta,
        "efficiency_odds_far": odds,
        "odds_identity_residual_ohm": odds_residual,
        "A0_estimate_real_ohm_s": (solution.motional_impedance_ohm / (1j * omega)).real,
        "A0_estimate_imag_ohm_s": (solution.motional_impedance_ohm / (1j * omega)).imag,
        "Ainf_estimate_real_ohm_per_s": (-1j * omega * solution.motional_impedance_ohm).real,
        "Ainf_estimate_imag_ohm_per_s": (-1j * omega * solution.motional_impedance_ohm).imag,
        "axis_SPL_dB": solution.axis_SPL_dB,
        "solve_seconds": elapsed_s,
    }


def integrate_moments(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    data = frame.sort_values("omega_rad_s").copy()
    omega = data["omega_rad_s"].to_numpy(float)
    resistance = data["Zmot_real_ohm"].to_numpy(float)
    eta = data["eta_far"].to_numpy(float)
    one_minus_eta = np.maximum(1.0 - eta, 1e-300)
    useful = data["Zblocked_real_ohm"].to_numpy(float) * data["efficiency_odds_far"].to_numpy(float)
    internal = data["Rinternal_far_ohm"].to_numpy(float) / one_minus_eta
    data["useful_efficiency_budget_ohm"] = useful
    data["internal_loss_penalty_budget_ohm"] = internal
    data["cumulative_moment_0"] = np.r_[0.0, cumulative_trapezoid(resistance, omega)] * 2.0 / np.pi
    data["cumulative_moment_2"] = np.r_[0.0, cumulative_trapezoid(resistance / omega**2, omega)] * 2.0 / np.pi
    data["cumulative_useful_moment_0"] = np.r_[0.0, cumulative_trapezoid(useful, omega)] * 2.0 / np.pi
    data["cumulative_internal_moment_0"] = np.r_[0.0, cumulative_trapezoid(internal, omega)] * 2.0 / np.pi
    data["cumulative_useful_moment_2"] = np.r_[0.0, cumulative_trapezoid(useful / omega**2, omega)] * 2.0 / np.pi
    data["cumulative_internal_moment_2"] = np.r_[0.0, cumulative_trapezoid(internal / omega**2, omega)] * 2.0 / np.pi
    result = {
        "window_moment_0_ohm_per_s": float(data["cumulative_moment_0"].iloc[-1]),
        "window_moment_2_ohm_s": float(data["cumulative_moment_2"].iloc[-1]),
        "window_useful_efficiency_moment_0_ohm_per_s": float(data["cumulative_useful_moment_0"].iloc[-1]),
        "window_internal_penalty_moment_0_ohm_per_s": float(data["cumulative_internal_moment_0"].iloc[-1]),
        "window_useful_efficiency_moment_2_ohm_s": float(data["cumulative_useful_moment_2"].iloc[-1]),
        "window_internal_penalty_moment_2_ohm_s": float(data["cumulative_internal_moment_2"].iloc[-1]),
    }
    return data, result


def nested_quadrature_convergence(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compare the full logarithmic grid with its exactly nested 2x/4x subgrids."""
    data = frame.sort_values("omega_rad_s").reset_index(drop=True)
    result: dict[str, dict[str, float]] = {}
    for stride in (4, 2, 1):
        subset = data.iloc[::stride]
        if subset.index[-1] != data.index[-1]:
            subset = pd.concat([subset, data.iloc[[-1]]]).drop_duplicates("omega_rad_s")
        _, values = integrate_moments(subset)
        result[stride] = {
            "n_frequencies": int(len(subset)),
            "moment_0_ohm_per_s": values["window_moment_0_ohm_per_s"],
            "moment_2_ohm_s": values["window_moment_2_ohm_s"],
        }
    fine = result[1]
    medium = result[2]
    result["fine_vs_medium"] = {
        "moment_0_relative_change": abs(fine["moment_0_ohm_per_s"] - medium["moment_0_ohm_per_s"]) / max(abs(fine["moment_0_ohm_per_s"]), 1e-300),
        "moment_2_relative_change": abs(fine["moment_2_ohm_s"] - medium["moment_2_ohm_s"]) / max(abs(fine["moment_2_ohm_s"]), 1e-300),
    }
    return result


def make_plots(frame: pd.DataFrame, endpoint: dict, out: Path) -> list[str]:
    written = []
    f = frame["freq_Hz"].to_numpy(float)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].semilogx(f, frame["Zmot_real_ohm"], label="Re Zmot")
    axes[0].semilogx(f, frame["Rac_far_ohm"], label="Rac (far field)")
    axes[0].semilogx(f, frame["Rinternal_far_ohm"], label="Rinternal residual")
    axes[0].axhline(0, color="k", lw=0.7); axes[0].set_ylabel("Resistance [ohm]"); axes[0].legend()
    axes[1].semilogx(f, frame["eta_far"], label="eta")
    axes[1].semilogx(f, frame["efficiency_odds_far"], label="eta/(1-eta)")
    axes[1].set_xlabel("Frequency [Hz]"); axes[1].set_ylabel("Dimensionless"); axes[1].legend()
    for ax in axes: ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); path = out / "power_efficiency_audit.png"; fig.savefig(path, dpi=180); plt.close(fig); written.append(path.name)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].semilogx(f, frame["cumulative_moment_0"], label="finite-window M0")
    axes[0].axhline(endpoint["A_inf_matrix_ohm_per_s"], color="k", ls="--", label="matrix Ainf")
    axes[0].set_ylabel("(2/pi) integral ReZ dw")
    axes[1].semilogx(f, frame["cumulative_moment_2"], label="finite-window M2")
    axes[1].axhline(endpoint["A0_structure_only_ohm_s"], color="k", ls="--", label="structure-only A0")
    axes[1].set_ylabel("(2/pi) integral ReZ/w^2 dw"); axes[1].set_xlabel("Upper frequency [Hz]")
    for ax in axes: ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); path = out / "cumulative_sum_rules.png"; fig.savefig(path, dpi=180); plt.close(fig); written.append(path.name)
    return written


def render_report(summary: dict) -> str:
    checks = summary["checks"]
    moments = summary["moments"]
    end = summary["endpoints"]
    return f"""# 扬声器 FEM 求和规则深入验证

## 验证口径

- 固定配置：`{summary['config']}`，全频段不切换网格或离散阶次。
- 相量约定：`exp(+i omega t)`；求和规则中的端点符号已按该约定转换。
- 辐射功率由 1 m 半空间完整方向图积分得到，同时保留 Boundary 93 有符号通量作交叉检查。
- 当前结果是有限频窗数值审计，不把有限窗积分冒充为 0 至无穷的严格等式。

## 主要结果

- 频率窗：{summary['frequency_window_Hz'][0]:.6g}–{summary['frequency_window_Hz'][1]:.6g} Hz，共 {summary['n_frequencies']} 点。
- 离散高频端点 A_inf：{end['A_inf_matrix_ohm_per_s']:.9g} ohm/s。
- 结构静态端点 A0（不含压力公式在零频的密闭空气约束）：{end['A0_structure_only_ohm_s']:.9g} ohm s。
- 有限窗 M0：{moments['window_moment_0_ohm_per_s']:.9g} ohm/s，占 A_inf 的 {100*moments['moment_0_fraction_of_Ainf']:.4g}%。
- 有限窗 M2：{moments['window_moment_2_ohm_s']:.9g} ohm s；这里只与结构 A0 并列，不作为耦合系统严格闭合率。
- M0 中有用辐射效率项：{moments['window_useful_efficiency_moment_0_ohm_per_s']:.9g} ohm/s；内部损耗惩罚项：{moments['window_internal_penalty_moment_0_ohm_per_s']:.9g} ohm/s。
- M2 中有用辐射效率项：{moments['window_useful_efficiency_moment_2_ohm_s']:.9g} ohm s；内部损耗惩罚项：{moments['window_internal_penalty_moment_2_ohm_s']:.9g} ohm s。
- 嵌套网格中，最细与次细频率积分的 M0/M2 相对变化：{100*summary['quadrature_convergence']['fine_vs_medium']['moment_0_relative_change']:.4g}% / {100*summary['quadrature_convergence']['fine_vs_medium']['moment_2_relative_change']:.4g}%。
- 最小 Re Zmot：{checks['minimum_Re_Zmot_ohm']:.9g} ohm。
- 最小辐射后内部损耗余量：{checks['minimum_Rinternal_far_ohm']:.9g} ohm。
- 效率 odds 代数一致性最大绝对残差：{checks['max_abs_odds_identity_residual_ohm']:.3e} ohm。
- Boundary 93 通量与远场功率的全频中位相对差：{100*checks['median_boundary_vs_far_power_relative_difference']:.4g}%。
- 100 Hz 以上，Boundary 93 与远场功率的 95% 分位/最大相对差：{100*checks['p95_boundary_vs_far_power_relative_difference_above_100Hz']:.4g}% / {100*checks['max_boundary_vs_far_power_relative_difference_above_100Hz']:.4g}%。
- 低频 Boundary 93 有符号通量为负的频点比例：{100*checks['negative_boundary_flux_fraction']:.4g}%；这是小 `ka` 下入射/出射通量相减的数值消减误差，这些点不用于功率闭合判据。

## 判读边界

1. 生产材料采用频率无关的结构 loss factor（滞回损耗）。把它外推到零频会令 `Re Zmot/omega^2` 产生对数型低频问题，因此原文的 A0 双矩等式不能在该经验损耗模型上未经修改直接宣称严格成立。
2. 压力声学离散在严格零频不能自动恢复密闭空气的静态刚度；故上面的矩阵 A0 只是结构端点。
3. 最高频有限元仍受网格分辨率限制；M0 尚未覆盖的权重既包含 15 kHz 以上物理尾部，也包含有限带宽和离散误差。
4. `Rinternal_far >= 0` 是被动功率分解的必要数值检查；若出现负值，应先检查方向图半径、Boundary 93 恢复和网格收敛，而不能解释成主动增益。
5. 自动 P1/P2/网格路由适合工程预测，但不是单一解析传递函数；严格 Herglotz 验证必须像本次一样使用固定配置。

机器可读逐频数据见 `sum_rule_frequency_audit.csv`，完整参数、哈希和检查见 `sum_rule_summary.json`。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sum_rules_highest_accuracy.json")
    parser.add_argument("--freqs", default="log:1:15000:161")
    parser.add_argument("--outdir", default="runs/sum_rules_highest_accuracy")
    parser.add_argument("--magnetostatic-vtu", default="inputs/comsol_reference/magnetostatic_converged_55iter.vtu")
    parser.add_argument("--without-nra", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--blas-threads", type=int, default=2)
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    magnetic_path = (ROOT / args.magnetostatic_vtu).resolve()
    out = (ROOT / args.outdir).resolve(); out.mkdir(parents=True, exist_ok=True)
    frequencies = np.unique(parse_frequencies(args.freqs))
    checkpoint = out / "sum_rule_frequency_checkpoint.csv"
    rows: list[dict] = []
    if args.resume and checkpoint.exists():
        rows = pd.read_csv(checkpoint).to_dict("records")
    completed = {round(float(row["freq_Hz"]), 10) for row in rows}

    model = build_best_model(ROOT, config_path=config_path, magnetostatic_vtu=magnetic_path)
    endpoints = endpoint_budgets(model)
    model_metadata = {
        "solid": model.solid.summary(), "coupling": model.G_info,
        "acoustic_unknowns": int(model.acoustic_operator.n2),
    }
    model_config = model.config
    del model
    gc.collect()
    pending = [float(f) for f in frequencies if round(float(f), 10) not in completed]
    if int(args.jobs) <= 1:
        worker_init(str(config_path), str(magnetic_path), not args.without_nra, args.blas_threads)
        iterator = ((frequency, worker_solve(frequency)) for frequency in pending)
        for index, (frequency, row) in enumerate(iterator, len(rows) + 1):
            rows.append(row)
            pd.DataFrame(rows).sort_values("freq_Hz").to_csv(checkpoint, index=False)
            print(f"[{index}/{len(frequencies)}] {frequency:.8g} Hz in {row['solve_seconds']:.2f} s", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=int(args.jobs), initializer=worker_init,
            initargs=(str(config_path), str(magnetic_path), not args.without_nra, args.blas_threads),
        ) as pool:
            futures = {pool.submit(worker_solve, frequency): frequency for frequency in pending}
            for index, future in enumerate(as_completed(futures), len(rows) + 1):
                frequency = futures[future]
                row = future.result()
                rows.append(row)
                pd.DataFrame(rows).sort_values("freq_Hz").to_csv(checkpoint, index=False)
                print(f"[{index}/{len(frequencies)}] {frequency:.8g} Hz in {row['solve_seconds']:.2f} s", flush=True)

    frame, moments = integrate_moments(pd.DataFrame(rows))
    frame.to_csv(out / "sum_rule_frequency_audit.csv", index=False)
    moments["moment_0_fraction_of_Ainf"] = moments["window_moment_0_ohm_per_s"] / endpoints["A_inf_matrix_ohm_per_s"]
    quadrature = nested_quadrature_convergence(frame)
    far = frame["radiated_power_far_W"].to_numpy(float)
    flux = frame["radiated_power_boundary_flux_W"].to_numpy(float)
    valid = np.isfinite(flux) & (far > 0)
    resolved = valid & (far >= np.nanmax(far) * 1e-6)
    above_100 = valid & (frame["freq_Hz"].to_numpy(float) >= 100.0)
    relative_power_difference = np.abs(flux - far) / np.maximum(far, 1e-300)
    checks = {
        "minimum_Re_Zmot_ohm": float(frame["Zmot_real_ohm"].min()),
        "minimum_Rinternal_far_ohm": float(frame["Rinternal_far_ohm"].min()),
        "max_abs_odds_identity_residual_ohm": float(frame["odds_identity_residual_ohm"].abs().max()),
        "median_boundary_vs_far_power_relative_difference": float(np.median(np.abs(flux[valid] - far[valid]) / far[valid])) if np.any(valid) else float("nan"),
        "p95_boundary_vs_far_power_relative_difference_resolved": float(np.quantile(relative_power_difference[resolved], 0.95)) if np.any(resolved) else float("nan"),
        "p95_boundary_vs_far_power_relative_difference_above_100Hz": float(np.quantile(relative_power_difference[above_100], 0.95)) if np.any(above_100) else float("nan"),
        "max_boundary_vs_far_power_relative_difference_above_100Hz": float(np.max(relative_power_difference[above_100])) if np.any(above_100) else float("nan"),
        "negative_boundary_flux_fraction": float(np.mean(flux[valid] < 0)) if np.any(valid) else float("nan"),
        "passivity_on_sampled_real_axis": bool(frame["Zmot_real_ohm"].min() >= -1e-8),
        "nonnegative_internal_loss_on_sampled_real_axis": bool(frame["Rinternal_far_ohm"].min() >= -1e-6),
    }
    input_paths = [config_path, magnetic_path, ROOT / model_config["geometry"]["mesh"], ROOT / model_config["geometry"]["structure_mesh"]]
    summary = {
        "config": str(config_path.relative_to(ROOT)),
        "frequency_window_Hz": [float(frame.freq_Hz.min()), float(frame.freq_Hz.max())],
        "n_frequencies": int(len(frame)),
        "phasor_convention": model_config["exterior"].get("phasor_convention"),
        "nra_enabled": not args.without_nra,
        "endpoints": endpoints,
        "moments": moments,
        "quadrature_convergence": quadrature,
        "checks": checks,
        "mesh_and_dofs": model_metadata,
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in input_paths},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "total_frequency_solve_seconds": float(frame.solve_seconds.sum())},
        "command": " ".join(sys.argv),
    }
    summary["plots"] = make_plots(frame, endpoints, out)
    (out / "sum_rule_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "VALIDATION_REPORT_CN.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({"outdir": str(out), "checks": checks, "moments": moments}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
