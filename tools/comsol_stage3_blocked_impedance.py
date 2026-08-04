#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from loudspeaker_axisym_fem.axisym_magnetics import (
    load_tagged_meshio,
    solve_axisymmetric_magnetostatics,
    solve_blocked_coil_impedance,
    write_blocked_impedance_csv,
    write_blocked_impedance_vtu,
    plot_blocked_impedance,
    plot_induced_current_density,
    skin_depth_m,
)
from loudspeaker_axisym_fem.comsol_driver_model import SOFT_IRON_BH_TABLE, COMSOL_TARGETS, ComsolDriverParameters
from loudspeaker_axisym_fem.json_utils import write_json, dumps_json


def parse_freqs(s: str):
    if s.strip().lower() in ('comsol-low', 'default'):
        return [1,2,3,4,5,6,7,8,9,10,20,50,100,200,500,900,1000,2000,3500,5000,8000]
    vals=[]
    for part in s.replace(';', ',').split(','):
        part=part.strip()
        if part:
            vals.append(float(part))
    return vals


def main():
    ap = argparse.ArgumentParser(description='Stage-3 COMSOL loudspeaker blocked coil impedance / eddy-current perturbation reproduction.')
    ap.add_argument('--mesh', default=str(ROOT / 'outputs_comsol_reproduction_stage1' / 'comsol_geometry_polyline_debug.msh'))
    ap.add_argument('--outdir', default=str(ROOT / 'outputs_comsol_reproduction_stage3_blocked_impedance'))
    ap.add_argument('--freqs', default='default')
    ap.add_argument('--store-field-freqs', default='50,900')
    ap.add_argument('--static-max-iter', type=int, default=8)
    ap.add_argument('--static-relaxation', type=float, default=0.55)
    ap.add_argument('--static-mu-initial-soft', type=float, default=700.0)
    ap.add_argument('--calibrate-static-bl', action='store_true', default=True)
    ap.add_argument('--linearized-mu-mode', choices=['differential','effective','stage2'], default='differential')
    ap.add_argument('--Rdc-ohm', type=float, default=COMSOL_TARGETS['dc_resistance_ohm'])
    ap.add_argument('--sigma-soft-iron', type=float, default=1.12e7)
    ap.add_argument('--core-inductance-scale', type=float, default=1.0, help='Stage-3B terminal correction: scale FEM eddy-current core flux linkage')
    ap.add_argument('--leakage-inductance-mH', type=float, default=0.0, help='Stage-3B terminal correction: add unresolved leakage inductance in mH')
    ap.add_argument('--fit-figure6-two-path', action='store_true', help='Fit core scale and leakage inductance to visual COMSOL Figure 6 low/high anchors')
    ap.add_argument('--figure6-low-anchor-Hz', type=float, default=1.0)
    ap.add_argument('--figure6-low-anchor-mH', type=float, default=1.78)
    ap.add_argument('--figure6-high-anchor-Hz', type=float, default=8000.0)
    ap.add_argument('--figure6-high-anchor-mH', type=float, default=0.80)
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    params = ComsolDriverParameters()
    mesh = load_tagged_meshio(args.mesh)
    static = solve_axisymmetric_magnetostatics(
        mesh,
        soft_iron_domains=(6,23),
        magnet_domains=(24,),
        coil_domains=(17,18,19),
        N0=params.N0,
        remanence_T=0.4,
        target_BL_N_A=COMSOL_TARGETS['BL_N_per_A'],
        bh_table=SOFT_IRON_BH_TABLE,
        max_iter=args.static_max_iter,
        tol=5e-3,
        relaxation=args.static_relaxation,
        mu_r_initial_soft=args.static_mu_initial_soft,
        calibrate_to_BL=args.calibrate_static_bl,
    )
    freqs = parse_freqs(args.freqs)
    store_freqs = parse_freqs(args.store_field_freqs)
    core_scale = float(args.core_inductance_scale)
    leak_H = float(args.leakage_inductance_mH) * 1e-3
    calibration_note = 'none'
    if args.fit_figure6_two_path:
        raw_for_fit = solve_blocked_coil_impedance(
            static,
            freqs,
            bh_table=SOFT_IRON_BH_TABLE,
            soft_iron_domains=(6,23),
            conducting_domains=(6,23),
            coil_domains=(17,18,19),
            N0=params.N0,
            Rdc_ohm=args.Rdc_ohm,
            sigma_soft_iron_S_m=args.sigma_soft_iron,
            linearized_mu_mode=args.linearized_mu_mode,
            store_field_frequencies=(),
        )
        farr = raw_for_fit.frequencies_Hz
        Lraw = raw_for_fit.Lb_H
        def interp_L(f):
            return float(__import__('numpy').interp(float(f), farr, Lraw))
        Llo_raw = interp_L(args.figure6_low_anchor_Hz)
        Lhi_raw = interp_L(args.figure6_high_anchor_Hz)
        Llo_t = float(args.figure6_low_anchor_mH) * 1e-3
        Lhi_t = float(args.figure6_high_anchor_mH) * 1e-3
        denom = Llo_raw - Lhi_raw
        if abs(denom) < 1e-12:
            raise RuntimeError('cannot fit two-path correction: raw low/high inductance are too close')
        core_scale = (Llo_t - Lhi_t) / denom
        leak_H = Lhi_t - core_scale * Lhi_raw
        calibration_note = f'fit Figure 6 anchors: {args.figure6_low_anchor_Hz:g} Hz={args.figure6_low_anchor_mH:g} mH, {args.figure6_high_anchor_Hz:g} Hz={args.figure6_high_anchor_mH:g} mH'
    elif args.core_inductance_scale != 1.0 or args.leakage_inductance_mH != 0.0:
        calibration_note = 'manual two-path terminal correction'

    blocked = solve_blocked_coil_impedance(
        static,
        freqs,
        bh_table=SOFT_IRON_BH_TABLE,
        soft_iron_domains=(6,23),
        conducting_domains=(6,23),
        coil_domains=(17,18,19),
        N0=params.N0,
        Rdc_ohm=args.Rdc_ohm,
        sigma_soft_iron_S_m=args.sigma_soft_iron,
        linearized_mu_mode=args.linearized_mu_mode,
        store_field_frequencies=store_freqs,
        core_inductance_scale=core_scale,
        leakage_inductance_H=leak_H,
        calibration_note=calibration_note,
    )
    write_blocked_impedance_csv(outdir / 'blocked_coil_impedance.csv', blocked)
    for f in store_freqs:
        if float(f) in blocked.A_phi_by_frequency:
            write_blocked_impedance_vtu(outdir / f'blocked_impedance_field_{int(round(f))}Hz.vtu', blocked, float(f))
    if not args.no_plots:
        plot_blocked_impedance(outdir / 'figure6', blocked)
        for f in store_freqs:
            if float(f) in blocked.A_phi_by_frequency:
                plot_induced_current_density(outdir / 'figure5', blocked, float(f), quantity='real')
                plot_induced_current_density(outdir / 'figure5', blocked, float(f), quantity='abs')
    # Pull a few anchor values for comparison with PDF Figure 6.
    f_to_L = {float(f): float(L*1e3) for f, L in zip(blocked.frequencies_Hz, blocked.Lb_H)}
    f_to_Z = {float(f): {'real': float(z.real), 'imag': float(z.imag), 'abs': float(abs(z))} for f, z in zip(blocked.frequencies_Hz, blocked.Zb_ohm)}
    summary = {
        'stage': 'stage-3 blocked coil impedance / eddy-current perturbation',
        'mesh': str(args.mesh),
        'settings': vars(args),
        'static_summary': static.summary(),
        'blocked_summary': blocked.summary(),
        'stage3b_terminal_correction': {
            'core_inductance_scale': core_scale,
            'leakage_inductance_mH': leak_H * 1e3,
            'note': calibration_note,
        },
        'skin_depth_checks_m': {
            '50Hz_mu1200': skin_depth_m(50.0, args.sigma_soft_iron, 1200.0),
            '900Hz_mu1200': skin_depth_m(900.0, args.sigma_soft_iron, 1200.0),
            '8000Hz_mu1200': skin_depth_m(8000.0, args.sigma_soft_iron, 1200.0),
        },
        'anchors': {
            'L_mH_at_1Hz': f_to_L.get(1.0),
            'L_mH_at_50Hz': f_to_L.get(50.0),
            'L_mH_at_100Hz': f_to_L.get(100.0),
            'L_mH_at_900Hz': f_to_L.get(900.0),
            'L_mH_at_1000Hz': f_to_L.get(1000.0),
            'L_mH_at_8000Hz': f_to_L.get(8000.0),
            'Z_ohm_at_50Hz': f_to_Z.get(50.0),
            'Z_ohm_at_900Hz': f_to_Z.get(900.0),
        },
        'COMSOL_reference': {
            'Figure_5': 'induced conduction current density phi component at 50 Hz and 900 Hz; skin depth decreases with frequency',
            'Figure_6': 'blocked coil inductance decreases with frequency, roughly from 1.8 mH at low frequency to 0.8 mH at the high end of the 8 kHz sweep',
            'coil': 'homogenized multiturn, N0=100, user wire area 3.5e-8 m^2, voltage linper(V0)',
        },
        'outputs': {
            'csv': 'blocked_coil_impedance.csv',
            'figure6_inductance': 'figure6_blocked_coil_inductance.png',
            'figure6_impedance': 'figure6_blocked_impedance.png',
            'figure5_current_50Hz': 'figure5_Jphi_real_50Hz.png',
            'figure5_current_900Hz': 'figure5_Jphi_real_900Hz.png',
        },
    }
    write_json(outdir / 'stage3_blocked_impedance_summary.json', summary, indent=2)
    report = [
        '# Stage 3 blocked coil impedance / eddy-current 复现报告',
        '',
        '## COMSOL 目标',
        '',
        '- Figure 5：50 Hz 与 900 Hz 的 pole piece/top plate 感应电流密度。PDF 明确说明频率升高后 skin depth 下降，电流更靠近铁件表面。',
        '- Figure 6：blocked coil inductance 随频率下降。PDF 图中低频约 1.8 mH，高频端约 0.8 mH。',
        '',
        '## 本阶段实现',
        '',
        '- 使用 Stage 2 的 A_phi 静磁场作为线性化点。',
        f'- soft iron 频域线性化模式：`{args.linearized_mu_mode}`。',
        '- 频域扰动方程：`curl(nu curl A) + i omega sigma A = J_src`。',
        '- Domains 6/23 中加入 `sigma=1.12e7 S/m` 涡流项。',
        '- Domains 17–19 施加均匀多匝 unit-current coil source，再由 flux linkage 得到 blocked impedance。',
        '- 线圈 DC 电阻按 PDF/COMSOL 目标取 `Rdc=5.6 ohm`。',
        f'- Stage-3B terminal correction: core scale = `{core_scale:.6g}`, leakage = `{leak_H*1e3:.6g} mH`, note = `{calibration_note}`。',
        '',
        '## 当前数值锚点',
        '',
        f'- 网格节点数：{blocked.mesh.n_nodes}',
        f'- 三角形数：{blocked.mesh.n_triangles}',
        f'- coil rz 面积：{blocked.coil_area_m2:.6e} m²',
        f'- L(1 Hz)：{f_to_L.get(1.0, float("nan")):.6g} mH',
        f'- L(50 Hz)：{f_to_L.get(50.0, float("nan")):.6g} mH',
        f'- L(900 Hz)：{f_to_L.get(900.0, float("nan")):.6g} mH',
        f'- L(8 kHz)：{f_to_L.get(8000.0, float("nan")):.6g} mH',
        f'- skin depth 50 Hz, mu_r=1200：{skin_depth_m(50.0, args.sigma_soft_iron, 1200.0)*1e3:.4g} mm',
        f'- skin depth 900 Hz, mu_r=1200：{skin_depth_m(900.0, args.sigma_soft_iron, 1200.0)*1e3:.4g} mm',
        f'- skin depth 8 kHz, mu_r=1200：{skin_depth_m(8000.0, args.sigma_soft_iron, 1200.0)*1e3:.4g} mm',
        '',
        '## 与 COMSOL 的差异',
        '',
        '1. Stage-3B 已加入端口两路径校正：`lambda_eff=L_leak I + scale_core lambda_FEM`，用于表示粗网格 scalar A_phi 没有解析到的 residual/leakage flux；raw FEM 仍保留在 CSV 的 raw_* 列中。',
        '2. 严格的 COMSOL `Domain Coil` voltage terminal equation 尚未完全等价实现；当前端口校正用于对齐 Figure 6 的端口电感曲线，并不改变 Figure 5 的 raw eddy-current 分布。',
        '2. 当前使用 sampled Bezier polyline debug mesh，尚未复现 COMSOL mapped mesh + near-iron 0.5 mm boundary-layer mesh。Figure 5 的 skin current 对网格非常敏感。',
        '3. 当前 scalar A_phi 忽略 electric scalar potential gauge/coil terminal equation；因此 L(f) 的趋势比绝对值更可信。',
        '4. 当前 Stage 2 raw BL 仍有 14–16% 差异，虽然本阶段默认用 calibrated static field 对齐后续链路，磁静态差异仍会传递到 Zb(f)。',
        '',
        '## 阶段判断',
        '',
        '本阶段已经完成可运行的涡流频域扰动与 blocked inductance 曲线，能展示 50 Hz 到 900 Hz 的电流向铁件表面集中的趋势。',
        '下一步应修正 coil terminal voltage constraint 和局部 boundary-layer mesh，再将 Figure 6 的绝对电感曲线收敛到 COMSOL PDF 图示量级。',
    ]
    (outdir / 'STAGE3_BLOCKED_IMPEDANCE_REPORT_CN.md').write_text('\n'.join(report), encoding='utf-8')
    print(dumps_json(summary, indent=2))


if __name__ == '__main__':
    main()
