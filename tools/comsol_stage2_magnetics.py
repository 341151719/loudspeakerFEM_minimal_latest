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
    write_element_vtu,
    plot_magnetic_fields,
)
from loudspeaker_axisym_fem.comsol_driver_model import SOFT_IRON_BH_TABLE, COMSOL_TARGETS, ComsolDriverParameters
from loudspeaker_axisym_fem.json_utils import write_json, dumps_json


def main():
    ap = argparse.ArgumentParser(description='Stage-2 COMSOL loudspeaker magnetostatic reproduction: A_phi, H/B/mu, and BL.')
    ap.add_argument('--mesh', default=str(ROOT / 'outputs_comsol_reproduction_stage1' / 'comsol_geometry_polyline_debug.msh'))
    ap.add_argument('--outdir', default=str(ROOT / 'outputs_comsol_reproduction_stage2_magnetics'))
    ap.add_argument('--max-iter', type=int, default=35)
    ap.add_argument('--tol', type=float, default=1e-4)
    ap.add_argument('--mu-r-initial-soft', type=float, default=700.0)
    ap.add_argument('--relaxation', type=float, default=0.1)
    ap.add_argument('--remanence-T', type=float, default=0.4)
    ap.add_argument('--calibrate-to-BL', action='store_true', help='scale final field to COMSOL target BL while reporting raw BL separately')
    ap.add_argument('--nonlinear-update-mode', choices=['H_forward','B_inverse'], default='B_inverse', help='B_inverse uses BH_inv(|B|) secant permeability; H_forward is the legacy Picard update')
    ap.add_argument('--remanence-rhs-sign', type=float, default=1.0, help='diagnostic sign multiplier for the remanent-flux RHS')
    ap.add_argument('--no-plots', action='store_true', help='skip PNG plotting; still writes VTU and JSON')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mesh = load_tagged_meshio(args.mesh)
    params = ComsolDriverParameters()
    res = solve_axisymmetric_magnetostatics(
        mesh,
        soft_iron_domains=(6, 23),
        magnet_domains=(24,),
        coil_domains=(17, 18, 19),
        N0=params.N0,
        remanence_T=args.remanence_T,
        target_BL_N_A=COMSOL_TARGETS['BL_N_per_A'],
        bh_table=SOFT_IRON_BH_TABLE,
        max_iter=args.max_iter,
        tol=args.tol,
        relaxation=args.relaxation,
        mu_r_initial_soft=args.mu_r_initial_soft,
        calibrate_to_BL=args.calibrate_to_BL,
        nonlinear_update_mode=args.nonlinear_update_mode,
        remanence_rhs_sign=args.remanence_rhs_sign,
    )
    write_element_vtu(outdir / 'magnetostatic_solution.vtu', res)
    if not args.no_plots:
        plot_magnetic_fields(outdir / 'figure3_4_magnetostatic', res)
    summary = {
        'stage': 'stage-2 magnetostatic A_phi reproduction',
        'mesh': str(args.mesh),
        'settings': vars(args),
        'summary': res.summary(),
        'COMSOL_targets': {
            'BL_N_per_A': COMSOL_TARGETS['BL_N_per_A'],
            'Figure_3': 'magnetic field norm around motor; max in voice-coil gap',
            'Figure_4': 'effective relative permeability in pole/top plate; saturation near pole center',
        },
        'interpretation': {
            'raw_BL_error_percent': None if not res.bl_raw_N_A else 100.0 * (res.bl_raw_N_A - COMSOL_TARGETS['BL_N_per_A']) / COMSOL_TARGETS['BL_N_per_A'],
            'calibrated_BL_error_percent': None if not res.bl_calibrated_N_A else 100.0 * (res.bl_calibrated_N_A - COMSOL_TARGETS['BL_N_per_A']) / COMSOL_TARGETS['BL_N_per_A'],
            'note': 'Raw BL is the actual Stage-2 Python result. Default Stage-2 now uses B_inverse BH-consistent secant permeability; calibrated field remains optional and should not be used to hide raw discrepancy.',
        },
        'outputs': {
            'vtu': 'magnetostatic_solution.vtu',
            'H_plot': 'figure3_4_magnetostatic_H_norm_A_m.png',
            'B_plot': 'figure3_4_magnetostatic_B_norm_T.png',
            'mu_plot': 'figure3_4_magnetostatic_mu_r.png',
            'Br_plot': 'figure3_4_magnetostatic_B_r_T.png',
        },
    }
    write_json(outdir / 'stage2_magnetics_summary.json', summary, indent=2)
    report = [
        '# Stage 2 磁场复现报告',
        '',
        '## 目标',
        '',
        '- 复现 COMSOL Figure 3：磁路磁场 H 分布。',
        '- 复现 COMSOL Figure 4：soft iron 有效相对磁导率分布。',
        f'- 对齐硬锚点 BL = {COMSOL_TARGETS["BL_N_per_A"]:.2f} N/A。',
        '',
        '## 当前计算结果',
        '',
        f'- 网格节点数：{res.mesh.n_nodes}',
        f'- 三角形数：{res.mesh.n_triangles}',
        f'- Picard 迭代次数：{res.iterations}',
        f'- 残差历史：{res.residual_history}',
        f'- Raw BL：{res.bl_raw_N_A:.6g} N/A',
        f'- COMSOL 目标 BL：{COMSOL_TARGETS["BL_N_per_A"]:.6g} N/A',
        f'- Raw BL 误差：{summary["interpretation"]["raw_BL_error_percent"]:.3f} %',
        f'- Calibration factor：{res.calibration_factor:.6g}',
        f'- Calibrated BL：{res.bl_calibrated_N_A:.6g} N/A',
        f'- 最大 |B|：{res.summary()["B_norm_max_T"]:.6g} T',
        f'- 最大 |H|：{res.summary()["H_norm_max_A_m"]:.6g} A/m',
        f'- soft iron/全域 μr 范围：{res.summary()["mu_r_elem_min"]:.6g} – {res.summary()["mu_r_elem_max"]:.6g}',
        '',
        '## 差异说明',
        '',
        '本阶段使用 COMSOL mphtxt 的最终几何和 domain ID，采用真实 A_phi 轴对称 FEM 装配。默认 soft iron 更新使用 B_inverse：由 |B| 经 B-H 反函数得到 H，再形成 secant μ=B/(μ0H)。',
        '与 COMSOL 仍可能不同的部分：COMSOL 使用其内部 curved geometry/mapped mesh、非线性牛顿线性化和完整 magnetic-domain box selection；本阶段使用 sampled Bezier polyline debug mesh 和 Picard nonlinear update。',
        '因此 raw BL 用于暴露差异；calibrated BL 仅用于后续声固耦合链路对齐，不掩盖 raw 误差。',
        '',
        '## 输出',
        '',
        '- `magnetostatic_solution.vtu`：可视化场数据。',
        '- `figure3_4_magnetostatic_H_norm_A_m.png`：Figure 3 对照图。',
        '- `figure3_4_magnetostatic_mu_r.png`：Figure 4 对照图。',
        '- `stage2_magnetics_summary.json`：机器可读摘要。',
    ]
    (outdir / 'STAGE2_MAGNETICS_REPORT_CN.md').write_text('\n'.join(report), encoding='utf-8')
    print(dumps_json(summary, indent=2))


if __name__ == '__main__':
    main()
