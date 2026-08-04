#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, sys, time, shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from loudspeaker_axisym_fem.axisym_magnetics import load_tagged_meshio
from loudspeaker_axisym_fem.stage4_solid_fem import build_stage4_solid_model, default_stage4_materials, SolidMaterial
from loudspeaker_axisym_fem.mmcpl_lorentz_backemf import (
    assemble_lorentz_backemf_vector,
    rigid_reciprocity_tests,
    power_reciprocity_test,
    solve_fixed_structure_regression,
    assemble_minimal_mmcpl_block,
    solve_mmcpl_block_for_frequency,
)


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def write_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8'); return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open('w', newline='', encoding='utf-8') as fp:
        w=csv.DictWriter(fp, fieldnames=keys); w.writeheader(); w.writerows(rows)


def read_blocked_csv(path: Path):
    rows=[]
    with path.open(newline='', encoding='utf-8') as fp:
        for r in csv.DictReader(fp):
            rows.append(r)
    freqs=np.array([float(r['f_Hz']) for r in rows], dtype=float)
    Z=np.array([float(r['Zb_real_ohm']) + 1j*float(r['Zb_imag_ohm']) for r in rows], dtype=complex)
    return rows, freqs, Z


def materials_with_suspension_scale(scale: float) -> dict[int, SolidMaterial]:
    base = default_stage4_materials()
    mats = {}
    for d, m in base.items():
        mats[d] = SolidMaterial(
            E=m.E * (scale if d in (20,25) else 1.0),
            nu=m.nu,
            rho=m.rho,
            loss_factor=m.loss_factor,
            beta_dK=m.beta_dK,
            label=m.label,
        )
    return mats


def plot_regression(path: Path, freqs, Zin, Zreg):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.semilogx(freqs, np.abs(Zin), 'o-', label='Stage5C Zb input abs')
    ax.semilogx(freqs, np.abs(Zreg), 'x--', label='Stage7B fixed-structure regressed abs')
    ax.semilogx(freqs, Zin.imag, 'o-', alpha=0.7, label='Stage5C imag')
    ax.semilogx(freqs, Zreg.imag, 'x--', alpha=0.7, label='Stage7B imag')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Ohm')
    ax.set_title('Stage 7B fixed-structure blocked impedance regression')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description='Stage 7A/7B energy-conjugate Lorentz/back-EMF block assembly and fixed-structure blocked impedance regression')
    ap.add_argument('--mesh', default=str(ROOT/'meshes'/'comsol_stable_1mm_05gap.msh'))
    ap.add_argument('--magnetostatic-vtu', default=str(ROOT/'outputs'/'stage2_magnetics'/'magnetostatic_solution.vtu'))
    ap.add_argument('--blocked-csv', default=str(ROOT/'outputs'/'stage5ABC_figure_magnetics_coil_closure'/'stage5C_domain_coil_closure'/'stage5C_selected_sigma_blocked_impedance.csv'))
    ap.add_argument('--outdir', default=str(ROOT/'outputs'/'stage7AB_mmcpl_energy_conjugate'))
    ap.add_argument('--uniform-refine', type=int, default=0)
    ap.add_argument('--suspension-E-scale', type=float, default=0.7915555556)
    ap.add_argument('--N0', type=float, default=100.0)
    ap.add_argument('--V0', type=float, default=3.55)
    ap.add_argument('--block-demo-freqs', default='50,1000')
    args=ap.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    t0=time.time()

    mesh=load_tagged_meshio(args.mesh)
    mats=materials_with_suspension_scale(args.suspension_E_scale)
    model=build_stage4_solid_model(mesh, materials=mats, uniform_refine=args.uniform_refine)
    cpl=assemble_lorentz_backemf_vector(model, args.magnetostatic_vtu, N0=args.N0)
    write_json(out/'stage7A_lorentz_backemf_vector_summary.json', cpl.summary())
    np.savez_compressed(out/'stage7A_lorentz_backemf_vector.npz', g_full_N_per_A=cpl.g_full_N_per_A, g_free_N_per_A=cpl.g_free_N_per_A, free_dofs=cpl.free_dofs)

    rigid=rigid_reciprocity_tests(cpl)
    power=power_reciprocity_test(cpl, omega=2*math.pi*50.0, seed=7)
    tests={
        'rigid_reciprocity': rigid,
        'power_reciprocity': power,
        'acceptance': {
            'rigid_axial_abs_error_tol': 1e-10,
            'power_relative_difference_tol': 1e-12,
            'rigid_axial_pass': rigid['axial_reciprocity_abs_error'] < 1e-10,
            'power_pass': power['relative_difference'] < 1e-12,
        }
    }
    write_json(out/'stage7A_energy_conjugacy_tests.json', tests)

    source_rows, freqs, Zin = read_blocked_csv(Path(args.blocked_csv))
    reg=solve_fixed_structure_regression(freqs, Zin, V0=args.V0)
    rows=[]
    for i,f in enumerate(freqs):
        rows.append({
            'f_Hz': float(f),
            'Zb_input_real_ohm': float(reg['Z_input_ohm'][i].real),
            'Zb_input_imag_ohm': float(reg['Z_input_ohm'][i].imag),
            'Zb_input_abs_ohm': float(abs(reg['Z_input_ohm'][i])),
            'I_fixed_real_A_peak': float(reg['I_fixed_A_peak'][i].real),
            'I_fixed_imag_A_peak': float(reg['I_fixed_A_peak'][i].imag),
            'Z_regressed_real_ohm': float(reg['Z_regressed_ohm'][i].real),
            'Z_regressed_imag_ohm': float(reg['Z_regressed_ohm'][i].imag),
            'Z_regressed_abs_ohm': float(abs(reg['Z_regressed_ohm'][i])),
            'abs_error_ohm': float(reg['abs_error_ohm'][i]),
            'rel_error': float(reg['rel_error'][i]),
        })
    write_rows(out/'stage7B_fixed_structure_blocked_impedance_regression.csv', rows)
    plot_regression(out/'figure6_stage7B_fixed_structure_blocked_impedance_regression.png', freqs, Zin, reg['Z_regressed_ohm'])

    # Demonstrate actual energy-conjugate block assembly dimensions without using gamma.
    demo_rows=[]
    for fs in [float(x) for x in args.block_demo_freqs.split(',') if x.strip()]:
        Zb = np.interp(fs, freqs, Zin.real) + 1j*np.interp(fs, freqs, Zin.imag)
        A = assemble_minimal_mmcpl_block(model, cpl, Zb, fs)
        demo_rows.append({
            'f_Hz': fs,
            'block_unknowns': int(A.shape[0]),
            'nnz': int(A.nnz),
            'Zb_real_ohm': float(Zb.real),
            'Zb_imag_ohm': float(Zb.imag),
            'top_right_uses': 'i*omega*g_free^T',
            'bottom_left_uses': '-g_free',
            'gamma_used': False,
        })
    write_rows(out/'stage7A_block_assembly_dimensions.csv', demo_rows)

    # Optional solid-coupled demonstration only for small model; it is not Stage 7C acceptance.
    coupled_rows=[]
    for fs in [float(x) for x in args.block_demo_freqs.split(',') if x.strip()]:
        Zb = np.interp(fs, freqs, Zin.real) + 1j*np.interp(fs, freqs, Zin.imag)
        try:
            I,u,Z = solve_mmcpl_block_for_frequency(model, cpl, Zb, fs, V0=args.V0)
            coupled_rows.append({
                'f_Hz': fs,
                'I_real_A_peak': float(I.real),
                'I_imag_A_peak': float(I.imag),
                'Z_mmcpl_abs_ohm_demo': float(abs(Z)),
                'Z_mmcpl_real_ohm_demo': float(Z.real),
                'Z_mmcpl_imag_ohm_demo': float(Z.imag),
                'note': 'diagnostic only; Stage7C/D will validate solid motional impedance',
            })
        except Exception as e:
            coupled_rows.append({'f_Hz': fs, 'error': repr(e)})
    write_rows(out/'stage7A_minimal_block_solve_diagnostic.csv', coupled_rows)

    summary={
        'stage': 'Stage 7A-7B energy-conjugate mmcpl assembly and fixed-structure blocked impedance regression',
        'inputs': vars(args),
        'model_summary': model.summary(),
        'coupling_summary': cpl.summary(),
        'reciprocity_tests': tests,
        'stage7B_regression': {
            'n_freqs': int(len(freqs)),
            'max_abs_error_ohm': float(np.max(reg['abs_error_ohm'])),
            'max_rel_error': float(np.max(reg['rel_error'])),
            'rms_abs_error_ohm': float(np.sqrt(np.mean(reg['abs_error_ohm']**2))),
            'pass': bool(np.max(reg['rel_error']) < 1e-14),
        },
        'block_demo': demo_rows,
        'elapsed_s': time.time()-t0,
        'interpretation': {
            'gamma_removed': True,
            'stage7A_status': 'PASS' if tests['acceptance']['rigid_axial_pass'] and tests['acceptance']['power_pass'] else 'CHECK',
            'stage7B_status': 'PASS' if np.max(reg['rel_error']) < 1e-14 else 'CHECK',
            'important_note': 'Stage7A/7B validate the weak-form coupling vector and fixed-structure electric regression. They do not yet claim full Figure 8/10 closure without structural/acoustic validation; that is Stage7C-E.',
        }
    }
    write_json(out/'stage7AB_summary.json', summary)

    report=[
        '# Stage 7A–7B：energy-conjugate Lorentz/back-EMF block assembly 与 fixed-structure blocked impedance regression',
        '',
        '## 目标',
        '',
        '- Stage 7A：把 Lorentz force 和 back EMF 写成同一个逐自由度耦合向量 `g`，删除 Stage 6 的 `gamma` 投影修正。',
        '- Stage 7B：在结构固定 `u=0` 时，新的 block formulation 必须退化为 Stage 5C 的 blocked impedance `Zb(ω)`。',
        '',
        '## 弱式实现',
        '',
        '对 coil domain，采用 COMSOL tutorial 的 homogenized multiturn 截面电流密度：',
        '',
        '`J_phi = I * N0 / A_coil`，其中 `A_coil = ∫_coil dA` 是 rz 平面截面积。',
        '',
        '静态磁场 `B0 = Br e_r + Bz e_z` 时，Lorentz 力密度为：',
        '',
        '`J_phi x B0 = J_phi * (Bz e_r - Br e_z)`。',
        '',
        '装配得到同一个 `g`：机械端 `F_u = I g`，电端 `V_be = g^T v = g^T iωu`。',
        '',
        '## 计算结果',
        '',
        f'- 结构节点：{model.summary()["n_structural_nodes"]}',
        f'- 结构三角形：{model.summary()["n_structural_triangles"]}',
        f'- 自由结构 DOF：{model.summary()["ndof_free"]}',
        f'- Coil rz 截面积 `A_coil`：{cpl.Acoil_rz_m2:.9e} m²',
        f'- Coil 旋转体积：{cpl.coil_volume_m3:.9e} m³',
        f'- `N0/A_coil`：{cpl.current_shape_per_m2:.9e} 1/m²',
        f'- 由 `g_z` 积分得到的 axial BL：{cpl.axial_BL_N_per_A:.9f} N/A',
        f'- radial resultant：{cpl.radial_resultant_N_per_A:.9f} N/A',
        '',
        '## Stage 7A 验收',
        '',
        f'- 刚体轴向互易误差：{rigid["axial_reciprocity_abs_error"]:.3e}',
        f'- 功率互易相对误差：{power["relative_difference"]:.3e}',
        f'- Stage 7A 状态：{summary["interpretation"]["stage7A_status"]}',
        '',
        '## Stage 7B 验收',
        '',
        f'- 回归频点数：{len(freqs)}',
        f'- fixed-structure `Zb` 最大绝对误差：{summary["stage7B_regression"]["max_abs_error_ohm"]:.3e} Ω',
        f'- fixed-structure `Zb` 最大相对误差：{summary["stage7B_regression"]["max_rel_error"]:.3e}',
        f'- Stage 7B 状态：{summary["interpretation"]["stage7B_status"]}',
        '',
        '## 输出文件',
        '',
        '- `stage7A_lorentz_backemf_vector.npz`：逐自由度 `g_full` / `g_free`。',
        '- `stage7A_energy_conjugacy_tests.json`：刚体互易与功率互易测试。',
        '- `stage7A_block_assembly_dimensions.csv`：minimal `[I,u]` block 维度与稀疏度。',
        '- `stage7B_fixed_structure_blocked_impedance_regression.csv`：固定结构 blocked impedance 回归。',
        '- `figure6_stage7B_fixed_structure_blocked_impedance_regression.png`：Zb 输入/回归图。',
        '',
        '## 仍未声明完成的内容',
        '',
        'Stage 7A–7B 只证明 `mmcpl` 的能量共轭弱式和固定结构电磁退化正确。完整 Figure 10 motional impedance、Figure 8 sensitivity、不使用外场 transfer correction 的闭合，需要 Stage 7C–7E 继续验证。',
    ]
    (out/'STAGE7AB_MMCPL_ASSEMBLY_REPORT_CN.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
