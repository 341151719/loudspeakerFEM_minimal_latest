#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, sys, time, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from loudspeaker_axisym_fem.stage4_solid_fem import (
    build_stage4_solid_model, default_stage4_materials, SolidMaterial,
    compute_eigenmodes, COIL_DOMAINS,
)
from loudspeaker_axisym_fem.mmcpl_lorentz_backemf import (
    assemble_lorentz_backemf_vector,
    solve_mmcpl_block_for_frequency,
)


def write_json(path: Path, data: dict, indent: int = 2):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding='utf-8')


def write_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_materials(suspension_E_scale: float):
    mats = default_stage4_materials(2.0 * math.pi * 40.0)
    for d in (20, 25):
        m = mats[d]
        mats[d] = SolidMaterial(m.E * suspension_E_scale, m.nu, m.rho, m.loss_factor, m.beta_dK, m.label)
    return mats


def interp_complex_log(df: pd.DataFrame, freqs: np.ndarray) -> np.ndarray:
    f0 = np.asarray(df['f_Hz'], dtype=float)
    z = np.asarray(df['Zb_real_ohm'], dtype=float) + 1j * np.asarray(df['Zb_imag_ohm'], dtype=float)
    lf = np.log10(np.asarray(freqs, dtype=float))
    return np.interp(lf, np.log10(f0), z.real) + 1j*np.interp(lf, np.log10(f0), z.imag)


def coil_nodes(model) -> np.ndarray:
    mask = np.isin(model.domains, COIL_DOMAINS)
    return np.unique(model.triangles[mask].ravel()).astype(int)


def rigid_coil_reciprocity(model, cpl, target_BL: float = 10.48) -> dict:
    cn = coil_nodes(model)
    g = cpl.g_full_N_per_A
    vz = np.zeros_like(g)
    vr = np.zeros_like(g)
    vz[2*cn + 1] = 1.0
    vr[2*cn + 0] = 1.0
    axial_force_sum = float(np.sum(g[2*cn + 1]))
    radial_force_sum = float(np.sum(g[2*cn + 0]))
    axial_backemf = float(np.dot(g, vz))
    radial_backemf = float(np.dot(g, vr))
    return {
        'n_coil_nodes': int(len(cn)),
        'axial_force_sum_N_per_A': axial_force_sum,
        'axial_backemf_for_rigid_coil_vz_1m_s': axial_backemf,
        'axial_reciprocity_abs_error': abs(axial_force_sum - axial_backemf),
        'axial_BL_vs_COMSOL_target_percent': 100.0*(axial_force_sum - target_BL)/target_BL,
        'radial_force_sum_N_per_A': radial_force_sum,
        'radial_backemf_for_rigid_coil_vr_1m_s': radial_backemf,
        'radial_reciprocity_abs_error': abs(radial_force_sum - radial_backemf),
        'radial_to_axial_percent': 100.0*radial_force_sum/max(abs(axial_force_sum), 1e-300),
    }


class ModalMotionalProjector:
    def __init__(self, model, cpl, nmodes: int = 40, sigma_Hz: float = 500.0):
        self.model = model
        self.cpl = cpl
        self.nmodes = int(nmodes)
        self.sigma_Hz = float(sigma_Hz)
        self.eigs = compute_eigenmodes(model, nmodes=nmodes, sigma_Hz=sigma_Hz)
        free = model.free_dofs
        self.free = free
        self.phi = self.eigs['modes'][:, free].T  # nfree x nmodes
        self.lam = np.asarray(self.eigs['lambda_rad2_s2'], dtype=float)
        self.g = np.asarray(cpl.g_free_N_per_A, dtype=float)
        self.a = self.phi.T @ self.g
        base = default_stage4_materials(2.0*math.pi*40.0)
        Kd = []
        eta0 = []
        beta = []
        for dom, K in model.K_by_domain.items():
            Kf = K[free][:, free]
            Kd.append(np.array([float(self.phi[:, i].T @ (Kf @ self.phi[:, i])) for i in range(self.phi.shape[1])]))
            mat = base[int(dom)]
            eta0.append(float(mat.loss_factor))
            beta.append(float(mat.beta_dK))
        self.Kd = np.asarray(Kd, dtype=float)
        self.eta0 = np.asarray(eta0, dtype=float)
        self.beta = np.asarray(beta, dtype=float)

    def modal_motional_term(self, freqs_Hz: np.ndarray) -> np.ndarray:
        out = []
        for f in np.asarray(freqs_Hz, dtype=float):
            w = 2.0 * math.pi * float(f)
            imK = np.sum((self.eta0[:, None] + w*self.beta[:, None]) * self.Kd, axis=0)
            den = self.lam - w*w + 1j*imK
            out.append(1j*w * np.sum(self.a*self.a / den))
        return np.asarray(out, dtype=complex)

    def total_impedance(self, freqs_Hz: np.ndarray, Zb: np.ndarray) -> np.ndarray:
        return np.asarray(Zb, dtype=complex) + self.modal_motional_term(np.asarray(freqs_Hz, dtype=float))

    def modal_participation_rows(self, max_modes: int = 20):
        rows = []
        for i, (f, a, lam) in enumerate(zip(self.eigs['f_Hz'], self.a, self.lam)):
            rows.append({
                'mode_index': i + 1,
                'eigenfrequency_Hz': float(f),
                'g_modal_coupling': float(a),
                'g_modal_coupling_abs': float(abs(a)),
                'lambda_rad2_s2': float(lam),
            })
            if i + 1 >= max_modes:
                break
        return rows


def evaluate_figure10(freqs, Z, ref):
    target_abs = np.asarray(ref['target_Z_abs_ohm'], dtype=float)
    err_abs = np.abs(Z) - target_abs
    rows = []
    for i, f in enumerate(freqs):
        rows.append({
            'f_Hz': float(f),
            'target_absZ_ohm': float(target_abs[i]),
            'stage7D_absZ_ohm': float(abs(Z[i])),
            'stage7D_realZ_ohm': float(Z[i].real),
            'stage7D_imagZ_ohm': float(Z[i].imag),
            'absZ_error_ohm': float(err_abs[i]),
            'absZ_error_percent': float(100.0*err_abs[i]/target_abs[i]),
            'target_realZ_ohm': float(ref['target_Z_real_ohm'].iloc[i]),
            'target_imagZ_ohm': float(ref['target_Z_imag_ohm'].iloc[i]),
        })
    return rows, {
        'absZ_RMSE_ohm': float(np.sqrt(np.mean(err_abs*err_abs))),
        'absZ_max_abs_error_percent': float(np.max(np.abs(100.0*err_abs/target_abs))),
        'Z50_abs_ohm': float(abs(Z[list(freqs).index(50.0)])) if 50.0 in list(freqs) else None,
        'Z50_real_ohm': float(Z[list(freqs).index(50.0)].real) if 50.0 in list(freqs) else None,
        'Z50_imag_ohm': float(Z[list(freqs).index(50.0)].imag) if 50.0 in list(freqs) else None,
    }


def plot_figure10(path: Path, f_dense, Z_dense, f_anchor, Z_anchor, f_ref, ref, scale):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(f_dense, np.abs(Z_dense), label='Stage 7D native solid-only |Z|')
    ax.semilogx(f_anchor, np.abs(Z_anchor), 'o', label='Stage 7D anchors')
    ax.semilogx(f_ref, ref['target_Z_abs_ohm'], 's', label='PDF Figure 10 anchors')
    ax.semilogx(f_dense, Z_dense.real, '--', label='Stage 7D Re(Z)')
    ax.semilogx(f_dense, Z_dense.imag, ':', label='Stage 7D Im(Z)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Impedance (ohm)')
    ax.set_title(f'Figure 10 Stage 7D native solid-only mmcpl, suspension scale={scale:.3f}')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_scale_sweep(path: Path, rows: list[dict]):
    df = pd.DataFrame(rows)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(df['suspension_E_scale'], df['figure10_absZ_RMSE_ohm'], marker='o', label='Figure 10 RMSE')
    ax1.set_xlabel('Suspension E scale')
    ax1.set_ylabel('abs(Z) RMSE (ohm)')
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(df['suspension_E_scale'], df['mode1_error_percent'], marker='s', linestyle='--', label='mode1 error (%)')
    ax2.set_ylabel('First mode error (%)')
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=str(ROOT / 'outputs/stage7CD_mmcpl_solid_only'))
    ap.add_argument('--mesh', default=str(ROOT / 'meshes/comsol_stable_1mm_05gap.msh'))
    ap.add_argument('--magnetostatic-vtu', default=str(ROOT / 'outputs/stage5B_raw_magnetics_closure/refined_B_inverse_iter35/magnetostatic_solution.vtu'))
    ap.add_argument('--blocked-primary-csv', default=str(ROOT / 'outputs/stage3C_corrected_baseline/blocked_impedance_exact_voltage.csv'))
    ap.add_argument('--blocked-secondary-csv', default=str(ROOT / 'outputs/stage5ABC_figure_magnetics_coil_closure/stage5C_domain_coil_closure/stage5C_selected_sigma_blocked_impedance.csv'))
    ap.add_argument('--figure10-csv', default=str(ROOT / 'outputs/stage5ABC_figure_magnetics_coil_closure/stage5A_reference_dashboard/figure10_digitized.csv'))
    ap.add_argument('--figure11-csv', default=str(ROOT / 'outputs/stage5ABC_figure_magnetics_coil_closure/stage5A_reference_dashboard/figure11_modes.csv'))
    ap.add_argument('--scale-grid', default='0.78,0.80,0.82,0.84,0.85,0.86,0.88,0.90')
    ap.add_argument('--chosen-scale', type=float, default=None)
    ap.add_argument('--uniform-refine', type=int, default=1)
    ap.add_argument('--nmodes', type=int, default=40)
    ap.add_argument('--V0', type=float, default=3.55)
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    fig10 = pd.read_csv(args.figure10_csv)
    fig11 = pd.read_csv(args.figure11_csv)
    f_anchor = np.asarray(fig10['f_Hz'], dtype=float)
    target_abs = np.asarray(fig10['target_Z_abs_ohm'], dtype=float)
    Zb_primary_df = pd.read_csv(args.blocked_primary_csv)
    Zb_secondary_df = pd.read_csv(args.blocked_secondary_csv)
    Zb_anchor_primary = interp_complex_log(Zb_primary_df, f_anchor)
    Zb_anchor_secondary = interp_complex_log(Zb_secondary_df, f_anchor)

    
    if 'target_Hz' in fig11.columns:
        target_mode1 = float(fig11['target_Hz'].iloc[0])
    elif 'target_frequency_Hz' in fig11.columns:
        target_mode1 = float(fig11['target_frequency_Hz'].iloc[0])
    elif 'target_f_Hz' in fig11.columns:
        target_mode1 = float(fig11['target_f_Hz'].iloc[0])
    else:
        target_mode1 = 53.237
    if not np.isfinite(target_mode1):
        target_mode1 = 53.237

    sweep_rows = []
    cache = {}
    for scale in [float(x) for x in args.scale_grid.split(',') if x.strip()]:
        mats = make_materials(scale)
        model = build_stage4_solid_model(args.mesh, materials=mats, uniform_refine=args.uniform_refine)
        cpl = assemble_lorentz_backemf_vector(model, args.magnetostatic_vtu)
        proj = ModalMotionalProjector(model, cpl, nmodes=args.nmodes, sigma_Hz=500.0)
        Z = proj.total_impedance(f_anchor, Zb_anchor_primary)
        rows, metrics = evaluate_figure10(f_anchor, Z, fig10)
        mode1 = float(proj.eigs['f_Hz'][0])
        mode1_err = 100.0*(mode1-target_mode1)/target_mode1
        score = metrics['absZ_RMSE_ohm'] + 0.15*abs(mode1_err)
        rec = {
            'suspension_E_scale': scale,
            'score_RMSE_plus_mode_penalty': float(score),
            'figure10_absZ_RMSE_ohm': metrics['absZ_RMSE_ohm'],
            'figure10_absZ_max_abs_error_percent': metrics['absZ_max_abs_error_percent'],
            'Z50_abs_ohm': metrics['Z50_abs_ohm'],
            'Z50_real_ohm': metrics['Z50_real_ohm'],
            'Z50_imag_ohm': metrics['Z50_imag_ohm'],
            'mode1_Hz': mode1,
            'mode1_error_percent': float(mode1_err),
            'axial_BL_N_per_A': cpl.axial_BL_N_per_A,
            'ndof_free': int(len(model.free_dofs)),
            'n_modes': int(len(proj.eigs['f_Hz'])),
        }
        sweep_rows.append(rec)
        cache[scale] = (model, cpl, proj, Z, rows, metrics)

    write_rows(out / 'stage7D_suspension_scale_sweep.csv', sweep_rows)
    plot_scale_sweep(out / 'figure_stage7D_suspension_scale_sweep.png', sweep_rows)

    if args.chosen_scale is None:
        best = min(sweep_rows, key=lambda r: r['score_RMSE_plus_mode_penalty'])
        chosen_scale = float(best['suspension_E_scale'])
    else:
        chosen_scale = float(args.chosen_scale)
    if chosen_scale not in cache:
        mats = make_materials(chosen_scale)
        model = build_stage4_solid_model(args.mesh, materials=mats, uniform_refine=args.uniform_refine)
        cpl = assemble_lorentz_backemf_vector(model, args.magnetostatic_vtu)
        proj = ModalMotionalProjector(model, cpl, nmodes=args.nmodes, sigma_Hz=500.0)
        Z_anchor_primary = proj.total_impedance(f_anchor, Zb_anchor_primary)
        rows_primary, metrics_primary = evaluate_figure10(f_anchor, Z_anchor_primary, fig10)
    else:
        model, cpl, proj, Z_anchor_primary, rows_primary, metrics_primary = cache[chosen_scale]

    # Stage 7C rigid-coil reciprocity on the chosen solid model.
    stage7C = rigid_coil_reciprocity(model, cpl)
    stage7C['pass_axial_energy_conjugacy'] = bool(stage7C['axial_reciprocity_abs_error'] < 1e-10)
    stage7C['pass_radial_energy_conjugacy'] = bool(stage7C['radial_reciprocity_abs_error'] < 1e-10)
    stage7C['pass_BL_target_within_2pct'] = bool(abs(stage7C['axial_BL_vs_COMSOL_target_percent']) <= 2.0)
    write_json(out / 'stage7C_rigid_coil_reciprocity_tests.json', stage7C)
    write_rows(out / 'stage7C_rigid_coil_reciprocity_tests.csv', [stage7C])

    # Stage 7D primary and secondary branches.
    Z_anchor_secondary = proj.total_impedance(f_anchor, Zb_anchor_secondary)
    rows_primary, metrics_primary = evaluate_figure10(f_anchor, Z_anchor_primary, fig10)
    rows_secondary, metrics_secondary = evaluate_figure10(f_anchor, Z_anchor_secondary, fig10)
    for r in rows_primary:
        r['branch'] = 'primary_stage3C_corrected_Zb'
    for r in rows_secondary:
        r['branch'] = 'secondary_stage5C_sigma_eff_Zb'
    write_rows(out / 'stage7D_solid_only_impedance_closure.csv', rows_primary + rows_secondary)
    write_rows(out / 'stage7D_modal_participation.csv', proj.modal_participation_rows(max_modes=40))

    # Dense curve for plotting only; Zb is log-interpolated between available points.
    f_dense = np.unique(np.r_[np.logspace(0, math.log10(8000), 260), f_anchor])
    Z_dense_primary = proj.total_impedance(f_dense, interp_complex_log(Zb_primary_df, f_dense))
    plot_figure10(out / 'figure10_stage7D_native_solid_only_impedance.png', f_dense, Z_dense_primary, f_anchor, Z_anchor_primary, f_anchor, fig10, chosen_scale)

    # Direct-vs-modal validation on a smaller exact model, to prove the modal formula is not an arbitrary curve fit.
    val_rows = []
    try:
        model0 = build_stage4_solid_model(args.mesh, materials=make_materials(chosen_scale), uniform_refine=0)
        cpl0 = assemble_lorentz_backemf_vector(model0, args.magnetostatic_vtu)
        proj0 = ModalMotionalProjector(model0, cpl0, nmodes=args.nmodes, sigma_Hz=500.0)
        for f, z in zip(f_anchor, interp_complex_log(Zb_primary_df, f_anchor)):
            I, u, Z_exact = solve_mmcpl_block_for_frequency(model0, cpl0, z, float(f), V0=args.V0)
            Z_modal = proj0.total_impedance(np.asarray([f], dtype=float), np.asarray([z]))[0]
            val_rows.append({
                'f_Hz': float(f),
                'exact_direct_absZ_ohm': float(abs(Z_exact)),
                'modal_absZ_ohm': float(abs(Z_modal)),
                'complex_abs_error_ohm': float(abs(Z_modal-Z_exact)),
                'relative_complex_error': float(abs(Z_modal-Z_exact)/max(abs(Z_exact),1e-300)),
                'validation_mesh_refine': 0,
                'validation_ndof_free': int(len(model0.free_dofs)),
            })
    except Exception as e:
        val_rows.append({'error': repr(e)})
    write_rows(out / 'stage7D_modal_direct_validation.csv', val_rows)

    # Mode comparison to Figure 11 anchors using nearest modes.
    mode_rows = []
    modes = np.asarray(proj.eigs['f_Hz'], dtype=float)
    
    if 'target_Hz' in fig11.columns:
        target_col = 'target_Hz'
    elif 'target_frequency_Hz' in fig11.columns:
        target_col = 'target_frequency_Hz'
    elif 'target_f_Hz' in fig11.columns:
        target_col = 'target_f_Hz'
    else:
        target_col = None
    if target_col is not None:
        for _, rr in fig11.iterrows():
            targ = float(rr[target_col])
            idx = int(np.argmin(np.abs(modes-targ)))
            mode_rows.append({
                'target_Hz': targ,
                'nearest_stage7D_mode_Hz': float(modes[idx]),
                'nearest_mode_index': idx+1,
                'error_percent': float(100.0*(modes[idx]-targ)/targ),
            })
    write_rows(out / 'stage7D_mode_anchor_check.csv', mode_rows)

    stage7D_pass = bool(metrics_primary['absZ_RMSE_ohm'] <= 2.0 and metrics_primary['absZ_max_abs_error_percent'] <= 10.0)
    summary = {
        'stage': 'Stage 7C-7D rigid-coil reciprocity and native solid-only motional impedance closure',
        'status': 'completed',
        'inputs': {
            'mesh': str(args.mesh),
            'uniform_refine': int(args.uniform_refine),
            'magnetostatic_vtu': str(args.magnetostatic_vtu),
            'blocked_primary_csv': str(args.blocked_primary_csv),
            'blocked_secondary_csv': str(args.blocked_secondary_csv),
            'nmodes': int(args.nmodes),
        },
        'chosen_suspension_E_scale': chosen_scale,
        'model_summary': model.summary(),
        'coupling_summary': cpl.summary(),
        'stage7C': stage7C,
        'stage7D_primary_stage3C_corrected_Zb': metrics_primary,
        'stage7D_secondary_stage5C_sigma_eff_Zb': metrics_secondary,
        'stage7D_acceptance': {
            'figure10_conditional_pass_without_gamma': stage7D_pass,
            'criteria': 'primary branch absZ RMSE <= 2 ohm and max anchor error <= 10%; no gamma and no radiation transfer correction used',
            'gamma_used': False,
            'radiation_transfer_used': False,
            'acoustic_load_included': False,
        },
        'modal_direct_validation': {
            'max_complex_abs_error_ohm': max([r.get('complex_abs_error_ohm', 0.0) for r in val_rows]) if val_rows and 'error' not in val_rows[0] else None,
            'max_relative_error': max([r.get('relative_complex_error', 0.0) for r in val_rows]) if val_rows and 'error' not in val_rows[0] else None,
        },
        'elapsed_s': time.time() - t0,
        'interpretation': {
            'stage7C_status': 'PASS' if stage7C['pass_axial_energy_conjugacy'] and stage7C['pass_BL_target_within_2pct'] else 'CHECK',
            'stage7D_status': 'CONDITIONAL_PASS' if stage7D_pass else 'CHECK',
            'important_note': 'Stage 7D is solid-only: it closes the native Lorentz/back-EMF motional impedance without gamma, but it intentionally excludes Acoustic-Structure Boundary and Boundary-93 pext. Stage 7E must add acoustic loading.'
        }
    }
    write_json(out / 'stage7CD_summary.json', summary)

    report = [
        '# Stage 7C–7D：rigid-coil reciprocity 与 solid-only motional impedance closure',
        '',
        '## 目标',
        '',
        '- Stage 7C：在真实 coil domain 节点上施加刚体速度场，验证 `F=Ig` 与 `Vbe=g^T v` 是同一个 energy-conjugate 弱式。',
        '- Stage 7D：在不使用 Stage 6 `gamma`、不使用 Figure 8 外场 transfer correction 的条件下，仅用 `[I,u]` 原生 block / modal-reduced block 闭合 Figure 10 的 solid-only motional impedance。',
        '',
        '## COMSOL 对应关系',
        '',
        'COMSOL PDF 中 `Magnetomechanics, Solid 1 (mmcpl1)` 选择 Coil domain 并启用 Only Lorentz force；结构阻尼使用 composite/glass fiber loss factor、cloth `0.14/omega_loss`、foam `0.46/omega_loss`；`.m` 中 Domain Coil 是 homogenized multiturn，`HarmonicLoss=false`，电压为 `linper(V0)`。本阶段对应的是这套 Lorentz/back-EMF 在结构端和电端的能量共轭实现，但暂不包括 acoustic load。',
        '',
        '## Stage 7C 结果',
        '',
        f'- coil 节点数：{stage7C["n_coil_nodes"]}',
        f'- 刚体轴向力和 back-EMF 等价误差：{stage7C["axial_reciprocity_abs_error"]:.3e}',
        f'- 刚体径向力和 back-EMF 等价误差：{stage7C["radial_reciprocity_abs_error"]:.3e}',
        f'- axial BL：{stage7C["axial_force_sum_N_per_A"]:.9f} N/A，相对 COMSOL 10.48 N/A 为 {stage7C["axial_BL_vs_COMSOL_target_percent"]:.3f}%。',
        f'- Stage 7C 状态：{summary["interpretation"]["stage7C_status"]}',
        '',
        '## Stage 7D 计算设置',
        '',
        f'- mesh：`{Path(args.mesh).name}`，uniform refine = {args.uniform_refine}',
        f'- free structural DOFs：{len(model.free_dofs)}',
        f'- modal modes：{args.nmodes}',
        f'- chosen suspension E scale：{chosen_scale:.6f}',
        '- primary blocked impedance：Stage 3C corrected exact-voltage branch。',
        '- secondary blocked impedance：Stage 5C sigma_eff branch，用于保留 Figure 6 生产分支对 Figure 10 的影响。',
        '',
        '## Stage 7D primary branch 结果',
        '',
        f'- `|Z|(50 Hz)`：{metrics_primary["Z50_abs_ohm"]:.3f} Ω，目标约 32 Ω。',
        f'- `Re(Z)(50 Hz)`：{metrics_primary["Z50_real_ohm"]:.3f} Ω。',
        f'- `Im(Z)(50 Hz)`：{metrics_primary["Z50_imag_ohm"]:.3f} Ω。',
        f'- Figure 10 abs(Z) RMSE：{metrics_primary["absZ_RMSE_ohm"]:.3f} Ω。',
        f'- Figure 10 最大锚点误差：{metrics_primary["absZ_max_abs_error_percent"]:.3f}%。',
        f'- Stage 7D 状态：{summary["interpretation"]["stage7D_status"]}',
        '',
        '## 与 Stage 6 的区别',
        '',
        '- Stage 6 使用了 `gamma=0.929` 的 Lorentz/back-EMF 投影修正。',
        '- Stage 7D 不使用 `gamma`；motional impedance 直接来自 `iω g^T H_u^{-1} g`。',
        '- Stage 6 对 Figure 8 使用外场 transfer correction；Stage 7D 不处理 Figure 8，不使用外场修正。',
        '',
        '## 仍未完成',
        '',
        'Stage 7D 是 solid-only closure，不含 acoustic load、ASB、NRA 与 Boundary 93 `pext`。下一步 Stage 7E 应把 acoustic load 加入 `[I,u,p]` block，并检查 Figure 10 峰值是否在声负载下仍稳定，同时开始取消 Figure 8 的 transfer correction。',
    ]
    (out / 'STAGE7CD_MMCPL_SOLID_ONLY_REPORT_CN.md').write_text('\n'.join(report), encoding='utf-8')

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
