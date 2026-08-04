#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, sys, time, shutil
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import splu, spsolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from loudspeaker_axisym_fem.axisym_magnetics import load_tagged_meshio
from loudspeaker_axisym_fem.stage4_solid_fem import (
    build_stage4_solid_model, default_stage4_materials, SolidMaterial,
)
from loudspeaker_axisym_fem.stage4C_acoustic_structure import (
    build_stage4C_acoustic_structure_model, Stage4CParameters,
    _complex_solid_stiffness, _acoustic_matrix,
)
from loudspeaker_axisym_fem.stage4F_hk_refinement import hk_axis_and_power_recovered
from loudspeaker_axisym_fem.mmcpl_lorentz_backemf import assemble_lorentz_backemf_vector


def write_json(path: Path, data: dict, indent: int = 2):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding='utf-8')


def write_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    keys=[]
    for r in rows:
        for k in r.keys():
            if k not in keys: keys.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def make_materials(suspension_E_scale: float):
    mats = default_stage4_materials(2.0 * math.pi * 40.0)
    for d in (20, 25):
        m = mats[d]
        mats[d] = SolidMaterial(m.E * suspension_E_scale, m.nu, m.rho, m.loss_factor, m.beta_dK, m.label)
    return mats


def interp_complex_log(df: pd.DataFrame, freqs: np.ndarray) -> np.ndarray:
    f0=np.asarray(df['f_Hz'], dtype=float)
    # support several naming conventions
    if {'Zb_real_ohm','Zb_imag_ohm'}.issubset(df.columns):
        z=np.asarray(df['Zb_real_ohm'], dtype=float)+1j*np.asarray(df['Zb_imag_ohm'], dtype=float)
    elif {'Z_real_ohm','Z_imag_ohm'}.issubset(df.columns):
        z=np.asarray(df['Z_real_ohm'], dtype=float)+1j*np.asarray(df['Z_imag_ohm'], dtype=float)
    else:
        raise KeyError('blocked impedance CSV must contain Zb_real_ohm/Zb_imag_ohm')
    lf=np.log10(np.asarray(freqs, dtype=float))
    return np.interp(lf, np.log10(f0), z.real)+1j*np.interp(lf, np.log10(f0), z.imag)


def parse_freqs(spec: str, fig8: pd.DataFrame, fig10: pd.DataFrame) -> np.ndarray:
    if spec == 'figure8_10':
        vals=np.r_[np.asarray(fig8['f_Hz'], dtype=float), np.asarray(fig10['f_Hz'], dtype=float), [600.0, 630.0, 1300.0]]
        return np.asarray(sorted(set(float(x) for x in vals)), dtype=float)
    if spec == 'figure10':
        return np.asarray(sorted(set(float(x) for x in fig10['f_Hz'])), dtype=float)
    if spec == 'figure8':
        return np.asarray(sorted(set(float(x) for x in fig8['f_Hz'])), dtype=float)
    vals=[]
    for p in spec.replace(';', ',').split(','):
        p=p.strip()
        if p: vals.append(float(p))
    return np.asarray(sorted(set(vals)), dtype=float)


def build_stage7e_model(mesh_path: str, mphtxt_path: str, suspension_E_scale: float):
    mesh = load_tagged_meshio(mesh_path)
    # Stage4C constructor creates acoustic matrices, ASB matrix, and a default solid.
    model = build_stage4C_acoustic_structure_model(mesh, mphtxt_path, solid_uniform_refine=0)
    scaled = build_stage4_solid_model(mesh, materials=make_materials(suspension_E_scale), uniform_refine=0)
    if model.solid.ndof != scaled.ndof or not np.array_equal(model.solid.global_node_ids, scaled.global_node_ids):
        raise RuntimeError('scaled solid node ordering differs from acoustic-structure model')
    model.solid = scaled
    return model


def solve_native_frequency(model, cpl, Zb: complex, freq_Hz: float, params: Stage4CParameters, *, nra_enabled: bool = True):
    """Solve Stage-7E native minimal Study-2 block with x=[I,u_free,p_free].

    Electrical:  Zb I + i*w*g^T u = V0
    Solid:      -g I + Hs u - G p = 0
    Acoustic:    -rho*w^2 G^T u + Ap p = 0
    """
    f=float(freq_Hz); w=2.0*math.pi*f
    solid=model.solid; sf=solid.free_dofs; pf=model.pressure_free_dofs
    gf=np.asarray(cpl.g_free_N_per_A, dtype=complex)
    Hs=(_complex_solid_stiffness(solid, w)[sf][:, sf].astype(complex) - (w*w)*solid.M[sf][:, sf].astype(complex)).tocsr()
    Ap=_acoustic_matrix(model, w, rho0=params.rho0_kg_m3, c0=params.c0_m_s, nra_enabled=nra_enabled)[pf][:, pf].astype(complex).tocsr()
    Gsf=model.G_sp[sf][:, pf].astype(complex).tocsr()
    GT=model.G_sp.T[pf][:, sf].astype(complex).tocsr()
    A00=csr_matrix([[complex(Zb)]])
    A01=csr_matrix((1j*w*gf.reshape(1, -1)))
    A02=csr_matrix((1, len(pf)), dtype=complex)
    A10=csr_matrix((-gf.reshape(-1, 1)))
    A20=csr_matrix((len(pf), 1), dtype=complex)
    Ablock=bmat([
        [A00, A01, A02],
        [A10, Hs, -Gsf],
        [A20, -params.rho0_kg_m3*w*w*GT, Ap],
    ], format='csc')
    rhs=np.zeros(Ablock.shape[0], dtype=complex); rhs[0]=params.V0_peak_V
    lu=splu(Ablock)
    sol=lu.solve(rhs)
    I=sol[0]
    us=sol[1:1+len(sf)]
    pp=sol[1+len(sf):]
    ufull=np.zeros(solid.ndof, dtype=complex); ufull[sf]=us
    pfull=np.zeros(len(model.acoustic_nodes_global), dtype=complex); pfull[pf]=pp
    Ztot=params.V0_peak_V/I
    Vbe=1j*w*np.dot(gf, us)
    motional=Ztot - Zb
    return {
        'f_Hz': f,
        'I_A_peak': I,
        'Z_total_ohm': Ztot,
        'Zb_ohm': complex(Zb),
        'Z_motional_ohm': motional,
        'V_backemf_V_peak': Vbe,
        'u_full_m': ufull,
        'p_full_Pa': pfull,
        'n_unknowns': int(Ablock.shape[0]),
        'nnz': int(Ablock.nnz),
    }


def solve_native_sweep(model, cpl, freqs: np.ndarray, Zb: np.ndarray, params: Stage4CParameters, *, nra_enabled: bool = True, outdir: Path | None = None, branch: str = 'with_NRA'):
    results=[]
    for f,z in zip(freqs, Zb):
        ck = (outdir / f'checkpoint_{branch}_{float(f):09.3f}Hz.npz') if outdir is not None else None
        if ck is not None and ck.exists():
            data=np.load(ck, allow_pickle=False)
            r={
                'f_Hz': float(data['f_Hz']),
                'I_A_peak': complex(data['I_A_peak']),
                'Z_total_ohm': complex(data['Z_total_ohm']),
                'Zb_ohm': complex(data['Zb_ohm']),
                'Z_motional_ohm': complex(data['Z_motional_ohm']),
                'V_backemf_V_peak': complex(data['V_backemf_V_peak']),
                'u_full_m': data['u_full_m'],
                'p_full_Pa': data['p_full_Pa'],
                'n_unknowns': int(1 + len(model.solid.free_dofs) + len(model.pressure_free_dofs)),
                'nnz': -1,
            }
        else:
            r=solve_native_frequency(model, cpl, z, float(f), params, nra_enabled=nra_enabled)
            if ck is not None:
                np.savez_compressed(ck,
                    f_Hz=float(f), I_A_peak=r['I_A_peak'], Z_total_ohm=r['Z_total_ohm'], Zb_ohm=r['Zb_ohm'],
                    Z_motional_ohm=r['Z_motional_ohm'], V_backemf_V_peak=r['V_backemf_V_peak'],
                    u_full_m=r['u_full_m'], p_full_Pa=r['p_full_Pa'])
        results.append(r)
    # arrange as result dict compatible with HK postprocessing
    f=np.asarray([r['f_Hz'] for r in results], dtype=float)
    Z=np.asarray([r['Z_total_ohm'] for r in results], dtype=complex)
    I=np.asarray([r['I_A_peak'] for r in results], dtype=complex)
    Zb_arr=np.asarray([r['Zb_ohm'] for r in results], dtype=complex)
    p=np.vstack([r['p_full_Pa'] for r in results])
    u=np.vstack([r['u_full_m'] for r in results])
    # A kinematic scalar for diagnostics: back-EMF divided by g norm is not a real displacement; keep Vbe/I form.
    omega=2*np.pi*f
    Vbe=np.asarray([r['V_backemf_V_peak'] for r in results], dtype=complex)
    coil_power=0.5*np.real(params.V0_peak_V*np.conj(I))
    return {
        'f_Hz': f,
        'Z_total_ohm': Z,
        'Zb_ohm': Zb_arr,
        'Z_motional_ohm': Z-Zb_arr,
        'I_A_peak': I,
        'V_backemf_V_peak': Vbe,
        'solid_displacement_m': u,
        'acoustic_pressure_field_Pa': p,
        'coil_power_W': coil_power,
        'n_unknowns': np.asarray([r['n_unknowns'] for r in results], dtype=int),
        'nnz': np.asarray([r['nnz'] for r in results], dtype=int),
    }


def add_hk_metrics(result: dict, model, params: Stage4CParameters) -> dict:
    hk=hk_axis_and_power_recovered(result, model, params, nphi_axis=24, mirror=True)
    out=dict(result)
    out['p_1m_Pa_peak']=hk['p_1m_hk_recovered_Pa_peak']
    out['SPL_1m_dB']=hk['SPL_1m_hk_recovered_dB']
    out['phase_deg']=hk['phase_hk_recovered_deg']
    out['acoustic_power_W']=np.maximum(hk['hk_recovered_halfspace_power_W'], 0.0)
    out['hk_flux_raw_W']=hk['hk_recovered_flux_raw_W']
    out['acoustic_efficiency_percent']=100.0*out['acoustic_power_W']/np.maximum(out['coil_power_W'], 1e-300)
    out['hk_boundary_info']=hk['hk_recovered_boundary_info']
    return out


def rows_response(res: dict, branch: str) -> list[dict]:
    rows=[]
    for i,f in enumerate(res['f_Hz']):
        Z=res['Z_total_ohm'][i]; Zb=res['Zb_ohm'][i]; Zm=res['Z_motional_ohm'][i]
        rows.append({
            'branch': branch,
            'f_Hz': float(f),
            'SPL_1m_hk_recovered_dB': float(res.get('SPL_1m_dB', np.full_like(res['f_Hz'], np.nan))[i]),
            'phase_hk_recovered_deg': float(res.get('phase_deg', np.full_like(res['f_Hz'], np.nan))[i]),
            'Z_abs_ohm': float(abs(Z)), 'Z_real_ohm': float(Z.real), 'Z_imag_ohm': float(Z.imag),
            'Zb_abs_ohm': float(abs(Zb)), 'Zb_real_ohm': float(Zb.real), 'Zb_imag_ohm': float(Zb.imag),
            'Z_motional_abs_ohm': float(abs(Zm)), 'Z_motional_real_ohm': float(Zm.real), 'Z_motional_imag_ohm': float(Zm.imag),
            'I_abs_A_peak': float(abs(res['I_A_peak'][i])),
            'Vbe_abs_V_peak': float(abs(res['V_backemf_V_peak'][i])),
            'coil_power_W': float(res['coil_power_W'][i]),
            'acoustic_power_W': float(res.get('acoustic_power_W', np.full_like(res['f_Hz'], np.nan))[i]),
            'acoustic_efficiency_percent': float(res.get('acoustic_efficiency_percent', np.full_like(res['f_Hz'], np.nan))[i]),
            'n_unknowns': int(res['n_unknowns'][i]),
            'matrix_nnz': int(res['nnz'][i]),
        })
    return rows


def compare_figure10(res: dict, fig10: pd.DataFrame) -> tuple[list[dict], dict]:
    rows=[]
    for _,rr in fig10.iterrows():
        f=float(rr['f_Hz']); idx=int(np.argmin(np.abs(res['f_Hz']-f)))
        Z=res['Z_total_ohm'][idx]
        targ=float(rr['target_Z_abs_ohm'])
        err=abs(Z)-targ
        rows.append({
            'figure':'Figure 10','f_Hz':f,'target_absZ_ohm':targ,
            'stage7E_absZ_ohm':float(abs(Z)), 'stage7E_realZ_ohm':float(Z.real), 'stage7E_imagZ_ohm':float(Z.imag),
            'absZ_error_ohm':float(err), 'absZ_error_percent':float(100*err/targ),
            'target_realZ_ohm':float(rr['target_Z_real_ohm']), 'target_imagZ_ohm':float(rr['target_Z_imag_ohm']),
        })
    errs=np.asarray([r['absZ_error_ohm'] for r in rows], dtype=float)
    per=np.asarray([r['absZ_error_percent'] for r in rows], dtype=float)
    return rows, {'absZ_RMSE_ohm':float(np.sqrt(np.mean(errs*errs))), 'absZ_max_abs_error_percent':float(np.max(np.abs(per)))}


def compare_figure8(res: dict, fig8: pd.DataFrame) -> tuple[list[dict], dict]:
    rows=[]
    for _,rr in fig8.iterrows():
        f=float(rr['f_Hz']); idx=int(np.argmin(np.abs(res['f_Hz']-f)))
        spl=float(res['SPL_1m_dB'][idx]); targ=float(rr['target_SPL_dB']); err=spl-targ
        rows.append({'figure':'Figure 8','f_Hz':f,'target_SPL_dB':targ,'stage7E_SPL_dB':spl,'SPL_error_dB':err})
    errs=np.asarray([r['SPL_error_dB'] for r in rows], dtype=float)
    return rows, {'SPL_RMSE_dB':float(np.sqrt(np.mean(errs*errs))), 'SPL_max_abs_error_dB':float(np.max(np.abs(errs))), 'SPL_mean_error_dB':float(np.mean(errs))}


def plot_figure10(path: Path, res: dict, fig10: pd.DataFrame, title: str):
    f=res['f_Hz']; Z=res['Z_total_ohm']
    fig,ax=plt.subplots(figsize=(8,5))
    ax.semilogx(f, np.abs(Z), '-o', ms=3, label='Stage 7E native [I,u,p] |Z|')
    ax.semilogx(f, Z.real, '--', label='Stage 7E Re(Z)')
    ax.semilogx(f, Z.imag, ':', label='Stage 7E Im(Z)')
    ax.semilogx(fig10['f_Hz'], fig10['target_Z_abs_ohm'], 's', label='PDF Figure 10 anchors')
    ax.axhline(5.6, lw=0.8, ls=':', label='DC 5.6 Ω')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Impedance (ohm)'); ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_figure8(path: Path, res_with: dict, res_without: dict | None, fig8: pd.DataFrame, title: str):
    fig,ax=plt.subplots(figsize=(8,5))
    ax.semilogx(res_with['f_Hz'], res_with['SPL_1m_dB'], '-o', ms=3, label='Stage 7E native ASB + recovered HK, NRA on')
    if res_without is not None:
        ax.semilogx(res_without['f_Hz'], res_without['SPL_1m_dB'], '--x', ms=3, label='Stage 7E native ASB + HK, NRA off')
    ax.semilogx(fig8['f_Hz'], fig8['target_SPL_dB'], 's', label='PDF Figure 8 anchors')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('SPL at 1 m (dB)'); ax.set_title(title)
    ax.set_ylim(40,110); ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description='Stage 7E native minimal COMSOL Study-2 [I,u,p] block: no gamma, no Figure-8 transfer correction.')
    ap.add_argument('--outdir', default=str(ROOT/'outputs/stage7E_native_study2_asb'))
    ap.add_argument('--mesh', default=str(ROOT/'meshes/comsol_geometry_polyline_coarse_2p5mm.msh'))
    ap.add_argument('--mphtxt', default=str(ROOT/'comsol_reference_inputs/Untitled.mphtxt'))
    ap.add_argument('--magnetostatic-vtu', default=str(ROOT/'outputs/stage5B_raw_magnetics_closure/refined_B_inverse_iter35/magnetostatic_solution.vtu'))
    ap.add_argument('--blocked-csv', default=str(ROOT/'outputs/stage3C_corrected_baseline/blocked_impedance_exact_voltage.csv'))
    ap.add_argument('--figure8-csv', default=str(ROOT/'outputs/stage5ABC_figure_magnetics_coil_closure/stage5A_reference_dashboard/figure8_digitized.csv'))
    ap.add_argument('--figure10-csv', default=str(ROOT/'outputs/stage5ABC_figure_magnetics_coil_closure/stage5A_reference_dashboard/figure10_digitized.csv'))
    ap.add_argument('--freqs', default='figure8_10')
    ap.add_argument('--suspension-E-scale', type=float, default=0.82)
    ap.add_argument('--V0-peak', type=float, default=3.55)
    ap.add_argument('--run-no-nra', action='store_true')
    ap.add_argument('--no-plots', action='store_true')
    args=ap.parse_args()
    t0=time.time(); out=Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    fig8=pd.read_csv(args.figure8_csv); fig10=pd.read_csv(args.figure10_csv)
    freqs=parse_freqs(args.freqs, fig8, fig10)
    Zb=interp_complex_log(pd.read_csv(args.blocked_csv), freqs)
    params=Stage4CParameters(V0_peak_V=args.V0_peak)
    model=build_stage7e_model(args.mesh, args.mphtxt, args.suspension_E_scale)
    cpl=assemble_lorentz_backemf_vector(model.solid, args.magnetostatic_vtu)
    res_with=solve_native_sweep(model, cpl, freqs, Zb, params, nra_enabled=True, outdir=out, branch='with_NRA')
    res_with=add_hk_metrics(res_with, model, params)
    res_without=None
    if args.run_no_nra:
        res_without=solve_native_sweep(model, cpl, freqs, Zb, params, nra_enabled=False, outdir=out, branch='without_NRA')
        res_without=add_hk_metrics(res_without, model, params)
    write_rows(out/'stage7E_native_study2_response.csv', rows_response(res_with, 'with_NRA') + (rows_response(res_without, 'without_NRA') if res_without else []))
    f10_rows, f10_metrics=compare_figure10(res_with, fig10)
    f8_rows, f8_metrics=compare_figure8(res_with, fig8)
    write_rows(out/'stage7E_figure10_dashboard.csv', f10_rows)
    write_rows(out/'stage7E_figure8_dashboard.csv', f8_rows)
    if res_without is not None:
        nra_rows=[]
        for f in [600.0,630.0,1300.0]:
            iw=int(np.argmin(np.abs(res_with['f_Hz']-f))); io=int(np.argmin(np.abs(res_without['f_Hz']-f)))
            nra_rows.append({'f_Hz':f,'SPL_with_NRA_dB':float(res_with['SPL_1m_dB'][iw]),'SPL_without_NRA_dB':float(res_without['SPL_1m_dB'][io]),'with_minus_without_dB':float(res_with['SPL_1m_dB'][iw]-res_without['SPL_1m_dB'][io])})
        write_rows(out/'stage7E_NRA_on_off_delta.csv', nra_rows)
    else:
        nra_rows=[]
    if not args.no_plots:
        plot_figure10(out/'figure10_stage7E_native_asb_impedance.png', res_with, fig10, 'Figure 10 Stage 7E native [I,u,p] impedance')
        plot_figure8(out/'figure8_stage7E_native_asb_sensitivity.png', res_with, res_without, fig8, 'Figure 8 Stage 7E native ASB + Boundary 93 recovered HK')
    summary={
        'stage':'Stage 7E native minimal COMSOL Study-2 ASB block',
        'status':'completed',
        'meaning':'Solves x=[I,u_free,p_free] with energy-conjugate g, ASB acoustic load, blocked Zb, and Boundary-93 recovered HK. No gamma and no Figure-8 transfer correction.',
        'inputs':{'mesh':str(args.mesh),'blocked_csv':str(args.blocked_csv),'magnetostatic_vtu':str(args.magnetostatic_vtu),'suspension_E_scale':float(args.suspension_E_scale),'frequencies_Hz':[float(x) for x in freqs]},
        'model_summary':model.summary(),
        'coupling_summary':cpl.summary(),
        'figure10_metrics':f10_metrics,
        'figure8_metrics':f8_metrics,
        'NRA_on_off':nra_rows,
        'acceptance':{
            'gamma_used':False,
            'figure8_transfer_correction_used':False,
            'native_ASB_acoustic_load_included':True,
            'figure10_conditional_pass': bool(f10_metrics['absZ_RMSE_ohm'] <= 3.0 and f10_metrics['absZ_max_abs_error_percent'] <= 20.0),
            'figure8_status':'diagnostic_only_until_Boundary93_native_pext_is_further_closed',
        },
        'elapsed_s':time.time()-t0,
    }
    write_json(out/'stage7E_summary.json', summary)
    report=[
        '# Stage 7E：native minimal COMSOL Study-2 `[I,u,p]` ASB closure', '',
        '## 本阶段解决的问题', '',
        'Stage 7D 已经用 `iω g^T H_u^{-1} g` 在 solid-only 层面闭合 Figure 10，但还没有把 Acoustic–Structure Boundary 的声负载放入同一个矩阵。Stage 7E 新增原生最小 Study-2 block：', '',
        '```text',
        '[ Zb        iω gᵀ            0 ] [I]   [V0]',
        '[ -g        Hs          -G_sp ] [u] = [0 ]',
        '[ 0   -ρ0ω² G_spᵀ       Hp  ] [p]   [0 ]',
        '```', '',
        '其中 `g` 是 Stage 7A 的同一个 energy-conjugate Lorentz/back-EMF 向量；本阶段不使用 Stage 6 的 `gamma`，也不使用 Figure 8 transfer correction。', '',
        '## 计算设置', '',
        f'- mesh：`{Path(args.mesh).name}`',
        f'- suspension E scale：{args.suspension_E_scale}',
        f'- 频点：{", ".join(str(float(x)) for x in freqs)} Hz',
        f'- matrix unknowns：{int(res_with["n_unknowns"][0])}',
        f'- matrix nnz：{int(res_with["nnz"][0])}',
        f'- axial BL from g：{cpl.axial_BL_N_per_A:.9f} N/A', '',
        '## Figure 10 阻抗结果', '',
        f'- abs(Z) RMSE：{f10_metrics["absZ_RMSE_ohm"]:.3f} Ω',
        f'- max abs anchor error：{f10_metrics["absZ_max_abs_error_percent"]:.3f} %', '',
        '## Figure 8 灵敏度结果', '',
        f'- SPL RMSE：{f8_metrics["SPL_RMSE_dB"]:.3f} dB',
        f'- SPL max abs error：{f8_metrics["SPL_max_abs_error_dB"]:.3f} dB',
        f'- SPL mean error：{f8_metrics["SPL_mean_error_dB"]:.3f} dB', '',
        '## 判断', '',
        '- Stage 7E 完成了比 Stage 7D 更深入的原生 `[I,u,p]` block：声负载现在直接进入 solid 方程，而不是后处理。',
        '- Figure 10 的 50 Hz 峰需要和 Stage 7D 对比：若声负载引入后峰值稳定，说明 `mmcpl` 弱式已经能进入完整 Study 2；若峰值漂移，优先查 ASB 符号、声压载荷方向和 acoustic damping。',
        '- Figure 8 现在仍是诊断指标。只要误差仍大，就说明 Boundary 93 `pext`/HK、P1 单元或几何重建还没有和 COMSOL 完全等价；但本阶段已经消除了 `gamma` 和 transfer correction 两个人工闭合项。',
    ]
    (out/'STAGE7E_NATIVE_STUDY2_ASB_REPORT_CN.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__=='__main__':
    main()
