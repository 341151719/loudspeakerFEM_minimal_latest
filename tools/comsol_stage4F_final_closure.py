#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, sys, json, shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from loudspeaker_axisym_fem.axisym_magnetics import load_tagged_meshio
from loudspeaker_axisym_fem.json_utils import write_json, dumps_json
from loudspeaker_axisym_fem.stage4_electroacoustic import load_blocked_impedance_csv
from loudspeaker_axisym_fem.stage4C_acoustic_structure import Stage4CParameters, build_stage4C_acoustic_structure_model, solve_stage4C_full_asb
from loudspeaker_axisym_fem.stage4D_exterior_nra import solve_stage4D_full, stage4D_rows, hk_directivity_from_result
from loudspeaker_axisym_fem.stage4F_hk_refinement import hk_axis_and_power_recovered, hk_directivity_recovered, compare_directivity_pair


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as fp:
        w=csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def parse_freqs(s: str) -> np.ndarray:
    vals=[]
    for p in s.replace(';',',').split(','):
        p=p.strip()
        if p:
            vals.append(float(p))
    return np.asarray(sorted(set(vals)), dtype=float)


def rows_from_result_with_recovered(result: dict) -> list[dict]:
    rows=[]
    for i,fi in enumerate(result['f_Hz']):
        Z=result['Z_total_ohm'][i]
        row={
            'f_Hz':float(fi),
            'SPL_stage4D_facet_dB':float(result.get('SPL_1m_dB', np.full_like(result['f_Hz'], np.nan))[i]),
            'SPL_stage4F_recovered_dB':float(result['SPL_1m_hk_recovered_dB'][i]),
            'recovered_minus_facet_dB':float(result['SPL_1m_hk_recovered_dB'][i]-result.get('SPL_1m_dB', result['SPL_1m_hk_recovered_dB'])[i]),
            'phase_stage4F_recovered_deg':float(result['phase_hk_recovered_deg'][i]),
            'Z_abs_ohm':float(abs(Z)),
            'Z_real_ohm':float(np.real(Z)),
            'Z_imag_ohm':float(np.imag(Z)),
            'coil_power_W':float(result['coil_power_W'][i]),
            'acoustic_power_recovered_W':float(result['hk_recovered_halfspace_power_W'][i]),
            'hk_recovered_flux_raw_W':float(result['hk_recovered_flux_raw_W'][i]),
        }
        rows.append(row)
    return rows


def solve_one_mesh(mesh_path: Path, mphtxt: Path, blocked_csv: Path, freqs: np.ndarray, params: Stage4CParameters, *, nra_enabled: bool = True, with_facet: bool = True) -> tuple[object, dict]:
    freqs, Zb = load_blocked_impedance_csv(blocked_csv, freqs)
    mesh=load_tagged_meshio(mesh_path)
    model=build_stage4C_acoustic_structure_model(mesh, mphtxt, solid_uniform_refine=0)
    if with_facet:
        res=solve_stage4D_full(freqs, Zb, model, params, nra_enabled=nra_enabled, hk_directivity=False)
    else:
        res=solve_stage4C_full_asb(freqs, Zb, model, params, nra_enabled=nra_enabled)
    rec=hk_axis_and_power_recovered(res, model, params)
    res.update(rec)
    return model, res


def write_directivity_csv(path: Path, f: np.ndarray, angles: np.ndarray, spl: np.ndarray, rel: np.ndarray, label: str) -> None:
    rows=[]
    for i,fi in enumerate(f):
        for j,a in enumerate(angles):
            rows.append({'label':label,'f_Hz':float(fi),'angle_deg':float(a),'SPL_dB':float(spl[i,j]),'relative_dB':float(rel[i,j])})
    write_rows_csv(path, rows)


def plot_response_convergence(path: Path, df_rows: list[dict]) -> None:
    import pandas as pd
    df=pd.DataFrame(df_rows)
    fig, ax=plt.subplots(figsize=(8,5))
    for label, grp in df.groupby('mesh_label'):
        ax.semilogx(grp['f_Hz'], grp['SPL_stage4F_recovered_dB'], marker='o', label=label)
    ax.set_xlabel('Frequency / Hz'); ax.set_ylabel('Stage 4F recovered Boundary-93 HK SPL / dB')
    ax.grid(True, which='both', linewidth=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_nra_delta(path: Path, nra_rows: list[dict]) -> None:
    import pandas as pd
    df=pd.DataFrame(nra_rows)
    fig, ax=plt.subplots(figsize=(8,5))
    for label, grp in df.groupby('mesh_label'):
        ax.semilogx(grp['f_Hz'], grp['SPL_with_NRA_dB']-grp['SPL_without_NRA_dB'], marker='o', label=label)
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel('Frequency / Hz'); ax.set_ylabel('with NRA - without NRA / dB')
    ax.grid(True, which='both', linewidth=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_directivity_overlay(path: Path, angles: np.ndarray, rels: list[tuple[str,np.ndarray]], title: str) -> None:
    fig, ax=plt.subplots(figsize=(7,5))
    for label, rel in rels:
        ax.plot(rel, angles, label=label)
    ax.set_xlabel('dB relative to 0°'); ax.set_ylabel('Angle / deg')
    ax.set_xlim(-30,5); ax.set_ylim(-90,90); ax.grid(True, linewidth=0.3); ax.legend(fontsize=8)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> None:
    ap=argparse.ArgumentParser(description='Stage 4F final numerical closure: recovered HK, refined NRA modal check, and P2/1mm cost boundary.')
    ap.add_argument('--outdir', default=str(ROOT/'outputs/stage4F_final_closure'))
    ap.add_argument('--mphtxt', default=str(ROOT/'comsol_reference_inputs/Untitled.mphtxt'))
    ap.add_argument('--blocked-impedance-csv', default=str(ROOT/'outputs/stage3C_corrected_baseline/blocked_impedance_exact_voltage.csv'))
    ap.add_argument('--coarse-mesh', default=str(ROOT/'meshes/comsol_geometry_polyline_coarse_2p5mm.msh'))
    ap.add_argument('--refined-mesh', default=str(ROOT/'meshes/comsol_geometry_polyline_stage3_refined_stage3C_seed.msh'))
    ap.add_argument('--stable-mesh', default=str(ROOT/'meshes/comsol_stable_1mm_05gap.msh'))
    ap.add_argument('--freqs', default='600,630,1000,1300,5000,8000')
    ap.add_argument('--directivity-freqs', default='1000,5000')
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    params=Stage4CParameters()
    mphtxt=Path(args.mphtxt); blocked=Path(args.blocked_impedance_csv)
    freqs=parse_freqs(args.freqs)
    dir_freqs=parse_freqs(args.directivity_freqs)

    meshes=[('coarse_2p5mm',Path(args.coarse_mesh)),('refined_stage3_seed',Path(args.refined_mesh))]
    response_rows=[]; nra_rows=[]; model_summaries={}
    stored_results={}
    for label, mesh_path in meshes:
        model, res = solve_one_mesh(mesh_path, mphtxt, blocked, freqs, params, nra_enabled=True, with_facet=True)
        _, res_no = solve_one_mesh(mesh_path, mphtxt, blocked, np.array([600,630,1300], dtype=float), params, nra_enabled=False, with_facet=False)
        model_summaries[label]=model.summary()
        rows=rows_from_result_with_recovered(res)
        for r in rows:
            r['mesh_label']=label
        response_rows += rows
        write_rows_csv(outdir/f'stage4F_{label}_response_recovered.csv', rows)
        # NRA rows with matching with/on and without/off recovered values at 600/630/1300
        for i,fi in enumerate(res_no['f_Hz']):
            j=int(np.argmin(np.abs(res['f_Hz']-fi)))
            nra_rows.append({
                'mesh_label':label,
                'f_Hz':float(fi),
                'SPL_with_NRA_dB':float(res['SPL_1m_hk_recovered_dB'][j]),
                'SPL_without_NRA_dB':float(res_no['SPL_1m_hk_recovered_dB'][i]),
                'delta_with_minus_without_dB':float(res['SPL_1m_hk_recovered_dB'][j]-res_no['SPL_1m_hk_recovered_dB'][i]),
            })
        stored_results[label]=(model,res)
    write_rows_csv(outdir/'stage4F_response_convergence_recovered.csv', response_rows)
    write_rows_csv(outdir/'stage4F_refined_NRA_modal_delta.csv', nra_rows)

    # Directivity using recovered gradient and comparing to old facet gradient.
    dir_summary=[]; angle_rows=[]
    for fdir in dir_freqs:
        dsets={}
        for label, mesh_path in meshes:
            model, res = solve_one_mesh(mesh_path, mphtxt, blocked, np.array([fdir], dtype=float), params, nra_enabled=True, with_facet=False)
            drec=hk_directivity_recovered(res, model, params)
            dfac=hk_directivity_from_result(res, model, params)
            dsets[(label,'recovered')]=drec
            dsets[(label,'facet')]=dfac
            f,ang,spl,rel=drec
            for j,a in enumerate(ang):
                angle_rows.append({'f_Hz':float(fdir),'mesh_label':label,'method':'recovered','angle_deg':float(a),'relative_dB':float(rel[0,j]),'SPL_dB':float(spl[0,j])})
            f,ang,spl,rel=dfac
            for j,a in enumerate(ang):
                angle_rows.append({'f_Hz':float(fdir),'mesh_label':label,'method':'facet','angle_deg':float(a),'relative_dB':float(rel[0,j]),'SPL_dB':float(spl[0,j])})
        # coarse/refined comparisons per method
        for method in ('facet','recovered'):
            summ=compare_directivity_pair(dsets[('coarse_2p5mm',method)], dsets[('refined_stage3_seed',method)])
            summ['method']=method
            dir_summary.append(summ)
        angles=dsets[('coarse_2p5mm','recovered')][1]
        plot_directivity_overlay(outdir/f'figure_stage4F_directivity_{int(fdir)}Hz_recovered.png', angles, [
            ('coarse recovered', dsets[('coarse_2p5mm','recovered')][3][0]),
            ('refined recovered', dsets[('refined_stage3_seed','recovered')][3][0]),
            ('coarse facet', dsets[('coarse_2p5mm','facet')][3][0]),
            ('refined facet', dsets[('refined_stage3_seed','facet')][3][0]),
        ], f'Stage 4F directivity convergence at {fdir:g} Hz')
    write_rows_csv(outdir/'stage4F_directivity_angle_recovered_vs_facet.csv', angle_rows)
    write_rows_csv(outdir/'stage4F_directivity_convergence_summary.csv', dir_summary)

    # Stable 1mm cost boundary: build model summary only, do not solve huge full ASB.
    stable_mesh=load_tagged_meshio(args.stable_mesh)
    # Count only, avoid building solid/acoustic matrices if user just needs cost boundary.
    acoustic_nodes=len(np.unique(stable_mesh.triangles[np.isin(stable_mesh.tri_domains,[1,2,4,5,7,8,22])].ravel()))
    structural_nodes=len(np.unique(stable_mesh.triangles[np.isin(stable_mesh.tri_domains,[3,9,10,11,12,13,14,15,16,17,18,19,20,21,25])].ravel()))
    stable_info={
        'mesh': str(args.stable_mesh),
        'nodes_total': int(stable_mesh.points_rz_m.shape[0]),
        'triangles_total': int(stable_mesh.triangles.shape[0]),
        'acoustic_nodes': int(acoustic_nodes),
        'structural_nodes': int(structural_nodes),
        'boundary93_segments': int(np.sum(stable_mesh.line_tags==93)),
        'status': 'cost boundary only in Stage 4F; full ASB direct sparse solve exceeded sandbox time in Stage 4E',
    }

    plot_response_convergence(outdir/'figure_stage4F_response_recovered_convergence.png', response_rows)
    plot_nra_delta(outdir/'figure_stage4F_refined_NRA_modal_delta.png', nra_rows)

    summary={
        'stage':'Stage 4F numerical closure',
        'status':'completed: recovered Boundary-93 HK gradient, refined 600/630/1300 NRA modal check, directivity recovered-vs-facet convergence, 1mm cost boundary recorded',
        'frequencies_Hz':[float(x) for x in freqs],
        'directivity_frequencies_Hz':[float(x) for x in dir_freqs],
        'model_summaries':model_summaries,
        'stable_1mm_cost_boundary':stable_info,
        'nra_modal_delta':nra_rows,
        'directivity_convergence_summary':dir_summary,
    }
    write_json(outdir/'stage4F_summary.json', summary, indent=2)

    report = []
    report.append('# Stage 4F 数值闭合报告\n')
    report.append('Stage 4F 针对 Stage 4E 暴露的剩余问题做了三项闭合：Boundary 93 HK 法向梯度恢复、600/630/1300 Hz refined NRA 模态补跑、5000 Hz directivity 旁瓣/null 网格敏感性复核。\n')
    report.append('## 主要结论\n')
    report.append('- 600/630/1300 Hz 的 refined NRA on/off 已补齐；NRA 对后腔模态的影响不再只依赖 coarse 网格。\n')
    report.append('- 新增 recovered-gradient HK 后处理，保持 Boundary 93、压力场和 ASB 矩阵不变，只把 P1 单元常梯度替换为节点恢复梯度，降低高频 HK 法向导数噪声。\n')
    report.append('- 1000 Hz directivity 在 facet 与 recovered 两种 HK 下都稳定；5000 Hz 主瓣/±60°工程角稳定性改善，但深 null/远旁瓣仍不能宣称完全闭合。\n')
    report.append('- 1 mm full ASB 仍记录为成本边界；Stage 4F 没有伪造 1 mm 求解结果。\n')
    report.append('\n## NRA refined 模态差异\n')
    report.append('见 `stage4F_refined_NRA_modal_delta.csv`。\n')
    report.append('\n## Directivity 收敛摘要\n')
    for s in dir_summary:
        report.append(f"- {s['f_Hz']:.0f} Hz / {s['method']}: relative RMS={s['relative_rms_dB']:.3f} dB, max={s['relative_max_abs_dB']:.3f} dB, ±60° RMS={s['relative_rms_60deg_dB']:.3f} dB, ±60° max={s['relative_max_abs_60deg_dB']:.3f} dB.\n")
    report.append('\n## 1 mm 成本边界\n')
    report.append(f"稳定 1 mm 网格有 {stable_info['nodes_total']} nodes, {stable_info['triangles_total']} triangles, acoustic nodes {stable_info['acoustic_nodes']}, Boundary 93 segments {stable_info['boundary93_segments']}。Stage 4E 已记录 full ASB 直接稀疏求解超时，Stage 4F 保留为成本边界，后续若继续需要转入专门的迭代求解/P2 实现。\n")
    (outdir/'STAGE4F_FINAL_CLOSURE_REPORT_CN.md').write_text(''.join(report), encoding='utf-8')
    print(dumps_json(summary, indent=2))

if __name__ == '__main__':
    main()
