#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from loudspeaker_axisym_fem.json_utils import write_json, dumps_json
from loudspeaker_axisym_fem.axisym_magnetics import load_tagged_meshio
from loudspeaker_axisym_fem.stage4_electroacoustic import comsol_frequency_vector, load_blocked_impedance_csv, visual_anchor_errors
from loudspeaker_axisym_fem.stage4C_acoustic_structure import Stage4CParameters, build_stage4C_acoustic_structure_model
from loudspeaker_axisym_fem.stage4D_exterior_nra import solve_stage4D_full, stage4D_rows
from loudspeaker_axisym_fem.narrow_region_acoustics import equivalent_narrow_region_coefficients, COMSOL_NARROW_REGIONS


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    with path.open('w', newline='', encoding='utf-8') as fp:
        w=csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def parse_freqs(spec: str) -> np.ndarray:
    if spec == 'comsol': return comsol_frequency_vector()
    if spec == 'anchors': return np.array([20,50,100,200,500,600,630,900,1000,1300,1500,2000,3000,5000,8000], dtype=float)
    vals=[]
    for p in spec.replace(';', ',').split(','):
        p=p.strip()
        if p: vals.append(float(p))
    return np.asarray(sorted(set(vals)), dtype=float)


def plot_sensitivity(path: Path, complete: dict, no_nra: dict) -> None:
    f=complete['f_Hz']
    fig, ax1=plt.subplots(figsize=(8,5))
    ax1.semilogx(f, complete['SPL_1m_dB'], marker='o', markersize=3, label='Stage 4D Boundary-93 HK + NRA')
    ax1.semilogx(f, no_nra['SPL_1m_dB'], marker='x', markersize=3, linestyle='--', label='Stage 4D Boundary-93 HK without NRA')
    ax1.semilogx(f, complete['SPL_1m_piston_dB'], marker='.', markersize=2, alpha=0.5, linestyle=':', label='Stage 4C piston reference')
    ax1.set_xlabel('Frequency / Hz'); ax1.set_ylabel('SPL / dB re 20 µPa')
    ax1.set_ylim(40, 110); ax1.grid(True, which='both', linewidth=0.3)
    ax2=ax1.twinx(); ax2.semilogx(f, complete['phase_deg'], alpha=0.45, label='HK phase')
    ax2.set_ylabel('Phase / deg'); ax2.set_ylim(-3200, 270)
    lines, labels=ax1.get_legend_handles_labels(); lines2, labels2=ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, loc='lower right', fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_impedance(path: Path, result: dict) -> None:
    f=result['f_Hz']; Z=result['Z_total_ohm']
    fig, ax=plt.subplots(figsize=(8,5))
    ax.semilogx(f, np.abs(Z), marker='o', markersize=3, label='abs(Z)')
    ax.semilogx(f, np.real(Z), marker='s', markersize=3, label='real(Z)')
    ax.semilogx(f, np.imag(Z), marker='^', markersize=3, label='imag(Z)')
    ax.axhline(5.6, linewidth=0.8, linestyle=':', label='DC resistance 5.6 Ω')
    ax.axhline(6.3, linewidth=0.8, linestyle='-.', label='nominal Z 6.3 Ω')
    ax.set_xlabel('Frequency / Hz'); ax.set_ylabel('Z / Ω')
    ax.set_ylim(-20,60); ax.grid(True, which='both', linewidth=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_power(path: Path, result: dict) -> None:
    f=result['f_Hz']
    fig, ax1=plt.subplots(figsize=(8,5))
    ax1.semilogx(f, result['coil_power_W'], marker='o', markersize=3, label='Coil power')
    ax1.set_xlabel('Frequency / Hz'); ax1.set_ylabel('Coil power / W'); ax1.grid(True, which='both', linewidth=0.3)
    ax2=ax1.twinx(); ax2.semilogx(f, result['acoustic_efficiency_percent'], marker='x', markersize=3, label='HK acoustic efficiency')
    ax2.set_ylabel('Acoustic efficiency / %')
    lines, labels=ax1.get_legend_handles_labels(); lines2, labels2=ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, loc='center left', fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_directivity(path: Path, result: dict) -> None:
    f=result['directivity_f_Hz']; angles=result['directivity_angles_deg']; rel=result['directivity_relative_hk_dB']
    fig, ax=plt.subplots(figsize=(8,5))
    if len(f) == 1:
        ax.plot(rel[0], angles, marker='o', markersize=2)
        ax.set_xlabel('dB relative to 0°'); ax.set_ylabel('Angle / deg')
        ax.set_title(f'Directivity at {float(f[0]):g} Hz')
        ax.set_xlim(-30, 5)
    else:
        cf=ax.contourf(f, angles, rel.T, levels=[-17,-15,-12,-9,-6,-3,-2,-1,1,2,3], extend='both')
        ax.set_xscale('log'); ax.set_xlim(max(20,float(np.min(f))), min(8000,float(np.max(f))))
        ax.set_xlabel('Frequency / Hz'); ax.set_ylabel('Angle / deg')
        fig.colorbar(cf, ax=ax, label='dB relative to 0°')
    ax.set_ylim(-90,90)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_displacement(path: Path, model, result: dict, target: float) -> None:
    f=result['f_Hz']; idx=int(np.argmin(np.abs(f-target)))
    u=result['solid_displacement_m'][idx]; solid=model.solid
    tri=mtri.Triangulation(solid.points_rz_m[:,0]*1000, solid.points_rz_m[:,1]*1000, solid.triangles)
    mag=np.sqrt(np.abs(u[0::2])**2 + np.abs(u[1::2])**2)
    fig, ax=plt.subplots(figsize=(6,6)); tpc=ax.tripcolor(tri, mag*1e3, shading='gouraud')
    ax.triplot(tri, linewidth=0.25, alpha=0.25); ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('r / mm'); ax.set_ylabel('z / mm'); ax.set_title(f'Stage 4D displacement at {f[idx]:g} Hz')
    fig.colorbar(tpc, ax=ax, label='displacement magnitude / mm peak')
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_pressure_field(path: Path, model, result: dict, target: float) -> None:
    f=result['f_Hz']; idx=int(np.argmin(np.abs(f-target)))
    pvals=np.real(result['acoustic_pressure_field_Pa'][idx]); ag=model.acoustic_nodes_global
    pts=model.mesh.points_rz_m[ag]; amap=model.acoustic_node_map
    mask=np.isin(model.mesh.tri_domains, [1,2,4,5,7,8,22])
    tris=[]
    for tri in model.mesh.triangles[mask]:
        if all(int(x) in amap for x in tri): tris.append([amap[int(x)] for x in tri])
    triang=mtri.Triangulation(pts[:,0]*1000, pts[:,1]*1000, np.asarray(tris,dtype=int))
    lim=np.nanpercentile(np.abs(pvals),98) if np.any(np.isfinite(pvals)) else 1.0
    fig, ax=plt.subplots(figsize=(6,6)); tpc=ax.tripcolor(triang, pvals, shading='gouraud', vmin=-lim, vmax=lim)
    ax.set_aspect('equal', adjustable='box'); ax.set_xlabel('r / mm'); ax.set_ylabel('z / mm')
    ax.set_title(f'Stage 4D real acoustic pressure at {f[idx]:g} Hz')
    fig.colorbar(tpc, ax=ax, label='Pa peak, real part')
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def nra_coeff_table(freqs: np.ndarray) -> list[dict]:
    rows=[]
    for f in freqs:
        for key,slit in COMSOL_NARROW_REGIONS.items():
            c=equivalent_narrow_region_coefficients(float(f), slit.height_m)
            rows.append({
                'f_Hz':float(f), 'feature':key, 'domain':slit.domain_id, 'height_mm':slit.height_m*1000,
                'rho_ratio_real':float(c.rho_eq_over_rho0.real), 'rho_ratio_imag':float(c.rho_eq_over_rho0.imag),
                'bulk_ratio_real':float(c.bulk_eq_over_bulk0.real), 'bulk_ratio_imag':float(c.bulk_eq_over_bulk0.imag),
                'k_ratio_real':float(c.complex_wavenumber_over_k0.real), 'k_ratio_imag':float(c.complex_wavenumber_over_k0.imag),
                'viscous_delta_over_half_height':float(c.boundary_layer['viscous_delta_over_half_height']),
                'thermal_delta_over_half_height':float(c.boundary_layer['thermal_delta_over_half_height']),
            })
    return rows


def safe_visual_anchor_errors(result: dict) -> dict:
    try:
        return visual_anchor_errors(result)
    except Exception as exc:
        return {"skipped": True, "reason": str(exc)}


def main() -> None:
    ap=argparse.ArgumentParser(description='Stage 4D: Boundary-93 HK/pext exterior field and slit NRA equivalent coefficients.')
    ap.add_argument('--mesh', default=str(ROOT/'meshes/comsol_stable_1mm_05gap.msh'))
    ap.add_argument('--mphtxt', default=str(ROOT/'comsol_reference_inputs/Untitled.mphtxt'))
    ap.add_argument('--blocked-impedance-csv', default=str(ROOT/'outputs/stage3C_corrected_baseline/blocked_impedance_exact_voltage.csv'))
    ap.add_argument('--outdir', default=str(ROOT/'outputs/stage4D_exterior_nra'))
    ap.add_argument('--freqs', default='anchors')
    ap.add_argument('--BL', type=float, default=10.482177800)
    ap.add_argument('--V0-peak', type=float, default=3.55)
    ap.add_argument('--radiation-radius-mm', type=float, default=70.0)
    ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--skip-directivity', action='store_true', help='skip HK directivity map for split/high-cost runs')
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    freqs=parse_freqs(args.freqs)
    freqs,Zb=load_blocked_impedance_csv(args.blocked_impedance_csv, freqs)
    mesh=load_tagged_meshio(args.mesh)
    model=build_stage4C_acoustic_structure_model(mesh, args.mphtxt, solid_uniform_refine=0)
    params=Stage4CParameters(BL_N_A=args.BL, V0_peak_V=args.V0_peak, radiation_radius_m=args.radiation_radius_mm*1e-3)
    complete=solve_stage4D_full(freqs, Zb, model, params, nra_enabled=True, hk_directivity=(not args.skip_directivity))
    no_nra=solve_stage4D_full(freqs, Zb, model, params, nra_enabled=False, hk_directivity=False)
    write_rows_csv(outdir/'stage4D_complete_response.csv', stage4D_rows(complete))
    write_rows_csv(outdir/'stage4D_without_nra_response.csv', stage4D_rows(no_nra))
    write_rows_csv(outdir/'stage4D_nra_equivalent_coefficients.csv', nra_coeff_table(freqs))
    if 'directivity_f_Hz' in complete:
        with (outdir/'stage4D_directivity_relative.csv').open('w', newline='', encoding='utf-8') as fp:
            w=csv.writer(fp); w.writerow(['f_Hz','angle_deg','SPL_dB','relative_dB'])
            for i,fi in enumerate(complete['directivity_f_Hz']):
                for j,a in enumerate(complete['directivity_angles_deg']):
                    w.writerow([float(fi),float(a),float(complete['directivity_spl_hk_dB'][i,j]),float(complete['directivity_relative_hk_dB'][i,j])])
    summary={
        'stage':'Stage 4D Boundary-93 HK/pext exterior + thermoviscous slit NRA equivalent',
        'status':'Boundary 93 HK exterior field is now primary SPL/directivity; domains 8/22 use frequency-dependent slit thermoviscous equivalent coefficients. Still P1/coarse-mesh unless run with finer mesh.',
        'mesh':str(args.mesh), 'mphtxt':str(args.mphtxt), 'frequencies_Hz':[float(x) for x in freqs],
        'model':model.summary(), 'parameters':asdict(params),
        'complete_visual_anchor_errors':safe_visual_anchor_errors(complete),
        'without_nra_visual_anchor_errors':safe_visual_anchor_errors(no_nra),
        'hk_boundary_info_first_frequency':complete['hk_boundary_info'][0] if complete['hk_boundary_info'] else None,
        'nra_coefficients_selected':nra_coeff_table(np.array([600.0, 1300.0, 8000.0])),
    }
    write_json(outdir/'stage4D_summary.json', summary, indent=2)
    if not args.no_plots:
        plot_sensitivity(outdir/'figure8_stage4D_sensitivity_phase.png', complete, no_nra)
        plot_impedance(outdir/'figure10_stage4D_total_electric_impedance.png', complete)
        plot_power(outdir/'stage4D_coil_power_efficiency.png', complete)
        if 'directivity_f_Hz' in complete:
            plot_directivity(outdir/'figure12_stage4D_directivity.png', complete)
        plot_displacement(outdir/'figure7_stage4D_displacement_8000Hz.png', model, complete, 8000.0)
        plot_pressure_field(outdir/'figure7_stage4D_pressure_8000Hz.png', model, complete, 8000.0)
    report=f'''# Stage 4D：Boundary 93 pext/HK + Narrow Region Acoustics 等效模型\n\n## 已实现\n\n- 使用 COMSOL Boundary 93 作为 HK 外场积分边界。\n- 从 Stage 4C ASB 压力场提取 P1 压力与梯度，构造 `p` 与 `∂p/∂n`。\n- 1 m 轴上 SPL、phase、声功率、directivity 改用 Boundary-93 HK/pext 风格结果。\n- Domain 8 / 22 Narrow Region Acoustics 从占位阻尼改为频率相关 parallel-plate slit thermoviscous 等效参数。\n- 继续保留 without-NRA study，用于 Figure 8/9 的损耗对比。\n\n## 模型规模\n\n```text\n{dumps_json(model.summary(), indent=2)}\n```\n\n## 当前限制\n\n- 当前运行仍使用 coarse/P1 ASB 网格；结果代表 Stage 4D 链路完成，不是最终 COMSOL Figure 8/12 定量闭合。\n- HK 外场对 Boundary 93 压力/梯度和 PML 边界条件较敏感；下一步需要 h=1 mm/0.5 mm 局部细化和 P2 声学单元。\n- NRA 使用公开 slit 等效参数，不是 COMSOL 专有内部实现的逐项复制。\n'''
    (outdir/'STAGE4D_EXTERIOR_NRA_REPORT_CN.md').write_text(report, encoding='utf-8')
    print(dumps_json(summary, indent=2))

if __name__=='__main__':
    main()
