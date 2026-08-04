#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from loudspeaker_axisym_fem.json_utils import write_json, dumps_json
from loudspeaker_axisym_fem.axisym_magnetics import load_tagged_meshio
from loudspeaker_axisym_fem.stage4_electroacoustic import comsol_frequency_vector, load_blocked_impedance_csv, directivity_map, visual_anchor_errors
from loudspeaker_axisym_fem.stage4C_acoustic_structure import (
    Stage4CParameters,
    build_stage4C_acoustic_structure_model,
    solve_stage4C_full_asb,
    result_to_rows_stage4C,
)


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    with path.open('w', newline='', encoding='utf-8') as fp:
        w=csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def parse_freqs(spec: str) -> np.ndarray:
    if spec == 'comsol':
        return comsol_frequency_vector()
    if spec == 'anchors':
        return np.array([20,50,100,200,500,600,630,900,1000,1300,1500,2000,3000,5000,8000], dtype=float)
    vals=[]
    for p in spec.replace(';', ',').split(','):
        p=p.strip()
        if p: vals.append(float(p))
    return np.asarray(sorted(set(vals)), dtype=float)


def plot_sensitivity(path: Path, complete: dict, no_nra: dict) -> None:
    f=complete['f_Hz']
    fig, ax1=plt.subplots(figsize=(8,5))
    ax1.semilogx(f, complete['SPL_1m_dB'], marker='o', markersize=3, label='Stage 4C full ASB / NRA')
    ax1.semilogx(f, no_nra['SPL_1m_dB'], marker='x', markersize=3, linestyle='--', label='Stage 4C without NRA')
    ax1.set_xlabel('Frequency / Hz'); ax1.set_ylabel('SPL / dB re 20 µPa')
    ax1.set_ylim(64, 96); ax1.grid(True, which='both', linewidth=0.3)
    ax2=ax1.twinx(); ax2.semilogx(f, complete['phase_deg'], alpha=0.45, label='Phase')
    ax2.set_ylabel('Phase / deg'); ax2.set_ylim(-3200, 270)
    lines, labels=ax1.get_legend_handles_labels(); lines2, labels2=ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, loc='lower right', fontsize=8)
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
    ax.set_ylim(-20, 60); ax.grid(True, which='both', linewidth=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_power(path: Path, result: dict) -> None:
    f=result['f_Hz']
    fig, ax1=plt.subplots(figsize=(8,5))
    ax1.semilogx(f, result['coil_power_W'], marker='o', markersize=3, label='Coil power')
    ax1.set_xlabel('Frequency / Hz'); ax1.set_ylabel('Coil power / W'); ax1.grid(True, which='both', linewidth=0.3)
    ax2=ax1.twinx(); ax2.semilogx(f, result['acoustic_efficiency_percent'], marker='x', markersize=3, label='Acoustic efficiency')
    ax2.set_ylabel('Acoustic efficiency / %')
    lines, labels=ax1.get_legend_handles_labels(); lines2, labels2=ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, loc='center left', fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_directivity(path: Path, freqs: np.ndarray, angles: np.ndarray, rel: np.ndarray) -> None:
    fig, ax=plt.subplots(figsize=(8,5))
    cf=ax.contourf(freqs, angles, rel.T, levels=[-17,-15,-12,-9,-6,-3,-2,-1,1,2,3], extend='both')
    ax.set_xscale('log'); ax.set_xlim(max(20, float(np.min(freqs))), min(8000, float(np.max(freqs))))
    ax.set_ylim(-90,90); ax.set_xlabel('Frequency / Hz'); ax.set_ylabel('Angle / deg')
    fig.colorbar(cf, ax=ax, label='dB relative to 0°')
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_displacement(path: Path, model, result: dict, target: float) -> None:
    f=result['f_Hz']; idx=int(np.argmin(np.abs(f-target)))
    u=result['solid_displacement_m'][idx]
    solid=model.solid
    r=solid.points_rz_m[:,0]*1000; z=solid.points_rz_m[:,1]*1000
    tri=mtri.Triangulation(r,z,solid.triangles)
    mag=np.sqrt(np.abs(u[0::2])**2 + np.abs(u[1::2])**2)
    fig, ax=plt.subplots(figsize=(6,6))
    tpc=ax.tripcolor(tri, mag*1e3, shading='gouraud')
    ax.triplot(tri, linewidth=0.25, alpha=0.25)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('r / mm'); ax.set_ylabel('z / mm')
    ax.set_title(f'Stage 4C ASB displacement at {f[idx]:g} Hz')
    fig.colorbar(tpc, ax=ax, label='displacement magnitude / mm peak')
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_pressure_field(path: Path, model, result: dict, target: float) -> None:
    f=result['f_Hz']; idx=int(np.argmin(np.abs(f-target)))
    pvals=np.real(result['acoustic_pressure_field_Pa'][idx])
    ag=model.acoustic_nodes_global
    pts=model.mesh.points_rz_m[ag]
    # Plot only acoustic triangles with local remapping.
    amap=model.acoustic_node_map
    mask=np.isin(model.mesh.tri_domains, [1,2,4,5,7,8,22])
    tris=[]
    for tri in model.mesh.triangles[mask]:
        if all(int(x) in amap for x in tri):
            tris.append([amap[int(x)] for x in tri])
    tri=np.asarray(tris, dtype=int)
    fig, ax=plt.subplots(figsize=(6,6))
    triang=mtri.Triangulation(pts[:,0]*1000, pts[:,1]*1000, tri)
    lim=np.nanpercentile(np.abs(pvals), 98) if np.any(np.isfinite(pvals)) else 1.0
    tpc=ax.tripcolor(triang, pvals, shading='gouraud', vmin=-lim, vmax=lim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('r / mm'); ax.set_ylabel('z / mm')
    ax.set_title(f'Stage 4C real acoustic pressure at {f[idx]:g} Hz')
    fig.colorbar(tpc, ax=ax, label='Pa peak, real part')
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> None:
    ap=argparse.ArgumentParser(description='Stage 4C: coupled pressure-acoustic FEM + solid FEM Acoustic-Structure Boundary matrix.')
    ap.add_argument('--mesh', default=str(ROOT/'meshes/comsol_stable_1mm_05gap.msh'))
    ap.add_argument('--mphtxt', default=str(ROOT/'comsol_reference_inputs/Untitled.mphtxt'))
    ap.add_argument('--blocked-impedance-csv', default=str(ROOT/'outputs/stage3C_corrected_baseline/blocked_impedance_exact_voltage.csv'))
    ap.add_argument('--outdir', default=str(ROOT/'outputs/stage4C_acoustic_structure'))
    ap.add_argument('--freqs', default='anchors', help="'anchors', 'comsol', or comma-separated Hz")
    ap.add_argument('--BL', type=float, default=10.482177800)
    ap.add_argument('--V0-peak', type=float, default=3.55)
    ap.add_argument('--radiation-radius-mm', type=float, default=70.0)
    ap.add_argument('--no-plots', action='store_true')
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    freqs=parse_freqs(args.freqs)
    freqs,Zb=load_blocked_impedance_csv(args.blocked_impedance_csv, freqs)
    mesh=load_tagged_meshio(args.mesh)
    model=build_stage4C_acoustic_structure_model(mesh, args.mphtxt, solid_uniform_refine=0)
    params=Stage4CParameters(BL_N_A=args.BL, V0_peak_V=args.V0_peak, radiation_radius_m=args.radiation_radius_mm*1e-3)
    complete=solve_stage4C_full_asb(freqs, Zb, model, params, nra_enabled=True)
    no_nra=solve_stage4C_full_asb(freqs, Zb, model, params, nra_enabled=False)
    write_rows_csv(outdir/'stage4C_complete_response.csv', result_to_rows_stage4C(complete))
    write_rows_csv(outdir/'stage4C_without_nra_response.csv', result_to_rows_stage4C(no_nra))
    fd,ang,rel=directivity_map(freqs, type('P',(),{'effective_radius_m':params.radiation_radius_m, 'c0_m_s':params.c0_m_s})())
    with (outdir/'stage4C_directivity_relative.csv').open('w', newline='', encoding='utf-8') as fp:
        w=csv.writer(fp); w.writerow(['f_Hz','angle_deg','relative_dB'])
        for i,fi in enumerate(fd):
            for j,a in enumerate(ang): w.writerow([float(fi),float(a),float(rel[i,j])])
    summary={
        'stage':'Stage 4C pressure-acoustic FEM + solid FEM Acoustic-Structure Boundary',
        'status':'full coupled ASB matrix implemented for mechanical/acoustic compliance; Boundary-93 exact COMSOL pext/HK replacement remains Stage 4D',
        'mesh':str(args.mesh),
        'mphtxt':str(args.mphtxt),
        'frequencies_Hz':[float(x) for x in freqs],
        'model':model.summary(),
        'parameters':asdict(params),
        'complete_visual_anchor_errors':visual_anchor_errors(complete),
        'without_nra_visual_anchor_errors':visual_anchor_errors(no_nra),
    }
    write_json(outdir/'stage4C_summary.json', summary, indent=2)
    if not args.no_plots:
        plot_sensitivity(outdir/'figure8_stage4C_sensitivity_phase.png', complete, no_nra)
        plot_impedance(outdir/'figure10_stage4C_total_electric_impedance.png', complete)
        plot_power(outdir/'stage4C_coil_power_efficiency.png', complete)
        plot_directivity(outdir/'figure12_stage4C_directivity.png', fd, ang, rel)
        plot_displacement(outdir/'figure7_stage4C_displacement_8000Hz.png', model, complete, 8000.0)
        plot_pressure_field(outdir/'figure7_stage4C_pressure_8000Hz.png', model, complete, 8000.0)
    report=f"""# Stage 4C：Pressure Acoustics FEM + Acoustic-Structure Boundary\n\n本阶段新增完整线性声固耦合矩阵，不再只用 piston radiation 作为结构负载。\n\n## 已实现\n\n- Pressure-acoustic FEM on COMSOL Air/PML domains 1,2,4,5,7,8,22.\n- Solid FEM structural domains from Stage 4B.\n- Acoustic-Structure Boundary interface detection from mphtxt boundary adjacency.\n- Coupled block matrix: `[S  -G; -rho*omega^2 G^T  A]`.\n- Unit-force coupled mechanical compliance -> electrical back-EMF loop `Z=Zb+BL^2/Zm_asb`.\n- NRA on/off coefficient in domains 8 and 22.\n- Figure 8/10/12/7 Stage-4C outputs.\n\n## 模型规模\n\n```text\n{dumps_json(model.summary(), indent=2)}\n```\n\n## 状态判断\n\nStage 4C 已完成 `solid + acpr + asb1` 的核心矩阵耦合。仍未完成的是 COMSOL Boundary 93 `pext()` 的严格外场替代；当前 1 m SPL 仍使用 ASB 求得的耦合速度加半空间体积速度表达式作为外场锚点，Boundary 93 的 facet-HK/pext 应作为 Stage 4D。\n"""
    (outdir/'STAGE4C_ACOUSTIC_STRUCTURE_REPORT_CN.md').write_text(report, encoding='utf-8')
    print(dumps_json(summary, indent=2))

if __name__=='__main__':
    main()
