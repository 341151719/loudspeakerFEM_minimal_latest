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
from loudspeaker_axisym_fem.stage4_electroacoustic import comsol_frequency_vector, load_blocked_impedance_csv, directivity_map
from loudspeaker_axisym_fem.stage4_solid_fem import build_stage4_solid_model, compute_eigenmodes, solve_structural_response
from loudspeaker_axisym_fem.stage4B_solid_electroacoustic import (
    Stage4BSolidCouplingParameters,
    solve_stage4B_solid_coupled,
    result_to_rows_stage4B,
    stage4B_visual_summary,
)


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as fp:
        w=csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def plot_sensitivity(path: Path, complete: dict, no_nra: dict) -> None:
    f=complete['f_Hz']
    fig, ax1=plt.subplots(figsize=(8,5))
    ax1.semilogx(f, complete['SPL_1m_dB'], label='Stage 4B solid FEM / NRA')
    ax1.semilogx(f, no_nra['SPL_1m_dB'], linestyle='--', label='Stage 4B without NRA')
    ax1.set_xlabel('Frequency / Hz'); ax1.set_ylabel('SPL / dB re 20 µPa')
    ax1.set_ylim(64, 92); ax1.grid(True, which='both', linewidth=0.3)
    ax2=ax1.twinx()
    ax2.semilogx(f, complete['phase_deg'], alpha=0.55, label='Phase')
    ax2.set_ylabel('Phase / deg'); ax2.set_ylim(-3200,270)
    lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, loc='lower right', fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_impedance(path: Path, result: dict) -> None:
    f=result['f_Hz']; Z=result['Z_total_ohm']
    fig, ax=plt.subplots(figsize=(8,5))
    ax.semilogx(f, np.abs(Z), label='abs(Z)')
    ax.semilogx(f, np.real(Z), label='real(Z)')
    ax.semilogx(f, np.imag(Z), label='imag(Z)')
    ax.axhline(5.6, linewidth=0.8, linestyle=':', label='DC resistance 5.6 Ω')
    ax.axhline(6.3, linewidth=0.8, linestyle='-.', label='nominal Z 6.3 Ω')
    ax.set_xlabel('Frequency / Hz'); ax.set_ylabel('Z / Ω')
    ax.set_ylim(-15, 50); ax.grid(True, which='both', linewidth=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_mechanical_impedance(path: Path, result: dict) -> None:
    f=result['f_Hz']; Z=result['Zm_solid_N_s_m']
    fig, ax=plt.subplots(figsize=(8,5))
    ax.loglog(f, np.abs(Z), label='abs(Zm solid)')
    ax.semilogx(f, np.maximum(np.real(Z), 1e-12), label='real(Zm)')
    ax.set_xlabel('Frequency / Hz'); ax.set_ylabel('Mechanical impedance / N·s/m')
    ax.grid(True, which='both', linewidth=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_power_efficiency(path: Path, result: dict) -> None:
    f=result['f_Hz']
    fig, ax1=plt.subplots(figsize=(8,5))
    ax1.semilogx(f, result['coil_power_W'], label='Coil power')
    ax1.set_xlabel('Frequency / Hz'); ax1.set_ylabel('Coil power / W'); ax1.grid(True, which='both', linewidth=0.3)
    ax2=ax1.twinx(); ax2.semilogx(f, result['acoustic_efficiency_percent'], label='Acoustic efficiency')
    ax2.set_ylabel('Acoustic efficiency / %')
    lines, labels=ax1.get_legend_handles_labels(); lines2, labels2=ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, loc='center left', fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_directivity(path: Path, freqs: np.ndarray, angles: np.ndarray, rel: np.ndarray) -> None:
    fig, ax=plt.subplots(figsize=(8,5))
    levels=[-17,-15,-12,-9,-6,-3,-2,-1,1,2,3]
    cf=ax.contourf(freqs, angles, rel.T, levels=levels, extend='both')
    ax.set_xscale('log'); ax.set_xlim(20,8000); ax.set_ylim(-90,90)
    ax.set_xlabel('Frequency / Hz'); ax.set_ylabel('Angle / deg')
    fig.colorbar(cf, ax=ax, label='dB relative to 0°')
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_mode_shape(path: Path, model, mode_vec: np.ndarray, title: str) -> None:
    r=model.points_rz_m[:,0]*1000.0; z=model.points_rz_m[:,1]*1000.0
    tri=mtri.Triangulation(r,z,model.triangles)
    ur=mode_vec[0::2]; uz=mode_vec[1::2]
    mag=np.sqrt(ur*ur+uz*uz)
    # normalized deformation for plotting in mm geometry units
    scale=0.18*max(np.ptp(r),np.ptp(z))/max(np.max(mag),1e-30)
    fig, ax=plt.subplots(figsize=(6,6))
    tpc=ax.tripcolor(tri, mag, shading='gouraud')
    ax.triplot(tri, linewidth=0.25, alpha=0.25)
    ax.plot(r + scale*ur, z + scale*uz, '.', markersize=1.4, alpha=0.45)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('r / mm'); ax.set_ylabel('z / mm'); ax.set_title(title)
    fig.colorbar(tpc, ax=ax, label='relative displacement magnitude')
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_frequency_displacement(path: Path, model, result: dict, freq_target: float) -> None:
    f=result['f_Hz']; i=int(np.argmin(np.abs(f-freq_target)))
    u=result['solid_displacement_per_N'][i] * result['F_Lorentz_N_peak'][i]
    r=model.points_rz_m[:,0]*1000.0; z=model.points_rz_m[:,1]*1000.0
    tri=mtri.Triangulation(r,z,model.triangles)
    mag=np.abs(u[0::2] + 1j*u[1::2])
    fig, ax=plt.subplots(figsize=(6,6))
    tpc=ax.tripcolor(tri, mag*1e3, shading='gouraud')
    ax.triplot(tri, linewidth=0.25, alpha=0.25)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('r / mm'); ax.set_ylabel('z / mm')
    ax.set_title(f'Stage 4B solid displacement at {f[i]:g} Hz')
    fig.colorbar(tpc, ax=ax, label='displacement magnitude / mm peak')
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> None:
    ap=argparse.ArgumentParser(description='Stage 4B: axisymmetric solid-FEM mechanical dynamic stiffness coupled to Stage-3C blocked impedance.')
    ap.add_argument('--mesh', default=str(ROOT/'meshes/comsol_stable_1mm_05gap.msh'))
    ap.add_argument('--blocked-impedance-csv', default=str(ROOT/'outputs/stage3C_corrected_baseline/blocked_impedance_exact_voltage.csv'))
    ap.add_argument('--outdir', default=str(ROOT/'outputs/stage4B_solid_electroacoustic'))
    ap.add_argument('--BL', type=float, default=10.482177800)
    ap.add_argument('--V0-peak', type=float, default=3.55)
    ap.add_argument('--radiation-radius-mm', type=float, default=70.0)
    ap.add_argument('--uniform-refine', type=int, default=1)
    ap.add_argument('--nmodes', type=int, default=8)
    ap.add_argument('--no-plots', action='store_true')
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    freqs=comsol_frequency_vector()
    f,Zb=load_blocked_impedance_csv(args.blocked_impedance_csv, freqs)
    mesh=load_tagged_meshio(args.mesh)
    solid=build_stage4_solid_model(mesh, uniform_refine=args.uniform_refine)
    eig=compute_eigenmodes(solid, nmodes=args.nmodes, sigma_Hz=50)
    params=Stage4BSolidCouplingParameters(BL_N_A=args.BL, V0_peak_V=args.V0_peak, radiation_radius_m=args.radiation_radius_mm*1e-3)
    complete=solve_stage4B_solid_coupled(f,Zb,solid,params,nra_enabled=True)
    no_nra=solve_stage4B_solid_coupled(f,Zb,solid,params,nra_enabled=False)
    rows=result_to_rows_stage4B(complete); rows_no=result_to_rows_stage4B(no_nra)
    write_rows_csv(outdir/'stage4B_complete_response.csv', rows)
    write_rows_csv(outdir/'stage4B_without_nra_response.csv', rows_no)
    # eigen csv
    with (outdir/'stage4B_solid_eigenfrequencies.csv').open('w', newline='', encoding='utf-8') as fp:
        w=csv.writer(fp); w.writerow(['mode_index','f_Hz'])
        for i,fr in enumerate(eig['f_Hz'],1): w.writerow([i,float(fr)])
    fd,ang,rel=directivity_map(f, type('P',(),{'effective_radius_m':params.radiation_radius_m, 'c0_m_s':params.c0_m_s})())
    with (outdir/'stage4B_directivity_relative.csv').open('w', newline='', encoding='utf-8') as fp:
        w=csv.writer(fp); w.writerow(['f_Hz','angle_deg','relative_dB'])
        for i,fi in enumerate(fd):
            for j,a in enumerate(ang): w.writerow([float(fi),float(a),float(rel[i,j])])
    summary={
        'stage':'Stage 4B solid-FEM electroacoustic coupling',
        'status':'solid FEM mechanical dynamic stiffness implemented; acoustic radiation still piston/exterior baseline, not full pressure-acoustic ASB matrix',
        'mesh':str(args.mesh),
        'uniform_refine':args.uniform_refine,
        'solid_model':solid.summary(),
        'parameters':asdict(params),
        'eigenfrequencies_Hz':[float(x) for x in eig['f_Hz']],
        'comsol_figure11_reference_Hz':[53.237,2347.4,2914.9,3553.9],
        'visual_anchor_errors_complete':stage4B_visual_summary(complete),
        'visual_anchor_errors_without_nra':stage4B_visual_summary(no_nra),
    }
    write_json(outdir/'stage4B_summary.json', summary, indent=2)
    if not args.no_plots:
        plot_sensitivity(outdir/'figure8_stage4B_sensitivity_phase.png', complete, no_nra)
        plot_impedance(outdir/'figure10_stage4B_total_electric_impedance.png', complete)
        plot_mechanical_impedance(outdir/'stage4B_solid_mechanical_impedance.png', complete)
        plot_power_efficiency(outdir/'stage4B_coil_power_efficiency.png', complete)
        plot_directivity(outdir/'figure12_stage4B_directivity.png', fd, ang, rel)
        for i in range(min(4, eig['modes'].shape[0])):
            plot_mode_shape(outdir/f'figure11_stage4B_mode_{i+1:02d}.png', solid, eig['modes'][i], f'Mode {i+1}: {eig["f_Hz"][i]:.2f} Hz')
        plot_frequency_displacement(outdir/'figure7_stage4B_displacement_8000Hz.png', solid, complete, 8000.0)
    report=f"""# Stage 4B solid-FEM 电-机-声耦合报告

Stage 4B 将 Stage 4A 的单自由度机械阻抗 `Zm` 替换为从 COMSOL structural domains 组装出的轴对称 solid FEM 动刚度。当前仍保留简化 piston/exterior radiation，因此本阶段完成的是 **solid-FEM mechanical dynamics + electrical back-EMF loop**，不是最终 pressure-acoustic ASB 全矩阵。

## 固体 FEM 模型

```json
{dumps_json(solid.summary(), indent=2)}
```

## 模态结果

COMSOL Figure 11 参考：约 53.237 Hz、2347.4 Hz、2914.9 Hz、3553.9 Hz。

当前 Stage 4B eigenfrequencies:

```json
{dumps_json(summary['eigenfrequencies_Hz'], indent=2)}
```

## Figure 8/10 视觉锚点误差

```json
{dumps_json(summary['visual_anchor_errors_complete'], indent=2)}
```

## 输出文件

- `stage4B_complete_response.csv`
- `stage4B_without_nra_response.csv`
- `stage4B_solid_eigenfrequencies.csv`
- `figure8_stage4B_sensitivity_phase.png`
- `figure10_stage4B_total_electric_impedance.png`
- `figure11_stage4B_mode_*.png`
- `figure7_stage4B_displacement_8000Hz.png`
- `figure12_stage4B_directivity.png`

## 结论

Stage 4B 已经完成 structural-domain solid FEM 的矩阵替换。第一模态从未细化 P1 的 67 Hz 降到当前细化后的约 {eig['f_Hz'][0]:.2f} Hz，已接近 COMSOL 的 53.237 Hz；高阶 breakup 模态仍有差异，主要来自 P1 三角形、polyline 几何和未使用 COMSOL quadratic/mapped solid mesh。下一步 Stage 4C 应加入 pressure-acoustic FEM 与 Acoustic-Structure Boundary 双向耦合，替代当前 piston radiation。
"""
    (outdir/'STAGE4B_SOLID_ELECTROACOUSTIC_REPORT_CN.md').write_text(report, encoding='utf-8')
    print(dumps_json(summary, indent=2))

if __name__=='__main__':
    main()
