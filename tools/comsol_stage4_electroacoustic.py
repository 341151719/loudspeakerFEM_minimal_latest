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

from loudspeaker_axisym_fem.json_utils import write_json, dumps_json
from loudspeaker_axisym_fem.stage4_electroacoustic import (
    Stage4ElectroacousticParameters,
    comsol_frequency_vector,
    load_blocked_impedance_csv,
    solve_stage4_lumped,
    result_to_rows,
    directivity_map,
    visual_anchor_errors,
)


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def plot_sensitivity(path: Path, complete: dict, no_nra: dict) -> None:
    fig, ax1 = plt.subplots(figsize=(8,5))
    f=complete['f_Hz']
    ax1.semilogx(f, complete['SPL_1m_dB'], label='Stage 4 complete / NRA')
    ax1.semilogx(f, no_nra['SPL_1m_dB'], linestyle='--', label='Stage 4 without NRA')
    ax1.set_xlabel('Frequency / Hz')
    ax1.set_ylabel('SPL / dB re 20 µPa')
    ax1.set_ylim(64, 92)
    ax1.grid(True, which='both', linewidth=0.3)
    ax2=ax1.twinx()
    ax2.semilogx(f, complete['phase_deg'], alpha=0.55, label='Phase')
    ax2.set_ylabel('Phase / deg')
    ax2.set_ylim(-3200, 270)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, loc='lower right', fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_impedance(path: Path, result: dict) -> None:
    f=result['f_Hz']; Z=result['Z_total_ohm']
    fig, ax = plt.subplots(figsize=(8,5))
    ax.semilogx(f, np.abs(Z), label='abs(Z)')
    ax.semilogx(f, np.real(Z), label='real(Z)')
    ax.semilogx(f, np.imag(Z), label='imag(Z)')
    ax.axhline(5.6, linewidth=0.8, linestyle=':', label='DC resistance 5.6 Ω')
    ax.axhline(6.3, linewidth=0.8, linestyle='-.', label='nominal Z 6.3 Ω')
    ax.set_xlabel('Frequency / Hz'); ax.set_ylabel('Z / Ω')
    ax.set_ylim(-15, 48)
    ax.grid(True, which='both', linewidth=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_power_efficiency(path: Path, result: dict) -> None:
    f=result['f_Hz']
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.semilogx(f, result['coil_power_W'], label='Coil power')
    ax1.set_xlabel('Frequency / Hz'); ax1.set_ylabel('Coil power / W')
    ax1.grid(True, which='both', linewidth=0.3)
    ax2=ax1.twinx()
    ax2.semilogx(f, result['acoustic_efficiency_percent'], label='Acoustic efficiency')
    ax2.set_ylabel('Acoustic efficiency / %')
    lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, loc='middle left' if False else 'center left', fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_directivity(path: Path, freqs: np.ndarray, angles: np.ndarray, rel: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8,5))
    levels=[-17,-15,-12,-9,-6,-3,-2,-1,1,2,3]
    cf=ax.contourf(freqs, angles, rel.T, levels=levels, extend='both')
    ax.set_xscale('log')
    ax.set_xlim(20,8000); ax.set_ylim(-90,90)
    ax.set_xlabel('Frequency / Hz'); ax.set_ylabel('Angle / deg')
    fig.colorbar(cf, ax=ax, label='dB relative to 0°')
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> None:
    ap=argparse.ArgumentParser(description='Stage 4 Study-2 electro-acoustic baseline using Stage-2 BL and Stage-3C blocked impedance.')
    ap.add_argument('--blocked-impedance-csv', default=str(ROOT/'outputs/stage3C_terminal_baseline/blocked_impedance_exact_voltage_sigma_eff.csv'))
    ap.add_argument('--outdir', default=str(ROOT/'outputs/stage4_electroacoustic'))
    ap.add_argument('--BL', type=float, default=10.482177800)
    ap.add_argument('--V0-peak', type=float, default=3.55)
    ap.add_argument('--effective-radius-mm', type=float, default=70.0)
    ap.add_argument('--Mms-g', type=float, default=12.0)
    ap.add_argument('--f0-Hz', type=float, default=53.237)
    ap.add_argument('--Rms', type=float, default=3.9333333333333336)
    ap.add_argument('--no-plots', action='store_true')
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    freqs=comsol_frequency_vector()
    f,Zb=load_blocked_impedance_csv(args.blocked_impedance_csv, freqs)
    params=Stage4ElectroacousticParameters(
        BL_N_A=args.BL,
        V0_peak_V=args.V0_peak,
        effective_radius_m=args.effective_radius_mm*1e-3,
        Mms_kg=args.Mms_g*1e-3,
        f0_Hz=args.f0_Hz,
        Rms_N_s_m=args.Rms,
    )
    complete=solve_stage4_lumped(f,Zb,params,narrow_region_enabled=True)
    no_nra=solve_stage4_lumped(f,Zb,params,narrow_region_enabled=False)
    rows=result_to_rows(complete)
    rows_no=result_to_rows(no_nra)
    write_rows_csv(outdir/'stage4_complete_response.csv', rows)
    write_rows_csv(outdir/'stage4_without_nra_response.csv', rows_no)
    fd,ang,rel=directivity_map(f,params)
    # save directivity matrix in long CSV
    with (outdir/'stage4_directivity_relative.csv').open('w', newline='', encoding='utf-8') as fp:
        w=csv.writer(fp); w.writerow(['f_Hz','angle_deg','relative_dB'])
        for i,fi in enumerate(fd):
            for j,a in enumerate(ang):
                w.writerow([float(fi),float(a),float(rel[i,j])])
    cmp={
        'stage':'Stage 4A electro-mechanical-acoustic baseline',
        'status':'usable scalar Study-2 baseline; not yet full solid/acoustic FEM ASB matrix',
        'parameters':asdict(params),
        'blocked_impedance_csv':str(args.blocked_impedance_csv),
        'visual_anchor_errors_complete':visual_anchor_errors(complete),
        'visual_anchor_errors_without_nra':visual_anchor_errors(no_nra),
        'figure_targets':{
            'Figure_8':'SPL at 1 m for V0=3.55 V; flat preferred range roughly 100-1500 Hz; lossless NRA shows steep resonances around 600 and 1300 Hz.',
            'Figure_10':'Total electric impedance Z=V0/I; peak about 50 Hz, DC resistance 5.6 ohm, 100 Hz-1 kHz roughly 6.3-10.4 ohm, high frequency rises by voice-coil inductance.',
            'Figure_12':'Directivity normalized to 0 deg, angles -90..90 deg, radius 1 m.'
        }
    }
    write_json(outdir/'stage4_summary.json', cmp, indent=2)
    if not args.no_plots:
        plot_sensitivity(outdir/'figure8_stage4_sensitivity_phase.png', complete, no_nra)
        plot_impedance(outdir/'figure10_stage4_total_electric_impedance.png', complete)
        plot_power_efficiency(outdir/'stage4_coil_power_efficiency.png', complete)
        plot_directivity(outdir/'figure12_stage4_directivity.png', fd, ang, rel)
    report=f"""# Stage 4A 电-机-声 Study 2 基线报告\n\n本阶段接入 Stage 2 的 RAW BL 闭合值和 Stage 3C 的 blocked impedance，建立完整标量电-机-声链路：\n\n```text\nZ_total = Zb + BL^2 / Zm\nI = V0 / Z_total\nv = BL*I / Zm\np_1m = j*omega*rho0*Sd*v/(2*pi*r)\n```\n\n该阶段用于复现 COMSOL Figure 8/10 的一维标量结果，并生成 Figure 12 的活塞/破裂模态指向性基线。它不是最终的 full solid/acoustic FEM Acoustic-Structure Boundary 矩阵。\n\n## 参数\n\n```json\n{dumps_json(asdict(params), indent=2)}\n```\n\n## 与 PDF 图示目标的误差摘要\n\n```json\n{dumps_json(cmp['visual_anchor_errors_complete'], indent=2)}\n```\n\n## 输出\n\n- `stage4_complete_response.csv`\n- `stage4_without_nra_response.csv`\n- `figure8_stage4_sensitivity_phase.png`\n- `figure10_stage4_total_electric_impedance.png`\n- `figure12_stage4_directivity.png`\n- `stage4_coil_power_efficiency.png`\n\n## 判断\n\nStage 4A 已可用于后续声固耦合 FEM 的数值基线：阻抗峰、DC resistance、100–1500 Hz 灵敏度平台、NRA/no-NRA 600/1300 Hz 差异均可检查。下一步应把 `Zm` 从 lumped SDOF 替换为 axisymmetric solid FEM 动刚度，并把 piston far-field 替换为 COMSOL geometry 上的 acoustic-structure boundary + exterior field。\n"""
    (outdir/'STAGE4_ELECTROACOUSTIC_REPORT_CN.md').write_text(report, encoding='utf-8')
    print(dumps_json(cmp, indent=2))


if __name__=='__main__':
    main()
