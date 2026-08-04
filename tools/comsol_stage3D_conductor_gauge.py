#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from loudspeaker_axisym_fem.axisym_magnetics import (
    load_tagged_meshio,
    solve_axisymmetric_magnetostatics,
    solve_conductor_gauge_voltage_coil_impedance,
    solve_conductor_gauge_fixed_current_coil_impedance,
    solve_voltage_constrained_blocked_coil_impedance,
    write_blocked_impedance_csv,
    write_blocked_impedance_vtu,
    plot_blocked_impedance,
    plot_induced_current_density,
    _coil_conductor_sigma_from_Rdc,
)
from loudspeaker_axisym_fem.comsol_driver_model import SOFT_IRON_BH_TABLE, COMSOL_TARGETS, ComsolDriverParameters
from loudspeaker_axisym_fem.json_utils import write_json, dumps_json


def parse_freqs(s: str):
    if s.strip().lower() in ('default', 'anchors'):
        return [1, 50, 100, 900, 1000, 8000]
    return [float(x.strip()) for x in s.replace(';', ',').split(',') if x.strip()]


def mesh_summary(mesh):
    import numpy as np
    ids, counts = np.unique(mesh.tri_domains, return_counts=True)
    d = dict(zip([int(x) for x in ids], [int(x) for x in counts]))
    return {
        'nodes': mesh.n_nodes,
        'triangles': mesh.n_triangles,
        'domain_triangle_counts': {str(k): d.get(k, 0) for k in [6, 23, 24, 17, 18, 19, 8, 22]},
    }


def visual_targets():
    return {1:1.78, 50:1.75, 100:1.58, 900:1.28, 1000:1.24, 8000:0.80}


def make_comparison(freqs, result):
    import csv
    rows=[]
    targets=visual_targets()
    for f,z,L,I in zip(result.frequencies_Hz, result.Zb_ohm, result.Lb_H, result.coil_current_A):
        target=targets.get(int(round(float(f))))
        rows.append({
            'f_Hz': float(f),
            'Z_real_ohm': float(z.real),
            'Z_imag_ohm': float(z.imag),
            'Z_abs_ohm': float(abs(z)),
            'Lb_mH': float(L*1e3),
            'I_abs_A': float(abs(I)),
            'target_L_mH_visual': target,
            'err_percent_vs_visual': None if target is None else 100.0*(float(L*1e3)-target)/target,
        })
    return rows


def write_comparison_csv(path, rows):
    import csv
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as fp:
        w=csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description='Stage-3D COMSOL-like homogenized multiturn conductor/gauge voltage coil.')
    ap.add_argument('--mesh', default=str(ROOT / 'reports/comsol_reproduction_stage1/comsol_geometry_polyline_coarse_2p5mm.msh'))
    ap.add_argument('--outdir', default=str(ROOT / 'outputs_comsol_reproduction_stage3D_conductor_gauge'))
    ap.add_argument('--freqs', default='anchors')
    ap.add_argument('--sigma-soft-iron', type=float, default=1.12e7)
    ap.add_argument('--linearized-mu-mode', choices=['differential','effective','stage2'], default='differential')
    ap.add_argument('--voltage-V', type=float, default=1.0)
    ap.add_argument('--Rdc-ohm', type=float, default=COMSOL_TARGETS['dc_resistance_ohm'])
    ap.add_argument('--sigma-coil', type=float, default=None, help='default computes homogenized conductor conductivity to match Rdc on this geometry')
    ap.add_argument('--voltage-distribution', choices=['series_per_turn','total_loop'], default='series_per_turn')
    ap.add_argument('--no-coil-induced-current', action='store_true')
    ap.add_argument('--gauge-solve-mode', choices=['prescribed_voltage','fixed_current_global_V'], default='prescribed_voltage')
    ap.add_argument('--static-max-iter', type=int, default=35)
    ap.add_argument('--calibrate-static-bl', action='store_true', default=False)
    ap.add_argument('--static-relaxation', type=float, default=0.1)
    ap.add_argument('--static-nonlinear-update-mode', choices=['H_forward','B_inverse'], default='B_inverse')
    ap.add_argument('--static-tol', type=float, default=1e-4)
    ap.add_argument('--store-field-freqs', default='50,900')
    ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--also-run-stage3c-terminal', action='store_true', help='write previous global-current voltage terminal baseline into same summary for comparison')
    args=ap.parse_args()

    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    params=ComsolDriverParameters()
    mesh=load_tagged_meshio(args.mesh)
    static=solve_axisymmetric_magnetostatics(
        mesh,
        soft_iron_domains=(6,23), magnet_domains=(24,), coil_domains=(17,18,19),
        N0=params.N0, remanence_T=0.4, target_BL_N_A=COMSOL_TARGETS['BL_N_per_A'],
        bh_table=SOFT_IRON_BH_TABLE,
        max_iter=args.static_max_iter, relaxation=args.static_relaxation, tol=args.static_tol,
        mu_r_initial_soft=700.0, calibrate_to_BL=args.calibrate_static_bl,
        nonlinear_update_mode=args.static_nonlinear_update_mode,
    )
    sigma_coil=args.sigma_coil
    if sigma_coil is None:
        sigma_coil = _coil_conductor_sigma_from_Rdc(mesh, coil_domains=(17,18,19), N0=params.N0, Rdc_ohm=args.Rdc_ohm, voltage_distribution=args.voltage_distribution)
    freqs=parse_freqs(args.freqs)
    store_freqs=parse_freqs(args.store_field_freqs)
    if args.gauge_solve_mode == 'fixed_current_global_V':
        result=solve_conductor_gauge_fixed_current_coil_impedance(
            static, freqs,
            bh_table=SOFT_IRON_BH_TABLE,
            soft_iron_domains=(6,23), conducting_domains=(6,23), coil_domains=(17,18,19),
            N0=params.N0, Rdc_ohm=args.Rdc_ohm,
            sigma_soft_iron_S_m=args.sigma_soft_iron,
            sigma_coil_S_m=sigma_coil,
            linearized_mu_mode=args.linearized_mu_mode,
            current_A=1.0,
            voltage_distribution=args.voltage_distribution,
            store_field_frequencies=store_freqs,
        )
    else:
        result=solve_conductor_gauge_voltage_coil_impedance(
            static, freqs,
            bh_table=SOFT_IRON_BH_TABLE,
            soft_iron_domains=(6,23), conducting_domains=(6,23), coil_domains=(17,18,19),
            N0=params.N0, Rdc_ohm=args.Rdc_ohm,
            sigma_soft_iron_S_m=args.sigma_soft_iron,
            sigma_coil_S_m=sigma_coil,
            linearized_mu_mode=args.linearized_mu_mode,
            voltage_V=args.voltage_V,
            voltage_distribution=args.voltage_distribution,
            include_coil_induced_current=not args.no_coil_induced_current,
            store_field_frequencies=store_freqs,
        )
    write_blocked_impedance_csv(outdir/'blocked_impedance_conductor_gauge.csv', result)
    comp=make_comparison(freqs, result)
    write_comparison_csv(outdir/'stage3D_figure6_comparison.csv', comp)
    for f in store_freqs:
        if float(f) in result.A_phi_by_frequency:
            write_blocked_impedance_vtu(outdir/f'conductor_gauge_field_{int(round(f))}Hz.vtu', result, float(f))
    if not args.no_plots:
        plot_blocked_impedance(outdir/'figure6_stage3D_conductor_gauge', result)
        for f in store_freqs:
            if float(f) in result.A_phi_by_frequency:
                plot_induced_current_density(outdir/'figure5_stage3D_conductor_gauge', result, float(f), quantity='real')
    baseline_summary=None
    if args.also_run_stage3c_terminal:
        base=solve_voltage_constrained_blocked_coil_impedance(
            static, freqs,
            bh_table=SOFT_IRON_BH_TABLE,
            soft_iron_domains=(6,23), conducting_domains=(6,23), coil_domains=(17,18,19),
            N0=params.N0, Rdc_ohm=args.Rdc_ohm,
            sigma_soft_iron_S_m=args.sigma_soft_iron,
            linearized_mu_mode=args.linearized_mu_mode,
            voltage_V=args.voltage_V,
            store_field_frequencies=(),
            solve_mode='schur',
        )
        write_blocked_impedance_csv(outdir/'blocked_impedance_stage3C_global_terminal_baseline.csv', base)
        baseline_summary=base.summary()
    summary={
        'stage': 'Stage 3D conductor/gauge homogenized multiturn voltage coil',
        'settings': vars(args),
        'mesh': str(args.mesh),
        'mesh_summary': mesh_summary(mesh),
        'sigma_coil_S_m': float(sigma_coil),
        'static_summary': static.summary(),
        'blocked_summary': result.summary(),
        'figure6_comparison': comp,
        'stage3C_global_terminal_baseline_summary': baseline_summary,
        'notes': [
            'This formulation prescribes electric scalar potential gradient inside the coil domain rather than imposing a lumped current source only.',
            'J_phi in coil includes drive term sigma*V/(N*2*pi*r) and induced term -i*omega*sigma*A_phi.',
            'The terminal current is I=(1/N)*integral_A J_phi dA and Z=V/I.',
        ],
    }
    write_json(outdir/'stage3D_conductor_gauge_summary.json', summary, indent=2)
    print(dumps_json(summary, indent=2))


if __name__ == '__main__':
    main()
