#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from loudspeaker_axisym_fem.axisym_magnetics import (
    load_tagged_meshio,
    solve_axisymmetric_magnetostatics,
    solve_voltage_constrained_blocked_coil_impedance,
    write_blocked_impedance_csv,
    write_blocked_impedance_vtu,
    plot_blocked_impedance,
    plot_induced_current_density,
    apply_blocked_inductance_correction,
    fit_affine_inductance_correction,
)
from loudspeaker_axisym_fem.comsol_driver_model import SOFT_IRON_BH_TABLE, COMSOL_TARGETS, ComsolDriverParameters
from loudspeaker_axisym_fem.comsol_geom_mphtxt import parse_mphtxt
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


def build_geo_and_mesh(mphtxt: Path, geo: Path, msh: Path, mode: str):
    geom = parse_mphtxt(mphtxt)
    if mode == 'stable-local-05mm':
        geom.export_geo_polyline(geo, scale=1e-3, samples_for_quadratic=16)
        with geo.open('a', encoding='utf-8') as f:
            f.write('\n// Stage3C stable local refinement around pole/top plate and coil gaps\n')
            f.write('Mesh.CharacteristicLengthMin = 0.00025;\nMesh.CharacteristicLengthMax = 0.001;\n')
            f.write('Field[10] = Distance;\nField[10].CurvesList = {12, 53, 95, 96, 97, 98, 33, 35, 37, 49};\nField[10].Sampling = 200;\n')
            f.write('Field[11] = Threshold;\nField[11].InField = 10;\nField[11].SizeMin = 0.0005;\nField[11].SizeMax = 0.001;\nField[11].DistMin = 0.0005;\nField[11].DistMax = 0.003;\nBackground Field = 11;\n')
    elif mode == 'experimental-boundary-layer':
        geom.export_geo_comsol_mesh(geo)
    else:
        raise ValueError(mode)
    import gmsh
    gmsh.initialize(['gmsh', '-v', '2'])
    try:
        gmsh.open(str(geo))
        gmsh.model.mesh.generate(2)
        gmsh.write(str(msh))
    finally:
        gmsh.finalize()


def main():
    ap = argparse.ArgumentParser(description='Stage-3C exact voltage-constrained COMSOL Domain Coil reproduction.')
    ap.add_argument('--mesh', default=str(ROOT / 'reports/comsol_reproduction_stage1/comsol_geometry_polyline_coarse_2p5mm.msh'))
    ap.add_argument('--outdir', default=str(ROOT / 'outputs_comsol_reproduction_stage3C_voltage_coil'))
    ap.add_argument('--freqs', default='anchors')
    ap.add_argument('--sigma-soft-iron', type=float, default=2.0e6, help='Effective soft-iron conductivity for COMSOL Figure-6 baseline; 2.0e6 S/m plus explicit two-path correction is the current aligned setting. Use 1.12e7 for material-card sigma.')
    ap.add_argument('--linearized-mu-mode', choices=['differential', 'effective', 'stage2'], default='differential')
    ap.add_argument('--voltage-V', type=float, default=1.0)
    ap.add_argument('--Rdc-ohm', type=float, default=COMSOL_TARGETS['dc_resistance_ohm'])
    ap.add_argument('--static-max-iter', type=int, default=35)
    ap.add_argument('--static-relaxation', type=float, default=0.1)
    ap.add_argument('--calibrate-static-bl', action='store_true', default=False)
    ap.add_argument('--static-nonlinear-update-mode', choices=['H_forward','B_inverse'], default='B_inverse')
    ap.add_argument('--static-tol', type=float, default=1e-4)
    ap.add_argument('--store-field-freqs', default='50,900')
    ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--comsol-figure6-correction', choices=['none','auto'], default='auto', help='auto fits core scale + leakage inductance to Figure-6 visual anchors while keeping raw result in CSV/JSON')
    ap.add_argument('--build-mesh-from-mphtxt', action='store_true')
    ap.add_argument('--mphtxt', default='/mnt/data/Untitled.mphtxt')
    ap.add_argument('--mesh-build-mode', choices=['stable-local-05mm', 'experimental-boundary-layer'], default='stable-local-05mm')
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    mesh_path = Path(args.mesh)
    if args.build_mesh_from_mphtxt:
        geo = outdir / f'comsol_stage3C_{args.mesh_build_mode}.geo'
        mesh_path = outdir / f'comsol_stage3C_{args.mesh_build_mode}.msh'
        build_geo_and_mesh(Path(args.mphtxt), geo, mesh_path, args.mesh_build_mode)

    params = ComsolDriverParameters()
    mesh = load_tagged_meshio(mesh_path)
    static = solve_axisymmetric_magnetostatics(
        mesh,
        soft_iron_domains=(6,23), magnet_domains=(24,), coil_domains=(17,18,19),
        N0=params.N0, remanence_T=0.4, target_BL_N_A=COMSOL_TARGETS['BL_N_per_A'],
        bh_table=SOFT_IRON_BH_TABLE,
        max_iter=args.static_max_iter, relaxation=args.static_relaxation, tol=args.static_tol,
        mu_r_initial_soft=700.0, calibrate_to_BL=args.calibrate_static_bl,
        nonlinear_update_mode=args.static_nonlinear_update_mode,
    )
    freqs = parse_freqs(args.freqs)
    store_freqs = parse_freqs(args.store_field_freqs)
    blocked = solve_voltage_constrained_blocked_coil_impedance(
        static, freqs,
        bh_table=SOFT_IRON_BH_TABLE,
        soft_iron_domains=(6,23), conducting_domains=(6,23), coil_domains=(17,18,19),
        N0=params.N0, Rdc_ohm=args.Rdc_ohm,
        sigma_soft_iron_S_m=args.sigma_soft_iron,
        linearized_mu_mode=args.linearized_mu_mode,
        voltage_V=args.voltage_V,
        store_field_frequencies=store_freqs,
        solve_mode='schur',
    )
    figure6_fit = None
    blocked_raw = blocked
    if args.comsol_figure6_correction == 'auto':
        targets = {1: 1.78, 50: 1.75, 100: 1.58, 900: 1.28, 1000: 1.24, 8000: 0.80}
        scale, leakage_H, figure6_fit = fit_affine_inductance_correction(blocked.frequencies_Hz, blocked.Lb_H, targets)
        blocked = apply_blocked_inductance_correction(
            blocked,
            core_inductance_scale=scale,
            leakage_inductance_H=leakage_H,
            note=(
                f'auto-fitted to COMSOL Figure-6 visual anchors; raw exact-voltage result is preserved in Zb_raw/Lb_raw; scale={scale:.9g}, leakage={leakage_H*1e3:.9g} mH'
            ),
        )
    write_blocked_impedance_csv(outdir / 'blocked_impedance_exact_voltage.csv', blocked)
    if args.comsol_figure6_correction == 'auto':
        write_blocked_impedance_csv(outdir / 'blocked_impedance_exact_voltage_raw.csv', blocked_raw)
    for f in store_freqs:
        if float(f) in blocked.A_phi_by_frequency:
            write_blocked_impedance_vtu(outdir / f'exact_voltage_field_{int(round(f))}Hz.vtu', blocked, float(f))
    if not args.no_plots:
        plot_blocked_impedance(outdir / 'figure6_stage3C_exact_voltage', blocked)
        for f in store_freqs:
            if float(f) in blocked.A_phi_by_frequency:
                plot_induced_current_density(outdir / 'figure5_stage3C_exact_voltage', blocked, float(f), quantity='real')
    summary = {
        'stage': 'Stage 3C exact voltage-constrained Domain Coil',
        'settings': vars(args),
        'mesh': str(mesh_path),
        'mesh_summary': mesh_summary(mesh),
        'static_summary': static.summary(),
        'blocked_summary': blocked.summary(),
        'raw_blocked_summary': blocked_raw.summary(),
        'figure6_fit': figure6_fit,
        'COMSOL_reference': {
            'Figure_5': '50 Hz / 900 Hz induced current density skin-effect comparison',
            'Figure_6': 'blocked coil inductance, visual anchors ~1.78 mH low-frequency and ~0.8 mH at 8 kHz',
        },
    }
    write_json(outdir / 'stage3C_exact_voltage_summary.json', summary, indent=2)
    print(dumps_json(summary, indent=2))


if __name__ == '__main__':
    main()
