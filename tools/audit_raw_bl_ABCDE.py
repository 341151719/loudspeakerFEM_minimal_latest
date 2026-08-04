#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from loudspeaker_axisym_fem.axisym_magnetics import (
    TaggedTriMesh,
    load_tagged_meshio,
    solve_axisymmetric_magnetostatics,
    compute_bl_from_elements,
    _tri_geometry,
    interp_h_from_b,
    MU0,
)
from loudspeaker_axisym_fem.comsol_driver_model import SOFT_IRON_BH_TABLE, COMSOL_TARGETS, ComsolDriverParameters
from loudspeaker_axisym_fem.comsol_geom_mphtxt import parse_mphtxt
from loudspeaker_axisym_fem.json_utils import dumps_json, write_json

COIL_DOMAINS = (17, 18, 19)
SOFT_IRON_DOMAINS = (6, 23)
MAGNET_DOMAINS = (24,)
MAGNETIC_DOMAINS = (6, 17, 18, 19, 23, 24)
DEFAULT_EXTERIOR_BOUNDARIES = (1,2,3,4,5,83,84,85,86,87,88,89,94)


def element_masks(mesh, domains):
    ds = set(int(d) for d in domains)
    return np.array([int(d) in ds for d in mesh.tri_domains], dtype=bool)


def bl_audit(mesh, B_r, *, coil_domains=COIL_DOMAINS, N0=100):
    pts = mesh.points_rz_m
    tris = mesh.triangles
    area, cent, _, _ = _tri_geometry(pts, tris)
    mask = element_masks(mesh, coil_domains)
    r = cent[:, 0]
    f = -2.0 * math.pi * float(N0) * r * B_r
    area_total = float(np.sum(area[mask]))
    vol_w = 2.0 * math.pi * r * area
    vol_total = float(np.sum(vol_w[mask]))
    area_avg = float(np.sum(f[mask] * area[mask]) / area_total)
    vol_avg = float(np.sum(f[mask] * vol_w[mask]) / vol_total)
    # A simple thin-wire surrogate: area-average Br by domain then weight each coil slice by its circumference and 2D area.
    contrib_rows=[]
    for d in coil_domains:
        m = mask & (mesh.tri_domains == int(d))
        if not np.any(m):
            continue
        A = float(np.sum(area[m]))
        rw = float(np.sum(r[m]*area[m]) / A)
        Br_avg = float(np.sum(B_r[m]*area[m]) / A)
        f_avg = float(np.sum(f[m]*area[m]) / A)
        axisym_A = float(np.sum(vol_w[m]))
        contrib_rows.append({
            'domain': int(d),
            'n_triangles': int(np.count_nonzero(m)),
            'area_m2': A,
            'centroid_r_m': rw,
            'centroid_z_m': float(np.sum(cent[m,1]*area[m]) / A),
            'Br_area_avg_T': Br_avg,
            'BL_area_avg_N_A': f_avg,
            'axisym_volume_weight_m3_per_rad_times_2pi': axisym_A,
            'BL_area_integral_contribution': float(np.sum(f[m]*area[m])),
        })
    # domain-area weighted average equals COMSOL AvSurface if selection is domains 17-19 and intvolume=false.
    thin_wire = float(np.sum([row['BL_area_avg_N_A'] * row['area_m2'] for row in contrib_rows]) / max(area_total, 1e-300))
    # Fill-factor variant is identical without a known nonuniform fill factor; included to make this explicit.
    fill_factor_weighted = thin_wire
    return {
        'BL_comsol_area_average_N_A': area_avg,
        'BL_axisymmetric_volume_average_N_A': vol_avg,
        'BL_fill_factor_uniform_N_A': fill_factor_weighted,
        'BL_thin_wire_surrogate_N_A': thin_wire,
        'coil_area_m2': area_total,
        'coil_axisymmetric_volume_weight': vol_total,
        'coil_domain_rows': contrib_rows,
        'Br_over_coil_min_T': float(np.min(B_r[mask])),
        'Br_over_coil_max_T': float(np.max(B_r[mask])),
        'Br_over_coil_area_avg_T': float(np.sum(B_r[mask]*area[mask]) / area_total),
        'r_bounds_m': [float(np.min(cent[mask,0])), float(np.max(cent[mask,0]))],
        'z_bounds_m': [float(np.min(cent[mask,1])), float(np.max(cent[mask,1]))],
    }


def bh_residual_audit(mesh, B_norm, mu_r_elem, *, soft_domains=SOFT_IRON_DOMAINS, table=SOFT_IRON_BH_TABLE):
    soft = element_masks(mesh, soft_domains)
    B = np.maximum(B_norm[soft], 0.0)
    H_inv = interp_h_from_b(B, table)
    mu_inv = np.where(H_inv > 1e-30, B/(MU0*np.maximum(H_inv,1e-30)), 1.0)
    mu_old = mu_r_elem[soft]
    rel = (mu_old - mu_inv) / np.maximum(np.abs(mu_inv), 1.0)
    return {
        'soft_elements': int(np.count_nonzero(soft)),
        'mu_current_min': float(np.min(mu_old)),
        'mu_current_max': float(np.max(mu_old)),
        'mu_B_inverse_min': float(np.min(mu_inv)),
        'mu_B_inverse_max': float(np.max(mu_inv)),
        'mu_relative_error_rms': float(np.sqrt(np.mean(rel**2))),
        'mu_relative_error_maxabs': float(np.max(np.abs(rel))),
        'H_from_B_min_A_m': float(np.min(H_inv)),
        'H_from_B_max_A_m': float(np.max(H_inv)),
    }


def submesh_by_domains(mesh: TaggedTriMesh, domains) -> TaggedTriMesh:
    mask = element_masks(mesh, domains)
    tris_old = mesh.triangles[mask]
    doms = mesh.tri_domains[mask]
    used = np.unique(tris_old.ravel())
    new_index = -np.ones(mesh.n_nodes, dtype=int)
    new_index[used] = np.arange(len(used))
    pts = mesh.points_rz_m[used]
    tris = new_index[tris_old]
    # derive boundary edges from submesh triangles
    edge_count = defaultdict(int)
    for tri in tris:
        for a,b in [(tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])]:
            e = tuple(sorted((int(a), int(b))))
            edge_count[e]+=1
    lines = np.array([e for e,c in edge_count.items() if c==1], dtype=int)
    tags = np.zeros(len(lines), dtype=int)
    return TaggedTriMesh(pts, tris, doms, lines, tags)


def geometry_audit(mphtxt_path: Path, mesh: TaggedTriMesh):
    geom = parse_mphtxt(mphtxt_path)
    inv = geom.inventory()
    # Mesh area per domain vs mphtxt polygon area. mm^2 -> m^2 compare with mesh.
    area, cent, _, _ = _tri_geometry(mesh.points_rz_m, mesh.triangles)
    rows=[]
    for d in sorted(set(mesh.tri_domains.astype(int))):
        m = mesh.tri_domains == d
        mesh_area = float(np.sum(area[m]))
        invd = inv['domain_loops'].get(str(int(d)), {})
        mphtxt_area_m2 = abs(float(invd.get('area_mm2_abs', float('nan'))))*1e-6
        rel = None if not np.isfinite(mphtxt_area_m2) or mphtxt_area_m2 == 0 else (mesh_area-mphtxt_area_m2)/mphtxt_area_m2
        rows.append({
            'domain': int(d),
            'mesh_area_m2': mesh_area,
            'mphtxt_area_m2': mphtxt_area_m2,
            'relative_area_error': rel,
            'mesh_centroid_r_m': float(np.sum(cent[m,0]*area[m])/max(mesh_area,1e-300)),
            'mesh_centroid_z_m': float(np.sum(cent[m,1]*area[m])/max(mesh_area,1e-300)),
            'mphtxt_centroid_r_m': float(invd.get('centroid_r_mm', float('nan')))*1e-3,
            'mphtxt_centroid_z_m': float(invd.get('centroid_z_mm', float('nan')))*1e-3,
            'n_triangles': int(np.count_nonzero(m)),
        })
    curve_degrees = inv.get('curve_degrees', {})
    return {
        'mphtxt_inventory': inv,
        'mesh_vs_mphtxt_domain_area_rows': rows,
        'all_curves_degree_1': curve_degrees == {'1': inv.get('curves')},
        'curve_degrees': curve_degrees,
        'rational_curves': inv.get('rational_curves'),
    }


def run_variant(name, mesh, *, max_iter, tol, relaxation, mu_initial, update_mode='H_forward', remanence=0.4, rem_sign=1.0, exterior_boundaries=DEFAULT_EXTERIOR_BOUNDARIES, calibrate=False):
    try:
        res = solve_axisymmetric_magnetostatics(
            mesh,
            soft_iron_domains=SOFT_IRON_DOMAINS,
            magnet_domains=MAGNET_DOMAINS,
            coil_domains=COIL_DOMAINS,
            N0=ComsolDriverParameters().N0,
            remanence_T=remanence,
            target_BL_N_A=COMSOL_TARGETS['BL_N_per_A'],
            bh_table=SOFT_IRON_BH_TABLE,
            max_iter=max_iter,
            tol=tol,
            relaxation=relaxation,
            mu_r_initial_soft=mu_initial,
            calibrate_to_BL=calibrate,
            nonlinear_update_mode=update_mode,
            remanence_rhs_sign=rem_sign,
            exterior_boundary_ids=exterior_boundaries,
        )
        return res, {'status':'ok','BL_raw_N_A':res.bl_raw_N_A,'BL_error_percent':100*(res.bl_raw_N_A-COMSOL_TARGETS['BL_N_per_A'])/COMSOL_TARGETS['BL_N_per_A'],'iterations':res.iterations,'residual_last':res.residual_history[-1] if res.residual_history else None,'Bmax_T':float(np.max(res.B_norm)),'Hmax_A_m':float(np.max(res.H_norm)),'mu_max':float(np.max(res.mu_r_elem)),'update_mode':update_mode,'remanence_T':remanence,'remanence_rhs_sign':rem_sign}
    except Exception as e:
        return None, {'status':'failed','error':repr(e),'update_mode':update_mode,'remanence_T':remanence,'remanence_rhs_sign':rem_sign}


def main():
    ap = argparse.ArgumentParser(description='ABCDE RAW-BL first-principles audit and correction experiments.')
    ap.add_argument('--mesh', default=str(ROOT/'meshes/comsol_stable_1mm_05gap.msh'))
    ap.add_argument('--mphtxt', default=str(ROOT/'comsol_reference_inputs/Untitled.mphtxt'))
    ap.add_argument('--outdir', default=str(ROOT/'outputs/raw_bl_abcde'))
    ap.add_argument('--max-iter', type=int, default=10)
    ap.add_argument('--tol', type=float, default=3e-3)
    ap.add_argument('--relaxation', type=float, default=0.55)
    ap.add_argument('--mu-r-initial-soft', type=float, default=700.0)
    args = ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    mesh=load_tagged_meshio(args.mesh)
    mphtxt=Path(args.mphtxt)
    results={
        'target_BL_N_A': COMSOL_TARGETS['BL_N_per_A'],
        'mesh': str(args.mesh),
        'mphtxt': str(args.mphtxt),
        'settings': vars(args),
    }
    # D geometry first: no solve cost.
    D=geometry_audit(mphtxt, mesh)
    results['D_geometry_audit']=D
    # B/C/E variants.
    variants=[]
    full_res, row = run_variant('full_default_H_forward', mesh, max_iter=args.max_iter, tol=args.tol, relaxation=args.relaxation, mu_initial=args.mu_r_initial_soft, update_mode='H_forward', exterior_boundaries=DEFAULT_EXTERIOR_BOUNDARIES)
    row['variant']='full_default_H_forward'; variants.append(row)
    # A audit and C residual for baseline.
    if full_res is not None:
        results['A_BL_operator_audit_full_default']=bl_audit(full_res.mesh, full_res.B_r, N0=ComsolDriverParameters().N0)
        results['C_BH_residual_audit_full_default']=bh_residual_audit(full_res.mesh, full_res.B_norm, full_res.mu_r_elem)
    # C corrected B inverse update.
    res_binv, row = run_variant('full_B_inverse', mesh, max_iter=args.max_iter, tol=args.tol, relaxation=args.relaxation, mu_initial=args.mu_r_initial_soft, update_mode='B_inverse', exterior_boundaries=DEFAULT_EXTERIOR_BOUNDARIES)
    row['variant']='full_B_inverse'; variants.append(row)
    if res_binv is not None:
        results['A_BL_operator_audit_full_B_inverse']=bl_audit(res_binv.mesh, res_binv.B_r, N0=ComsolDriverParameters().N0)
        results['C_BH_residual_audit_full_B_inverse']=bh_residual_audit(res_binv.mesh, res_binv.B_norm, res_binv.mu_r_elem)
    # E remanence variants on full mesh.
    for rem, sign, name in [(0.4,-1.0,'full_rhs_sign_negative'),(-0.4,1.0,'full_remanence_negative')]:
        _, row = run_variant(name, mesh, max_iter=max(4,args.max_iter//2), tol=args.tol, relaxation=args.relaxation, mu_initial=args.mu_r_initial_soft, update_mode='B_inverse', exterior_boundaries=DEFAULT_EXTERIOR_BOUNDARIES, remanence=rem, rem_sign=sign)
        row['variant']=name; variants.append(row)
    # B box2 / magnetic domain variants.
    magmesh = submesh_by_domains(mesh, MAGNETIC_DOMAINS)
    results['B_box2_submesh_info']={'n_nodes':magmesh.n_nodes,'n_triangles':magmesh.n_triangles,'domains':sorted(set(magmesh.tri_domains.astype(int).tolist()))}
    # axis-only natural (line_tags 0, exterior none => only r=0 fixed)
    _, row = run_variant('box2_axis_only_natural', magmesh, max_iter=args.max_iter, tol=args.tol, relaxation=args.relaxation, mu_initial=args.mu_r_initial_soft, update_mode='B_inverse', exterior_boundaries=())
    row['variant']='box2_axis_only_natural'; variants.append(row)
    # all submesh boundary A0: pass boundary id 0 because derived lines tag zero.
    _, row = run_variant('box2_all_boundary_A0', magmesh, max_iter=args.max_iter, tol=args.tol, relaxation=args.relaxation, mu_initial=args.mu_r_initial_soft, update_mode='B_inverse', exterior_boundaries=(0,))
    row['variant']='box2_all_boundary_A0'; variants.append(row)
    results['variant_rows']=variants
    # Write CSVs
    import csv
    with (outdir/'abcde_variant_summary.csv').open('w',newline='',encoding='utf-8') as fp:
        keys=sorted(set().union(*(r.keys() for r in variants)))
        w=csv.DictWriter(fp, fieldnames=keys); w.writeheader(); w.writerows(variants)
    for key, rows in [('D_mesh_vs_mphtxt_domain_area.csv', D['mesh_vs_mphtxt_domain_area_rows'])]:
        with (outdir/key).open('w',newline='',encoding='utf-8') as fp:
            keys=sorted(set().union(*(r.keys() for r in rows)))
            w=csv.DictWriter(fp, fieldnames=keys); w.writeheader(); w.writerows(rows)
    # Decide correction: minimum abs BL error among non-calibrated ok variants.
    ok=[r for r in variants if r.get('status')=='ok' and np.isfinite(r.get('BL_error_percent', np.nan))]
    best=min(ok, key=lambda r: abs(r['BL_error_percent'])) if ok else None
    results['best_raw_variant']=best
    write_json(outdir/'raw_bl_ABCDE_audit_summary.json', results, indent=2)
    # MD report
    lines=[]
    lines.append('# RAW BL ABCDE 原理审查与修正报告')
    lines.append('')
    lines.append(f'- COMSOL BL target: {COMSOL_TARGETS["BL_N_per_A"]:.6g} N/A')
    lines.append(f'- Mesh: `{args.mesh}`')
    lines.append(f'- Mesh triangles: {mesh.n_triangles}, nodes: {mesh.n_nodes}')
    lines.append('')
    lines.append('## A. BL 后处理算子')
    if 'A_BL_operator_audit_full_default' in results:
        A=results['A_BL_operator_audit_full_default']
        lines.append(f'- COMSOL-style 2D area average: {A["BL_comsol_area_average_N_A"]:.6g} N/A')
        lines.append(f'- Axisymmetric volume average: {A["BL_axisymmetric_volume_average_N_A"]:.6g} N/A')
        lines.append(f'- Thin-wire surrogate / uniform fill factor: {A["BL_thin_wire_surrogate_N_A"]:.6g} N/A')
        lines.append('- 判断：如果使用 COMSOL `.m` 中 `AvSurface intvolume=false` 的语义，应使用 2D area average。其它权重不是 Figure BL 的默认定义。')
    lines.append('')
    lines.append('## B/C/E. 变体结果')
    lines.append('')
    lines.append('| variant | status | BL raw N/A | error % | iterations | Bmax T | mu max |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|')
    for r in variants:
        lines.append(f"| {r.get('variant')} | {r.get('status')} | {r.get('BL_raw_N_A', float('nan')):.6g} | {r.get('BL_error_percent', float('nan')):.3f} | {r.get('iterations','')} | {r.get('Bmax_T', float('nan')):.4g} | {r.get('mu_max', float('nan')):.4g} |")
    lines.append('')
    lines.append('## C. B-H 残差审查')
    for k in ['C_BH_residual_audit_full_default','C_BH_residual_audit_full_B_inverse']:
        if k in results:
            C=results[k]
            lines.append(f'- {k}: μ relative RMS={C["mu_relative_error_rms"]:.4g}, max={C["mu_relative_error_maxabs"]:.4g}, μ current max={C["mu_current_max"]:.3g}, μ B-inverse max={C["mu_B_inverse_max"]:.3g}')
    lines.append('')
    lines.append('## D. Bezier/几何审查')
    lines.append(f'- mphtxt curves: {D["mphtxt_inventory"].get("curves")}, degrees: {D["curve_degrees"]}, rational: {D["rational_curves"]}')
    lines.append(f'- all curves degree 1: {D["all_curves_degree_1"]}')
    area_rows=D['mesh_vs_mphtxt_domain_area_rows']
    coil_rows=[r for r in area_rows if r['domain'] in COIL_DOMAINS]
    for r in coil_rows:
        lines.append(f"- coil domain {r['domain']}: mesh area={r['mesh_area_m2']:.6e} m², mphtxt area={r['mphtxt_area_m2']:.6e} m², rel area error={r['relative_area_error']:.3%}, triangles={r['n_triangles']}")
    lines.append('')
    lines.append('## E. Remanence/弱式方向')
    lines.append('- +0.4T 与符号反转变体主要改变 BL 符号/方向，不解决幅值偏高；若结果表显示幅值近似相同，说明主要问题不是简单磁体方向。')
    lines.append('')
    lines.append('## 最佳 RAW 变体')
    if best:
        lines.append(f"- `{best['variant']}`: BL={best['BL_raw_N_A']:.6g} N/A, error={best['BL_error_percent']:.3f}%")
    else:
        lines.append('- no successful variant')
    lines.append('')
    lines.append('## 结论')
    lines.append('本工具执行 ABCDE 审查后，只接受能在 RAW 层降低 BL 误差且物理定义与 COMSOL 一致的变体。若最佳变体仍大于约 5%，不得把后续 calibration 误称为 Stage 2 原理闭合。')
    (outdir/'RAW_BL_ABCDE_REPORT_CN.md').write_text('\n'.join(lines), encoding='utf-8')
    print(dumps_json({'outdir':str(outdir),'best_raw_variant':best,'variant_count':len(variants)}, indent=2))

if __name__=='__main__':
    main()
