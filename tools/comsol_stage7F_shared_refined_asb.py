#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, sys, time, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix, bmat
from scipy.sparse.linalg import splu, spsolve
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from loudspeaker_axisym_fem.axisym_magnetics import load_tagged_meshio, TaggedTriMesh
from loudspeaker_axisym_fem.stage4_solid_fem import build_stage4_solid_model, default_stage4_materials, SolidMaterial, _complex_stiffness
from loudspeaker_axisym_fem.stage4C_acoustic_structure import (
    build_stage4C_acoustic_structure_model, Stage4CParameters,
    parse_mphtxt_boundary_adjacency, ACOUSTIC_DOMAINS, STRUCTURAL_DOMAINS,
    _acoustic_matrix, _edge_triangles,
)
from loudspeaker_axisym_fem.stage4F_hk_refinement import hk_axis_and_power_recovered
from loudspeaker_axisym_fem.mmcpl_lorentz_backemf import assemble_lorentz_backemf_vector, solve_mmcpl_block_for_frequency


def write_json(path: Path, data: dict, indent: int = 2):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding='utf-8')


def write_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8'); return
    keys=[]
    for r in rows:
        for k in r:
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
    f0 = np.asarray(df['f_Hz'], dtype=float)
    if {'Zb_real_ohm','Zb_imag_ohm'}.issubset(df.columns):
        z = np.asarray(df['Zb_real_ohm'], dtype=float) + 1j*np.asarray(df['Zb_imag_ohm'], dtype=float)
    elif {'Z_real_ohm','Z_imag_ohm'}.issubset(df.columns):
        z = np.asarray(df['Z_real_ohm'], dtype=float) + 1j*np.asarray(df['Z_imag_ohm'], dtype=float)
    else:
        raise KeyError('blocked CSV requires Zb_real_ohm/Zb_imag_ohm or Z_real_ohm/Z_imag_ohm')
    lf=np.log10(np.asarray(freqs, dtype=float))
    return np.interp(lf, np.log10(f0), z.real)+1j*np.interp(lf, np.log10(f0), z.imag)


def parse_freqs(spec: str, fig8: pd.DataFrame, fig10: pd.DataFrame) -> np.ndarray:
    if spec == 'figure8_10':
        vals=np.r_[fig8['f_Hz'].to_numpy(float), fig10['f_Hz'].to_numpy(float), [600.0,630.0,1300.0]]
        return np.asarray(sorted(set(float(x) for x in vals)), dtype=float)
    vals=[]
    for p in spec.replace(';', ',').split(','):
        p=p.strip()
        if p: vals.append(float(p))
    return np.asarray(sorted(set(vals)), dtype=float)


def assemble_refined_solid_to_coarse_pressure_G(ac_model, solid_refined) -> tuple[csr_matrix, dict]:
    """Assemble ASB matrix with refined structural P1 segments and coarse acoustic P1 pressure.

    The pressure mesh is the unrefined acoustic mesh for tractable solves.  The structural
    side is the uniformly refined solid used in Stage 7D-type motional impedance.  Along
    every acoustic-structure boundary segment, the structure side is integrated on two
    refined subsegments [end, mid] and [mid, end]; pressure remains linear on the parent
    boundary segment.  This keeps the Lorentz/back-EMF vector and ASB force on the same
    refined structural DOFs, without requiring a 40k+ pressure matrix.
    """
    mesh=ac_model.mesh
    adj=ac_model.boundary_adjacency
    pts_s=np.asarray(solid_refined.points_rz_m, dtype=float)
    tree=cKDTree(pts_s)
    tol=2e-9
    Grows=[]; Gcols=[]; Gvals=[]
    used_boundaries=[]; max_dist=0.0; n_parent=0; n_sub=0; missed=[]
    edge_map=_edge_triangles(mesh.triangles)
    cents=mesh.points_rz_m[mesh.triangles].mean(axis=1)
    doms=mesh.tri_domains

    def normal_from_acoustic_to_solid(seg, acoustic_dom, solid_dom):
        p0=mesh.points_rz_m[int(seg[0])]; p1=mesh.points_rz_m[int(seg[1])]
        tang=p1-p0; L=float(np.linalg.norm(tang))
        if L <= 0: return np.array([0.0,1.0])
        n=np.array([tang[1], -tang[0]], dtype=float)/L
        key=tuple(sorted(map(int, seg)))
        ac=None; st=None
        for it in edge_map.get(key, []):
            d=int(doms[it])
            if d == int(acoustic_dom): ac=cents[it]
            if d == int(solid_dom): st=cents[it]
        if ac is not None and st is not None:
            if np.dot(n, st-ac) < 0: n=-n
        return n

    # Gaussian points on [0,1]
    xg,wg=np.polynomial.legendre.leggauss(2); xi=0.5*(xg+1.0); wi=0.5*wg
    for seg, tag in zip(mesh.line_cells, mesh.line_tags):
        tag=int(tag); a=adj.get(tag)
        if a is None: continue
        dom_pair={int(a.up_domain), int(a.down_domain)}
        acoustic_sides=list(dom_pair & ACOUSTIC_DOMAINS)
        solid_sides=list(dom_pair & STRUCTURAL_DOMAINS)
        if not acoustic_sides or not solid_sides:
            continue
        g0=int(seg[0]); g1=int(seg[1])
        if g0 not in ac_model.acoustic_node_map or g1 not in ac_model.acoustic_node_map:
            continue
        p0=mesh.points_rz_m[g0]; p1=mesh.points_rz_m[g1]; pm=0.5*(p0+p1)
        L=float(np.linalg.norm(p1-p0))
        if L <= 0: continue
        # Find refined structural nodes at endpoints and midpoint.
        sd=[]
        for x in (p0, pm, p1):
            dist, idx=tree.query(x, k=1)
            max_dist=max(max_dist, float(dist))
            if dist > tol:
                missed.append({'boundary':tag,'dist_m':float(dist),'r':float(x[0]),'z':float(x[1])})
            sd.append(int(idx))
        if any(m['boundary']==tag and m['dist_m']>tol for m in missed[-3:]):
            # Do not abort; nearest projection is still better than dropping the ASB term.
            pass
        nvec=normal_from_acoustic_to_solid(seg, acoustic_sides[0], solid_sides[0])
        pcols=[ac_model.acoustic_node_map[g0], ac_model.acoustic_node_map[g1]]
        parent_intervals=[(0.0,0.5,sd[0],sd[1]), (0.5,1.0,sd[1],sd[2])]
        for t0,t1,s0,s1 in parent_intervals:
            for s,w in zip(xi,wi):
                t=t0+(t1-t0)*float(s)
                x=(1.0-t)*p0+t*p1
                weight=2.0*math.pi*max(float(x[0]),1e-9)*L*(t1-t0)*float(w)
                # structural local coordinate on this half segment
                tau=(t-t0)/(t1-t0)
                Ns=[1.0-tau, tau]
                Np=[1.0-t, t]
                for si,snode in enumerate((s0,s1)):
                    for pj,pcol in enumerate(pcols):
                        val=weight*Ns[si]*Np[pj]
                        Grows.append(2*snode);   Gcols.append(pcol); Gvals.append(float(nvec[0]*val))
                        Grows.append(2*snode+1); Gcols.append(pcol); Gvals.append(float(nvec[1]*val))
            n_sub += 1
        used_boundaries.append(tag); n_parent += 1
    G=coo_matrix((Gvals,(Grows,Gcols)), shape=(solid_refined.ndof, len(ac_model.acoustic_nodes_global))).tocsr()
    info={'n_parent_interface_segments':int(n_parent),'n_refined_subsegments':int(n_sub),'n_interface_boundaries':int(len(set(used_boundaries))), 'interface_boundaries':sorted(set(int(x) for x in used_boundaries)), 'max_solid_node_projection_distance_m':float(max_dist), 'n_projection_warnings':int(sum(1 for m in missed if m['dist_m']>tol)), 'projection_warning_first10':missed[:10], 'G_shape':[int(G.shape[0]),int(G.shape[1])], 'G_nnz':int(G.nnz), 'description':'mortar-like refined-solid / coarse-pressure ASB; structural side uses refined Stage-7D DOFs; pressure remains tractable P1 coarse mesh'}
    return G, info


def build_stage7f_model(mesh_path: str|Path, mphtxt_path: str|Path, suspension_E_scale: float, solid_uniform_refine: int):
    mesh=load_tagged_meshio(mesh_path)
    ac_model=build_stage4C_acoustic_structure_model(mesh, mphtxt_path, solid_uniform_refine=0)
    solid_ref=build_stage4_solid_model(mesh, materials=make_materials(suspension_E_scale), uniform_refine=solid_uniform_refine)
    G_ref, G_info=assemble_refined_solid_to_coarse_pressure_G(ac_model, solid_ref)
    # Replace solid and G in the model object so HK/reused helpers can still consume pressure fields.
    ac_model.solid=solid_ref
    ac_model.G_sp=G_ref
    return ac_model, G_info


def solve_stage7f_frequency(model, cpl, Zb: complex, freq_Hz: float, params: Stage4CParameters, *, nra_enabled: bool=True):
    f=float(freq_Hz); w=2.0*math.pi*f
    solid=model.solid; sf=solid.free_dofs; pf=model.pressure_free_dofs
    gf=np.asarray(cpl.g_free_N_per_A, dtype=complex)
    Hs=(_complex_stiffness(solid, w)[sf][:,sf].astype(complex) - (w*w)*solid.M[sf][:,sf].astype(complex)).tocsr()
    Ap=_acoustic_matrix(model, w, rho0=params.rho0_kg_m3, c0=params.c0_m_s, nra_enabled=nra_enabled)[pf][:,pf].astype(complex).tocsr()
    Gsf=model.G_sp[sf][:,pf].astype(complex).tocsr()
    GT=model.G_sp.T[pf][:,sf].astype(complex).tocsr()
    A=bmat([
        [csr_matrix([[complex(Zb)]]), csr_matrix((1j*w*gf.reshape(1,-1))), csr_matrix((1,len(pf)), dtype=complex)],
        [csr_matrix((-gf.reshape(-1,1))), Hs, -Gsf],
        [csr_matrix((len(pf),1), dtype=complex), -params.rho0_kg_m3*w*w*GT, Ap],
    ], format='csc')
    rhs=np.zeros(A.shape[0], dtype=complex); rhs[0]=params.V0_peak_V
    lu=splu(A)
    sol=lu.solve(rhs)
    I=sol[0]; us=sol[1:1+len(sf)]; pp=sol[1+len(sf):]
    ufull=np.zeros(solid.ndof, dtype=complex); ufull[sf]=us
    pfull=np.zeros(len(model.acoustic_nodes_global), dtype=complex); pfull[pf]=pp
    return {'f_Hz':f,'I_A_peak':I,'Z_total_ohm':params.V0_peak_V/I,'Zb_ohm':complex(Zb),'Z_motional_ohm':params.V0_peak_V/I-complex(Zb),'V_backemf_V_peak':1j*w*np.dot(gf,us), 'u_full_m':ufull,'p_full_Pa':pfull,'n_unknowns':int(A.shape[0]),'nnz':int(A.nnz)}


def solve_stage7f_sweep(model, cpl, freqs, Zb, params, outdir: Path, branch: str, nra_enabled=True):
    results=[]
    for f,z in zip(freqs,Zb):
        ck=outdir/f'checkpoint_{branch}_{float(f):09.3f}Hz.npz'
        if ck.exists():
            d=np.load(ck, allow_pickle=False)
            r={'f_Hz':float(d['f_Hz']),'I_A_peak':complex(d['I_A_peak']),'Z_total_ohm':complex(d['Z_total_ohm']),'Zb_ohm':complex(d['Zb_ohm']),'Z_motional_ohm':complex(d['Z_motional_ohm']),'V_backemf_V_peak':complex(d['V_backemf_V_peak']),'u_full_m':d['u_full_m'],'p_full_Pa':d['p_full_Pa'],'n_unknowns':int(d['n_unknowns']),'nnz':int(d['nnz'])}
        else:
            t=time.time(); r=solve_stage7f_frequency(model,cpl,z,float(f),params,nra_enabled=nra_enabled); r['solve_elapsed_s']=time.time()-t
            np.savez_compressed(ck, f_Hz=float(f), I_A_peak=r['I_A_peak'], Z_total_ohm=r['Z_total_ohm'], Zb_ohm=r['Zb_ohm'], Z_motional_ohm=r['Z_motional_ohm'], V_backemf_V_peak=r['V_backemf_V_peak'], u_full_m=r['u_full_m'], p_full_Pa=r['p_full_Pa'], n_unknowns=r['n_unknowns'], nnz=r['nnz'])
        results.append(r)
    f=np.asarray([r['f_Hz'] for r in results],float); I=np.asarray([r['I_A_peak'] for r in results],complex); Z=np.asarray([r['Z_total_ohm'] for r in results],complex); Zbarr=np.asarray([r['Zb_ohm'] for r in results],complex)
    out={'f_Hz':f,'I_A_peak':I,'Z_total_ohm':Z,'Zb_ohm':Zbarr,'Z_motional_ohm':Z-Zbarr,'V_backemf_V_peak':np.asarray([r['V_backemf_V_peak'] for r in results],complex),'solid_displacement_m':np.vstack([r['u_full_m'] for r in results]),'acoustic_pressure_field_Pa':np.vstack([r['p_full_Pa'] for r in results]),'coil_power_W':0.5*np.real(params.V0_peak_V*np.conj(I)),'n_unknowns':np.asarray([r['n_unknowns'] for r in results],int),'nnz':np.asarray([r['nnz'] for r in results],int)}
    return out


def add_hk_metrics(result, model, params):
    hk=hk_axis_and_power_recovered(result, model, params, nphi_axis=24, mirror=True)
    result=dict(result)
    result['p_1m_Pa_peak']=hk['p_1m_hk_recovered_Pa_peak']
    result['SPL_1m_dB']=hk['SPL_1m_hk_recovered_dB']
    result['phase_deg']=hk['phase_hk_recovered_deg']
    result['acoustic_power_W']=np.maximum(hk['hk_recovered_halfspace_power_W'],0.0)
    result['hk_flux_raw_W']=hk['hk_recovered_flux_raw_W']
    result['acoustic_efficiency_percent']=100.0*result['acoustic_power_W']/np.maximum(result['coil_power_W'],1e-300)
    result['hk_boundary_info']=hk['hk_recovered_boundary_info']
    return result


def solve_solid_only_sweep(solid, cpl, freqs, Zb, params):
    rows=[]; Zs=[]
    for f,z in zip(freqs,Zb):
        I,u,Z=solve_mmcpl_block_for_frequency(solid,cpl,z,float(f),V0=params.V0_peak_V)
        Zs.append(Z)
        rows.append({'f_Hz':float(f),'Z_solid_only_abs_ohm':float(abs(Z)),'Z_solid_only_real_ohm':float(Z.real),'Z_solid_only_imag_ohm':float(Z.imag),'I_solid_only_abs_A_peak':float(abs(I))})
    return np.asarray(Zs,complex), rows


def rows_response(res, Z_solid=None, branch='with_NRA'):
    rows=[]
    for i,f in enumerate(res['f_Hz']):
        Z=res['Z_total_ohm'][i]; Zb=res['Zb_ohm'][i]; Zm=res['Z_motional_ohm'][i]
        row={'branch':branch,'f_Hz':float(f),'SPL_1m_hk_recovered_dB':float(res.get('SPL_1m_dB',np.full(len(res['f_Hz']),np.nan))[i]),'phase_hk_recovered_deg':float(res.get('phase_deg',np.full(len(res['f_Hz']),np.nan))[i]),'Z_abs_ohm':float(abs(Z)),'Z_real_ohm':float(Z.real),'Z_imag_ohm':float(Z.imag),'Zb_abs_ohm':float(abs(Zb)),'Zb_real_ohm':float(Zb.real),'Zb_imag_ohm':float(Zb.imag),'Z_motional_abs_ohm':float(abs(Zm)),'Z_motional_real_ohm':float(Zm.real),'Z_motional_imag_ohm':float(Zm.imag),'I_abs_A_peak':float(abs(res['I_A_peak'][i])),'Vbe_abs_V_peak':float(abs(res['V_backemf_V_peak'][i])),'coil_power_W':float(res['coil_power_W'][i]),'acoustic_power_W':float(res.get('acoustic_power_W',np.full(len(res['f_Hz']),np.nan))[i]),'acoustic_efficiency_percent':float(res.get('acoustic_efficiency_percent',np.full(len(res['f_Hz']),np.nan))[i]),'n_unknowns':int(res['n_unknowns'][i]),'matrix_nnz':int(res['nnz'][i])}
        if Z_solid is not None:
            row['Z_solid_only_abs_ohm']=float(abs(Z_solid[i])); row['Z_solid_only_real_ohm']=float(Z_solid[i].real); row['Z_solid_only_imag_ohm']=float(Z_solid[i].imag); row['ASB_minus_solid_absZ_ohm']=float(abs(Z)-abs(Z_solid[i]))
        rows.append(row)
    return rows


def compare_figure10(res, fig10):
    rows=[]
    for _,rr in fig10.iterrows():
        f=float(rr['f_Hz']); idx=int(np.argmin(np.abs(res['f_Hz']-f))); Z=res['Z_total_ohm'][idx]; targ=float(rr['target_Z_abs_ohm']); err=abs(Z)-targ
        rows.append({'figure':'Figure 10','f_Hz':f,'target_absZ_ohm':targ,'stage7F_absZ_ohm':float(abs(Z)),'stage7F_realZ_ohm':float(Z.real),'stage7F_imagZ_ohm':float(Z.imag),'absZ_error_ohm':float(err),'absZ_error_percent':float(100*err/targ),'target_realZ_ohm':float(rr['target_Z_real_ohm']),'target_imagZ_ohm':float(rr['target_Z_imag_ohm'])})
    errs=np.asarray([r['absZ_error_ohm'] for r in rows]); per=np.asarray([r['absZ_error_percent'] for r in rows])
    return rows, {'absZ_RMSE_ohm':float(np.sqrt(np.mean(errs*errs))),'absZ_max_abs_error_percent':float(np.max(np.abs(per))), 'Z50_abs_ohm':float(rows[[r['f_Hz'] for r in rows].index(50.0)]['stage7F_absZ_ohm']) if 50.0 in [r['f_Hz'] for r in rows] else None}


def compare_figure8(res, fig8):
    rows=[]
    for _,rr in fig8.iterrows():
        f=float(rr['f_Hz']); idx=int(np.argmin(np.abs(res['f_Hz']-f))); spl=float(res['SPL_1m_dB'][idx]); targ=float(rr['target_SPL_dB']); err=spl-targ
        rows.append({'figure':'Figure 8','f_Hz':f,'target_SPL_dB':targ,'stage7F_SPL_dB':spl,'SPL_error_dB':err})
    errs=np.asarray([r['SPL_error_dB'] for r in rows])
    return rows, {'SPL_RMSE_dB':float(np.sqrt(np.mean(errs*errs))),'SPL_max_abs_error_dB':float(np.max(np.abs(errs))),'SPL_mean_error_dB':float(np.mean(errs))}


def plot_figure10(path, res, Z_solid, fig10):
    f=res['f_Hz']; Z=res['Z_total_ohm']
    fig,ax=plt.subplots(figsize=(8,5))
    ax.semilogx(f,np.abs(Z),'-o',ms=3,label='Stage 7F refined-solid ASB |Z|')
    ax.semilogx(f,np.abs(Z_solid),'--x',ms=3,label='same refined solid-only |Z|')
    ax.semilogx(f,Z.real,':',label='Stage 7F Re(Z)')
    ax.semilogx(f,Z.imag,'-.',label='Stage 7F Im(Z)')
    ax.semilogx(fig10['f_Hz'],fig10['target_Z_abs_ohm'],'s',label='PDF Figure 10 anchors')
    ax.axhline(5.6,lw=0.8,ls=':',label='DC 5.6 Ω')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Impedance (ohm)'); ax.set_title('Stage 7F refined-solid ASB impedance: preserve native motional peak')
    ax.grid(True,which='both',alpha=0.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def plot_figure8(path,res,fig8):
    fig,ax=plt.subplots(figsize=(8,5))
    ax.semilogx(res['f_Hz'],res['SPL_1m_dB'],'-o',ms=3,label='Stage 7F refined-solid ASB + recovered HK')
    ax.semilogx(fig8['f_Hz'],fig8['target_SPL_dB'],'s',label='PDF Figure 8 anchors')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('SPL at 1 m (dB)'); ax.set_ylim(40,110); ax.set_title('Stage 7F Figure 8 diagnostic')
    ax.grid(True,which='both',alpha=0.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description='Stage 7F refined-structure ASB preserving Stage-7D motional impedance with same refined structural DOFs.')
    ap.add_argument('--outdir', default=str(ROOT/'outputs/stage7F_shared_refined_asb'))
    ap.add_argument('--mesh', default=str(ROOT/'meshes/comsol_geometry_polyline_coarse_2p5mm.msh'))
    ap.add_argument('--mphtxt', default=str(ROOT/'comsol_reference_inputs/Untitled.mphtxt'))
    ap.add_argument('--magnetostatic-vtu', default=str(ROOT/'outputs/stage5B_raw_magnetics_closure/refined_B_inverse_iter35/magnetostatic_solution.vtu'))
    ap.add_argument('--blocked-csv', default=str(ROOT/'outputs/stage3C_corrected_baseline/blocked_impedance_exact_voltage.csv'))
    ap.add_argument('--figure8-csv', default=str(ROOT/'outputs/stage5ABC_figure_magnetics_coil_closure/stage5A_reference_dashboard/figure8_digitized.csv'))
    ap.add_argument('--figure10-csv', default=str(ROOT/'outputs/stage5ABC_figure_magnetics_coil_closure/stage5A_reference_dashboard/figure10_digitized.csv'))
    ap.add_argument('--freqs', default='figure8_10')
    ap.add_argument('--suspension-E-scale', type=float, default=0.65)
    ap.add_argument('--solid-uniform-refine', type=int, default=1)
    ap.add_argument('--V0-peak', type=float, default=3.55)
    ap.add_argument('--no-plots', action='store_true')
    args=ap.parse_args()
    t0=time.time(); out=Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    fig8=pd.read_csv(args.figure8_csv); fig10=pd.read_csv(args.figure10_csv)
    freqs=parse_freqs(args.freqs, fig8, fig10)
    Zb=interp_complex_log(pd.read_csv(args.blocked_csv), freqs)
    params=Stage4CParameters(V0_peak_V=args.V0_peak)
    model, G_info=build_stage7f_model(args.mesh,args.mphtxt,args.suspension_E_scale,args.solid_uniform_refine)
    cpl=assemble_lorentz_backemf_vector(model.solid,args.magnetostatic_vtu)
    Z_solid, solid_rows=solve_solid_only_sweep(model.solid,cpl,freqs,Zb,params)
    write_rows(out/'stage7F_same_refined_solid_only_reference.csv', solid_rows)
    res=solve_stage7f_sweep(model,cpl,freqs,Zb,params,out,'with_NRA',nra_enabled=True)
    res=add_hk_metrics(res,model,params)
    write_rows(out/'stage7F_shared_refined_asb_response.csv', rows_response(res,Z_solid,'with_NRA'))
    f10_rows,f10_metrics=compare_figure10(res,fig10); write_rows(out/'stage7F_figure10_dashboard.csv',f10_rows)
    f8_rows,f8_metrics=compare_figure8(res,fig8); write_rows(out/'stage7F_figure8_dashboard.csv',f8_rows)
    # peak retention at Figure-10 50 Hz against same refined solid-only branch
    i50=int(np.argmin(np.abs(freqs-50.0)))
    peak_retention=float(abs(res['Z_total_ohm'][i50])/max(abs(Z_solid[i50]),1e-300))
    if not args.no_plots:
        plot_figure10(out/'figure10_stage7F_refined_solid_asb_impedance.png',res,Z_solid,fig10)
        plot_figure8(out/'figure8_stage7F_refined_solid_asb_sensitivity.png',res,fig8)
    summary={'stage':'Stage 7F shared-refined ASB / refined structural DOF preservation','status':'completed','meaning':'Uses the same uniformly refined structural DOFs for Lorentz/back-EMF and ASB pressure loading. Pressure mesh remains coarse P1 for tractable block solves; ASB is a nonmatching refined-solid/coarse-pressure mortar projection. No gamma and no SPL transfer correction.', 'inputs':{'mesh':str(args.mesh),'solid_uniform_refine':int(args.solid_uniform_refine),'suspension_E_scale':float(args.suspension_E_scale),'frequencies_Hz':[float(x) for x in freqs]}, 'model_summary':model.summary(), 'G_refined_ASB_info':G_info, 'coupling_summary':cpl.summary(), 'figure10_metrics':f10_metrics, 'figure8_metrics':f8_metrics, 'peak_retention_at_50Hz_absZ_ASB_over_solid_only':peak_retention, 'acceptance':{'gamma_used':False,'figure8_transfer_correction_used':False,'same_refined_structural_DOFs_for_g_and_ASB':True,'pressure_mesh':'coarse P1 retained for tractable solve','figure10_peak_preserved': bool(peak_retention > 0.8),'figure10_conditional_pass': bool(f10_metrics['absZ_RMSE_ohm'] <= 3.0 and f10_metrics['absZ_max_abs_error_percent'] <= 20.0)}, 'elapsed_s':time.time()-t0}
    write_json(out/'stage7F_summary.json',summary)
    report=['# Stage 7F：shared-refined ASB / refined structural DOF preservation','', '## 目标','', 'Stage 7E 的原生 `[I,u,p]` block 使用未细化结构节点，导致 50 Hz motional impedance 峰相对 Stage 7D refined solid-only 分支退化。本阶段把 Lorentz/back-EMF 的 `g` 向量和 ASB 压力载荷都投影到同一套 refined structural DOFs。', '', '## 方法','', '```text','Electrical:  Zb I + iω gᵀu = V0','Solid:      -g I + Hs_ref u - G_ref p = 0','Acoustic:        -ρ0ω² G_refᵀu + Hp p = 0','```','', '`G_ref` 不是 Stage 7E 的粗结构节点矩阵，而是在每条 acoustic-solid 边界上把结构侧分成 `[端点-中点]` 和 `[中点-端点]` 两个 refined P1 子边积分；压力侧保留父边 P1 形函数。这样 ASB 的结构载荷和 `g` 使用同一套 refined structure DOFs。', '', '## 计算设置','', f'- mesh: `{Path(args.mesh).name}`', f'- solid_uniform_refine: {args.solid_uniform_refine}', f'- suspension E scale: {args.suspension_E_scale}', f'- refined structural free DOFs: {model.solid.summary()["ndof_free"]}', f'- pressure free DOFs: {model.summary()["n_pressure_free_dofs"]}', f'- coupled unknowns: {int(res["n_unknowns"][0])}', f'- axial BL from g: {cpl.axial_BL_N_per_A:.9f} N/A', f'- ASB parent interface segments: {G_info["n_parent_interface_segments"]}', f'- ASB refined subsegments: {G_info["n_refined_subsegments"]}', '', '## Figure 10 / motional peak','', f'- same refined solid-only |Z|(50 Hz): {abs(Z_solid[i50]):.3f} Ω', f'- Stage 7F ASB |Z|(50 Hz): {abs(res["Z_total_ohm"][i50]):.3f} Ω', f'- ASB/solid-only peak retention: {peak_retention:.3f}', f'- Figure 10 abs(Z) RMSE: {f10_metrics["absZ_RMSE_ohm"]:.3f} Ω', f'- Figure 10 max anchor error: {f10_metrics["absZ_max_abs_error_percent"]:.3f} %', '', '## Figure 8 diagnostic','', f'- SPL RMSE: {f8_metrics["SPL_RMSE_dB"]:.3f} dB', f'- SPL max abs error: {f8_metrics["SPL_max_abs_error_dB"]:.3f} dB', '', '## 判断','', '- 本阶段完成了 Stage 7F 的核心结构：`g` 与 ASB 共享 refined structural DOFs，不再把 ASB 施加到粗结构节点。', '- 压力侧为计算成本仍保留 coarse P1；因此它是 shared-refined structural ASB，而不是 full shared-refined acoustic P2。', '- 若 50 Hz peak retention 接近 1，说明 Stage 7D 的 refined motional impedance 已经能在 full `[I,u,p]` block 中保持；若仍明显小于 1，下一步应检查 ASB 声压载荷符号、声学质量项符号、pressure Dirichlet/PML 设置。']
    (out/'STAGE7F_SHARED_REFINED_ASB_REPORT_CN.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__':
    main()
