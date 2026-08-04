#!/usr/bin/env python3
"""Compare COMSOL/native complex fields by structural and acoustic domain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "best_model"), str(ROOT / "src")]
from coupled_solver import build_best_model


def cmplx(d, a, b): return d[a].to_numpy(float) + 1j * d[b].to_numpy(float)


def metrics(pred, ref, distance_note=None):
    mask = np.isfinite(pred.real) & np.isfinite(pred.imag) & np.isfinite(ref.real) & np.isfinite(ref.imag)
    pred, ref = pred[mask], ref[mask]
    if not len(pred): return {"n": 0}
    scale = np.vdot(pred, ref) / max(np.vdot(pred, pred).real, 1e-300)
    corr = abs(np.vdot(pred, ref)) / max(np.linalg.norm(pred) * np.linalg.norm(ref), 1e-300)
    out = {
        "n": int(len(pred)), "coverage_pct": float(100 * np.mean(mask)),
        "raw_complex_NRMSE_pct": float(100 * np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1e-300)),
        "shape_NRMSE_pct": float(100 * np.linalg.norm(scale * pred - ref) / max(np.linalg.norm(ref), 1e-300)),
        "complex_correlation": float(corr), "diagnostic_scale_abs": float(abs(scale)),
        "diagnostic_scale_phase_deg": float(np.angle(scale, deg=True)),
    }
    return out


def subdivide6(c):
    a,b,d,ab,bd,da = map(int,c)
    return [[a,ab,da],[ab,b,bd],[da,bd,d],[ab,bd,da]]


def interp(points, triangles, values, xy):
    tri=mtri.Triangulation(points[:,0],points[:,1],np.asarray(triangles,int))
    rr=mtri.LinearTriInterpolator(tri,np.asarray(values).real)(xy[:,0],xy[:,1])
    ii=mtri.LinearTriInterpolator(tri,np.asarray(values).imag)(xy[:,0],xy[:,1])
    return np.asarray(np.ma.filled(rr,np.nan))+1j*np.asarray(np.ma.filled(ii,np.nan))


def interp_scattered(points, triangles, values, xy):
    used=np.unique(np.asarray(triangles,int).ravel()); p=points[used]; v=np.asarray(values)[used]
    rr=np.asarray(LinearNDInterpolator(p,v.real,fill_value=np.nan)(xy))
    ii=np.asarray(LinearNDInterpolator(p,v.imag,fill_value=np.nan)(xy))
    missing=~np.isfinite(rr)|~np.isfinite(ii)
    if np.any(missing):
        rr[missing]=NearestNDInterpolator(p,v.real)(xy[missing]); ii[missing]=NearestNDInterpolator(p,v.imag)(xy[missing])
    return rr+1j*ii


def solid_rows(model, native_u, csv, freq):
    d=pd.read_csv(csv); d=d[np.isclose(d.freq_Hz.astype(float),freq)].copy()
    rows=[]
    for dom,g in d.groupby("domain_id"):
        con=model.solid.triangles6[model.solid.domains==int(dom)]
        tris=[t for c in con for t in subdivide6(c)]
        xy=g[["r/1[m]_real","z/1[m]_real"]].to_numpy(float)
        vector_pred=[]; vector_ref=[]
        for component,pred_values,recol,imcol in (
            ("u_r",native_u[0::2],"u/1[m]_real","u/1[m]_imag"),
            ("u_z",native_u[1::2],"w/1[m]_real","w/1[m]_imag")):
            pred=interp(model.solid.points_rz_m,tris,pred_values,xy); ref=cmplx(g,recol,imcol)
            vector_pred.append(pred); vector_ref.append(ref)
            rows.append({"layer":"solid","domain_id":int(dom),"component":component,
                         "reference_RMS":float(np.sqrt(np.mean(np.abs(ref)**2))),**metrics(pred,ref)})
        vp=np.concatenate(vector_pred); vr=np.concatenate(vector_ref)
        rows.append({"layer":"solid","domain_id":int(dom),"component":"u_vector",
                     "reference_RMS":float(np.sqrt(np.mean(np.abs(vr)**2))),**metrics(vp,vr)})
    return rows


def acoustic_domain_triangles(model, dom):
    op=model.acoustic_operator; amap=model.acoustic_model.acoustic_node_map
    enriched={int(r["triangle_id"]):r for r in getattr(op,"pml_triangles",[])}
    out=[]
    for it,(tri,d) in enumerate(zip(model.mesh.triangles,model.mesh.tri_domains.astype(int))):
        if int(d)!=int(dom): continue
        if it in enriched:
            rec=enriched[it]; conn=list(map(int,rec["base_dofs"]))
            for e in rec["edges"]:
                if e in op.edge_dof: conn.append(int(op.edge_dof[e]))
                else: conn.append(-1)
            # Interface-constrained midpoint values are not independent.  For
            # domain comparisons, fall back to the P1 triangle if any are absent.
            if min(conn)>=0: out.extend(subdivide6(conn))
            else: out.append([amap[int(x)] for x in tri])
        else: out.append([amap[int(x)] for x in tri])
    return out


def acoustic_rows(model, pressures, labels, csv, freq):
    d=pd.read_csv(csv); d=d[np.isclose(d.freq_Hz.astype(float),freq)].copy(); rows=[]
    points=model.acoustic_operator.mixed_points_and_cells()[0]
    for dom,g in d.groupby("domain_id"):
        tris=acoustic_domain_triangles(model,int(dom)); xy=g[["r/1[m]_real","z/1[m]_real"]].to_numpy(float)
        ref=cmplx(g,"acpr.p_t_real","acpr.p_t_imag")
        for pressure,label in zip(pressures,labels):
            vals=model.acoustic_operator.pressure_for_mixed_points(pressure)
            pred=interp_scattered(points,tris,vals,xy)
            rows.append({"layer":"acoustic","domain_id":int(dom),"component":label,**metrics(pred,ref)})
    return rows


def main():
    a=argparse.ArgumentParser(); a.add_argument("--config",default="configs/stage34_structure_refined2.json")
    a.add_argument("--frequency",type=float,required=True); a.add_argument("--native-npz",type=Path,required=True)
    a.add_argument("--diagnostic-npz",type=Path); a.add_argument("--comsol-solid",type=Path,required=True)
    a.add_argument("--comsol-acoustic",type=Path,required=True); a.add_argument("--outdir",type=Path,required=True); x=a.parse_args()
    model=build_best_model(ROOT,config_path=x.config,magnetostatic_vtu=ROOT/"inputs/comsol_reference/magnetostatic_converged_55iter.vtu",build_blocked_coil=False)
    z=np.load(x.native_npz); rows=solid_rows(model,z["solid_displacement"],x.comsol_solid,x.frequency)
    pressures=[z["pressure_mixed"]]; labels=["native_full_chain"]
    if x.diagnostic_npz:
        q=np.load(x.diagnostic_npz); pressures.append(q["COMSOL_all_ASB_motion_native_acoustic_pressure_mixed"]); labels.append("COMSOL_motion_native_acoustic")
    rows+=acoustic_rows(model,pressures,labels,x.comsol_acoustic,x.frequency)
    table=pd.DataFrame(rows); x.outdir.mkdir(parents=True,exist_ok=True); table.to_csv(x.outdir/"domain_field_metrics.csv",index=False)
    summary={"frequency_Hz":x.frequency,"benchmark_policy":"diagnostic only","worst_shape_by_layer":{}}
    for layer,g in table.groupby("layer"):
        w=g.sort_values("shape_NRMSE_pct",ascending=False).iloc[0]
        summary["worst_shape_by_layer"][layer]={"domain_id":int(w.domain_id),"component":w.component,"shape_NRMSE_pct":float(w.shape_NRMSE_pct),"raw_complex_NRMSE_pct":float(w.raw_complex_NRMSE_pct)}
    (x.outdir/"domain_field_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print(table.sort_values(["layer","shape_NRMSE_pct"],ascending=[True,False]).to_string(index=False))

if __name__=="__main__": main()
