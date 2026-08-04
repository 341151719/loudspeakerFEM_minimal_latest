#!/usr/bin/env python3
"""Rank every ASB boundary by error and acoustic-motion importance."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"best_model"),str(ROOT/"src")]
from coupled_solver import build_best_model
from p2_axisym_solid import P2BoundarySampler

def cplx(d,a,b): return d[a].to_numpy(float)+1j*d[b].to_numpy(float)

def integral(frame,value):
    x=frame[["r_m","z_m"]].to_numpy(float); v=np.asarray(value)
    ds=np.linalg.norm(np.diff(x,axis=0),axis=1)
    q=2*np.pi*np.maximum(x[:,0],1e-12)*np.abs(v)**2
    return float(np.sum(.5*(q[:-1]+q[1:])*ds)) if len(ds) else 0.0

def metrics(pred,ref):
    scale=np.vdot(pred,ref)/max(np.vdot(pred,pred).real,1e-300)
    corr=abs(np.vdot(pred,ref))/max(np.linalg.norm(pred)*np.linalg.norm(ref),1e-300)
    return {"raw_complex_NRMSE_pct":100*float(np.linalg.norm(pred-ref)/max(np.linalg.norm(ref),1e-300)),
            "shape_NRMSE_pct":100*float(np.linalg.norm(scale*pred-ref)/max(np.linalg.norm(ref),1e-300)),
            "complex_correlation":float(corr),"scale_abs":float(abs(scale)),"scale_phase_deg":float(np.angle(scale,deg=True))}

def main():
    a=argparse.ArgumentParser(); a.add_argument("--config",default="configs/stage34_structure_refined2.json")
    a.add_argument("--comsol-motion",type=Path,required=True); a.add_argument("--checkpoints",type=Path,required=True)
    a.add_argument("--frequencies",default="8000,10000,12000,13500,15000"); a.add_argument("--outdir",type=Path,required=True); x=a.parse_args()
    model=build_best_model(ROOT,config_path=x.config,magnetostatic_vtu=ROOT/"inputs/comsol_reference/magnetostatic_converged_55iter.vtu",build_blocked_coil=False)
    d=pd.read_csv(x.comsol_motion); rows=[]
    for freq in map(float,x.frequencies.split(',')):
        u=np.load(x.checkpoints/f"{freq:g}Hz"/f"solution_{freq:g}Hz.npz")["solid_displacement"]
        ff=d[np.isclose(d.freq_Hz.astype(float),freq)]
        temp=[]
        for b,g in ff.groupby("boundary_id",sort=True):
            g=g.sort_values("node_id").reset_index(drop=True).rename(columns={"nr":"normal_r","nz":"normal_z"})
            sampler=P2BoundarySampler(model.solid,g); _,_,pred=sampler.sample(u)
            ref=cplx(g,"un_SI_real","un_SI_imag")
            ne,ce=integral(g,pred),integral(g,ref)
            temp.append({"freq_Hz":freq,"boundary_id":int(b),**metrics(pred,ref),"native_motion_energy":ne,"COMSOL_motion_energy":ce,"energy_ratio":ne/max(ce,1e-300),"max_projection_distance_m":float(np.max(sampler.dist))})
        total=sum(q["COMSOL_motion_energy"] for q in temp)
        for q in temp:
            q["COMSOL_energy_share_pct"]=100*q["COMSOL_motion_energy"]/max(total,1e-300)
            q["impact_score_pct2"]=q["COMSOL_energy_share_pct"]*q["shape_NRMSE_pct"]/100
        rows+=temp
    table=pd.DataFrame(rows); x.outdir.mkdir(parents=True,exist_ok=True); table.to_csv(x.outdir/"all_asb_boundary_metrics.csv",index=False)
    agg=table.groupby("boundary_id").agg(frequency_count=("freq_Hz","size"),energy_share_pct_mean=("COMSOL_energy_share_pct","mean"),shape_NRMSE_pct_mean=("shape_NRMSE_pct","mean"),shape_NRMSE_pct_max=("shape_NRMSE_pct","max"),raw_NRMSE_pct_mean=("raw_complex_NRMSE_pct","mean"),impact_score_mean=("impact_score_pct2","mean"),energy_ratio_geomean=("energy_ratio",lambda v:float(np.exp(np.mean(np.log(np.maximum(v,1e-300))))))).reset_index().sort_values("impact_score_mean",ascending=False)
    agg.to_csv(x.outdir/"all_asb_boundary_ranking.csv",index=False)
    summary={"criterion":"impact = COMSOL normal-motion energy share times complex-shape error; ranking avoids tiny inactive boundaries dominating", "top10":agg.head(10).to_dict(orient="records")}
    (x.outdir/"all_asb_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print(agg.head(15).to_string(index=False))
if __name__=="__main__":main()
