#!/usr/bin/env python3
"""Causal one-boundary-at-a-time COMSOL motion substitutions using one LU/frequency."""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from scipy.sparse.linalg import splu

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"best_model"),str(ROOT/"src"),str(ROOT/"tools")]
from coupled_solver import build_best_model, _exterior_field
from stage35_layer_substitution import _replace_motion, _reference_directivity, _metrics

def main():
    a=argparse.ArgumentParser(); a.add_argument("--config",default="configs/stage34_structure_refined2.json")
    a.add_argument("--motion",type=Path,required=True); a.add_argument("--directivity",type=Path,required=True)
    a.add_argument("--checkpoints",type=Path,required=True); a.add_argument("--frequencies",default="8000,10000,12000,13500,15000")
    a.add_argument("--outdir",type=Path,required=True); x=a.parse_args(); t0=time.perf_counter()
    model=build_best_model(ROOT,config_path=x.config,magnetostatic_vtu=ROOT/"inputs/comsol_reference/magnetostatic_converged_55iter.vtu",build_blocked_coil=False)
    motion=pd.read_csv(x.motion); interface=set(map(int,model.G_info["interface_boundaries"])); rows=[]
    groups={"all_ASB":interface,"cone_46_47":{46,47},"dustcap_91_92":{91,92},"surround_99_102":{99,100,101,102}}
    for freq in map(float,x.frequencies.split(',')):
        z=np.load(x.checkpoints/f"{freq:g}Hz"/f"solution_{freq:g}Hz.npz"); native_u=z["solid_displacement"]
        angles,ref=_reference_directivity(x.directivity,freq); base=_metrics(z["directivity_pressure_Pa_peak"],ref,angles)
        rows.append({"freq_Hz":freq,"substitution":"native_baseline","kind":"baseline",**base})
        A,_=model.acoustic_operator.matrix(freq,nra_enabled=True); lu=splu(A.tocsc()); G=model.G
        if G.shape[1]<model.acoustic_operator.n2: G=hstack([G,csr_matrix((G.shape[0],model.acoustic_operator.n2-G.shape[1]))],format="csr")
        ff=motion[np.isclose(motion.freq_Hz.astype(float),freq)]
        cases={**groups,**{f"boundary_{b}":{b} for b in sorted(interface)}}
        for name,tags in cases.items():
            u,info=_replace_motion(model,native_u,ff,tags); rhs=model.config["air"]["rho0_kg_m3"]*(2*math.pi*freq)**2*(G.T@u); p=lu.solve(rhs)
            aa,pdir,_,_,_=_exterior_field(model,freq,p); pred=np.interp(angles,aa,pdir.real)+1j*np.interp(angles,aa,pdir.imag); m=_metrics(pred,ref,angles)
            row={"freq_Hz":freq,"substitution":name,"kind":"group" if name in groups else "single_boundary","replaced_nodes":info["replaced_nodes"],**m}
            for metric in ("main_ge_minus20dB_RMSE_dB","field_energy_weighted_relative_RMSE_dB","shape_NRMSE_pct","raw_complex_NRMSE_pct"):
                row[metric+"_reduction_vs_native_pct"]=100*(1-m[metric]/max(base[metric],1e-300))
            rows.append(row)
        print(f"completed {freq:g} Hz",flush=True)
    table=pd.DataFrame(rows); x.outdir.mkdir(parents=True,exist_ok=True); table.to_csv(x.outdir/"causal_boundary_substitution.csv",index=False)
    singles=table[table.kind=="single_boundary"]
    rank=singles.groupby("substitution").agg(frequency_count=("freq_Hz","size"),main_RMSE_reduction_pct_mean=("main_ge_minus20dB_RMSE_dB_reduction_vs_native_pct","mean"),field_weighted_reduction_pct_mean=("field_energy_weighted_relative_RMSE_dB_reduction_vs_native_pct","mean"),shape_reduction_pct_mean=("shape_NRMSE_pct_reduction_vs_native_pct","mean"),raw_reduction_pct_mean=("raw_complex_NRMSE_pct_reduction_vs_native_pct","mean")).reset_index()
    rank["robust_causal_score_pct"]=rank[["main_RMSE_reduction_pct_mean","field_weighted_reduction_pct_mean","shape_reduction_pct_mean","raw_reduction_pct_mean"]].median(axis=1)
    rank=rank.sort_values("robust_causal_score_pct",ascending=False); rank.to_csv(x.outdir/"causal_boundary_ranking.csv",index=False)
    summary={"elapsed_seconds":time.perf_counter()-t0,"criterion":"median reduction across main-field, field-energy-weighted, shape and raw complex errors; positive across metrics is required","top10":rank.head(10).to_dict(orient="records")}
    (x.outdir/"causal_boundary_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8"); print(rank.head(15).to_string(index=False))
if __name__=="__main__":main()
