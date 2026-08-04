#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, sys, pickle
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from loudspeaker_axisym_fem.axisym_magnetics import load_tagged_meshio
from loudspeaker_axisym_fem.stage4_electroacoustic import load_blocked_impedance_csv
from loudspeaker_axisym_fem.stage4C_acoustic_structure import Stage4CParameters, build_stage4C_acoustic_structure_model, solve_stage4C_full_asb
from loudspeaker_axisym_fem.stage4D_exterior_nra import hk_directivity_from_result
from loudspeaker_axisym_fem.stage4F_hk_refinement import hk_directivity_recovered, hk_axis_and_power_recovered
from loudspeaker_axisym_fem.json_utils import write_json, dumps_json

def write_rows(path, rows):
    if not rows: return
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mesh', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--freq', type=float, required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--mphtxt', default=str(ROOT/'comsol_reference_inputs/Untitled.mphtxt'))
    ap.add_argument('--blocked-impedance-csv', default=str(ROOT/'outputs/stage3C_corrected_baseline/blocked_impedance_exact_voltage.csv'))
    args=ap.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    params=Stage4CParameters()
    freqs,Zb=load_blocked_impedance_csv(args.blocked_impedance_csv, np.array([args.freq],float))
    mesh=load_tagged_meshio(args.mesh)
    model=build_stage4C_acoustic_structure_model(mesh,args.mphtxt,solid_uniform_refine=0)
    res=solve_stage4C_full_asb(freqs,Zb,model,params,nra_enabled=True)
    rec_axis=hk_axis_and_power_recovered(res,model,params)
    res.update(rec_axis)
    frec,ang,spl_rec,rel_rec=hk_directivity_recovered(res,model,params)
    ffac,ang2,spl_fac,rel_fac=hk_directivity_from_result(res,model,params)
    rows=[]
    for method,spl,rel in [('recovered',spl_rec,rel_rec),('facet',spl_fac,rel_fac)]:
        for j,a in enumerate(ang):
            rows.append({'mesh_label':args.label,'f_Hz':float(args.freq),'method':method,'angle_deg':float(a),'SPL_dB':float(spl[0,j]),'relative_dB':float(rel[0,j])})
    write_rows(out/f'directivity_{args.label}_{int(args.freq)}.csv',rows)
    summ={
        'mesh_label':args.label,
        'f_Hz':float(args.freq),
        'model':model.summary(),
        'SPL_axis_recovered_dB':float(res['SPL_1m_hk_recovered_dB'][0]),
        'Z_abs_ohm':float(abs(res['Z_total_ohm'][0])),
        'recovered_boundary_info':res['hk_recovered_boundary_info'][0],
    }
    write_json(out/f'directivity_{args.label}_{int(args.freq)}.json',summ,indent=2)
    print(dumps_json(summ,indent=2))
if __name__=='__main__': main()
