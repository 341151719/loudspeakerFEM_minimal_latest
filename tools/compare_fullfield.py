#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
import meshio


def metrics(pred,ref):
    s=np.vdot(pred,ref)/max(np.vdot(pred,pred).real,1e-300);a=pred*s
    return {'complex_corr':float(abs(np.vdot(pred,ref))/max(np.linalg.norm(pred)*np.linalg.norm(ref),1e-300)),'normalized_residual_after_scale':float(np.linalg.norm(a-ref)/max(np.linalg.norm(ref),1e-300)),'best_scale_abs':float(abs(s)),'best_scale_phase_deg':float(np.angle(s,deg=True))}

def compare_solid(vtu,comsol_csv,f,out):
    m=meshio.read(vtu);pts=m.points[:,:2];tree=cKDTree(pts);d=pd.read_csv(comsol_csv);d=d[np.isclose(d.freq_Hz.astype(float),f)].copy();xy=d[['r/1[m]_real','z/1[m]_real']].to_numpy(float);dist,idx=tree.query(xy)
    ur=m.point_data['u_r_m_real'][idx]+1j*m.point_data['u_r_m_imag'][idx];uz=m.point_data['u_z_m_real'][idx]+1j*m.point_data['u_z_m_imag'][idx]
    rr=d.u_real.to_numpy()+1j*d.u_imag.to_numpy();rz=d.w_real.to_numpy()+1j*d.w_imag.to_numpy();tab=d[['domain_id','node_id','r/1[m]_real','z/1[m]_real']].copy();tab['distance_m']=dist;tab['our_ur_real']=ur.real;tab['our_ur_imag']=ur.imag;tab['COMSOL_ur_real']=rr.real;tab['COMSOL_ur_imag']=rr.imag;tab['our_uz_real']=uz.real;tab['our_uz_imag']=uz.imag;tab['COMSOL_uz_real']=rz.real;tab['COMSOL_uz_imag']=rz.imag;tab.to_csv(out/'solid_fullfield_pointwise.csv',index=False);return {'solid_ur':metrics(ur,rr),'solid_uz':metrics(uz,rz),'max_nearest_distance_m':float(dist.max())}

def compare_acoustic(vtu,comsol_csv,f,out):
    m=meshio.read(vtu);pts=m.points[:,:2];tree=cKDTree(pts);d=pd.read_csv(comsol_csv);d=d[np.isclose(d.freq_Hz.astype(float),f)].copy();xy=d[['r/1[m]_real','z/1[m]_real']].to_numpy(float);dist,idx=tree.query(xy);p=m.point_data['p_Pa_peak_real'][idx]+1j*m.point_data['p_Pa_peak_imag'][idx];ref=d['acpr.p_t_real'].to_numpy()+1j*d['acpr.p_t_imag'].to_numpy();tab=d[['domain_id','node_id','r/1[m]_real','z/1[m]_real']].copy();tab['distance_m']=dist;tab['our_p_real']=p.real;tab['our_p_imag']=p.imag;tab['COMSOL_p_real']=ref.real;tab['COMSOL_p_imag']=ref.imag;tab.to_csv(out/'acoustic_fullfield_pointwise.csv',index=False);return {'acoustic_p':metrics(p,ref),'max_nearest_distance_m':float(dist.max())}

def main():
    a=argparse.ArgumentParser();a.add_argument('--freq',type=float,required=True);a.add_argument('--solid-vtu');a.add_argument('--acoustic-vtu');a.add_argument('--comsol-solid-csv');a.add_argument('--comsol-acoustic-csv');a.add_argument('--outdir',type=Path,required=True);x=a.parse_args();x.outdir.mkdir(parents=True,exist_ok=True);s={'freq_Hz':x.freq}
    if x.solid_vtu and x.comsol_solid_csv:s.update(compare_solid(x.solid_vtu,x.comsol_solid_csv,x.freq,x.outdir))
    if x.acoustic_vtu and x.comsol_acoustic_csv:s.update(compare_acoustic(x.acoustic_vtu,x.comsol_acoustic_csv,x.freq,x.outdir))
    (x.outdir/'fullfield_comparison_summary.json').write_text(json.dumps(s,indent=2),encoding='utf-8');print(json.dumps(s,indent=2))
if __name__=='__main__':main()
