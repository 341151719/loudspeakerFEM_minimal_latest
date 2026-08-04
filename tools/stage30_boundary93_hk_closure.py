#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,time,sys
from pathlib import Path
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'best_model')]
from best_model.coupled_solver import build_best_model,solve_frequency
from loudspeaker_axisym_fem.stage4F_hk_refinement import boundary93_hk_samples_recovered
from loudspeaker_axisym_fem.exterior_field import hk_pressure_from_samples

def complex_metrics(z,ref,theta):
 e=z-ref;scale=np.vdot(z,ref)/max(np.vdot(z,z).real,1e-300);amp=20*np.log10(np.maximum(abs(z),1e-300)/np.maximum(abs(ref),1e-300));phase=np.angle(z/np.where(abs(ref)>0,ref,1),deg=True);i0=int(np.argmin(abs(theta)));rel=20*np.log10(np.maximum(abs(z)/max(abs(z[i0]),1e-300),1e-300));relref=20*np.log10(np.maximum(abs(ref)/max(abs(ref[i0]),1e-300),1e-300));main=relref>=-20
 return {'complex_NRMSE':float(np.linalg.norm(e)/max(np.linalg.norm(ref),1e-300)),'shape_NRMSE_after_complex_scale':float(np.linalg.norm(z*scale-ref)/max(np.linalg.norm(ref),1e-300)),'best_scale_abs':float(abs(scale)),'best_scale_phase_deg':float(np.angle(scale,deg=True)),'amplitude_RMSE_dB':float(np.sqrt(np.mean(amp**2))),'phase_RMSE_deg':float(np.sqrt(np.mean(phase**2))),'phase_main_ge_minus20dB_RMSE_deg':float(np.sqrt(np.mean(phase[main]**2))),'relative_directivity_RMSE_dB':float(np.sqrt(np.mean((rel-relref)**2))),'axis_amplitude_error_dB':float(amp[i0]),'axis_phase_error_deg':float(phase[i0])},amp,phase,rel,relref

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--req5-raw',required=True);ap.add_argument('--magnetostatic-vtu',required=True);ap.add_argument('--blocked-impedance-csv');ap.add_argument('--config');ap.add_argument('--outdir',required=True);ap.add_argument('--frequencies',default='20,50,100,600,630,1000,1320,2000,5000,6300,8000');ap.add_argument('--voltage-peak',type=float,default=3.55);a=ap.parse_args()
 out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);raw=Path(a.req5_raw);l9=pd.read_csv(raw/'layer09_farfield_directivity_matrix.csv');blocked=Path(a.blocked_impedance_csv) if a.blocked_impedance_csv else raw/'layer02_blocked_impedance_full_sweep.csv';freqs=[float(x) for x in a.frequencies.split(',') if x.strip()];model=build_best_model(ROOT,config_path=a.config,magnetostatic_vtu=a.magnetostatic_vtu);rows=[]
 for f in freqs:
  t=time.time();sol=solve_frequency(model,f,drive='voltage',voltage_V_peak=a.voltage_peak,blocked_impedance_csv=blocked,nra_enabled=True);d=l9[np.isclose(l9.freq_Hz,f)].sort_values('theta_deg');theta=d.theta_deg.to_numpy(float);ref=d.pext_real_Pa.to_numpy()+1j*d.pext_imag_Pa.to_numpy();th=np.radians(theta);ro=abs(np.sin(th));zo=np.cos(th)
  for method,radial,label in [('zz',False,'legacy_ZZ'),('ppr',True,'PPR')]:
   sm,info=boundary93_hk_samples_recovered(model.acoustic_model,sol.pressure_base,recovery_method=method,force_radial_normals=radial);z=hk_pressure_from_samples(f,model.config['air']['c0_m_s'],*sm,obs_r=ro,obs_z=zo,nphi=int(model.config['exterior']['azimuth_quadrature_points']),mirror=True,sign=-1);m,amp,phase,rel,relref=complex_metrics(z,ref,theta);rows.append({'freq_Hz':f,'method':label,'elapsed_s':time.time()-t,'mean_abs_dpdn_Pa_per_m':info.mean_abs_dpdn_Pa_per_m,**m});pd.DataFrame({'freq_Hz':f,'method':label,'theta_deg':theta,'python_real_Pa':z.real,'python_imag_Pa':z.imag,'COMSOL_real_Pa':ref.real,'COMSOL_imag_Pa':ref.imag,'amplitude_error_dB':amp,'phase_error_deg':phase,'relative_python_dB':rel,'relative_COMSOL_dB':relref}).to_csv(out/f'farfield_{f:07.1f}Hz_{label}.csv',index=False)
  pd.DataFrame(rows).to_csv(out/'stage30_boundary93_hk_metrics.csv',index=False);print(json.dumps(rows[-1]),flush=True)
if __name__=='__main__':main()
