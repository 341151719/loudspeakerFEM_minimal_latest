#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,time,math,os,sys,hashlib
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'best_model')]
from coupled_solver import build_best_model,solve_frequency

def atomic_npz(p:Path,**kw):
 t=p.with_suffix('.tmp.npz');np.savez_compressed(t,**kw);os.replace(t,p)
def pherr(a,b):return float(np.degrees(np.angle(a*np.conj(b))))
def dberr(a,b):return float(20*np.log10(max(abs(a),1e-300)/max(abs(b),1e-300)))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--req5-raw',required=True);ap.add_argument('--outdir',required=True);ap.add_argument('--config');ap.add_argument('--magnetostatic-vtu');ap.add_argument('--resume',action='store_true');a=ap.parse_args();out=Path(a.outdir);solout=out/'checkpoints';solout.mkdir(parents=True,exist_ok=True)
 raw=Path(a.req5_raw);ref=pd.read_csv(raw/'layer03_impedance_power_decomposition.csv').sort_values('freq_Hz');blocked=raw/'layer02_blocked_impedance_full_sweep.csv';model=build_best_model(ROOT,config_path=a.config,magnetostatic_vtu=a.magnetostatic_vtu);rows=[]
 for i,r in ref.reset_index(drop=True).iterrows():
  f=float(r.freq_Hz);cp=solout/f'figure8_{f:09.3f}Hz.npz';t=time.time()
  if a.resume and cp.exists():
   d=np.load(cp,allow_pickle=False);I=complex(d['I']);Zb=complex(d['Zb']);Zm=complex(d['Zm']);Z=complex(d['Z']);pa=complex(d['p_axis']);u=d['u'];pb=d['p_base'];elapsed=float(d['elapsed_s'])
  else:
   s=solve_frequency(model,f,drive='voltage',voltage_V_peak=3.55,blocked_impedance_csv=blocked,nra_enabled=True);I=s.current_A_peak;Zb=s.blocked_impedance_ohm;Zm=s.motional_impedance_ohm;Z=s.total_impedance_ohm;pa=s.p_axis_1m_Pa_peak;u=s.solid_displacement;pb=s.pressure_base;elapsed=time.time()-t;atomic_npz(cp,f=np.array([f]),I=np.array([I]),Zb=np.array([Zb]),Zm=np.array([Zm]),Z=np.array([Z]),p_axis=np.array([pa]),u=u,p_base=pb,elapsed_s=np.array([elapsed]))
  Ic=complex(r.I_total_real_A,r.I_total_imag_A);Zc=complex(r.Z_total_real_ohm,r.Z_total_imag_ohm);Zmc=complex(r.Z_motional_real_ohm,r.Z_motional_imag_ohm);pc=complex(r.axis_pext_real_Pa,r.axis_pext_imag_Pa);Hp=pa/I;Hc=pc/Ic
  row={'freq_Hz':f,'python_I_real_A_peak':I.real,'python_I_imag_A_peak':I.imag,'COMSOL_I_real_A_peak':Ic.real,'COMSOL_I_imag_A_peak':Ic.imag,'I_abs_error_percent':100*(abs(I)/max(abs(Ic),1e-300)-1),'I_phase_error_deg':pherr(I,Ic),'python_Ztotal_real_ohm':Z.real,'python_Ztotal_imag_ohm':Z.imag,'COMSOL_Ztotal_real_ohm':Zc.real,'COMSOL_Ztotal_imag_ohm':Zc.imag,'Ztotal_complex_error_ohm':abs(Z-Zc),'python_Zmotional_real_ohm':Zm.real,'python_Zmotional_imag_ohm':Zm.imag,'COMSOL_Zmotional_real_ohm':Zmc.real,'COMSOL_Zmotional_imag_ohm':Zmc.imag,'Zmotional_complex_error_ohm':abs(Zm-Zmc),'python_p_axis_real_Pa_peak':pa.real,'python_p_axis_imag_Pa_peak':pa.imag,'COMSOL_p_axis_real_Pa_peak':pc.real,'COMSOL_p_axis_imag_Pa_peak':pc.imag,'p_axis_amplitude_error_dB':dberr(pa,pc),'p_axis_phase_error_deg':pherr(pa,pc),'p_axis_complex_NRE':abs(pa-pc)/max(abs(pc),1e-300),'python_HpI_real_Pa_per_A':Hp.real,'python_HpI_imag_Pa_per_A':Hp.imag,'COMSOL_HpI_real_Pa_per_A':Hc.real,'COMSOL_HpI_imag_Pa_per_A':Hc.imag,'HpI_amplitude_error_dB':dberr(Hp,Hc),'HpI_phase_error_deg':pherr(Hp,Hc),'HpI_complex_NRE':abs(Hp-Hc)/max(abs(Hc),1e-300),'python_SPL_RMS_dB':20*np.log10(max(abs(pa)/math.sqrt(2),1e-300)/20e-6),'COMSOL_SPL_RMS_dB':20*np.log10(max(abs(pc)/math.sqrt(2),1e-300)/20e-6),'elapsed_s':elapsed}
  rows.append(row);pd.DataFrame(rows).to_csv(out/'stage32_figure8_full126_partial.csv',index=False);print(json.dumps({'done':i+1,'total':len(ref),'freq_Hz':f,'amp_error_dB':row['p_axis_amplitude_error_dB'],'phase_error_deg':row['p_axis_phase_error_deg'],'elapsed_s':elapsed}),flush=True)
 pd.DataFrame(rows).to_csv(out/'stage32_figure8_full126.csv',index=False)
if __name__=='__main__':main()
