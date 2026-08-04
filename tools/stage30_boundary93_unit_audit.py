#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'best_model')]
from best_model.coupled_solver import build_best_model
from loudspeaker_axisym_fem.stage4F_hk_refinement import recover_acoustic_nodal_gradients_ppr
from loudspeaker_axisym_fem.exterior_field import hk_pressure_from_samples

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--magnetostatic-vtu',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();m=build_best_model(ROOT,magnetostatic_vtu=a.magnetostatic_vtu);ac=m.acoustic_model;gnodes=sorted({int(g) for seg,tag in zip(ac.mesh.line_cells,ac.mesh.line_tags) if int(tag)==93 for g in seg});xy=ac.mesh.points_rz_m[ac.acoustic_nodes_global];r,z=xy[:,0],xy[:,1];c=np.array([1.2+.3j,-2.1+.4j,.7-1.1j,3.2+.2j,-1.4+.8j,.9-.5j]);p=c[0]+c[1]*r+c[2]*z+c[3]*r*r+c[4]*r*z+c[5]*z*z;g=recover_acoustic_nodal_gradients_ppr(ac,p,target_global_nodes=gnodes);e=[];rn=[]
 for q in gnodes:
  li=ac.acoustic_node_map[q];rq,zq=ac.mesh.points_rz_m[q];ref=np.array([c[1]+2*c[3]*rq+c[4]*zq,c[2]+c[4]*rq+2*c[5]*zq]);e.append(np.linalg.norm(g[li]-ref));rn.append(np.linalg.norm(ref))
 ppr={'relative_L2_gradient_error':float(np.linalg.norm(e)/np.linalg.norm(rn)),'max_abs_gradient_error':float(max(e))}
 mono=[];aa=.165;R=1.;xg,wg=np.polynomial.legendre.leggauss(3)
 for f in [100.,1000.,8000.]:
  k=2*np.pi*f/m.config['air']['c0_m_s'];arr=[[] for _ in range(7)]
  for i in range(96):
   t0=np.pi*i/192;t1=np.pi*(i+1)/192
   for x,w in zip(xg,wg):
    th=.5*(t0+t1)+.5*(t1-t0)*x;wt=.5*(t1-t0)*w;p0=np.exp(-1j*k*aa)/(4*np.pi*aa);vals=[aa*np.sin(th),aa*np.cos(th),np.sin(th),np.cos(th),aa*wt,p0,p0*(-1j*k-1/aa)]
    for v,y in zip(arr,vals):v.append(y)
  pred=hk_pressure_from_samples(f,m.config['air']['c0_m_s'],*(np.asarray(x) for x in arr),obs_r=0.,obs_z=R,nphi=96,mirror=True,sign=-1)[0];ref=np.exp(-1j*k*R)/(4*np.pi*R);mono.append({'freq_Hz':f,'relative_error':float(abs(pred-ref)/abs(ref))})
 out={'PPR_quadratic_reproduction':ppr,'HK_monopole':mono,'pass':ppr['relative_L2_gradient_error']<1e-10 and max(x['relative_error'] for x in mono)<1e-8};Path(a.out).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
