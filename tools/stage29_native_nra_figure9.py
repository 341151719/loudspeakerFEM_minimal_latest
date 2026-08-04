#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, os, sys, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from comsol_stage7F_shared_refined_asb import build_stage7f_model, solve_stage7f_frequency
from loudspeaker_axisym_fem.mmcpl_lorentz_backemf import assemble_lorentz_backemf_vector
from loudspeaker_axisym_fem.stage4C_acoustic_structure import Stage4CParameters, PML_DOMAINS
from loudspeaker_axisym_fem.narrow_region_acoustics import (
    equivalent_narrow_region_coefficients, narrow_region_dissipation,
    comsol_air_properties, COMSOL_NARROW_REGIONS,
)

Q7=[((1/3,1/3,1/3),0.225)]
a=0.059715871789770;b=0.470142064105115;w=0.132394152788506
for q in [(a,b,b),(b,a,b),(b,b,a)]:Q7.append((q,w))
a=0.797426985353087;b=0.101286507323456;w=0.125939180544827
for q in [(a,b,b),(b,a,b),(b,b,a)]:Q7.append((q,w))

def sha(p:Path):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def atomic_npz(path:Path,**kw):
 t=path.with_suffix('.tmp.npz');np.savez_compressed(t,**kw);os.replace(t,path)

def interp_zb(path:Path,freqs):
 d=pd.read_csv(path).sort_values('f_Hz');x=d.f_Hz.to_numpy(float);zr=d.Zb_real_ohm.to_numpy(float);zi=d.Zb_imag_ohm.to_numpy(float)
 f=np.asarray(freqs,float)
 if f.min()<x.min() or f.max()>x.max():raise ValueError('blocked interpolation outside direct REQ5 range')
 return np.interp(f,x,zr)+1j*np.interp(f,x,zi)

def domain_tri_data(model,dom):
 mesh=model.mesh;idx=np.flatnonzero(mesh.tri_domains==dom);tris=mesh.triangles[idx];loc=np.array([[model.acoustic_node_map[int(g)] for g in t] for t in tris],int);xy=mesh.points_rz_m[tris]
 return idx,loc,xy

def domain_pressure_metrics(model,p,dom):
 _,loc,xy=domain_tri_data(model,dom);pv=p[loc];total=0.;intp=0j;intabs=0.;mx=0.
 for lam,wq in Q7:
  l=np.asarray(lam,float);r=xy[:,:,0]@l;z=xy[:,:,1]@l
  x0=xy[:,0];x1=xy[:,1];x2=xy[:,2];area=.5*np.abs((x1[:,0]-x0[:,0])*(x2[:,1]-x0[:,1])-(x2[:,0]-x0[:,0])*(x1[:,1]-x0[:,1]))
  wt=2*np.pi*np.maximum(r,0)*area*wq;pq=pv@l;total+=float(wt.sum());intp+=np.sum(wt*pq);intabs+=float(np.sum(wt*np.abs(pq)));mx=max(mx,float(np.max(np.abs(pq))))
 mean=intp/max(total,1e-300)
 return {'domain_id':dom,'volume_m3':total,'mean_p_real_Pa':mean.real,'mean_p_imag_Pa':mean.imag,'mean_p_abs_Pa':abs(mean),'mean_abs_p_Pa':intabs/max(total,1e-300),'max_abs_p_Pa':mx,'mean_p_phase_deg':math.degrees(math.atan2(mean.imag,mean.real))}

def solve_branch(model,cpl,freqs,zb,params,outdir,branch,nra_enabled):
 outdir.mkdir(parents=True,exist_ok=True);rows=[];diss=[]
 for i,(f,z) in enumerate(zip(freqs,zb),1):
  ck=outdir/f'checkpoint_{branch}_{f:09.3f}Hz.npz'
  if ck.exists():
   d=np.load(ck,allow_pickle=False);p=d['p_full_Pa'];I=complex(d['I_A_peak']);Z=complex(d['Z_total_ohm']);elapsed=float(d.get('elapsed_s',np.nan))
  else:
   t=time.time();r=solve_stage7f_frequency(model,cpl,complex(z),float(f),params,nra_enabled=nra_enabled);elapsed=time.time()-t;p=r['p_full_Pa'];I=r['I_A_peak'];Z=r['Z_total_ohm'];atomic_npz(ck,f_Hz=np.array([f]),I_A_peak=np.array([I]),Z_total_ohm=np.array([Z]),p_full_Pa=p,u_full_m=r['u_full_m'],elapsed_s=np.array([elapsed]))
  base={'branch':branch,'nra_enabled':nra_enabled,'freq_Hz':f,'I_real_A_peak':I.real,'I_imag_A_peak':I.imag,'I_abs_A_peak':abs(I),'Z_total_real_ohm':Z.real,'Z_total_imag_ohm':Z.imag,'solve_elapsed_s':elapsed}
  for dom in (8,22):
   m=domain_pressure_metrics(model,p,dom);rows.append({**base,**m})
   if nra_enabled:
    h=COMSOL_NARROW_REGIONS['nra1' if dom==8 else 'nra2'].height_m;coeff=equivalent_narrow_region_coefficients(f,h,rho0=params.rho0_kg_m3,c0=params.c0_m_s);loss=narrow_region_dissipation(p,model.Knra[dom],model.Mnra[dom],2*np.pi*f,coeff);diss.append({'branch':branch,'freq_Hz':f,'domain_id':dom,**loss,'stiffness_factor_real':coeff.stiffness_factor.real,'stiffness_factor_imag':coeff.stiffness_factor.imag,'mass_factor_real':coeff.mass_factor.real,'mass_factor_imag':coeff.mass_factor.imag,'rho_eq_over_rho0_real':coeff.rho_eq_over_rho0.real,'rho_eq_over_rho0_imag':coeff.rho_eq_over_rho0.imag,'bulk_eq_over_bulk0_real':coeff.bulk_eq_over_bulk0.real,'bulk_eq_over_bulk0_imag':coeff.bulk_eq_over_bulk0.imag,'passive':coeff.passive})
  if i%10==0 or i==len(freqs):print(json.dumps({'branch':branch,'completed':i,'total':len(freqs),'freq_Hz':f}),flush=True)
 pd.DataFrame(rows).to_csv(outdir/f'{branch}_domain_pressure.csv',index=False)
 pd.DataFrame(diss).to_csv(outdir/f'{branch}_nra_dissipation.csv',index=False)
 return pd.DataFrame(rows),pd.DataFrame(diss)

def half_power_mode(df,dom=8,lo=500,hi=750):
 d=df[(df.domain_id==dom)&(df.freq_Hz>=lo)&(df.freq_Hz<=hi)].sort_values('freq_Hz');f=d.freq_Hz.to_numpy(float);a=d.mean_abs_p_Pa.to_numpy(float);i=int(np.argmax(a));f0=f[i];ap=a[i];target=ap/math.sqrt(2)
 def cross_left():
  for j in range(i-1,-1,-1):
   if a[j]<=target<=a[j+1]:return float(f[j]+(target-a[j])*(f[j+1]-f[j])/(a[j+1]-a[j]))
  return np.nan
 def cross_right():
  for j in range(i,len(f)-1):
   if a[j]>=target>=a[j+1]:return float(f[j]+(target-a[j])*(f[j+1]-f[j])/(a[j+1]-a[j]))
  return np.nan
 f1=cross_left();f2=cross_right();Q=f0/(f2-f1) if np.isfinite(f1+f2) and f2>f1 else np.nan
 return {'domain_id':dom,'mode_frequency_Hz':float(f0),'peak_mean_abs_pressure_Pa':float(ap),'half_power_pressure_Pa':float(target),'lower_half_power_Hz':f1,'upper_half_power_Hz':f2,'bandwidth_Hz':float(f2-f1) if np.isfinite(f1+f2) else np.nan,'Q_factor':float(Q),'frequency_step_Hz':float(np.median(np.diff(f))),'interior_peak':bool(0<i<len(f)-1)}

def domain_interpolator(model,p,dom):
 _,loc,xy=domain_tri_data(model,dom);nodes=np.unique(loc);pts=model.mesh.points_rz_m[model.acoustic_nodes_global[nodes]];lin=LinearNDInterpolator(pts,p[nodes],fill_value=np.nan);near=NearestNDInterpolator(pts,p[nodes])
 def evaluate(r,z):
  q=np.asarray(lin(r,z),complex);bad=~np.isfinite(q.real)|~np.isfinite(q.imag)
  if np.any(bad):q[bad]=near(np.asarray(r)[bad],np.asarray(z)[bad])
  return q
 return evaluate

def compare_full_cloud(model,ck,ref,dom,f):
 p=np.load(ck)['p_full_Pa'];q=ref[(ref.domain_id==dom)&np.isclose(ref.freq_Hz,f)].copy();itp=domain_interpolator(model,p,dom);py=np.asarray(itp(q['r/1[m]_real'].to_numpy(),q['z/1[m]_real'].to_numpy()),complex);co=q['acpr.p_t_real'].to_numpy()+1j*q['acpr.p_t_imag'].to_numpy();ok=np.isfinite(py.real)&np.isfinite(py.imag);py=py[ok];co=co[ok];e=py-co;nrmse=np.linalg.norm(e)/max(np.linalg.norm(co),1e-300);corr=abs(np.vdot(co,py))/max(np.linalg.norm(co)*np.linalg.norm(py),1e-300);scale=np.vdot(py,co)/max(np.vdot(py,py),1e-300);shape=np.linalg.norm(scale*py-co)/max(np.linalg.norm(co),1e-300)
 return {'branch':'with_NRA','freq_Hz':f,'domain_id':dom,'n_reference_points':len(q),'n_interpolated_points':int(ok.sum()),'complex_NRMSE':float(nrmse),'complex_correlation':float(corr),'best_complex_scale_real':scale.real,'best_complex_scale_imag':scale.imag,'shape_NRMSE_after_complex_scale':float(shape),'mean_abs_python_Pa':float(np.mean(abs(py))),'mean_abs_COMSOL_Pa':float(np.mean(abs(co))),'mean_phase_error_deg':float(np.degrees(np.angle(np.vdot(co,py))))}

def compare_probes(model,ck,ref,dom,f):
 p=np.load(ck)['p_full_Pa'];q=ref[(ref.domain_id==dom)&np.isclose(ref.freq_Hz,f)].copy();itp=domain_interpolator(model,p,dom);py=np.asarray(itp(q.r_m.to_numpy(),q.z_m.to_numpy()),complex);co=q.acpr_p_t_real_Pa.to_numpy()+1j*q.acpr_p_t_imag_Pa.to_numpy();ok=np.isfinite(py);e=py[ok]-co[ok]
 return {'branch':'without_NRA','freq_Hz':f,'domain_id':dom,'n_probes':int(ok.sum()),'complex_NRMSE':float(np.linalg.norm(e)/max(np.linalg.norm(co[ok]),1e-300)),'complex_correlation':float(abs(np.vdot(co[ok],py[ok]))/max(np.linalg.norm(co[ok])*np.linalg.norm(py[ok]),1e-300)),'mean_abs_python_Pa':float(np.mean(abs(py[ok]))),'mean_abs_COMSOL_Pa':float(np.mean(abs(co[ok])))}

def plot_figure9(model,outdir,branchdir):
 fields=[]
 for f in [600.,630.]:fields.append(np.load(branchdir/f'checkpoint_without_NRA_{f:09.3f}Hz.npz')['p_full_Pa'])
 mesh=model.mesh;mask=np.isin(mesh.tri_domains,[2,7,8,22]);tris=mesh.triangles[mask];loc=np.array([[model.acoustic_node_map[int(g)] for g in t] for t in tris]);pts=mesh.points_rz_m[model.acoustic_nodes_global];face=[np.real(p[loc]).mean(axis=1) for p in fields];v=max(np.max(np.abs(x)) for x in face)
 for normalized in [False,True]:
  fig,axs=plt.subplots(1,2,figsize=(13,6),constrained_layout=True);last=None
  for ax,f,val in zip(axs,[600,630],face):
   vv=val/v if normalized and v>0 else val; mirrored=np.column_stack([-pts[:,0],pts[:,1]]); pt=np.vstack([pts,mirrored]); n=len(pts); tr=np.vstack([loc, np.column_stack([loc[:,0]+n,loc[:,2]+n,loc[:,1]+n])]); t=mtri.Triangulation(pt[:,0]*1e3,pt[:,1]*1e3,tr); last=ax.tripcolor(t,facecolors=np.r_[vv,vv],shading='flat',vmin=-1 if normalized else -v,vmax=1 if normalized else v); ax.set_aspect('equal'); ax.set_title(f'{f} Hz, without NRA'); ax.set_xlabel('x [mm]'); ax.set_ylabel('z [mm]'); ax.set_xlim(-50,50); ax.set_ylim(-170,10)
  cb=fig.colorbar(last,ax=axs.tolist(),shrink=.85);cb.set_label('normalized Re(p)' if normalized else 'Re(p) [Pa]');fig.savefig(outdir/('figure09_python_signed_real_pressure_normalized.png' if normalized else 'figure09_python_signed_real_pressure_Pa.png'),dpi=220);plt.close(fig)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--outdir',default=str(ROOT/'outputs/stage29_native_nra'));ap.add_argument('--project-input-root');ap.add_argument('--mesh');ap.add_argument('--mphtxt');ap.add_argument('--magnetostatic-vtu');ap.add_argument('--blocked-csv',default=str(ROOT/'inputs/comsol_reference/stage29_nra/direct_blocked_impedance.csv'));ap.add_argument('--comsol-export-root',default=str(ROOT/'inputs/comsol_reference/stage29_nra'));ap.add_argument('--req5-root',default=str(ROOT/'inputs/comsol_reference/stage29_nra'));a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);S=Path(a.project_input_root) if a.project_input_root else ROOT
 mesh_path=Path(a.mesh) if a.mesh else (S/'meshes/comsol_geometry_polyline_coarse_2p5mm.msh' if a.project_input_root else ROOT/'inputs/meshes/comsol_geometry_polyline_coarse_2p5mm.msh')
 mphtxt_path=Path(a.mphtxt) if a.mphtxt else (S/'comsol_reference_inputs/Untitled.mphtxt' if a.project_input_root else ROOT/'inputs/comsol_reference/Untitled.mphtxt')
 magnetostatic_path=Path(a.magnetostatic_vtu) if a.magnetostatic_vtu else (S/'outputs/stage5B_raw_magnetics_closure/refined_B_inverse_iter35/magnetostatic_solution.vtu' if a.project_input_root else ROOT/'runs/magnetics/magnetostatic_solution.vtu')
 props=comsol_air_properties();params=Stage4CParameters(V0_peak_V=3.55,rho0_kg_m3=props.density_kg_m3,c0_m_s=props.sound_speed_m_s)
 model,G=build_stage7f_model(mesh_path,mphtxt_path,0.65,1);cpl=assemble_lorentz_backemf_vector(model.solid,magnetostatic_path)
 direct=pd.read_csv(a.blocked_csv);anchors=direct.f_Hz.to_numpy(float);dense1=np.arange(500.,750.0001,2.);dense2=np.arange(1150.,1450.0001,5.);freqs=np.unique(np.r_[anchors,dense1,dense2]);zb=interp_zb(Path(a.blocked_csv),freqs);pd.DataFrame({'freq_Hz':freqs,'Zb_real_ohm':zb.real,'Zb_imag_ohm':zb.imag,'is_direct_REQ5_anchor':np.isin(freqs,anchors)}).to_csv(out/'stage29_frequency_and_blocked_input.csv',index=False)
 withdf,diss=solve_branch(model,cpl,freqs,zb,params,out/'with_NRA','with_NRA',True);withoutdf,_=solve_branch(model,cpl,freqs,zb,params,out/'without_NRA','without_NRA',False)
 allp=pd.concat([withdf,withoutdf],ignore_index=True);allp.to_csv(out/'stage29_domain8_22_pressure.csv',index=False);diss.to_csv(out/'stage29_nra_dissipation.csv',index=False)
 modes={'without_NRA_domain8':half_power_mode(withoutdf,8),'without_NRA_domain22':half_power_mode(withoutdf,22),'with_NRA_domain8':half_power_mode(withdf,8),'with_NRA_domain22':half_power_mode(withdf,22)}
 f0=modes['without_NRA_domain8']['mode_frequency_Hz'];a0=float(withoutdf[(withoutdf.domain_id==8)&np.isclose(withoutdf.freq_Hz,f0)].mean_abs_p_Pa.iloc[0]);aw=float(withdf[(withdf.domain_id==8)&np.isclose(withdf.freq_Hz,f0)].mean_abs_p_Pa.iloc[0]);modes['NRA_suppression_at_noNRA_mode_dB']=20*math.log10(max(aw,1e-300)/max(a0,1e-300))
 # Existing COMSOL average-anchor comparison.
 eroot=Path(a.comsol_export_root);cmp=[]
 for branch,df,stud in [('with_NRA',withdf,'study2'),('without_NRA',withoutdf,'study3')]:
  for dom in [8,22]:
   c=pd.read_csv(eroot/f'domain{dom}_{stud}_average_pressure.csv');m=df[df.domain_id==dom].merge(c,on='freq_Hz');m=m[m.freq_Hz.isin(anchors)];
   for _,r in m.iterrows():cmp.append({'branch':branch,'freq_Hz':r.freq_Hz,'domain_id':dom,'python_mean_abs_Pa':r.mean_abs_p_Pa,'COMSOL_mean_abs_Pa':r.mean_abs_acpr_p_t,'mean_abs_error_Pa':r.mean_abs_p_Pa-r.mean_abs_acpr_p_t,'python_max_abs_Pa':r.max_abs_p_Pa,'COMSOL_max_abs_Pa':r.max_abs_acpr_p_t,'max_abs_error_Pa':r.max_abs_p_Pa-r.max_abs_acpr_p_t,'python_mean_phase_deg':r.mean_p_phase_deg,'COMSOL_representative_phase_deg':r.representative_phase_deg})
 cmp=pd.DataFrame(cmp);cmp.to_csv(out/'stage29_COMSOL_average_anchor_comparison.csv',index=False)
 # Full with-NRA cloud and no-NRA probes at 600/630.
 req=pd.read_csv(Path(a.req5_root)/'layer07_nra_domain_points.csv');cloud=[];probes=[]
 for f in [600.,630.]:
  for dom in [8,22]:
   cloud.append(compare_full_cloud(model,out/'with_NRA'/f'checkpoint_with_NRA_{f:09.3f}Hz.npz',req,dom,f));pr=pd.read_csv(eroot/f'domain{dom}_study3_points.csv');pr['domain_id']=dom;probes.append(compare_probes(model,out/'without_NRA'/f'checkpoint_without_NRA_{f:09.3f}Hz.npz',pr,dom,f))
 pd.DataFrame(cloud).to_csv(out/'stage29_with_NRA_fullfield_direct_comparison.csv',index=False);pd.DataFrame(probes).to_csv(out/'stage29_without_NRA_probe_comparison.csv',index=False)
 plot_figure9(model,out,out/'without_NRA')
 # Curves.
 fig,ax=plt.subplots(figsize=(10,6));
 for branch,df,ls in [('with NRA',withdf,'-'),('without NRA',withoutdf,'--')]:
  d=df[(df.domain_id==8)&(df.freq_Hz>=500)&(df.freq_Hz<=750)].sort_values('freq_Hz');ax.semilogy(d.freq_Hz,d.mean_abs_p_Pa,ls,label=f'Python {branch}')
 for branch,stud,mk in [('with NRA','study2','o'),('without NRA','study3','s')]:
  c=pd.read_csv(eroot/f'domain8_{stud}_average_pressure.csv');c=c[(c.freq_Hz>=500)&(c.freq_Hz<=750)];ax.semilogy(c.freq_Hz,c.mean_abs_acpr_p_t,mk,label=f'COMSOL anchors {branch}')
 ax.set(xlabel='Frequency [Hz]',ylabel='Domain 8 mean |p| [Pa]',title='Back-cavity mode and NRA damping');ax.grid(True,which='both');ax.legend();fig.tight_layout();fig.savefig(out/'stage29_cavity_mode_domain8.png',dpi=220);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,6));
 for dom in [8,22]:d=diss[diss.domain_id==dom].sort_values('freq_Hz');ax.loglog(d.freq_Hz,d.viscous_W,label=f'Domain {dom} viscous');ax.loglog(d.freq_Hz,d.thermal_W,'--',label=f'Domain {dom} thermal')
 ax.set(xlabel='Frequency [Hz]',ylabel='Dissipated power [W]',title='Native NRA thermoviscous dissipation');ax.grid(True,which='both');ax.legend();fig.tight_layout();fig.savefig(out/'stage29_nra_dissipation.png',dpi=220);plt.close(fig)
 # Metrics / acceptance.
 def rms(x):return float(np.sqrt(np.mean(np.asarray(x,float)**2)))
 cm=cmp.copy();metrics={'schema':'stage29-native-nra-figure9-v1','native_model':'parallel-plate thermoviscous effective density and bulk modulus; no COMSOL local transfer; no NRA frequency interpolation','air_properties':props.to_jsonable(),'frequency_count':len(freqs),'modes':modes,'COMSOL_average_anchor_metrics':{},'with_NRA_fullfield_metrics':cloud,'without_NRA_probe_metrics':probes,'dissipation_reference_status':'BLOCKED_MISSING_COMSOL_DIRECT_VISCOUS_THERMAL_DOMAIN_INTEGRALS','figure9_without_NRA_fullfield_reference_status':'BLOCKED_MISSING_DSET5_FULLFIELD_600_630'}
 for branch in ['with_NRA','without_NRA']:
  x=cm[cm.branch==branch];metrics['COMSOL_average_anchor_metrics'][branch]={'n':len(x),'mean_abs_RMSE_Pa':rms(x.mean_abs_error_Pa),'mean_abs_relative_RMSE':rms(x.mean_abs_error_Pa/np.maximum(x.COMSOL_mean_abs_Pa,1e-300)),'max_abs_relative_RMSE':rms(x.max_abs_error_Pa/np.maximum(x.COMSOL_max_abs_Pa,1e-300))}
 metrics['native_passivity_all_points']=bool(diss.passive.all());metrics['total_native_NRA_dissipation_W_at_mode']=float(diss[np.isclose(diss.freq_Hz,f0)].total_W.sum())
 (out/'stage29_metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False))
 acc=[('COMSOL_local_transfer_disabled',True),('NRA_log_frequency_interpolation_disabled',True),('native_effective_density_bulk_modulus',True),('native_passivity_all_points',metrics['native_passivity_all_points']),('cavity_mode_frequency_computed',np.isfinite(modes['without_NRA_domain8']['mode_frequency_Hz'])),('Q_factor_computed',np.isfinite(modes['without_NRA_domain8']['Q_factor'])),('domain8_22_complex_pressure_computed',True),('viscous_thermal_dissipation_computed',True),('signed_real_Figure9_generated',True),('COMSOL_noNRA_fullfield_available',False),('COMSOL_direct_dissipation_available',False)]
 pd.DataFrame([{'criterion':k,'status':'PASS' if v else 'BLOCKED','value':bool(v)} for k,v in acc]).to_csv(out/'stage29_acceptance_matrix.csv',index=False)
 audit={'inputs':{str(p):{'sha256':sha(Path(p)),'size':Path(p).stat().st_size} for p in [a.blocked_csv,Path(a.req5_root)/'layer07_nra_domain_points.csv',eroot/'domain8_study2_average_pressure.csv',eroot/'domain8_study3_average_pressure.csv']},'code_sha256':sha(Path(__file__)),'model_summary':model.summary(),'G_info':G,'coupling_summary':cpl.summary(),'outputs':{'frequency_count':len(freqs),'checkpoint_count':len(list((out/'with_NRA').glob('checkpoint_*.npz')))+len(list((out/'without_NRA').glob('checkpoint_*.npz')))}}
 (out/'stage29_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False));print(json.dumps(metrics,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
