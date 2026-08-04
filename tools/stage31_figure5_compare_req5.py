from __future__ import annotations
from pathlib import Path
import json,math,glob,sys
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
BASE=Path('/mnt/data/fig5_work');REQ=BASE/'req5/COMSOL_loudspeaker_req5_export_package/comsol_req5_raw';OUT=BASE/'figure5';OUT.mkdir(exist_ok=True)
cur=pd.read_csv(REQ/'layer02_induction_current_points.csv');zb=pd.read_csv(REQ/'layer02_blocked_impedance_full_sweep.csv').set_index('freq_Hz')

def p2shape(l):
 l0,l1,l2=l
 return np.array([l0*(2*l0-1),l1*(2*l1-1),l2*(2*l2-1),4*l0*l1,4*l1*l2,4*l2*l0])
def bary(p,a,b,c):
 T=np.column_stack([b-a,c-a]);uv=np.linalg.solve(T,p-a);return np.array([1-uv.sum(),uv[0],uv[1]])
def eval_python(data,pts,domains):
 xy=data['points'];tri=data['triangles'];td=data['tri_domains'];elem=data['elem_dofs'];A=data['A_phi_p2'];out=np.empty(len(pts),complex);eid=np.empty(len(pts),int)
 finder=mtri.Triangulation(xy[:,0],xy[:,1],tri).get_trifinder();cand=finder(pts[:,0],pts[:,1])
 cents=xy[tri].mean(1);trees={d:cKDTree(cents[td==d]) for d in np.unique(domains)};inds={d:np.flatnonzero(td==d) for d in np.unique(domains)}
 for i,(p,d) in enumerate(zip(pts,domains)):
  e=int(cand[i])
  if e<0 or int(td[e])!=int(d): e=int(inds[d][trees[d].query(p)[1]])
  lam=bary(p,xy[tri[e,0]],xy[tri[e,1]],xy[tri[e,2]]);out[i]=p2shape(lam)@A[elem[e]];eid[i]=e
 return out,eid

def boundary_segments(xy,tri,td,dom):
 from collections import defaultdict
 edge=defaultdict(list)
 for i in np.flatnonzero(td==dom):
  a,b,c=tri[i]
  for u,v in [(a,b),(b,c),(c,a)]:edge[tuple(sorted((int(u),int(v))))].append(i)
 seg=[]
 for e,adj in edge.items():
  # boundary of domain if edge has only one triangle of this domain; edge map built only on dom, so shared internal has 2
  if len(adj)==1:seg.append(xy[list(e)])
 return np.asarray(seg)
def point_seg_dist(p,a,b):
 ab=b-a;t=np.clip(np.sum((p-a)*ab,axis=1)/np.maximum(np.sum(ab*ab,axis=1),1e-30),0,1);q=a+t[:,None]*ab;return np.linalg.norm(p-q,axis=1)
def distances_to_boundary(pts,segs):
 mids=segs.mean(1);tree=cKDTree(mids);k=min(16,len(segs));_,ii=tree.query(pts,k=k);ii=np.atleast_2d(ii)
 if ii.shape[0]!=len(pts):ii=ii.T
 out=np.empty(len(pts))
 for n,p in enumerate(pts):
  s=segs[np.asarray(ii[n]).ravel()];out[n]=point_seg_dist(np.repeat(p[None,:],len(s),0),s[:,0],s[:,1]).min()
 return out
def corr(a,b):return abs(np.vdot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)+1e-300)
def metrics(py,co,pts):
 e=py-co;magc=np.abs(co);mask=magc>=magc.max()*0.1;phase=np.angle(py[mask]/co[mask],deg=True);alpha=np.vdot(py,co)/np.vdot(py,py)
 wp=np.abs(py);wc=np.abs(co);cp=(pts*wp[:,None]).sum(0)/wp.sum();cc=(pts*wc[:,None]).sum(0)/wc.sum();ip=np.argmax(wp);ic=np.argmax(wc)
 return {'complex_NRMSE':float(np.linalg.norm(e)/np.linalg.norm(co)),'magnitude_NRMSE':float(np.linalg.norm(np.abs(py)-np.abs(co))/np.linalg.norm(np.abs(co))),'phase_RMSE_deg_above_minus20dB':float(np.sqrt(np.mean(phase**2))),'complex_correlation_abs':float(corr(py,co)),'best_complex_scale_real':float(alpha.real),'best_complex_scale_imag':float(alpha.imag),'shape_NRMSE_after_complex_scale':float(np.linalg.norm(alpha*py-co)/np.linalg.norm(co)),'current_centroid_error_mm':float(np.linalg.norm(cp-cc)*1e3),'maximum_location_error_mm':float(np.linalg.norm(pts[ip]-pts[ic])*1e3),'python_peak_A_m2':float(wp.max()),'comsol_peak_A_m2':float(wc.max()),'n_points':len(py)}

def profile(dist,mag,maxd=0.002,nb=24):
 bins=np.linspace(0,maxd,nb+1);mid=.5*(bins[:-1]+bins[1:]);q=[];n=[]
 for lo,hi in zip(bins[:-1],bins[1:]):
  x=mag[(dist>=lo)&(dist<hi)];q.append(np.nan if len(x)<3 else np.percentile(x,90));n.append(len(x))
 q=np.asarray(q);good=np.isfinite(q)&(q>0);qnorm=q/np.nanmax(q[:max(2,nb//8)])
 fit=good&(qnorm>.1)&(qnorm<1.2)&(mid>0)
 if fit.sum()>=3:
  sl,ic=np.polyfit(mid[fit],np.log(qnorm[fit]),1);delta=-1/sl if sl<0 else np.nan
 else:delta=np.nan
 return pd.DataFrame({'depth_m':mid,'p90_abs':q,'normalized':qnorm,'n':n}),delta
allm=[];allp=[]
for f in [50.,900.]:
 d=np.load(BASE/f'global_p2_native/global_p2_{int(f)}Hz.npz');q=cur[cur.freq_Hz==f].copy();pts=q[['r/1[m]_real','z/1[m]_real']].to_numpy();dom=q.domain_id.to_numpy(int);A,eid=eval_python(d,pts,dom);Jper=-1j*2*np.pi*f*1.12e7*A;I=complex(zb.loc[f,'I_blocked_real_A'],zb.loc[f,'I_blocked_imag_A']);Jpy=Jper*I;Jco=q.mf_Jiphi_real.to_numpy()+1j*q.mf_Jiphi_imag.to_numpy() if 'mf_Jiphi_real' in q else q['mf.Jiphi_real'].to_numpy()+1j*q['mf.Jiphi_imag'].to_numpy()
 qout=q[['freq_Hz','domain_id','node_id','r/1[m]_real','z/1[m]_real']].copy();qout['python_J_real']=Jpy.real;qout['python_J_imag']=Jpy.imag;qout['python_J_abs']=abs(Jpy);qout['comsol_J_real']=Jco.real;qout['comsol_J_imag']=Jco.imag;qout['comsol_J_abs']=abs(Jco);qout['error_abs']=abs(Jpy-Jco);qout.to_csv(OUT/f'figure5_pointwise_{int(f)}Hz.csv',index=False)
 for dd in [6,23]:
  m=dom==dd;me=metrics(Jpy[m],Jco[m],pts[m]);me.update({'freq_Hz':f,'domain_id':dd,'comparison':'absolute_scaled_by_COMSOL_blocked_current'});allm.append(me)
  men=metrics(Jper[m],Jco[m]/I,pts[m]);men.update({'freq_Hz':f,'domain_id':dd,'comparison':'current_normalized_J_per_A'});allm.append(men)
  seg=boundary_segments(d['points'],d['triangles'],d['tri_domains'],dd);dist=distances_to_boundary(pts[m],seg);pp,dp=profile(dist,abs(Jpy[m]),.003 if f==50 else .0015);pc,dc=profile(dist,abs(Jco[m]),.003 if f==50 else .0015);pr=pp[['depth_m']].copy();pr['python_p90_abs']=pp.p90_abs;pr['python_norm']=pp.normalized;pr['comsol_p90_abs']=pc.p90_abs;pr['comsol_norm']=pc.normalized;pr['python_delta_fit_m']=dp;pr['comsol_delta_fit_m']=dc;pr.to_csv(OUT/f'figure5_skin_profile_{int(f)}Hz_domain{dd}.csv',index=False);allp.append({'freq_Hz':f,'domain_id':dd,'python_delta_fit_mm':dp*1e3 if np.isfinite(dp) else None,'comsol_delta_fit_mm':dc*1e3 if np.isfinite(dc) else None,'delta_error_percent':100*(dp/dc-1) if np.isfinite(dp) and np.isfinite(dc) else None})
# plots
pd.DataFrame(allm).to_csv(OUT/'figure5_direct_field_metrics.csv',index=False);pd.DataFrame(allp).to_csv(OUT/'figure5_skin_depth_metrics.csv',index=False)
for f in [50.,900.]:
 q=pd.read_csv(OUT/f'figure5_pointwise_{int(f)}Hz.csv');fig,axs=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
 for row,dd in enumerate([6,23]):
  x=q[q.domain_id==dd];vmin=np.log10(max(min(x.python_J_abs[x.python_J_abs>0].min(),x.comsol_J_abs[x.comsol_J_abs>0].min()),1e-30));vmax=np.log10(max(x.python_J_abs.max(),x.comsol_J_abs.max()))
  for col,(name,title) in enumerate([('comsol_J_abs','COMSOL'),('python_J_abs','Python global P2'),('error_abs','|complex error|')]):
   val=np.log10(np.maximum(x[name],1e-30));sc=axs[row,col].scatter(x['r/1[m]_real']*1e3,x['z/1[m]_real']*1e3,c=val,s=5);axs[row,col].set_aspect('equal');axs[row,col].set_title(f'{title}, domain {dd}');axs[row,col].set_xlabel('r [mm]');axs[row,col].set_ylabel('z [mm]');fig.colorbar(sc,ax=axs[row,col],label='log10 A/m²')
 fig.suptitle(f'Figure 5 direct Jphi comparison, {f:g} Hz');fig.savefig(OUT/f'figure5_direct_comparison_{int(f)}Hz.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
for ax,(f,dd) in zip(axs.ravel(),[(50,6),(50,23),(900,6),(900,23)]):
 p=pd.read_csv(OUT/f'figure5_skin_profile_{f}Hz_domain{dd}.csv');ax.plot(p.depth_m*1e3,p.comsol_norm,'o-',label='COMSOL');ax.plot(p.depth_m*1e3,p.python_norm,'s--',label='Python');ax.set(title=f'{f} Hz, domain {dd}',xlabel='distance from iron boundary [mm]',ylabel='normalized p90 |Jphi|',ylim=(0,None));ax.grid(True);ax.legend()
fig.savefig(OUT/'figure5_skin_depth_profiles.png',dpi=180);plt.close(fig)
summary={'metrics':allm,'skin_profiles':allp,'normalization':'Python unit-current field multiplied by direct COMSOL blocked current; normalized metrics compare J/I','phasor':'complex values used without best-fit correction; best complex scale reported only as diagnostic'}
(OUT/'figure5_direct_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(pd.DataFrame(allm).to_string(index=False));print(pd.DataFrame(allp).to_string(index=False))
