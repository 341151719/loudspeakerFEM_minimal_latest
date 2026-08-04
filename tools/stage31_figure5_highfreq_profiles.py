from pathlib import Path
from collections import defaultdict
import numpy as np,pandas as pd,json
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
BASE=Path('/mnt/data/fig5_work');OUT=BASE/'figure5';direct=pd.read_csv(BASE/'req5/COMSOL_loudspeaker_req5_export_package/comsol_req5_raw/layer02_blocked_impedance_full_sweep.csv').set_index('freq_Hz')
def edgeseg(points,tri,doms,d):
 ed=defaultdict(list)
 for i in np.flatnonzero(doms==d):
  a,b,c=tri[i]
  for u,v in [(a,b),(b,c),(c,a)]:ed[tuple(sorted((int(u),int(v))))].append(i)
 return np.asarray([points[list(e)] for e,a in ed.items() if len(a)==1])
def pdist(points,segs):
 mid=segs.mean(1);tree=cKDTree(mid);k=min(20,len(segs));_,idx=tree.query(points,k=k);idx=np.atleast_2d(idx);idx=idx if idx.shape[0]==len(points) else idx.T;out=np.empty(len(points))
 for i,p in enumerate(points):
  s=segs[np.asarray(idx[i]).ravel()];ab=s[:,1]-s[:,0];t=np.clip(np.sum((p-s[:,0])*ab,1)/np.maximum(np.sum(ab*ab,1),1e-30),0,1);q=s[:,0]+t[:,None]*ab;out[i]=np.linalg.norm(p-q,axis=1).min()
 return out
def profile(dist,mag,maxd,nb=30):
 bins=np.linspace(0,maxd,nb+1);mid=.5*(bins[:-1]+bins[1:]);q=[];rms=[];n=[]
 for lo,hi in zip(bins[:-1],bins[1:]):
  x=mag[(dist>=lo)&(dist<hi)];q.append(np.percentile(x,90) if len(x)>=3 else np.nan);rms.append(np.sqrt(np.mean(x*x)) if len(x)>=3 else np.nan);n.append(len(x))
 q=np.asarray(q);norm=q/np.nanmax(q[:max(2,nb//8)]);good=np.isfinite(norm)&(norm>.1)&(norm<1.2)&(mid>0)
 delta=np.nan
 if good.sum()>=3:
  sl,_=np.polyfit(mid[good],np.log(norm[good]),1);delta=-1/sl if sl<0 else np.nan
 return pd.DataFrame({'depth_m':mid,'p90_abs_A_m2':q,'rms_abs_A_m2':rms,'normalized_p90':norm,'n':n}),delta
rows=[]
for f in [2000,5000,8000]:
 x=np.load(BASE/f'skin_p2_native/global_p2_{f}Hz.npz');pts=x['points'];tri=x['triangles'];dom=x['tri_domains'];cent=pts[tri].mean(1);J=x['Jphi_centroid_per_1A'];I=complex(direct.loc[f,'I_blocked_real_A'],direct.loc[f,'I_blocked_imag_A']);J=J*I
 for d in [6,23]:
  m=dom==d;seg=edgeseg(pts,tri,dom,d);dist=pdist(cent[m],seg);mag=abs(J[m]);pr,delta=profile(dist,mag,.002 if f==2000 else .0012);pr.to_csv(OUT/f'figure5_python_skin_profile_{f}Hz_domain{d}.csv',index=False)
  # area/axisymmetric volume weights and near-surface loss fractions using centroid approximation
  p=pts[tri[m]];area=.5*np.abs((p[:,1,0]-p[:,0,0])*(p[:,2,1]-p[:,0,1])-(p[:,2,0]-p[:,0,0])*(p[:,1,1]-p[:,0,1]));vol=2*np.pi*cent[m,0]*area;wloss=mag**2/1.12e7*vol
  total=wloss.sum();fra={thr:float(wloss[dist<=thr].sum()/total) for thr in [5e-5,1e-4,2.5e-4,5e-4]}
  rms=np.sqrt(np.sum(mag**2*vol)/np.sum(vol));imax=np.argmax(mag);wc=(cent[m]*wloss[:,None]).sum(0)/total
  rows.append({'freq_Hz':f,'domain_id':d,'fitted_skin_depth_mm':delta*1e3 if np.isfinite(delta) else np.nan,'current_crowding_factor_peak_over_rms':float(mag.max()/rms),'peak_r_mm':cent[m][imax,0]*1e3,'peak_z_mm':cent[m][imax,1]*1e3,'loss_centroid_r_mm':wc[0]*1e3,'loss_centroid_z_mm':wc[1]*1e3,'loss_fraction_within_0p05mm':fra[5e-5],'loss_fraction_within_0p1mm':fra[1e-4],'loss_fraction_within_0p25mm':fra[2.5e-4],'loss_fraction_within_0p5mm':fra[5e-4]})
pd.DataFrame(rows).to_csv(OUT/'figure5_highfreq_skin_crowding_predictions.csv',index=False)
# profile plot
fig,axs=plt.subplots(3,2,figsize=(12,13),constrained_layout=True)
for ax,(f,d) in zip(axs.ravel(),[(2000,6),(2000,23),(5000,6),(5000,23),(8000,6),(8000,23)]):
 p=pd.read_csv(OUT/f'figure5_python_skin_profile_{f}Hz_domain{d}.csv');ax.plot(p.depth_m*1e3,p.normalized_p90,'o-');ax.set(title=f'{f} Hz, domain {d}',xlabel='distance from iron boundary [mm]',ylabel='normalized p90 |Jphi|',ylim=(0,None));ax.grid(True)
fig.savefig(OUT/'figure5_highfreq_python_skin_profiles.png',dpi=180);plt.close(fig)
# field plots for all five frequencies
fig,axs=plt.subplots(5,3,figsize=(13,20),constrained_layout=True)
for row,f in enumerate([50,900,2000,5000,8000]):
 p=BASE/(f'global_p2_native/global_p2_{f}Hz.npz' if f in [50,900] else f'skin_p2_native/global_p2_{f}Hz.npz');x=np.load(p);pts=x['points'];tri=x['triangles'];dom=x['tri_domains'];J=x['Jphi_centroid_per_1A'];I=complex(direct.loc[f,'I_blocked_real_A'],direct.loc[f,'I_blocked_imag_A']);J=J*I;m=np.isin(dom,[6,23]);cen=pts[tri].mean(1)[m];vals=[J.real[m],J.imag[m],abs(J[m])];titles=['Re Jphi','Im Jphi','|Jphi|']
 for col,(v,t) in enumerate(zip(vals,titles)):
  if col<2:
   lim=np.percentile(abs(v),99.5);sc=axs[row,col].scatter(cen[:,0]*1e3,cen[:,1]*1e3,c=v,s=1,vmin=-lim,vmax=lim,cmap='coolwarm')
  else:
   sc=axs[row,col].scatter(cen[:,0]*1e3,cen[:,1]*1e3,c=np.log10(np.maximum(v,1e-30)),s=1,cmap='viridis')
  axs[row,col].set_aspect('equal');axs[row,col].set_title(f'{f} Hz {t}');axs[row,col].set_xlabel('r [mm]');axs[row,col].set_ylabel('z [mm]');fig.colorbar(sc,ax=axs[row,col])
fig.savefig(OUT/'figure5_python_complex_fields_50_900_2000_5000_8000Hz.png',dpi=180);plt.close(fig)
print(pd.DataFrame(rows).to_string(index=False))
