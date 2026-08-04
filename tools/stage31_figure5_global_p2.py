from __future__ import annotations
from pathlib import Path
import json,math,time,resource,sys,gc,ctypes
import numpy as np,pandas as pd,meshio
from scipy.interpolate import PchipInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spilu,LinearOperator,gmres
P=Path('/mnt/data/fig5_work/project/loudspeaker_comsol_reproduction_best_v1');sys.path[:0]=[str(P/'src')]
from loudspeaker_axisym_fem.axisym_magnetics import MU0,load_tagged_meshio,_tri_geometry
from loudspeaker_axisym_fem.comsol_driver_model import SOFT_IRON_BH_TABLE
_Q=[((1/3,1/3,1/3),0.225)]
a=.059715871789770;b=.470142064105115;w=.132394152788506
_Q += [((a,b,b),w),((b,a,b),w),((b,b,a),w)]
a=.797426985353087;b=.101286507323456;w=.125939180544827
_Q += [((a,b,b),w),((b,a,b),w),((b,b,a),w)]

def topology(mesh):
 t=mesh.triangles;ne=len(t);E=np.vstack([np.sort(t[:,[0,1]],1),np.sort(t[:,[1,2]],1),np.sort(t[:,[2,0]],1)])
 ue,inv=np.unique(E,axis=0,return_inverse=True);ed=np.column_stack([inv[:ne],inv[ne:2*ne],inv[2*ne:]])+mesh.n_nodes
 return np.column_stack([t,ed]),{tuple(map(int,e)):i for i,e in enumerate(ue)},ue

def shape_batch(l,dr,dz):
 l0,l1,l2=l;N=np.array([l0*(2*l0-1),l1*(2*l1-1),l2*(2*l2-1),4*l0*l1,4*l1*l2,4*l2*l0])
 gr=np.column_stack([(4*l0-1)*dr[:,0],(4*l1-1)*dr[:,1],(4*l2-1)*dr[:,2],4*(l0*dr[:,1]+l1*dr[:,0]),4*(l1*dr[:,2]+l2*dr[:,1]),4*(l2*dr[:,0]+l0*dr[:,2])])
 gz=np.column_stack([(4*l0-1)*dz[:,0],(4*l1-1)*dz[:,1],(4*l2-1)*dz[:,2],4*(l0*dz[:,1]+l1*dz[:,0]),4*(l1*dz[:,2]+l2*dz[:,1]),4*(l2*dz[:,0]+l0*dz[:,2])])
 return N,gr,gz

def incremental_tensor(dom,Br,Bz,B,H,mu_static):
 n=len(dom);nrr=np.empty(n);nrz=np.zeros(n);nzz=np.empty(n)
 # default isotropic static secant for non-soft domains
 nu0=1/(MU0*np.maximum(mu_static,1.0));nrr[:]=nu0;nzz[:]=nu0
 soft=np.isin(dom,[6,23]);tab=np.asarray(SOFT_IRON_BH_TABLE,float);fun=PchipInterpolator(tab[:,0],tab[:,1],extrapolate=True);d=fun.derivative()
 hs=np.maximum(H[soft],0.0);bs=np.maximum(B[soft],1e-12)
 mupar=np.clip(d(hs),MU0,4000*MU0);muperp=np.clip(bs/np.maximum(hs,1e-12),MU0,4000*MU0)
 npar=1/mupar;nperp=1/muperp;bx=Br[soft]/bs;bz=Bz[soft]/bs;delta=npar-nperp
 nrr[soft]=nperp+delta*bx*bx;nrz[soft]=delta*bx*bz;nzz[soft]=nperp+delta*bz*bz
 return nrr,nrz,nzz

def main():
 mesh=load_tagged_meshio('/mnt/data/fig5_work/global_p2_mesh.msh');v=meshio.read('/mnt/data/fig5_work/global_p2_static_projected.vtu');cd=v.cell_data_dict
 Br=np.asarray(cd['B_r_T']['triangle'],float);Bz=np.asarray(cd['B_z_T']['triangle'],float);Bn=np.asarray(cd['B_norm_T']['triangle'],float);Hn=np.asarray(cd['H_norm_A_m']['triangle'],float);must=np.asarray(cd['mu_r']['triangle'],float)
 elem,emap,uedges=topology(mesh);ndof=mesh.n_nodes+len(uedges);area,cent,dr,dz=_tri_geometry(mesh.points_rz_m,mesh.triangles);rv=mesh.points_rz_m[mesh.triangles,0];ne=mesh.n_triangles
 nrr,nrz,nzz=incremental_tensor(mesh.tri_domains,Br,Bz,Bn,Hn,must)
 sigma=np.zeros(ne);sigma[np.isin(mesh.tri_domains,[6,23])]=1.12e7
 coil=np.isin(mesh.tri_domains,[17,18,19]);Acoil=float(area[coil].sum());N0=100
 ke=np.zeros((ne,6,6));me=np.zeros_like(ke);ce=np.zeros((ne,6))
 for bary,wq in _Q:
  lam=np.asarray(bary);N,gr,gz=shape_batch(lam,dr,dz);rq=np.maximum(rv@lam,1e-12);fac=2*np.pi*rq*area*wq;Brs=-gz;Bzs=gr+N[None,:]/rq[:,None]
  ke += fac[:,None,None]*(nrr[:,None,None]*(Brs[:,:,None]*Brs[:,None,:])+nrz[:,None,None]*(Brs[:,:,None]*Bzs[:,None,:]+Bzs[:,:,None]*Brs[:,None,:])+nzz[:,None,None]*(Bzs[:,:,None]*Bzs[:,None,:]))
  me += (fac*sigma)[:,None,None]*(N[:,None]*N[None,:])[None,:,:]
  ce[coil] += (fac[coil]*(N0/Acoil))[:,None]*N[None,:]
 rows=np.repeat(elem,6,1).ravel();cols=np.tile(elem,(1,6)).ravel();shape=(ndof,ndof)
 K=coo_matrix((ke.ravel(),(rows,cols)),shape=shape).tocsr();M=coo_matrix((me.ravel(),(rows,cols)),shape=shape).tocsr();c=np.bincount(elem.ravel(),weights=ce.ravel(),minlength=ndof)
 tags={1,2,3,4,5,83,84,85,86,87,88,89,94};fv=set();fe=[]
 for edge,tag in zip(mesh.line_cells,mesh.line_tags):
  if int(tag) in tags:
   k=tuple(sorted(map(int,edge)));fv.update(k)
   if k in emap:fe.append(mesh.n_nodes+emap[k])
 fv.update(map(int,np.flatnonzero(np.abs(mesh.points_rz_m[:,0])<1e-12)));fixed=np.array(sorted(fv.union(fe)),int);mask=np.ones(ndof,bool);mask[fixed]=False;free=np.flatnonzero(mask)
 Kf=K[free][:,free].tocsr();Mf=M[free][:,free].tocsr();cf=c[free]
 out=Path('/mnt/data/fig5_work/global_p2_native');out.mkdir(exist_ok=True)
 cache=Path('/mnt/data/fig5_work/global_p2_cache');cache.mkdir(exist_ok=True)
 from scipy.sparse import save_npz
 save_npz(cache/'K.npz',Kf,compressed=True);save_npz(cache/'M.npz',Mf,compressed=True)
 np.savez_compressed(cache/'arrays.npz',c=cf,free=free,elem=elem,sigma=sigma,ndof=np.array([ndof]),points=mesh.points_rz_m,triangles=mesh.triangles,tri_domains=mesh.tri_domains)
 if __import__('os').environ.get('BUILD_ONLY')=='1':
  print(json.dumps({'cache':str(cache),'n_p2_dofs':ndof,'n_free':len(free),'K_nnz':int(Kf.nnz),'M_nnz':int(Mf.nnz)}));return
 freqs=[50.,900.,2000.,5000.,8000.];Rdc=5.566962641411444;results=[];x0=None
 for f in freqs:
  om=2*np.pi*f;op=(Kf.astype(complex)+1j*om*Mf.astype(complex)).tocsc();t=time.time();ilu=spilu(op,drop_tol=1e-3,fill_factor=10,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0);Prc=LinearOperator(op.shape,matvec=ilu.solve,dtype=complex);hist=[];x,info=gmres(op,cf.astype(complex),M=Prc,x0=x0,rtol=1e-11,atol=0,restart=100,maxiter=30,callback=lambda z:hist.append(float(z)),callback_type='pr_norm');rr=float(np.linalg.norm(op@x-cf)/np.linalg.norm(cf));
  if info or rr>2e-11: raise RuntimeError((f,info,rr,len(hist)))
  x0=x.copy();A=np.zeros(ndof,complex);A[free]=x;lam=complex(c@A);Z=Rdc+1j*om*lam
  # centroid J and quadrature losses per 1 A peak.
  Ncent=np.array([-1/9,-1/9,-1/9,4/9,4/9,4/9]);Acent=A[elem]@Ncent;Jcent=np.zeros(ne,complex);cond=sigma>0;Jcent[cond]=-1j*om*sigma[cond]*Acent[cond]
  losses={}
  for dom in [6,23]:
   sel=mesh.tri_domains==dom;loss=0.0
   for bary,wq in _Q:
    N,_,_=shape_batch(np.asarray(bary),dr,dz);rq=np.maximum(rv@np.asarray(bary),1e-12);Aq=A[elem]@N;J=-1j*om*sigma*Aq;loss += float(np.sum(np.pi*rq[sel]*area[sel]*wq*np.abs(J[sel])**2/sigma[sel])) # 0.5*2pi r
   losses[dom]=loss
  np.savez_compressed(out/f'global_p2_{int(f)}Hz.npz',points=mesh.points_rz_m,triangles=mesh.triangles,tri_domains=mesh.tri_domains,elem_dofs=elem,A_phi_p2=A,Jphi_centroid_per_1A=Jcent,sigma_elem=sigma,mu_static=must,Zb=np.array([Z]),relative_residual=np.array([rr]),loss_domain6_W_per_1Apeak=np.array([losses[6]]),loss_domain23_W_per_1Apeak=np.array([losses[23]]))
  row={'freq_Hz':f,'Zb_real_ohm':Z.real,'Zb_imag_ohm':Z.imag,'Zb_abs_ohm':abs(Z),'relative_residual':rr,'iterations':len(hist),'factor_s':time.time()-t,'loss_domain6_W_per_1Apeak':losses[6],'loss_domain23_W_per_1Apeak':losses[23],'energy_error_ohm':(Z.real-Rdc)-2*(losses[6]+losses[23]),'peak_rss_MB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024};results.append(row);pd.DataFrame(results).to_csv(out/'global_p2_selected.csv',index=False);print(json.dumps(row),flush=True)
  del op,ilu,Prc,x,A;gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0)
 (out/'summary.json').write_text(json.dumps({'n_nodes':mesh.n_nodes,'n_triangles':mesh.n_triangles,'n_p2_dofs':ndof,'n_free':len(free),'K_nnz':int(Kf.nnz),'M_nnz':int(Mf.nnz),'frequencies':results},indent=2))
if __name__=='__main__':main()
