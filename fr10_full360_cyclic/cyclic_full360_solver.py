from __future__ import annotations
import json,math,os,time
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix,csc_matrix
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
import meshio
import base_p2_local_solver as b

HERE=Path(__file__).resolve().parent
R90=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])


def default_output_root():
    override=os.environ.get('FR10_FULL360_OUTPUT_ROOT')
    if override:return Path(override)
    return HERE.parent.parent/'runs'/f'{HERE.parent.name}_fr10_full360'


def load_cfg(): return json.loads((HERE/'configs/fr10_full360_cyclic.json').read_text())

def build_sector_model(cfg):
    m=b.build_model(cfg,perturb=False)
    # Undo x/y mirror constraints. Retain only physical fixed attachments at surround/spider outer radii.
    rr=lambda x:np.hypot(x[:,0],x[:,1]); fixed=set()
    for part,radmm in [(next(p for p in m['parts'] if p.name=='surround'),float(cfg['geometry']['surround_outer_radius_mm'])),
                       (next(p for p in m['parts'] if p.name=='spider'),31.5)]:
        ids=np.where(rr(part.p)>=radmm*1e-3-0.00018)[0]
        for j in ids:
            for q in range(3): fixed.add(3*(part.offset_node+int(j))+q)
    m['fixed_physical']=np.asarray(sorted(fixed),int)
    m['free_physical']=np.setdiff1d(np.arange(m['Nd']),m['fixed_physical'])
    return m


def _pair_part_seams(model,tol=2e-8):
    pairs=[]; report=[]
    for part in model['parts']:
        p=part.p; A=np.flatnonzero((np.abs(p[:,1])<tol)&(p[:,0]>tol)); B=np.flatnonzero((np.abs(p[:,0])<tol)&(p[:,1]>tol))
        if len(A)==0 or len(B)==0: continue
        target=(R90@p[A].T).T; tree=cKDTree(p[B]);d,j=tree.query(target)
        ok=d<5e-8
        if not np.all(ok): raise RuntimeError(f'struct seam match failed {part.name}: max {d.max()}')
        for aa,bb in zip(A,B[j]): pairs.append((part.offset_node+int(aa),part.offset_node+int(bb)))
        report.append((part.name,len(A),float(d.max())))
    return pairs,report


def structural_transform(model,phase):
    free=model['free_physical']; pos={int(d):i for i,d in enumerate(free)}; pairs,rep=_pair_part_seams(model)
    slave_nodes={bb:aa for aa,bb in pairs}
    slave_dofs={3*n+q for n in slave_nodes for q in range(3)}
    retained=[int(d) for d in free if int(d) not in slave_dofs]; col={d:i for i,d in enumerate(retained)}
    rows=[];cols=[];vals=[]
    for row,d in enumerate(free):
        d=int(d); n,q=divmod(d,3)
        if n not in slave_nodes:
            rows.append(row);cols.append(col[d]);vals.append(1+0j)
        else:
            ma=slave_nodes[n]
            # u_B = phase * R90 * u_A
            for qa,c in enumerate(R90[q]):
                if abs(c)<1e-15: continue
                md=3*ma+qa
                if md in col:
                    rows.append(row);cols.append(col[md]);vals.append(phase*c)
    T=coo_matrix((vals,(rows,cols)),shape=(len(free),len(retained)),dtype=complex).tocsc()
    return T,free,rep


def acoustic_transform(ac,phase,tol=2e-8):
    p=ac['p']; A=np.flatnonzero((np.abs(p[:,1])<tol)&(p[:,0]>tol)); B=np.flatnonzero((np.abs(p[:,0])<tol)&(p[:,1]>tol));
    target=(R90@p[A].T).T;tree=cKDTree(p[B]);d,j=tree.query(target)
    if len(A) and (not np.all(d<5e-8)): raise RuntimeError(f'acoustic seam match max={d.max()}')
    slave={int(bb):int(aa) for aa,bb in zip(A,B[j])}
    axis=np.flatnonzero((np.abs(p[:,0])<tol)&(np.abs(p[:,1])<tol))
    zero_axis=(abs(phase-1)>1e-10)
    excluded=set(slave)
    if zero_axis: excluded.update(int(x) for x in axis)
    retained=[i for i in range(len(p)) if i not in excluded];col={n:i for i,n in enumerate(retained)}
    rows=[];cols=[];vals=[]
    for n in range(len(p)):
        if n in slave:
            rows.append(n);cols.append(col[slave[n]]);vals.append(phase)
        elif zero_axis and n in set(axis):
            pass
        else:
            rows.append(n);cols.append(col[n]);vals.append(1+0j)
    T=coo_matrix((vals,(rows,cols)),shape=(len(p),len(retained)),dtype=complex).tocsc()
    return T,{'seam_pairs':len(slave),'axis_nodes':len(axis),'reduced_nodes':len(retained),'seam_max_mismatch_m':float(d.max() if len(d) else 0)}


def reduced_system(cfg,model,front,rear,G,freq,sscale,kclass):
    phase=np.exp(1j*kclass*math.pi/2);w=2*math.pi*freq;rho=float(cfg['air']['rho_kg_m3']);c=float(cfg['air']['c_m_s']);kk=w/c
    Tu,sfree,srep=structural_transform(model,phase);Tp,prep=acoustic_transform(front,phase)
    # rear uses identical undeformed topology and same cyclic transform dimensions
    Tpr,rprep=acoustic_transform(rear,phase)
    mscale=float(cfg['targets']['Mms_kg'])/model['raw_mass_full_kg']
    H0=(b.structural_blocks(model,cfg,sscale)-w*w*(model['M']*mscale))[sfree][:,sfree].tocsc()
    H=(Tu.conj().T@(H0@Tu)).tocsc()
    def Aac(ac,T):
        A=(ac['K'].astype(complex)-kk*kk*ac['M'].astype(complex)+(1/ac['outer_radius_m']+1j*kk)*ac['Babs'].astype(complex)).tocsc()
        return (T.conj().T@(A@T)).tocsc()
    Af=Aac(front,Tp);Ar=Aac(rear,Tpr)
    G0=G[sfree,:].tocsr();Bf=(Tu.conj().T@(G0@Tp)).tocsc();Br=(Tu.conj().T@(G0@Tpr)).tocsc()
    C_f=(Tp.conj().T@(G0.T@Tu)).tocsc();C_r=(Tpr.conj().T@(G0.T@Tu)).tocsc()
    f0=model['fL'][sfree].astype(complex);f=(Tu.conj().T@f0)
    return {'phase':phase,'Tu':Tu,'Tp':Tp,'Tpr':Tpr,'sfree':sfree,'H':H,'Af':Af,'Ar':Ar,'Bf':Bf,'Br':Br,'Cf':C_f,'Cr':C_r,'f':np.asarray(f).ravel(),'struct_report':srep,'ac_report':prep}


def _active_trace_indices(B, C):
    """Return the reduced acoustic nodes touched by either coupling direction."""
    return np.unique(np.r_[B.nonzero()[1], C.nonzero()[0]]).astype(int)


def _relative_sparse_difference(A, B):
    if A.shape != B.shape:
        return math.inf
    delta = (A - B).tocoo()
    numerator = float(np.linalg.norm(delta.data)) if delta.nnz else 0.0
    scale = float(np.linalg.norm(A.data)) if A.nnz else 0.0
    return numerator / max(scale, 1e-300)


def _interface_green(L, volume_size, trace_indices, block_size):
    """Form E^H A^-1 E without materializing a volume-sized dense identity."""
    ntrace = len(trace_indices)
    result = np.empty((ntrace, ntrace), dtype=complex)
    for j0 in range(0, ntrace, block_size):
        j1 = min(j0 + block_size, ntrace)
        rhs = np.zeros((volume_size, j1 - j0), dtype=complex)
        rhs[trace_indices[j0:j1], np.arange(j1 - j0)] = 1.0
        result[:, j0:j1] = L.solve(rhs)[trace_indices, :]
    return result


def _structural_trace_compliance(L, C, B, block_size):
    """Form C H^-1 B using block sparse right-hand sides."""
    result = np.empty((C.shape[0], B.shape[1]), dtype=complex)
    for j0 in range(0, B.shape[1], block_size):
        j1 = min(j0 + block_size, B.shape[1])
        response = L.solve(B[:, j0:j1].toarray())
        result[:, j0:j1] = C @ response
    return result


def _coupled_backward_error(H, Af, Ar, Bf, Br, Cf, Cr, rw2, u, pf, pr, f):
    rs = H @ u + Bf @ pf - Br @ pr - f
    rf = Af @ pf + rw2 * (Cf @ u)
    rr = Ar @ pr - rw2 * (Cr @ u)
    raw_residual_2 = math.sqrt(
        np.linalg.norm(rs) ** 2
        + np.linalg.norm(rf) ** 2
        + np.linalg.norm(rr) ** 2
    )
    structural_scale=(np.linalg.norm(H@u)+np.linalg.norm(Bf@pf)+np.linalg.norm(Br@pr)+np.linalg.norm(f))
    front_scale=np.linalg.norm(Af@pf)+rw2*np.linalg.norm(Cf@u)
    rear_scale=np.linalg.norm(Ar@pr)+rw2*np.linalg.norm(Cr@u)
    block_residuals={
        'structural':float(np.linalg.norm(rs)/max(structural_scale,1e-300)),
        'front_acoustic':float(np.linalg.norm(rf)/max(front_scale,1e-300)),
        'rear_acoustic':float(np.linalg.norm(rr)/max(rear_scale,1e-300)),
    }
    h_rows = np.asarray(np.abs(H).sum(axis=1)).ravel()
    bf_rows = np.asarray(np.abs(Bf).sum(axis=1)).ravel()
    br_rows = np.asarray(np.abs(Br).sum(axis=1)).ravel()
    af_rows = np.asarray(np.abs(Af).sum(axis=1)).ravel()
    ar_rows = np.asarray(np.abs(Ar).sum(axis=1)).ravel()
    cf_rows = np.asarray(np.abs(Cf).sum(axis=1)).ravel()
    cr_rows = np.asarray(np.abs(Cr).sum(axis=1)).ravel()
    operator_inf = float(
        max(
            np.max(h_rows + bf_rows + br_rows),
            np.max(af_rows + rw2 * cf_rows),
            np.max(ar_rows + rw2 * cr_rows),
        )
    )
    solution_inf = max(
        float(np.max(np.abs(u))),
        float(np.max(np.abs(pf))),
        float(np.max(np.abs(pr))),
    )
    rhs_inf = float(np.max(np.abs(f)))
    max_residual = float(
        max(np.max(np.abs(rs)), np.max(np.abs(rf)), np.max(np.abs(rr)))
    )
    return {
        'relative_residual_2':float(max(block_residuals.values())),
        'block_relative_residual_2':block_residuals,
        'raw_residual_over_structural_rhs_2':float(raw_residual_2/max(np.linalg.norm(f),1e-300)),
        'normwise_backward_error_inf': float(
            max_residual / max(operator_inf * solution_inf + rhs_inf, 1e-300)
        ),
    }


def solve_phase(cfg,model,front,rear,G,freq,sscale,kclass,force_scale=1.0):
    t0=time.time();r=reduced_system(cfg,model,front,rear,G,freq,sscale,kclass);w=2*math.pi*freq;rho=float(cfg['air']['rho_kg_m3']);rw2=rho*w*w
    H,Af,Ar=r['H'],r['Af'],r['Ar'];Bf,Br,Cf,Cr=r['Bf'],r['Br'],r['Cf'],r['Cr'];f=r['f']*force_scale
    If=_active_trace_indices(Bf,Cf);Ir=_active_trace_indices(Br,Cr)
    Bfi=Bf[:,If].tocsc();Bri=Br[:,Ir].tocsc();Cfi=Cf[If,:].tocsc();Cri=Cr[Ir,:].tocsc()
    same_trace=bool(np.array_equal(If,Ir))
    bdiff=_relative_sparse_difference(Bfi,Bri) if same_trace else math.inf
    cdiff=_relative_sparse_difference(Cfi,Cri) if same_trace else math.inf
    match_tol=float(cfg['numerics'].get('trace_operator_match_tolerance',1e-12))
    if not same_trace or bdiff>match_tol or cdiff>match_tol:
        raise RuntimeError(
            'front/rear periodic ASB trace topology is not compatible: '
            f'same_indices={same_trace}, Bdiff={bdiff:.3e}, Cdiff={cdiff:.3e}'
        )
    # The two acoustic domains use the same periodic trace coordinates and the same
    # local ASB map.  Verify that contract before sharing the trace displacement x=C*u.
    B=(0.5*(Bfi+Bri)).tocsc();C=(0.5*(Cfi+Cri)).tocsc();I=If;ni=len(I)
    lu_start=time.time();Ls=splu(H,permc_spec='MMD_AT_PLUS_A');struct_lu_s=time.time()-lu_start
    front_start=time.time();Lf=splu(Af,permc_spec='COLAMD');front_lu_s=time.time()-front_start
    rear_start=time.time();Lr=splu(Ar,permc_spec='COLAMD');rear_lu_s=time.time()-rear_start
    print(f'[cyclic] LU structural={struct_lu_s:.2f}s front={front_lu_s:.2f}s rear={rear_lu_s:.2f}s',flush=True)
    block_size=int(cfg['numerics'].get('condensation_block_rhs',32))
    cond_start=time.time()
    S=_structural_trace_compliance(Ls,C,B,block_size)
    Hf=_interface_green(Lf,Af.shape[0],I,block_size)
    Hr=_interface_green(Lr,Ar.shape[0],I,block_size)
    condensation_s=time.time()-cond_start
    u0=Ls.solve(f);x0=np.asarray(C@u0).ravel()
    Red=np.eye(ni,dtype=complex)-rw2*(S@(Hf+Hr))
    dense_start=time.time();x=np.linalg.solve(Red,x0);dense_s=time.time()-dense_start
    pfI=-rw2*(Hf@x);prI=rw2*(Hr@x)
    bf=np.zeros(Af.shape[0],complex);br=np.zeros(Ar.shape[0],complex);bf[I]=-rw2*x;br[I]=rw2*x
    pfr=Lf.solve(bf);prr=Lr.solve(br)
    structural_rhs=f-Bf@pfr+Br@prr
    condensed_structural_rhs=f+rw2*(B@(Hf@x+Hr@x))
    ur=Ls.solve(structural_rhs)
    # Recover physical quarter-sector fields.
    us=np.zeros(model['Nd'],complex);us[r['sfree']]=r['Tu']@ur;pf=r['Tp']@pfr;pr=r['Tpr']@prr
    errors=_coupled_backward_error(H,Af,Ar,Bf,Br,Cf,Cr,rw2,ur,pfr,prr,f)
    trace_consistency=float(np.linalg.norm(x-C@ur)/max(np.linalg.norm(x)+np.linalg.norm(C@ur),1e-300))
    dense_trace_residual=float(np.linalg.norm(Red@x-x0)/max(np.linalg.norm(x0),1e-300))
    front_lu_residual=float(np.linalg.norm(Af@pfr-bf)/max(np.linalg.norm(bf),1e-300))
    rear_lu_residual=float(np.linalg.norm(Ar@prr-br)/max(np.linalg.norm(br),1e-300))
    structural_rhs_consistency=float(np.linalg.norm(structural_rhs-condensed_structural_rhs)/max(np.linalg.norm(structural_rhs)+np.linalg.norm(condensed_structural_rhs),1e-300))
    front_trace_green_consistency=float(np.linalg.norm(pfr[I]+rw2*(Hf@x))/max(np.linalg.norm(pfr[I])+np.linalg.norm(rw2*(Hf@x)),1e-300))
    rear_trace_green_consistency=float(np.linalg.norm(prr[I]-rw2*(Hr@x))/max(np.linalg.norm(prr[I])+np.linalg.norm(rw2*(Hr@x)),1e-300))
    return us,np.asarray(pf).ravel(),np.asarray(pr).ravel(),{
        'kclass':kclass,
        'phase':[r['phase'].real,r['phase'].imag],
        'solver':'exact periodic local-ASB trace condensation (three sparse volume LU + dense trace LU)',
        'struct_reduced_dofs':H.shape[0],
        'acoustic_reduced_dofs_front':Af.shape[0],
        'acoustic_reduced_dofs_rear':Ar.shape[0],
        'trace_dofs':ni,
        'trace_indices_front_rear_identical':same_trace,
        'Bf_Br_relative_difference':bdiff,
        'Cf_Cr_relative_difference':cdiff,
        'structural_LU_s':struct_lu_s,
        'front_acoustic_LU_s':front_lu_s,
        'rear_acoustic_LU_s':rear_lu_s,
        'trace_condensation_s':condensation_s,
        'dense_trace_solve_s':dense_s,
        'reduced_condition_number_2':float(np.linalg.cond(Red)),
        'dense_trace_relative_residual_2':dense_trace_residual,
        'trace_displacement_consistency_relative_2':trace_consistency,
        'front_acoustic_LU_relative_residual_2':front_lu_residual,
        'rear_acoustic_LU_relative_residual_2':rear_lu_residual,
        'structural_rhs_condensation_consistency_relative_2':structural_rhs_consistency,
        'front_trace_green_consistency_relative_2':front_trace_green_consistency,
        'rear_trace_green_consistency_relative_2':rear_trace_green_consistency,
        **errors,
        'elapsed_s':time.time()-t0,
    }


def reconstruct_full_points_cells(points,tets,field,kclass,vector=False):
    # Four explicit sectors, no seam merging in output; geometric duplication on sector planes is harmless for VTU visualization.
    Ps=[];Ts=[];Fs=[];off=0;phase=np.exp(1j*kclass*math.pi/2)
    for q in range(4):
        ang=q*math.pi/2;ca,sa=math.cos(ang),math.sin(ang);R=np.array([[ca,-sa,0],[sa,ca,0],[0,0,1.]])
        Ps.append((R@points.T).T);Ts.append(tets+off);off+=len(points)
        if vector: Fs.append((phase**q)*(field@R.T))  # row-vector field rotates by R
        else: Fs.append((phase**q)*field)
    return np.vstack(Ps),np.vstack(Ts),np.concatenate(Fs,axis=0)


def full360_metrics(model,u_sector,kclass=0):
    # Fourier energy of axial displacement on original P1 radiating surface vertices after explicit reconstruction.
    rings={}
    U=u_sector.reshape(-1,3)
    for part in model['parts']:
        if part.name not in ('surround','cone','dustcap'):continue
        faces,cent,area,nout=b.boundary_triangle_data(part.p4,part.t4);mask=nout[:,2]>0.15
        if part.name=='cone':mask&=np.hypot(cent[:,0],cent[:,1])>=0.0175*0.985
        vids=np.unique(faces[mask].ravel())
        for vid in vids:
            x=part.p4[int(vid)];r=float(np.hypot(x[0],x[1]));z=float(x[2]);th=float(math.atan2(x[1],x[0])%(2*math.pi));val=U[part.offset_node+int(vid),2]
            for q in range(4):
                thq=(th+q*math.pi/2)%(2*math.pi);vq=(np.exp(1j*kclass*math.pi/2)**q)*val;rings.setdefault((round(r,7),round(z,7)),[]).append((thq,vq))
    E=np.zeros(17);tot=0.;nr=0
    for vals in rings.values():
        # de-duplicate angle points at seams
        d={round(t,10):v for t,v in vals}; vals=sorted(d.items());
        if len(vals)<16:continue
        th=np.array([x[0] for x in vals]);v=np.array([x[1] for x in vals]);
        # direct nonuniform DFT projection; ring nodes are uniform for structured structure mesh.
        for m in range(17):
            a=np.mean(v*np.exp(-1j*m*th));E[m]+=abs(a)**2
        tot+=np.mean(np.abs(v)**2);nr+=1
    frac=E/max(E.sum(),1e-300);expected=np.asarray([m%4==int(kclass)%4 for m in range(len(E))])
    class_fraction=float(np.sum(frac[expected]))
    return {'ring_count':nr,'m_energy_fraction_0_to_16':frac.tolist(),'dominant_m':int(np.argmax(E)),'expected_mod4_class':int(kclass)%4,'expected_mod4_class_energy_fraction':class_fraction,'mod4_leakage_fraction':float(1-class_fraction),'nonaxisymmetric_fraction_m_ge_1':float(1-frac[0])}


def _full360_structure_arrays(model,u_sector,kclass):
    points=[];cells=[];fields=[];point_parts=[];point_sectors=[];cell_parts=[];cell_sectors=[];point_offset=0
    U=u_sector.reshape(-1,3)
    for part_id,part in enumerate(model['parts']):
        local=U[part.offset_node:part.offset_node+len(part.p)]
        P,T,F=reconstruct_full_points_cells(part.p,part.t10,local,kclass,True)
        points.append(P);cells.append(T+point_offset);fields.append(F)
        point_parts.append(np.full(len(P),part_id,np.int32));point_sectors.append(np.repeat(np.arange(4,dtype=np.int32),len(part.p)))
        cell_parts.append(np.full(len(T),part_id,np.int32));cell_sectors.append(np.repeat(np.arange(4,dtype=np.int32),len(part.t10)))
        point_offset+=len(P)
    return np.vstack(points),np.vstack(cells),np.vstack(fields),np.concatenate(point_parts),np.concatenate(point_sectors),np.concatenate(cell_parts),np.concatenate(cell_sectors)


def write_full360_fields(out,model,front,rear,u,pf,pr,kclass,tag,cfg):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);written={}
    P,T,U,point_parts,point_sectors,cell_parts,cell_sectors=_full360_structure_arrays(model,u,kclass)
    structure=out/f'structure_full360_{tag}_k{kclass}_P2.vtu'
    meshio.write_points_cells(structure,P,[('tetra10',T)],point_data={
        'u_real_m':U.real,'u_imag_m':U.imag,'u_abs_m':np.linalg.norm(U,axis=1),
        'u_z_phase_deg':np.degrees(np.angle(U[:,2])),
        'part_id':point_parts,'sector_id':point_sectors},cell_data={
        'part_id':[cell_parts],'sector_id':[cell_sectors]})
    written['structure_vtu']=str(structure)
    for name,ac,field in (('front',front,pf),('rear',rear,pr)):
        Pa,Ta,Fa=reconstruct_full_points_cells(ac['p'],ac['t'],field,kclass,False)
        path=out/f'acoustic_{name}_full360_{tag}_k{kclass}.vtu'
        meshio.write_points_cells(path,Pa,[('tetra',Ta)],point_data={
            'p_real_Pa':Fa.real,'p_imag_Pa':Fa.imag,'p_abs_Pa':np.abs(Fa),
            'p_phase_deg':np.degrees(np.angle(Fa)),
            'SPL_rms_dB':20*np.log10(np.maximum(np.abs(Fa)/math.sqrt(2)/float(cfg['air']['p_ref_Pa']),1e-300)),
            'sector_id':np.repeat(np.arange(4,dtype=np.int32),len(ac['p']))},cell_data={
            'sector_id':[np.repeat(np.arange(4,dtype=np.int32),len(ac['t']))]})
        written[f'acoustic_{name}_vtu']=str(path)
    return written


def _equal_3d_axes(ax,points):
    lo=points.min(axis=0);hi=points.max(axis=0);center=0.5*(lo+hi);radius=0.5*float(np.max(hi-lo))
    ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)


def _sample_indices(count,maximum):
    if count<=maximum:return np.arange(count)
    return np.linspace(0,count-1,maximum,dtype=int)


def write_full360_visualizations(out,model,front,rear,u,pf,pr,kclass,metrics,tag,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(parents=True,exist_ok=True);written={}
    maximum=int(cfg.get('visualization',{}).get('maximum_scatter_points',18000))
    P,_,U,_,_,_,_=_full360_structure_arrays(model,u,kclass);umag=np.linalg.norm(U,axis=1);scale=0.0
    if np.max(umag)>0:scale=float(cfg.get('visualization',{}).get('deformation_span_fraction',0.08))*float(np.ptp(P,axis=0).max())/float(np.max(umag))
    Pd=P+scale*U.real;ii=_sample_indices(len(Pd),maximum)
    fig=plt.figure(figsize=(9,7));ax=fig.add_subplot(111,projection='3d')
    sc=ax.scatter(Pd[ii,0]*1e3,Pd[ii,1]*1e3,Pd[ii,2]*1e3,c=umag[ii]*1e6,s=1.5,cmap='viridis')
    _equal_3d_axes(ax,Pd[ii]*1e3);ax.set_xlabel('x / mm');ax.set_ylabel('y / mm');ax.set_zlabel('z / mm')
    ax.set_title(f'FR10 full-360 structure |u|, {tag}, k={kclass}, deformation x{scale:.3g}')
    fig.colorbar(sc,ax=ax,pad=0.08,label='|u| / micrometre');fig.tight_layout()
    path=out/f'structure_full360_{tag}_k{kclass}.png';fig.savefig(path,dpi=180);plt.close(fig);written['structure_png']=str(path)
    uz_real=U[ii,2].real*1e6;limit=max(float(np.max(np.abs(uz_real))),1e-30)
    fig=plt.figure(figsize=(9,7));ax=fig.add_subplot(111,projection='3d')
    sc=ax.scatter(Pd[ii,0]*1e3,Pd[ii,1]*1e3,Pd[ii,2]*1e3,c=uz_real,s=1.5,cmap='coolwarm',vmin=-limit,vmax=limit)
    _equal_3d_axes(ax,Pd[ii]*1e3);ax.set_xlabel('x / mm');ax.set_ylabel('y / mm');ax.set_zlabel('z / mm')
    ax.set_title(f'FR10 full-360 Re(u_z), {tag}, k={kclass}, deformation x{scale:.3g}')
    fig.colorbar(sc,ax=ax,pad=0.08,label='Re(u_z) / micrometre');fig.tight_layout()
    path=out/f'structure_uz_real_full360_{tag}_k{kclass}.png';fig.savefig(path,dpi=180);plt.close(fig);written['structure_uz_real_png']=str(path)
    acoustic_points=[];acoustic_values=[]
    for ac,field in ((front,pf),(rear,pr)):
        Pa,_,Fa=reconstruct_full_points_cells(ac['p'],ac['t'],field,kclass,False)
        outer=np.linalg.norm(Pa,axis=1)>0.98*float(ac['outer_radius_m'])
        acoustic_points.append(Pa[outer]);acoustic_values.append(Fa[outer])
    Pa=np.vstack(acoustic_points);Fa=np.concatenate(acoustic_values);ii=_sample_indices(len(Pa),maximum)
    spl=20*np.log10(np.maximum(np.abs(Fa[ii])/math.sqrt(2)/float(cfg['air']['p_ref_Pa']),1e-300))
    fig=plt.figure(figsize=(9,7));ax=fig.add_subplot(111,projection='3d')
    sc=ax.scatter(Pa[ii,0]*1e3,Pa[ii,1]*1e3,Pa[ii,2]*1e3,c=spl,s=2,cmap='magma')
    _equal_3d_axes(ax,Pa[ii]*1e3);ax.set_xlabel('x / mm');ax.set_ylabel('y / mm');ax.set_zlabel('z / mm')
    ax.set_title(f'FR10 full-360 outer acoustic field, {tag}, k={kclass}')
    fig.colorbar(sc,ax=ax,pad=0.08,label='SPL / dB re 20 uPa RMS');fig.tight_layout()
    path=out/f'acoustic_outer_full360_{tag}_k{kclass}.png';fig.savefig(path,dpi=180);plt.close(fig);written['acoustic_png']=str(path)
    energy=np.asarray(metrics['m_energy_fraction_0_to_16'],float)
    fig,ax=plt.subplots(figsize=(9,4.5));ax.bar(np.arange(len(energy)),energy)
    ax.set_yscale('log');ax.set_ylim(max(float(np.max(energy))*1e-12,1e-16),1.1);ax.set_xlabel('circumferential order m');ax.set_ylabel('energy fraction')
    ax.set_title(f'Circumferential displacement energy, {tag}, k={kclass}');ax.grid(True,which='both',axis='y',alpha=0.25);fig.tight_layout()
    path=out/f'circumferential_m_energy_{tag}_k{kclass}.png';fig.savefig(path,dpi=180);plt.close(fig);written['m_energy_png']=str(path)
    csv_path=out/f'circumferential_m_energy_{tag}_k{kclass}.csv';orders=np.arange(len(energy),dtype=int)
    np.savetxt(csv_path,np.column_stack((orders,energy,orders%4==int(kclass)%4)),delimiter=',',header='m,energy_fraction,expected_mod4_class',comments='',fmt=['%d','%.17g','%d'])
    written['m_energy_csv']=str(csv_path)
    return written


def _quarter_reference(freq):
    path=HERE.parent/'benchmarks'/'fr10_3d_quarter'/f'quarter_p2_local_asb_{freq:g}Hz.json'
    if not path.exists():return None
    return json.loads(path.read_text())


def _quarter_comparison(freq,Zmot,coil_displacement,spl):
    ref=_quarter_reference(freq)
    if ref is None:return None
    zr=complex(*ref['electrical']['Z_motional_ohm']);xr=complex(*ref['mechanical']['coil_displacement_peak_m']);sr=float(ref['acoustic']['SPL_1m_from_outer_axis_dB'])
    return {
        'reference_file':str((HERE.parent/'benchmarks'/'fr10_3d_quarter'/f'quarter_p2_local_asb_{freq:g}Hz.json').relative_to(HERE.parent)),
        'Zmot_complex_relative_error':float(abs(Zmot-zr)/max(abs(zr),1e-300)),
        'coil_displacement_complex_relative_error':float(abs(coil_displacement-xr)/max(abs(xr),1e-300)),
        'SPL_difference_dB':float(spl-sr),
    }


def run_baseline(freqs=(500.,1000.,2000.),sscale=None,outroot=None):
    cfg=load_cfg();sscale=float(sscale or cfg['calibration']['p2_local_asb_suspension_scale']);outroot=Path(outroot or default_output_root()/'baseline');outroot.mkdir(parents=True,exist_ok=True)
    t=time.time();model=build_sector_model(cfg);front,rear=b.build_acoustic_domains(cfg);G,grep=b.build_local_G(model,front,cfg);print('build',time.time()-t,'sector Nd',model['Nd'],'trace raw',len(np.unique(G.nonzero()[1])),flush=True)
    allsum=[]
    for f in freqs:
        u,pf,pr,meta=solve_phase(cfg,model,front,rear,G,float(f),sscale,0)
        w=2*math.pi*f;Bl=float(cfg['electrical']['Bl_Tm']);x=complex(model['gcoil']@u);Zmot=1j*w*Bl*x;Zb=complex(cfg['electrical']['Rdc_ohm'],w*cfg['electrical']['Le_H']);Z=Zb+Zmot;I=1/Z;uv=u*I;pfv=pf*I;prv=pr*I
        R=front['outer_radius_m'];ia=int(np.argmin(np.linalg.norm(front['p']-np.array([0,0,R]),axis=1)));iar=int(np.argmin(np.linalg.norm(rear['p']-np.array([0,0,-R]),axis=1)));pR=complex(pfv[ia]);pRr=complex(prv[iar]);k=w/cfg['air']['c_m_s'];p1=pR*R*np.exp(-1j*k*(1-R));p1r=pRr*R*np.exp(-1j*k*(1-R));spl=20*math.log10(max(abs(p1)/math.sqrt(2)/cfg['air']['p_ref_Pa'],1e-300));splr=20*math.log10(max(abs(p1r)/math.sqrt(2)/cfg['air']['p_ref_Pa'],1e-300))
        fm=full360_metrics(model,uv,0);coh=b.radiating_surface_metrics(model,uv,G,cfg)['volume_displacement_coherence'];coil=x*I
        sm={'frequency_Hz':f,'full360_method':'four-sector cyclic Bloch FEM, k=0 electrical excitation; no mirror BC','electrical':{'Z_motional_ohm':[Zmot.real,Zmot.imag],'Z_total_ohm':[Z.real,Z.imag],'current_A_peak':[I.real,I.imag]},'mechanical':{'coil_displacement_peak_m':[coil.real,coil.imag],'max_sector_nodal_displacement_peak_m':float(np.max(np.linalg.norm(uv.reshape(-1,3),axis=1))),'volume_displacement_coherence':coh,**fm},'acoustic':{'front_axis_pressure_at_outer_radius_Pa_peak':[pR.real,pR.imag],'rear_axis_pressure_at_outer_radius_Pa_peak':[pRr.real,pRr.imag],'front_SPL_1m_axis_dB':spl,'rear_SPL_1m_axis_dB':splr,'max_front_pressure_Pa_peak':float(np.max(np.abs(pfv))),'max_rear_pressure_Pa_peak':float(np.max(np.abs(prv))),'rms_front_nodal_pressure_Pa_peak':float(np.sqrt(np.mean(np.abs(pfv)**2))),'rms_rear_nodal_pressure_Pa_peak':float(np.sqrt(np.mean(np.abs(prv)**2)))},'quarter_reference_comparison':_quarter_comparison(f,Zmot,coil,spl),'cyclic_solver':meta}
        od=outroot/f'{f:g}Hz';od.mkdir(exist_ok=True)
        field_files=write_full360_fields(od,model,front,rear,uv,pfv,prv,0,f'{f:g}Hz',cfg)
        plot_files=write_full360_visualizations(od,model,front,rear,uv,pfv,prv,0,fm,f'{f:g}Hz',cfg)
        sm['outputs']={**field_files,**plot_files};(od/'summary.json').write_text(json.dumps(sm,indent=2))
        np.savez_compressed(od/f'sector_solution_{f:g}Hz.npz',struct_points=model['P'],u_real=uv.real,u_imag=uv.imag,front_points=front['p'],p_front_real=pfv.real,p_front_imag=pfv.imag)
        allsum.append(sm);print(json.dumps(sm,indent=2),flush=True)
    compared=[x['quarter_reference_comparison'] for x in allsum if x['quarter_reference_comparison'] is not None]
    resonance=[x for x in allsum if abs(x['frequency_Hz']-90.0)<1e-9]
    backward_tol=float(cfg['numerics']['backward_error_tolerance']);residual_tol=float(cfg['numerics']['relative_residual_tolerance'])
    acceptance={'all_backward_errors_below_tolerance':all(x['cyclic_solver']['normwise_backward_error_inf']<backward_tol for x in allsum),'all_block_relative_residuals_below_tolerance':all(x['cyclic_solver']['relative_residual_2']<residual_tol for x in allsum),'backward_error_tolerance':backward_tol,'block_relative_residual_tolerance':residual_tol,'quarter_SPL_difference_below_0p5_dB':all(abs(x['SPL_difference_dB'])<0.5 for x in compared),'quarter_Zmot_relative_error_below_5_percent':all(x['Zmot_complex_relative_error']<0.05 for x in compared),'full360_90Hz_motional_reactance_below_0p05_ohm':bool(resonance and abs(resonance[0]['electrical']['Z_motional_ohm'][1])<0.05)}
    acceptance['status']='pass' if all(value for key,value in acceptance.items() if key not in ('backward_error_tolerance','block_relative_residual_tolerance')) else 'fail'
    (outroot/'run_summary.json').write_text(json.dumps({'suspension_scale':sscale,'local_asb':grep,'acceptance':acceptance,'frequencies':allsum},indent=2,default=float))
    return allsum


def run_phase_diagnostics(freq=2000.,phase_classes=(1,2,3),sscale=None,outroot=None):
    cfg=load_cfg();sscale=float(sscale or cfg['calibration']['p2_local_asb_suspension_scale']);outroot=Path(outroot or default_output_root()/'phase_diagnostics');outroot.mkdir(parents=True,exist_ok=True)
    model=build_sector_model(cfg);front,rear=b.build_acoustic_domains(cfg);G,grep=b.build_local_G(model,front,cfg);rows=[]
    for kclass in phase_classes:
        u,pf,pr,meta=solve_phase(cfg,model,front,rear,G,float(freq),sscale,int(kclass));fm=full360_metrics(model,u,int(kclass));od=outroot/f'{freq:g}Hz_k{kclass}';od.mkdir(exist_ok=True)
        fields=write_full360_fields(od,model,front,rear,u,pf,pr,int(kclass),f'{freq:g}Hz_diagnostic',cfg)
        plots=write_full360_visualizations(od,model,front,rear,u,pf,pr,int(kclass),fm,f'{freq:g}Hz_diagnostic',cfg)
        row={'frequency_Hz':float(freq),'phase_class':int(kclass),'excitation_identity':'unit sector generalized-force projection for Bloch-space diagnostic; not a physical symmetric electrical drive','mechanical':fm,'solver':meta,'outputs':{**fields,**plots}}
        row['non_mirror_space_active']=bool(fm['dominant_m']!=0 and fm['nonaxisymmetric_fraction_m_ge_1']>0.95 and fm['expected_mod4_class_energy_fraction']>0.90)
        (od/'summary.json').write_text(json.dumps(row,indent=2));rows.append(row)
    status='pass' if all(r['non_mirror_space_active'] and r['solver']['normwise_backward_error_inf']<1e-10 for r in rows) else 'fail'
    (outroot/'diagnostic_summary.json').write_text(json.dumps({'status':status,'local_asb':grep,'phase_classes':rows},indent=2))
    return rows

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--freq',type=float,nargs='*',default=[90,500,1000,2000]);ap.add_argument('--scale',type=float,default=None);ap.add_argument('--out',type=Path,default=None);ap.add_argument('--diagnostic-phases',type=int,nargs='*',default=None);a=ap.parse_args()
    if a.diagnostic_phases is None:run_baseline(a.freq,a.scale,a.out)
    else:run_phase_diagnostics(a.freq[0],a.diagnostic_phases,a.scale,a.out)
