from __future__ import annotations
import json, math, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, block_diag, bmat
from scipy.sparse.linalg import splu, spsolve, gmres, LinearOperator
from scipy.spatial import cKDTree

try:
    import meshio
except ImportError:
    meshio = None

HERE = Path(__file__).resolve().parent

# ----------------------------- P2 tetrahedra -----------------------------
P2_EDGES = ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))


def read_tets_mm(path: Path):
    if meshio is None:
        raise RuntimeError('meshio required')
    m=meshio.read(path)
    ts=[c.data for c in m.cells if c.type=='tetra']
    if not ts: raise RuntimeError(f'no tetra in {path}')
    return np.asarray(m.points[:,:3],float)*1e-3, np.vstack(ts).astype(np.int32)


def boundary_faces(tets):
    faces={}
    loc=((0,1,2),(0,1,3),(0,2,3),(1,2,3))
    for it,t in enumerate(tets):
        for q in loc:
            f=tuple(int(t[x]) for x in q); key=tuple(sorted(f))
            if key in faces: faces[key]=None
            else: faces[key]=(f,it)
    out=[v for v in faces.values() if v is not None]
    return np.asarray([x[0] for x in out],np.int32),np.asarray([x[1] for x in out],np.int32)


def boundary_triangle_data(p,t):
    faces,owners=boundary_faces(t)
    cent=p[faces].mean(axis=1); area=np.empty(len(faces)); nout=np.empty((len(faces),3))
    for i,(f,owner) in enumerate(zip(faces,owners)):
        a,b,c=p[f]; cr=np.cross(b-a,c-a); norm=np.linalg.norm(cr); area[i]=0.5*norm
        n=cr/max(norm,1e-300); tc=p[t[owner]].mean(axis=0); fc=(a+b+c)/3
        if np.dot(n,tc-fc)>0: n=-n
        nout[i]=n
    return faces,cent,area,nout


def elevate_p2(p,t):
    edge_to_node={}; pts=[x.copy() for x in p]; t10=np.empty((len(t),10),np.int32)
    t10[:,:4]=t
    for ie,tet in enumerate(t):
        for j,(a,b) in enumerate(P2_EDGES):
            va,vb=int(tet[a]),int(tet[b]); key=(va,vb) if va<vb else (vb,va)
            n=edge_to_node.get(key)
            if n is None:
                n=len(pts); edge_to_node[key]=n; pts.append(0.5*(p[va]+p[vb]))
            t10[ie,4+j]=n
    return np.asarray(pts,float),t10,edge_to_node


def iso_D(E,nu):
    lam=E*nu/((1+nu)*(1-2*nu)); mu=E/(2*(1+nu))
    return np.array([[lam+2*mu,lam,lam,0,0,0],[lam,lam+2*mu,lam,0,0,0],
                     [lam,lam,lam+2*mu,0,0,0],[0,0,0,mu,0,0],
                     [0,0,0,0,mu,0],[0,0,0,0,0,mu]],float)


def _poly_integral_bary(exps):
    # integral / V over tetra of lambda_1^a ... lambda_4^d
    s=sum(exps); num=6
    for a in exps: num*=math.factorial(a)
    return num/math.factorial(s+3)


def _p2_mass_coeff():
    # polynomials as dict exponent tuple -> coefficient
    pol=[]
    for i in range(4):
        e2=[0]*4; e2[i]=2; e1=[0]*4; e1[i]=1
        pol.append({tuple(e2):2.0,tuple(e1):-1.0})
    for i,j in P2_EDGES:
        e=[0]*4;e[i]=1;e[j]=1;pol.append({tuple(e):4.0})
    C=np.zeros((10,10))
    for i in range(10):
      for j in range(10):
        val=0.0
        for ei,ci in pol[i].items():
          for ej,cj in pol[j].items():
            ex=tuple(a+b for a,b in zip(ei,ej)); val+=ci*cj*_poly_integral_bary(ex)
        C[i,j]=val
    return C

P2_MASS_C=_p2_mass_coeff()
# Symmetric degree-2 exact tetra rule, normalized weights sum to 1.
_QA=0.5854101966249685; _QB=0.1381966011250105
P2_Q_L=np.array([[_QA,_QB,_QB,_QB],[_QB,_QA,_QB,_QB],[_QB,_QB,_QA,_QB],[_QB,_QB,_QB,_QA]])


def assemble_elasticity_p2(p4,t4,p2,t10,E,nu,rho,element_E_scale=None):
    D0=iso_D(E,nu); ne=len(t4); nd=3*len(p2)
    nent=ne*900
    rows=np.empty(nent,np.int32); cols=np.empty(nent,np.int32); kval=np.empty(nent,float); mval=np.empty(nent,float)
    ptr=0; totalV=0.0
    for ie,(tet,nodes10) in enumerate(zip(t4,t10)):
        x=p4[tet]; A=np.c_[np.ones(4),x]; det=np.linalg.det(A); V=abs(det)/6.0
        if V<1e-22: raise RuntimeError('degenerate tetra')
        totalV+=V; inv=np.linalg.inv(A); gradL=inv[1:,:].T  # 4x3
        Ke=np.zeros((30,30))
        D=D0 if element_E_scale is None else D0*float(element_E_scale[ie])
        for L in P2_Q_L:
            gradN=np.empty((10,3))
            for i in range(4): gradN[i]=(4*L[i]-1)*gradL[i]
            for j,(a,b) in enumerate(P2_EDGES): gradN[4+j]=4*(L[a]*gradL[b]+L[b]*gradL[a])
            B=np.zeros((6,30))
            for a,g in enumerate(gradN):
                bx,by,bz=g; k=3*a
                B[:,k:k+3]=((bx,0,0),(0,by,0),(0,0,bz),(by,bx,0),(0,bz,by),(bz,0,bx))
            Ke += (V/4.0)*(B.T@D@B)
        Me_s=rho*V*P2_MASS_C; Me=np.kron(Me_s,np.eye(3))
        gd=np.array([[3*int(n)+q for q in range(3)] for n in nodes10],np.int32).ravel()
        sl=slice(ptr,ptr+900); rows[sl]=np.repeat(gd,30); cols[sl]=np.tile(gd,30); kval[sl]=Ke.ravel(); mval[sl]=Me.ravel(); ptr+=900
    K=coo_matrix((kval,(rows,cols)),shape=(nd,nd)).tocsr(); M=coo_matrix((mval,(rows,cols)),shape=(nd,nd)).tocsr()
    return K,M,rho*totalV


def volume_average_vector_p2(p4,t4,t10,n2):
    # integral of each quadratic tetra basis: vertices -V/20; edges V/5
    w=np.zeros(n2)
    for tet,n10 in zip(t4,t10):
        x=p4[tet]; V=abs(np.linalg.det(np.c_[np.ones(4),x]))/6.0
        w[n10[:4]] += -V/20.0; w[n10[4:]] += V/5.0
    s=w.sum()
    return w/max(s,1e-300)


def p2_face_nodes(face, edge_to_node):
    a,b,c=(int(x) for x in face)
    def em(i,j):
        key=(i,j) if i<j else (j,i); return edge_to_node[key]
    return np.array([a,b,c,em(a,b),em(a,c),em(b,c)],np.int32)

# ----------------------------- acoustic P1 -----------------------------
def assemble_acoustic(p,t):
    n=len(p); rk=[];ck=[];vk=[];rm=[];cm=[];vm=[]; sm=np.ones((4,4));np.fill_diagonal(sm,2.0)
    for tet in t:
        x=p[tet]; A=np.c_[np.ones(4),x]; V=abs(np.linalg.det(A))/6.0; inv=np.linalg.inv(A); grad=inv[1:,:]
        Ke=V*(grad.T@grad);Me=V/20.0*sm;rr=np.repeat(tet,4);cc=np.tile(tet,4)
        rk.extend(rr);ck.extend(cc);vk.extend(Ke.ravel());rm.extend(rr);cm.extend(cc);vm.extend(Me.ravel())
    return coo_matrix((vk,(rk,ck)),shape=(n,n)).tocsr(),coo_matrix((vm,(rm,cm)),shape=(n,n)).tocsr()


def scalar_boundary_mass(n,faces,areas,mask):
    r=[];c=[];v=[];triM=np.array([[2,1,1],[1,2,1],[1,1,2]],float)/12
    for f,A in zip(faces[mask],areas[mask]):
        r.extend(np.repeat(f,3));c.extend(np.tile(f,3));v.extend((A*triM).ravel())
    return coo_matrix((v,(r,c)),shape=(n,n)).tocsr()

# ----------------------------- analytic diaphragm profile -----------------------------
def profile_z_and_dzdr(r, cfg):
    """Front acoustic interface midsurface, SI units. Piecewise cone/dustcap/roll.
    Outside surround outer radius the rigid baffle remains z=0.
    """
    g=cfg['geometry']; r=np.asarray(r,float); z=np.zeros_like(r); dz=np.zeros_like(r)
    ri_dc=g['dustcap_inner_radius_mm']*1e-3; ro_dc=g['dustcap_outer_radius_mm']*1e-3
    ri_c=g['cone_inner_radius_mm']*1e-3; ro_c=g['cone_outer_radius_mm']*1e-3
    ro_s=g['surround_outer_radius_mm']*1e-3
    z_apex=g['dustcap_apex_z_mm']*1e-3; z_ci=g['cone_inner_z_mm']*1e-3; z_co=g['cone_outer_z_mm']*1e-3
    roll=g['surround_roll_height_mm']*1e-3
    # center/apex continuation
    m=r<=ri_dc; z[m]=z_apex; dz[m]=0
    m=(r>ri_dc)&(r<ro_dc); a=(z_ci+(ro_dc-ri_c)/(ro_c-ri_c)*(z_co-z_ci)-z_apex)/(ro_dc-ri_dc)
    z[m]=z_apex+a*(r[m]-ri_dc);dz[m]=a
    # cone, beginning at dustcap outer edge to avoid hidden cone in the acoustic interface
    m=(r>=ro_dc)&(r<=ro_c); a2=(z_co-z_ci)/(ro_c-ri_c);z[m]=z_ci+a2*(r[m]-ri_c);dz[m]=a2
    # roll surround arch, endpoints at z_co
    m=(r>ro_c)&(r<=ro_s); s=(r[m]-ro_c)/(ro_s-ro_c); zso=float(g.get('surround_outer_z_mm',-0.35))*1e-3; z[m]=z_co+(zso-z_co)*s+roll*np.sin(np.pi*s); dz[m]=(zso-z_co)/(ro_s-ro_c)+roll*np.pi/(ro_s-ro_c)*np.cos(np.pi*s)
    return z,dz


def build_acoustic_domains(cfg):
    p0,t=read_tets_mm(HERE/'meshes'/'acoustic_base_quarter.msh')
    R=float(cfg['geometry']['acoustic_radius_mm'])*1e-3; rr=np.hypot(p0[:,0],p0[:,1]); zmax=np.sqrt(np.maximum(R*R-rr*rr,0.0))
    h,_=profile_z_and_dzdr(rr,cfg); rb=float(cfg['geometry']['surround_outer_radius_mm'])*1e-3; h=np.where(rr<=rb*1.0001,h,0.0)
    frac=np.divide(p0[:,2],zmax,out=np.zeros_like(zmax),where=zmax>1e-12); frac=np.clip(frac,0,1)
    pf=p0.copy(); pr=p0.copy(); pf[:,2]=h+frac*(zmax-h); pr[:,2]=h+frac*(-zmax-h)
    # boundary topology from original undeformed upper quarter hemisphere
    faces,owners=boundary_faces(t); c0=p0[faces].mean(axis=1); base=np.max(np.abs(p0[faces,2]),axis=1)<2e-9
    rad0=np.linalg.norm(p0[faces].mean(axis=1),axis=1); outer=rad0>0.985*R
    # select interface faces by projected centroid, excluding baffle outside surround
    rfc=np.hypot(c0[:,0],c0[:,1]); interface=base&(rfc<=rb*1.001)
    # areas on deformed meshes, normals not required for scalar Robin
    def info(p):
        cent=p[faces].mean(axis=1); area=0.5*np.linalg.norm(np.cross(p[faces[:,1]]-p[faces[:,0]],p[faces[:,2]]-p[faces[:,0]]),axis=1)
        K,M=assemble_acoustic(p,t); B=scalar_boundary_mass(len(p),faces,area,outer)
        return {'p':p,'t':t,'K':K,'M':M,'Babs':B,'faces':faces,'area':area,'interface_faces':faces[interface],
                'interface_face_ids':np.flatnonzero(interface),'outer_face_ids':np.flatnonzero(outer)}
    front=info(pf); rear=info(pr)
    front['outer_radius_m']=rear['outer_radius_m']=R
    return front,rear

# ----------------------------- local mortar ASB -----------------------------
TRI_Q_L=np.array([[1/3,1/3,1/3],[0.6,0.2,0.2],[0.2,0.6,0.2],[0.2,0.2,0.6]])
TRI_Q_W=np.array([-27/48,25/48,25/48,25/48],float)  # sum=1, degree 3 exact


def bary_xy(q, tri_xy):
    a,b,c=tri_xy; M=np.column_stack((b-a,c-a)); rhs=q-a
    sol=np.linalg.lstsq(M,rhs,rcond=None)[0]; l1,l2=sol; return np.array([1-l1-l2,l1,l2])


def build_local_G(model, ac_front, cfg):
    # acoustic interface locator in projected x-y coordinates
    af=ac_front['interface_faces']; acp=ac_front['p']; cxy=acp[af,:2].mean(axis=1); tree=cKDTree(cxy)
    rows=[];cols=[];vals=[]; report=[]; total_area=0; total_proj=0; map_fail=0; z_mismatch=[]
    part_by={p.name:p for p in model['parts']}
    for nm in ('surround','cone','dustcap'):
        part=part_by[nm]; faces,cent,area,nout=boundary_triangle_data(part.p4,part.t4)
        rr=np.hypot(cent[:,0],cent[:,1]); mask=nout[:,2]>0.15
        if nm=='cone': mask &= rr>=float(cfg['geometry']['dustcap_outer_radius_mm'])*1e-3*0.985
        ids=np.flatnonzero(mask); p_area=0;p_proj=0;nface=0
        for fid in ids:
            f=faces[fid]; n=nout[fid]; A=area[fid]
            # reject radial/axial end caps accidentally selected by nz threshold if tiny projected contribution
            if A<=0: continue
            fn6=p2_face_nodes(f,part.edge_to_node); xyz=part.p4[f]
            for L,wq in zip(TRI_Q_L,TRI_Q_W):
                q=L@xyz; _,idx=tree.query(q[:2],k=min(24,len(cxy)))
                idx=np.atleast_1d(idx); found=None
                for j in idx:
                    tri=af[int(j)]; lam=bary_xy(q[:2],acp[tri,:2])
                    if np.min(lam)>=-2e-7 and np.max(lam)<=1+2e-7:
                        found=(tri,lam);break
                if found is None:
                    # boundary fallback: closest triangle with clipped barycentric coordinates
                    j=int(idx[0]);tri=af[j];lam=bary_xy(q[:2],acp[tri,:2]);lam=np.maximum(lam,0);lam/=max(lam.sum(),1e-300);map_fail+=1
                else: tri,lam=found
                zh=lam@acp[tri,2]; z_mismatch.append(abs(zh-q[2]))
                # P2 triangle basis with ordering a,b,c,ab,ac,bc
                l0,l1,l2=L; Ns=np.array([l0*(2*l0-1),l1*(2*l1-1),l2*(2*l2-1),4*l0*l1,4*l0*l2,4*l1*l2])
                fac=A*wq
                for ii,ns in zip(fn6,Ns):
                    gi=part.offset_node+int(ii)
                    for comp in range(3):
                        sv=ns*n[comp]*fac
                        if abs(sv)<1e-30: continue
                        row=3*gi+comp
                        for aj,na in zip(tri,lam):
                            rows.append(row);cols.append(int(aj));vals.append(sv*na)
            p_area+=A; p_proj+=A*n[2]; nface+=1
        total_area+=p_area;total_proj+=p_proj;report.append({'part':nm,'surface_faces':nface,'area_quarter_m2':p_area,'projected_area_quarter_m2':p_proj})
    G=coo_matrix((vals,(rows,cols)),shape=(model['Nd'],len(acp))).tocsr()
    return G,{'parts':report,'surface_area_full_m2':4*total_area,'projected_area_full_m2':4*total_proj,
              'mapping_fallback_quadrature_points':int(map_fail),'max_interface_z_mismatch_mm':float(max(z_mismatch,default=0)*1e3),
              'rms_interface_z_mismatch_mm':float(np.sqrt(np.mean(np.square(z_mismatch)))*1e3 if z_mismatch else 0)}

# ----------------------------- structural model -----------------------------
@dataclass
class Part:
    name:str; p4:np.ndarray; t4:np.ndarray; p:np.ndarray; t10:np.ndarray; edge_to_node:dict; offset_node:int; K:csr_matrix; M:csr_matrix; mass_kg_quarter:float


def build_model(cfg, perturb=False):
    names=['surround','cone','dustcap','spider','former','coil','neck_glue']; parts=[]; off=0
    for nm in names:
        p4,t4=read_tets_mm(HERE/'meshes'/f'{nm}.msh');p2,t10,emap=elevate_p2(p4,t4);m=cfg['materials'][nm]
        escale=None
        if perturb and nm=='cone':
            # small deliberate angular stiffness asymmetry, diagnostic only, not a claimed FR10 property
            cent=p4[t4].mean(axis=1);theta=np.arctan2(cent[:,1],cent[:,0]);amp=float(cfg.get('diagnostic_nonaxisymmetry',{}).get('cone_E_sector_perturbation_fraction',0.03))
            escale=1+amp*np.cos(2*theta+0.37)
        K,M,mass=assemble_elasticity_p2(p4,t4,p2,t10,float(m['E']),float(m['nu']),float(m['rho']),escale)
        parts.append(Part(nm,p4,t4,p2,t10,emap,off,K,M,mass));off+=len(p2)
    N=off;Nd=3*N;P=np.vstack([p.p for p in parts]);ranges={p.name:(p.offset_node,p.offset_node+len(p.p)) for p in parts}
    Kparts=block_diag([p.K for p in parts],format='csr');M=block_diag([p.M for p in parts],format='csr')
    # penalty bonded pairs on all quadratic nodes near interfaces
    kt=float(cfg['numerics']['tie_penalty_N_m']); rr=lambda x:np.hypot(x[:,0],x[:,1]); tr=[];tc=[];tv=[];trep=[]
    def pn(nm):
        p=next(x for x in parts if x.name==nm);return p.p,p.offset_node
    def tie(A,selA,B,selB,label,maxgap):
        pA,oA=pn(A);pB,oB=pn(B);IA=np.asarray(selA(pA),int);IB=np.asarray(selB(pB),int)
        tree=cKDTree(pB[IB]);d,j=tree.query(pA[IA]);keep=d<=maxgap;IA=IA[keep];J=IB[j[keep]];d=d[keep]
        if not len(IA): raise RuntimeError(f'empty tie {label}')
        # de-duplicate identical (A,B) pairs
        pairs={ (int(a),int(b)) for a,b in zip(IA,J)}
        for ia,jb in pairs:
            ga=oA+ia;gb=oB+jb
            for q in range(3):
                da=3*ga+q;db=3*gb+q;tr.extend((da,db,da,db));tc.extend((da,db,db,da));tv.extend((kt,kt,-kt,-kt))
        trep.append({'interface':label,'pairs':len(pairs),'max_gap_mm':float(d.max()*1e3),'mean_gap_mm':float(d.mean()*1e3)})
    tie('surround',lambda p:np.where(rr(p)<0.04055)[0],'cone',lambda p:np.where((rr(p)>0.03955)&(p[:,2]>-0.0042))[0],'surround-cone',0.0015)
    tie('dustcap',lambda p:np.where(rr(p)>0.0170)[0],'cone',lambda p:np.where((rr(p)>0.0170)&(rr(p)<0.0180))[0],'dustcap-cone',0.0006)
    tie('cone',lambda p:np.where((rr(p)<0.0107)&(np.abs(p[:,2]+0.0152)<0.00035))[0],'neck_glue',lambda p:np.where(np.abs(p[:,2]+0.0152)<0.00035)[0],'cone-neck',0.0006)
    tie('neck_glue',lambda p:np.where(rr(p)<0.01008)[0],'former',lambda p:np.where((rr(p)>0.00982)&(p[:,2]>-0.01555)&(p[:,2]<-0.01455))[0],'neck-former',0.0004)
    tie('spider',lambda p:np.where(rr(p)<0.0103)[0],'former',lambda p:np.where((rr(p)>0.00982)&(p[:,2]>-0.0221)&(p[:,2]<-0.0209))[0],'spider-former',0.0005)
    tie('coil',lambda p:np.where(rr(p)<0.01012)[0],'former',lambda p:np.where((rr(p)>0.00982)&(p[:,2]>-0.0311)&(p[:,2]<-0.0244))[0],'coil-former',0.00035)
    Ktie=coo_matrix((tv,(tr,tc)),shape=(Nd,Nd)).tocsr()
    # symmetry and fixed attachments
    fixed=set();tol=float(cfg['numerics'].get('symmetry_tolerance_mm',1e-6))*1e-3
    for i,x in enumerate(P):
        if abs(x[0])<tol:fixed.add(3*i)
        if abs(x[1])<tol:fixed.add(3*i+1)
    for nm,radmm in [('surround',float(cfg['geometry']['surround_outer_radius_mm'])),('spider',31.5)]:
        pp,o=pn(nm);ids=np.where(rr(pp)>=radmm*1e-3-0.00018)[0]
        for j in ids:
            for q in range(3):fixed.add(3*(o+int(j))+q)
    fixed=np.asarray(sorted(fixed),int);free=np.setdiff1d(np.arange(Nd),fixed)
    coil=next(p for p in parts if p.name=='coil');wc=volume_average_vector_p2(coil.p4,coil.t4,coil.t10,len(coil.p));gcoil=np.zeros(Nd)
    for j,wj in enumerate(wc):gcoil[3*(coil.offset_node+j)+2]=wj
    Bl=float(cfg['electrical']['Bl_Tm']);fL=(Bl/4)*gcoil
    return {'parts':parts,'P':P,'Nd':Nd,'Kparts':Kparts,'M':M,'Ktie':Ktie,'free':free,'fixed':fixed,'gcoil':gcoil,'fL':fL,
            'raw_mass_full_kg':4*sum(p.mass_kg_quarter for p in parts),'tie_report':trep}


def structural_blocks(model,cfg,sscale):
    blocks=[];eta_s=1/float(cfg['targets']['Qms'])
    for p in model['parts']:
        eta=eta_s if p.name in ('surround','spider') else float(cfg['materials'][p.name]['eta']);sc=sscale if p.name in ('surround','spider') else 1
        blocks.append(p.K.astype(complex)*(sc*(1+1j*eta)))
    return block_diag(blocks,format='csr')+model['Ktie'].astype(complex)


def solve_unit_current(cfg,model,front,rear,G,freq,sscale):
    """Exact interface condensation of the monolithic local-ASB FEM system.

    Only acoustic trace nodes actually touched by G are retained in the dense Schur
    system. All structural and acoustic volume DOFs are eliminated by sparse LU.
    The retained interface has O(10^2) DOFs, versus rank=1 in the previous Sd ASB.
    """
    w=2*math.pi*freq;rho=float(cfg['air']['rho_kg_m3']);c=float(cfg['air']['c_m_s']);k=w/c
    mscale=float(cfg['targets']['Mms_kg'])/model['raw_mass_full_kg'];sf=model['free'];Gfull=G[sf,:].tocsr()
    Hs=(structural_blocks(model,cfg,sscale)-w*w*(model['M']*mscale))[sf][:,sf].tocsc()
    def Aac(ac):return (ac['K'].astype(complex)-k*k*ac['M'].astype(complex)+(1/ac['outer_radius_m']+1j*k)*ac['Babs'].astype(complex)).tocsc()
    Af=Aac(front);Ar=Aac(rear);ns=Hs.shape[0];nf=Af.shape[0];nr=Ar.shape[0];rw2=rho*w*w
    I=np.unique(Gfull.nonzero()[1]);Gs=Gfull[:,I].tocsc();ni=len(I)
    tlu=time.time();Ls=splu(Hs);Lf=splu(Af);Lr=splu(Ar);lu_time=time.time()-tlu
    bs=int(cfg['numerics'].get('condensation_block_rhs',32))
    # Structural interface compliance S = G^T Hs^-1 G.
    S=np.empty((ni,ni),complex)
    for j0 in range(0,ni,bs):
        j1=min(j0+bs,ni); X=Ls.solve(Gs[:,j0:j1].toarray()); S[:,j0:j1]=Gs.T@X
    # Acoustic interface Green matrices H=(A^-1)_II, retaining all interface trace DOFs.
    def interface_green(L,n):
        H=np.empty((ni,ni),complex)
        for j0 in range(0,ni,bs):
            j1=min(j0+bs,ni);B=np.zeros((n,j1-j0),complex)
            for k,j in enumerate(range(j0,j1)):B[I[j],k]=1.0
            X=L.solve(B);H[:,j0:j1]=X[I,:]
        return H
    Hf=interface_green(Lf,nf);Hr=interface_green(Lr,nr);Hsum=Hf+Hr
    f=model['fL'][sf].astype(complex);u0=Ls.solve(f);x0=np.asarray(Gs.T@u0).ravel()
    Red=np.eye(ni,dtype=complex)-rw2*(S@Hsum);x=np.linalg.solve(Red,x0)
    pfI=-rw2*(Hf@x);prI=rw2*(Hr@x)
    # Recover full structural and acoustic fields.
    ufree=Ls.solve(f-Gs@pfI+Gs@prI)
    bf=np.zeros(nf,complex);br=np.zeros(nr,complex);bf[I]=-rw2*x;br[I]=rw2*x
    pf=Lf.solve(bf);pr=Lr.solve(br);u=np.zeros(model['Nd'],complex);u[sf]=ufree
    # Full coupled residuals.
    rs=Hs@ufree+Gfull@pf-Gfull@pr-f;rf=Af@pf+rw2*(Gfull.T@ufree);rr=Ar@pr-rw2*(Gfull.T@ufree)
    rnorm=math.sqrt(np.linalg.norm(rs)**2+np.linalg.norm(rf)**2+np.linalg.norm(rr)**2);bnorm=np.linalg.norm(f)
    hsrs=np.asarray(np.abs(Hs).sum(axis=1)).ravel();grs=np.asarray(np.abs(Gfull).sum(axis=1)).ravel();gtrs=np.asarray(np.abs(Gfull.T).sum(axis=1)).ravel();afrs=np.asarray(np.abs(Af).sum(axis=1)).ravel();arrs=np.asarray(np.abs(Ar).sum(axis=1)).ravel()
    Ainf=float(max(np.max(hsrs+2*grs),np.max(rw2*gtrs+afrs),np.max(rw2*gtrs+arrs)));xinf=max(float(np.max(np.abs(ufree))),float(np.max(np.abs(pf))),float(np.max(np.abs(pr))));binf=float(np.max(np.abs(f)))
    backward=float(max(np.max(np.abs(rs)),np.max(np.abs(rf)),np.max(np.abs(rr)))/max(Ainf*xinf+binf,1e-300))
    return u,pf,pr,{'SYS_shape':(ns+nf+nr,ns+nf+nr),'SYS_nnz':int(Hs.nnz+Af.nnz+Ar.nnz+4*Gfull.nnz),'backward_error_inf':backward,
                    'relative_residual_2':float(rnorm/max(bnorm,1e-300)),'solver':'exact 287-ish local-interface condensation (sparse volume LU + dense trace solve)',
                    'interface_trace_dofs':int(ni),'reduced_condition_number_2':float(np.linalg.cond(Red)),'block_LU_time_s':lu_time}

def calibrate_fs(cfg,model,front,rear,G,target=90.0):
    # Secant in log stiffness scale using motional reactance at target fs.
    Bl=float(cfg['electrical']['Bl_Tm']);w=2*math.pi*target
    def fun(s):
        u,pf,pr,meta=solve_unit_current(cfg,model,front,rear,G,target,s);x=complex(model['gcoil']@u);z=1j*w*Bl*x
        return z.imag,z.real,meta
    seeds=cfg.get('calibration',{}).get('p2_fs_scale_bracket',[0.5,5.0]);a,b=map(float,seeds);fa,ra,_=fun(a);fb,rb,_=fun(b);hist=[(a,fa,ra),(b,fb,rb)]
    # expand bracket if necessary
    for _ in range(5):
        if fa*fb<=0:break
        if abs(fa)<abs(fb):b*=2;fb,rb,_=fun(b);hist.append((b,fb,rb))
        else:a/=2;fa,ra,_=fun(a);hist.append((a,fa,ra))
    if fa*fb>0:
        # fallback secant without guaranteed bracket
        pass
    for _ in range(7):
        if fa*fb<=0:c=math.sqrt(a*b)  # log-space bisection, stiffness is multiplicative
        else:c=b-fb*(b-a)/max(fb-fa,1e-30);c=max(c,1e-4)
        fc,rc,_=fun(c);hist.append((c,fc,rc))
        if abs(fc)<2e-3:return c,hist
        if fa*fc<=0:b,fb=c,fc
        else:a,fa=c,fc
    best=min(hist,key=lambda x:abs(x[1]));return best[0],hist


def radiating_surface_metrics(model,u,G,cfg):
    # Local volume-displacement coherence. 1 = all patches in phase; lower values indicate breakup/cancellation.
    q=G.T@u; qsum=complex(np.sum(q)); qabs=float(np.sum(np.abs(q))); coherence=abs(qsum)/max(qabs,1e-300)
    # True circumferential nonuniformity: compare original mesh vertices that share the same meridional (r,z)
    # station but differ in theta. This avoids contaminating the metric with radial breakup gradients.
    U=u.reshape(-1,3);num=den=0.0;ng=0
    for p in model['parts']:
        if p.name not in ('surround','cone','dustcap'):continue
        faces,cent,area,nout=boundary_triangle_data(p.p4,p.t4);mask=nout[:,2]>0.15
        if p.name=='cone':mask&=np.hypot(cent[:,0],cent[:,1])>=float(cfg['geometry']['dustcap_outer_radius_mm'])*1e-3*0.985
        vids=np.unique(faces[mask].ravel());groups={}
        for vid in vids:
            c=p.p4[int(vid)];key=(round(float(np.hypot(c[0],c[1])),7),round(float(c[2]),7));groups.setdefault(key,[]).append(U[p.offset_node+int(vid),2])
        for vals in groups.values():
            if len(vals)<5:continue
            vals=np.asarray(vals);av=vals.mean();num+=float(np.sum(np.abs(vals-av)**2));den+=float(np.sum(np.abs(vals)**2));ng+=1
    az=float(math.sqrt(num/max(den,1e-300)))
    return {'volume_displacement_coherence':float(coherence),'azimuthal_variation_ratio':az,'azimuthal_metric_ring_groups':int(ng)}

def run_frequency(cfg,model,front,rear,G,sscale,freq,voltage=1.0):
    t0=time.time();u,pf,pr,meta=solve_unit_current(cfg,model,front,rear,G,freq,sscale);w=2*math.pi*freq;Bl=float(cfg['electrical']['Bl_Tm'])
    xcoil=complex(model['gcoil']@u);Zmot=1j*w*Bl*xcoil;Zb=complex(float(cfg['electrical']['Rdc_ohm']),w*float(cfg['electrical']['Le_H']));Z=Zb+Zmot;I=complex(voltage)/Z
    uv=u*I;pfv=pf*I;prv=pr*I
    # axis node at outer sphere and 1m spherical extrapolation
    R=front['outer_radius_m'];ia=int(np.argmin(np.linalg.norm(front['p']-np.array([0,0,R]),axis=1)));pR=complex(pfv[ia]);c=float(cfg['air']['c_m_s']);k=w/c
    p1=pR*R*np.exp(-1j*k*(1-R));pref=float(cfg['air']['p_ref_Pa']);spl=20*math.log10(max(abs(p1)/math.sqrt(2)/pref,1e-300))
    metrics=radiating_surface_metrics(model,uv,G,cfg)
    maxu=float(np.max(np.linalg.norm(uv.reshape(-1,3),axis=1)))
    summary={'frequency_Hz':freq,'voltage_peak_V':voltage,'electrical':{'Z_blocked_ohm':[Zb.real,Zb.imag],'Z_motional_ohm':[Zmot.real,Zmot.imag],'Z_total_ohm':[Z.real,Z.imag],'current_A_peak':[I.real,I.imag]},
             'mechanical':{'coil_displacement_peak_m':[(xcoil*I).real,(xcoil*I).imag],'max_nodal_displacement_peak_m':maxu,'mass_used_full_kg':float(cfg['targets']['Mms_kg']),'suspension_stiffness_scale':sscale,**metrics},
             'acoustic':{'front_outer_axis_pressure_R_Pa_peak':[pR.real,pR.imag],'SPL_1m_from_outer_axis_dB':spl,'max_front_pressure_Pa_peak':float(np.max(np.abs(pfv))),'max_rear_pressure_Pa_peak':float(np.max(np.abs(prv)))},
             'mesh':{'structural_p2_nodes':len(model['P']),'structural_vector_dofs':model['Nd'],'structural_free_dofs':len(model['free']),'acoustic_front_nodes':len(front['p']),'acoustic_rear_nodes':len(rear['p']),'coupled_unknowns':int(meta['SYS_shape'][0]),'system_nnz':int(meta['SYS_nnz'])},
             'solver':{'normwise_backward_error_inf':meta['backward_error_inf'],'elapsed_s':time.time()-t0}}
    return uv,pfv,prv,summary


def write_vtu(out,model,front,rear,u,pf,pr,tag):
    if meshio is None:return
    out.mkdir(parents=True,exist_ok=True)
    # P2 displacement is written on quadratic tetra connectivity.
    for p in model['parts']:
        uu=u.reshape(-1,3)[p.offset_node:p.offset_node+len(p.p)]
        meshio.write_points_cells(out/f'{p.name}_{tag}_P2.vtu',p.p,[('tetra10',p.t10)],point_data={'u_real_m':uu.real,'u_imag_m':uu.imag,'u_abs_m':np.linalg.norm(uu,axis=1)})
    meshio.write_points_cells(out/f'acoustic_front_{tag}.vtu',front['p'],[('tetra',front['t'])],point_data={'p_real_Pa':pf.real,'p_imag_Pa':pf.imag,'p_abs_Pa':np.abs(pf)})
    meshio.write_points_cells(out/f'acoustic_rear_{tag}.vtu',rear['p'],[('tetra',rear['t'])],point_data={'p_real_Pa':pr.real,'p_imag_Pa':pr.imag,'p_abs_Pa':np.abs(pr)})


def run_all(cfg_path=None,freqs=(500,1000,2000),voltage=1.0,calibrate=True,perturb=False,outroot=None):
    cfg=json.loads(Path(cfg_path or HERE/'configs/fr10_p2_local_asb.json').read_text());t0=time.time();model=build_model(cfg,perturb=perturb);front,rear=build_acoustic_domains(cfg);G,grep=build_local_G(model,front,cfg)
    if calibrate:sscale,hist=calibrate_fs(cfg,model,front,rear,G,float(cfg['targets']['fs_Hz']))
    else:sscale=float(cfg['calibration']['p2_local_asb_suspension_scale']);hist=[]
    outroot=Path(outroot or HERE/'results'/'baseline');outroot.mkdir(parents=True,exist_ok=True)
    (outroot/'local_asb_report.json').write_text(json.dumps(grep,indent=2),encoding='utf-8')
    (outroot/'fs_calibration.json').write_text(json.dumps({'scale':sscale,'history':[{'scale':a,'ImZmot_ohm':b,'ReZmot_ohm':c} for a,b,c in hist]},indent=2),encoding='utf-8')
    summaries=[]
    for f in freqs:
        u,pf,pr,s=run_frequency(cfg,model,front,rear,G,sscale,float(f),voltage);tag=(f'{f:g}Hz').replace('.','p');od=outroot/tag;od.mkdir(exist_ok=True);(od/'summary.json').write_text(json.dumps(s,indent=2),encoding='utf-8');write_vtu(od,model,front,rear,u,pf,pr,tag)
        np.savez_compressed(od/f'solution_{tag}.npz',struct_points=model['P'],u_real=u.real,u_imag=u.imag,front_points=front['p'],front_tets=front['t'],p_front_real=pf.real,p_front_imag=pf.imag,rear_points=rear['p'],p_rear_real=pr.real,p_rear_imag=pr.imag)
        summaries.append(s)
    meta={'local_asb':grep,'suspension_scale':sscale,'frequencies':summaries,'total_elapsed_s':time.time()-t0,'perturbation_diagnostic':perturb}
    (outroot/'run_summary.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return meta

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--freq',type=float,nargs='*',default=[500,1000,2000]);ap.add_argument('--voltage',type=float,default=1);ap.add_argument('--no-calibrate',action='store_true');ap.add_argument('--perturb',action='store_true');ap.add_argument('--out',type=Path,default=None);a=ap.parse_args()
    m=run_all(freqs=a.freq,voltage=a.voltage,calibrate=not a.no_calibrate,perturb=a.perturb,outroot=a.out);print(json.dumps(m,indent=2))
