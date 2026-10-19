"""Attach native mixed acoustics to the existing independent structural/BEM model.

Assembly reads geometry, connectivity, material settings and cached element
matrices only. COMSOL solution vectors and reference fields are not inputs.
Mechanical elements/connectors and the faceted P0 BEM remain approximations.
"""
from pathlib import Path
import csv,json
import numpy as np
from scipy import sparse
from vented1.core import AIR_DOMAINS,SOLID_DOMAINS,atomic_json
from vented1.mesh import load,incidence,trace_positions
from vented1.model import build,Accumulator
from vented1.elements import tri_rule,tri_shape
from .native_surface_fields import read_matrix,surface_inverse,surface_shapes


class NativePressureSpace:
    def __init__(self,mechanical_space,count):
        self.mechanical_space=mechanical_space;self.np=count
    def __getattr__(self,name):return getattr(self.mechanical_space,name)


def geometry_patch(points,query,old_corners,reference_bary,kind,local):
    """Native surface pullback over an existing mechanical triangle."""
    q=surface_inverse(old_corners,query,kind)
    Np,dNp=surface_shapes(kind,local,q)
    nativeJ=np.einsum('ia,qib->qab',points,dNp)
    axes=np.column_stack([old_corners[1]-old_corners[0],old_corners[2]-old_corners[0]])
    if kind=='tri':coarseJ=np.broadcast_to(axes,(len(q),3,2))
    else:
        cross=old_corners[3]-old_corners[2]-old_corners[1]+old_corners[0]
        coarseJ=np.stack([axes[:,0]+q[:,1,None]*cross,axes[:,1]+q[:,0,None]*cross],axis=2)
    Jtriangle=np.column_stack([reference_bary[1]-reference_bary[0],reference_bary[2]-reference_bary[0]])
    rhs=np.einsum('qai,aj->qij',coarseJ,Jtriangle)
    dq=np.linalg.solve(np.einsum('qai,qaj->qij',coarseJ,coarseJ),rhs)
    J=np.einsum('qai,qij->qaj',nativeJ,dq)
    return Np,np.cross(J[:,:,0],J[:,:,1])


def build_native_coupled(source,cfg,output,progress=None):
    source,output=Path(source),Path(output)
    geom=source/'native';cache=output/'acoustic_cache'
    from .inputs import prepare_kernels, acoustic_faces
    prepare_kernels(source,cache,progress)
    if progress:progress('legacy_mechanical_assembly')
    model=build(load(source,cfg),cfg)
    mesh,space=model.mesh,model.space;inc=incidence(mesh)
    xyz=read_matrix(geom/'nodes_coords_model_unit.f64.gz','>f8')
    types={};all_dofs=[]
    for kind in ['tet','pyr','hex']:
        names=np.array((geom/(kind+'_dof_names.txt')).read_text().splitlines())
        pi=np.flatnonzero(names=='comp1.p');en=read_matrix(geom/(kind+'_nodes.i32.gz'),'>i4')
        pd=read_matrix(geom/(kind+'_dofs.i32.gz'),'>i4')[:,pi]
        domain=np.loadtxt(source/f'elements_{kind}.csv',delimiter=',',skiprows=1,usecols=0,dtype=int)
        active=np.isin(domain,list(AIR_DOMAINS))
        if np.any(pd[active]<0):raise ValueError('Missing native fluid DOF')
        types[kind]=(en,pd);all_dofs.append(pd[active].ravel())
    native=np.unique(np.concatenate(all_dofs));n=len(native)
    lookup=np.full(int(native.max())+1,-1,int);lookup[native]=np.arange(n)
    surfaces={}
    for kind in ['tri','quad']:
        en=read_matrix(geom/(kind+'_nodes.i32.gz'),'>i4');local=read_matrix(geom/(kind+'_local_coords.f64.gz'),'>f8')
        corners=np.array([[0.,0],[1,0],[0,1]]+([[1,1]] if kind=='quad' else []))
        order=np.linalg.norm(corners[:,None]-local[None],axis=2).argmin(axis=1)
        surfaces[kind]=(en,local,order)
    selections={r[0]:set(json.loads(r[2])) for r in csv.reader((source/'live_settings.csv').open())
                if len(r)==3 and r[1]=='selection' and r[0].startswith('coupling/asb')}
    L,w=tri_rule(6);Nm=np.array([tri_shape(l) for l in L])
    Bi=Accumulator((space.nm,n),250000);Cp=Accumulator((n,len(model.ext)),250000)
    ext_lookup={int(bi):j for j,bi in enumerate(model.ext)}
    faces=acoustic_faces(mesh,inc)
    triangle_count=len(np.loadtxt(source/'elements_tri.csv',delimiter=',',skiprows=1,usecols=0,dtype=int))
    interior_solid_faces=interior_shell_faces=0
    if progress:progress('native_pressure_interfaces')
    for face in faces:
        bi,ci,tag=face['bi'],face['ci'],face['boundary']
        kind,parent=('tri',bi) if bi<triangle_count else ('quad',(bi-triangle_count)//4)
        en,local,corner_order=surfaces[kind];points=xyz[en[parent]];old_corners=points[corner_order]
        nodes=mesh.b6[bi];query=L@mesh.x[mesh.btri[bi]]
        Np,vec=geometry_patch(points,query,old_corners,mesh.x[mesh.btri[bi]],kind,local)
        if np.dot(vec.mean(axis=0),face['normal'])<0:vec=-vec
        if np.any(vec@np.array(face['normal'])<=0):raise ValueError('Native/structural patch has folded orientation')
        vkind=['tet','pyr','hex'][mesh.parent[ci,0]];vp=int(mesh.parent[ci,1]);vn,vp_dofs=types[vkind]
        distance=np.linalg.norm(points[:,None]-xyz[vn[vp]][None],axis=2);order=distance.argmin(axis=1)
        if distance.min(axis=1).max()>2e-10:raise ValueError('Native interface node mismatch')
        pd=lookup[vp_dofs[vp,order]]
        if np.any(pd<0):raise ValueError('Native interface pressure ownership missing')
        if tag==277:
            panel=ext_lookup[bi];force=np.einsum('qi,q,q->i',Np,np.linalg.norm(vec,axis=1),w)
            Cp.add(pd,[panel],force[:,None]);continue
        coupling=np.einsum('qi,qj,qk,q->ikj',Nm,Np,vec,w).reshape(18,len(pd))
        if tag in selections['coupling/asb2']:
            md=(space.offset+6*space.hcell[bi,:,None]+np.arange(3)).ravel()
            Bi.add(md,pd,coupling);interior_shell_faces+=1
        if tag in selections['coupling/asb1']:
            solid=[c for c,_ in inc[tuple(sorted(mesh.btri[bi]))] if mesh.domain[c] in SOLID_DOMAINS]
            if len(solid)!=1:raise ValueError('Ambiguous structural side of native acoustic/solid coupling')
            sd=space.scell[solid[0],trace_positions(mesh,solid[0],nodes)]
            md=(3*sd[:,None]+np.arange(3)).ravel();Bi.add(md,pd,coupling);interior_solid_faces+=1
    full_Bi=Bi.finish();native_Cp=Cp.finish()
    # The external P0 BEM retains its faceted geometry; repair its two explicit
    # solid/shell overlap selections using the same work-adjoint load map.
    extra=Accumulator(model.full['Be'].shape,20000)
    overlap=selections['coupling/asb3']&selections['coupling/asb4']
    Nv=np.einsum('q,qi->i',w,Nm);outside_count=0
    for bi,panel in ext_lookup.items():
        if int(mesh.tag[bi]) not in overlap:continue
        solid=[c for c,_ in inc[tuple(sorted(mesh.btri[bi]))] if mesh.domain[c] in SOLID_DOMAINS]
        if len(solid)!=1:raise ValueError('Ambiguous exterior solid overlap')
        sd=space.scell[solid[0],trace_positions(mesh,solid[0],mesh.b6[bi])]
        md=(3*sd[:,None]+np.arange(3)).ravel()
        force=(2*model.areas[panel]*Nv[:,None]*model.normals[panel]).ravel()
        extra.add(md,[panel],force[:,None]);outside_count+=1
    model.full['Bi']=full_Bi;model.full['Be']=model.full['Be']+extra.finish()
    model.reduced['Bi']=(model.T.T@full_Bi).tocsr()
    model.reduced['Be']=(model.T.T@model.full['Be']).tocsr()
    model.Cp=native_Cp;model.space=NativePressureSpace(space,n)
    model.ak={d:sparse.load_npz(cache/f'ak_{d}.npz') for d in AIR_DOMAINS}
    model.am={d:sparse.load_npz(cache/f'am_{d}.npz') for d in AIR_DOMAINS}
    rng=np.random.default_rng(8619);u=rng.normal(size=space.nm);p=rng.normal(size=n)
    work_error=abs(u@(full_Bi@p)-(full_Bi.T@u)@p)/max(abs(u@(full_Bi@p)),1e-30)
    if work_error>1e-11:raise ArithmeticError('Native pressure/structure work adjoint failed')
    report={'native_pressure_dofs':n,'legacy_pressure_dofs':space.np,
        'interior_solid_side_records':interior_solid_faces,'interior_shell_side_records':interior_shell_faces,
        'internal_overlap_tags':sorted(selections['coupling/asb1']&selections['coupling/asb2']),
        'external_overlap_tags':sorted(overlap),'external_solid_overlap_panels':outside_count,
        'interface_virtual_work_relative_error':work_error,
        'COMSOL_solution_vectors_read_for_assembly':False,'reference_boundary_values_used_in_solve':False,
        'remaining_approximations':['legacy straight converted structural solids','faceted shell Reissner-Mindlin',
            'nodal-spin solid/shell connector','faceted P0 BEM and aperture mortar'],
        'standard_legacy_geometry_export_not_valid_for_native_pressure_space':True}
    model.report['native_acoustic_repair']=report
    output.mkdir(exist_ok=True)
    atomic_json(output/'assembly_audit.json',model.report)
    np.savez_compressed(output/'native_pressure_space.npz',native_dof=native)
    sparse.save_npz(output/'native_Bi.npz',full_Bi);sparse.save_npz(output/'native_Cp.npz',native_Cp)
    return model
