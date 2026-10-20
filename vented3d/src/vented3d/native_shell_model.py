"""Attach native MITC shells, director folds and thickness connections."""
from dataclasses import replace
import numpy as np
from scipy import sparse
from vented1.elements import shell_material,tri_rule,cross
from vented1.model import Accumulator,reduce_constraints
from vented1.core import atomic_json,normmax
from .mitc_shell import shell_matrices
from .shell_topology import build_topology
from .shell_constraints import director_constraints,fold_constraints
from .native_connections import connection_constraints


class DirectorSpace:
    def __init__(self,legacy,hnode):
        self.legacy=legacy;self.hnode=np.asarray(hnode);self.offset=legacy.offset
        self.nm=self.offset+6*len(hnode)
    def __getattr__(self,name):
        if name in ['hcell','hdofs']:raise RuntimeError('Native shell fields require their saved primitive topology')
        return getattr(self.legacy,name)


def rigid_modes(mesh,space,topology):
    R=np.zeros((space.nm,6))
    for i,g in enumerate(space.snode):
        R[3*i:3*i+3,:3]=np.eye(3);R[3*i:3*i+3,3:]=-cross(mesh.x[g])
    for i,(x,n) in enumerate(zip(topology.xyz,topology.normals)):
        start=space.offset+6*i
        R[start:start+3,:3]=np.eye(3);R[start:start+3,3:]=-cross(x)
        R[start+3:start+6,3:]=-cross(n)
    return R


def symmetry(mesh,space,topology):
    fixed=list(3*np.flatnonzero(abs(mesh.x[space.snode,1])<1e-11)+1)
    for i,(x,n) in enumerate(zip(topology.xyz,topology.normals)):
        if abs(x[1])>1e-11:continue
        fixed.append(space.offset+6*i+1)
        if abs(n[1])<1e-6:fixed.append(space.offset+6*i+4)
        elif abs(n[1])>1-1e-6:fixed.extend([space.offset+6*i+3,space.offset+6*i+5])
        else:raise ValueError('Shell director at the symmetry plane has no symmetric orientation')
    return sparse.coo_matrix((np.ones(len(fixed)),(np.arange(len(fixed)),fixed)),shape=(len(fixed),space.nm)).tocsr()


def upgrade(model,source,cfg,output,context,progress=None):
    from .model import geometry_patch
    from .native_surface_fields import surface_inverse
    from .shell_settings import load_contract
    physical_contract=load_contract(source,cfg)
    if cfg['connection'].get('method')!='native_thickness_hard':
        raise ValueError('Native MITC route requires connection.method=native_thickness_hard')
    if progress:progress('native_shell_topology')
    topology=build_topology(source,physical_contract['fold_angle_rad'])
    legacy=model.space.mechanical_space;offset=legacy.offset
    oldmesh=model.mesh
    hnode=np.arange(len(oldmesh.x),len(oldmesh.x)+len(topology.xyz))
    mesh=replace(oldmesh,x=np.vstack([oldmesh.x,topology.xyz]));space=DirectorSpace(legacy,hnode)
    width=6*len(topology.xyz);acc={k:Accumulator((width,width),cfg['triplet_entries']) for k in ['K','M','loss','C']}
    if progress:progress('native_mitc_shell_assembly')
    minimum=np.inf;area=0.
    for element in topology.elements:
        material=shell_material(element.tag,cfg['temperature_K'])
        extra=cfg['connection'].get('offset_by_boundary_m',{}).get(str(element.tag),0.)
        K,M,a=shell_matrices(element.shape,element.points,material,cfg['shell'].get('quadrature',5),
            directors=element.directors,offset=extra)
        dofs=(6*element.nodes[:,None]+np.arange(6)).ravel()
        for name,value in [('K',K),('M',M),('loss',material.loss*K),('C',material.beta*K)]:acc[name].add(dofs,dofs,value)
        minimum=min(minimum,a['minimum_volume_jacobian']);area+=a['area']
    # The original full matrices have uncoupled solid/shell blocks before
    # constraints, so their solid block is retained exactly.
    model.reduced.clear()
    for name in ['K','M','loss','C']:
        solid=model.full[name][:offset,:offset].tocsr()
        model.full[name]=sparse.block_diag((solid,acc[name].finish()),format='csr')
    model.full['drill']=sparse.csr_matrix((space.nm,space.nm))
    for name in ['gc','gm','f']:model.full[name]=np.r_[model.full[name][:offset],np.zeros(width)]
    Bi=Accumulator((width,model.space.np),250000);Be=Accumulator((width,len(model.ext)),250000)
    L,w=tri_rule(6);ext_lookup={int(bi):j for j,bi in enumerate(model.ext)}
    surfaces=context['surfaces'];xyz=context['xyz'];types=context['types'];lookup=context['lookup']
    if progress:progress('native_shell_work_adjoint_interfaces')
    for face in context['faces']:
        bi,ci,tag=face['bi'],face['ci'],face['boundary']
        kind,parent=('tri',bi) if bi<context['triangle_count'] else ('quad',(bi-context['triangle_count'])//4)
        location=topology.element_index.get((kind,parent))
        if location is None or tag not in context['selections']['coupling/asb2']:continue
        element=topology.elements[location]
        en,local,corner_order=surfaces[kind];points=xyz[en[parent]];corners=points[corner_order]
        query=L@oldmesh.x[oldmesh.btri[bi]];q=surface_inverse(corners,query,kind)
        N=element.shape.evaluate(q)[0]
        Np,vec=geometry_patch(points,query,corners,oldmesh.x[oldmesh.btri[bi]],kind,local)
        if np.dot(vec.mean(axis=0),face['normal'])<0:vec=-vec
        vkind=['tet','pyr','hex'][oldmesh.parent[ci,0]];vp=int(oldmesh.parent[ci,1]);vn,pdofs=types[vkind]
        distances=np.linalg.norm(points[:,None]-xyz[vn[vp]][None],axis=2);order=distances.argmin(axis=1)
        if distances.min(axis=1).max()>2e-10:raise ValueError('Native shell/acoustic ownership differs')
        pd=lookup[pdofs[vp,order]]
        if np.any(pd<0):raise ValueError('Native shell acoustic DOF missing')
        block=np.einsum('qi,qj,qk,q->ikj',N,Np,vec,w).reshape(3*len(element.nodes),len(pd))
        md=(6*element.nodes[:,None]+np.arange(3)).ravel();Bi.add(md,pd,block)
    for bi,panel in ext_lookup.items():
        kind,parent=('tri',bi) if bi<context['triangle_count'] else ('quad',(bi-context['triangle_count'])//4)
        location=topology.element_index.get((kind,parent))
        if location is None:continue
        element=topology.elements[location]
        if element.tag not in context['selections']['coupling/asb4']:continue
        en,local,corner_order=surfaces[kind];corners=xyz[en[parent]][corner_order]
        q=surface_inverse(corners,L@oldmesh.x[oldmesh.btri[bi]],kind)
        N=element.shape.evaluate(q)[0]
        force=np.einsum('q,qi,a->ia',2*model.areas[panel]*w,N,model.normals[panel]).ravel()
        Be.add((6*element.nodes[:,None]+np.arange(3)).ravel(),[panel],force[:,None])
    model.full['Bi']=sparse.vstack([model.full['Bi'][:offset],Bi.finish()],format='csr')
    model.full['Be']=sparse.vstack([model.full['Be'][:offset],Be.finish()],format='csr')
    if progress:progress('native_thickness_connection')
    Cconnect,connection_report=connection_constraints(oldmesh,legacy,topology,cfg,offset)
    Cnormal=director_constraints(np.arange(len(topology.xyz)),topology.normals,space.nm,offset)
    Cfold,fold_report=fold_constraints(topology.groups,topology.normals,space.nm,offset)
    R=rigid_modes(mesh,space,topology)
    unconstrained=sparse.vstack([Cconnect,Cnormal,Cfold],format='csr')
    motion_error=normmax(unconstrained@R)/max(normmax(unconstrained),1.)
    if motion_error>2e-9:raise ValueError(f'Native connection/fold constraints remove rigid motion: {motion_error:g}')
    C=sparse.vstack([Cconnect,Cnormal,Cfold,symmetry(mesh,space,topology)],format='csr')
    if progress:progress('native_director_constraint_reduction')
    settings={**cfg,'_preferred_constraint_pivots':set(range(offset,space.nm))}
    C,T,free,constraint_report=reduce_constraints(C,settings)
    model.mesh=mesh;model.space.mechanical_space=space
    model.constraints=C;model.T=T;model.free=free
    model.reduced={k:(T.T@model.full[k]@T).tocsr() for k in ['K','M','loss','C','drill']}
    model.reduced.update({k:np.asarray(T.T@model.full[k]).ravel() for k in ['gc','gm','f']})
    model.reduced.update({k:(T.T@model.full[k]).tocsr() for k in ['Bi','Be']})
    report={**topology.report,**fold_report,'physical_settings_contract':physical_contract,'formulation':'published MITC6a/MITC9 curved director shell',
        'director_unknown':'normal displacement a; not axial theta','normal_tangency_rows':Cnormal.shape[0],
        'connection':connection_report,'constraints':constraint_report,'rigid_constraint_relative_error':motion_error,
        'minimum_volume_jacobian':minimum,'native_shell_area_m2':area,
        'solid_curvature_repaired':False,'COMSOL_internal_algorithm_identity_verified':False}
    model.report['native_shell_repair']=report
    model.report['connection']=connection_report;model.report['constraints']=constraint_report
    model.report['native_acoustic_repair']['remaining_approximations']=['legacy straight converted structural solids','faceted P0 BEM and aperture mortar','published MITC variant not certified identical to COMSOL internals']
    atomic_json(output/'assembly_audit.json',model.report)
    sparse.save_npz(output/'native_Bi.npz',model.full['Bi']);sparse.save_npz(output/'native_Cp.npz',model.Cp)
    np.savez_compressed(output/'native_shell_topology.npz',xyz=topology.xyz,normals=topology.normals,native_nodes=topology.native_nodes)
    return model
