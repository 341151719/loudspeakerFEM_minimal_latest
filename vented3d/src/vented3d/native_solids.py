"""Native tet10/pyr13/hex20 structural fields and their curved geometry.

Solid fields use serendipity orders, unlike the full quadratic acoustic field.
Missing geometry-node entries belonging to other orders are never indexed.
"""
from dataclasses import dataclass
import numpy as np
from vented1.core import SOLID_DOMAINS
from vented1.elements import constitutive,solid_material,Motor
from vented1.model import Accumulator
from .native_surface_fields import read_matrix
from .native_acoustic_elements import basis_with_gradient,volume_rule
from .native_geometry_map import basis


class SolidShape:
    def __init__(self,kind,local):
        self.kind=kind;self.local=np.asarray(local,float)
        if kind=='hex':self.powers=np.array([[i,j,k] for k in range(3) for j in range(3) for i in range(3)
            if sum(a for a in [i,j,k] if a>1)<=2])
        else:self.powers=None
        self.inverse=np.linalg.inv(self.values(self.local))
        grid=np.array([[x,y,z] for z in [0.,.5,1] for y in [0.,.5,1] for x in [0.,.5,1]])
        if kind=='pyr':grid[:,:2]*=1-grid[:,2,None]
        T=np.array([[1.,0,0],[-.5,2,-.5],[0,0,1]])
        self.bounds_transform=np.kron(T,np.kron(T,T))@(self.values(grid)@self.inverse)
    def values(self,q):
        if self.kind=='hex':return np.prod(np.asarray(q)[:,None]**self.powers[None],axis=2)
        if self.kind=='pyr':return np.delete(basis(np.asarray(q),'pyr'),8,axis=1)
        return basis_with_gradient(np.asarray(q),'tet')[0]
    def evaluate(self,q,gradient=True):
        q=np.atleast_2d(q);V=self.values(q);N=V@self.inverse
        if not gradient:return N
        if self.kind=='hex':
            d=np.zeros((len(q),20,3))
            for a in range(3):
                selected=self.powers[:,a]>0;p=self.powers[selected].copy();p[:,a]-=1
                d[:,selected,a]=np.prod(q[:,None]**p[None],axis=2)*self.powers[selected,a]
        else:
            d=basis_with_gradient(q,self.kind)[1]
            if self.kind=='pyr':d=np.delete(d,8,axis=1)
        return N,np.einsum('qia,ij->qja',d,self.inverse)
    def margin(self,q):
        q=np.atleast_2d(q)
        if self.kind=='tet':return np.min(np.column_stack([q,1-q.sum(axis=1)]),axis=1)
        if self.kind=='hex':return np.min(np.column_stack([q,1-q]),axis=1)
        return np.min(np.column_stack([q,1-q[:,2],1-q[:,0]-q[:,2],1-q[:,1]-q[:,2]]),axis=1)
    def inverse_points(self,points,query,seed=None):
        query=np.atleast_2d(query)
        center=np.full(3,.25 if self.kind=='tet' else .2 if self.kind=='pyr' else .5)
        if seed is None:
            targets=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
            corner=np.linalg.norm(targets[:,None]-self.local[None],axis=2).argmin(axis=1)
            axes=(points[corner[1:]]-points[corner[0]]).T
            try:q=np.linalg.solve(axes,(query-points[corner[0]]).T).T
            except np.linalg.LinAlgError:q=np.tile(center,(len(query),1))
            if self.kind=='pyr':q[q[:,2]>=1-1e-12]=center
        else:q=np.broadcast_to(seed,(len(query),3)).copy()
        valid=np.ones(len(query),bool);done=np.zeros(len(query),bool)
        for _ in range(22):
            N=self.evaluate(q,False);r=N@points-query
            done|=np.linalg.norm(r,axis=1)<2e-12
            active=valid&~done
            if self.kind=='pyr':active&=q[:,2]<1-1e-12;valid&=(q[:,2]<1-1e-12)|done
            if not np.any(active):break
            _,dN=self.evaluate(q[active]);J=np.einsum('ia,qib->qab',points,dN)
            try:step=np.linalg.solve(J,r[active,:,None])[:,:,0]
            except np.linalg.LinAlgError:valid[active]=False;break
            q[active]-=step
            valid&=np.isfinite(q).all(axis=1)&(np.linalg.norm(q,axis=1)<100)
        return q,done&valid


@dataclass
class SolidElement:
    kind: str
    parent: int
    domain: int
    shape: SolidShape
    points: np.ndarray
    nodes: np.ndarray
    native_nodes: np.ndarray

    def bounds(self):
        control=self.shape.bounds_transform@self.points
        return control.min(axis=0),control.max(axis=0)


@dataclass
class SolidTopology:
    elements: list
    xyz: np.ndarray
    element_index: dict
    surface_owners: dict
    surface_coordinates: dict
    report: dict


def load_solids(source):
    geom=source/'native';xyz=read_matrix(geom/'nodes_coords_model_unit.f64.gz','>f8')
    elements=[];unique={};positions=[];element_index={};faces={}
    for kind in ['tet','pyr','hex']:
        names=np.array((geom/f'{kind}_dof_names.txt').read_text().splitlines())
        dl=read_matrix(geom/f'{kind}_local_dof_coords.f64.gz','>f8');field=dl[names=='comp1.u'];shape=SolidShape(kind,field)
        local=read_matrix(geom/f'{kind}_local_coords.f64.gz','>f8')
        which=np.linalg.norm(field[:,None]-local[None],axis=2).argmin(axis=1)
        if np.linalg.norm(field-local[which],axis=1).max()>1e-14:raise ValueError('Solid geometry/field local nodes differ')
        geometry=read_matrix(geom/f'{kind}_nodes.i32.gz','>i4')
        dofs=read_matrix(geom/f'{kind}_dofs.i32.gz','>i4')[:,names=='comp1.u']
        domains=np.loadtxt(source/f'elements_{kind}.csv',delimiter=',',skiprows=1,usecols=0,dtype=int)
        q=field
        if kind=='tet':tests=[q[:,0],q[:,1],q[:,2],q.sum(axis=1)-1]
        elif kind=='hex':tests=[q[:,a]-v for a in range(3) for v in [0,1]]
        else:tests=[q[:,0],q[:,1],q[:,2],q[:,0]+q[:,2]-1,q[:,1]+q[:,2]-1]
        corners=np.all((abs(q)<1e-12)|(abs(q-1)<1e-12),axis=1)
        face_local=[np.flatnonzero((abs(test)<1e-12)&corners) for test in tests]
        for parent in np.flatnonzero(np.isin(domains,list(SOLID_DOMAINS))):
            native=geometry[parent,which];raw=dofs[parent]
            if np.any(native<0) or np.any(raw<0):raise ValueError('Missing active native solid geometry/DOF')
            nodes=[]
            for gid,point in zip(raw,xyz[native]):
                if int(gid) not in unique:unique[int(gid)]=len(positions);positions.append(point)
                index=unique[int(gid)]
                if np.linalg.norm(positions[index]-point)>2e-10:raise ValueError('Shared native solid DOF has conflicting positions')
                nodes.append(index)
            element=SolidElement(kind,int(parent),int(domains[parent]),shape,xyz[native],np.array(nodes),native)
            index=len(elements);elements.append(element);element_index[kind,int(parent)]=index
            for indices in face_local:faces.setdefault(tuple(sorted(map(int,native[indices]))),[]).append(index)
    owners={};coordinates={}
    for kind in ['tri','quad']:
        en=read_matrix(geom/f'{kind}_nodes.i32.gz','>i4');local=read_matrix(geom/f'{kind}_local_coords.f64.gz','>f8')
        ref=np.array([[0.,0],[1,0],[0,1]]+([[1,1]] if kind=='quad' else []))
        ci=np.linalg.norm(ref[:,None]-local[None],axis=2).argmin(axis=1)
        for parent,row in enumerate(en):
            candidates=faces.get(tuple(sorted(map(int,row[ci]))),[])
            if not candidates:continue
            owners[kind,parent]=candidates
            for index in candidates:
                e=elements[index]
                corner_local=[]
                for gid in row[ci]:
                    match=np.flatnonzero(e.native_nodes==gid)
                    if len(match)!=1:raise ValueError('Native solid face corner ownership differs')
                    corner_local.append(e.shape.local[match[0]])
                coordinates[kind,parent,index]=np.array(corner_local)
    return SolidTopology(elements,np.array(positions),element_index,owners,coordinates,
        {'native_solid_elements':len(elements),'native_solid_nodes':len(positions),
         'native_element_types':{k:sum(e.kind==k for e in elements) for k in ['tet','pyr','hex']},
         'displacement_orders':{'tet':10,'pyr':13,'hex':20}})


def surface_volume_coordinates(kind,q,corners):
    r,s=np.asarray(q).T
    N=np.column_stack([1-r-s,r,s]) if kind=='tri' else np.column_stack([(1-r)*(1-s),r*(1-s),(1-r)*s,r*s])
    return N@corners


def solid_matrices(element,material,order=5):
    shape=element.shape;q,w=volume_rule(element.kind,order);N,dN=shape.evaluate(q)
    J=np.einsum('ia,qib->qab',element.points,dN);det=np.linalg.det(J)
    sign=np.sign(det[0])
    if np.any(det*sign<=0):raise ValueError('Folded native structural geometry')
    measure=abs(det)*w;grad=np.einsum('qib,qba->qia',dN,np.linalg.inv(J))
    B=np.zeros((len(q),6,3*len(shape.local)))
    for j in range(len(shape.local)):
        x,y,z=grad[:,j].T
        B[:,0,3*j]=x;B[:,1,3*j+1]=y;B[:,2,3*j+2]=z
        B[:,3,3*j]=y;B[:,3,3*j+1]=x
        B[:,4,3*j+1]=z;B[:,4,3*j+2]=y
        B[:,5,3*j]=z;B[:,5,3*j+2]=x
    D=constitutive(material.E,material.nu)
    K=np.einsum('q,qai,ab,qbj->ij',measure,B,D,B,optimize=True)
    M=np.kron(np.einsum('q,qi,qj->ij',measure*material.rho,N,N),np.eye(3))
    load=np.einsum('q,qi->i',measure,N)
    return K,M,load,float(measure.sum()),float(abs(det).min())


def assemble_solids(topology,cfg):
    width=3*len(topology.xyz);acc={k:Accumulator((width,width),cfg['triplet_entries']) for k in ['K','M','loss']}
    gc=np.zeros(width);gm=np.zeros(width);vc=vm=0.;minimum=np.inf
    for e in topology.elements:
        material=solid_material(e.domain,cfg['temperature_K']);K,M,load,volume,jac=solid_matrices(e,material,cfg['shell'].get('solid_quadrature',5))
        dofs=(3*e.nodes[:,None]+np.arange(3)).ravel()
        for name,value in [('K',K),('M',M),('loss',material.loss*K)]:acc[name].add(dofs,dofs,value)
        if e.domain==26:np.add.at(gc,3*e.nodes,load);vc+=volume
        if e.domain in [23,27]:np.add.at(gm,3*e.nodes,load);vm+=volume
        minimum=min(minimum,jac)
    if min(vc,vm)<=0:raise ValueError('Native motor integration domains are absent')
    gc/=vc;gm/=vm
    result={k:v.finish() for k,v in acc.items()};result.update(gc=gc,gm=gm,f=.5*Motor.BL*(gc-gm))
    return result,{'minimum_volume_jacobian':minimum,'coil_volume_half_m3':vc,'magnetic_volume_half_m3':vm}
