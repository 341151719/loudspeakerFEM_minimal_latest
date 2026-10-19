"""Entity-local native surface interpolation, retaining original DOF ownership."""
from pathlib import Path
import csv
import gzip
import struct
import numpy as np


def read_matrix(path,dtype):
    with gzip.open(path,'rb') as stream:raw=stream.read()
    dims=struct.unpack('>2i',raw[:8])
    return np.frombuffer(raw,dtype=dtype,offset=8).reshape(dims).T


def read_vector(path):
    with gzip.open(path,'rb') as stream:raw=stream.read()
    version,n=struct.unpack('>2i',raw[:8])
    if version!=1 or len(raw)!=8+16*n:raise ValueError('Native vector layout mismatch')
    a=np.frombuffer(raw,dtype='>f8',offset=8).reshape(n,2)
    return a[:,0]+1j*a[:,1]


def surface_basis(q,kind,count):
    powers=([(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)] if kind=='tri' else
            [(i,j) for j in range(3) for i in range(3) if count==9 or (i,j)!=(2,2)])
    if len(powers)!=count:raise ValueError('Unsupported native surface shape count')
    return np.prod(np.asarray(q)[:,None,:]**np.array(powers)[None],axis=2)


def surface_shapes(kind,local,q):
    """Native surface basis and its two reference derivatives."""
    count=len(local)
    powers=([(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)] if kind=='tri' else
            [(i,j) for j in range(3) for i in range(3)])
    inverse=np.linalg.inv(surface_basis(local,kind,count))
    N=surface_basis(q,kind,count)@inverse;D=np.zeros((len(q),count,2));p=np.array(powers)
    for a in range(2):
        keep=p[:,a]>0;lower=p[keep].copy();lower[:,a]-=1
        values=np.zeros((len(q),count));values[:,keep]=np.prod(q[:,None]**lower[None],axis=2)*p[keep,a]
        D[:,:,a]=values@inverse
    return N,D


def surface_inverse(corners,query,kind):
    origin=corners[0]
    axes=np.column_stack([corners[1]-origin,corners[2]-origin])
    q=(query-origin)@np.linalg.pinv(axes).T
    if kind=='tri':
        if np.max(np.linalg.norm(q@axes.T+origin-query,axis=1))>1e-10:
            raise ValueError('Query does not belong to original planar triangle')
        return q
    cross=corners[3]-corners[2]-corners[1]+corners[0]
    for _ in range(15):
        mapped=origin+q[:,0,None]*axes[:,0]+q[:,1,None]*axes[:,1]+(q[:,0]*q[:,1])[:,None]*cross
        r=mapped-query
        if np.max(np.linalg.norm(r,axis=1),initial=0)<1e-11:return q
        J=np.stack([axes[:,0]+q[:,1,None]*cross,axes[:,1]+q[:,0,None]*cross],axis=2)
        q-=np.linalg.solve(np.einsum('eai,eaj->eij',J,J),np.einsum('eai,ea->ei',J,r)[:,:,None])[:,:,0]
    raise ValueError('Native quad inverse map did not converge')


class NativeSurfaceFields:
    def __init__(self,geometry,vectors,dof_file,triangle_count=6572):
        self.triangle_count=triangle_count
        self.xyz=read_matrix(geometry/'nodes_coords_model_unit.f64.gz','>f8')
        self.vectors=np.array([read_vector(path) for path in vectors])
        with dof_file.open() as stream:rows=list(csv.DictReader(stream))
        self.to_solution=np.full(max(int(r['dof_index']) for r in rows)+1,-1,int)
        for r in rows:self.to_solution[int(r['dof_index'])]=int(r['solution_vector_index'])
        if self.to_solution.max()>=self.vectors.shape[1]:raise ValueError('DOF/solution vector mismatch')
        self.elements={}
        self.shapes={}
        for kind in ['tri','quad']:
            local=read_matrix(geometry/(kind+'_local_coords.f64.gz'),'>f8')
            names=(geometry/(kind+'_dof_names.txt')).read_text().splitlines()
            ld=read_matrix(geometry/(kind+'_local_dof_coords.f64.gz'),'>f8')
            dofs=read_matrix(geometry/(kind+'_dofs.i32.gz'),'>i4')
            nodes=read_matrix(geometry/(kind+'_nodes.i32.gz'),'>i4')
            refcorners=np.array([[0.,0],[1,0],[0,1]]+([[1,1]] if kind=='quad' else []))
            ci=np.linalg.norm(refcorners[:,None]-local[None],axis=2).argmin(axis=1)
            if not np.array_equal(local[ci],refcorners):raise ValueError('Native surface reference coordinates mismatch')
            self.elements[kind]=(nodes,dofs,local,ci)
            for component in ['u','v','w','u2','v2','w2']:
                rows=np.flatnonzero(np.array(names)=='comp1.'+component)
                self.shapes[kind,component]=(rows,np.linalg.inv(surface_basis(ld[rows],kind,len(rows))))

    def primitive(self,bi):
        return ('tri',bi) if bi<self.triangle_count else ('quad',(bi-self.triangle_count)//4)

    def coordinates(self,bi,query):
        kind,index=self.primitive(bi)
        nodes,_,local,ci=self.elements[kind]
        en=nodes[index]
        if np.any(en<0):raise ValueError('Missing native surface geometry node')
        q=surface_inverse(self.xyz[en[ci]],query,kind)
        N=surface_basis(q,kind,len(local))@np.linalg.inv(surface_basis(local,kind,len(local)))
        return q,N@self.xyz[en]

    def displacement(self,bi,q,shell):
        kind,index=self.primitive(bi)
        _,dofs,_,_=self.elements[kind]
        result=[]
        for component in (['u2','v2','w2'] if shell else ['u','v','w']):
            rows,inverse=self.shapes[kind,component]
            owned=dofs[index,rows]
            if np.any(owned<0):raise ValueError(f'Missing native {component} on {kind} row {index}')
            si=self.to_solution[owned]
            if np.any(si<0):raise ValueError('Unmapped native DOF')
            N=surface_basis(q,kind,len(rows))@inverse
            result.append(self.vectors[:,si]@N.T)
        return np.stack(result,axis=2)
