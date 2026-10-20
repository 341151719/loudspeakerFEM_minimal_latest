"""Linear curved director shells with MITC6a/MITC9 tensor tying.

Unknowns per node are translation u and displacement a of the unit normal.
Tangency (n.a=0) belongs to the assembled constraints, not a drilling penalty.
MITC6a follows Lee/Bathe (2004); MITC9 uses component-specific tensor grids.
These published elements do not establish identity with COMSOL's internals.
"""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from vented1.elements import tri_rule


class SurfaceShape:
    def __init__(self,kind,local):
        self.kind=kind;self.local=np.asarray(local,float)
        self.powers=np.array([(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)] if kind=='tri'
            else [(i,j) for j in range(3) for i in range(3)])
        if len(local)!=len(self.powers):raise ValueError('MITC requires native tri6 or quad9 nodes')
        self.inverse=np.linalg.inv(self.monomials(self.local))

    def monomials(self,q,powers=None):
        p=self.powers if powers is None else powers
        return np.prod(np.asarray(q)[:,None,:]**p[None],axis=2)

    def evaluate(self,q):
        q=np.atleast_2d(q);N=self.monomials(q)@self.inverse
        D=np.zeros((len(q),len(self.local),2));H=np.zeros((len(q),len(self.local),2,2))
        for a in range(2):
            selected=self.powers[:,a]>0;p=self.powers[selected].copy();p[:,a]-=1
            values=np.zeros((len(q),len(self.local)))
            values[:,selected]=self.monomials(q,p)*self.powers[selected,a]
            D[:,:,a]=values@self.inverse
            for b in range(2):
                factors=self.powers[:,a]*(self.powers[:,b]-(a==b));selected=factors>0
                p=self.powers[selected].copy();p[:,a]-=1;p[:,b]-=1
                values=np.zeros((len(q),len(self.local)))
                values[:,selected]=self.monomials(q,p)*factors[selected]
                H[:,:,a,b]=values@self.inverse
        return N,D,H

    def nodal_directors(self,points):
        _,D,_=self.evaluate(self.local)
        J=np.einsum('ia,qib->qab',points,D)
        normal=np.cross(J[:,:,0],J[:,:,1])
        return normal/np.linalg.norm(normal,axis=1)[:,None]

    def geometry(self,points,q,directors=None):
        N,D,H=self.evaluate(q);points=np.asarray(points,float)
        J=np.einsum('ia,qib->qab',points,D)
        J2=np.einsum('ia,qibc->qabc',points,H)
        c=np.cross(J[:,:,0],J[:,:,1]);jac=np.linalg.norm(c,axis=1)
        if np.any(jac<=1e-20):raise ValueError('Degenerate native shell geometry')
        # Degenerated shell geometry interpolates the nodal directors, just as
        # its normal-displacement field. This preserves every rigid rotation.
        directors=self.nodal_directors(points) if directors is None else np.asarray(directors)
        normal=N@directors;dn=np.einsum('ia,qib->qab',directors,D)
        return N,D,J,normal,dn,jac


def displacement_strains(shape,points,q,directors=None,z=0.):
    """Covariant engineering membrane/bending/shear B matrices at arbitrary q."""
    N,D,J,n,dn,jac=shape.geometry(points,q,directors);count=len(shape.local);width=6*count
    Bm=np.zeros((len(N),3,width));Bb=np.zeros_like(Bm);Bs=np.zeros((len(N),2,width))
    for i in range(count):
        u=slice(6*i,6*i+3);a=slice(6*i+3,6*i+6)
        for row,(alpha,beta) in enumerate([(0,0),(1,1),(0,1)]):
            factor=.5 if alpha==beta else 1.
            Bm[:,row,u]=factor*(J[:,:,alpha]*D[:,i,beta,None]+J[:,:,beta]*D[:,i,alpha,None])
            Bb[:,row,u]=factor*(dn[:,:,alpha]*D[:,i,beta,None]+dn[:,:,beta]*D[:,i,alpha,None])
            Bb[:,row,a]=factor*(J[:,:,alpha]*D[:,i,beta,None]+J[:,:,beta]*D[:,i,alpha,None])
        for alpha in range(2):
            Bs[:,alpha,u]=n*D[:,i,alpha,None]
            Bs[:,alpha,a]=J[:,:,alpha]*N[:,i,None]
    if z:
        Bm=Bm+z*Bb
        for i in range(count):
            u=slice(6*i,6*i+3);a=slice(6*i+3,6*i+6)
            for row,(alpha,beta) in enumerate([(0,0),(1,1),(0,1)]):
                factor=.5 if alpha==beta else 1.
                Bm[:,row,a]+=z*z*factor*(dn[:,:,alpha]*D[:,i,beta,None]+dn[:,:,beta]*D[:,i,alpha,None])
            for alpha in range(2):
                Bs[:,alpha,a]+=z*(dn[:,:,alpha]*N[:,i,None]+n*D[:,i,alpha,None])
        J=J+z*dn
    return Bm,Bb,Bs,(N,J,n,jac)


def interpolation(points,query,powers):
    points,query=np.asarray(points),np.asarray(query);p=np.array(powers)
    V=np.prod(points[:,None]**p[None],axis=2)
    return np.prod(query[:,None]**p[None],axis=2)@np.linalg.inv(V)


@lru_cache(None)
def triangular_tying():
    low=.5-.5/np.sqrt(3);high=1-low;center=1/3
    edge=np.array([[low,0],[high,0],[0,low],[0,high],[high,low],[low,high],[center,center]])
    # Rotated RT1 vector space: two P1 components plus (s,-r)*(d*r+e*s).
    def space(q):
        r,s=np.asarray(q).T;one=np.ones(len(r));zero=np.zeros(len(r))
        return np.stack([np.stack([one,r,s,zero,zero,zero,r*s,s*s],axis=1),
            np.stack([zero,zero,zero,one,r,s,-r*r,-r*s],axis=1)],axis=1)
    E=space(edge)
    V=np.stack([E[0,0],E[1,0],E[2,1],E[3,1],E[4,1]-E[4,0],E[5,1]-E[5,0],E[6,0],E[6,1]])
    return edge,np.linalg.inv(V),space


def mitc_strains(shape,points,q,directors=None,z=0.):
    def raw(locations):return displacement_strains(shape,points,locations,directors,z)
    Bm,Bb,_,geometry=raw(q)
    if shape.kind=='quad':
        a=(leggauss(2)[0]+1)/2;b=(leggauss(3)[0]+1)/2;c=np.array([0.,.5,1.])
        def grid(x,y):return np.array([[i,j] for j in y for i in x])
        for row,(x,y,powers) in enumerate([(a,b,[(i,j) for j in range(3) for i in range(2)]),
                (b,a,[(i,j) for j in range(2) for i in range(3)]),
                (a,a,[(i,j) for j in range(2) for i in range(2)])]):
            tying=grid(x,y);sample=raw(tying)[0][:,row]
            Bm[:,row]=interpolation(tying,q,powers)@sample
        shear=[]
        for row,(x,y,powers) in enumerate([(a,c,[(i,j) for j in range(3) for i in range(2)]),
                (c,a,[(i,j) for j in range(2) for i in range(3)])]):
            tying=grid(x,y);sample=raw(tying)[2][:,row]
            shear.append(interpolation(tying,q,powers)@sample)
        Bs=np.stack(shear,axis=1)
    else:
        edge,inverse,space=triangular_tying();sample=raw(edge)[2]
        values=np.stack([sample[0,0],sample[1,0],sample[2,1],sample[3,1],sample[4,1]-sample[4,0],
            sample[5,1]-sample[5,0],sample[6,0],sample[6,1]])
        Bs=np.einsum('qai,ij,jd->qad',space(q),inverse,values)
        low=.5-.5/np.sqrt(3);high=1-low;c=1/np.sqrt(3)
        tying=[np.array([[low,0],[high,0],[low,c]]),
            np.array([[0,low],[0,high],[c,low]]),
            np.array([[high,low],[low,high],[low,low]])]
        normal=[]
        for row,locations in enumerate(tying):
            sample=raw(locations)[0]
            selected=sample[:,row] if row<2 else .5*(sample[:,0]+sample[:,1]-sample[:,2])
            normal.append(interpolation(locations,q,[(0,0),(1,0),(0,1)])@selected)
        Bm=np.stack([normal[0],normal[1],normal[0]+normal[1]-2*normal[2]],axis=1)
    return Bm,Bb,Bs,geometry


def cartesian_strains(B,J,n,shear=False):
    t1=J[:,:,0]/np.linalg.norm(J[:,:,0],axis=1)[:,None];t2=np.cross(n,t1)
    frame=np.stack([t1,t2],axis=2)
    inverse=np.linalg.inv(np.einsum('qia,qib->qab',frame,J))
    if shear:return np.einsum('qai,qad->qid',inverse,B)
    tensor=np.empty((len(B),2,2,B.shape[-1]))
    tensor[:,0,0]=B[:,0];tensor[:,1,1]=B[:,1]
    tensor[:,0,1]=tensor[:,1,0]=.5*B[:,2]
    local=np.einsum('qai,qabd,qbj->qijd',inverse,tensor,inverse)
    return np.stack([local[:,0,0],local[:,1,1],2*local[:,0,1]],axis=1)


def shell_matrices(shape,points,material,order=5,mitc=True,directors=None,offset=0.):
    if shape.kind=='tri':
        L,w=tri_rule(order);q=L[:,1:]
    else:
        a,b=leggauss(order);a=(a+1)/2;b=b/2
        q=np.array([[x,y] for y in a for x in a]);w=np.outer(b,b).ravel()
    E,nu,t,rho=material.E,material.nu,material.thickness,material.rho
    if min(E,t,rho)<=0 or not -1<nu<.5:raise ValueError('Invalid native shell material')
    D=E/(1-nu**2)*np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    size=6*len(shape.local);K=np.zeros((size,size));M=np.zeros_like(K)
    extension=np.zeros_like(K);shear_energy=np.zeros_like(K);minimum=np.inf
    zg,zw=leggauss(3);orientation=None
    for unit,weight in zip(zg,zw):
        z=.5*t*unit-offset
        Bm,_,Bs,(N,J,n,jac)=(mitc_strains if mitc else displacement_strains)(shape,points,q,directors,z)
        F=np.concatenate([J,n[:,:,None]],axis=2);det=np.linalg.det(F)
        if orientation is None:orientation=np.sign(det[0])
        if np.any(det*orientation<=1e-20):raise ValueError('Folded shell through-thickness geometry')
        minimum=min(minimum,float(abs(det).min()))
        en=n/np.linalg.norm(n,axis=1)[:,None]
        t1=np.cross(J[:,:,1],en);t1/=np.linalg.norm(t1,axis=1)[:,None]
        frame=np.stack([t1,np.cross(en,t1),en],axis=2)
        Q=np.linalg.solve(F,frame)
        tensor=np.zeros((len(q),3,3,size))
        tensor[:,0,0]=Bm[:,0];tensor[:,1,1]=Bm[:,1]
        tensor[:,0,1]=tensor[:,1,0]=.5*Bm[:,2]
        tensor[:,0,2]=tensor[:,2,0]=.5*Bs[:,0]
        tensor[:,1,2]=tensor[:,2,1]=.5*Bs[:,1]
        for i in range(len(shape.local)):tensor[:,2,2,6*i+3:6*i+6]=N[:,i,None]*n
        local=np.einsum('qai,qabd,qbj->qijd',Q,tensor,Q,optimize=True)
        B=np.stack([local[:,0,0],local[:,1,1],2*local[:,0,1]],axis=1)
        S=2*np.stack([local[:,0,2],local[:,1,2]],axis=1)
        measure=w*abs(det)*.5*t*weight
        Ke=np.einsum('q,qad,ab,qbe->de',measure,B,D,B,optimize=True)
        Ks=np.einsum('q,qad,qae->de',measure*(5/6)*E/(2*(1+nu)),S,S,optimize=True)
        extension+=Ke;shear_energy+=Ks;K+=Ke+Ks
        mass=np.einsum('q,qi,qj->ij',measure*rho,N,N)
        section=np.array([[1,z],[z,z*z]])
        M+=np.kron(mass,np.kron(section,np.eye(3)))
    area=float(np.sum(w*shape.geometry(points,q,directors)[-1]))
    return K,M,{'in_plane':extension,'shear':shear_energy,'area':area,'minimum_volume_jacobian':minimum}
