"""Native quadratic tetrahedron, hex and rational-pyramid acoustic elements."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from .native_geometry_map import basis


def basis_with_gradient(q,kind):
    q=np.asarray(q,float);x,y,z=q.T
    if kind=='tet':
        powers=np.array([(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),
                         (1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)])
    elif kind=='hex':powers=np.array([[i,j,k] for k in range(3) for j in range(3) for i in range(3)])
    else:powers=None
    if powers is not None:
        B=np.prod(q[:,None]**powers[None],axis=2)
        D=np.zeros((len(q),len(powers),3))
        for a in range(3):
            selected=powers[:,a]>0
            lower=powers[selected].copy();lower[:,a]-=1
            D[:,selected,a]=np.prod(q[:,None]**lower[None],axis=2)*powers[selected,a]
        return B,D
    h=1-z;values=[];grad=[]
    if np.any(h<=0):raise ValueError('Pyramid derivative requested at apex or beyond')
    for middle,limit in [(False,3),(True,2)]:
        for j in range(limit):
            for i in range(limit):
                power=(1 if middle else 2)-i-j
                factor=z if middle else np.ones(len(q))
                term=x**i*y**j*h**power
                values.append(factor*term)
                dx=factor*i*x**(i-1)*y**j*h**power if i else np.zeros(len(q))
                dy=factor*j*x**i*y**(j-1)*h**power if j else np.zeros(len(q))
                dz=(term if middle else 0)-factor*power*x**i*y**j*h**(power-1)
                grad.append(np.column_stack([dx,dy,dz]))
    values.append(z*z);grad.append(np.column_stack([np.zeros(len(q)),np.zeros(len(q)),2*z]))
    return np.column_stack(values),np.stack(grad,axis=1)


@lru_cache(None)
def volume_rule(kind,order):
    q,w=leggauss(order);q=(q+1)/2;w=w/2
    a,b,c=np.meshgrid(q,q,q,indexing='ij');wa,wb,wc=np.meshgrid(w,w,w,indexing='ij')
    weight=(wa*wb*wc).ravel();a,b,c=a.ravel(),b.ravel(),c.ravel()
    if kind=='tet':
        points=np.column_stack([a,(1-a)*b,(1-a)*(1-b)*c]);weight*=((1-a)**2*(1-b))
    elif kind=='pyr':
        points=np.column_stack([(1-c)*a,(1-c)*b,c]);weight*=(1-c)**2
    else:points=np.column_stack([a,b,c])
    return points,weight


class NativeShape:
    def __init__(self,kind,local):
        self.kind=kind;self.local=np.asarray(local,float)
        if kind=='pyr':V=basis(self.local,kind)
        else:V=basis_with_gradient(self.local,kind)[0]
        self.inverse=np.linalg.inv(V)

    def evaluate(self,q,gradient=True):
        if not gradient and self.kind=='pyr':return basis(q,'pyr')@self.inverse
        B,D=basis_with_gradient(q,self.kind)
        N=B@self.inverse
        if not gradient:return N
        return N,np.einsum('qia,ij->qja',D,self.inverse)

    def matrices(self,points,order=5):
        q,w=volume_rule(self.kind,order);N,dN=self.evaluate(q)
        J=np.einsum('eia,qib->eqab',points,dN)
        det=np.linalg.det(J)
        sign=np.sign(det[:,0])
        if np.any(det*sign[:,None]<=0):raise ValueError('Folded or degenerate native geometry')
        measure=abs(det)*w
        grad=np.einsum('qib,eqba->eqia',dN,np.linalg.inv(J))
        K=np.einsum('eq,eqia,eqja->eij',measure,grad,grad,optimize=True)
        M=np.einsum('eq,qi,qj->eij',measure,N,N,optimize=True)
        return K,M,measure.sum(axis=1),float(abs(det).min())

    def inverse_coordinates(self,points,query,q):
        q=np.asarray(q,float).copy()
        for _ in range(20):
            N,dN=self.evaluate(q)
            residual=np.einsum('ei,eia->ea',N,points)-query
            if np.max(np.linalg.norm(residual,axis=1),initial=0)<5e-12:return q
            J=np.einsum('eia,eib->eab',points,dN)
            q-=np.linalg.solve(J,residual[:,:,None])[:,:,0]
        raise ValueError('Native volume inverse map failed')
