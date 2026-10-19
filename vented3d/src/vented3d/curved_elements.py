"""Batched isoparametric P2 acoustic tetrahedra for geometry diagnostics."""
import numpy as np
from vented1.elements import tet_rule, tet_shape
from vented1.mesh import EDGES


def inverse_shape(points, queries, tolerance=5e-12):
    """Evaluate P2 shapes at physical points using the curved inverse map."""
    points,queries=np.asarray(points,float),np.asarray(queries,float)
    if points.shape!=(len(queries),10,3) or queries.shape!=(len(points),3):
        raise ValueError('Inverse map input shape mismatch')
    q=np.full((len(points),3),.25)
    g=np.array([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    for iteration in range(20):
        L=np.column_stack([1-q.sum(axis=1),q])
        N=np.column_stack([L*(2*L-1),*[4*L[:,i]*L[:,j] for i,j in EDGES]])
        residual=np.einsum('ei,eia->ea',N,points)-queries
        if np.max(np.linalg.norm(residual,axis=1),initial=0)<=tolerance:
            if np.any(L< -1e-8):raise ValueError('Reference query lies outside curved donor cell')
            return N,L
        dN=np.concatenate([(4*L-1)[:,:,None]*g[None],
            np.stack([4*(L[:,i,None]*g[j]+L[:,j,None]*g[i]) for i,j in EDGES],axis=1)],axis=1)
        J=np.einsum('eia,eib->eab',points,dN)
        if np.any(np.linalg.det(J)<=0):raise ValueError('Nonpositive inverse-map Jacobian')
        q-=np.linalg.solve(J,residual[:,:,None])[:,:,0]
    raise ValueError('Curved inverse map did not converge')


def scalar_batch(points, order=4):
    """Return K, M, volumes, minimum sampled det(J); reject inverted cells."""
    points = np.asarray(points, float)
    if points.ndim != 3 or points.shape[1:] != (10,3):
        raise ValueError('Expected (elements,10,3) P2 geometry')
    L, weights = tet_rule(order)
    gradients = np.array([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    shapes = [tet_shape(l, gradients) for l in L]
    N = np.array([a[0] for a in shapes])
    dN = np.array([a[1] for a in shapes])
    J = np.einsum('eia,qib->eqab',points,dN)
    det = np.linalg.det(J)
    if np.any(det <= 0):
        bad = np.flatnonzero(np.any(det<=0,axis=1))
        raise ValueError(f'Nonpositive sampled Jacobians in local elements {bad[:10].tolist()}')
    g = np.einsum('qib,eqba->eqia',dN,np.linalg.inv(J))
    measure = det*weights
    K = np.einsum('eq,eqia,eqja->eij',measure,g,g,optimize=True)
    M = np.einsum('eq,qi,qj->eij',measure,N,N,optimize=True)
    return K, M, measure.sum(axis=1), float(det.min())
