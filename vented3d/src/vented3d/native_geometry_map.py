"""Reference geometry interpolation for native Q2 hexes and 14-node pyramids."""
import numpy as np


def linear_shape(q, kind):
    if kind=='hex':
        bits=np.array([[i,j,k] for k in [0,1] for j in [0,1] for i in [0,1]])
        f=np.where(bits[None],q[:,None],1-q[:,None])
        N=np.prod(f,axis=2)
        g=np.empty((len(q),8,3))
        for a in range(3):g[:,:,a]=np.prod(f[:,:,[i for i in range(3) if i!=a]],axis=2)*(2*bits[:,a]-1)
        return N,g
    x,y,z=q.T;h=np.maximum(1-z,1e-14);xy=x*y/h
    N=np.column_stack([1-z-x-y+xy,x-xy,y-xy,xy,z])
    a,b,c=x/h,y/h,x*y/h**2
    g=np.array([[-1+b,-1+a,-1+c],[1-b,-a,-c],[-b,1-a,-c],[b,a,c],
                [np.zeros(len(q)),np.zeros(len(q)),np.ones(len(q))]]).transpose(2,0,1)
    return N,g


def inverse_linear(corners, query, kind):
    axes=[1,2,4] if kind=='hex' else [1,2,4]
    J=np.stack([corners[:,i]-corners[:,0] for i in axes],axis=2)
    q=np.linalg.solve(J,(query-corners[:,0])[:,:,None])[:,:,0]
    for _ in range(15):
        N,g=linear_shape(q,kind)
        r=np.einsum('ei,eia->ea',N,corners)-query
        keep=np.linalg.norm(r,axis=1)>1e-12
        if not np.any(keep):return q
        jac=np.einsum('eia,eib->eab',corners[keep],g[keep])
        q[keep]-=np.linalg.solve(jac,r[keep,:,None])[:,:,0]
    raise ValueError('Native linear geometry inverse did not converge')


def basis(q,kind):
    if kind=='hex':
        powers=np.array([[i,j,k] for k in range(3) for j in range(3) for i in range(3)])
        return np.prod(q[:,None,:]**powers[None],axis=2)
    z=q[:,2];h=1-z
    a=np.divide(q[:,0],h,out=np.zeros(len(q)),where=abs(h)>1e-14)
    b=np.divide(q[:,1],h,out=np.zeros(len(q)),where=abs(h)>1e-14)
    return np.column_stack([*[h*h*a**i*b**j for j in range(3) for i in range(3)],
                            *[z*h*a**i*b**j for j in range(2) for i in range(2)],z*z])


def map_points(native_points, local, physical_query, kind):
    corners_local=(np.array([[i,j,k] for k in [0,1] for j in [0,1] for i in [0,1]])
        if kind=='hex' else np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1]]))
    order=np.linalg.norm(corners_local[:,None]-local[None],axis=2).argmin(axis=1)
    if not np.array_equal(local[order],corners_local):raise ValueError('Unknown native corner coordinates')
    q=inverse_linear(native_points[:,order],physical_query,kind)
    V=basis(local,kind)
    if np.linalg.matrix_rank(V)!=len(local):raise ValueError('Native reference basis is not unisolvent')
    N=basis(q,kind)@np.linalg.inv(V)
    return np.einsum('ei,eia->ea',N,native_points)
