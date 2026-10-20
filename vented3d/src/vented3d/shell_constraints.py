"""Director tangency, physical fold angles, and finite-thickness hard mortar."""
import numpy as np
from scipy import sparse
from vented1.model import Accumulator


def director_constraints(nodes,normals,width,offset=0):
    rows=np.repeat(np.arange(len(nodes)),3)
    cols=(offset+6*np.asarray(nodes)[:,None]+np.arange(3,6)).ravel()
    return sparse.coo_matrix((np.asarray(normals).ravel(),(rows,cols)),shape=(len(nodes),width)).tocsr()


def fold_constraints(groups,normals,width,offset=0,parallel_tol=1e-10):
    """Shared translations and linearized preservation of every meeting angle.

    Each group contains distinct director nodes at one geometric location.
    Smooth nodes must already have been merged using the fold-angle threshold.
    Opposite/parallel normals use a vector condition instead of a singular
    angle condition. No artificial hinge stiffness or shared drilling DOF.
    """
    rr=[];cc=[];vv=[];row=0;folds=opposite=0
    def add(cols,values):
        nonlocal row
        rr.extend([row]*len(cols));cc.extend(cols);vv.extend(values);row+=1
    for group in groups:
        first=group[0]
        for node in group[1:]:
            for axis in range(3):add([offset+6*node+axis,offset+6*first+axis],[1.,-1.])
        for j,node in enumerate(group):
            for other in group[:j]:
                a,b=normals[node],normals[other];dot=float(a@b)
                if np.linalg.norm(a-b)<parallel_tol or np.linalg.norm(a+b)<parallel_tol:
                    sign=1. if dot>0 else -1.
                    for axis in range(3):add([offset+6*node+3+axis,offset+6*other+3+axis],[1.,-sign])
                    opposite+=int(dot<0)
                else:
                    # d(n_i.n_j)=a_i.n_j+n_i.a_j=0.
                    add([offset+6*node+3+k for k in range(3)]+[offset+6*other+3+k for k in range(3)],list(b)+list(a))
                    folds+=1
    return sparse.coo_matrix((vv,(rr,cc)),shape=(row,width)).tocsr(),{'fold_angle_rows':folds,'opposite_normal_pairs':opposite}


def point_connection(solid_dofs,Nsolid,shell_dofs,Nshell,z,width):
    """Hard kinematics u_solid(x+z*n)=u_mid(x)+z*a(x)."""
    solid_dofs=np.asarray(solid_dofs).reshape(-1,3);shell_dofs=np.asarray(shell_dofs).reshape(-1,6)
    rr=[];cc=[];vv=[]
    for axis in range(3):
        rr.extend([axis]*len(Nsolid));cc.extend(solid_dofs[:,axis]);vv.extend(Nsolid)
        rr.extend([axis]*len(Nshell));cc.extend(shell_dofs[:,axis]);vv.extend(-np.asarray(Nshell))
        rr.extend([axis]*len(Nshell));cc.extend(shell_dofs[:,axis+3]);vv.extend(-z*np.asarray(Nshell))
    return sparse.coo_matrix((vv,(rr,cc)),shape=(3,width)).tocsr()


def thickness_mortar(primitives,width,entry_limit=250000):
    """Integrate shell translation/director tests against bounded solid traces.

    A primitive supplies shell_dofs and a sequence of dictionaries containing
    Nsolid, solid_dofs, Nshell, z, weight. Geometry ownership and thickness
    coverage are validated by the caller before any constraints are accepted.
    """
    accumulator=Accumulator((width,width),entry_limit);samples=0
    for primitive in primitives:
        hdofs=np.asarray(primitive['shell_dofs']).reshape(-1,6)
        for point in primitive['samples']:
            Ns=np.asarray(point['Nsolid']);Nh=np.asarray(point['Nshell'])
            z,weight=point['z'],point['weight']
            if weight<=0:raise ValueError('Mortar quadrature weights must be positive')
            sdofs=np.asarray(point['solid_dofs']).reshape(-1,3)
            # C = integral S_shell^T (S_solid-S_shell), including moments.
            Sh=np.zeros((3,hdofs.size));Ss=np.zeros((3,sdofs.size))
            for i,value in enumerate(Nh):
                Sh[:,6*i:6*i+3]=value*np.eye(3)
                Sh[:,6*i+3:6*i+6]=z*value*np.eye(3)
            for i,value in enumerate(Ns):Ss[:,3*i:3*i+3]=value*np.eye(3)
            accumulator.add(hdofs.ravel(),sdofs.ravel(),weight*Sh.T@Ss)
            accumulator.add(hdofs.ravel(),hdofs.ravel(),-weight*Sh.T@Sh)
            samples+=1
    C=accumulator.finish();magnitudes=np.asarray(abs(C).max(axis=1).toarray()).ravel()
    active=np.flatnonzero(magnitudes>0)
    C=sparse.diags(1/magnitudes[active])@C[active]
    return C.tocsr(),{'quadrature_samples':samples,'rows':len(active),'method':'finite_thickness_hard_mortar'}
