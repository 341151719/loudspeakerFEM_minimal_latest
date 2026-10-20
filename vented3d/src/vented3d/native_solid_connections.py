"""Selected shared-boundary thickness connections on native curved solids."""
import numpy as np
from scipy import sparse
from vented1.elements import shell_material


def fiber_segments(element,x,normal,h,offset=0.):
    """Bounded inverse-map search; integrate only points inside this element."""
    lo,hi=-h/2-offset,h/2-offset;grid=np.linspace(lo,hi,33)
    def margins(z):
        q,valid=element.shape.inverse_points(element.points,x+np.asarray(z)[:,None]*normal)
        result=element.shape.margin(q);result[~valid]=-np.inf
        return result,q
    values,_=margins(grid);inside=values>=-1e-10
    def root(a,b,fa,fb):
        flag=fa>=-1e-10
        for _ in range(32):
            mid=(a+b)/2;fm=margins([mid])[0][0]
            # A ray on a solid face has zero margin over an interval. Search
            # membership, not an isolated zero of the minimum face margin.
            if (fm>=-1e-10)==flag:a,fa=mid,fm
            else:b,fb=mid,fm
            if b-a<max(2e-12,1e-8*h):break
        return (a+b)/2
    intervals=[];start=None
    for j,active in enumerate(inside):
        if active and start is None:
            start=grid[j] if j==0 else root(grid[j-1],grid[j],values[j-1],values[j])
        if start is not None and (not active or j==len(grid)-1):
            end=grid[j] if active else root(grid[j-1],grid[j],values[j-1],values[j])
            if end-start>1e-8*h:intervals.append((float(start),float(end)))
            start=None
    return intervals


def connection_constraints(solids,shell,cfg,offset):
    width=offset+6*len(shell.xyz);assignments={};selected=set()
    by_geometry={}
    bounds=[e.bounds() for e in solids.elements]
    for index,e in enumerate(solids.elements):
        for gid in e.native_nodes:by_geometry.setdefault(int(gid),set()).add(index)
    for e in shell.elements:
        owners=solids.surface_owners.get((e.kind,e.parent),[])
        if not owners:continue
        selected.add(e.tag)
        for node in e.nodes:assignments.setdefault(int(node),[]).append((e.tag,owners))
    rows=[];cols=[];values=[];row=0;fractions=[];conditions=[];connected=[];corner_limits=[]
    gauss,weights=np.polynomial.legendre.leggauss(3)
    for node,options in sorted(assignments.items()):
        tag=min((option[0] for option in options),key=lambda t:shell_material(t,cfg['temperature_K']).thickness)
        h=shell_material(tag,cfg['temperature_K']).thickness
        extra=cfg['connection'].get('offset_by_boundary_m',{}).get(str(tag),0.)
        original={j for _,owners in options for j in owners};domains={solids.elements[j].domain for j in original}
        candidates=set(original)
        for index in original:
            for gid in solids.elements[index].native_nodes:candidates.update(by_geometry[int(gid)])
        candidates={j for j in candidates if solids.elements[j].domain in domains}
        x,n=shell.xyz[node],shell.normals[node];intervals=[]
        endpoints=x+np.array([-h/2-extra,h/2-extra])[:,None]*n
        low,high=endpoints.min(axis=0),endpoints.max(axis=0)
        for index in sorted(candidates):
            minimum,maximum=bounds[index]
            if np.any(maximum<low-2e-12) or np.any(minimum>high+2e-12):continue
            for a,b in fiber_segments(solids.elements[index],x,n,h,extra):intervals.append((a,b,index))
        cuts=sorted({v for a,b,_ in intervals for v in [a,b]});segments=[]
        for a,b in zip(cuts[:-1],cuts[1:]):
            owners=[index for lo,hi,index in intervals if lo<=(a+b)/2<=hi]
            if owners and b-a>1e-8*h:segments.append((a,b,min(owners)))
        fraction=sum(b-a for a,b,_ in segments)/h
        use_limit=fraction<cfg['connection'].get('min_fraction',.05)
        limit_owner=None
        if use_limit:
            # At the edge/corner of a selected face a normal fiber may only
            # touch the material at z=0. Use the one-sided plane-section limit
            # of the native solid shape, evaluated on that exact solid boundary.
            # This is never used to bridge a shell/solid geometry gap.
            for index in sorted(original):
                e=solids.elements[index];q,valid=e.shape.inverse_points(e.points,[x])
                if not valid[0] or e.shape.margin(q)[0]<-5e-9:continue
                p=q[0]
                boundaries=(list(p)+[1-p.sum()]) if e.kind=='tet' else (list(p)+list(1-p)) if e.kind=='hex' else [*p,1-p[2],1-p[0]-p[2],1-p[1]-p[2]]
                if np.count_nonzero(np.abs(boundaries)<2e-7)<2:continue
                N=e.shape.evaluate(q,False)[0]
                if np.linalg.norm(N@e.points-x)>2e-12:continue
                if e.kind=='pyr' and p[2]>1-1e-10:q=.99999999*q+.00000001*np.full((1,3),.2)
                _,dN=e.shape.evaluate(q);J=np.einsum('ia,qib->qab',e.points,dN)
                grad=np.einsum('qib,qba->qia',dN,np.linalg.inv(J))[0]
                coefficients={int(dof):np.array([value,h*derivative@n]) for dof,value,derivative in zip(e.nodes,N,grad)}
                limit_owner=index;corner_limits.append(node);break
            if limit_owner is None:
                raise ValueError(f'Native curved solid connection node {node}, boundary {tag}: thickness coverage {fraction:g}; no exact boundary limit exists')
        for left,right in zip(segments[:-1],segments[1:]):
            if right[0]-left[1]>1e-7*h:raise ValueError('Disconnected native solid layers in one connection')
        G=np.zeros((2,2));moments={}
        for a,b,index in ([] if use_limit else segments):
            e=solids.elements[index];z=(a+b)/2+(b-a)*gauss/2
            q,valid=e.shape.inverse_points(e.points,x+z[:,None]*n)
            if not valid.all() or e.shape.margin(q).min() < -5e-9:
                raise ValueError('Native thickness quadrature left its owning solid')
            N=e.shape.evaluate(q,False)
            for local,w,phi in zip(z,weights*(b-a)/(2*h),N):
                basis=np.array([1.,local/h]);G+=w*np.outer(basis,basis)
                for dof,value in zip(e.nodes,phi):moments[int(dof)]=moments.get(int(dof),np.zeros(2))+w*basis*value
        cond=1. if use_limit else float(np.linalg.cond(G))
        if cond>1e10:raise ValueError('Degenerate native connection thickness moments')
        if not use_limit:
            inverse=np.linalg.inv(G)
            coefficients={dof:inverse@value for dof,value in moments.items()}
        for mode in range(2):
            for axis in range(3):
                rows.append(row);cols.append(offset+6*node+3*mode+axis);values.append(1.)
                for dof,c in coefficients.items():
                    rows.append(row);cols.append(3*dof+axis);values.append(-c[mode]/(h if mode else 1.))
                row+=1
        if not use_limit:fractions.append(fraction)
        conditions.append(cond);connected.append(node)
    C=sparse.coo_matrix((values,(rows,cols)),shape=(row,width)).tocsr()
    return C,{'method':'native_pointwise_finite_thickness_hard_connection','connection_scope':'shared_boundaries_only',
        'shared_shell_boundaries':sorted(selected),'connected_nodes':len(connected),'connected_node_ids':connected,
        'minimum_thickness_coverage':min(fractions,default=0.),'maximum_section_condition':max(conditions,default=0.),
        'solid_geometry':'native tet10/pyr13/hex20 curved geometry','bounded_solid_ownership':True,
        'edge_corner_plane_section_limit_nodes':corner_limits,'edge_corner_limit_count':len(corner_limits)}
