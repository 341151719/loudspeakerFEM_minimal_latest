"""Bounded finite-thickness solid traces for hard shell connections.

At each connected director node, translation and normal displacement are the
zeroth/first moments of solid motion through the physical shell thickness.
This is a pointwise hard plane-section connection, not a solid-curl tie.
"""
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from vented1.core import SOLID_DOMAINS
from vented1.mesh import EDGES,Locator
from vented1.elements import shell_material,tet_shape


def bounded_segments(locator,x,director,h,domains,offset=0.):
    """Intersect only the finite physical thickness interval with owned solids.

    Unlike a midsurface-containing search, a skin bonded to a solid may have
    material on just one side. No point is extrapolated outside a solid cell.
    """
    lower,upper=-h/2-offset,h/2-offset;radius=max(abs(lower),abs(upper))*np.linalg.norm(director)
    intervals=[]
    for domain in domains:
        item=locator.domain_trees.get(int(domain))
        if item is None:continue
        ids,tree,max_radius=item
        for j in ids[tree.query_ball_point(x,max_radius+radius)]:
            if np.linalg.norm(x-locator.center[j])>locator.radius[j]+radius:continue
            L=locator.bary(j,x);slope=locator.inv[j]@director;dL=np.r_[-slope.sum(),slope]
            lo,hi=lower,upper
            for value,gradient in zip(L,dL):
                if abs(gradient)<1e-14/max(h,locator.radius[j]):
                    if value < -2e-11:hi=lo;break
                elif gradient>0:lo=max(lo,(-2e-11-value)/gradient)
                else:hi=min(hi,(-2e-11-value)/gradient)
            if hi-lo>1e-11*h:intervals.append((float(lo),float(hi),int(locator.cells[j]),int(j)))
    if not intervals:return []
    cuts=sorted({v for a,b,_,_ in intervals for v in [a,b]});pieces=[]
    for a,b in zip(cuts[:-1],cuts[1:]):
        middle=(a+b)/2;owners=[(ci,j) for lo,hi,ci,j in intervals if lo<=middle<=hi]
        if owners and b-a>1e-11*h:
            ci,j=min(owners);pieces.append((a,b,ci,j))
    # Do not accidentally connect two physically separated solid layers.
    for left,right in zip(pieces[:-1],pieces[1:]):
        if right[0]-left[1]>1e-7*h:raise ValueError('Disconnected solid layers inside one shell connection fiber')
    return pieces


def native_edges(element):
    local=element.shape.local
    corners=np.array([[0.,0],[1,0],[0,1]]+([[1,1]] if element.kind=='quad' else []))
    ids=np.linalg.norm(corners[:,None]-local[None],axis=2).argmin(axis=1)
    pairs=[(0,1),(1,2),(2,0)] if element.kind=='tri' else [(0,1),(1,3),(3,2),(2,0)]
    for a,b in pairs:
        p,q=corners[a],corners[b];direction=q-p
        f=(local-p)@direction/(direction@direction)
        on=np.linalg.norm(local-(p+f[:,None]*direction),axis=1)<1e-12
        yield int(ids[a]),int(ids[b]),np.flatnonzero(on&(f>=-1e-12)&(f<=1+1e-12))


def connection_constraints(mesh,space,topology,cfg,offset=0):
    width=offset+6*len(topology.xyz);solid=np.flatnonzero(np.isin(mesh.domain,list(SOLID_DOMAINS)))
    locator=Locator(mesh,solid)
    solid_edges={}
    for ci in solid:
        for a,b in EDGES:
            edge=tuple(sorted((int(mesh.tet[ci,a]),int(mesh.tet[ci,b]))))
            solid_edges.setdefault(edge,set()).add(int(mesh.domain[ci]))
    # Native and converted corner vertices are matched only within 0.1 nm;
    # this is a bounded identity check, never a nearest-boundary field mapping.
    tree=cKDTree(mesh.x);assignments={};shared_tags=set();edge_count=0
    for element in topology.elements:
        domains=set(mesh.adjacency[element.tag])&SOLID_DOMAINS
        if domains:
            shared_tags.add(element.tag)
            for node in element.nodes:assignments.setdefault(int(node),[]).append((element.tag,domains,'face'))
        for a,b,locals_on_edge in native_edges(element):
            old=[tree.query_ball_point(point,1e-10) for point in element.points[[a,b]]]
            if not all(old):raise ValueError('Native/solid corner correspondence missing')
            attached=set()
            for first in old[0]:
                for second in old[1]:attached.update(solid_edges.get(tuple(sorted((first,second))),set()))
            if not attached:continue
            for local in locals_on_edge:
                node=int(element.nodes[local])
                assignments.setdefault(node,[]).append((element.tag,attached,'edge'));edge_count+=1
    gauss,weights=np.polynomial.legendre.leggauss(3)
    rows=[];cols=[];data=[];row=0;fractions=[];conditions=[];physical_offset=[];connected=[]
    for node,options in sorted(assignments.items()):
        # One physical section per director node; use its thinnest incident
        # material section, and preserve the union of topology-owned solids.
        tag=min((o[0] for o in options),key=lambda t:shell_material(t,cfg['temperature_K']).thickness)
        h=shell_material(tag,cfg['temperature_K']).thickness
        extra=cfg['connection'].get('offset_by_boundary_m',{}).get(str(tag),0.)
        domains=set().union(*(o[1] for o in options));x=topology.xyz[node];n=topology.normals[node]
        segments=bounded_segments(locator,x,n,h,domains,extra)
        fraction=sum(b-a for a,b,_,_ in segments)/h
        if fraction<cfg['connection'].get('min_fraction',.05):
            raise ValueError(f'Native hard connection node {node}, boundary {tag}: thickness coverage {fraction:g}; coordinate {x.tolist()}')
        G=np.zeros((2,2));moments={}
        for a,b,ci,j in segments:
            sdofs=space.scell[ci]
            for g,w in zip(gauss,weights):
                z=(a+b)/2+(b-a)*g/2;weight=w*(b-a)/(2*h);basis=np.array([1.,z/h])
                point=x+z*n;L=locator.bary(j,point)
                if L.min() < -5e-10:raise ValueError('Connection trace extrapolates outside its owning solid')
                N=tet_shape(L);G+=weight*np.outer(basis,basis)
                for dof,value in zip(sdofs,N):moments[int(dof)]=moments.get(int(dof),np.zeros(2))+weight*basis*value
        cond=float(np.linalg.cond(G))
        if cond>1e10:raise ValueError('Degenerate through-thickness connection moment matrix')
        inverse=np.linalg.inv(G)
        coefficients={d:inverse@v for d,v in moments.items()}
        for mode in range(2):
            for axis in range(3):
                rows.append(row);cols.append(offset+6*node+3*mode+axis);data.append(1.)
                for dof,values in coefficients.items():
                    rows.append(row);cols.append(3*dof+axis);data.append(-values[mode]/(h if mode else 1.))
                row+=1
        fractions.append(fraction);conditions.append(cond);physical_offset.append([segments[0][0],segments[-1][1]]);connected.append(node)
    C=sparse.coo_matrix((data,(rows,cols)),shape=(row,width)).tocsr()
    return C,{'method':'native_pointwise_finite_thickness_hard_connection','connected_nodes':len(connected),
        'shared_shell_boundaries':sorted(shared_tags),'edge_node_records':edge_count,
        'minimum_thickness_coverage':min(fractions,default=0.),'maximum_section_condition':max(conditions,default=0.),
        'offset_range_m':[min((v[0] for v in physical_offset),default=0.),max((v[1] for v in physical_offset),default=0.)],
        'bounded_solid_ownership':True,'solid_geometry':'existing straight P2 solids',
        'connected_node_ids':connected}
