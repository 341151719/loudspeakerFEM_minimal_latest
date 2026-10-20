"""Native curved shell patches, shared motion and separate fold directors."""
from dataclasses import dataclass
import numpy as np
from .mitc_shell import SurfaceShape
from .native_surface_fields import read_matrix
from vented1.core import SHELL_TAGS


@dataclass
class ShellElement:
    kind: str
    parent: int
    tag: int
    native_nodes: np.ndarray
    nodes: np.ndarray
    shape: SurfaceShape
    points: np.ndarray
    directors: np.ndarray


@dataclass
class ShellTopology:
    elements: list
    xyz: np.ndarray
    normals: np.ndarray
    native_nodes: np.ndarray
    groups: list
    element_index: dict
    report: dict


def build_topology(source,fold_angle=.001):
    if not 0<fold_angle<np.pi/2:raise ValueError('Fold angle must be in (0,pi/2) radians')
    geom=source/'native';xyz=read_matrix(geom/'nodes_coords_model_unit.f64.gz','>f8')
    records=[];averages={};positions={};by_geometry={}
    for kind in ['tri','quad']:
        en=read_matrix(geom/f'{kind}_nodes.i32.gz','>i4')
        local=read_matrix(geom/f'{kind}_local_coords.f64.gz','>f8');shape=SurfaceShape(kind,local)
        names=np.array((geom/f'{kind}_dof_names.txt').read_text().splitlines())
        ld=read_matrix(geom/f'{kind}_local_dof_coords.f64.gz','>f8')
        dofs=read_matrix(geom/f'{kind}_dofs.i32.gz','>i4')
        director_rows=[]
        for name in ['arx','ary','arz']:
            selected=np.flatnonzero(names=='comp1.'+name)
            order=np.linalg.norm(local[:,None]-ld[selected][None],axis=2).argmin(axis=1)
            if np.linalg.norm(local-ld[selected[order]],axis=1).max()>1e-14:raise ValueError('Native shell/director reference nodes differ')
            director_rows.append(selected[order])
        tags=np.loadtxt(source/f'elements_{kind}.csv',delimiter=',',skiprows=1,usecols=0,dtype=int)
        for parent in np.flatnonzero(np.isin(tags,list(SHELL_TAGS))):
            nodes=en[parent];tag=int(tags[parent]);normals=shape.nodal_directors(xyz[nodes])
            keys=[]
            for j,(node,normal) in enumerate(zip(nodes,normals)):
                owner=tuple(int(dofs[parent,indices[j]]) for indices in director_rows)
                if min(owner)<0:raise ValueError('Missing native director ownership')
                key=(int(node),owner);keys.append(key)
                if key not in averages:
                    averages[key]=np.zeros(3);positions[key]=xyz[node]
                    by_geometry.setdefault(int(node),[]).append(key)
                if averages[key]@normal<0:normal=-normal
                averages[key]+=normal
            records.append((kind,int(parent),tag,nodes,shape,keys))
    for key,value in averages.items():
        if np.linalg.norm(value)<1e-8:raise ValueError('Inconsistent normals inside one native shell boundary')
        averages[key]=value/np.linalg.norm(value)
    lookup={};coordinates=[];normal_nodes=[];native_nodes=[];groups=[]
    for node,owners in sorted(by_geometry.items()):
        group=[]
        for key in sorted(owners):
            index=len(coordinates);lookup[key]=index
            coordinates.append(positions[key]);normal_nodes.append(averages[key]);native_nodes.append(node);group.append(index)
        if len(group)>1:groups.append(group)
    normal_nodes=np.asarray(normal_nodes)
    # A smooth surface may be represented by boundaries with opposite
    # orientations. Use one geometric normal line before imposing a_i=-a_j.
    for group in groups:
        for j,node in enumerate(group):
            for other in group[:j]:
                dot=normal_nodes[node]@normal_nodes[other]
                if abs(dot)>np.cos(fold_angle):
                    normal_nodes[node]=(1. if dot>0 else -1.)*normal_nodes[other];break
    elements=[];index={}
    for kind,parent,tag,nodes,shape,keys in records:
        owned=np.array([lookup[key] for key in keys])
        index[kind,parent]=len(elements)
        elements.append(ShellElement(kind,parent,tag,nodes,owned,shape,xyz[nodes],normal_nodes[owned]))
    return ShellTopology(elements,np.array(coordinates),normal_nodes,np.array(native_nodes),groups,index,
        {'native_shell_elements':len(elements),'native_tri6':sum(e.kind=='tri' for e in elements),
         'native_quad9':sum(e.kind=='quad' for e in elements),'director_nodes':len(coordinates),
         'director_topology_source':'native arx/ary/arz DOF ownership; source fold-angle contract checked',
         'geometric_nodes':len(by_geometry),'fold_node_groups':len(groups),'fold_angle_rad':fold_angle})
