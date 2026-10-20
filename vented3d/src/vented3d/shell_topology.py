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
    records=[];averages={};counts={}
    for kind in ['tri','quad']:
        en=read_matrix(geom/f'{kind}_nodes.i32.gz','>i4')
        local=read_matrix(geom/f'{kind}_local_coords.f64.gz','>f8');shape=SurfaceShape(kind,local)
        tags=np.loadtxt(source/f'elements_{kind}.csv',delimiter=',',skiprows=1,usecols=0,dtype=int)
        for parent in np.flatnonzero(np.isin(tags,list(SHELL_TAGS))):
            nodes=en[parent];tag=int(tags[parent]);normals=shape.nodal_directors(xyz[nodes])
            for node,normal in zip(nodes,normals):
                key=(int(node),tag)
                if key not in averages:averages[key]=np.zeros(3);counts[key]=0
                averages[key]+=normal;counts[key]+=1
            records.append((kind,int(parent),tag,nodes,shape))
    for key,value in averages.items():
        if np.linalg.norm(value)<1e-8:raise ValueError('Inconsistent normals inside one native shell boundary')
        averages[key]=value/np.linalg.norm(value)
    by_geometry={}
    for node,tag in averages:by_geometry.setdefault(node,[]).append(tag)
    lookup={};positions=[];normal_nodes=[];native_nodes=[];groups=[]
    for node,tags in sorted(by_geometry.items()):
        clusters=[]
        for tag in sorted(tags):
            candidates=[j for j,cluster in enumerate(clusters) if all(averages[node,tag]@averages[node,t]>np.cos(fold_angle) for t in cluster)]
            if candidates:clusters[candidates[0]].append(tag)
            else:clusters.append([tag])
        group=[]
        for cluster in clusters:
            index=len(positions);normal=sum(averages[node,tag] for tag in cluster);normal/=np.linalg.norm(normal)
            positions.append(xyz[node]);normal_nodes.append(normal);native_nodes.append(node);group.append(index)
            for tag in cluster:lookup[node,tag]=index
        if len(group)>1:groups.append(group)
    normal_nodes=np.asarray(normal_nodes)
    # A smooth surface may be represented by boundaries with opposite
    # orientations. Use one geometric normal line before imposing a_i=-a_j.
    for group in groups:
        for j,node in enumerate(group):
            for other in group[:j]:
                if normal_nodes[node]@normal_nodes[other]<-np.cos(fold_angle):
                    normal_nodes[node]=-normal_nodes[other];break
    elements=[];index={}
    for kind,parent,tag,nodes,shape in records:
        owned=np.array([lookup[int(node),tag] for node in nodes])
        index[kind,parent]=len(elements)
        elements.append(ShellElement(kind,parent,tag,nodes,owned,shape,xyz[nodes],normal_nodes[owned]))
    return ShellTopology(elements,np.array(positions),normal_nodes,np.array(native_nodes),groups,index,
        {'native_shell_elements':len(elements),'native_tri6':sum(e.kind=='tri' for e in elements),
         'native_quad9':sum(e.kind=='quad' for e in elements),'director_nodes':len(positions),
         'geometric_nodes':len(by_geometry),'fold_node_groups':len(groups),'fold_angle_rad':fold_angle})
