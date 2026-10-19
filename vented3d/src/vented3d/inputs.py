"""Geometry-only native inputs and reproducible acoustic element caches."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy import sparse
from vented1.core import AIR_DOMAINS,SOLID_DOMAINS,SHELL_TAGS,SYMMETRY,atomic_json,sha256
from vented1.model import Accumulator
from .native_surface_fields import read_matrix
from .native_acoustic_elements import NativeShape


def volume_inputs(source):
    source=Path(source);geom=source/'native'
    xyz=read_matrix(geom/'nodes_coords_model_unit.f64.gz','>f8')
    types={};all_dofs=[]
    for kind in ['tet','pyr','hex']:
        local=read_matrix(geom/(kind+'_local_coords.f64.gz'),'>f8')
        nodes=read_matrix(geom/(kind+'_nodes.i32.gz'),'>i4')
        dofs=read_matrix(geom/(kind+'_dofs.i32.gz'),'>i4')
        ld=read_matrix(geom/(kind+'_local_dof_coords.f64.gz'),'>f8')
        names=np.array((geom/(kind+'_dof_names.txt')).read_text().splitlines())
        pi=np.flatnonzero(names=='comp1.p')
        if not np.array_equal(local,ld[pi]):raise ValueError('Native geometry/pressure coordinates differ')
        domain=np.loadtxt(source/f'elements_{kind}.csv',delimiter=',',skiprows=1,usecols=0,dtype=int)
        if len(nodes)!=len(domain):raise ValueError('Native/base volume row correspondence differs')
        active=np.flatnonzero(np.isin(domain,list(AIR_DOMAINS)));pd=dofs[:,pi]
        if np.any(nodes[active]<0) or np.any(pd[active]<0):raise ValueError('Missing active native field node')
        types[kind]={'nodes':nodes,'dofs':pd,'local':local,'domain':domain,'active':active,'shape':NativeShape(kind,local)}
        all_dofs.append(pd[active].ravel())
    native=np.unique(np.concatenate(all_dofs))
    return xyz,types,native


def acoustic_faces(mesh,inc):
    faces=[]
    for bi,tag in enumerate(mesh.tag):
        tag=int(tag)
        if tag in SYMMETRY:continue
        adjacent=inc[tuple(sorted(mesh.btri[bi]))]
        moving=tag in SHELL_TAGS or any(mesh.domain[c] in SOLID_DOMAINS for c,_ in adjacent)
        if tag!=277 and not moving:continue
        for ci,oriented in adjacent:
            if mesh.domain[ci] not in AIR_DOMAINS:continue
            p=mesh.x[oriented];normal=np.cross(p[1]-p[0],p[2]-p[0]);area2=np.linalg.norm(normal)
            faces.append({'bi':bi,'ci':ci,'boundary':tag,'normal':normal/area2,'area2':float(area2)})
    return faces


def prepare_kernels(source,cache,progress=None):
    source,cache=Path(source),Path(cache);cache.mkdir(parents=True,exist_ok=True)
    xyz,types,native=volume_inputs(source);n=len(native)
    lookup=np.full(int(native.max())+1,-1,int);lookup[native]=np.arange(n)
    fingerprint={'schema':1,'order':5,
        'element_source':sha256(Path(__file__).with_name('native_acoustic_elements.py')),
        'geometry_source':sha256(Path(__file__).with_name('native_geometry_map.py')),
        'input_source':sha256(Path(__file__)),
        'geometry_files':{p.name:sha256(p) for p in sorted((source/'native').iterdir()) if p.is_file()},
        'domain_files':{k:sha256(source/f'elements_{k}.csv') for k in types}}
    manifest=cache/'assembly_manifest.json'
    ready=manifest.exists() and json.loads(manifest.read_text())==fingerprint and all((cache/f'{s}_{d}.npz').exists() for s in ['ak','am'] for d in AIR_DOMAINS)
    if not ready:
        if progress:progress('native_acoustic_kernel_assembly')
        ak={d:Accumulator((n,n),250000) for d in AIR_DOMAINS};am={d:Accumulator((n,n),250000) for d in AIR_DOMAINS}
        for kind,item in types.items():
            for first in range(0,len(item['active']),96):
                batch=item['active'][first:first+96]
                K,M,_,_=item['shape'].matrices(xyz[item['nodes'][batch]],5)
                for j,parent in enumerate(batch):
                    domain=int(item['domain'][parent]);pd=lookup[item['dofs'][parent]]
                    ak[domain].add(pd,pd,K[j]);am[domain].add(pd,pd,M[j])
        for d in AIR_DOMAINS:
            sparse.save_npz(cache/f'ak_{d}.npz',ak[d].finish());sparse.save_npz(cache/f'am_{d}.npz',am[d].finish())
        atomic_json(manifest,fingerprint)
    return {'native_pressure_dofs':n,'native_elements':{k:len(v['active']) for k,v in types.items()},'cache_reused':ready}
