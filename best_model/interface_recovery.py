from __future__ import annotations
import math
import numpy as np


def _edge_quadrature(order:int=5):
    x,w=np.polynomial.legendre.leggauss(order)
    return (x+1)/2,w/2


def asb_edge_fields(model, u:np.ndarray, p_base:np.ndarray, freq_Hz:float, *, boundary_ids=None, reference_normal_trees=None, order:int=5):
    """Recover local ASB pressure, solid traction and kinematics on P2 edges.

    The returned normal points from the structural domain into the acoustic
    domain. Pressure traction acting on the solid is ``-p*n``. All quantities
    are SI and use the exp(+i omega t) convention.
    """
    solid=model.solid;ac=model.acoustic_model;mesh=ac.mesh
    sg=solid.global_to_vertex_local;am=ac.acoustic_node_map
    if boundary_ids is None:
        boundary_ids=sorted({int(t) for t in mesh.line_tags if int(t) in ac.boundary_adjacency})
    bset=set(map(int,boundary_ids));ts,ws=_edge_quadrature(order);omega=2*math.pi*float(freq_Hz)
    tri_cent=mesh.points_rz_m[mesh.triangles].mean(axis=1)
    edge_to_tri={}
    for it,t in enumerate(mesh.triangles):
        for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):edge_to_tri.setdefault(tuple(sorted((int(a),int(b)))),[]).append(it)
    rows=[]
    for seg,tag in zip(mesh.line_cells,mesh.line_tags):
        bid=int(tag)
        if bid not in bset:continue
        ga,gb=map(int,seg)
        if ga not in sg or gb not in sg or ga not in am or gb not in am:continue
        p0=mesh.points_rz_m[ga];p1=mesh.points_rz_m[gb];vec=p1-p0;L=float(np.linalg.norm(vec))
        if L<=0:continue
        n=np.array([vec[1],-vec[0]])/L
        key=tuple(sorted((ga,gb)));adj=edge_to_tri.get(key,[])
        # Orient toward an adjacent acoustic triangle and away from structural triangle.
        mid=(p0+p1)/2
        ac_tri=[i for i in adj if int(mesh.tri_domains[i]) in (2,4,7,8,22)]
        if ac_tri and np.dot(n,tri_cent[ac_tri[0]]-mid)<0:n=-n
        if reference_normal_trees and bid in reference_normal_trees:
            tree,nvals=reference_normal_trees[bid];nr=nvals[tree.query(mid)[1]]
            if np.dot(n,nr)<0:n=-n
        a=sg[ga];b=sg[gb];m=solid.edge_mid_nodes[tuple(sorted((a,b)))]
        ua=np.array([u[2*a],u[2*a+1]]);ub=np.array([u[2*b],u[2*b+1]]);um=np.array([u[2*m],u[2*m+1]])
        pa=p_base[am[ga]];pb=p_base[am[gb]]
        for t,w in zip(ts,ws):
            N0=(1-t)*(1-2*t);N1=t*(2*t-1);Nm=4*t*(1-t)
            uv=N0*ua+N1*ub+Nm*um;pv=(1-t)*pa+t*pb;x=(1-t)*p0+t*p1;r=float(x[0])
            un=complex(np.dot(uv,n));vn=1j*omega*un;an=-(omega**2)*un
            fac=float(2*math.pi*r*L*w);traction=-pv*n
            rows.append({'boundary_id':bid,'r_m':float(x[0]),'z_m':float(x[1]),'normal_r':float(n[0]),'normal_z':float(n[1]),'weight_axisym_m2':fac,'p_Pa':pv,'u_n_m':un,'v_n_m_s':vn,'a_n_m_s2':an,'traction_r_Pa':complex(traction[0]),'traction_z_Pa':complex(traction[1]),'complex_work_integrand_W_m2':pv*np.conj(vn)})
    return rows


def integrate_asb_edge_fields(rows):
    out={}
    for r in rows:
        b=int(r['boundary_id']);q=out.setdefault(b,{'complex_work_W':0j,'time_average_power_W':0.0,'int_abs_p2_Pa2_m2':0.0,'int_abs_vn2_m4_s2':0.0,'axisymmetric_area_m2':0.0})
        w=r['weight_axisym_m2'];work=r['complex_work_integrand_W_m2']
        q['complex_work_W']+=w*work;q['time_average_power_W']+=0.5*w*work.real
        q['int_abs_p2_Pa2_m2']+=w*abs(r['p_Pa'])**2;q['int_abs_vn2_m4_s2']+=w*abs(r['v_n_m_s'])**2;q['axisymmetric_area_m2']+=w
    return out
