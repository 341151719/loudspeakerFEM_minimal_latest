from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Mapping
import math

import numpy as np

from .stage4C_acoustic_structure import (
    AcousticStructureModel,
    Stage4CParameters,
    solve_stage4C_full_asb,
    EXTERIOR_FIELD_BOUNDARY,
    ACOUSTIC_DOMAINS,
)
from .exterior_field import hk_pressure_from_samples, intensity_power_from_samples


def _tri_area_grad(points: np.ndarray, tri: np.ndarray):
    p = points[np.asarray(tri, dtype=int)]
    r0,z0=p[0]; r1,z1=p[1]; r2,z2=p[2]
    det=(r1-r0)*(z2-z0)-(r2-r0)*(z1-z0)
    area=0.5*abs(det)
    if area <= 0: return None, None
    grads=np.array([
        [(z1-z2)/det, (r2-r1)/det],
        [(z2-z0)/det, (r0-r2)/det],
        [(z0-z1)/det, (r1-r0)/det],
    ], dtype=float)
    return area, grads


def _edge_to_triangles(triangles: np.ndarray) -> dict[tuple[int,int], list[int]]:
    out: dict[tuple[int,int], list[int]] = {}
    for it, tri in enumerate(triangles):
        a,b,c=map(int, tri)
        for e in ((a,b),(b,c),(c,a)):
            k=e if e[0] < e[1] else (e[1], e[0])
            out.setdefault(k, []).append(it)
    return out


@dataclass
class HKBoundaryInfo:
    n_samples: int
    boundary_id: int
    source_segments: int
    mean_radius_m: float
    mean_abs_pressure_Pa: float
    mean_abs_dpdn_Pa_per_m: float


def boundary93_hk_samples_from_p1(
    model: AcousticStructureModel,
    p_acoustic_nodes: np.ndarray,
    *,
    boundary_id: int = EXTERIOR_FIELD_BOUNDARY,
    intorder: int = 2,
):
    """Create axisymmetric HK quadrature samples on COMSOL Boundary 93.

    The Stage-4C acoustic unknowns are P1 nodal pressures on the acoustic mesh.
    For each physical line element tagged as Boundary 93, the adjacent physical-air
    triangle is used for the pressure gradient; the normal is oriented from physical
    air toward the PML/exterior domain using mphtxt up/down domain adjacency and
    element centroids.
    """
    mesh=model.mesh
    pvals=np.asarray(p_acoustic_nodes, dtype=complex)
    amap=model.acoustic_node_map
    edge_map=_edge_to_triangles(mesh.triangles)
    cent=mesh.points_rz_m[mesh.triangles].mean(axis=1)
    # 2-point Gauss on line is enough for P1 pressure and constant gradient.
    if intorder <= 1:
        xi=np.array([0.5]); wi=np.array([1.0])
    else:
        xg,wg=np.polynomial.legendre.leggauss(2)
        xi=0.5*(xg+1.0); wi=0.5*wg
    rs=[]; zs=[]; nr=[]; nz=[]; ds=[]; pb=[]; dpdn=[]
    nseg=0
    for seg, tag in zip(mesh.line_cells, mesh.line_tags):
        if int(tag) != int(boundary_id):
            continue
        key=tuple(sorted(map(int,seg)))
        adj=model.boundary_adjacency.get(int(tag))
        if adj is None:
            continue
        # Choose physical acoustic side, not PML side.  Boundary 93 is domain 4 / 5.
        physical_air = [d for d in (adj.up_domain, adj.down_domain) if d in ACOUSTIC_DOMAINS and d not in (1,5)]
        pml_or_ext = [d for d in (adj.up_domain, adj.down_domain) if d in (1,5,0)]
        adom=int(physical_air[0]) if physical_air else int(adj.up_domain if adj.up_domain in ACOUSTIC_DOMAINS else adj.down_domain)
        # find adjacent triangle in physical air domain
        tri_id=None; other_id=None
        for it in edge_map.get(key, []):
            d=int(mesh.tri_domains[it])
            if d == adom:
                tri_id=it
            elif pml_or_ext and d in pml_or_ext:
                other_id=it
        if tri_id is None:
            # fallback: any non-PML acoustic triangle
            for it in edge_map.get(key, []):
                d=int(mesh.tri_domains[it])
                if d in ACOUSTIC_DOMAINS and d not in (1,5):
                    tri_id=it; break
        if tri_id is None:
            continue
        tri=mesh.triangles[tri_id]
        area, grads=_tri_area_grad(mesh.points_rz_m, tri)
        if grads is None:
            continue
        ptri=np.array([pvals[amap[int(g)]] for g in tri], dtype=complex)
        grad_p=ptri @ grads  # [dp/dr, dp/dz]
        p0=mesh.points_rz_m[int(seg[0])]; p1=mesh.points_rz_m[int(seg[1])]
        tang=p1-p0; L=float(np.linalg.norm(tang))
        if L <= 0: continue
        n1=np.array([tang[1], -tang[0]], dtype=float)/L
        # Orient normal from physical air triangle centroid toward PML/exterior.
        if other_id is not None:
            direction=cent[other_id]-cent[tri_id]
        else:
            mid=0.5*(p0+p1); direction=mid-cent[tri_id]
        if np.dot(n1, direction) < 0:
            n1=-n1
        # local shape functions on line endpoints
        for s,w in zip(xi,wi):
            x=(1.0-s)*p0+s*p1
            # pressure interpolation along boundary segment from endpoint nodal p if possible
            pa=pvals[amap[int(seg[0])]] if int(seg[0]) in amap else 0j
            pc=pvals[amap[int(seg[1])]] if int(seg[1]) in amap else 0j
            pp=(1.0-s)*pa+s*pc
            rs.append(float(x[0])); zs.append(float(x[1])); nr.append(float(n1[0])); nz.append(float(n1[1]))
            ds.append(float(L*w)); pb.append(pp); dpdn.append(complex(grad_p[0]*n1[0]+grad_p[1]*n1[1]))
        nseg += 1
    if not rs:
        raise RuntimeError(f'No usable Boundary {boundary_id} HK samples were generated')
    return (np.asarray(rs), np.asarray(zs), np.asarray(nr), np.asarray(nz), np.asarray(ds), np.asarray(pb), np.asarray(dpdn)), HKBoundaryInfo(
        n_samples=len(rs),
        boundary_id=boundary_id,
        source_segments=nseg,
        mean_radius_m=float(np.mean(rs)),
        mean_abs_pressure_Pa=float(np.mean(np.abs(pb))),
        mean_abs_dpdn_Pa_per_m=float(np.mean(np.abs(dpdn))),
    )


def hk_axis_and_power_from_result(
    result: Mapping[str, np.ndarray],
    model: AcousticStructureModel,
    params: Stage4CParameters,
    *,
    nphi_axis: int = 72,
    nphi_power: int = 48,
    mirror: bool = True,
) -> dict[str, np.ndarray | list[dict]]:
    f=np.asarray(result['f_Hz'], dtype=float)
    pfields=np.asarray(result['acoustic_pressure_field_Pa'])
    p_axis=[]; power=[]; flux_raw=[]; infos=[]
    for i,fi in enumerate(f):
        samples, info=boundary93_hk_samples_from_p1(model, pfields[i])
        pa=hk_pressure_from_samples(float(fi), params.c0_m_s, *samples, obs_r=0.0, obs_z=params.observation_distance_m, nphi=nphi_axis, mirror=mirror, sign=-1)[0]
        Praw=float(intensity_power_from_samples(float(fi), params.rho0_kg_m3, samples[0], samples[4], samples[5], samples[6]))
        # Boundary-orientation and coarse P1 gradients can make the direct flux negative;
        # keep the signed diagnostic and use a half-space pressure estimate as a stable
        # positive acoustic-power fallback for efficiency plots.
        Pfallback=(abs(pa)**2/(2.0*params.rho0_kg_m3*params.c0_m_s))*2.0*math.pi*params.observation_distance_m**2
        P=Praw if Praw > 0 else Pfallback
        p_axis.append(pa); power.append(P); flux_raw.append(Praw); infos.append(asdict(info))
    p_axis=np.asarray(p_axis, dtype=complex); power=np.asarray(power, dtype=float); flux_raw=np.asarray(flux_raw, dtype=float)
    spl=20.0*np.log10(np.maximum(np.abs(p_axis)/math.sqrt(2.0), 1e-300)/params.p_ref_Pa)
    phase=np.unwrap(np.angle(p_axis))*180.0/math.pi
    return {'p_1m_hk_Pa_peak':p_axis, 'SPL_1m_hk_dB':spl, 'phase_hk_deg':phase, 'hk_halfspace_power_W':power, 'hk_flux_raw_W':flux_raw, 'hk_boundary_info':infos}


def hk_directivity_from_result(
    result: Mapping[str, np.ndarray],
    model: AcousticStructureModel,
    params: Stage4CParameters,
    *,
    angles_deg: np.ndarray | None = None,
    nphi: int = 48,
    mirror: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if angles_deg is None:
        angles_deg=np.linspace(-90,90,181)
    angles=np.asarray(angles_deg, dtype=float)
    th=np.deg2rad(angles)
    obs_r=np.abs(np.sin(th))*params.observation_distance_m
    obs_z=np.cos(th)*params.observation_distance_m
    f=np.asarray(result['f_Hz'], dtype=float)
    pfields=np.asarray(result['acoustic_pressure_field_Pa'])
    spl=np.zeros((len(f), len(angles)), dtype=float)
    for i,fi in enumerate(f):
        samples,_=boundary93_hk_samples_from_p1(model, pfields[i])
        p=hk_pressure_from_samples(float(fi), params.c0_m_s, *samples, obs_r=obs_r, obs_z=obs_z, nphi=nphi, mirror=mirror, sign=-1)
        spl[i]=20.0*np.log10(np.maximum(np.abs(p)/math.sqrt(2.0), 1e-300)/params.p_ref_Pa)
    i0=int(np.argmin(np.abs(angles)))
    rel=spl-spl[:,[i0]]
    return f, angles, spl, rel


def solve_stage4D_full(
    freqs_Hz: Iterable[float],
    Zb_ohm: np.ndarray,
    model: AcousticStructureModel,
    params: Stage4CParameters,
    *,
    nra_enabled: bool = True,
    hk_directivity: bool = True,
) -> dict:
    res=solve_stage4C_full_asb(freqs_Hz, Zb_ohm, model, params, nra_enabled=nra_enabled)
    hk=hk_axis_and_power_from_result(res, model, params)
    # replace primary Stage-4C far field by Boundary-93 HK/pext-style field
    res['p_1m_piston_Pa_peak']=res['p_1m_Pa_peak'].copy()
    res['SPL_1m_piston_dB']=res['SPL_1m_dB'].copy()
    res['phase_piston_deg']=res['phase_deg'].copy()
    res['p_1m_Pa_peak']=hk['p_1m_hk_Pa_peak']
    res['SPL_1m_dB']=hk['SPL_1m_hk_dB']
    res['phase_deg']=hk['phase_hk_deg']
    res['hk_halfspace_power_W']=hk['hk_halfspace_power_W']
    res['hk_flux_raw_W']=hk['hk_flux_raw_W']
    res['hk_boundary_info']=hk['hk_boundary_info']
    res['acoustic_power_W']=np.maximum(hk['hk_halfspace_power_W'], 0.0)
    res['acoustic_efficiency_percent']=100.0*res['acoustic_power_W']/np.maximum(res['coil_power_W'], 1e-300)
    if hk_directivity:
        fd,ang,spl,rel=hk_directivity_from_result(res, model, params)
        res['directivity_f_Hz']=fd; res['directivity_angles_deg']=ang; res['directivity_spl_hk_dB']=spl; res['directivity_relative_hk_dB']=rel
    return res


def stage4D_rows(result: Mapping[str,np.ndarray]) -> list[dict]:
    rows=[]
    for i,fi in enumerate(result['f_Hz']):
        Z=result['Z_total_ohm'][i]; Zb=result['Zb_ohm'][i]; Zm=result['Zm_asb_N_s_m'][i]
        rows.append({
            'f_Hz':float(fi),
            'SPL_1m_hk_dB':float(result['SPL_1m_dB'][i]),
            'SPL_1m_piston_dB':float(result['SPL_1m_piston_dB'][i]),
            'hk_minus_piston_dB':float(result['SPL_1m_dB'][i]-result['SPL_1m_piston_dB'][i]),
            'phase_hk_deg':float(result['phase_deg'][i]),
            'Z_abs_ohm':float(abs(Z)),
            'Z_real_ohm':float(np.real(Z)),
            'Z_imag_ohm':float(np.imag(Z)),
            'Zb_abs_ohm':float(abs(Zb)),
            'Zb_real_ohm':float(np.real(Zb)),
            'Zb_imag_ohm':float(np.imag(Zb)),
            'Zm_abs_N_s_m':float(abs(Zm)),
            'I_abs_A_peak':float(abs(result['I_A_peak'][i])),
            'F_abs_N_peak':float(abs(result['F_Lorentz_N_peak'][i])),
            'v_abs_m_s_peak':float(abs(result['v_coil_m_s_peak'][i])),
            'coil_power_W':float(result['coil_power_W'][i]),
            'acoustic_power_hk_W':float(result['acoustic_power_W'][i]),
            'hk_flux_raw_W':float(result['hk_flux_raw_W'][i]),
            'acoustic_efficiency_percent':float(result['acoustic_efficiency_percent'][i]),
        })
    return rows
