from __future__ import annotations

from dataclasses import asdict
import math
from typing import Mapping, Iterable, Sequence

import numpy as np

from .stage4C_acoustic_structure import AcousticStructureModel, Stage4CParameters, EXTERIOR_FIELD_BOUNDARY, ACOUSTIC_DOMAINS, PML_DOMAINS
from .stage4D_exterior_nra import _edge_to_triangles, _tri_area_grad, HKBoundaryInfo, boundary93_hk_samples_from_p1, hk_directivity_from_result, hk_axis_and_power_from_result
from .exterior_field import hk_pressure_from_samples, intensity_power_from_samples


def recover_acoustic_nodal_gradients(model: AcousticStructureModel, p_acoustic_nodes: np.ndarray, *, include_pml: bool = False) -> np.ndarray:
    """Area-weighted nodal recovery of P1 pressure gradients on the acoustic mesh.

    Stage-4D used the constant gradient from the single triangle adjacent to Boundary 93.
    That is exact for a P1 element but noisy for HK differentiation at high frequency.
    This function performs a conservative ZZ-style recovery: each node receives the
    area-weighted average of gradients from adjacent acoustic triangles.  It does not
    change the acoustic solution; it only stabilizes the exterior-field normal derivative.
    """
    mesh = model.mesh
    pvals = np.asarray(p_acoustic_nodes, dtype=complex)
    n = len(model.acoustic_nodes_global)
    grad_sum = np.zeros((n, 2), dtype=complex)
    weight_sum = np.zeros(n, dtype=float)
    allowed = set(ACOUSTIC_DOMAINS) if include_pml else (set(ACOUSTIC_DOMAINS) - set(PML_DOMAINS))
    amap = model.acoustic_node_map
    for tri, dom in zip(mesh.triangles, mesh.tri_domains):
        if int(dom) not in allowed:
            continue
        local = [amap.get(int(g)) for g in tri]
        if any(x is None for x in local):
            continue
        area, grads = _tri_area_grad(mesh.points_rz_m, tri)
        if grads is None or area is None or area <= 0:
            continue
        ptri = np.array([pvals[int(x)] for x in local], dtype=complex)
        g = ptri @ grads
        # Axisymmetric weight: 2*pi*rbar*area, but 2*pi cancels in averaging.
        rbar = max(float(np.mean(mesh.points_rz_m[tri, 0])), 1e-9)
        w = rbar * float(area)
        for li in local:
            li = int(li)
            grad_sum[li] += w * g
            weight_sum[li] += w
    ok = weight_sum > 0
    grad = np.zeros((n, 2), dtype=complex)
    grad[ok] = grad_sum[ok] / weight_sum[ok, None]
    return grad



def recover_acoustic_nodal_gradients_ppr(
    model: AcousticStructureModel,
    p_acoustic_nodes: np.ndarray,
    *,
    target_global_nodes: Sequence[int] | None = None,
    include_pml: bool = False,
    polynomial_degree: int = 2,
    minimum_points: int | None = None,
    maximum_patch_rings: int = 6,
) -> np.ndarray:
    """One-sided polynomial-preserving recovery of pressure gradients.

    COMSOL's Exterior Field Calculation uses polynomial-preserving recovery
    (``UsePPR=1``).  For Boundary 93 this must be evaluated from the physical
    acoustic side, not averaged with the PML-side element gradient.  A local
    pressure polynomial is fitted in scaled ``(r,z)`` coordinates and its
    derivative at the target node is returned.  Quadratic recovery is used by
    default; rank-deficient patches fall back to a linear fit and finally to
    the conservative ZZ recovery.
    """
    mesh = model.mesh
    pvals = np.asarray(p_acoustic_nodes, dtype=complex)
    n = len(model.acoustic_nodes_global)
    amap = model.acoustic_node_map
    allowed = set(ACOUSTIC_DOMAINS) if include_pml else (set(ACOUSTIC_DOMAINS) - set(PML_DOMAINS))

    adjacency: dict[int, set[int]] = {}
    available: set[int] = set()
    for tri, dom in zip(mesh.triangles, mesh.tri_domains):
        if int(dom) not in allowed:
            continue
        nodes = [int(x) for x in tri if int(x) in amap]
        if len(nodes) != 3:
            continue
        for g in nodes:
            available.add(g)
            adjacency.setdefault(g, set()).update(nodes)

    if target_global_nodes is None:
        targets = sorted(available)
    else:
        targets = [int(g) for g in target_global_nodes if int(g) in available]
    grad = np.zeros((n, 2), dtype=complex)
    legacy = None
    degree = 2 if int(polynomial_degree) >= 2 else 1
    ncoef = 6 if degree == 2 else 3
    minpts = int(minimum_points or (10 if degree == 2 else 5))

    for g in targets:
        x0 = mesh.points_rz_m[g]
        patch = {g}
        frontier = {g}
        for _ in range(int(maximum_patch_rings)):
            nxt: set[int] = set()
            for q in frontier:
                nxt.update(adjacency.get(q, ()))
            patch.update(nxt)
            frontier = nxt
            if len(patch) >= minpts:
                break
        nodes = sorted(q for q in patch if q in amap)
        if len(nodes) < 3:
            if legacy is None:
                legacy = recover_acoustic_nodal_gradients(model, pvals, include_pml=include_pml)
            grad[amap[g]] = legacy[amap[g]]
            continue
        xy = mesh.points_rz_m[nodes] - x0
        h = max(float(np.sqrt(np.mean(np.sum(xy * xy, axis=1)))), 1e-12)
        X = xy / h
        y = np.asarray([pvals[amap[q]] for q in nodes], dtype=complex)
        if degree == 2 and len(nodes) >= 6:
            A = np.column_stack([
                np.ones(len(nodes)), X[:, 0], X[:, 1],
                X[:, 0] ** 2, X[:, 0] * X[:, 1], X[:, 1] ** 2,
            ])
        else:
            A = np.column_stack([np.ones(len(nodes)), X[:, 0], X[:, 1]])
        coef, _, rank, _ = np.linalg.lstsq(A, y, rcond=1e-12)
        if rank < min(A.shape[1], ncoef) or len(coef) < 3:
            A1 = np.column_stack([np.ones(len(nodes)), X[:, 0], X[:, 1]])
            coef, _, rank, _ = np.linalg.lstsq(A1, y, rcond=1e-12)
        if rank < 3:
            if legacy is None:
                legacy = recover_acoustic_nodal_gradients(model, pvals, include_pml=include_pml)
            grad[amap[g]] = legacy[amap[g]]
        else:
            grad[amap[g]] = coef[1:3] / h
    return grad

def boundary93_hk_samples_recovered(
    model: AcousticStructureModel,
    p_acoustic_nodes: np.ndarray,
    *,
    boundary_id: int = EXTERIOR_FIELD_BOUNDARY,
    intorder: int = 3,
    include_pml_in_recovery: bool = False,
    recovery_method: str = "ppr",
    force_radial_normals: bool = True,
):
    """Boundary-93 HK quadrature samples using recovered nodal pressure gradients.

    Uses one-sided quadratic polynomial-preserving recovery by default, matching
    COMSOL ExteriorFieldCalculation ``UsePPR=1``.  The legacy ZZ recovery remains
    available through ``recovery_method='zz'`` for audit comparisons.
    """
    mesh = model.mesh
    pvals = np.asarray(p_acoustic_nodes, dtype=complex)
    amap = model.acoustic_node_map
    boundary_nodes = sorted({int(g) for seg, tag in zip(mesh.line_cells, mesh.line_tags) if int(tag) == int(boundary_id) for g in seg})
    method = str(recovery_method).lower()
    if method in {"ppr", "ppr_q2", "polynomial_preserving"}:
        grad_nodes = recover_acoustic_nodal_gradients_ppr(
            model, pvals, target_global_nodes=boundary_nodes,
            include_pml=include_pml_in_recovery, polynomial_degree=2,
        )
    elif method in {"zz", "legacy_zz"}:
        grad_nodes = recover_acoustic_nodal_gradients(model, pvals, include_pml=include_pml_in_recovery)
    else:
        raise ValueError(f"Unknown Boundary93 recovery method: {recovery_method}")
    edge_map = _edge_to_triangles(mesh.triangles)
    cent = mesh.points_rz_m[mesh.triangles].mean(axis=1)
    if intorder <= 1:
        xi=np.array([0.5]); wi=np.array([1.0])
    elif intorder == 2:
        xg,wg=np.polynomial.legendre.leggauss(2); xi=0.5*(xg+1.0); wi=0.5*wg
    else:
        xg,wg=np.polynomial.legendre.leggauss(3); xi=0.5*(xg+1.0); wi=0.5*wg
    rs=[]; zs=[]; nr=[]; nz=[]; ds=[]; pb=[]; dpdn=[]; nseg=0
    for seg, tag in zip(mesh.line_cells, mesh.line_tags):
        if int(tag) != int(boundary_id):
            continue
        adj = model.boundary_adjacency.get(int(tag))
        if adj is None:
            continue
        key = tuple(sorted(map(int, seg)))
        physical_air = [d for d in (adj.up_domain, adj.down_domain) if d in ACOUSTIC_DOMAINS and d not in PML_DOMAINS]
        pml_or_ext = [d for d in (adj.up_domain, adj.down_domain) if d in (1,5,0)]
        adom = int(physical_air[0]) if physical_air else int(adj.up_domain if adj.up_domain in ACOUSTIC_DOMAINS else adj.down_domain)
        tri_id=None; other_id=None
        for it in edge_map.get(key, []):
            d=int(mesh.tri_domains[it])
            if d == adom:
                tri_id=it
            elif pml_or_ext and d in pml_or_ext:
                other_id=it
        if tri_id is None:
            for it in edge_map.get(key, []):
                d=int(mesh.tri_domains[it])
                if d in ACOUSTIC_DOMAINS and d not in PML_DOMAINS:
                    tri_id=it; break
        if tri_id is None:
            continue
        p0=mesh.points_rz_m[int(seg[0])]; p1=mesh.points_rz_m[int(seg[1])]
        tang=p1-p0; L=float(np.linalg.norm(tang))
        if L <= 0:
            continue
        n1=np.array([tang[1], -tang[0]], dtype=float)/L
        if other_id is not None:
            direction=cent[other_id]-cent[tri_id]
        else:
            mid=0.5*(p0+p1); direction=mid-cent[tri_id]
        if np.dot(n1, direction) < 0:
            n1=-n1
        g0=int(seg[0]); g1=int(seg[1])
        if g0 not in amap or g1 not in amap:
            continue
        l0=amap[g0]; l1=amap[g1]
        p0v=pvals[l0]; p1v=pvals[l1]
        gr0=grad_nodes[l0]; gr1=grad_nodes[l1]
        for s,w in zip(xi,wi):
            x=(1.0-s)*p0+s*p1
            if force_radial_normals:
                rad=max(float(np.linalg.norm(x)), 1e-14)
                n_eval=np.asarray(x, dtype=float)/rad
            else:
                n_eval=n1
            pp=(1.0-s)*p0v+s*p1v
            gg=(1.0-s)*gr0+s*gr1
            rs.append(float(x[0])); zs.append(float(x[1])); nr.append(float(n_eval[0])); nz.append(float(n_eval[1])); ds.append(float(L*w)); pb.append(pp); dpdn.append(complex(gg[0]*n_eval[0]+gg[1]*n_eval[1]))
        nseg += 1
    if not rs:
        raise RuntimeError(f'No usable Boundary {boundary_id} recovered-HK samples were generated')
    return (np.asarray(rs), np.asarray(zs), np.asarray(nr), np.asarray(nz), np.asarray(ds), np.asarray(pb), np.asarray(dpdn)), HKBoundaryInfo(
        n_samples=len(rs), boundary_id=boundary_id, source_segments=nseg,
        mean_radius_m=float(np.mean(rs)), mean_abs_pressure_Pa=float(np.mean(np.abs(pb))),
        mean_abs_dpdn_Pa_per_m=float(np.mean(np.abs(dpdn)))
    )


def hk_axis_and_power_recovered(result: Mapping[str, np.ndarray], model: AcousticStructureModel, params: Stage4CParameters, *, nphi_axis: int = 96, mirror: bool = True) -> dict:
    f=np.asarray(result['f_Hz'], dtype=float)
    pfields=np.asarray(result['acoustic_pressure_field_Pa'])
    p_axis=[]; power=[]; flux_raw=[]; infos=[]
    for i,fi in enumerate(f):
        samples, info = boundary93_hk_samples_recovered(model, pfields[i])
        pa=hk_pressure_from_samples(float(fi), params.c0_m_s, *samples, obs_r=0.0, obs_z=params.observation_distance_m, nphi=nphi_axis, mirror=mirror, sign=-1)[0]
        Praw=float(intensity_power_from_samples(float(fi), params.rho0_kg_m3, samples[0], samples[4], samples[5], samples[6]))
        Pfallback=(abs(pa)**2/(2.0*params.rho0_kg_m3*params.c0_m_s))*2.0*math.pi*params.observation_distance_m**2
        p_axis.append(pa); power.append(Praw if Praw > 0 else Pfallback); flux_raw.append(Praw); infos.append(asdict(info))
    p_axis=np.asarray(p_axis, dtype=complex); power=np.asarray(power, dtype=float); flux_raw=np.asarray(flux_raw, dtype=float)
    spl=20.0*np.log10(np.maximum(np.abs(p_axis)/math.sqrt(2.0), 1e-300)/params.p_ref_Pa)
    phase=np.unwrap(np.angle(p_axis))*180.0/math.pi
    return {'p_1m_hk_recovered_Pa_peak':p_axis,'SPL_1m_hk_recovered_dB':spl,'phase_hk_recovered_deg':phase,'hk_recovered_halfspace_power_W':power,'hk_recovered_flux_raw_W':flux_raw,'hk_recovered_boundary_info':infos}


def hk_directivity_recovered(result: Mapping[str, np.ndarray], model: AcousticStructureModel, params: Stage4CParameters, *, angles_deg: np.ndarray | None = None, nphi: int = 72, mirror: bool = True):
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
        samples,_=boundary93_hk_samples_recovered(model, pfields[i])
        p=hk_pressure_from_samples(float(fi), params.c0_m_s, *samples, obs_r=obs_r, obs_z=obs_z, nphi=nphi, mirror=mirror, sign=-1)
        spl[i]=20.0*np.log10(np.maximum(np.abs(p)/math.sqrt(2.0), 1e-300)/params.p_ref_Pa)
    i0=int(np.argmin(np.abs(angles)))
    rel=spl-spl[:,[i0]]
    return f, angles, spl, rel


def compare_directivity_pair(coarse: tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray], refined: tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]) -> dict:
    fc, ac, splc, relc = coarse
    fr, ar, splr, relr = refined
    if len(ac) != len(ar) or np.max(np.abs(ac-ar)) > 1e-9:
        raise ValueError('angle grids must match')
    d_abs=splr[0]-splc[0]
    d_rel=relr[0]-relc[0]
    # common engineering sector excludes the deepest numerical nulls but keeps main/side lobe area
    sector=np.abs(ac) <= 75
    main=np.abs(ac) <= 60
    return {
        'f_Hz': float(fr[0]),
        'absolute_rms_dB': float(np.sqrt(np.mean(d_abs**2))),
        'relative_rms_dB': float(np.sqrt(np.mean(d_rel**2))),
        'relative_max_abs_dB': float(np.max(np.abs(d_rel))),
        'relative_rms_75deg_dB': float(np.sqrt(np.mean(d_rel[sector]**2))),
        'relative_max_abs_75deg_dB': float(np.max(np.abs(d_rel[sector]))),
        'relative_rms_60deg_dB': float(np.sqrt(np.mean(d_rel[main]**2))),
        'relative_max_abs_60deg_dB': float(np.max(np.abs(d_rel[main]))),
    }
